from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from llm_val.utils.ner_metrics import split_entities_by_words
from llm_val.utils.report_helper import semaphore_by_threshold


MIN_OBSERVATIONS = 101
ID_COLUMNS = ('session_id', 'question_id', 'sample_id', 'example_id', 'id', 'question')
COUNT_COLUMNS = ('tp', 'fp', 'fn', 'tn')
TERMS = {
    'accuracy': ((1, 0, 0, 1), (1, 1, 1, 1)),
    'precision': ((1, 0, 0, 0), (1, 1, 0, 0)),
    'recall': ((1, 0, 0, 0), (1, 0, 1, 0)),
    'f1': ((2, 0, 0, 0), (2, 1, 1, 0)),
}
METRIC_ALIASES = {
    'accuracy_score': 'accuracy',
    'precision_score': 'precision',
    'recall_score': 'recall',
    'f1_score': 'f1',
    'single_mean': 'mean',
}
METHOD_NAMES = {
    'binomial': 'Точный биномиальный критерий',
    'mcnemar': 'Точный критерий Мак-Немара',
    'fisher': 'Точный критерий Фишера',
    'delta': 'Дельта-метод',
    'student': 'Одновыборочный t-критерий',
    'welch': 'Критерий Уэлча',
    'wilcoxon': 'Парный критерий Уилкоксона',
    'bootstrap': 'Бутстрэп',
}
EXACT_METHODS = {'binomial', 'mcnemar', 'fisher'}


class UndefinedStatistic(ValueError):
    """The data define a valid comparison, but the statistic cannot be estimated."""


@dataclass(frozen=True)
class NerMatching:
    """Entity types and matching rules used to count NER errors per document."""

    entity_types: tuple[str, ...] | None
    exact: bool
    ignore_repetitions: bool


@dataclass(frozen=True)
class Hypothesis:
    """Metric definition, input column and reference for one comparison."""

    label: str
    kind: str
    reference: float
    metric: Callable[[pd.DataFrame], float] | None
    compare_versions: bool
    main_column: str | None


@dataclass(frozen=True)
class Comparison:
    """An aligned comparison and the statistic evaluated on each resample."""

    a: pd.DataFrame
    b: pd.DataFrame | None
    paired: bool
    statistic: Callable[[pd.DataFrame], float]
    reference: float
    alternative: str

    def estimate(self, a, b):
        """Return the absolute value or the difference A − B."""
        value_a = self.statistic(a)
        return value_a if b is None else value_a - self.statistic(b)


@dataclass(frozen=True)
class Inference:
    """Inferential results for one method and the displayed estimate."""

    p_value: float
    estimate: float
    ci_95: tuple[float, float] | None
    ci_99: tuple[float, float] | None
    note: str = ''


def metric_kind(name):
    """Resolve the supported metric names while keeping custom metrics intact."""
    name = name.strip().lower().replace('-', '_')
    return METRIC_ALIASES.get(name, name)


def _ids(y, x, column):
    if column in y:
        return pd.Index(y[column])
    if column == y.index.name:
        return y.index
    if column in x and x.index.equals(y.index):
        return pd.Index(x[column])
    return None


def _align(a, b, id_column):
    y_a, x_a = a
    y_b, x_b = b
    if id_column:
        candidates = (id_column,)
    else:
        shared_index = (y_a.index.name,) if y_a.index.name and y_a.index.name == y_b.index.name else ()
        candidates = (*shared_index, *ID_COLUMNS)
    for column in candidates:
        ids_a, ids_b = _ids(y_a, x_a, column), _ids(y_b, x_b, column)
        if ids_a is None or ids_b is None:
            continue
        if ids_a.hasnans or ids_b.hasnans or not ids_a.is_unique or not ids_b.is_unique:
            if not id_column and column == 'question':
                continue
            raise ValueError(f'Идентификатор {column!r} должен быть уникальным и непустым в обеих выборках.')
        positions = ids_b.get_indexer(ids_a)
        if len(ids_a) != len(ids_b) or (positions < 0).any():
            raise ValueError(f'Наборы идентификаторов {column!r} в парных выборках различаются.')
        return y_a.reset_index(drop=True), y_b.iloc[positions].reset_index(drop=True)
    raise ValueError('Для парного сравнения нужен общий идентификатор объекта. Укажите id_column.')


def _classification_rows(frame, kind, main_column, positive_label, ner):
    lookup = {str(column).lower(): column for column in frame}
    required = COUNT_COLUMNS if kind == 'accuracy' else COUNT_COLUMNS[:3]
    if all(column in lookup for column in required):
        rows = frame[[lookup[column] for column in required]].copy()
        rows.columns = list(required)
        rows = rows.apply(pd.to_numeric)
        valid = rows.dropna().to_numpy(dtype=float)
        if not np.isfinite(valid).all() or (valid < 0).any() or (valid != np.floor(valid)).any():
            raise ValueError('Счётчики TP, FP, FN и TN должны быть конечными неотрицательными целыми.')
        return rows.reindex(columns=COUNT_COLUMNS, fill_value=0)
    if kind != 'accuracy' and {'extracted', 'target'} <= set(frame):
        return _ner_rows(frame, ner)
    if {'true_answer', 'model_answer'} <= set(frame):
        return _counts_from_labels(frame['true_answer'], frame['model_answer'], kind, positive_label)
    column = main_column or 'metric_value'
    if kind == 'accuracy' and column in frame:
        scores = pd.to_numeric(frame[column])
        if scores.dropna().isin((0, 1)).all():
            return _accuracy_rows(scores)
    raise ValueError(f'Для {kind} нужны пообъектные счётчики или true_answer и model_answer.')


def _accuracy_rows(scores):
    return pd.DataFrame({'tp': scores, 'fp': 1 - scores, 'fn': 0, 'tn': 0})


def _counts_from_labels(truth, prediction, kind, positive_label):
    valid = truth.notna() & prediction.notna()
    if kind == 'accuracy':
        return _accuracy_rows(truth.eq(prediction).astype('Int64').where(valid))
    labels = pd.unique(pd.concat([truth[valid], prediction[valid]], ignore_index=True))
    if len(labels) > 2 or (len(labels) == 2 and positive_label not in labels):
        raise ValueError(
            'Для бинарной метрики укажите positive_label. Для другой агрегации используйте счётчики или custom.'
        )
    actual, predicted = truth.eq(positive_label), prediction.eq(positive_label)
    rows = pd.DataFrame(
        {'tp': actual & predicted, 'fp': ~actual & predicted, 'fn': actual & ~predicted, 'tn': ~actual & ~predicted}
    )
    return rows.astype('Int64').where(valid, axis=0)


def _ner_rows(frame, matching):
    counts = []
    for predicted, target in frame[['extracted', 'target']].itertuples(index=False, name=None):
        values = (predicted, target)
        missing = [pd.api.types.is_scalar(value) and pd.isna(value) for value in values]
        if any(not absent and not isinstance(values[index], dict) for index, absent in enumerate(missing)):
            raise ValueError('Для NER колонки extracted и target должны содержать словари типов сущностей.')
        counts.append((np.nan,) * 4 if any(missing) else _entity_counts(predicted, target, matching))
    return pd.DataFrame(counts, index=frame.index, columns=COUNT_COLUMNS)


def _entity_counts(predicted, target, matching):
    if not matching.exact:
        predicted, target = map(split_entities_by_words, (predicted, target))
    labels = matching.entity_types if matching.entity_types is not None else predicted.keys() | target.keys()
    tp = fp = fn = 0
    for label in labels:
        a, b = predicted.get(label, ()), target.get(label, ())
        if matching.ignore_repetitions:
            a, b = set(a), set(b)
        a, b = Counter(a), Counter(b)
        tp += sum((a & b).values())
        fp += sum((a - b).values())
        fn += sum((b - a).values())
    return tp, fp, fn, 0


def _mean_rows(frame, column):
    if column is not None:
        values = pd.to_numeric(frame[column])
    elif 'metric_value' in frame:
        values = pd.to_numeric(frame['metric_value'])
    else:
        candidates = frame.drop(columns=list(ID_COLUMNS), errors='ignore')
        if not len(candidates.columns):
            raise ValueError('Укажите main_column с пообъектной оценкой для метрики single_mean.')
        values = candidates.apply(pd.to_numeric).mean(axis=1, skipna=False)
    if not np.isfinite(values.dropna().to_numpy(dtype=float)).all():
        raise ValueError('Пообъектные оценки должны быть конечными.')
    return values.to_frame('score')


def _use_rows(frame, kind, main_column, positive_label, ner):
    if kind in TERMS:
        return _classification_rows(frame, kind, main_column, positive_label, ner)
    if kind == 'mean':
        return _mean_rows(frame, main_column)
    return frame


def _valid_rows(frame, kind, main_column):
    if kind in TERMS or kind == 'mean':
        return frame.notna().all(axis=1)
    if main_column is not None:
        return frame[main_column].notna()
    return pd.Series(True, index=frame.index)


def _samples(a, b, paired, kind, main_column, id_column, positive_label, ner):
    raw_a, raw_b = _align(a, b, id_column) if paired else (a[0], None if b is None else b[0])
    rows_a = _use_rows(raw_a, kind, main_column, positive_label, ner)
    valid_a = _valid_rows(rows_a, kind, main_column)
    if raw_b is None:
        return rows_a.loc[valid_a], None
    rows_b = _use_rows(raw_b, kind, main_column, positive_label, ner)
    valid_b = _valid_rows(rows_b, kind, main_column)
    if not paired:
        return rows_a.loc[valid_a], rows_b.loc[valid_b]
    valid = valid_a & valid_b
    if kind in TERMS:
        for target in ('true_answer', 'target'):
            if (
                target in raw_a
                and target in raw_b
                and not raw_a.loc[valid, target].eq(raw_b.loc[valid, target]).fillna(False).all()
            ):
                raise ValueError('Эталонные метки различаются между выровненными объектами.')
    return rows_a.loc[valid], rows_b.loc[valid]


def _ratio(counts, kind):
    numerator, denominator = TERMS[kind]
    total = counts.to_numpy(dtype=float).sum(axis=0)
    trials = total @ denominator
    if trials == 0:
        raise UndefinedStatistic(f'Нулевой знаменатель метрики {kind}.')
    return float((total @ numerator) / trials)


def _mean(rows):
    return float(rows['score'].mean())


def _finite_metric(rows, metric):
    value = float(metric(rows))
    if not np.isfinite(value):
        raise UndefinedStatistic('Значение метрики не определено.')
    return value


def _binomial_trials(frame, kind):
    numerator, denominator = TERMS[kind]
    counts = frame.to_numpy(dtype=float)
    return (counts @ numerator).astype(int), (counts @ denominator).astype(int)


def _exact_method_is_valid(comparison, kind):
    _, trials_a = _binomial_trials(comparison.a, kind)
    if (trials_a > 1).any():
        return False
    if comparison.b is None:
        return True
    _, trials_b = _binomial_trials(comparison.b, kind)
    if comparison.reference != 0 or (trials_b > 1).any():
        return False
    return not comparison.paired or np.array_equal(trials_a, trials_b)


def _default_method(kind, compare_versions, paired):
    if kind == 'f1' or (kind == 'precision' and paired):
        return 'delta'
    if kind in TERMS:
        if not compare_versions:
            return 'binomial'
        return 'mcnemar' if paired else 'fisher'
    if kind == 'mean':
        if not compare_versions:
            return 'student'
        return 'wilcoxon' if paired else 'welch'
    return 'bootstrap'


def _method_selection(default, methods, include_bootstrap, several_hypotheses, kind):
    if methods:
        selected = tuple(method for method in methods if method in (default, 'bootstrap'))
        if not several_hypotheses and selected != methods:
            invalid = next(method for method in methods if method not in selected)
            raise ValueError(f'Метод {invalid!r} не соответствует метрике {kind!r} и режиму сравнения.')
        if selected:
            return selected, True
    automatic = (default, 'bootstrap') if include_bootstrap and default != 'bootstrap' else (default,)
    return automatic, False


def _require_bootstrap_support(comparison, kind):
    for name, frame in (('A', comparison.a), ('B', comparison.b)):
        if frame is not None:
            _, trials = _binomial_trials(frame, kind)
            if np.count_nonzero(trials) < 2:
                raise UndefinedStatistic(
                    f'Для бутстрэпа {kind} у модели {name} нужно не менее двух объектов с ненулевым знаменателем.'
                )


def _resample(comparison, iterations, rng, statistic=None, *, kind=None):
    if kind in TERMS:
        _require_bootstrap_support(comparison, kind)
    estimate = statistic or comparison.estimate
    n_a = len(comparison.a)
    n_b = len(comparison.b) if comparison.b is not None else 0
    draws = []
    for _ in range(iterations):
        index_a = rng.integers(n_a, size=n_a)
        a = comparison.a.iloc[index_a]
        b = None
        if comparison.b is not None:
            index_b = index_a if comparison.paired else rng.integers(n_b, size=n_b)
            b = comparison.b.iloc[index_b]
        try:
            draws.append(estimate(a, b))
        except UndefinedStatistic:
            continue
    undefined = iterations - len(draws)
    if undefined:
        raise UndefinedStatistic(f'{undefined} из {iterations} бутстрэп-выборок дали неопределённую метрику.')
    return np.asarray(draws)


def _intervals(estimate, scale, quantile):
    return tuple((estimate - scale * quantile(level), estimate + scale * quantile(level)) for level in (0.975, 0.995))


def _bootstrap_intervals(draws, estimate, ci_type):
    interval_95 = tuple(np.quantile(draws, (0.025, 0.975)))
    interval_99 = tuple(np.quantile(draws, (0.005, 0.995)))
    if ci_type == 'basic':
        return tuple((2 * estimate - high, 2 * estimate - low) for low, high in (interval_95, interval_99))
    if ci_type == 'normal':
        return _intervals(estimate, np.std(draws, ddof=1) if len(draws) > 1 else 0.0, stats.norm.ppf)
    return interval_95, interval_99


def _tail(draws, reference, alternative):
    lower = float(np.mean(draws <= reference))
    upper = float(np.mean(draws >= reference))
    if alternative == 'two-sided':
        return min(1.0, 2 * min(lower, upper))
    return lower if alternative == 'greater' else upper


def _bootstrap(comparison, kind, iterations, rng, ci_type):
    estimate = comparison.estimate(comparison.a, comparison.b)
    draws = _resample(comparison, iterations, rng, kind=kind)
    note = f'Эмпирическая доля по {iterations} бутстрэп-выборкам.'
    return Inference(
        _tail(draws, comparison.reference, comparison.alternative),
        estimate,
        *_bootstrap_intervals(draws, estimate, ci_type),
        note,
    )


def _exact(comparison, kind, method, iterations, rng, ci_type):
    estimate = comparison.estimate(comparison.a, comparison.b)
    successes_a, trials_a = _binomial_trials(comparison.a, kind)
    if method == 'binomial':
        result = stats.binomtest(
            int(successes_a.sum()), int(trials_a.sum()), p=comparison.reference, alternative=comparison.alternative
        )
        two_sided = stats.binomtest(result.k, result.n)
        intervals = [tuple(two_sided.proportion_ci(level)) for level in (0.95, 0.99)]
        return Inference(result.pvalue, estimate, *intervals)
    successes_b, trials_b = _binomial_trials(comparison.b, kind)
    if method == 'mcnemar':
        n10 = int(np.sum(successes_a > successes_b))
        n01 = int(np.sum(successes_a < successes_b))
        p_value = stats.binomtest(n10, n10 + n01, alternative=comparison.alternative).pvalue if n10 + n01 else 1.0
    else:
        table = [
            [successes.sum(), trials.sum() - successes.sum()]
            for successes, trials in ((successes_a, trials_a), (successes_b, trials_b))
        ]
        p_value = stats.fisher_exact(table, alternative=comparison.alternative).pvalue
    try:
        draws = _resample(comparison, iterations, rng, kind=kind)
    except UndefinedStatistic as error:
        return Inference(p_value, estimate, None, None, f'Доверительный интервал недоступен. {error}')
    return Inference(
        p_value,
        estimate,
        *_bootstrap_intervals(draws, estimate, ci_type),
        'Доверительный интервал оценён бутстрэпом по объектам.',
    )


def _influence(frame, kind):
    numerator, denominator = map(np.asarray, TERMS[kind])
    values = frame.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = mean @ denominator
    if scale == 0:
        raise UndefinedStatistic(f'Нулевой знаменатель метрики {kind}.')
    gradient = (numerator - ((mean @ numerator) / scale) * denominator) / scale
    return (values - mean) @ gradient


def _delta(comparison, kind):
    estimate = comparison.estimate(comparison.a, comparison.b)
    a = _influence(comparison.a, kind)
    b = _influence(comparison.b, kind) if comparison.b is not None else None
    if comparison.paired:
        variance = np.var(a - b, ddof=1) / len(a)
    else:
        variance = np.var(a, ddof=1) / len(a)
        if b is not None:
            variance += np.var(b, ddof=1) / len(b)
    error = np.sqrt(variance)
    if not np.isfinite(error) or error <= np.finfo(float).eps:
        raise UndefinedStatistic('Стандартная ошибка дельта-метода равна нулю или не определена.')
    z_score = (estimate - comparison.reference) / error
    tails = {
        'greater': stats.norm.sf(z_score),
        'less': stats.norm.cdf(z_score),
        'two-sided': 2 * stats.norm.sf(abs(z_score)),
    }
    return Inference(float(tails[comparison.alternative]), estimate, *_intervals(estimate, error, stats.norm.ppf))


def _median_difference(a, b):
    return float(np.median(a['score'].to_numpy() - b['score'].to_numpy()))


def _mean_test(comparison, method, iterations, rng, ci_type):
    a = comparison.a['score'].to_numpy(dtype=float)
    b = None if comparison.b is None else comparison.b['score'].to_numpy(dtype=float)
    reference, alternative = comparison.reference, comparison.alternative
    if method == 'wilcoxon':
        differences = a - b - reference
        p_value = stats.wilcoxon(differences, alternative=alternative).pvalue if np.any(differences) else 1.0
        estimate = _median_difference(comparison.a, comparison.b)
        draws = _resample(comparison, iterations, rng, _median_difference)
        return Inference(
            p_value, estimate, *_bootstrap_intervals(draws, estimate, ci_type), 'Эффект — медиана парных разностей.'
        )
    error = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)) if method == 'welch' else stats.sem(a)
    if not np.isfinite(error) or error == 0:
        raise UndefinedStatistic('t-критерий не определён при нулевой стандартной ошибке.')
    if method == 'welch':
        result = stats.ttest_ind(a - reference, b, equal_var=False, alternative=alternative)
    else:
        result = stats.ttest_1samp(a, reference, alternative=alternative)
    estimate = comparison.estimate(comparison.a, comparison.b)
    return Inference(result.pvalue, estimate, *_intervals(estimate, error, partial(stats.t.ppf, df=result.df)))


def _run_method(comparison, kind, method, iterations, rng, ci_type):
    if method == 'bootstrap':
        return _bootstrap(comparison, kind, iterations, rng, ci_type)
    if method == 'delta':
        return _delta(comparison, kind)
    if method in EXACT_METHODS:
        return _exact(comparison, kind, method, iterations, rng, ci_type)
    return _mean_test(comparison, method, iterations, rng, ci_type)


def _method_for_data(comparison, kind, method, explicit_methods):
    if method not in EXACT_METHODS or _exact_method_is_valid(comparison, kind):
        return method, ''
    reason = 'Точный метод неприменим при этом пороге или структуре наблюдений.'
    if explicit_methods:
        raise UndefinedStatistic(reason)
    return 'bootstrap', f'{reason} Использован бутстрэп по объектам.'


def _evaluate(comparison, kind, methods, explicit_methods, iterations, rng, ci_type):
    has_enough_rows = len(comparison.a) >= MIN_OBSERVATIONS and (
        comparison.b is None or len(comparison.b) >= MIN_OBSERVATIONS
    )
    if not has_enough_rows:
        yield from (
            (method, None, 'Недостаточно наблюдений: требуется N > 100 в каждой выборке.') for method in methods
        )
        return
    emitted = set()
    for method in methods:
        note = ''
        selected = method
        try:
            selected, note = _method_for_data(comparison, kind, method, explicit_methods)
            if selected in emitted:
                continue
            emitted.add(selected)
            result = _run_method(comparison, kind, selected, iterations, rng, ci_type)
        except UndefinedStatistic as error:
            yield selected, None, ' '.join(part for part in (note, str(error)) if part)
        else:
            yield selected, result, ' '.join(part for part in (note, result.note) if part).strip()


def _metric_value(frame, statistic):
    try:
        return statistic(frame) if frame is not None and len(frame) else None
    except UndefinedStatistic:
        return None


def _plot(results):
    eligible = [row for row in results if row['interval_95'] is not None]
    if not eligible:
        return []
    fig, axes = plt.subplots(len(eligible), 1, figsize=(8, 1.8 * len(eligible)), squeeze=False)
    for index, row in enumerate(eligible):
        ax = axes[index, 0]
        low, high = row['interval_95']
        ax.hlines(0, low, high, linewidth=2, label='95% ДИ')
        ax.plot(row['estimate'], 0, 'o', label='Оценка')
        ax.axvline(row['reference'], color='gray', linestyle='--', label='Граница H₀')
        ax.set_title(f'{row["metric"]} · {row["comparison"]} · {row["method"]}', fontsize=10)
        ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    return [fig]


def _p_value_color(p_value, threshold):
    if p_value is None or not np.isfinite(p_value):
        return 'gray'
    return semaphore_by_threshold(p_value, threshold, greater_is_better=False, right_border_is_bigger=False)


def _report(results, *, is_info, semaphore_threshold):
    rows = pd.DataFrame(
        {
            'Метрика': row['metric'],
            'Сравнение': row['comparison'],
            'Критерий': row['method'],
            'Оцениваемая величина': row['effect'],
            'Альтернатива': row['alternative'],
            'N A': row['n_a'],
            'N B': row['n_b'],
            'Метрика A': row['metric_a'],
            'Метрика B': row['metric_b'],
            'Оценка': row['estimate'],
            'Граница H₀': row['reference'],
            'p-value': row['p_value'],
            '95% ДИ': _interval_text(row['interval_95']),
            '99% ДИ': _interval_text(row['interval_99']),
            'Вывод': row['conclusion'],
            'Обоснование': row['justification'],
            'Комментарий': row['comment'],
        }
        for row in results
    )
    rows = rows.astype(object).where(rows.notna(), None)
    color = 'gray'
    if not is_info:
        colors = [_p_value_color(row['p_value'], semaphore_threshold) for row in results]
        rows['Результат теста'] = colors
        color = colors[0]
    primary = results[0]
    precomputed = {'p_value': primary['p_value'], 'methodology_version': 'genai_v4', 'comparisons': results}
    if primary['interval_95'] is not None:
        precomputed['ci_low'], precomputed['ci_high'] = primary['interval_95']
        precomputed['ci_low_99'], precomputed['ci_high_99'] = primary['interval_99']
    return {
        'report': {'semaphore': color, 'result_dataframes': [rows], 'result_plots': _plot(results)},
        'precomputed': precomputed,
    }


def _interval_text(interval):
    return None if interval is None else f'[{interval[0]:.6g}, {interval[1]:.6g}]'


def _validate_options(alternative, significance_level, is_info, semaphore_threshold, iterations, ci_type):
    if alternative not in {None, 'greater', 'less', 'two-sided'}:
        raise ValueError(f'Неизвестное направление сравнения: {alternative!r}.')
    if significance_level is not None and not 0 < significance_level < 1:
        raise ValueError('Уровень значимости должен находиться между 0 и 1.')
    if not is_info:
        low, high = semaphore_threshold
        if not 0 <= low < high <= 1:
            raise ValueError('Границы светофора должны удовлетворять условию 0 ≤ low < high ≤ 1.')
    if iterations < 2 or ci_type not in {'percentile', 'basic', 'normal'}:
        raise ValueError('Для бутстрэпа нужны минимум две итерации и поддерживаемый тип интервала.')


def _statistic_for_hypothesis(hypothesis):
    kind, reference = hypothesis.kind, hypothesis.reference
    proportion_threshold = not hypothesis.compare_versions and kind in TERMS
    if not np.isfinite(reference) or (proportion_threshold and not 0 <= reference <= 1):
        raise ValueError(f'Недопустимый порог для метрики {hypothesis.label!r}: {reference}.')
    statistic = partial(_ratio, kind=kind) if kind in TERMS else _mean if kind == 'mean' else hypothesis.metric
    if statistic is None:
        raise ValueError(f'Для метрики {hypothesis.label!r} укажите scorer или calc_metric_fn.')
    return partial(_finite_metric, metric=statistic)


def run_significance(
    a,
    baselines,
    hypotheses,
    *,
    paired,
    id_column,
    positive_label,
    ner,
    alternative,
    methods,
    include_bootstrap,
    iterations,
    ci_type,
    random_state,
    significance_level,
    is_info,
    semaphore_threshold,
):
    """Evaluate and render all requested hypotheses for the GenAI v4 test."""
    _validate_options(alternative, significance_level, is_info, semaphore_threshold, iterations, ci_type)
    if isinstance(methods, str):
        methods = (methods,)
    methods = tuple(dict.fromkeys(methods or ()))
    if set(methods) - METHOD_NAMES.keys():
        raise ValueError('methods содержит неизвестный статистический критерий.')
    results = []
    rng = np.random.default_rng(0 if random_state is None else random_state)
    for index, hypothesis in enumerate(hypotheses):
        kind, reference = hypothesis.kind, hypothesis.reference
        statistic = _statistic_for_hypothesis(hypothesis)
        is_paired = paired and hypothesis.compare_versions
        default = _default_method(kind, hypothesis.compare_versions, is_paired)
        selected, explicit_methods = _method_selection(default, methods, include_bootstrap, len(hypotheses) > 1, kind)
        direction = 'less' if {'welch', 'wilcoxon'}.intersection(selected) else 'greater'
        if index == 0 and alternative is not None:
            direction = alternative
        pairs = baselines.items() if hypothesis.compare_versions else (('Бизнес-порог', None),)
        for name, b in pairs:
            rows_a, rows_b = _samples(a, b, is_paired, kind, hypothesis.main_column, id_column, positive_label, ner)
            comparison = Comparison(rows_a, rows_b, is_paired, statistic, reference, direction)
            results.extend(
                _result_rows(
                    hypothesis.label,
                    name,
                    comparison,
                    kind,
                    selected,
                    explicit_methods,
                    iterations,
                    rng,
                    ci_type,
                    significance_level,
                )
            )
    return _report(results, is_info=is_info, semaphore_threshold=semaphore_threshold)


def _result_rows(label, name, comparison, kind, methods, explicit_methods, iterations, rng, ci_type, alpha):
    value_a = _metric_value(comparison.a, comparison.statistic)
    value_b = _metric_value(comparison.b, comparison.statistic)
    alternative = {'greater': '>', 'less': '<', 'two-sided': '≠'}[comparison.alternative]
    parameter = 'A' if comparison.b is None else 'A − B'
    rows = []
    for method, result, note in _evaluate(comparison, kind, methods, explicit_methods, iterations, rng, ci_type):
        ci_95 = None if result is None else result.ci_95
        ci_99 = None if result is None else result.ci_99
        rows.append(
            {
                'metric': label,
                'comparison': name,
                'method': METHOD_NAMES[method],
                'method_id': method,
                'effect': _effect_name(comparison, kind, method),
                'n_a': len(comparison.a),
                'n_b': None if comparison.b is None else len(comparison.b),
                'metric_a': value_a,
                'metric_b': value_b,
                'reference': comparison.reference,
                'alternative': f'{parameter} {alternative} {comparison.reference:g}',
                'estimate': None if result is None else float(result.estimate),
                'p_value': None if result is None else float(result.p_value),
                'interval_95': None if ci_95 is None else tuple(map(float, ci_95)),
                'interval_99': None if ci_99 is None else tuple(map(float, ci_99)),
                'conclusion': _conclusion(result, alpha),
                'justification': None,
                'comment': note,
            }
        )
    if len(rows) > 1:
        rows[0]['justification'] = _method_justification(rows, alpha)
    return rows


def _effect_name(comparison, kind, method):
    if method == 'wilcoxon':
        return 'Медиана парных разностей'
    if comparison.paired and kind == 'mean':
        return 'Средняя парная разность'
    return 'Значение метрики' if comparison.b is None else 'Разность метрик'


def _method_justification(rows, alpha):
    available = [row for row in rows if row['p_value'] is not None]
    if not available:
        return 'Для выбранных методов p-value недоступны. Причины указаны в комментариях.'
    if len(available) < len(rows):
        conclusion = available[0]['conclusion'] or 'Для решения по H₀ укажите уровень значимости.'
        return f'Один из методов недоступен. Вывод опирается на доступный критерий. {conclusion}'
    if len({row['effect'] for row in available}) > 1:
        return 'Методы оценивают разные величины. Вывод о каждой величине приведён в её строке.'
    if alpha is None:
        return 'Методы оценивают одну величину. Для общего решения по H₀ укажите уровень значимости.'
    rejected = {row['p_value'] < alpha for row in available}
    if len(rejected) > 1:
        return f'При α={alpha:g} методы дают разные решения по H₀. Итог зависит от согласованного критерия.'
    decision = 'отклоняют' if rejected == {True} else 'не отклоняют'
    return f'Оба метода {decision} H₀ при α={alpha:g}.'


def _conclusion(result, alpha):
    if result is None or alpha is None:
        return None
    return f'H₀ отклоняется при α={alpha:g}.' if result.p_value < alpha else f'H₀ не отклоняется при α={alpha:g}.'
