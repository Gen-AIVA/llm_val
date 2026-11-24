from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from llm_val.utils.annotation_aggregation import (
    SET_LABEL_TYPES,
    aggregate_dawid_skene,
    as_label_set,
    canonical_integer_labels,
)
from llm_val.utils.report_helper import semaphore_by_threshold, worst_semaphore


VERSION = 'genai_v4'
CACHE_VERSION = 2
IDS = ['session_id', 'assessor_id']
COHEN = 'Каппа Коэна'
FLEISS = 'Каппа Флейса'
ALPHA = 'Альфа Криппендорфа'
GWET = 'Gwet AC1'
CLASS_AGREEMENT = 'Согласованность по классу'
SCALES = {'nominal', 'ordinal', 'interval', 'multilabel'}


class UnavailableAgreement(ValueError):
    """Required data are absent or insufficient for a criterion."""


@dataclass(frozen=True)
class Criterion:
    """Measurement scale and calculation options agreed for a criterion."""

    scale: str
    categories: tuple
    weights: str
    multilabel_method: str
    assumed_scale: bool


@dataclass(frozen=True)
class Ratings:
    """Matrices for initial agreement, the main coefficient and final labels."""

    initial: pd.DataFrame
    main: pd.DataFrame
    aggregation: pd.DataFrame
    full: bool
    novel: int


def _is_missing(value):
    return pd.api.types.is_scalar(value) and pd.isna(value)


def _criterion(series, options):
    ordered = isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered
    scale = options.get('scale')
    assumed = scale is None and not ordered
    if scale is None:
        scale = 'nominal'
        if ordered:
            scale = 'ordinal'
        elif any(isinstance(value, SET_LABEL_TYPES) for value in series):
            scale = 'multilabel'
        elif series.nunique() > 2 and (
            pd.api.types.infer_dtype(series) in {'integer', 'floating', 'mixed-integer-float'}
            or canonical_integer_labels(series.dropna()) is not None
        ):
            scale = 'ordinal'
    if scale not in SCALES:
        raise ValueError(f'Неизвестная шкала: {scale!r}.')
    categories = tuple(options.get('categories', list(series.cat.categories) if ordered else ()))
    if len(set(categories)) != len(categories) or any(_is_missing(v) for v in categories):
        raise ValueError('Категории шкалы должны быть уникальными и непустыми.')
    weights = options.get('weights', 'linear')
    multilabel = options.get('multilabel_method', 'per_label')
    if weights not in {'linear', 'quadratic'} or multilabel not in {'per_label', 'jaccard'}:
        raise ValueError('Используйте веса linear/quadratic и многометочный режим per_label/jaccard.')
    return Criterion(scale, categories, weights, multilabel, assumed)


def _initial_mask(frame, flag, order):
    if flag:
        if flag not in frame.columns or not frame[flag].isin((True, False)).all():
            raise UnavailableAgreement('Нужен корректный признак первоначальной оценки.')
        return frame[flag].astype(bool)
    if not order or order not in frame:
        raise UnavailableAgreement('Для динамического перекрытия нужен признак или порядок первоначальных оценок.')
    ranks = pd.to_numeric(frame[order], errors='coerce')
    if not np.isfinite(ranks.to_numpy(dtype=float)).all():
        raise UnavailableAgreement('Порядок динамических оценок должен содержать конечные числа.')
    ordered = frame[IDS].assign(sequence=ranks).drop_duplicates()
    if ordered.duplicated(IDS).any():
        raise UnavailableAgreement('Для одного асессора и объекта заданы разные порядковые номера.')
    ranks = ordered.groupby('session_id', sort=False)['sequence'].rank(method='min')
    initial_ids = ordered.loc[ranks <= 2, IDS]
    if (initial_ids.groupby('session_id').size() > 2).any():
        raise UnavailableAgreement('Не удалось однозначно выделить две первоначальные оценки.')
    selected = pd.MultiIndex.from_frame(frame[IDS]).isin(pd.MultiIndex.from_frame(initial_ids))
    return pd.Series(selected, index=frame.index)


def _normalize_scores(series, scale):
    if scale == 'interval':
        numeric = pd.to_numeric(series, errors='raise')
        if not np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)).all():
            raise ValueError('Количественные оценки должны быть конечными числами.')
        return numeric
    if scale == 'multilabel':
        return series.map(as_label_set)
    for value in series:
        hash(value)
    return series


def _build_matrix(rows, sessions):
    if rows.empty:
        return pd.DataFrame(index=sessions)
    matrix = rows.assign(score=rows['score'].astype(object)).pivot(
        index='session_id', columns='assessor_id', values='score'
    )
    return matrix.reindex(sessions)


def _ordered_initial_mask(frame, order):
    assigned = frame.loc[frame[order].notna() | frame['score'].notna()]
    if assigned[IDS].isna().any().any():
        raise UnavailableAgreement('В порядке оценок отсутствует ID объекта или асессора.')
    return _initial_mask(assigned, None, order)


def _valid_rating_rows(frame, scale):
    frame = frame.dropna(subset=['score']).copy()
    if frame[IDS].isna().any().any():
        raise UnavailableAgreement('У оценок отсутствует ID объекта или асессора.')
    frame['score'] = _normalize_scores(frame['score'], scale)
    frame = frame.drop_duplicates()
    if frame.duplicated(IDS).any():
        raise UnavailableAgreement('Обнаружено несколько разных оценок одного асессора для объекта.')
    return frame


def _select_ratings(data, column, scale, *, sessions, dynamic, flag, order):
    metadata = flag or order
    require = [metadata] if dynamic and metadata in data.columns else []
    frame = data.loc[:, [*IDS, column, *require]].rename(columns={column: 'score'})
    initial_by_order = _ordered_initial_mask(frame, order) if dynamic and not flag and order in frame else None
    frame = _valid_rating_rows(frame, scale)
    if dynamic:
        mask = _initial_mask(frame, flag, order) if initial_by_order is None else initial_by_order.reindex(frame.index)
        initial = frame.loc[mask]
    else:
        initial = frame
    matrix = _build_matrix(initial, sessions)
    deficient = int((matrix.notna().sum(axis=1) < 2).sum())
    if matrix.empty or deficient:
        raise UnavailableAgreement(f'Объектов с менее чем двумя независимыми оценками: {deficient} из {len(sessions)}.')

    aggregation = matrix
    if dynamic:
        votes = initial.groupby('session_id', sort=False)['score'].agg(set).to_dict()
        further = frame.loc[~frame.index.isin(initial.index)]
        if not further.empty:
            aggregation = _build_matrix(frame, sessions)
        novel = further.loc[
            [
                score not in votes[session]
                for session, score in further[['session_id', 'score']].itertuples(index=False, name=None)
            ]
        ]
        if len(novel):
            independent = further.loc[further['assessor_id'].isin(novel['assessor_id'])]
            main = _build_matrix(pd.concat([initial, independent]), sessions)
            return Ratings(matrix, main, aggregation, False, len(novel))
    return Ratings(matrix, matrix, aggregation, bool(matrix.notna().all().all()), 0)


def _label_rows(matrix):
    return [
        tuple(value for value in row if not _is_missing(value)) for row in matrix.itertuples(index=False, name=None)
    ]


def _target_rates(final, categories):
    counts = Counter(final.dropna())
    return {category: counts.get(category, 0) for category in categories}, sum(counts.values())


def _rate_band(count, total):
    pct = count * 100
    if count == 0 or count == total:
        return 'absent' if count == 0 else 'single'
    if total <= pct < 5 * total or 95 * total < pct <= 99 * total:
        return 'imbalanced'
    if pct < total or pct > 99 * total:
        return 'extreme'
    return 'standard'


def _agreement_by_object(rows):
    return np.array([sum(n * (n - 1) for n in Counter(row).values()) / (len(row) * (len(row) - 1)) for row in rows])


def _ac1(rows, categories):
    # The Gwet marginal probabilities give equal weight to each object.
    probabilities = dict.fromkeys(categories, 0.0)
    for row in rows:
        for label, n in Counter(row).items():
            probabilities[label] += n / len(row)
    if len(categories) < 2:
        return None
    marginals = np.array(list(probabilities.values())) / len(rows)
    expected = (marginals * (1 - marginals)).sum() / (len(categories) - 1)
    return float((_agreement_by_object(rows).mean() - expected) / (1 - expected))


def _fleiss(rows):
    counts = Counter(label for row in rows for label in row)
    n = sum(counts.values())
    expected = sum((count / n) ** 2 for count in counts.values())
    return float((_agreement_by_object(rows).mean() - expected) / (1 - expected)) if expected < 1 else None


def _cohen(matrix, categories, weights=None):
    to_code = {label: i for i, label in enumerate(categories)}
    a, b = (matrix[column].map(to_code).to_numpy(dtype=int) for column in matrix.columns)
    if len(set(a) | set(b)) < 2:
        return None
    return float(cohen_kappa_score(a, b, labels=list(range(len(categories))), weights=weights))


def _alpha_interval(rows):
    origin = rows[0][0]
    groups = [np.asarray([value - origin for value in row], dtype=float) for row in rows]
    scale = max(np.abs(group).max() for group in groups)
    if scale == 0:
        return None
    if np.isinf(scale):
        scale = max(abs(value) for row in rows for value in row)
        groups = [np.asarray(row, dtype=float) for row in rows]
    groups = [group / scale for group in groups]
    # Ordered pair squared differences sum to twice n times the centred sum of squares.
    merged = np.concatenate(groups)
    expected = 2 * len(merged) * np.square(merged - merged.mean()).sum() / (len(merged) - 1)
    observed = sum(2 * len(group) * np.square(group - group.mean()).sum() / (len(group) - 1) for group in groups)
    return float(1 - observed / expected) if expected > 0 else None


def _nominal_alpha(rows, totals):
    n = sum(totals.values())
    expected = (n * n - sum(c * c for c in totals.values())) / (n - 1)
    observed = sum((len(row) ** 2 - sum(c * c for c in Counter(row).values())) / (len(row) - 1) for row in rows)
    return float(1 - observed / expected) if expected > 0 else None


def _weighted_alpha(rows, totals, distance):
    n = sum(totals.values())
    observed = sum(sum(distance(a, b) for a, b in combinations(row, 2)) / (len(row) - 1) for row in rows)
    expected = sum(totals[a] * totals[b] * distance(a, b) for a, b in combinations(totals, 2)) / (n - 1)
    return float(1 - observed / expected) if expected > 0 else None


def _ordinal_alpha(rows, categories):
    totals = Counter(label for row in rows for label in row)
    centres, prefix = {}, 0
    for label in categories:
        centres[label] = prefix + totals[label] / 2
        prefix += totals[label]
    return _weighted_alpha(rows, totals, lambda a, b: (centres[a] - centres[b]) ** 2)


def _jaccard(a, b):
    union = a | b
    return 1 - len(a & b) / len(union) if union else 0.0


def _statistic(ratings, spec, categories):
    rows = _label_rows(ratings.main)
    if spec.scale == 'interval':
        return f'{ALPHA} (interval)', _alpha_interval(rows)
    if spec.scale == 'multilabel':
        totals = Counter(label for row in rows for label in row)
        return f'{ALPHA} (Jaccard)', _weighted_alpha(rows, totals, _jaccard)
    if ratings.full and ratings.main.shape[1] == 2:
        weights = spec.weights if spec.scale == 'ordinal' else None
        return f'{COHEN} ({weights})' if weights else COHEN, _cohen(ratings.main, categories, weights)
    if spec.scale == 'ordinal':
        return f'{ALPHA} (ordinal)', _ordinal_alpha(rows, categories)
    if ratings.full:
        return FLEISS, _fleiss(rows)
    totals = Counter(label for row in rows for label in row)
    return ALPHA, _nominal_alpha(rows, totals)


def _categories(matrix, spec):
    observed = {v: None for row in _label_rows(matrix) for v in row}
    if spec.categories:
        missing = set(observed) - set(spec.categories)
        if missing:
            raise ValueError('Обнаружены метки вне заданного набора категорий.')
        return spec.categories
    if spec.scale == 'ordinal':
        strings = canonical_integer_labels(pd.Series(list(observed)))
        if strings is not None:
            return tuple(sorted(observed, key=int))
        if not all(isinstance(v, (int, float, np.number)) for v in observed):
            raise UnavailableAgreement('Для нечисловой порядковой шкалы нужен заданный порядок категорий.')
        return tuple(sorted(observed))
    return tuple(observed)


def _dawid_skene(matrix):
    rows = matrix.rename_axis(index='task', columns='worker').stack().rename('label').reset_index()
    try:
        predicted = aggregate_dawid_skene(rows)
    except ImportError:
        return None, 'Финальные метки недоступны. Для агрегации Дэвида–Скина нужен crowd-kit.'
    note = (
        'Единственная наблюдаемая категория.'
        if rows['label'].nunique() == 1
        else 'Финальные метки рассчитаны Дэвидом–Скином.'
    )
    return predicted.reindex(matrix.index), note


def _final_series(raw, column, sessions):
    if column not in raw:
        return None
    if 'session_id' in raw.columns:
        raw = raw.set_index('session_id')
    if raw.index.name != 'session_id':
        return None
    if not raw.index.is_unique or raw.index.hasnans:
        raise UnavailableAgreement('Финальные оценки должны иметь уникальные и непустые session_id.')
    return raw[column].reindex(sessions)


def _label_text_keys(label):
    keys = {str(label)}
    if isinstance(label, (float, np.floating)) and np.isfinite(label) and label.is_integer():
        keys.add(str(int(label)))
    return keys


def _align_final_categories(final, categories):
    domain = set(categories)
    by_text = {}
    for category in categories:
        for key in _label_text_keys(category):
            by_text.setdefault(key, set()).add(category)
    replacements = {}
    for label in final.unique():
        if label in domain:
            continue
        candidates = set().union(*(by_text.get(key, ()) for key in _label_text_keys(label)))
        if len(candidates) != 1:
            raise UnavailableAgreement('В финальной разметке есть метки вне шкалы индивидуальных оценок.')
        replacements[label] = candidates.pop()
    aligned = final.map(lambda label: replacements.get(label, label))
    note = ' Типы финальных меток согласованы с исходной шкалой.' if replacements else ''
    return aligned, note


def _row(check, value=None, *, primary=False, label=None, rate=None, comment=''):
    value = float(value) if value is not None and np.isfinite(value) else None
    return dict(check=check, value=value, primary=primary, label=label, target_rate=rate, comment=comment)


def _main_row(ratings, spec, categories, *, primary=True):
    method, value = _statistic(ratings, spec, categories)
    reason = '' if value is not None else 'Коэффициент не определён при отсутствии ожидаемого расхождения.'
    return _row(method, value, primary=primary, comment=reason)


def _class_rows(ratings, final, rates, bands, regime, label):
    by_object = pd.Series(_agreement_by_object(_label_rows(ratings.initial)), index=ratings.initial.index)
    for category, count in rates.items():
        band = bands[category]
        rate = count / len(final)
        match = final.map(lambda value, category=category: category == value)
        if band == 'absent':
            yield _row(
                CLASS_AGREEMENT, label=label(category), rate=0.0, comment='Класс отсутствует в финальной разметке.'
            )
            continue
        if band == 'extreme':
            yield _row(
                CLASS_AGREEMENT,
                label=label(category),
                rate=rate,
                comment='Класс исключён из расчёта из-за экстремального дисбаланса.',
            )
            continue
        primary = regime in {'extreme', 'single'} or (regime == 'imbalanced' and rate < 0.05)
        yield _row(
            CLASS_AGREEMENT, by_object[match[match].index].mean(), primary=primary, label=label(category), rate=rate
        )


def _nominal_rows(ratings, spec, final, source, *, label=str):
    categories = _categories(ratings.main, spec)
    final, normalized = _align_final_categories(final.dropna(), categories)
    source += normalized
    counts, n = _target_rates(final, categories)
    bands = {category: _rate_band(count, n) for category, count in counts.items()}
    diagnostic = [
        _row('Target rate', label=label(category), rate=count / n, comment=f'{count} из {n} финальных меток. {source}')
        for category, count in counts.items()
    ]
    present = set(bands.values())
    if 'single' in present:
        regime = 'single'
    elif 'extreme' in present:
        regime = 'extreme'
    elif 'imbalanced' in present:
        regime = 'imbalanced'
    else:
        return [_main_row(ratings, spec, categories), *diagnostic]
    computed = list(_class_rows(ratings, final, counts, bands, regime, label))
    if regime == 'imbalanced':
        coefficient = _ac1(_label_rows(ratings.main), categories)
        computed.insert(0, _row(GWET, coefficient, primary=True))
    return [*computed, *diagnostic]


def _binary_matrix(matrix, label):
    return matrix.apply(lambda col: col.map(lambda value: None if _is_missing(value) else int(label in value)))


def _binary_ratings(ratings, label):
    initial = _binary_matrix(ratings.initial, label)
    if not ratings.novel:
        return replace(ratings, initial=initial, main=initial)
    aggregation = _binary_matrix(ratings.aggregation, label)
    initial_with_all_raters = initial.reindex(columns=aggregation.columns)
    additional = aggregation.where(initial_with_all_raters.isna())
    novel = additional
    for vote in (0, 1):
        repeated = additional.eq(vote).mul(initial.eq(vote).any(axis=1), axis=0)
        novel = novel.mask(repeated)
    unique = novel.notna()
    count = int(unique.sum().sum())
    if not count:
        return replace(ratings, initial=initial, main=initial, full=bool(initial.notna().all().all()), novel=0)
    independent = additional.loc[:, unique.any(axis=0)]
    main = initial.combine_first(independent)
    return replace(ratings, initial=initial, main=main, full=False, novel=count)


def _agreement_for_label(ratings, spec, final, source, column, label):
    binary = _binary_ratings(ratings, label)
    target = final.map(lambda value: int(label in value))
    binary_spec = replace(spec, scale='nominal', categories=(0, 1))
    rows = _nominal_rows(binary, binary_spec, target, source, label=lambda value: f'{label}={value}')
    for row in rows:
        if row['label'] is None:
            row['label'] = str(label)
    return _annotate(rows, column, spec, binary)


def _multilabel_rows(ratings, spec, final, source, column):
    checked = list(value for row in _label_rows(ratings.main) for value in row)
    if final is not None:
        final = final.dropna().map(as_label_set)
        checked.extend(final)
    observed = set().union(*checked)
    if spec.categories and observed - set(spec.categories):
        raise ValueError('Обнаружены метки вне заданного набора многометочных категорий.')
    if spec.multilabel_method == 'jaccard':
        return list(_annotate([_main_row(ratings, spec, ())], column, spec, ratings))
    result = []
    for label in spec.categories or sorted(observed, key=str):
        result.extend(_agreement_for_label(ratings, spec, final, source, column, label))
    if not result:
        rows = [_row('Применимость', comment='В размеченных наборах нет ни одной метки.')]
        return list(_annotate(rows, column, spec, ratings))
    return result


def _annotate(rows, column, spec, ratings):
    overlap = ratings.initial.notna().sum(axis=1)
    notes = []
    if spec.assumed_scale:
        notes.append(f'Шкала определена без метаданных: {spec.scale}.')
    if ratings.novel:
        notes.append(f'Новых меток арбитров: {ratings.novel}. Основной расчёт выполнен для неполного перекрытия.')
    for row in rows:
        comment = ' '.join([*notes, row['comment']]).strip()
        if row['value'] is not None and not comment:
            comment = 'Основной коэффициент по правилам v4.' if row['primary'] else 'Дополнительный результат.'
        yield dict(
            row,
            criterion=column,
            scale=spec.scale,
            n_objects=len(overlap),
            n_assessors=ratings.main.shape[1],
            min_overlap=int(overlap.min()),
            max_overlap=int(overlap.max()),
            overlap='полное' if ratings.full else 'неполное',
            comment=comment,
        )


def _criterion_final_labels(final, column, sessions, ratings):
    labels = _final_series(final, column, sessions)
    if labels is None:
        return _dawid_skene(ratings.aggregation)
    return labels, 'Использованы переданные финальные метки.'


def _criterion_results(data, final, column, options, *, sessions, dynamic, flag, order):
    try:
        spec = _criterion(data[column], options)
        ratings = _select_ratings(data, column, spec.scale, sessions=sessions, dynamic=dynamic, flag=flag, order=order)
        requires_final = spec.scale == 'nominal' or (
            spec.scale == 'multilabel' and spec.multilabel_method == 'per_label'
        )
        labels, source = (None, '')
        if requires_final:
            labels, source = _criterion_final_labels(final, column, sessions, ratings)
            if labels is None or labels.dropna().empty:
                reason = source if labels is None else 'Нет валидных финальных меток.'
                return list(_annotate([_row('Применимость', comment=reason)], column, spec, ratings))
        if spec.scale == 'nominal':
            rows = _nominal_rows(ratings, spec, labels, source)
        elif spec.scale == 'multilabel':
            return _multilabel_rows(ratings, spec, labels, source, column)
        else:
            rows = [_main_row(ratings, spec, _categories(ratings.main, spec) if spec.scale == 'ordinal' else ())]
        return list(_annotate(rows, column, spec, ratings))
    except (ValueError, TypeError, KeyError) as error:
        return [dict(_row('Применимость', comment=str(error)), criterion=column, n_objects=len(sessions))]


def _color(value, method, thresholds):
    if value is None:
        return 'gray'
    group = 'class' if method in {GWET, CLASS_AGREEMENT} else 'coefficient'
    low, high = thresholds.get(method.partition(' (')[0], thresholds[group])
    # Allow rounding noise from the coefficient arithmetic at inclusive boundaries.
    for bound in (low, high):
        if abs(value - bound) <= 8 * np.finfo(float).eps:
            value = bound
            break
    return semaphore_by_threshold(value, (low, high), left_border_is_bigger=False, right_border_is_bigger=False)


def report_agreement(rows, *, is_info, thresholds, cache_version=CACHE_VERSION):
    """Build the library result and summarize the worst primary color."""
    colored_rows = [(row, 'gray' if is_info else _color(row['value'], row['check'], thresholds)) for row in rows]
    primary = [color for row, color in colored_rows if row['primary']]
    table = pd.DataFrame(
        [
            {
                'Критерий': row.get('criterion'),
                'Шкала': row.get('scale'),
                'Перекрытие': row.get('overlap'),
                'Объектов': row.get('n_objects'),
                'Асессоров': row.get('n_assessors'),
                'Мин. оценок': row.get('min_overlap'),
                'Макс. оценок': row.get('max_overlap'),
                'Наименование проверки': row['check'],
                'Класс': row['label'],
                'Target rate': row['target_rate'],
                'Согласие': row['value'],
                'Основная проверка': 'да' if row['primary'] else 'нет',
                'Светофор': color,
                'Комментарий': row['comment'],
            }
            for row, color in colored_rows
        ]
    )
    return {
        'report': {
            'semaphore': worst_semaphore(primary),
            'result_plots': [],
            'result_dataframes': [table.astype(object).where(table.notna(), None)],
        },
        'precomputed': {'criteria_stats': rows, 'methodology_version': VERSION, 'cache_version': cache_version},
    }


def run_agreement(data, final, columns, specs, *, dynamic, flag, order, is_info, thresholds):
    """Calculate GenAI v4 agreement on a native Sampler's rating tables."""
    columns = (
        list(dict.fromkeys([*[c for c in data.columns if str(c).endswith('_metric')], *specs]))
        if columns is None
        else list(dict.fromkeys(columns))
    )
    if 'session_id' not in data.columns or 'assessor_id' not in data.columns or not columns:
        return report_agreement(
            [_row('Применимость', comment='Нужны session_id, assessor_id и столбцы оценок.')],
            is_info=is_info,
            thresholds=thresholds,
        )
    data = data.reset_index(drop=True)
    flag = flag or ('is_initial' if 'is_initial' in data and order is None else None)
    order = order or ('rating_order' if flag is None and 'rating_order' in data else None)
    dynamic = bool(flag or order) if dynamic is None else dynamic
    sessions = pd.Index(data['session_id'].dropna().unique(), name='session_id')
    rows = []
    for column in columns:
        rows.extend(
            _criterion_results(
                data, final, column, specs.get(column, {}), sessions=sessions, dynamic=dynamic, flag=flag, order=order
            )
        )
    return report_agreement(rows, is_info=is_info, thresholds=thresholds)
