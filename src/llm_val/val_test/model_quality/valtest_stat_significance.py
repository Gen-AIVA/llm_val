import numpy as np
import typing as tp
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon, ttest_1samp, ttest_ind
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from llm_val.sampler.Sampler import Sampler
from llm_val.scorer.BinaryClassScorer import BinaryClassScorer
from llm_val.scorer.NerScorer import NerScorer
from llm_val.scorer.Scorer import Scorer
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.utils.metrics import METRICS
from llm_val.utils.ner_metrics import NER_METRICS
from llm_val.utils.report_helper import semaphore_by_threshold
from llm_val.val_test.model_quality.stat_significance_v4 import Hypothesis, NerMatching, TERMS, run_significance
from llm_val.val_test.model_quality.stat_significance_v4 import metric_kind as normalize_metric_kind


_BINARY_SCORERS = {'accuracy': accuracy_score, 'precision': precision_score, 'recall': recall_score, 'f1': f1_score}
_NER_MICRO_F1 = {'ner_exact_match_micro_f1', 'ner_partial_match_micro_f1'}


def report_valtest_stat_significance(p_value: float,
                                     extra_samplers_keys: tp.List=[],
                                     data_type: str='test',
                                     semaphore_threshold: tp.Tuple[float, float]=(0.01, 0.05),
                                     figure: plt.Figure=None,
                                     is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        p_value: Значение p-value
        extra_samplers_keys: Названия сравниваемых моделей
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold: Границы светофора. Указываются в порядке возрастания
        figure: График с визуализацией доверительных интервалов
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    samplers_str = '' if not extra_samplers_keys else ' ' + ', '.join(extra_samplers_keys)
    color = 'gray' if is_info else semaphore_by_threshold(value=p_value,
                                                          threshold=semaphore_threshold,
                                                          greater_is_better=False,
                                                          left_border_is_bigger=True,
                                                          right_border_is_bigger=False)
    res_table = pd.DataFrame({f'Значение p-value на {data_type}{samplers_str}': [p_value],
                              'Результат теста': [color]})
    fig_list = [figure] if figure is not None else []
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = fig_list
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_stat_significance(sampler: tp.Union[Sampler, tp.Dict],
                              scorer: tp.Optional[Scorer] = None,
                              metric_agg: str = 'single_mean',
                              data_type: str = 'test',
                              main_column: tp.Optional[str] = None,
                              value_threshold: float = 0.5,
                              extra_samplers: tp.Optional[tp.Dict[str, tp.Any]] = None,
                              same_questions: bool = False,
                              semaphore_threshold: tp.Tuple[float, float] = (0.01, 0.05),
                              bootstrap: bool = False,
                              bootstrap_iterations: int = 1000,
                              bootstrap_ci_type: str = 'percentile',
                              alternative: tp.Optional[str] = None,
                              exec_env: str = 'default',
                              random_state: tp.Optional[int] = None,
                              is_info: bool = True,
                              p_value: tp.Optional[float] = None,
                              calc_metric_fn: tp.Optional[tp.Callable] = None,
                              *,
                              metric_kind: tp.Optional[str] = None,
                              metric_thresholds: tp.Optional[tp.Mapping[str, float]] = None,
                              comparison_margin: float = 0.0,
                              id_column: tp.Optional[str] = None,
                              positive_label: tp.Any = 1,
                              methods: tp.Optional[tp.Sequence[str]] = None,
                              significance_level: tp.Optional[float] = None,
                              main_metric: tp.Optional[str] = None,
                              ner_exact_match: tp.Optional[bool] = None,
                              ner_ignore_repetitions: bool = False,
                              **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    """Run GenAI 5.3.4 with the v4 methodology.

    Set metric_agg or metric_kind to accuracy, precision, recall, f1, single_mean,
    or custom. Custom metrics use the scorer key given by metric_agg or the
    supplied calc_metric_fn. Classification metrics use one row per independent
    object with true_answer/model_answer or TP/FP/FN/TN counts. Set positive_label
    for binary labels other than 0/1. For accuracy, binary scores are also accepted.
    For NER, extracted/target entity dictionaries are supported for pooled
    Precision, Recall and F1 using the sampler's entity_types. Configure the
    matching mode with ner_exact_match and ner_ignore_repetitions when needed.
    A configured scorer with a different calculation for a classification
    alias is evaluated as a custom metric. Use main_metric to select one of
    several outputs, or set metric_kind explicitly to choose the calculation.

    Provide value_threshold for a business threshold and metric_thresholds for
    additional classification thresholds. For version comparisons, use
    extra_samplers and comparison_margin. For paired versions also set
    same_questions=True and provide a common id_column if it cannot be inferred.
    The default direction follows the v4 method table: less for comparisons of
    per-object numeric scores with Welch/Wilcoxon, greater for the other cases.
    alternative applies to the primary hypothesis. Additional classification
    thresholds use the upper tail. A second threshold for the primary metric
    reuses the selected metric definition.

    Set bootstrap=True to report bootstrap alongside the default method or pass
    the agreed methods explicitly. With several hypotheses, each uses the
    compatible methods from that list, or its default if none apply. Pass
    significance_level to include a conclusion at that level. The report names
    the effect estimated by each method and explains how their results relate.
    The default is_info=True yields a gray result. With
    is_info=False, semaphore_threshold sets the p-value bounds (0.01 and 0.05
    by default). The overall color follows the primary p-value. The precomputed
    p_value argument is accepted for compatibility and the test recomputes it
    from the supplied data.
    """
    baselines = {
        name: _significance_frames(sample, data_type, exec_env)
        for name, sample in (extra_samplers or {}).items()
    }
    frames = _significance_frames(sampler, data_type, exec_env)
    accuracy_column = None if 'metric_value' in frames[0] else main_column
    metric_name = _metric_alias(metric_agg)
    scorer_metrics = getattr(scorer, 'metrics', {}) or {}
    score_key = metric_agg
    if calc_metric_fn is None and (metric_kind is None or normalize_metric_kind(metric_kind) == 'custom'):
        score_key = _scorer_metric_key(scorer_metrics, metric_agg, metric_name)
    kind = _significance_metric_kind(
        metric_kind, metric_name, scorer, scorer_metrics, score_key, calc_metric_fn, main_metric,
    )
    if kind == 'mean' and main_metric is not None and _uses_standard_scorer(
        scorer, scorer_metrics.get(score_key), kind, metric_name, main_metric, score_key,
    ):
        main_column = main_metric[len(score_key) + 1:]
    ner = _significance_ner_matching(sampler, extra_samplers, metric_name, ner_exact_match, ner_ignore_repetitions)
    reference = comparison_margin if baselines else value_threshold
    metric = calc_metric_fn
    if metric is None and scorer is not None and kind == 'custom':
        metric = partial(_score_significance_metric, scorer=scorer, metric_agg=score_key,
                         data_type=data_type, main_column=main_column, main_metric=main_metric, ner=ner,
                         is_ner_micro_f1=metric_name in _NER_MICRO_F1)
    primary = Hypothesis(metric_agg, kind, float(reference), metric, bool(baselines), main_column)
    hypotheses = [primary, *_additional_hypotheses(primary, metric_name, metric_thresholds, accuracy_column)]
    return run_significance(
        frames, baselines, hypotheses,
        paired=same_questions, id_column=id_column, positive_label=positive_label, ner=ner,
        alternative=alternative, methods=methods, include_bootstrap=bootstrap, iterations=bootstrap_iterations,
        ci_type=bootstrap_ci_type, random_state=random_state, significance_level=significance_level,
        is_info=is_info, semaphore_threshold=semaphore_threshold,
    )


def _additional_hypotheses(primary, metric_name, metric_thresholds, accuracy_column):
    for name, threshold in (metric_thresholds or {}).items():
        additional_kind = _metric_alias(name)
        if _is_primary_metric(additional_kind, primary.kind, metric_name):
            yield replace(primary, label=name, reference=float(threshold), compare_versions=False)
            continue
        if additional_kind not in TERMS:
            raise ValueError(f'Неизвестная метрика классификации в metric_thresholds: {name!r}.')
        column = accuracy_column if additional_kind == 'accuracy' else None
        yield Hypothesis(name, additional_kind, float(threshold), None, False, column)


def _is_primary_metric(candidate, primary_kind, metric_name):
    if primary_kind == 'custom':
        return candidate == metric_name
    if candidate in TERMS or candidate == 'mean':
        return candidate == primary_kind
    return candidate == metric_name and (candidate not in _NER_MICRO_F1 or primary_kind == 'f1')


def _significance_frames(sampler, data_type, exec_env):
    data = sampler[data_type] if exec_env == 'sberds' or isinstance(sampler, dict) else getattr(sampler, data_type)
    return pd.DataFrame(data['y']), pd.DataFrame(data.get('X'))


def _metric_alias(name):
    return normalize_metric_kind(name).replace('ner_metrics_', 'ner_', 1)


def _scorer_metric_key(metrics, metric_agg, name):
    if metric_agg in metrics or (name not in TERMS and name not in _NER_MICRO_F1 and name != 'mean'):
        return metric_agg
    aliases = [key for key in metrics if _metric_alias(key) == name]
    if len(aliases) > 1:
        raise ValueError(f'Для {metric_agg!r} найдено несколько метрик scorer. Укажите точный ключ: {aliases}.')
    return aliases[0] if aliases else metric_agg


def _significance_metric_kind(explicit, name, scorer, scorer_metrics, score_key, metric, main_metric):
    if explicit is not None:
        kind = normalize_metric_kind(explicit)
        if kind not in TERMS and kind not in {'mean', 'custom'}:
            raise ValueError(f'Неизвестный тип метрики: {explicit!r}.')
        return kind
    if metric is not None:
        return 'custom'
    kind = 'f1' if name in _NER_MICRO_F1 else name
    if kind not in TERMS and kind != 'mean':
        return 'custom'
    configured = scorer_metrics.get(score_key)
    if configured is not None:
        return kind if _uses_standard_scorer(scorer, configured, kind, name, main_metric, score_key) else 'custom'
    if kind == 'mean' and len(scorer_metrics) == 1:
        (key, config), = scorer_metrics.items()
        accuracy = _metric_alias(key) == 'accuracy' and _uses_standard_scorer(
            scorer, config, 'accuracy', 'accuracy', main_metric, key,
        )
        return 'accuracy' if accuracy else kind
    return kind


def _uses_standard_scorer(scorer, configured, kind, name, main_metric, score_key):
    if not isinstance(configured, tp.Mapping):
        return False
    implementation = getattr(scorer.calc, '__func__', None)
    call = configured.get('call')
    if kind == 'mean':
        selector = f'{score_key}_'
        return (implementation is UniversalScorer.calc and call is METRICS['single_mean']['call']
                and configured.get('is_singlecol')
                and (main_metric is None or (main_metric.startswith(selector) and len(main_metric) > len(selector))))
    if configured.get('is_singlecol', True):
        return False
    if name in _NER_MICRO_F1:
        return (implementation is NerScorer.calc and call is NER_METRICS[name]['call']
                and main_metric in (None, 'micro_f1'))
    return (implementation is BinaryClassScorer.calc and call is _BINARY_SCORERS[kind]
            and main_metric in (None, score_key))


def _significance_ner_matching(sampler, extra_samplers, ner_key, exact_match, ignore_repetitions):
    entities = getattr(sampler, 'entity_types', None)
    if entities is not None:
        entities = tuple(entities)
        for baseline in (extra_samplers or {}).values():
            baseline_types = getattr(baseline, 'entity_types', None)
            if baseline_types is not None and set(entities) != set(baseline_types):
                raise ValueError('Для сравнения версий NER должны использоваться одинаковые типы сущностей.')
    if exact_match is None:
        exact_match = ner_key != 'ner_partial_match_micro_f1'
    return NerMatching(entities, exact_match, ignore_repetitions)


def _score_significance_metric(rows, scorer, metric_agg, data_type, main_column, main_metric, ner, is_ner_micro_f1):
    sample = SimpleNamespace(**{data_type: {'y': rows}}, entity_types=ner.entity_types)
    values = scorer.calc(sampler=sample, data_type=data_type, metric_name=metric_agg)
    if main_metric is not None:
        if main_metric not in values:
            raise ValueError(f'Scorer не вернул указанную метрику {main_metric!r}.')
        return float(values[main_metric])
    if metric_agg in values:
        return float(values[metric_agg])
    if is_ner_micro_f1 and 'micro_f1' in values:
        return float(values['micro_f1'])
    key = f'{metric_agg}_{main_column}'
    if key in values:
        return float(values[key])
    if len(values) == 1:
        return float(next(iter(values.values())))
    raise ValueError(f'Scorer не вернул единственное значение для метрики {metric_agg!r}.')


def valtest_stat_significance_legacy(sampler: tp.Union[Sampler, tp.Dict],
                              scorer: tp.Optional[Scorer],
                              metric_agg: str,
                              data_type: str='test',
                              main_column: tp.Optional[str]=None,
                              value_threshold: float=0.5,
                              extra_samplers: tp.Optional[tp.Dict[str, tp.Any]]=None,
                              same_questions: bool=False,
                              semaphore_threshold: tp.Tuple[float, float]=(0.01, 0.05),
                              bootstrap: bool=False,
                              bootstrap_iterations: int=1000,
                              bootstrap_ci_type: str='percentile',
                              alternative: str='greater',
                              exec_env: str='default',
                              random_state: tp.Optional[int]=None,
                              is_info: bool=False,
                              p_value: tp.Optional[float]=None,
                              calc_metric_fn: tp.Optional[tp.Callable]=None,
                              **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на статзначимость различия ключевой метрики качества
    Parameters:
        sampler: Объект класса Sampler
        scorer: Объект класса Scorer. Используется при exec_env='default'
        metric_agg: Название агрегации метрики.
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        main_column: Название колонки, содержащей ключевую метрику качества; если None, то усредняются все колонки в y
        value_threshold: Пороговое значение ключевой метрики качества при использовании 1-выбор-ого крит-ия Стьюдента
        extra_samplers: Дополнительные объекты Sampler, используемые, если в тесте участвует более одной модели
        same_questions: True, если разные модели отвечали на одинаковые вопросы; False, если на разные
        semaphore_threshold: Границы светофора
        bootstrap: Флаг для использования bootstrap
        bootstrap_iterations: Число bootstrap итераций
        bootstrap_ci_type: Тип доверительного интервала ('percentile', 'basic', 'normal')
        alternative: Тип альтернативной гипотезы ('greater', 'less', 'two-sided')
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
        random_state: seed генератора случайных чисел
        is_info: True, если тест информативный
        p_value: Опциональное значение p-value.
            Если не None, то подсчет статистики не производится, и на основе переданного аргумента выставляется светофор.
        calc_metric_fn: Пользовательская функция расчета метрики.
            Используется при metric_agg != 'single_mean' и scorer is None.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    extra_samplers_keys = [] if extra_samplers is None else list(extra_samplers.keys())
    original_metric_y = get_real_metric(sampler,
                                        data_type,
                                        exec_env,
                                        main_column)
    figure = None
    metric_fn = np.mean
    bootstrap_data = None
    precomputed = {}
    if bootstrap:
        if not main_column:
            raise ValueError('Для режима bootstrap необходимо явно указать параметр main_column.')

        y_data = get_data(sampler, data_type, 'y', exec_env)
        valid_mask = y_data[main_column].notna()
        bootstrap_data = y_data[valid_mask].reset_index(drop=True)
        
        if metric_agg == 'single_mean':
            strategy = 'rigorous'
            bootstrap_data = bootstrap_data[main_column].values
            metric_fn = np.mean
        else:
            strategy = 'empirical'
            if scorer is not None:
                metric_fn = partial(calc_metric_values,
                                    scorer=scorer,
                                    metric_agg=metric_agg,
                                    data_type=data_type)
            elif calc_metric_fn is not None:
                metric_fn = calc_metric_fn
            else:
                raise ValueError(
                    "Для режима bootstrap с metric_agg != 'single_mean' необходимо передать "
                    "либо scorer, либо пользовательскую функцию расчета метрики через параметр calc_metric_fn."
                )

        ci_low, ci_high = bootstrap_confidence_interval(bootstrap_data,
                                                        bootstrap_iterations,
                                                        ci_type=bootstrap_ci_type,
                                                        random_state=random_state,
                                                        metric_fn=metric_fn)
        ci_low_99, ci_high_99 = bootstrap_confidence_interval(bootstrap_data,
                                                              bootstrap_iterations,
                                                              ci=0.99,  
                                                              ci_type=bootstrap_ci_type,
                                                              random_state=random_state,
                                                              metric_fn=metric_fn)
        precomputed['ci_low'] = ci_low
        precomputed['ci_high'] = ci_high
        precomputed['ci_low_99'] = ci_low_99
        precomputed['ci_high_99'] = ci_high_99
        figure = bootstrap_plot(bootstrap_data,
                                bootstrap_iterations,
                                ci_low,
                                ci_high,
                                random_state,
                                metric_fn=metric_fn,
                                ci_low_99=ci_low_99,
                                ci_high_99=ci_high_99)
    if p_value is None:
        if bootstrap and extra_samplers is None:
            p_value = bootstrap_p_value(bootstrap_data,
                                        value_threshold,
                                        bootstrap_iterations,
                                        alternative,
                                        random_state,
                                        metric_fn=metric_fn,
                                        strategy=strategy)
        elif extra_samplers is None:
            p_value = ttest_1samp(original_metric_y, 
                                  value_threshold, 
                                  alternative=alternative).pvalue
        else:
            p_value = test_many_samples(original_metric_y,
                                        extra_samplers,
                                        same_questions,
                                        data_type,
                                        exec_env,
                                        main_column)
    precomputed['p_value'] = p_value  
    report = report_valtest_stat_significance(p_value=p_value,
                                              extra_samplers_keys=extra_samplers_keys,
                                              data_type=data_type,
                                              semaphore_threshold=semaphore_threshold,
                                              figure=figure,
                                              is_info=is_info)
    return {'report': report, 
            'precomputed': precomputed}
      
            
def test_many_samples(original_metric_y: pd.Series,
                      extra_samplers: tp.Dict[str, tp.Any],
                      same_questions: bool=False,
                      data_type: str='test',
                      exec_env: str='default',
                      main_column: tp.Optional[str]=None) -> float:
    '''
    Функция для подсчета критериев для более чем одной модели
    Parameters:
        original_metric_y: Массив с метрикой валидируемой модели в разрезе объектов
        extra_samplers: Sampler'ы сравниваемых моделей
        same_questions: True, если разные модели отвечали на одинаковые вопросы, False, если на разные
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
        main_column: Название колонки, содержащей ключевую метрику качества; если None, то усредняются все колонки в y
    Return:
        p_value критериев для более чем одной модели
    '''
    if same_questions:
        p_value = test_many_samples_same_questions(original_metric_y,
                                                   extra_samplers,
                                                   data_type,
                                                   exec_env,
                                                   main_column)
    else:
        comparable_answers = get_real_metric(list(extra_samplers.values())[0],
                                             data_type,
                                             exec_env,
                                             main_column)
        p_value = ttest_ind(original_metric_y,
                            comparable_answers,
                            equal_var=False).pvalue
    return p_value
        
        
def test_many_samples_same_questions(original_metric_y: pd.Series,
                                     extra_samplers: tp.Dict[str, tp.Any],
                                     data_type: str='test',
                                     exec_env: str='default',
                                     main_column: tp.Optional[str]=None) -> float:
    '''
    Функция для подсчета критериев для более чем одной модели для случая с ответами на одни вопросы
    Parameters:
        original_metric_y: Массив с метрикой валидируемой модели в разрезе объектов
        extra_samplers: Sampler'ы сравниваемых моделей
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
        main_column: Название колонки, содержащей ключевую метрику качества; если None, то усредняются все колонки в y
    Return:
        p_value критериев для более чем одной модели для случая с ответами на одни вопросы
    '''
    if len(extra_samplers) == 1:
        p_value = calculate_wilcoxon(original_metric_y,
                                     extra_samplers,
                                     data_type,
                                     exec_env,
                                     main_column)
    else:
        p_value = calculate_friedman(original_metric_y,
                                     extra_samplers,
                                     data_type,
                                     exec_env,
                                     main_column)
    return p_value


def calculate_wilcoxon(original_metric_y: pd.Series,
                       extra_samplers: tp.Dict[str, tp.Any],
                       data_type: str='test',
                       exec_env: str='default',
                       main_column: tp.Optional[str]=None) -> float:
    '''
    Функция для подсчета критерия Уилкоксона
    Parameters:
        original_metric_y: Массив с метрикой валидируемой модели в разрезе объектов
        extra_samplers: Sampler сравниваемой модели
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
        main_column: Название колонки, содержащей ключевую метрику качества; если None, то усредняются все колонки в y
    Return:
        p_value критерия Уилкоксона
    '''
    comparable_answers = get_real_metric(list(extra_samplers.values())[0],
                                         data_type,
                                         exec_env,
                                         main_column)
    diff_for_test = original_metric_y - comparable_answers  
    p_value = wilcoxon(diff_for_test).pvalue
    return p_value


def calculate_friedman(original_metric_y: pd.Series, 
                       extra_samplers: tp.Dict[str, tp.Any],
                       data_type: str='test',
                       exec_env: str='default',
                       main_column: tp.Optional[str]=None) -> float:
    '''
    Функция для подсчета критерия Фридмана
    Parameters:
        original_metric_y: Массив с метрикой валидируемой модели в разрезе объектов
        extra_samplers: Sampler'ы сравниваемых моделей
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
        main_column: Название колонки, содержащей ключевую метрику качества; если None, то усредняются все колонки в y
    Return:
        p_value критерия Фридмана
    '''
    is_corrects = [original_metric_y]
    for sampler in extra_samplers.values():
        comparable_answers = get_real_metric(sampler,
                                             data_type,
                                             exec_env,
                                             main_column)
        is_corrects.append(comparable_answers)
    p_value = friedmanchisquare(*is_corrects).pvalue
    return p_value


def get_real_metric(sampler: tp.Union[Sampler, tp.Dict],
                    data_type: str='test',
                    exec_env: str='default',
                    main_column: tp.Optional[str]=None) -> pd.Series:
    '''
    Parameters:
        sampler: Sampler с данными, в которых хранится колонка с ключевой метрикой качества
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
        main_column: Название колонки, содержащей ключевую метрику качества; если None, то усредняются все колонки в y
    Return: 
        Массив с метрикой валидируемой модели в разрезе объектов
    '''
    y_data = get_data(sampler,
                      data_type,
                      'y',
                      exec_env)
    if main_column:
        if main_column not in y_data.columns:
            raise ValueError(
                f"Колонка '{main_column}' не найдена в y_data. "
                f"Доступные колонки: {list(y_data.columns)}"
            )
        original_metric_y = y_data[main_column].dropna()
    else:
        y_data = y_data.dropna().reset_index(drop=True)
        original_metric_y = y_data.mean(axis=1)
        
    return original_metric_y


def calc_metric_values(y_df: pd.DataFrame,
                       scorer: Scorer,
                       metric_agg: str,
                       data_type: str='test') -> float:
    '''
    Рассчитывает метрику через scorer для переданного DataFrame.
    Parameters:
        y_df: DataFrame с данными для расчета метрики 
        scorer: Объект класса Scorer
        metric_agg: Название агрегации метрики (ключ из словаря scorer.metrics)
        data_type: Тип данных, на которых производится расчет метрики
    Return: 
        Скалярное значение рассчитанной метрики 
    '''
    class _MockSampler: pass
    mock_sampler = _MockSampler()
    setattr(mock_sampler,
            data_type,
            {'y': y_df})

    res = scorer.calc(sampler=mock_sampler, 
                      data_type=data_type, 
                      metric_name=metric_agg)
    return float(res[metric_agg])


def bootstrap_p_value(sample: tp.Union[pd.DataFrame, pd.Series, np.ndarray],
                      value_threshold: float,
                      iterations: int=1000,
                      alternative: str='greater',
                      random_state: tp.Optional[int]=None,
                      metric_fn: tp.Callable = None,
                      strategy: str='empirical') -> float:
    '''
    Parameters:
        sample: Массив наблюдений
        value_threshold: Проверяемый порог 
        iterations: Количество bootstrap итераций
        alternative: Тип альтернативной гипотезы ('greater', 'less', 'two-sided')
        random_state: seed генератора случайных чисел
        metric_fn: Функция расчета метрики
        strategy: Стратегия бутстрапа ('rigorous' для теста с центровкой под H0).
    Return: 
        p-value одновыборочного критерия Стьюдента для бутстрапированной выборки
    '''
    rng = np.random.default_rng(random_state)
    n = len(sample)
    observed = metric_fn(sample)
    if strategy == 'rigorous':
        data = np.asarray(sample)
        centered_data = data - observed + value_threshold
        boot_metrics = np.empty(iterations)
        for i in range(iterations):
            boot_metrics[i] = metric_fn(rng.choice(centered_data, size=n, replace=True))
            
        if alternative == 'greater':
            count = np.sum(boot_metrics >= observed)
        elif alternative == 'less':
            count = np.sum(boot_metrics <= observed)
        else:
            count = np.sum(np.abs(boot_metrics - value_threshold) >= np.abs(observed - value_threshold))
    else:
        idx = np.arange(n)
        boot_metrics = np.empty(iterations)
        for i in range(iterations):
            if isinstance(sample, (pd.DataFrame, pd.Series)):
                boot_sample = sample.iloc[rng.choice(idx, size=n, replace=True)]
            else:
                boot_sample = sample[rng.choice(idx, size=n, replace=True)]
            boot_metrics[i] = metric_fn(boot_sample)
    
        if alternative == 'greater':
            count = np.sum(boot_metrics <= value_threshold)
        elif alternative == 'less':
            count = np.sum(boot_metrics >= value_threshold)
        else:
            count = np.sum(np.abs(boot_metrics - value_threshold) >= np.abs(observed - value_threshold))
            
    return float((count + 1) / (iterations + 1))


def bootstrap_confidence_interval(sample: tp.Union[pd.DataFrame, pd.Series, np.ndarray],
                                  iterations: int=1000,
                                  ci: float=0.95,
                                  ci_type: str='percentile',
                                  random_state: tp.Optional[int]=None,
                                  metric_fn: tp.Callable = None) -> tp.Tuple[float, float]:
    '''
    Parameters:
        sample: Массив наблюдений
        iterations: Количество bootstrap итераций
        ci: Уровень доверительного интервала
        ci_type: Тип доверительного интервала ('percentile', 'basic', 'normal')
        random_state: seed генератора случайных чисел
        metric_fn: Функция расчета метрики
    Return: 
        Кортеж с границами доверительного интервала
    '''
    rng = np.random.default_rng(random_state)
    boot_metrics = []
    n = len(sample)
    idx = np.arange(n)
    
    if isinstance(sample, (pd.DataFrame, pd.Series)):
        for _ in range(iterations):
            boot_metrics.append(metric_fn(sample.iloc[rng.choice(idx, size=n, replace=True)]))
    else:
        for _ in range(iterations):
            boot_metrics.append(metric_fn(sample[rng.choice(idx, size=n, replace=True)]))
            
    boot_metrics = np.array(boot_metrics)
    alpha = (1 - ci) / 2
    theta_hat = metric_fn(sample)
    
    if ci_type == 'percentile':
        lower = np.quantile(boot_metrics, alpha)
        upper = np.quantile(boot_metrics, 1 - alpha)
    elif ci_type == 'basic':
        lower = 2 * theta_hat - np.quantile(boot_metrics, 1 - alpha)
        upper = 2 * theta_hat - np.quantile(boot_metrics, alpha)
    elif ci_type == 'normal':
        std = np.std(boot_metrics)
        z = norm.ppf(1 - alpha)
        lower = theta_hat - z * std
        upper = theta_hat + z * std
    return (float(lower), float(upper))


def bootstrap_plot(sample: tp.Union[pd.DataFrame, pd.Series, np.ndarray],
                   iterations: int=1000,
                   ci_low: float=None,
                   ci_high: float=None,
                   random_state: tp.Optional[int]=None,
                   metric_fn: tp.Callable = None,
                   ci_low_99: float=None,   
                   ci_high_99: float=None) -> plt.Figure:
    '''
    Parameters:
        sample: Массив наблюдений
        iterations: Количество bootstrap итераций
        ci_low: Нижняя граница доверительного интервала
        ci_high: Верхняя граница доверительного интервала
        random_state: Seed генератора случайных чисел для воспроизводимости
        metric_fn: Функция расчета метрики
        ci_low_99: Нижняя граница 99% доверительного интервала
        ci_high_99: Верхняя граница 99% доверительного интервала
    Return: 
        График с визуализацией доверительных интервалов
    '''   
    rng = np.random.default_rng(random_state)
    n = len(sample)
    idx = np.arange(n)
    boot_metrics = []
    if isinstance(sample, (pd.DataFrame, pd.Series)):
        for _ in range(iterations):
            boot_metrics.append(metric_fn(sample.iloc[rng.choice(idx, size=n, replace=True)]))
    else:
        for _ in range(iterations):
            boot_metrics.append(metric_fn(sample[rng.choice(idx, size=n, replace=True)]))         
    fig, ax = plt.subplots()
    counts, bins, _ = ax.hist(boot_metrics, bins=30, color='blue', alpha=0.7)
    mu = np.mean(boot_metrics)
    sigma = max(np.std(boot_metrics), 1e-8)
    x = np.linspace(bins[0], bins[-1], 200)
    y = norm.pdf(x, mu, sigma)
    y_scaled = y * len(boot_metrics) * (bins[1] - bins[0])
    ax.plot(x,
            y_scaled,
            color='gray',
            label='normal distribution')
    if ci_low is not None and ci_high is not None:
        ax.axvline(ci_low,
                   color='red',
                   linestyle='--',
                   label='95% CI')
        ax.axvline(ci_high,
                   color='red',
                   linestyle='--')
    if ci_low_99 is not None and ci_high_99 is not None:
        ax.axvline(ci_low_99,
                   color='orange',
                   linestyle=':',
                   label='99% CI')
        ax.axvline(ci_high_99,
                   color='orange',
                   linestyle=':')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Count bootstrap')
    ax.set_title('Bootstrap method')
    ax.legend()
    fig.tight_layout()
    return fig


def get_data(sampler_obj: tp.Union[Sampler, tp.Dict],
             data_type: str, 
             key: str, 
             exec_env: str='default') -> tp.Any:
    '''
    Parameters:
        sampler_obj: Объект сэмплера, содержащий данные для извлечения
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        key: Ключ или имя атрибута для извлечения конкретных данных (например, 'X', 'y')
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
    Return: 
        Содержимое sampler_obj из data_type по ключу key
    '''
    res = None
    if exec_env == 'default':
        res = getattr(sampler_obj, data_type)[key]
    elif exec_env == 'sberds':
        res = sampler_obj[data_type][key]
    return res
