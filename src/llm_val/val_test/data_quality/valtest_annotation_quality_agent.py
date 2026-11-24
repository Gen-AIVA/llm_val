from itertools import combinations

import numpy as np
import typing as tp
import pandas as pd
import krippendorff
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold, worst_semaphore


def report_valtest_annotation_quality_agent(criteria_stats: tp.List[tp.Dict[str, tp.Any]],
                                            semaphore_threshold_spear: tp.Tuple[float, float]=(0.3, 0.6),
                                            semaphore_threshold_krip: tp.Tuple[float, float]=(0.1, 0.4),
                                            semaphore_threshold_cohen: tp.Tuple[float, float]=(0.4, 0.8),
                                            semaphore_threshold_fleiss: tp.Tuple[float, float]=(0.4, 0.8),
                                            is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а в соответствии с концепцией llm_val
    Parameters:
        criteria_stats: Список словарей по критериям с ключами 'criterion',
            'check' и 'value'
        semaphore_threshold_spear: Границы светофора для корреляции Спирмена
        semaphore_threshold_krip: Границы светофора для альфы Криппендорфа
        semaphore_threshold_cohen: Границы светофора для Каппы Коэна
        semaphore_threshold_fleiss: Границы светофора для Каппы Флейса
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    thresholds = {'Попарный коэф. Спирмена': semaphore_threshold_spear,
                  'Альфа Криппендорфа': semaphore_threshold_krip,
                  'Каппа Коэна': semaphore_threshold_cohen,
                  'Каппа Флейса': semaphore_threshold_fleiss}
    is_multi_criteria = len(criteria_stats) > 1
    checks, values, colors = [], [], []
    for stat in criteria_stats:
        color = ('gray' if (is_info or pd.isna(stat['value']))
                 else semaphore_by_threshold(stat['value'], thresholds[stat['check']]))
        check_name = (f'{stat["check"]} ({stat["criterion"]})'
                      if is_multi_criteria else stat['check'])
        checks.append(check_name)
        values.append(None if pd.isna(stat['value']) else stat['value'])
        colors.append(color)
    df = pd.DataFrame({'Наименование проверки': checks,
                       'Согласие': values,
                       'Светофор': colors})
    result_color = ('gray' if is_info
                    else worst_semaphore(
                        [color for color in colors if color != 'gray']))
    return {'semaphore': result_color,
            'result_plots': [],
            'result_dataframes': [df]}


def valtest_annotation_quality_agent(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                     data_type: str='test',
                                     min_periods: int=1,
                                     metric_columns: tp.Optional[tp.List[str]]=None,
                                     semaphore_threshold_spear: tp.Tuple[float, float]=(0.3, 0.6),
                                     semaphore_threshold_krip: tp.Tuple[float, float]=(0.1, 0.4),
                                     semaphore_threshold_cohen: tp.Tuple[float, float]=(0.4, 0.8),
                                     semaphore_threshold_fleiss: tp.Tuple[float, float]=(0.4, 0.8),
                                     exec_env: str='default',
                                     is_info: bool=False,
                                     criteria_stats: tp.Optional[tp.List[tp.Dict[str, tp.Any]]]=None,
                                     **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Оценка качества разметки на основании согласованности асессоров
    Parameters:
        sampler: Объект класса Sampler или словарь с данными
        data_type: Тип данных, на которых производится расчет метрики.
            Hint: название атрибута в sampler'е
        min_periods: Минимальное количество не np.nan элементов
            для подсчета корреляции Спирмена между асессорами
        metric_columns: Список названий колонок с метриками.
            Если None — используются все колонки с суффиксом _metric
        semaphore_threshold_spear: Границы светофора для корреляции Спирмена
        semaphore_threshold_krip: Границы светофора для альфы Криппендорфа
        semaphore_threshold_cohen: Границы светофора для Каппы Коэна
        semaphore_threshold_fleiss: Границы светофора для Каппы Флейса
        exec_env: Среда исполнения ('default' или 'sberds')
        is_info: True, если тест информативный
        criteria_stats: Предподсчитанный список статистик по критериям
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if criteria_stats is None:
        criteria_stats = compute_agreement_stats(
            sampler=sampler,
            data_type=data_type,
            min_periods=min_periods,
            metric_columns=metric_columns,
            exec_env=exec_env)
    report = report_valtest_annotation_quality_agent(
        criteria_stats=criteria_stats,
        semaphore_threshold_spear=semaphore_threshold_spear,
        semaphore_threshold_krip=semaphore_threshold_krip,
        semaphore_threshold_cohen=semaphore_threshold_cohen,
        semaphore_threshold_fleiss=semaphore_threshold_fleiss,
        is_info=is_info)
    return {'report': report,
            'precomputed': {'criteria_stats': criteria_stats}}


def compute_agreement_stats(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                            data_type: str,
                            min_periods: int,
                            metric_columns: tp.Optional[tp.List[str]],
                            exec_env: str) -> tp.List[tp.Dict[str, tp.Any]]:
    '''
    Вычисление статистик согласованности асессоров по каждому критерию
    Parameters:
        sampler: Объект класса Sampler или словарь с данными
        data_type: Тип данных, на которых производится расчет метрики.
            Hint: название атрибута в sampler'е
        min_periods: Минимальное количество наблюдений для корреляции Спирмена
        metric_columns: Список колонок с метриками
        exec_env: Среда исполнения
    Return:
        Список словарей по критериям с ключами 'criterion', 'check' и 'value'
    '''
    x_data = (pd.DataFrame.from_dict(sampler[data_type]['X'])
              if exec_env == 'sberds'
              else getattr(sampler, data_type)['X'])
    if metric_columns is None:
        metric_columns = list(
            x_data.columns[x_data.columns.str.endswith('_metric')])
    agg_data = x_data[['assessor_id', 'session_id'] + metric_columns]
    n_assessors = agg_data['assessor_id'].nunique()
    criteria_stats = []
    for metric_column in metric_columns:
        check = select_agreement_check(agg_data, metric_column, n_assessors)
        value = compute_criterion_value(
            agg_data, metric_column, check, min_periods)
        criteria_stats.append({'criterion': metric_column,
                               'check': check,
                               'value': value})
    return criteria_stats


def compute_criterion_value(df: pd.DataFrame,
                            metric_column: str,
                            check: str,
                            min_periods: int) -> float:
    '''
    Расчет значения согласия по одному критерию выбранной проверкой
    Parameters:
        df: DataFrame с колонками assessor_id, session_id и метриками
        metric_column: Название колонки с метрикой критерия
        check: Наименование выбранной проверки
        min_periods: Минимальное количество наблюдений для корреляции Спирмена
    Return:
        Значение согласия по критерию
    '''
    check_to_func = {
        'Попарный коэф. Спирмена': lambda: spearman_pair_correlation(
            df, metric_column, min_periods),
        'Альфа Криппендорфа': lambda: krippendorff_alpha(df, metric_column),
        'Каппа Коэна': lambda: cohen_kappa_pair(df, metric_column),
        'Каппа Флейса': lambda: fleiss_kappa_calc(df, metric_column)}
    return check_to_func[check]()


def classify_annotation_type(markup_column: pd.Series) -> str:
    '''
    Классификация типа разметки
    Parameters:
        markup_column: Series с данными разметки по одной из колонок
    Return:
        Тип разметки: 'nominal', 'ordinal' или 'binary'
    '''
    clean_column = markup_column.dropna()
    num_unique = clean_column.nunique()
    is_numeric = not pd.to_numeric(clean_column, errors='coerce').isna().any()
    result = ('binary' if num_unique == 2
              else 'ordinal' if is_numeric
              else 'nominal')
    return result


def select_agreement_check(df: pd.DataFrame,
                           metric_column: str,
                           n_assessors: int) -> str:
    '''
    Выбор проверки согласия по типу разметки и перекрытию для критерия
    Parameters:
        df: DataFrame с колонками assessor_id, session_id и метриками
        metric_column: Название колонки с метрикой критерия
        n_assessors: Количество уникальных асессоров
    Return:
        Наименование проверки для критерия
    '''
    column_type = classify_annotation_type(df[metric_column])
    clean_data = df.dropna(subset=['assessor_id', metric_column])
    overlaps = clean_data.groupby('session_id')['assessor_id'].nunique()
    is_full_overlap = (len(overlaps) > 0
                       and bool((overlaps == n_assessors).all()))
    type_to_check = {'ordinal': 'Попарный коэф. Спирмена',
                     'nominal': 'Альфа Криппендорфа'}
    return type_to_check.get(
        column_type,
        'Каппа Флейса' if is_full_overlap else 'Каппа Коэна')


def encode_nominal_labels(df: pd.DataFrame,
                          metric_column: str) -> pd.DataFrame:
    '''
    Кодирование произвольных меток критерия в числовые коды
    Parameters:
        df: DataFrame с колонками assessor_id, session_id и метриками
        metric_column: Название колонки с метрикой критерия
    Return:
        DataFrame с числовыми кодами меток вместо исходных значений
    '''
    encoded = df[['assessor_id', 'session_id', metric_column]].copy()
    labels = encoded[metric_column].dropna().astype(str)
    mapping = {label: code for code, label
               in enumerate(sorted(labels.unique()))}
    encoded[metric_column] = encoded[metric_column].map(
        lambda label: mapping[str(label)] if pd.notna(label) else np.nan)
    return encoded


def krippendorff_alpha(df: pd.DataFrame,
                       metric_column: str) -> float:
    '''
    Вычисление альфы Криппендорфа для неупорядоченного критерия
    Parameters:
        df: DataFrame с колонками assessor_id, session_id и метриками
        metric_column: Название колонки с неупорядоченной метрикой критерия
    Return:
        Коэффициент альфа Криппендорфа
    '''
    encoded = encode_nominal_labels(df, metric_column)
    test_data = pd.pivot_table(encoded,
                               values=metric_column,
                               index='assessor_id',
                               columns='session_id',
                               aggfunc='first')
    matrix = test_data.to_numpy(dtype=float)
    unique_vals = np.unique(matrix[~np.isnan(matrix)])
    if len(unique_vals) <= 1:
        return np.nan
    return krippendorff.alpha(reliability_data=matrix,
                              level_of_measurement='nominal')


def spearman_pair_correlation(df: pd.DataFrame,
                              metric_column: str,
                              min_periods: int=1) -> float:
    '''
    Вычисление попарного коэффициента корреляции Спирмена
    Parameters:
        df: DataFrame с колонками assessor_id, session_id и метриками
        metric_column: Название колонки с упорядоченной метрикой критерия
        min_periods: Минимальное количество не np.nan элементов
            для подсчета корреляции Спирмена между асессорами
    Return:
        Коэффициент Спирмена
    '''
    test_data = pd.pivot_table(df,
                               values=metric_column,
                               index='assessor_id',
                               columns='session_id',
                               aggfunc='first').T
    corr_matrix = test_data.corr(
        method='spearman', min_periods=min_periods).to_numpy()
    correlations_sum = (np.nansum(np.triu(corr_matrix))
                        - np.nansum(np.diagonal(corr_matrix)))
    cnt = (np.triu(~np.isnan(corr_matrix)).sum()
           - np.diagonal(~np.isnan(corr_matrix)).sum())
    return correlations_sum / cnt if cnt != 0 else np.nan


def cohen_kappa_pair(df: pd.DataFrame,
                     metric_column: str) -> float:
    '''
    Вычисление Каппы Коэна для бинарного критерия
    Parameters:
        df: DataFrame с колонками assessor_id, session_id и метриками
        metric_column: Название колонки с бинарной метрикой критерия
    Return:
        Значение Каппы Коэна
    '''
    encoded = encode_nominal_labels(df, metric_column)
    all_labels = sorted(encoded[metric_column].dropna().unique().tolist())
    test_data = pd.pivot_table(encoded,
                               values=metric_column,
                               index='assessor_id',
                               columns='session_id',
                               aggfunc='first').T
    kappas = []
    for first_ann, second_ann in combinations(test_data.columns, 2):
        pair_df = test_data[[first_ann, second_ann]].dropna()
        if len(pair_df) == 0:
            continue
        kappa = cohen_kappa_score(pair_df[first_ann], pair_df[second_ann],
                                  labels=all_labels)
        if not pd.isna(kappa):
            kappas.append(kappa)
    return float(np.mean(kappas)) if kappas else np.nan


def fleiss_kappa_calc(df: pd.DataFrame,
                      metric_column: str) -> float:
    '''
    Вычисление Каппы Флейса для бинарного критерия
    Parameters:
        df: DataFrame с колонками assessor_id, session_id и метриками
        metric_column: Название колонки с бинарной метрикой критерия
    Return:
        Значение Каппы Флейса
    '''
    encoded = encode_nominal_labels(df, metric_column)
    test_data = pd.pivot_table(encoded,
                               values=metric_column,
                               index='assessor_id',
                               columns='session_id',
                               aggfunc='first').T.dropna()
    if test_data.empty:
        return np.nan
    return fleiss_kappa(aggregate_raters(test_data)[0])
