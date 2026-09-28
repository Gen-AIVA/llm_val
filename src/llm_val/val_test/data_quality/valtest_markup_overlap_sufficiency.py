import typing as tp
import pandas as pd

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold_without_yellow, worst_semaphore


def report_valtest_markup_overlap_sufficiency(overlap_statistics: tp.Dict[str, int],
                                              green_semaphore_threshold: tp.Union[int, tp.Dict[str, int]]=3,
                                              is_justified: bool=False,
                                              is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а в соответствии с концепцией llm_val
    Parameters:
        overlap_statistics: Словарь со статистикой перекрытий по каждой метрике
        green_semaphore_threshold: Граница выставления зеленого светофора.
            Минимальное число асессоров для зеленого светофора.
            Если int — единый порог для всех метрик.
            Если dict — порог для каждой метрики отдельно
        is_justified: Есть ли обоснование меньшего числа разметчиков,
            чем необходимое перекрытие
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    df = pd.DataFrame(list(overlap_statistics.items()),
                       columns=['Название метрики', 'Минимальное перекрытие'])
    semaphore_list = [
        'gray' if is_info
        else semaphore_by_threshold_without_yellow(
            val,
            green_semaphore_threshold[metric] if isinstance(green_semaphore_threshold, dict) else green_semaphore_threshold,
            True)
        for metric, val in overlap_statistics.items()
    ]
    df['Светофор'] = [
        'yellow' if (is_justified and color == 'red') else color
        for color in semaphore_list
    ]
    result_color = 'gray' if is_info else worst_semaphore(df['Светофор'].tolist())
    return {'semaphore': result_color,
            'result_plots': [],
            'result_dataframes': [df]}


def valtest_markup_overlap_sufficiency(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                       data_type: str='test',
                                       green_semaphore_threshold: tp.Optional[tp.Union[int, tp.Dict[str, int]]]=None,
                                       n_categories: tp.Optional[tp.Union[int, tp.Dict[str, int]]]=None,
                                       is_justified: bool=False,
                                       metric_columns: tp.Optional[tp.List[str]]=None,
                                       exec_env: str='default',
                                       is_info: bool=False,
                                       overlap_statistics: tp.Optional[tp.Dict[str, int]]=None,
                                       **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Parameters:
        sampler: Объект класса Sampler или словарь с данными
        data_type: Тип данных, на которых производится расчет метрики.
            Hint: название атрибута в sampler'е
        green_semaphore_threshold: Граница выставления зеленого светофора.
            Минимальное число асессоров для зеленого светофора.
            Имеет наивысший приоритет.
            Если int — единый порог для всех метрик.
            Если dict — порог для каждой метрики отдельно
        n_categories: Количество возможных вариантов оценки (N).
            Порог вычисляется как N+1.
            Если int — единое N для всех метрик.
            Если dict — N для каждой метрики отдельно.
            Если None — N вычисляется автоматически как количество
            уникальных значений метрики в данных
        is_justified: Есть ли обоснование меньшего числа разметчиков,
            чем необходимое перекрытие
        metric_columns: Список названий колонок с метриками.
            Если None — используются все колонки с суффиксом _metric
        exec_env: Среда исполнения ('default' или 'sberds')
        is_info: True, если тест информативный
        overlap_statistics: Предподсчитанная статистика перекрытий
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    x_data = None
    if overlap_statistics is None or green_semaphore_threshold is None:
        x_data = (pd.DataFrame.from_dict(sampler[data_type]['X'])
                  if exec_env == 'sberds'
                  else getattr(sampler, data_type)['X'])
        if metric_columns is None:
            metric_columns = list(
                x_data.columns[x_data.columns.str.endswith('_metric')])
    if overlap_statistics is None:
        overlap_statistics = get_overlap_statistics(x_data, metric_columns)
    if green_semaphore_threshold is None:
        green_semaphore_threshold = resolve_threshold(
            n_categories, x_data, metric_columns)
    precomputed = {'overlap_statistics': overlap_statistics}
    report = report_valtest_markup_overlap_sufficiency(
        overlap_statistics=overlap_statistics,
        green_semaphore_threshold=green_semaphore_threshold,
        is_justified=is_justified,
        is_info=is_info)
    return {'report': report,
            'precomputed': precomputed}


def get_overlap_statistics(x_data: pd.DataFrame,
                           metric_columns: tp.List[str]) -> tp.Dict[str, int]:
    '''
    Подсчет статистики перекрытия по каждой метрике
    Parameters:
        x_data: DataFrame с данными
        metric_columns: Список названий колонок с метриками
    Return:
        Словарь со статистикой минимального перекрытия по каждой метрике
    '''
    overlap_statistics = {}
    for metric_column in metric_columns:
        test_data = x_data.dropna(subset=['assessor_id', metric_column])
        overlap_statistics[metric_column] = test_data.groupby('session_id')['assessor_id'].nunique().min()
    return overlap_statistics


def resolve_threshold(n_categories: tp.Optional[tp.Union[int, tp.Dict[str, int]]],
                      x_data: pd.DataFrame,
                      metric_columns: tp.List[str]) -> tp.Dict[str, int]:
    '''
    Вычисление порога перекрытия (N+1) для каждой метрики
    Parameters:
        n_categories: Количество возможных вариантов оценки (N).
            Если None — вычисляется из данных.
            Если int — единое N для всех метрик.
            Если dict — N для каждой метрики отдельно
        x_data: DataFrame с данными
        metric_columns: Список названий колонок с метриками
    Return:
        Словарь с порогом перекрытия (N+1) для каждой метрики
    '''
    result = ({col: n_categories[col] + 1 for col in metric_columns}
              if isinstance(n_categories, dict)
              else {col: n_categories + 1 for col in metric_columns}
              if isinstance(n_categories, int)
              else {col: x_data[col].dropna().nunique() + 1
                    for col in metric_columns})
    return result
