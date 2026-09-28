import typing as tp
import reprlib

import numpy as np
import pandas as pd

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold_without_red


def report_valtest_outliers_by_char_count(outlier_ratio: float,
                                          outlier_indices: tp.List[int],
                                          threshold: float=0.1,
                                          is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        outlier_ratio: Доля примеров-выбросов (худшая для диалогов)
        outlier_indices: Список индексов выбросов
        threshold: Граница светофора для доли выбросов
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    color = 'gray' if is_info else semaphore_by_threshold_without_red(
        outlier_ratio, threshold=threshold, greater_is_better=False
    )
    test_result_map = {'green': 'Пройдена', 'yellow': 'Не пройдена', 'gray': 'Пройдена'}
    test_result = test_result_map.get(color, 'Не пройдена')
    comment = '' if not outlier_indices else f'Номера строк с примерами, являющимися выбросами: {reprlib.repr(outlier_indices)}'
    result_df = pd.DataFrame(
        {
            'Проверка': 'Отсутствие выбросов среди примеров по количеству символов [вспомогательная проверка]',
            'Доля выбросов': [outlier_ratio],
            'Результат': [test_result],
            'Комментарий': comment
        }
    )
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [result_df]
    return res_dict


def valtest_outliers_by_char_count(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                   data_type: str='test',
                                   threshold: float=0.1,
                                   is_info: bool=False,
                                   input_stats: tp.Optional[tp.Dict[str, tp.Any]]=None,
                                   output_stats: tp.Optional[tp.Dict[str, tp.Any]]=None,
                                   exec_env: tp.Literal['default', 'sberds']='default',
                                   **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест отсутствия выбросов среди примеров по количеству символов. 
    На основе квартилей Q1, Q3 и межквартильного размаха IQR определяются примеры-выбросы. 
    Если их доля менее threshold (по умолчанию - 10%) - тест пройден.
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        threshold: Граница светофора для доли выбросов
        is_info: True, если тест информативный
        input_stats: Словарь со статистиками выбросов по текстам запросов из precomputed
        output_stats: Словарь со статистиками выбросов по текстам ответов (применимо для формата диалогов) из precomputed
        exec_env: Среда выполнения ('default' или 'sberds')
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if input_stats is None:
        data = sampler[data_type]['X'] if exec_env == 'sberds' else getattr(sampler, data_type)['X']
        if 'session_id' in data.columns:
            data = data.groupby('session_id', as_index=False).nth(0)
        has_dialogue = 'dialogue' in data
        input_char_series = data['dialogue'].apply(
            lambda dialog: sum(len(turn[1]) for turn in dialog)
        ) if has_dialogue else data['input_query'].apply(len)
        input_stats = calculate_char_outlier_stats(input_char_series)
        if has_dialogue:
            output_char_series = data['dialogue'].apply(
                lambda dialog: sum(len(turn[2]) for turn in dialog)
            )
            output_stats = calculate_char_outlier_stats(output_char_series)

    worst_ratio, combined_indices = get_worst_outlier_data(
        input_stats['outlier_ratio'], 
        input_stats['outlier_indices'],
        output_stats['outlier_ratio'] if output_stats else None,
        output_stats['outlier_indices'] if output_stats else None
    )

    report = report_valtest_outliers_by_char_count(outlier_ratio=worst_ratio,
                                                   outlier_indices=combined_indices,
                                                   threshold=threshold,
                                                   is_info=is_info)
    precomputed = {'input_stats': input_stats}
    if output_stats is not None:
        precomputed['output_stats'] = output_stats
    return {'report': report, 
            'precomputed': precomputed}


def get_worst_outlier_data(input_ratio: float,
                           input_indices: tp.List[int],
                           output_ratio: tp.Optional[float],
                           output_indices: tp.Optional[tp.List[int]]) -> tp.Tuple[float, tp.List[int]]:
    '''
    Определение худшей доли выбросов и объединение индексов
    Parameters:
        input_ratio: Доля выбросов среди инпутов
        input_indices: Индексы выбросов среди инпутов
        output_ratio: Доля выбросов среди аутпутов
        output_indices: Индексы выбросов среди аутпутов
    Return:
        Кортеж из худшей доли выбросов и списка уникальных индексов
    '''
    worst_ratio = input_ratio
    combined_indices = input_indices
    if output_ratio is not None:
        worst_ratio = max(input_ratio, output_ratio)
        combined_indices = list(set(input_indices) | set(output_indices))
    return worst_ratio, combined_indices


def calculate_char_outlier_stats(char_series: pd.Series) -> tp.Dict[str, tp.Any]:
    '''
    Расчет статистик выбросов по количеству символов
    Parameters:
        char_series: Серия с количеством символов
    Return:
        Словарь со статистиками выбросов
    '''
    stats = {
        'outlier_ratio': 0.0,
        'char_counts': {},
        'q1': 0.0,
        'q3': 0.0,
        'iqr': 0.0,
        'lower_bound': 0.0,
        'upper_bound': 0.0,
        'outlier_indices': []
    }
    if len(char_series) > 0:
        stats['q1'], stats['q3'] = np.percentile(char_series, 25), np.percentile(char_series, 75)
        stats['iqr'] = stats['q3'] - stats['q1']
        stats['lower_bound'], stats['upper_bound'] = stats['q1'] - 1.5 * stats['iqr'], stats['q3'] + 1.5 * stats['iqr']
        outlier_mask = (char_series < stats['lower_bound']) | (char_series > stats['upper_bound'])
        stats['outlier_indices'] = char_series[outlier_mask].index.tolist()
        stats['outlier_ratio'] = len(stats['outlier_indices']) / len(char_series)
        stats['char_counts'] = char_series.to_dict()
    return stats
