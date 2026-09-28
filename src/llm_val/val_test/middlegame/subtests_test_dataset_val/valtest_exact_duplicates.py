import reprlib
import typing as tp
import pandas as pd
from copy import deepcopy
from collections import defaultdict

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold_without_red


def report_valtest_exact_duplicates(duplicates_ratio: float,
                                    duplicate_indices: tp.List[tp.Tuple[int, ...]],
                                    threshold: float=0.1,
                                    is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        duplicates_ratio: Доля примеров-дубликатов
        duplicate_indices: Список индексов дубликатов
        threshold: Граница светофора для доли дубликатов
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''

    color = 'gray' if is_info else semaphore_by_threshold_without_red(
        duplicates_ratio, threshold=threshold, greater_is_better=False
    )
    test_result_map = {'green': 'Пройдена', 'yellow': 'Не пройдена', 'gray': 'Пройдена'}
    test_result = test_result_map.get(color, 'Не пройдена')
    if duplicate_indices:
        comment = f'Номера строк с примерами-дубликатами: {", ".join(str(t) for t in duplicate_indices)}'
    else:
        comment = ''
    result_df = pd.DataFrame({
        'Проверка': 'Отсутствие полных дубликатов среди примеров [вспомогательная проверка]',
        'Доля дубликатов': [duplicates_ratio],
        'Результат': [test_result],
        'Комментарий': comment
    })

    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [result_df]
    return res_dict


def valtest_exact_duplicates(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                             data_type: str='test',
                             threshold: float=0.1,
                             is_info: bool=False,
                             data_duplicates_dict: tp.Optional[tp.Dict[str, tp.Any]]=None,
                             exec_env: tp.Literal['default', 'sberds']='default',
                             **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест отсутствия дубликатов и внутри выборки.
    Если их доля менее threshold (по умолчанию - 10%) - тест пройден.
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на котором исследуется качество данных
        threshold: Граница светофора для доли дубликатов
        is_info: True, если тест информативный
        data_duplicates_dict: Опциональный словарь с предпосчитанными результатами теста.
            Если не None, то тест не запускается, и на основе переданного аргумента выставляется светофор.
        exec_env: Среда выполнения ('default' или 'sberds')
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if data_duplicates_dict is None:
        sampler_dict = tp.cast(tp.Dict[str, tp.Any], sampler)
        data = sampler_dict[data_type]['X'] if exec_env == 'sberds' else getattr(sampler, data_type)['X']
        if 'session_id' in data.columns:
            data = data.groupby('session_id', as_index=False).nth(0)
        if 'dialogue' in data:
            queries_series = data['dialogue'].apply(lambda msgs: [msg[1] for msg in msgs])
            data_duplicates_dict = compute_duplicates_stats(queries_series)
        else:
            data_duplicates_dict = compute_duplicates_stats(data['input_query'])
             
    precomputed = deepcopy(data_duplicates_dict)

    duplicates_ratio = data_duplicates_dict['duplicates_ratio']
    duplicates = data_duplicates_dict['duplicates']

    duplicate_indices = collect_duplicate_indices(duplicates)
    precomputed['duplicate_indices'] = duplicate_indices
    
    report = report_valtest_exact_duplicates(duplicates_ratio,
                                             duplicates,
                                             threshold,
                                             is_info)
    return {'report': report,
            'precomputed': precomputed}


def collect_duplicate_indices(duplicates: tp.List[tp.Tuple[int, ...]]) -> tp.Set[int]:
    '''
    Собирает плоское множество индексов полных дубликатов.
    Parameters:
        duplicates: Список групп индексов полных дубликатов.
    Return:
        Множество индексов полных дубликатов.
    '''
    duplicate_indices = set()
    for duplicate_group in duplicates:
        duplicate_indices.update(duplicate_group)
    return duplicate_indices


def compute_duplicates_stats(queries: pd.Series
                  ) -> tp.Dict[str, tp.Union[float, tp.List[tp.Tuple[int, ...]]]]:
    '''
    Подсчёт доли дубликатов и множества групп дублирующихся индексов.
    Parameters:
        queries: Серия с текстами запросов.
    Return:
        Словарь, содержащий долю дубликатов в датасете, а также индексы дубликатов
    '''
    text_to_indices = defaultdict(list)
    for idx, item in queries.items():
        if isinstance(item, list):
            normalized_key = ' '.join(' '.join(str(t).split()) for t in item)
        else:
            normalized_key = ' '.join(str(item).split())
        text_to_indices[normalized_key].append(idx)

    duplicates = []

    for indices in text_to_indices.values():
        if len(indices) > 1:
            duplicates.append(tuple(indices))

    duplicates.sort(key=lambda x: x[0])
    
    total = len(queries)
    if total == 0:
        ratio = 0.0
    else:
        duplicated_indices_count = len(set().union(*[set(g) for g in duplicates])) if duplicates else 0
        ratio = duplicated_indices_count / total

    return {
        'duplicates_ratio': ratio,
        'duplicates': duplicates
    }
