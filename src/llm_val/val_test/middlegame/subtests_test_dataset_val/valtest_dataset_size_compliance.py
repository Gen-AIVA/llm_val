import typing as tp
import pandas as pd

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold_without_red


def report_valtest_dataset_size_compliance(cleaned_count: int,
                                           agreed_min_count: int,
                                           is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        cleaned_count: Количество примеров после удаления дубликатов и выбросов.
        agreed_min_count: Согласованный минимальный объём (из теста 3.3.5).
        is_info: True, если тест информативный.
    Return:
        report в соответствии с концепцией llm_val
    '''
    color = 'gray' if is_info else semaphore_by_threshold_without_red(
        cleaned_count, threshold=agreed_min_count, greater_is_better=True
    )

    comment = ''
    if color != 'green':
        comment = (
            f'После исключения дубликатов и выбросов остаётся {cleaned_count} примеров, '
            f'что меньше требуемого минимума {agreed_min_count}. '
            f'Необходимо добавить не менее {agreed_min_count - cleaned_count} примеров '
            f'или обосновать сохранение части исключённых записей.'
        )

    result_df = pd.DataFrame(
        {
            'Проверка': 'Соответствие размера датасетов согласованному количеству примеров [основная проверка]',
            'Результат': ['Не пройдена' if color == 'yellow' else 'Пройдена'],
            'Комментарий': [comment],
        }
    )

    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [result_df]
    return res_dict


def valtest_dataset_size_compliance(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                    exec_env: tp.Literal['default', 'sberds']='default',
                                    data_type: str='test',
                                    agreed_min_count: int=250,
                                    outlier_indices: tp.Optional[tp.Set[int]]=None,
                                    duplicate_indices: tp.Optional[tp.Set[int]]=None,
                                    semantic_duplicate_indices: tp.Optional[tp.Set[int]]=None,
                                    is_info: bool=False,
                                    cleaned_count: tp.Optional[int]=None,
                                    indices_to_remove: tp.Optional[tp.Set[int]]=None,
                                    **kwargs: tp.Any) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Проверяет, что размер тестового датасета после удаления дубликатов и выбросов
    не меньше согласованного минимального объёма.
    Parameters:
        sampler: Объект Sampler или словарь (при exec_env='sberds').
        exec_env: 'default' – sampler.test['X'] возвращает DataFrame,
            'sberds' – sampler['test']['X'] возвращает DataFrame.
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        agreed_min_count: Согласованный минимальный объём (из теста 3.3.5).
        outlier_indices: Индексы строк-выбросов, подлежащих удалению.
        duplicate_indices: Индексы строк-дубликатов, подлежащих удалению.
        semantic_duplicate_indices: Индексы строк семантических дубликатов, подлежащих удалению.
        is_info: True, если тест информативный.
        cleaned_count: Предподсчитанное количество примеров после очистки.
        indices_to_remove: Агрегация индексов строк, подлежащих удалению.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if cleaned_count is None:
        data = get_dataset_size_compliance_data(sampler=sampler,
                                                data_type=data_type,
                                                exec_env=exec_env)
        if 'session_id' in data.columns:
            data = data.groupby('session_id', as_index=False).nth(0)
        if indices_to_remove is None:
            indices_to_remove = collect_indices_to_remove(
                duplicate_indices=duplicate_indices,
                outlier_indices=outlier_indices,
                semantic_duplicate_indices=semantic_duplicate_indices
            )
        cleaned_count = len(data) - len(indices_to_remove)

    if indices_to_remove is None:
        indices_to_remove = set()

    report = report_valtest_dataset_size_compliance(cleaned_count=cleaned_count,
                                                    agreed_min_count=agreed_min_count,
                                                    is_info=is_info)
    precomputed = {'cleaned_count': cleaned_count, 'indices_to_remove': indices_to_remove}

    return {'report': report, 'precomputed': precomputed}


def get_dataset_size_compliance_data(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                     data_type: str,
                                     exec_env: tp.Literal['default', 'sberds']) -> pd.DataFrame:
    '''
    Получает данные для проверки размера датасета из sampler.
    Parameters:
        sampler: Объект Sampler или словарь с данными при exec_env='sberds'.
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        exec_env: Среда выполнения теста.
    Return:
        Датафрейм с данными для проверки.
    '''
    if exec_env == 'default':
        data = getattr(sampler, data_type)['X']
    else:
        sampler_dict = tp.cast(tp.Dict[str, tp.Any], sampler)
        data = sampler_dict[data_type]['X']
    return data


def collect_indices_to_remove(duplicate_indices: tp.Optional[tp.Set[int]]=None,
                              outlier_indices: tp.Optional[tp.Set[int]]=None,
                              semantic_duplicate_indices: tp.Optional[tp.Set[int]]=None) -> tp.Set[int]:
    '''
    Объединяет индексы строк, подлежащих удалению.
    Parameters:
        duplicate_indices: Индексы строк-дубликатов, подлежащих удалению.
        outlier_indices: Индексы строк-выбросов, подлежащих удалению.
        semantic_duplicate_indices: Индексы строк семантических дубликатов, подлежащих удалению.
    Return:
        Множество уникальных индексов строк, подлежащих удалению.
    '''
    indices_to_remove = set()
    for indices in (duplicate_indices, outlier_indices, semantic_duplicate_indices):
        if indices:
            indices_to_remove.update(indices)
    return indices_to_remove
