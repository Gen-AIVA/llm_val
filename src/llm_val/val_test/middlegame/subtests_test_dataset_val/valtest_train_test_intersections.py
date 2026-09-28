import typing as tp
from copy import deepcopy

import pandas as pd

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold_without_red


def report_valtest_train_test_intersections(intersection_ratio: float,
                                            intersection_indices: tp.List[int],
                                            threshold: float=0.1,
                                            is_info: bool=False,
                                            is_applicable: bool=True) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        intersection_ratio: Доля пересечений
        intersection_indices: Список индексов-пересечений
        threshold: Граница светофора для доли выбросов
        is_info: True, если тест информативный
        is_applicable: True, если предоставлен валидационный датасет
    Return:
        report в соответствии с концепцией llm_val
    '''
    if not is_applicable:
        color = 'gray'
        test_result = 'Не проводилась'
        comment = 'Валидационный датасет не предоставлен'
    else:
        color = 'gray' if is_info else semaphore_by_threshold_without_red(
            intersection_ratio, threshold=threshold, greater_is_better=False
        )
        if is_info:
            test_result = 'Пройдена'
        else:
            test_result = 'Пройдена' if color == 'green' else 'Не пройдена'
        if intersection_indices:
            comment = (
                f'Номера строк с пересекающимися примерами (из тестовой выборки): '
                f'{", ".join(str(i) for i in intersection_indices)}. '
                f'Рекомендуется удалить дубликаты из валидационного датасета.'
            )
        else:
            comment = ''
    result_df = pd.DataFrame({
        'Проверка': 'Отсутствие пересечений примеров тестового датасета с примерами валидационного датасета [вспомогательная проверка]',
        'Доля пересечений': [intersection_ratio],
        'Результат': [test_result],
        'Комментарий': [comment]
    })
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [result_df]
    return res_dict


def valtest_train_test_intersections(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                     exec_env: tp.Literal['default', 'sberds']='default',
                                     train_data_type: tp.Literal['train', 'val']='train',
                                     test_data_type: tp.Literal['test', 'oot']='test',
                                     threshold: float=0.1,
                                     duplicates: tp.Optional[tp.List[tp.Tuple[int, ...]]]=None,
                                     is_info: bool=False,
                                     precomputed_intersections: tp.Optional[tp.Dict[str, tp.Any]]=None,
                                     **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на отсутствие пересечений между тестовой и валидационной выборками.
    Если валидационный датасет не предоставлен, тест не проводится (серый).
    Parameters:
        sampler: Объект класса Sampler
        exec_env: Среда выполнения ('default' или 'sberds')
        train_data_type: Тип данных с валидационным датасетом
        test_data_type: Тип данных с тестовым датасетом
        threshold: Граница светофора для доли пересечений
        duplicates: Список групп индексов полных дубликатов в тестовом датасете
        is_info: True, если тест информативный
        precomputed_intersections: Опциональный словарь с предпосчитанными результатами теста.
            Если не None, то тест не запускается, и на основе переданного аргумента выставляется светофор.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if precomputed_intersections is not None:
        stats = precomputed_intersections
    elif not has_sampler_data_type(sampler=sampler,
                                   data_type=train_data_type,
                                   exec_env=exec_env):
        stats = build_not_applicable_stats()
    else:
        train_data = get_sampler_data(sampler=sampler,
                                      data_type=train_data_type,
                                      exec_env=exec_env)
        test_data = get_sampler_data(sampler=sampler,
                                     data_type=test_data_type,
                                     exec_env=exec_env)
        if 'session_id' in train_data['X'].columns:
            train_data['X'] = train_data['X'].groupby('session_id', as_index=False).nth(0)
        if 'session_id' in test_data['X'].columns:
            test_data['X'] = test_data['X'].groupby('session_id', as_index=False).nth(0)    
        train_texts = extract_text_series(train_data['X'])
        test_texts = extract_text_series(test_data['X'])
        stats = compute_intersection_stats(train_texts=train_texts,
                                           test_texts=test_texts,
                                           duplicates=duplicates)

    precomputed = {'intersections_stats': deepcopy(stats)}

    report = report_valtest_train_test_intersections(
        intersection_ratio=stats['intersection_ratio'],
        intersection_indices=stats['intersection_indices'],
        threshold=threshold,
        is_info=is_info,
        is_applicable=stats.get('is_applicable', True)
    )
    return {'report': report, 'precomputed': precomputed}


def normalize_text(text: str) -> str:
    '''
    Нормализует текст.
    Parameters:
        text: Исходный текст.
    Return:
        Текст без лишних пробелов.
    '''
    return ' '.join(str(text).split())


def dialogue_to_text(dialogue_list: tp.Iterable[tp.Tuple[tp.Any, ...]]) -> str:
    '''
    Преобразует диалог в одну строку из пользовательских вводов.
    Parameters:
        dialogue_list: Список кортежей диалога.
    Return:
        Строка с нормализованными пользовательскими вводами.
    '''
    return ' '.join(normalize_text(turn[1]) for turn in dialogue_list)


def extract_text_series(data: pd.DataFrame) -> pd.Series:
    '''
    Извлекает тексты запросов из датафрейма.
    Parameters:
        data: Датафрейм с текстами запросов.
    Return:
        Серия нормализованных запросов из датасета
    '''
    if 'dialogue' in data.columns:
        text_series = data['dialogue'].apply(dialogue_to_text)
    else:
        text_series = data['input_query'].apply(normalize_text)
    return tp.cast(pd.Series, text_series)


def has_sampler_data_type(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                          data_type: tp.Literal['train', 'val', 'test', 'oot'],
                          exec_env: tp.Literal['default', 'sberds']) -> bool:
    '''
    Проверяет наличие выборки в sampler.
    Parameters:
        sampler: Объект Sampler или словарь с данными.
        data_type: Тип данных для проверки.
        exec_env: Среда выполнения теста.
    Return:
        True, если выборка есть в sampler.
    '''
    sampler_dict = tp.cast(tp.Dict[str, tp.Any], sampler)
    if exec_env == 'default':
        has_data_type = hasattr(sampler, data_type) and getattr(sampler, data_type) is not None
    else:
        has_data_type = data_type in sampler_dict and sampler_dict[data_type] is not None
    return has_data_type


def get_sampler_data(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                     data_type: tp.Literal['train', 'val', 'test', 'oot'],
                     exec_env: tp.Literal['default', 'sberds']) -> tp.Dict[str, pd.DataFrame]:
    '''
    Получает выборку из sampler.
    Parameters:
        sampler: Объект Sampler или словарь с данными.
        data_type: Тип данных для получения.
        exec_env: Среда выполнения теста.
    Return:
        Словарь с данными выборки.
    '''
    sampler_dict = tp.cast(tp.Dict[str, tp.Dict[str, pd.DataFrame]], sampler)
    data = getattr(sampler, data_type) if exec_env == 'default' else sampler_dict[data_type]
    return data


def build_not_applicable_stats() -> tp.Dict[str, tp.Any]:
    '''
    Формирует статистики для случая отсутствия валидационного датасета.
    Return:
        Словарь со статистиками неприменимого теста.
    '''
    stats = {
        'intersection_ratio': 0.0,
        'intersection_indices': [],
        'checked_test_indices': [],
        'is_applicable': False
    }
    return stats


def get_duplicate_indices_to_skip(duplicates: tp.Optional[tp.List[tp.Tuple[int, ...]]]=None) -> tp.Set[int]:
    '''
    Определяет индексы дубликатов, которые не нужно учитывать в тесте.
    Parameters:
        duplicates: Список групп индексов полных дубликатов.
    Return:
        Множество индексов дубликатов без первых примеров из каждой группы.
    '''
    indices_to_skip = set()
    if duplicates is not None:
        for duplicate_group in duplicates:
            sorted_duplicate_group = sorted(duplicate_group)
            indices_to_skip.update(sorted_duplicate_group[1:])
    return indices_to_skip


def get_checked_test_indices(test_texts: pd.Series,
                             duplicates: tp.Optional[tp.List[tp.Tuple[int, ...]]]=None) -> tp.List[int]:
    '''
    Определяет индексы тестовых примеров, участвующих в сравнении.
    Parameters:
        test_texts: Серия текстов тестового датасета.
        duplicates: Список групп индексов полных дубликатов.
    Return:
        Список индексов тестовых примеров для сравнения.
    '''
    indices_to_skip = get_duplicate_indices_to_skip(duplicates)
    checked_test_indices = [index for index in sorted(test_texts.index) if index not in indices_to_skip]
    return checked_test_indices


def compute_intersection_stats(train_texts: pd.Series,
                               test_texts: pd.Series,
                               duplicates: tp.Optional[tp.List[tp.Tuple[int, ...]]]=None) -> tp.Dict[str, tp.Any]:
    '''
    Вычисляет пересечение между train и test на основе полного совпадения текстов.
    Parameters:
        train_texts: Серия текстовых запросов валидационного датасета
        test_texts: Серия текстовых запросов тестового датасета
        duplicates: Список групп индексов полных дубликатов
    Return:
        Словарь с долей пересечений и списком индексов из test, которые пересекаются.
    '''
    train_unique_texts = set(train_texts)
    checked_test_indices = get_checked_test_indices(test_texts=test_texts,
                                                    duplicates=duplicates)
    intersection_indices = []
    for test_index in checked_test_indices:
        text = test_texts.loc[test_index]
        if text in train_unique_texts:
            intersection_indices.append(test_index)
    checked_count = len(checked_test_indices)
    ratio = len(intersection_indices) / checked_count if checked_count > 0 else 0.0
    return {
        'intersection_ratio': ratio,
        'intersection_indices': intersection_indices,
        'checked_test_indices': checked_test_indices,
        'is_applicable': True
    }
