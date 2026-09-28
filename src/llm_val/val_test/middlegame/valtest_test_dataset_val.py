import importlib
import pkgutil
import typing as tp

import pandas as pd

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import worst_semaphore
from llm_val.val_test.middlegame import subtests_test_dataset_val


def report_valtest_test_dataset_val(subtests_data: tp.Dict[str, tp.Dict[str, str]],
                                    is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        subtests_data: Словарь с данными подтестов, где ключ - название, значение - словарь с ключами 'semaphore', 'result', 'comment'
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    subtest_semaphores = [data['semaphore'] for data in subtests_data.values()]
    final_semaphore = 'gray' if is_info else worst_semaphore(subtest_semaphores)
    
    result_dataframe = pd.DataFrame(
        [
            {
                'Проверка': name,
                'Результат': data['result'],
                'Комментарий': data['comment']
            }
            for name, data in subtests_data.items()
        ]
    )
    
    res_dict = {}
    res_dict['semaphore'] = final_semaphore
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [result_dataframe]
    return res_dict


def valtest_test_dataset_val(sampler: Sampler,
                             data_type: str='test',
                             subtests: tp.Optional[tp.List[str]]=None,
                             exec_env: tp.Literal['default', 'sberds']='default',
                             is_info: bool=False,
                             subtests_kwargs: tp.Optional[tp.Dict[str, tp.Dict[str, tp.Any]]]=None,
                             **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Агрегирующий тест набора данных. Запускает все подтесты из пакета subtests_test_dataset_val и выставляет итоговый светофор.
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        subtests: Список названий подтестов для запуска. Если None, запускаются все найденные подтесты
        exec_env: Среда выполнения ('default' или 'sberds')
        is_info: True, если тест информативный
        subtests_kwargs: Дополнительные параметры отдельных подтестов, где ключ - название подтеста
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    available_subtests = order_dependent_subtests(discover_subtests(subtests))
    subtests_kwargs = {} if subtests_kwargs is None else subtests_kwargs
    subtests_data = {}
    precomputed = {}

    for module_name, subtest_func in available_subtests.items():
        subtest_kwargs = get_subtest_kwargs(module_name=module_name,
                                            precomputed=precomputed,
                                            user_kwargs=kwargs,
                                            subtests_kwargs=subtests_kwargs)
        subtest_result = subtest_func(
            sampler=sampler,
            data_type=data_type,
            exec_env=exec_env,
            is_info=is_info,
            **subtest_kwargs
        )
        subtest_report = subtest_result['report']
        
        subtest_name, subtest_res, subtest_comment = extract_subtest_info(
            subtest_report=subtest_report,
            default_name=module_name
        )
        subtests_data[subtest_name] = {
            'semaphore': subtest_report['semaphore'],
            'result': subtest_res,
            'comment': subtest_comment
        }
        precomputed[module_name] = subtest_result['precomputed']

    report = report_valtest_test_dataset_val(
        subtests_data=subtests_data,
        is_info=is_info
    )
    return {'report': report, 'precomputed': precomputed}


def discover_subtests(subtests: tp.Optional[tp.List[str]]=None) -> tp.Dict[str, tp.Any]:
    '''
    Автоматический поиск всех подтестов в пакете subtests_test_dataset_val
    Parameters:
        subtests: Список названий необходимых подтестов. Если None, возвращаются все найденные
    Return:
        Словарь, ключами которого являются названия функций подтестов, а значениями — соответствующие функции
    '''
    subtest_functions = {}
    for _, module_name, _ in pkgutil.iter_modules(subtests_test_dataset_val.__path__):
        if subtests is not None and module_name not in subtests:
            continue
            
        module = importlib.import_module(
            f'llm_val.val_test.middlegame.subtests_test_dataset_val.{module_name}'
        )
        subtest_func = getattr(module, module_name, None)
        if callable(subtest_func):
            subtest_functions[module_name] = subtest_func
    return subtest_functions


def order_dependent_subtests(available_subtests: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    '''
    Упорядочивает подтесты с учетом зависимостей между ними.
    Parameters:
        available_subtests: Словарь найденных подтестов.
    Return:
        Словарь подтестов в порядке запуска.
    '''
    ordered_subtests = {}
    exact_duplicates_name = 'valtest_exact_duplicates'
    semantic_duplicates_name = 'valtest_semantic_duplicates'
    train_test_intersections_name = 'valtest_train_test_intersections'
    dataset_size_compliance_name = 'valtest_dataset_size_compliance'

    for module_name, subtest_func in available_subtests.items():
        if module_name == dataset_size_compliance_name:
            continue
        if (
            (module_name == train_test_intersections_name or module_name == semantic_duplicates_name)
            and exact_duplicates_name in available_subtests
            and exact_duplicates_name not in ordered_subtests
        ):
            ordered_subtests[exact_duplicates_name] = available_subtests[exact_duplicates_name]
        if module_name not in ordered_subtests:
            ordered_subtests[module_name] = subtest_func
    if dataset_size_compliance_name in available_subtests:
        ordered_subtests[dataset_size_compliance_name] = available_subtests[dataset_size_compliance_name]
    return ordered_subtests


def get_subtest_kwargs(module_name: str,
                       precomputed: tp.Dict[str, tp.Dict[str, tp.Any]],
                       user_kwargs: tp.Dict[str, tp.Any],
                       subtests_kwargs: tp.Dict[str, tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Any]:
    '''
    Формирует дополнительные параметры для запуска подтеста.
    Parameters:
        module_name: Название модуля подтеста.
        precomputed: Словарь с предпосчитанными результатами уже выполненных подтестов.
        user_kwargs: Универсальные дополнительные параметры, переданные пользователем в агрегирующий тест.
        subtests_kwargs: Дополнительные параметры отдельных подтестов.
    Return:
        Словарь дополнительных параметров для подтеста.
    '''
    subtest_kwargs = get_subtest_user_kwargs(module_name=module_name,
                                             user_kwargs=user_kwargs,
                                             subtests_kwargs=subtests_kwargs)
    if module_name == 'valtest_dataset_size_compliance':
        subtest_kwargs.update(get_dataset_size_compliance_kwargs(precomputed=precomputed,
                                                                 user_kwargs=subtest_kwargs))
    if module_name == 'valtest_train_test_intersections':
        subtest_kwargs.update(get_train_test_intersections_kwargs(precomputed=precomputed,
                                                                  user_kwargs=subtest_kwargs))
    if module_name == 'valtest_semantic_duplicates':
        subtest_kwargs.update(get_semantic_duplicates_kwargs(precomputed=precomputed,
                                                             user_kwargs=subtest_kwargs))
    return subtest_kwargs


def get_subtest_user_kwargs(module_name: str,
                            user_kwargs: tp.Dict[str, tp.Any],
                            subtests_kwargs: tp.Dict[str, tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Any]:
    '''
    Объединяет универсальные параметры и параметры конкретного подтеста.
    Parameters:
        module_name: Название модуля подтеста.
        user_kwargs: Универсальные дополнительные параметры.
        subtests_kwargs: Дополнительные параметры отдельных подтестов.
    Return:
        Словарь пользовательских параметров для подтеста.
    '''
    subtest_user_kwargs = dict(user_kwargs)
    subtest_user_kwargs.update(subtests_kwargs.get(module_name, {}))
    return subtest_user_kwargs


def get_semantic_duplicates_kwargs(precomputed: tp.Dict[str, tp.Dict[str, tp.Any]],
                                   user_kwargs: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    '''
    Собирает параметры для проверки на семантические дубликаты.
    Parameters:
        precomputed: Словарь с предпосчитанными результатами уже выполненных подтестов.
        user_kwargs: Дополнительные параметры, переданные пользователем в агрегирующий тест.
    Return:
        Словарь параметров для проверки на семантические дубликаты.
    '''
    semantic_kwargs = {}
    if 'exact_duplicates' not in user_kwargs:
        semantic_kwargs['exact_duplicates'] = get_exact_duplicates(precomputed)
    return semantic_kwargs


def get_train_test_intersections_kwargs(precomputed: tp.Dict[str, tp.Dict[str, tp.Any]],
                                        user_kwargs: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    '''
    Собирает параметры для проверки пересечений тестового и валидационного датасетов.
    Parameters:
        precomputed: Словарь с предпосчитанными результатами уже выполненных подтестов.
        user_kwargs: Дополнительные параметры, переданные пользователем в агрегирующий тест.
    Return:
        Словарь параметров для проверки пересечений.
    '''
    intersection_kwargs = {}
    if 'duplicates' not in user_kwargs:
        intersection_kwargs['duplicates'] = get_exact_duplicates(precomputed)
    return intersection_kwargs


def get_dataset_size_compliance_kwargs(precomputed: tp.Dict[str, tp.Dict[str, tp.Any]],
                                       user_kwargs: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Set[int]]:
    '''
    Собирает индексы для передачи в проверку размера датасета.
    Parameters:
        precomputed: Словарь с предпосчитанными результатами уже выполненных подтестов.
        user_kwargs: Дополнительные параметры, переданные пользователем в агрегирующий тест.
    Return:
        Словарь индексов для проверки размера датасета.
    '''
    compliance_kwargs = {}
    if 'duplicate_indices' not in user_kwargs:
        compliance_kwargs['duplicate_indices'] = get_exact_duplicate_indices(precomputed)
    if 'outlier_indices' not in user_kwargs:
        compliance_kwargs['outlier_indices'] = get_outlier_indices(precomputed)
    if 'semantic_duplicate_indices' not in user_kwargs:
        compliance_kwargs['semantic_duplicate_indices'] = get_semantic_duplicate_indices(precomputed)
    return compliance_kwargs


def get_exact_duplicate_indices(precomputed: tp.Dict[str, tp.Dict[str, tp.Any]]) -> tp.Set[int]:
    '''
    Извлекает индексы полных дубликатов из precomputed.
    Parameters:
        precomputed: Словарь с предпосчитанными результатами подтестов.
    Return:
        Множество индексов полных дубликатов.
    '''
    duplicate_precomputed = precomputed.get('valtest_exact_duplicates', {})
    duplicate_indices = duplicate_precomputed.get('duplicate_indices', set())
    return set(duplicate_indices)


def get_exact_duplicates(precomputed: tp.Dict[str, tp.Dict[str, tp.Any]]) -> tp.List[tp.Tuple[int, ...]]:
    '''
    Извлекает группы полных дубликатов из precomputed.
    Parameters:
        precomputed: Словарь с предпосчитанными результатами подтестов.
    Return:
        Список групп индексов полных дубликатов.
    '''
    duplicate_precomputed = precomputed.get('valtest_exact_duplicates', {})
    duplicates = duplicate_precomputed.get('duplicates', [])
    return list(duplicates)


def get_outlier_indices(precomputed: tp.Dict[str, tp.Dict[str, tp.Any]]) -> tp.Set[int]:
    '''
    Извлекает индексы выбросов из precomputed.
    Parameters:
        precomputed: Словарь с предпосчитанными результатами подтестов.
    Return:
        Множество индексов выбросов.
    '''
    outlier_precomputed = precomputed.get('valtest_outliers_by_char_count', {})
    input_stats = outlier_precomputed.get('input_stats', {})
    output_stats = outlier_precomputed.get('output_stats', {})
    outlier_indices = set(input_stats.get('outlier_indices', []))
    outlier_indices.update(output_stats.get('outlier_indices', []))
    return outlier_indices


def get_semantic_duplicate_indices(precomputed: tp.Dict[str, tp.Dict[str, tp.Any]]) -> tp.Set[int]:
    '''
    Извлекает индексы семантических дубликатов из precomputed.
    Parameters:
        precomputed: Словарь с предпосчитанными результатами подтестов.
    Return:
        Множество индексов семантических дубликатов.
    '''
    semantic_precomputed = precomputed.get('valtest_semantic_duplicates', {})
    semantic_duplicate_indices = semantic_precomputed.get('semantic_duplicate_indices', set())
    return set(semantic_duplicate_indices)


def extract_subtest_info(subtest_report: tp.Dict[str, tp.Any],
                         default_name: str) -> tp.Tuple[str, str, str]:
    '''
    Извлечение названия, результата и комментария подтеста из отчета
    Parameters:
        subtest_report: Словарь с результатами отчета подтеста
        default_name: Название модуля по умолчанию
    Return:
        Кортеж из названия, результата и комментария подтеста
    '''
    subtest_name = default_name
    subtest_res = '-'
    subtest_comment = ''
    
    result_dfs = subtest_report.get('result_dataframes', [])
    if result_dfs and not result_dfs[0].empty:
        first_df = result_dfs[0]
        if 'Проверка' in first_df.columns:
            subtest_name = str(first_df['Проверка'].iloc[0])
        if 'Результат' in first_df.columns:
            subtest_res = str(first_df['Результат'].iloc[0])
        if 'Комментарий' in first_df.columns:
            subtest_comment = str(first_df['Комментарий'].iloc[0])
            
    return subtest_name, subtest_res, subtest_comment
