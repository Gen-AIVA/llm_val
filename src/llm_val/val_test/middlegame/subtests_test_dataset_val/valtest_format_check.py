import numpy as np
import pandas as pd
import typing as tp
from copy import deepcopy

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import worst_semaphore
from llm_val.utils.dataset_type_match import _compile_schema, _resolve_spec, X_TYPES, X_DIALOGUE_TYPES


def report_valtest_format_check(violation: tp.Dict[str, tp.Dict[str, tp.List[int]]],
                                semaphore_none: bool=None,
                                is_info: bool=False):
    '''
    Создание report'а
    Parameters:
        violation: результаты проверок на наличие пропусков в датасете ['NA'] несоответствий типа колонок ['type'] и их полного отсутствия ['col']
        semaphore_none: Светофорно ли проверяются пропуски в датасете
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    if is_info:
        color = 'gray'
    else:
        color_NA = 'red' if len(violation['NA']) > 0 and semaphore_none else 'green'
        color_type = 'red' if len(violation['type']) > 0 else 'green'
        color_column = 'red' if len(violation['col']) > 0 else 'green'
        color = worst_semaphore([color_NA, color_type, color_column])

    test_result_map = {'green': 'Пройдена', 'yellow': 'Не пройдена', 'red': 'Не пройдена', 'gray': 'Пройдена'}
    test_result = test_result_map.get(color, 'Не пройдена')
    violations_tmp = violation['col']
    comment = '' if len(violation['col']) < 1 else f'Наименования отсутствующих необходимых колонок: {violations_tmp}'
    if len(violation['type']) > 0:
        for col in violation['type'].keys():
            violation_local = violation['type'][col]
            comment += f'\n\nНомера строк в колонке {col} в которых не соблюдается формат: {violation_local}'
    if len(violation['NA']) > 0:
        for col in violation['NA'].keys():
            violation_local = violation['NA'][col]
            comment += f'\n\nНомера строк в колонке {col} в которых пропущены значения: {violation_local}'

    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = None
    result_df = pd.DataFrame(
        {
            'Проверка': 'Отсутствие пропусков в датасете, несоответствий типа колонок и их полного отсутствия [основная проверка]',
            'Наличие нарушений формата': [violation],
            'Результат': [test_result],
            'Комментарий': comment
        }
    )
    res_dict['result_dataframes'] = [result_df]
    return res_dict


def valtest_format_check(sampler: Sampler,
                         data_type: str='test',
                         optional_columns: tp.List[str]=[],
                         semaphore_none: bool=True,
                         is_info: bool=False,
                         exec_env: str='default',
                         violation: tp.Dict[str, tp.List[tp.Any]]=None) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на наличие пропусков в датасете, несоответствий типа колонок и их полного отсутствия
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        optional_columns: Колонки, для которых проверка не требуется
        semaphore_none: Светофорно ли проверяются пропуски в датасете
        is_info: True, если тест информативный
        exec_env: Переменная исполняемого окружения ('default' или 'sberds')
        violation: опицональный результат вычисления индексов значений с типом None (по ключу "NA") и значений c несоответсвущим типом (по ключу "type") в каждой из колонок
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if exec_env == 'default':
        data = getattr(sampler, data_type)
    elif exec_env == 'sberds':
        data = sampler[data_type]
    
    template_dict = X_DIALOGUE_TYPES if 'dialogue' in data['X'].columns else X_TYPES
    if violation is None or 'NA' not in violation or 'type' not in violation:
        violation = validate_df(df=data['X'],
                                type_dict=template_dict,
                                optional_columns=optional_columns,
                                violation=violation)
    if violation is None or 'col' not in violation:
        required_cols = [x for x in template_dict.keys() if not ('[' in x and ']' in x)]
        violation['col'] = list((set(required_cols) - set(optional_columns)) - set(data['X'].columns))
    report = report_valtest_format_check(violation,
                                         semaphore_none,
                                         is_info)
    precomputed = {'violation': violation}

    return {'report': report,
            'precomputed': precomputed}


def validate_df(df: pd.DataFrame,
                type_dict: tp.Dict[str, tp.Any],
                optional_columns: tp.List[str]=[],
                violation: tp.Dict[str, tp.List[tp.Any]]=None) -> pd.DataFrame:
    '''
    Проверка соответствия значений колонок DataFrame типам схемы.
    Пропуски и несоответствия типа учитываются отдельно.
    Parameters:
        df: Проверяемый DataFrame
        type_dict: Схема типов
        optional_columns: Колонки, для которых проверка не требуется
        violation: опицональный результат вычисления индексов значений с типом None (по ключу "NA") и значений c несоответсвущим типом (по ключу "type") в каждой из колонок
    Return:
        DataFrame с нарушениями формата или None, если нарушений нет
    '''
    exact, patterns = _compile_schema(type_dict)
    na_violations = dict() if violation is None or 'NA' not in violation else violation['NA']
    type_violations = dict() if violation is None or 'type' not in violation is None else violation['type']

    for column in df.columns:
        if column in optional_columns:
            continue
        spec, allow_none = _resolve_spec(column, exact, patterns)
        if spec is None:
            continue
        series = df[column]
        missing_mask = series.isna().to_numpy(dtype=bool)

        if violation is None or violation['type'] is None:
            type_ok = np.array([miss or spec.check(v) for miss, v in zip(missing_mask, series)],
                            dtype=bool)
            mismatch_pos = np.flatnonzero(~type_ok)
            if mismatch_pos.size:
                type_violations[column] = mismatch_pos.tolist()

        if violation is None or violation['NA'] is None:
            none_pos = np.array([], dtype=int) if allow_none else np.flatnonzero(missing_mask)
            if none_pos.size:
                na_violations[column] = none_pos.tolist()

    return {'NA': na_violations, 'type': type_violations}