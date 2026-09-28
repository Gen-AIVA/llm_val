import os
import datetime
import numpy as np
import pandas as pd
import typing as tp

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold


def report_valtest_node_usage_analysis(node_stats: pd.DataFrame,
                                       frequency_threshold: float=0.05,
                                       red_semaphore_threshold: float=0.5,
                                       yellow_semaphore_threshold: float=0.0,
                                       is_info: bool=False) -> tp.Dict[str, tp.Any]:

    '''
    Создание report'а.
    Parameters:
        node_stats: статистика использования нод
        frequency_threshold: минимальный процент трейсов для каждой ноды
        red_semaphore_threshold: порог для красного светофора
        yellow_semaphore_threshold: порог для желтого светофора
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    if len(node_stats) == 0:
        result_metric = 0.0
        bad_nodes = []
    else:
        result_metric = len(node_stats[node_stats['frequency'] < frequency_threshold]) / len(node_stats)
        bad_nodes = list(node_stats[node_stats['frequency'] < frequency_threshold]['span_name'])

    final_semaphore = semaphore_by_threshold(result_metric,
                                             threshold=[yellow_semaphore_threshold, red_semaphore_threshold],
                                             greater_is_better=False,
                                             left_border_is_bigger=False,
                                             right_border_is_bigger=False)
    dct_for_text = {'red': f'''Около {100 * round(result_metric, 3)}% нод или инструментов представлены недостаточно (менее, чем в {100 * round(frequency_threshold, 3)}% трейсов). Ноды, которые представлены в трейсах недостаточно: {bad_nodes}''',
                    'yellow': f'''Около {100 * round(result_metric, 3)}% нод или инструментов представлены недостаточно (менее, чем в {100 * round(frequency_threshold, 3)}% трейсов). Ноды, которые представлены в трейсах недостаточно: {bad_nodes}''',
                    'green': f'Все ноды и инструменты представлены достаточно (более чем в {100 * round(frequency_threshold, 3)}% трейсов).'}
    result_table = pd.DataFrame()
    result_table['Результат теста'] = [dct_for_text[final_semaphore]]
    result_table['Светофор'] = [final_semaphore] 
    result_color = 'gray' if is_info else final_semaphore
    res_dict = {
        'semaphore': result_color,
        'result_plots': [],
        'result_dataframes': [result_table]
    }
    return res_dict


def valtest_nodes_usage_analysis(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                 data_type: str='test',
                                 exec_env: str='default',
                                 frequency_threshold: float=0.05,
                                 red_semaphore_threshold: float=0.5,
                                 yellow_semaphore_threshold: float=0.0,
                                 nodes_usage_stats: tp.Optional[pd.DataFrame]=None,
                                 is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Тест на анализ использования нод и инструментов условного графа.
    Parameters:
        sampler: объект класса Sampler
        data_type: Ключ, где лежат трейсы агента
        exec_env: Среда выполнения ('default' или 'sberds')
        frequency_threshold: минимальный процент трейсов, в которых должна быть задействована каждая нода
        red_semaphore_threshold: верхний порог для красного светофора доли нод, которые встречаются в трейсах меньше, чем frequency_threshold (не включая порог)
        yellow_semaphore_threshold: верхний порог для желтого светофора доли нод, которые встречаются в трейсах меньше, чем frequency_threshold (не включая порог)
        is_info: True, если тест информативный
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if nodes_usage_stats is None:
        traces = get_data(sampler, data_type, 'traces', exec_env).copy()
        total_cases = traces['trace_id'].nunique()
        node_freq = traces.groupby('span_name')['trace_id'].nunique() / total_cases
        nodes_usage_stats = node_freq.reset_index(name='frequency')
    precomputed = {'nodes_usage_stats': nodes_usage_stats}
    report = report_valtest_node_usage_analysis(nodes_usage_stats,
                                        frequency_threshold,
                                        red_semaphore_threshold,
                                        yellow_semaphore_threshold,
                                        is_info) 
    return {'report': report,
            'precomputed': precomputed}


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