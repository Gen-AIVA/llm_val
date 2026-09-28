import os 
import numpy as np
import pandas as pd
import typing as tp
from collections import deque
from sberpm import DataHolder
from sberpm.visual import Graph
from sberpm.miners import SimpleMiner

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import worst_semaphore
from llm_val.utils.report_helper import semaphore_by_threshold
from llm_val.utils.report_helper import semaphore_by_threshold_without_red


def report_valtest_trace_connectivity(start_nodes: tp.List[str],
                                      end_nodes: tp.List[str],
                                      connected_components: tp.List[tp.List[str]],
                                      yellow_semaphore_threshold: int=1,
                                      is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'a
    Parameters:
        start_nodes: список начальных вершин графа
        end_nodes: список конечных вершин графа
        connected_components: список компонент связности
        yellow_semaphore_threshold: порог для желтого светофора за все проверки теста
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    checks = [
        ('Количество стартовых нод', len(start_nodes)),
        ('Количество выходных нод', len(end_nodes)),
        ('Количество компонент связности', len(connected_components)),
    ]
    results_data = []
    semaphores = []
    for check_name, check_value in checks:
        semaphore = semaphore_by_threshold_without_red(value=check_value,
                                                       threshold=yellow_semaphore_threshold,
                                                       greater_is_better=False,
                                                       border_is_bigger=False)
        semaphores.append(semaphore)
        results_data.append([check_name, check_value, semaphore])
    results_df = pd.DataFrame(results_data,
                              columns=['Проверка', 
                                       'Результат проверки', 
                                       'Светофор'])
    results_df['Результат проверки'] = results_df['Результат проверки'].astype(int)
    overall_color = 'gray' if is_info else worst_semaphore(results_df['Светофор'].tolist())
    res_dict = {}
    res_dict['semaphore'] = overall_color
    res_dict['result_plots'] = []  
    res_dict['result_dataframes'] = [results_df]
    return res_dict


def valtest_trace_connectivity(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                               data_type: str='test',
                               exec_env: str='default',
                               yellow_semaphore_threshold: int=1,
                               start_nodes: tp.List[str]=None,
                               end_nodes: tp.List[str]=None,
                               connected_components: tp.List[tp.List[str]]=None,
                               is_info: bool=False,
                               **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на связность трейсов.
    Parameters:
        sampler: Объект класса Sampler
        data_type: Ключ, где лежат трейсы агента
        exec_env: среда выполнения ('default' или 'sberds')
        yellow_semaphore_threshold: порог для желтого светофора за все проверки теста
        start_nodes: список начальный вершин графа агента 
        end_nodes: список конечных вершин графа агента
        connected_components: список компонент связности графа
        is_info: True, если тест информаивный 
        **kwargs: дополнительные параметры
    Return:
        Словарь в соответствии с концепцией с llm_val
    '''
    if start_nodes is None or end_nodes is None or connected_components is None:
        traces = get_data(sampler, data_type, 'traces', exec_env).copy()
        data_holder = DataHolder(data=traces,
                                 col_case='trace_id',
                                 col_stage='span_name',
                                 col_start_time='start_time_ns',
                                 col_end_time='end_time_ns',)
        graph_miner = SimpleMiner(data_holder)
        graph_miner.apply()
        graph = graph_miner.graph
        start_nodes, end_nodes, connected_components = extract_graph_components(graph)
    report = report_valtest_trace_connectivity(start_nodes=start_nodes,
                                               end_nodes=end_nodes,
                                               connected_components=connected_components,
                                               yellow_semaphore_threshold=yellow_semaphore_threshold,
                                               is_info=is_info)
    precomputed = {'start_nodes': start_nodes,
                   'end_nodes': end_nodes,
                   'connected_components': connected_components}
    return {'report': report,
            'precomputed': precomputed}


def find_connected_components(adj_matrix: tp.List[tp.List[int]]) -> tp.List[tp.List[int]]:
    '''
    Находит компоненты связности в графе.
    Parameters:
        adj_matrix: матрица смежности графа
    Return:
        Список компонент связности (каждая компонента - список индексов вершин)
    '''
    n = len(adj_matrix)
    visited = [False] * n
    components = []
    for node in range(n):
        if not visited[node]: 
            component = [] 
            queue = deque([node])
            visited[node] = True
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in range(n):
                    if adj_matrix[current][neighbor] == 1 and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(component) 
    return components


def extract_graph_components(graph: Graph) -> tp.Tuple[tp.List[str], tp.List[str], tp.List[tp.List[str]]]:
    '''
    Извлекает из графа начальные/конечные вершины и компоненты связности.
    Parameters:
        graph: граф процессов из библиотеки sberPM
    Return:
        Список начальных вершин, список конечных вершин, список компонент связности
    '''
    start_nodes = [str(edge)[17:] for edge in graph.get_edges() if 'startevent' in str(edge)]
    end_nodes = [str(edge)[6:-9] for edge in graph.get_edges() if 'endevent' in str(edge)]
    adj_matrix, nodes_mapping = graph.get_adjacency_matrix()
    components = find_connected_components(adj_matrix)
    reverse_nodes_map = {v: k for k, v in nodes_mapping.items()}
    readable_components = []
    for comp in components:
        new_components = [reverse_nodes_map[node] for node in comp]
        readable_components.append(new_components)
    return start_nodes, end_nodes, readable_components


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
