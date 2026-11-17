import typing as tp

import pandas as pd

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.get_data import get_data
from llm_val.utils.report_helper import worst_semaphore, semaphore_by_threshold_without_red
from llm_val.utils.traces import extract_graph_components as extract_graph_components
from llm_val.utils.traces import find_connected_components as find_connected_components
from llm_val.utils.traces import prepare_traces


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
    checks = [('Количество стартовых нод', len(start_nodes)),
              ('Количество выходных нод', len(end_nodes)),
              ('Количество компонент связности', len(connected_components))]
    results_data = []
    for check_name, check_value in checks:
        semaphore = semaphore_by_threshold_without_red(value=check_value,
                                                       threshold=yellow_semaphore_threshold,
                                                       greater_is_better=False,
                                                       border_is_bigger=False)
        results_data.append([check_name, check_value, semaphore])
    results_df = pd.DataFrame(results_data,
                              columns=['Проверка', 'Результат проверки', 'Светофор'])
    results_df['Результат проверки'] = results_df['Результат проверки'].astype(int)
    overall_color = 'gray' if is_info else worst_semaphore(results_df['Светофор'].tolist())
    res_dict: tp.Dict[str, tp.Any] = {}
    res_dict['semaphore'] = overall_color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [results_df]
    return res_dict


def valtest_trace_connectivity(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                               data_type: str='test',
                               exec_env: str='default',
                               yellow_semaphore_threshold: int=1,
                               start_nodes: tp.Optional[tp.List[str]]=None,
                               end_nodes: tp.Optional[tp.List[str]]=None,
                               connected_components: tp.Optional[tp.List[tp.List[str]]]=None,
                               is_info: bool=False,
                               **kwargs: tp.Any) -> tp.Dict[str, tp.Any]:
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
        traces = prepare_traces(get_data(sampler, data_type, 'traces', exec_env))
        start_nodes, end_nodes, connected_components = extract_graph_components(traces)
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
