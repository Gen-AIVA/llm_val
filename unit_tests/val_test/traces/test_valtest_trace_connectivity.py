import pytest
import pandas as pd
import typing as tp

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.traces import prepare_traces
from llm_val.sampler.TracesSampler import TracesSampler
from llm_val.val_test.traces.valtest_trace_connectivity import (valtest_trace_connectivity,
                                                             find_connected_components,
                                                             extract_graph_components)

DATA_PATH_0 = 'data/data_traces.parquet'
DATA_PATH_1 = 'data/data_traces_1.parquet'
DATA_PATH_2 = 'data/data_traces_2.parquet'
DATA_PATH_3 = 'data/data_traces_3.parquet'


@pytest.fixture
def create_traces_sampler() -> tp.Any:
    '''
    Результат проверки на полной тестовой корзине.
    '''
    data = pd.read_parquet(DATA_PATH_0)
    sampler = TracesSampler(data)
    res = valtest_trace_connectivity(sampler)
    assert isinstance(res, dict)
    assert set(res.keys()) == {'report', 'precomputed'}
    return res


@pytest.fixture
def create_traces_sampler_with_precomputed() -> tp.Any:
    '''
    Результат проверки с готовыми компонентами графа.
    '''
    data = pd.read_parquet(DATA_PATH_0)
    sampler = TracesSampler(data)
    start_nodes = ['start_1', 'start_2']
    end_nodes = ['end_1', 'end_2']
    connected_components = [['stage_1', 'stage_2'], ['stage_3', 'stage_4']]
    res = valtest_trace_connectivity(sampler,
                                     start_nodes=start_nodes,
                                     end_nodes=end_nodes,
                                     connected_components=connected_components)
    assert isinstance(res, dict)
    assert set(res.keys()) == {'report', 'precomputed'}
    return res, start_nodes, end_nodes, connected_components


@pytest.fixture
def sampler_connected_traces() -> tp.Any:
    '''
    Сэмплер с одним линейным трейсом.
    '''
    data = pd.read_parquet(DATA_PATH_1)
    traces = prepare_traces(data, 'traceid', 'name', 'starttimeunixnano', 'endtimeunixnano')
    sampler = Sampler(test={'X': pd.DataFrame(), 'y': pd.DataFrame(), 'traces': traces})
    return sampler


@pytest.fixture
def sampler_disconnected_traces() -> tp.Any:
    '''
    Сэмплер с несвязанными трейсами.
    '''
    data = pd.read_parquet(DATA_PATH_2)
    traces = prepare_traces(data, 'traceid', 'name', 'starttimeunixnano', 'endtimeunixnano')
    sampler = Sampler(test={'X': pd.DataFrame(), 'y': pd.DataFrame(), 'traces': traces})
    return sampler


@pytest.fixture
def sample_traces_multiple_starts_ends() -> tp.Any:
    '''
    Сэмплер с несколькими входами и выходами.
    '''
    data = pd.read_parquet(DATA_PATH_3)
    traces = prepare_traces(data, 'traceid', 'name', 'starttimeunixnano', 'endtimeunixnano')
    sampler = Sampler(test={'X': pd.DataFrame(), 'y': pd.DataFrame(), 'traces': traces})
    return sampler


def test_valtest_trace_connectivity_precomputed(create_traces_sampler: tp.Any) -> None:
    '''
    Проверить формат предрассчитанных компонент.
    '''
    test_res = create_traces_sampler
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'start_nodes', 'end_nodes', 'connected_components'}
    assert isinstance(precomputed['start_nodes'], list)
    assert isinstance(precomputed['end_nodes'], list)
    assert isinstance(precomputed['connected_components'], list)


def test_valtest_trace_connectivity_with_precomputed(create_traces_sampler_with_precomputed: tp.Any) -> None:
    '''
    Проверить переиспользование готовых компонент.
    '''
    test_res, start_nodes, end_nodes, components = create_traces_sampler_with_precomputed
    precomputed = test_res['precomputed']
    assert precomputed['start_nodes'] == start_nodes
    assert precomputed['end_nodes'] == end_nodes
    assert precomputed['connected_components'] == components

    report = test_res['report']
    df = report['result_dataframes'][0]
    assert len(df) == 3


def test_find_connected_components() -> None:
    '''
    Проверить обход связных, раздельных и пустых графов.
    '''
    adj_matrix_connected = [[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]]
    components = find_connected_components(adj_matrix_connected)
    assert len(components) == 1
    assert set(components[0]) == {0, 1, 2}

    adj_matrix_disconnected = [[0, 1, 0, 0],
                               [1, 0, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, 1, 0]]
    components = find_connected_components(adj_matrix_disconnected)
    assert len(components) == 2
    assert set(components[0]) == {0, 1}
    assert set(components[1]) == {2, 3}

    adj_matrix_isolated = [[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]]
    components = find_connected_components(adj_matrix_isolated)
    assert len(components) == 2
    assert set(components[0]) == {0, 1}
    assert set(components[1]) == {2}

    adj_matrix_empty: tp.List[tp.List[int]] = []
    components = find_connected_components(adj_matrix_empty)
    assert components == []

    adj_matrix_single = [[0]]
    components = find_connected_components(adj_matrix_single)
    assert len(components) == 1
    assert components[0] == [0]


def test_extract_graph_components(sampler_connected_traces: tp.Any) -> None:
    '''
    Проверить начальную и конечную ноды линейного трейса.
    '''
    start_nodes, end_nodes, components = extract_graph_components(prepare_traces(sampler_connected_traces.test['traces']))
    assert start_nodes == ['stage_1']
    assert end_nodes == ['stage_3']
    assert components == [['stage_1', 'stage_2', 'stage_3']]


def test_valtest_trace_connectivity_connected_graph(sampler_connected_traces: tp.Any) -> None:
    '''
    Проверить зелёный светофор связного графа.
    '''
    result = valtest_trace_connectivity(sampler_connected_traces)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}
    report = result['report']
    df = report['result_dataframes'][0]

    components_count = df[df['Проверка'] == 'Количество компонент связности']['Результат проверки'].values[0]
    assert components_count == 1

    components_semaphore = df[df['Проверка'] == 'Количество компонент связности']['Светофор'].values[0]
    assert components_semaphore == 'green'


def test_valtest_trace_connectivity_disconnected_graph(sampler_disconnected_traces: tp.Any) -> None:
    '''
    Проверить выявление нескольких компонент.
    '''
    result = valtest_trace_connectivity(sampler_disconnected_traces)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}
    report = result['report']
    df = report['result_dataframes'][0]

    components_count = df[df['Проверка'] == 'Количество компонент связности']['Результат проверки'].values[0]
    assert components_count > 1


def test_valtest_trace_connectivity_multiple_starts_ends(sample_traces_multiple_starts_ends: tp.Any) -> None:
    '''
    Проверить несколько входов и выходов.
    '''
    result = valtest_trace_connectivity(sample_traces_multiple_starts_ends)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}
    precomputed = result['precomputed']
    assert len(precomputed['start_nodes']) >= 2
    assert len(precomputed['end_nodes']) >= 2


def test_valtest_trace_connectivity_custom_threshold() -> None:
    '''
    Проверить влияние порогов на светофоры.
    '''
    data = pd.read_parquet(DATA_PATH_0)
    sampler = TracesSampler(data)

    result_threshold_1 = valtest_trace_connectivity(sampler, yellow_semaphore_threshold=1)
    assert isinstance(result_threshold_1, dict)
    assert set(result_threshold_1.keys()) == {'report', 'precomputed'}
    result_threshold_2 = valtest_trace_connectivity(sampler, yellow_semaphore_threshold=2)
    assert isinstance(result_threshold_2, dict)
    assert set(result_threshold_2.keys()) == {'report', 'precomputed'}
    result_threshold_3 = valtest_trace_connectivity(sampler, yellow_semaphore_threshold=4)
    assert isinstance(result_threshold_3, dict)
    assert set(result_threshold_3.keys()) == {'report', 'precomputed'}
    df1 = result_threshold_1['report']['result_dataframes'][0]
    df2 = result_threshold_2['report']['result_dataframes'][0]
    df3 = result_threshold_3['report']['result_dataframes'][0]
    amber_count_1 = sum(df1['Светофор'] == 'yellow')
    amber_count_2 = sum(df2['Светофор'] == 'yellow')
    amber_count_3 = sum(df3['Светофор'] == 'yellow')

    assert amber_count_1 == 2
    assert amber_count_2 == 1
    assert amber_count_3 == 0
