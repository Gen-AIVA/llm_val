import pytest
import numpy as np
import pandas as pd

from sberpm import DataHolder
from sberpm.miners import SimpleMiner
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from llm_val.sampler.Sampler import Sampler
from llm_val.sampler.TracesSampler import TracesSampler
from llm_val.val_test.traces.valtest_trace_connectivity import (
    valtest_trace_connectivity,
    find_connected_components,
    extract_graph_components,
)

DATA_PATH_0 = 'data/data_traces.parquet'
DATA_PATH_1 = 'data/data_traces_1.parquet'
DATA_PATH_2 = 'data/data_traces_2.parquet'
DATA_PATH_3 = 'data/data_traces_3.parquet'

@pytest.fixture()
def create_traces_sampler(request):
    data = pd.read_parquet(DATA_PATH_0)
    sampler = TracesSampler(data)
    res = valtest_trace_connectivity(sampler)
    return res


@pytest.fixture()
def create_traces_sampler_with_precomputed(request):
    data = pd.read_parquet(DATA_PATH_0)
    sampler = TracesSampler(data)
    start_nodes = ['start_1', 'start_2']
    end_nodes = ['end_1', 'end_2']
    connected_components = [['stage_1', 'stage_2'], ['stage_3', 'stage_4']]
    res = valtest_trace_connectivity(
        sampler,
        start_nodes=start_nodes,
        end_nodes=end_nodes,
        connected_components=connected_components
    )
    return res, start_nodes, end_nodes, connected_components


@pytest.fixture()
def sampler_connected_traces():
    data = pd.read_parquet(DATA_PATH_1)
    sampler = TracesSampler(data)
    return sampler


@pytest.fixture()
def sampler_disconnected_traces():
    data = pd.read_parquet(DATA_PATH_2)
    sampler = TracesSampler(data)
    return sampler


@pytest.fixture()
def sample_traces_multiple_starts_ends():
    data = pd.read_parquet(DATA_PATH_3)
    sampler = TracesSampler(data)
    return sampler


def test_valtest_trace_connectivity_precomputed(create_traces_sampler):
    test_res = create_traces_sampler
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'start_nodes', 'end_nodes', 'connected_components'}
    assert isinstance(precomputed['start_nodes'], list)
    assert isinstance(precomputed['end_nodes'], list)
    assert isinstance(precomputed['connected_components'], list)


def test_valtest_trace_connectivity_with_precomputed(create_traces_sampler_with_precomputed):
    test_res, start_nodes, end_nodes, components = create_traces_sampler_with_precomputed
    precomputed = test_res['precomputed']
    assert precomputed['start_nodes'] == start_nodes
    assert precomputed['end_nodes'] == end_nodes
    assert precomputed['connected_components'] == components
    
    report = test_res['report']
    df = report['result_dataframes'][0]
    assert len(df) == 3


def test_find_connected_components():
    adj_matrix_connected = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]
    components = find_connected_components(adj_matrix_connected)
    assert len(components) == 1
    assert set(components[0]) == {0, 1, 2}

    adj_matrix_disconnected = [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ]
    components = find_connected_components(adj_matrix_disconnected)
    assert len(components) == 2
    assert set(components[0]) == {0, 1}
    assert set(components[1]) == {2, 3}

    adj_matrix_isolated = [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ]
    components = find_connected_components(adj_matrix_isolated)
    assert len(components) == 2
    assert set(components[0]) == {0, 1}
    assert set(components[1]) == {2}
    
    adj_matrix_empty = []
    components = find_connected_components(adj_matrix_empty)
    assert components == []

    adj_matrix_single = [[0]]
    components = find_connected_components(adj_matrix_single)
    assert len(components) == 1
    assert components[0] == [0]


def test_extract_graph_components(sampler_connected_traces):
    data_holder = DataHolder(
        data=sampler_connected_traces.test['traces'],
        col_case='trace_id',
        col_stage='span_name',
        col_start_time='start_time_ns',
        col_end_time='end_time_ns'
    )
    graph_miner = SimpleMiner(data_holder)
    graph_miner.apply()
    graph = graph_miner.graph
    start_nodes, end_nodes, components = extract_graph_components(graph)
    assert start_nodes == ['stage_1']
    assert end_nodes == ['stage_3']
    assert components == [['stage_1', 'stage_2', 'stage_3']]


def test_valtest_trace_connectivity_connected_graph(sampler_connected_traces):
    result = valtest_trace_connectivity(sampler_connected_traces)
    report = result['report']
    df = report['result_dataframes'][0]
    
    components_count = df[df['Проверка'] == 'Количество компонент связности']['Результат проверки'].values[0]
    assert components_count == 1
    
    components_semaphore = df[df['Проверка'] == 'Количество компонент связности']['Светофор'].values[0]
    assert components_semaphore == 'green'


def test_valtest_trace_connectivity_disconnected_graph(sampler_disconnected_traces):
    result = valtest_trace_connectivity(sampler_disconnected_traces)
    report = result['report']
    df = report['result_dataframes'][0]

    components_count = df[df['Проверка'] == 'Количество компонент связности']['Результат проверки'].values[0]
    assert components_count > 1


def test_valtest_trace_connectivity_multiple_starts_ends(sample_traces_multiple_starts_ends):
    result = valtest_trace_connectivity(sample_traces_multiple_starts_ends)
    precomputed = result['precomputed']
    assert len(precomputed['start_nodes']) >= 2
    assert len(precomputed['end_nodes']) >= 2


def test_valtest_trace_connectivity_custom_threshold():
    data = pd.read_parquet(DATA_PATH_0)
    sampler = TracesSampler(data)

    result_threshold_1 = valtest_trace_connectivity(sampler, yellow_semaphore_threshold=1)
    result_threshold_2 = valtest_trace_connectivity(sampler, yellow_semaphore_threshold=2)
    result_threshold_3 = valtest_trace_connectivity(sampler, yellow_semaphore_threshold=4)
    df1 = result_threshold_1['report']['result_dataframes'][0]
    df2 = result_threshold_2['report']['result_dataframes'][0]
    df3 = result_threshold_3['report']['result_dataframes'][0]
    amber_count_1 = sum(df1['Светофор'] == 'yellow')
    amber_count_2 = sum(df2['Светофор'] == 'yellow')
    amber_count_3 = sum(df3['Светофор'] == 'yellow')

    assert amber_count_1 == 2
    assert amber_count_2 == 1
    assert amber_count_3 == 0