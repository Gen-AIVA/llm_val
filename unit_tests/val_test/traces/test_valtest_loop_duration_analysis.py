import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sberpm import DataHolder
from llm_val.sampler.Sampler import Sampler
from llm_val.sampler.TracesSampler import TracesSampler
from llm_val.val_test.traces.valtest_loop_duration_analysis import (
    valtest_loop_duration_analysis,
    calc_time_statistic,
    calc_loop_statistic,
)


DATA_PATH_0 = 'data/data_traces.parquet'
DATA_PATH_1 = 'data/data_traces_with_time_outliers.parquet'
DATA_PATH_2 = 'data/data_traces_with_loops.parquet'


@pytest.fixture()
def create_traces_sampler(request):
    data = pd.read_parquet(DATA_PATH_0)
    sampler = TracesSampler(data)
    res = valtest_loop_duration_analysis(sampler)
    return res


@pytest.fixture()
def create_traces_sampler_with_precomputed_stats(request):
    data = pd.read_parquet(DATA_PATH_0)
    sampler = TracesSampler(data)
    loop_time_stats = pd.DataFrame({
        'Нода': ['stage_1', 'stage_2', 'stage_3'],
        'Среднее время': ['5.12', '3.57', '7.89'],
        'Среднее время без выбросов': ['5.00', '3.50', '7.80'],
        'Зацикленность в начало': ['0.10', '0.05', '0.20'],
        'Зацикленность в себя': ['0.20', '0.10', '0.15'],
        'Зацикленность "Возврат"': ['0.05', '0.03', '0.08'],
        'Зацикленность "Пинг-Понг"': ['0.02', '0.01', '0.03'],
        'Зацикленность в прозвольную ноду': ['0.03', '0.02', '0.04']
    })
    res = valtest_loop_duration_analysis(sampler, loop_time_stats=loop_time_stats)
    return res, loop_time_stats


@pytest.fixture()
def sample_traces_data(request):
    np.random.seed(42)
    traces = []
    trace_ids = [f'trace_{i}' for i in range(10)]
    stages = ['stage_1', 'stage_2', 'stage_3', 'stage_4']
    for trace_id in trace_ids:
        for i, stage in enumerate(stages):
            start_time = datetime(2024, 1, 1, 10, 0, 0) + timedelta(seconds=i*5)
            end_time = start_time + timedelta(seconds=np.random.uniform(1, 10))
            traces.append({
                'trace_id': trace_id,
                'span_name': stage,
                'start_time_ns': start_time,
                'end_time_ns': end_time
            })
    return pd.DataFrame(traces)


@pytest.fixture()
def create_traces_with_time_outliers(request):
    data = pd.read_parquet(DATA_PATH_1)
    data_holder = DataHolder(
        data=data,
        col_case='trace_id',
        col_stage='span_name',
        col_start_time='starttimeunixnano',
        col_end_time='endtimeunixnano'
    )
    return data_holder


@pytest.fixture()
def create_traces_with_loops(request):
    data = pd.read_parquet(DATA_PATH_2)
    data_holder = DataHolder(
        data=data,
        col_case='trace_id',
        col_stage='span_name',
        col_start_time='starttime',
        col_end_time='endtime'
    )
    return data_holder


def test_valtest_loop_duration_precomputed(create_traces_sampler):
    test_res = create_traces_sampler
    precomputed = test_res['precomputed']

    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'loop_time_stats'}
    assert isinstance(precomputed['loop_time_stats'], pd.DataFrame)

    stats_df = precomputed['loop_time_stats']
    assert 'Нода' in stats_df.columns
    assert 'Среднее время' in stats_df.columns
    assert 'Среднее время без выбросов' in stats_df.columns


def test_valtest_loop_duration_with_precomputed_stats(create_traces_sampler_with_precomputed_stats):
    test_res, precomputed_stats = create_traces_sampler_with_precomputed_stats
    precomputed = test_res['precomputed']
    pd.testing.assert_frame_equal(precomputed['loop_time_stats'], precomputed_stats)

    report = test_res['report']
    df = report['result_dataframes'][0]
    pd.testing.assert_frame_equal(df, precomputed_stats)


def test_valtest_calc_time_statistics(sample_traces_data):
    data_holder = DataHolder(data=sample_traces_data,
                             col_case='trace_id',
                             col_stage='span_name',
                             col_start_time='start_time_ns',
                             col_end_time='end_time_ns')
    result = calc_time_statistic(data_holder)
    assert isinstance(result, pd.DataFrame)
    assert 'Нода' in result.columns
    assert 'Среднее время' in result.columns
    assert 'Среднее время без выбросов' in result.columns

    expected_stages = sample_traces_data['span_name'].unique()
    assert set(result['Нода']) == set(expected_stages)

    assert (result['Среднее время'] >= 0).all()
    assert (result['Среднее время без выбросов'] >= 0).all()


def test_calc_time_statistic_with_outliers(create_traces_with_time_outliers):
    data_holder = create_traces_with_time_outliers
    result = calc_time_statistic(data_holder)
    assert result.loc[0, 'Среднее время'] >= result.loc[0, 'Среднее время без выбросов']


def test_calc_loop_statistic_without_loop(sample_traces_data):
    data_holder = DataHolder(
        data=sample_traces_data,
        col_case='trace_id',
        col_stage='span_name',
        col_start_time='start_time_ns',
        col_end_time='end_time_ns'
    )
    result = calc_loop_statistic(data_holder)
    assert isinstance(result, pd.DataFrame)
    for column in result.columns[1:]:
        assert sum(result[column]) == 0

    
def test_calc_loop_statistic_with_loop(create_traces_with_loops):
    result = calc_loop_statistic(create_traces_with_loops)
    for column in result.columns[1:]:
        assert sum(result[column]) >= 1.0


def test_valtest_loop_duration_merge_columns_integrity(create_traces_sampler):
    test_res = create_traces_sampler
    stats_df = test_res['precomputed']['loop_time_stats']
    expected_columns = [
        'Нода',
        'Среднее время',
        'Среднее время без выбросов',
        'Зацикленность в начало',
        'Зацикленность в себя',
        'Зацикленность "Возврат"',
        'Зацикленность "Пинг-Понг"',
        'Зацикленность в прозвольную ноду'
    ]
    for col in expected_columns:
        assert col in stats_df.columns, f"Колонка '{col}' отсутствует в результате"

