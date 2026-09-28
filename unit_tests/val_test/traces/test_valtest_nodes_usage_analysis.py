import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.append('src')

from llm_val.sampler.Sampler import Sampler
from llm_val.val_test.traces.valtest_nodes_usage_analysis import (
    get_data,
    report_valtest_node_usage_analysis,
    valtest_nodes_usage_analysis,
)



@pytest.fixture
def sample_traces_df() -> pd.DataFrame:
    """Создаёт пример DataFrame с трейсами для тестирования."""
    return pd.DataFrame({
        'trace_id': ['t1', 't1', 't2', 't3', 't3', 't3'],
        'span_name': ['nodeA', 'nodeB', 'nodeA', 'nodeB', 'nodeC', 'nodeA']
    })


@pytest.fixture
def sampler_with_traces(sample_traces_df: pd.DataFrame) -> MagicMock:
    """Создаёт мок-объект Sampler с тестовыми трейсами."""
    mock_sampler = MagicMock(spec=Sampler)
    mock_sampler.test = {'traces': sample_traces_df}
    return mock_sampler


@pytest.fixture
def expected_node_stats(sample_traces_df: pd.DataFrame) -> pd.DataFrame:
    """Предварительно рассчитанная статистика использования нод."""
    total_cases = sample_traces_df['trace_id'].nunique()
    node_freq = sample_traces_df.groupby('span_name')['trace_id'].nunique() / total_cases
    return node_freq.reset_index(name='frequency')



def test_get_data_default_env(
    sampler_with_traces: MagicMock,
    sample_traces_df: pd.DataFrame
) -> None:
    """get_data должна извлекать данные из атрибута объекта sampler при exec_env='default'."""
    result = get_data(sampler_with_traces, 'test', 'traces', exec_env='default')
    pd.testing.assert_frame_equal(result, sample_traces_df)


def test_get_data_sberds_env() -> None:
    """get_data должна извлекать данные из словаря при exec_env='sberds'."""
    mock_dict = {'test': {'traces': pd.DataFrame({'a': [1]})}}
    result = get_data(mock_dict, 'test', 'traces', exec_env='sberds')
    pd.testing.assert_frame_equal(result, mock_dict['test']['traces'])


def test_get_data_unknown_env(sampler_with_traces: MagicMock) -> None:
    """При неизвестном exec_env функция возвращает None."""
    result = get_data(sampler_with_traces, 'test', 'traces', exec_env='unknown')
    assert result is None



def test_report_structure() -> None:
    """Словарь отчёта должен содержать обязательные ключи."""
    node_stats = pd.DataFrame({
        'span_name': ['A', 'B'],
        'frequency': [0.1, 0.01]
    })
    report = report_valtest_node_usage_analysis(node_stats, frequency_threshold=0.05)
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_plots'], list)
    assert isinstance(report['result_dataframes'], list)
    assert len(report['result_dataframes']) == 1


def test_report_all_nodes_above_threshold() -> None:
    """Если все ноды превышают порог частоты, светофор должен быть зелёным."""
    node_stats = pd.DataFrame({
        'span_name': ['A', 'B'],
        'frequency': [0.1, 0.2]
    })
    report = report_valtest_node_usage_analysis(
        node_stats,
        frequency_threshold=0.05,
        yellow_semaphore_threshold=0.0,
        red_semaphore_threshold=0.5
    )
    assert report['semaphore'] == 'green'
    table = report['result_dataframes'][0]
    assert 'все ноды' in table.iloc[0, 0].lower()


def test_report_some_nodes_below_threshold_yellow() -> None:
    """Если доля низкочастотных нод превышает жёлтый порог, светофор yellow."""
    node_stats = pd.DataFrame({
        'span_name': ['A', 'B', 'C'],
        'frequency': [0.1, 0.02, 0.01]
    })
    report = report_valtest_node_usage_analysis(
        node_stats,
        frequency_threshold=0.05,
        yellow_semaphore_threshold=0.0,
        red_semaphore_threshold=0.8
    )
    assert report['semaphore'] == 'yellow'
    table = report['result_dataframes'][0]
    assert 'недостаточно' in table.iloc[0, 0].lower()


def test_report_some_nodes_below_threshold_red() -> None:
    """Если доля низкочастотных нод превышает красный порог, светофор red."""
    node_stats = pd.DataFrame({
        'span_name': ['A', 'B'],
        'frequency': [0.01, 0.02]
    })
    report = report_valtest_node_usage_analysis(
        node_stats,
        frequency_threshold=0.05,
        yellow_semaphore_threshold=0.0,
        red_semaphore_threshold=0.5
    )
    assert report['semaphore'] == 'red'
    table = report['result_dataframes'][0]
    assert 'недостаточно' in table.iloc[0, 0].lower()


def test_report_info_mode() -> None:
    """При is_info=True светофор должен быть 'gray' независимо от метрики."""
    node_stats = pd.DataFrame({
        'span_name': ['A'],
        'frequency': [0.0]
    })
    report = report_valtest_node_usage_analysis(
        node_stats,
        frequency_threshold=0.05,
        is_info=True
    )
    assert report['semaphore'] == 'gray'


def test_report_empty_stats() -> None:
    """Пустая статистика должна приводить к доле 0 и зелёному светофору."""
    node_stats = pd.DataFrame(columns=['span_name', 'frequency'])
    report = report_valtest_node_usage_analysis(
        node_stats,
        frequency_threshold=0.05,
        yellow_semaphore_threshold=0.0,
        red_semaphore_threshold=0.5
    )
    assert report['semaphore'] == 'green'


def test_report_bad_nodes_list_in_text() -> None:
    """В тексте отчёта должны быть перечислены ноды, частота которых ниже порога."""
    node_stats = pd.DataFrame({
        'span_name': ['nodeX', 'nodeY', 'nodeZ'],
        'frequency': [0.1, 0.01, 0.02]
    })
    report = report_valtest_node_usage_analysis(
        node_stats,
        frequency_threshold=0.05,
        yellow_semaphore_threshold=0.0,
        red_semaphore_threshold=0.5
    )
    table = report['result_dataframes'][0]
    text = table.iloc[0, 0]
    assert 'nodeY' in text
    assert 'nodeZ' in text
    assert 'nodeX' not in text



def test_valtest_without_precomputed(
    sampler_with_traces: MagicMock,
    expected_node_stats: pd.DataFrame
) -> None:
    """Без переданного nodes_usage_stats статистика должна вычисляться из трейсов."""
    result = valtest_nodes_usage_analysis(
        sampler=sampler_with_traces,
        data_type='test',
        exec_env='default',
        frequency_threshold=0.05
    )
    assert 'report' in result
    assert 'precomputed' in result
    assert 'nodes_usage_stats' in result['precomputed']
    pd.testing.assert_frame_equal(
        result['precomputed']['nodes_usage_stats'],
        expected_node_stats
    )


def test_valtest_with_precomputed(sampler_with_traces: MagicMock) -> None:
    """Если передан nodes_usage_stats, вычисления пропускаются."""
    precomputed_stats = pd.DataFrame({'span_name': ['A'], 'frequency': [0.5]})
    with patch(
        'llm_val.val_test.traces.valtest_nodes_usage_analysis.get_data'
    ) as mock_get_data:
        result = valtest_nodes_usage_analysis(
            sampler=sampler_with_traces,
            data_type='test',
            exec_env='default',
            nodes_usage_stats=precomputed_stats
        )
        mock_get_data.assert_not_called()
    assert result['precomputed']['nodes_usage_stats'] is precomputed_stats


def test_valtest_return_structure(sampler_with_traces: MagicMock) -> None:
    """Возвращаемый словарь должен содержать ключи 'report' и 'precomputed'."""
    result = valtest_nodes_usage_analysis(sampler=sampler_with_traces)
    assert 'report' in result
    assert 'precomputed' in result
    assert 'semaphore' in result['report']
    assert 'result_plots' in result['report']
    assert 'result_dataframes' in result['report']


def test_valtest_calls_report_function(sampler_with_traces: MagicMock) -> None:
    """Проверка, что функция отчёта вызывается с правильными аргументами."""
    with patch(
        'llm_val.val_test.traces.valtest_nodes_usage_analysis.report_valtest_node_usage_analysis'
    ) as mock_report:
        mock_report.return_value = {
            'semaphore': 'green',
            'result_plots': [],
            'result_dataframes': []
        }
        result = valtest_nodes_usage_analysis(
            sampler=sampler_with_traces,
            frequency_threshold=0.1,
            red_semaphore_threshold=0.6,
            yellow_semaphore_threshold=0.2,
            is_info=True
        )
        mock_report.assert_called_once()
        args, kwargs = mock_report.call_args
        assert isinstance(args[0], pd.DataFrame)
        assert args[1] == 0.1
        assert args[2] == 0.6
        assert args[3] == 0.2
        assert args[4] is True


def test_valtest_with_sberds_env() -> None:
    """Тест для окружения sberds, где sampler представлен словарём."""
    mock_sampler_dict = {
        'test': {
            'traces': pd.DataFrame({
                'trace_id': ['t1', 't1'],
                'span_name': ['node1', 'node2']
            })
        }
    }
    result = valtest_nodes_usage_analysis(
        sampler=mock_sampler_dict,
        data_type='test',
        exec_env='sberds',
        frequency_threshold=0.5
    )
    stats = result['precomputed']['nodes_usage_stats']
    assert stats[stats['span_name'] == 'node1']['frequency'].iloc[0] == 1.0
    assert stats[stats['span_name'] == 'node2']['frequency'].iloc[0] == 1.0


def test_valtest_empty_traces() -> None:
    """При пустом DataFrame с трейсами тест должен отработать без ошибок."""
    empty_traces = pd.DataFrame(columns=['trace_id', 'span_name'])
    mock_sampler = MagicMock(spec=Sampler)
    mock_sampler.test = {'traces': empty_traces}
    result = valtest_nodes_usage_analysis(sampler=mock_sampler)
    stats = result['precomputed']['nodes_usage_stats']
    assert stats.empty
    assert result['report']['semaphore'] == 'green'


def test_valtest_trace_with_multiple_same_node_per_trace() -> None:
    """
    При расчёте частоты должны учитываться уникальные trace_id для каждой ноды,
    а не количество вхождений в трейсах.
    """
    traces = pd.DataFrame({
        'trace_id': ['t1', 't1', 't1', 't2'],
        'span_name': ['A', 'A', 'B', 'A']
    })
    mock_sampler = MagicMock(spec=Sampler)
    mock_sampler.test = {'traces': traces}
    result = valtest_nodes_usage_analysis(sampler=mock_sampler)
    stats = result['precomputed']['nodes_usage_stats']
    assert stats[stats['span_name'] == 'A']['frequency'].iloc[0] == 1.0
    assert stats[stats['span_name'] == 'B']['frequency'].iloc[0] == 0.5