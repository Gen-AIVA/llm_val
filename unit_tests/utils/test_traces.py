import json
import os
from pathlib import Path
import subprocess
import sys
import typing as tp

import pandas as pd
import pytest

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.traces import calc_loop_statistic, calc_time_statistic, extract_graph_components, prepare_traces
from llm_val.val_test.traces.valtest_loop_duration_analysis import valtest_loop_duration_analysis
from llm_val.val_test.traces.valtest_trace_connectivity import valtest_trace_connectivity


def _traces(cases: tp.Dict[str, tp.List[str]]) -> pd.DataFrame:
    '''
    Создать события с известными порядком и секундной длительностью.
    '''
    records = [{'trace_id': case,
                'span_name': stage,
                'start_time_ns': position * 2_000_000_000,
                'end_time_ns': (position * 2 + 1) * 1_000_000_000}
               for case, stages in cases.items()
               for position, stage in enumerate(stages)]
    return pd.DataFrame(records, columns=['trace_id', 'span_name', 'start_time_ns', 'end_time_ns'])


def test_loop_statistics_match_reference() -> None:
    '''
    Сравнить все пять метрик с сохранённым результатом прежнего анализатора 3.4.3.
    '''
    fixture = Path(__file__).resolve().parents[1] / 'data' / 'traces_loop_statistics.json'
    reference = json.loads(fixture.read_text(encoding='utf-8'))
    traces = prepare_traces(pd.DataFrame(reference['records']))
    pd.testing.assert_frame_equal(calc_loop_statistic(traces), pd.DataFrame(reference['expected']))


def test_prepare_traces_sorts_and_keeps_input_unchanged() -> None:
    '''
    Перемешанные события и нестандартный индекс не меняют порядок расчётов.
    '''
    data = _traces({'one': ['z', 'a', 'b'], 'two': ['c', 'd']})
    shuffled = data.sample(frac=1, random_state=42)
    shuffled.index = [90, 70, 50, 30, 10]
    original = shuffled.copy(deep=True)
    pd.testing.assert_frame_equal(prepare_traces(shuffled), prepare_traces(data))
    pd.testing.assert_frame_equal(shuffled, original)


@pytest.mark.parametrize('representation', ['numeric', 'datetime', 'string'])
def test_timestamps_and_duration_seconds(representation: str) -> None:
    '''
    Поддержать наносекунды, datetime и строки с той же длительностью в секундах.
    '''
    data = _traces({'one': ['a']})
    data['start_time_ns'] = 1_700_000_000_000_000_000
    data['end_time_ns'] = 1_700_000_001_125_000_000
    if representation != 'numeric':
        for column in ['start_time_ns', 'end_time_ns']:
            data[column] = pd.to_datetime(data[column], utc=True)
            if representation == 'string':
                data[column] = data[column].astype(str)
    assert prepare_traces(data)['duration'].tolist() == [1.125]


def test_time_statistics_keep_strict_quantiles() -> None:
    '''
    Исключить значения на границах квантилей и оставить NaN для константной ноды.
    '''
    data = _traces({'one': ['a'] * 5 + ['b'] * 2})
    durations = pd.Series([1, 2, 3, 4, 100, 2, 2])
    data['end_time_ns'] = data['start_time_ns'] + durations * 1_000_000_000
    statistics = calc_time_statistic(prepare_traces(data)).set_index('Нода')
    assert statistics.loc['a', 'Среднее время'] == 22.0
    assert statistics.loc['a', 'Среднее время без выбросов'] == 3.0
    assert pd.isna(statistics.loc['b', 'Среднее время без выбросов'])


def test_graph_does_not_join_independent_traces() -> None:
    '''
    Сохранить отдельные компоненты, порядок появления и одиночные ноды.
    '''
    traces = prepare_traces(_traces({'one': ['z', 'a'], 'two': ['b'], 'three': ['x', 'y']}))
    starts, ends, components = extract_graph_components(traces)
    assert starts == ['z', 'x', 'b']
    assert ends == ['a', 'y', 'b']
    assert components == [['z', 'a'], ['x', 'y'], ['b']]


def test_graph_stage_names_are_not_parsed_as_edges() -> None:
    '''
    Имена нод с зарезервированными подстроками возвращаются полностью.
    '''
    traces = prepare_traces(_traces({'one': ['startevent', 'a_b', 'endevent']}))
    assert extract_graph_components(traces) == (['startevent'], ['endevent'], [['startevent', 'a_b', 'endevent']])


def test_loops_do_not_cross_case_boundaries() -> None:
    '''
    Одинаковые соседние ноды разных трейсов не создают цикл.
    '''
    statistics = calc_loop_statistic(prepare_traces(_traces({'one': ['b', 'a'], 'two': ['a', 'b']})))
    assert statistics['Нода'].tolist() == ['b', 'a']
    assert (statistics.drop(columns='Нода') == 0).all().all()


def test_single_stage_has_zero_normalized_metrics() -> None:
    '''
    Сохранить нулевую min-max нормализацию константного столбца.
    '''
    statistics = calc_loop_statistic(prepare_traces(_traces({'one': ['a'] * 5})))
    assert (statistics.drop(columns='Нода') == 0).all().all()


def test_empty_traces_keep_output_columns() -> None:
    '''
    Пустые данные сохраняют схему статистик и дают пустой граф.
    '''
    traces = prepare_traces(_traces({}))
    assert extract_graph_components(traces) == ([], [], [])
    assert calc_time_statistic(traces).columns.tolist() == ['Нода', 'Среднее время', 'Среднее время без выбросов']
    assert calc_loop_statistic(traces).columns.tolist() == ['Нода',
                                                          'В начало',
                                                          'В себя',
                                                          '«Возврат»',
                                                          '«Пинг-Понг»',
                                                          'В произвольную операцию']
    result = valtest_loop_duration_analysis({'test': {'traces': traces}}, exec_env='sberds')
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}
    assert result['precomputed']['loop_time_stats'].empty


def test_unused_categories_do_not_add_nodes() -> None:
    '''
    Категории без наблюдений не создают фиктивные входы и ноды.
    '''
    data = _traces({'one': ['a', 'b']})
    data['trace_id'] = pd.Categorical(data['trace_id'], categories=['one', 'unused'])
    data['span_name'] = pd.Categorical(data['span_name'], categories=['a', 'b', 'unused'])
    traces = prepare_traces(data)
    assert extract_graph_components(traces) == (['a'], ['b'], [['a', 'b']])
    assert calc_loop_statistic(traces)['Нода'].tolist() == ['a', 'b']


def test_missing_values_follow_trace_preprocessing_contract() -> None:
    '''
    Пропустить неизвестные ID/ноды, сохранив событие с неизвестной длительностью.
    '''
    data = _traces({'one': ['a', '<unknown>', 'null', 'b']})
    data.loc[0, 'end_time_ns'] = None
    traces = prepare_traces(data)
    assert traces['span_name'].tolist() == ['a', 'b']
    assert pd.isna(traces.loc[0, 'duration'])


def test_missing_required_column_fails() -> None:
    '''
    Отсутствующая колонка не подменяется фиктивными данными.
    '''
    data = _traces({'one': ['a']}).drop(columns='span_name')
    with pytest.raises(KeyError, match='span_name'):
        prepare_traces(data)


def test_public_tests_support_both_environments() -> None:
    '''
    Обычный Sampler и формат SberDS возвращают одинаковые отчёты.
    '''
    data = {'test': {'traces': _traces({'one': ['a', 'b', 'a']}), 'X': pd.DataFrame(), 'y': pd.DataFrame()}}
    sampler = Sampler(test=data['test'])
    valtests: tp.List[tp.Callable[..., tp.Dict[str, tp.Any]]] = [valtest_trace_connectivity,
                                                              valtest_loop_duration_analysis]
    for valtest in valtests:
        result = valtest(sampler)
        assert isinstance(result, dict)
        assert set(result.keys()) == {'report', 'precomputed'}
        as_dictionary = valtest(data, exec_env='sberds')
        assert isinstance(as_dictionary, dict)
        assert set(as_dictionary.keys()) == {'report', 'precomputed'}
        pd.testing.assert_frame_equal(result['report']['result_dataframes'][0],
                                      as_dictionary['report']['result_dataframes'][0])


def test_precomputed_results_do_not_read_sampler() -> None:
    '''
    Переданные результаты возвращаются без доступа к данным сэмплера.
    '''
    table = pd.DataFrame({'Нода': ['a'], 'Среднее время': ['1.00']})
    result = valtest_loop_duration_analysis({}, loop_time_stats=table)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}
    assert result['precomputed']['loop_time_stats'] is table
    graph = valtest_trace_connectivity({}, start_nodes=['a'], end_nodes=['a'], connected_components=[['a']])
    assert isinstance(graph, dict)
    assert set(graph.keys()) == {'report', 'precomputed'}
    assert graph['report']['semaphore'] == 'green'


def test_trace_checks_run_with_dependency_blocked() -> None:
    '''
    Импорт и расчёт обоих тестов работают при заблокированном импорте старой зависимости.
    '''
    source = Path(__file__).resolve().parents[2] / 'src'
    script = """
import sys
sys.modules['sberpm'] = None
import pandas as pd
from llm_val.val_test.traces.valtest_trace_connectivity import valtest_trace_connectivity
from llm_val.val_test.traces.valtest_loop_duration_analysis import valtest_loop_duration_analysis
data = {'test': {'traces': pd.DataFrame({
    'trace_id': ['one', 'one'], 'span_name': ['a', 'b'],
    'start_time_ns': [0, 2000000000], 'end_time_ns': [1000000000, 3000000000],
})}}
for check in [valtest_trace_connectivity, valtest_loop_duration_analysis]:
    result = check(data, exec_env='sberds')
    assert set(result) == {'report', 'precomputed'}
"""
    subprocess.run([sys.executable, '-c', script],
                   check=True,
                   capture_output=True,
                   text=True,
                   env={**os.environ, 'PYTHONPATH': str(source), 'PYTHONDONTWRITEBYTECODE': '1'})
