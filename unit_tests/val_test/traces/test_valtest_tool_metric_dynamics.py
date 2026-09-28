import pandas as pd
import pytest
import typing as tp

from llm_val.sampler.ToolTracesSampler import ToolTracesSampler
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.utils.metrics import METRICS
from llm_val.val_test.traces.valtest_tool_metric_dynamics import (
    valtest_tool_metric_dynamics,
    compute_tool_metrics,
    format_result_table,
    calc_overall_metrics,
    calc_metric_from_data,
)


@pytest.fixture
def sample_traces() -> pd.DataFrame:
    '''Создаёт синтетические трейсы.'''
    return pd.DataFrame(
        {
            'traceid': ['t1', 't1', 't1', 't2', 't2', 't3', 't3', 't3', 't3'],
            'spanid': ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9'],
            'name': [
                'router',
                'GigaChat',
                'scenario_node',
                'router',
                'GigaChat',
                'router',
                'GigaChat',
                'scenario_node',
                'buttons_node',
            ],
            'session_id': ['sess1', 'sess1', 'sess1', 'sess2', 'sess2', 'sess3', 'sess3', 'sess3', 'sess3'],
            'aef_kind': ['tool', 'llm', 'tool', 'tool', 'llm', 'tool', 'llm', 'tool', 'tool'],
            'starttimeunixnano': [0] * 9,
            'endtimeunixnano': [1000] * 9,
            'estimation': [0.8, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4],
        }
    )


@pytest.fixture
def sample_sampler(sample_traces: pd.DataFrame) -> ToolTracesSampler:
    '''Создаёт ToolTracesSampler с трейсами.'''
    return ToolTracesSampler(test=sample_traces)


@pytest.fixture
def sample_sampler_y(sample_sampler: ToolTracesSampler) -> pd.DataFrame:
    '''Возвращает y сэмплера (для проверки индекса).'''
    return sample_sampler.test['y']


@pytest.fixture
def sample_scorer() -> UniversalScorer:
    '''Создаёт UniversalScorer для теста.'''
    return UniversalScorer(metrics=METRICS)


@pytest.fixture
def traces_without_aef_kind() -> pd.DataFrame:
    '''Трейсы без колонки aef_kind.'''
    return pd.DataFrame(
        {
            'traceid': ['t1', 't1', 't2', 't2'],
            'spanid': ['s1', 's2', 's3', 's4'],
            'name': ['router', 'GigaChat', 'router', 'GigaChat'],
            'session_id': ['sess1', 'sess1', 'sess2', 'sess2'],
            'starttimeunixnano': [0, 100, 200, 300],
            'endtimeunixnano': [1000, 1100, 1200, 1300],
            'estimation': [0.9, 0.8, 0.7, 0.6],
        }
    )


@pytest.fixture
def sampler_without_aef_kind(traces_without_aef_kind: pd.DataFrame) -> ToolTracesSampler:
    '''Сэмплер без aef_kind.'''
    return ToolTracesSampler(test=traces_without_aef_kind)


class TestCalcOverallMetrics:
    '''Тесты для calc_overall_metrics.'''

    def test_returns_dict_with_scorer(self,
                                      sample_sampler: ToolTracesSampler,
                                      sample_scorer: UniversalScorer) -> None:
        '''Со scorer возвращается словарь метрик.'''
        result = calc_overall_metrics(
            sampler=sample_sampler,
            scorer=sample_scorer,
            data_type='test',
            exec_env='default',
            metric_agg='single_mean',
        )
        assert isinstance(result, dict)
        assert len(result) > 0
        for key, val in result.items():
            assert isinstance(key, str)
            assert isinstance(val, float)


class TestCalcMetricFromData:
    '''Тесты для calc_metric_from_data.'''

    def test_single_mean_singlecol(self) -> None:
        '''single_mean с одной колонкой.'''
        data = pd.DataFrame({'estimation': [0.5, 0.7, 0.9]})
        result = calc_metric_from_data(data, metric_name='single_mean')
        assert isinstance(result, dict)
        # ключ вида single_mean_estimation
        assert any('single_mean' in k for k in result)
        assert abs(list(result.values())[0] - 0.7) < 1e-6


class TestComputeToolMetrics:
    '''Тесты для compute_tool_metrics.'''

    def test_basic_case(self,
                        sample_sampler: ToolTracesSampler,
                        sample_scorer: UniversalScorer) -> None:
        '''Базовая проверка: возвращается DataFrame с ожидаемыми колонками.'''
        overall = 0.6
        result = compute_tool_metrics(
            sampler=sample_sampler,
            scorer=sample_scorer,
            data_type='test',
            exec_env='default',
            metric_column='estimation',
            metric_agg='single_mean',
            overall_metrics={'default': overall},
        )

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {'span_name', 'metric_name', 'metric_value', 'abs_change', 'rel_change'}

    def test_tool_100_percent_removed(self,
                                      sample_sampler: ToolTracesSampler,
                                      sample_scorer: UniversalScorer) -> None:
        '''Инструменты в 100% сессий удаляются.'''
        overall = 0.5
        result = compute_tool_metrics(
            sampler=sample_sampler,
            scorer=sample_scorer,
            data_type='test',
            exec_env='default',
            metric_column='estimation',
            metric_agg='single_mean',
            overall_metrics={'default': overall},
        )

        assert len(result) > 0
        # router присутствует во всех 3 сессиях — должен быть удалён
        assert 'router' not in result['span_name'].values, 'router встречается во всех сессиях и должен быть удалён'

    def test_no_filter_column(self,
                              sample_sampler: ToolTracesSampler,
                              sample_scorer: UniversalScorer) -> None:
        '''Без aef_kind фильтр не применяется, инструменты считаются по всем нодам.'''
        overall = 0.5
        result = compute_tool_metrics(
            sampler=sample_sampler,
            scorer=sample_scorer,
            data_type='test',
            exec_env='default',
            metric_column='estimation',
            metric_agg='single_mean',
            overall_metrics={'default': overall},
        )

        assert isinstance(result, pd.DataFrame)
        # aef_kind есть — фильтруется по 'tool', router во всех сессиях удаляется
        assert len(result) > 0
        # router удалён, остались scenario_node и buttons_node
        assert 'buttons_node' in result['span_name'].values


class TestCalcMetricFn:
    '''Тесты для кастомного calc_metric_fn (без Scorer).'''

    def test_custom_calc_metric_fn_used(self,
                                        sample_sampler: ToolTracesSampler) -> None:
        '''При переданном calc_metric_fn используется именно он, а не calc_metric_from_data.'''
        custom_called_with: tp.List[pd.DataFrame] = []

        def custom_fn(data: pd.DataFrame, metric_agg: str) -> tp.Dict[str, float]:
            custom_called_with.append(data.copy())
            return {'custom_metric': 42.0}

        result = valtest_tool_metric_dynamics(
            sampler=sample_sampler,
            scorer=None,
            data_type='test',
            metric_column='estimation',
            metric_agg='single_mean',
            greater_is_better=True,
            exec_env='default',
            calc_metric_fn=custom_fn,
        )

        assert len(custom_called_with) > 0
        assert result['precomputed']['result_df']['metric_value'].iloc[0] == 42.0


class TestValtestToolMetricDynamics:
    '''Тесты для valtest_tool_metric_dynamics.'''

    def test_return_structure(self,
                              sample_sampler: ToolTracesSampler,
                              sample_scorer: UniversalScorer) -> None:
        '''Проверка структуры возврата.'''
        result = valtest_tool_metric_dynamics(
            sampler=sample_sampler,
            scorer=sample_scorer,
            data_type='test',
            metric_column='estimation',
            metric_agg='single_mean',
            greater_is_better=True,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {'report', 'precomputed'}

        report = result['report']
        assert isinstance(report, dict)
        assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}

    def test_gray_semaphore(self,
                            sample_sampler: ToolTracesSampler,
                            sample_scorer: UniversalScorer) -> None:
        '''Светофор всегда gray.'''
        result = valtest_tool_metric_dynamics(
            sampler=sample_sampler,
            scorer=sample_scorer,
            data_type='test',
            metric_column='estimation',
            metric_agg='single_mean',
            greater_is_better=True,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {'report', 'precomputed'}
        assert result['report']['semaphore'] == 'gray'

    def test_result_dataframe_structure(self,
                                        sample_sampler: ToolTracesSampler,
                                        sample_scorer: UniversalScorer) -> None:
        '''Проверка структуры выходной таблицы.'''
        result = valtest_tool_metric_dynamics(
            sampler=sample_sampler,
            scorer=sample_scorer,
            data_type='test',
            metric_column='estimation',
            metric_agg='single_mean',
            greater_is_better=True,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {'report', 'precomputed'}

        df = result['report']['result_dataframes'][0]
        assert isinstance(df, pd.DataFrame)
        expected_columns = {
            'Нода/инструмент',
            'Метрика',
            'Абсолютное (относительное) изменение',
        }
        assert set(df.columns) == expected_columns

    def test_precomputed(self,
                         sample_sampler: ToolTracesSampler,
                         sample_scorer: UniversalScorer) -> None:
        '''Если передан precomputed, вычисления не производятся.'''
        precomputed = pd.DataFrame(
            {
                'Нода/инструмент': ['test_tool'],
                'Метрика': ['single_mean'],
                'Значение метрики': [0.5],
                'Абсолютное Изменение': [0.0],
                'Относительное изменение': ['0.0%'],
            }
        )
        result = valtest_tool_metric_dynamics(
            sampler=sample_sampler,
            scorer=sample_scorer,
            data_type='test',
            metric_column='estimation',
            metric_agg='single_mean',
            greater_is_better=True,
            precomputed_report=precomputed,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {'report', 'precomputed'}
        df = result['report']['result_dataframes'][0]
        assert len(df) == 1
        assert df['Нода/инструмент'].iloc[0] == 'test_tool'

    def test_result_plots_empty(self,
                                sample_sampler: ToolTracesSampler,
                                sample_scorer: UniversalScorer) -> None:
        '''Графики отсутствуют.'''
        result = valtest_tool_metric_dynamics(
            sampler=sample_sampler,
            scorer=sample_scorer,
            data_type='test',
            metric_column='estimation',
            metric_agg='single_mean',
            greater_is_better=True,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {'report', 'precomputed'}
        assert result['report']['result_plots'] == []

    def test_precomputed_key(self,
                             sample_sampler: ToolTracesSampler,
                             sample_scorer: UniversalScorer) -> None:
        '''Precomputed содержит result_df.'''
        result = valtest_tool_metric_dynamics(
            sampler=sample_sampler,
            scorer=sample_scorer,
            data_type='test',
            metric_column='estimation',
            metric_agg='single_mean',
            greater_is_better=True,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {'report', 'precomputed'}
        assert 'result_df' in result['precomputed']


class TestComputeToolMetricsGreaterIsBetter:
    '''Тесты для greater_is_better в compute_tool_metrics.'''

    def test_sign_inversion(self,
                            sample_sampler: ToolTracesSampler,
                            sample_scorer: UniversalScorer) -> None:
        '''abs_change при greater_is_better=False инвертирован.'''
        overall = {'default': 0.6}
        result_with_true = compute_tool_metrics(
            sampler=sample_sampler,
            scorer=sample_scorer,
            data_type='test',
            exec_env='default',
            metric_column='estimation',
            metric_agg='single_mean',
            overall_metrics=overall,
            greater_is_better=True,
        )

        result_with_false = compute_tool_metrics(
            sampler=sample_sampler,
            scorer=sample_scorer,
            data_type='test',
            exec_env='default',
            metric_column='estimation',
            metric_agg='single_mean',
            overall_metrics=overall,
            greater_is_better=False,
        )

        assert isinstance(result_with_true, pd.DataFrame)
        assert isinstance(result_with_false, pd.DataFrame)
        pd.testing.assert_series_equal(
            result_with_false['abs_change'],
            -result_with_true['abs_change'],
            check_names=False,
        )


class TestFormatResultTable:
    '''Тесты для format_result_table.'''

    def test_columns_and_types(self) -> None:
        '''Проверка названий колонок, формата объединённого столбца и сортировки.'''
        tool_metrics = pd.DataFrame(
            {
                'span_name': ['tool_a', 'tool_a', 'tool_b', 'tool_b'],
                'metric_name': ['single_mean_estimation', 'multicol_mean', 'single_mean_estimation', 'multicol_mean'],
                'metric_value': [0.5, 0.6, 0.7, 0.8],
                'abs_change': [-0.1, 0.0, 0.1, 0.2],
                'rel_change': [-0.1667, 0.0, 0.1667, 0.3333],
            }
        )
        result = format_result_table(tool_metrics, greater_is_better=True)

        expected_columns = {
            'Нода/инструмент',
            'Метрика',
            'Абсолютное (относительное) изменение',
        }
        assert set(result.columns) == expected_columns

        change_col = 'Абсолютное (относительное) изменение'
        row_a_single = result[(result['Нода/инструмент'] == 'tool_a') & (result['Метрика'] == 0.5)]
        assert row_a_single[change_col].iloc[0] == '-0.1 (-16.7%)'
        row_b_single = result[(result['Нода/инструмент'] == 'tool_b') & (result['Метрика'] == 0.7)]
        assert row_b_single[change_col].iloc[0] == '0.1 (+16.7%)'

        sorted_span_names = result['Нода/инструмент'].tolist()
        expected_order = ['tool_a', 'tool_a', 'tool_b', 'tool_b']
        assert sorted_span_names == expected_order

    def test_empty_input(self) -> None:
        '''Пустой вход — пустой DataFrame с правильными колонками.'''
        empty = pd.DataFrame(columns=['span_name', 'metric_name', 'metric_value', 'abs_change', 'rel_change'])
        result = format_result_table(empty)
        assert len(result) == 0
        assert set(result.columns) == {
            'Нода/инструмент',
            'Метрика',
            'Абсолютное (относительное) изменение',
        }


class TestToolTracesSamplerInitialization:
    '''Тесты инициализации ToolTracesSampler с разными данными.'''

    def test_with_aef_kind(self,
                           sample_sampler: ToolTracesSampler,
                           sample_scorer: UniversalScorer) -> None:
        '''Сэмплер с aef_kind: фильтр работает, метрики считаются только по tool.'''
        result = valtest_tool_metric_dynamics(
            sampler=sample_sampler,
            scorer=sample_scorer,
            data_type='test',
            metric_column='estimation',
            metric_agg='single_mean',
            greater_is_better=True,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {'report', 'precomputed'}
        df = result['report']['result_dataframes'][0]
        tool_names = set(df['Нода/инструмент'])
        assert 'GigaChat' not in tool_names, 'GigaChat — llm, не должен войти'
        assert 'router' not in tool_names, 'router во всех сессиях — должен быть удалён'
        assert 'scenario_node' in tool_names
        assert 'buttons_node' in tool_names

    def test_without_aef_kind(self,
                              sampler_without_aef_kind: ToolTracesSampler,
                              sample_scorer: UniversalScorer) -> None:
        '''Сэмплер без aef_kind: все инструменты в 100% сессий — все удаляются.'''
        result = valtest_tool_metric_dynamics(
            sampler=sampler_without_aef_kind,
            scorer=sample_scorer,
            data_type='test',
            metric_column='estimation',
            metric_agg='single_mean',
            greater_is_better=True,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {'report', 'precomputed'}
        df = result['report']['result_dataframes'][0]
        tool_names = set(df['Нода/инструмент'])
        # router и GigaChat — в 100% сессий (2 из 2), поэтому удаляются
        assert len(tool_names) == 0, 'Все инструменты в 100% сессий — должны быть удалены'

    def test_y_index_is_session_id(self,
                                   sample_sampler_y: pd.DataFrame) -> None:
        '''В y индекс — session_id (по методике).'''
        assert sample_sampler_y.index.name == 'session_id'
        assert set(sample_sampler_y.index) == {'sess1', 'sess2', 'sess3'}


class TestSessionBasedUniqueness:
    '''Тесты уникальности по session_id (а не по trace_id).'''

    @pytest.fixture
    def multi_trace_sampler(self) -> ToolTracesSampler:
        '''Сэмплер: в sess1 два trace_id, в sess2 — один, tool_A встретился в обеих.

        В y попадает по одному значению estimation на сессию,
        поэтому estimation должен быть одинаков у всех строк сессии
        в basket_columns (session_id, aef_kind, estimation).
        '''
        traces = pd.DataFrame(
            {
                'traceid': ['t1', 't2', 't3', 't4'],
                'spanid': ['s1', 's2', 's3', 's4'],
                'name': ['tool_A', 'router', 'tool_A', 'tool_B'],
                'session_id': ['sess1', 'sess1', 'sess2', 'sess2'],
                'aef_kind': ['tool', 'tool', 'tool', 'tool'],
                'starttimeunixnano': [0, 100, 200, 300],
                'endtimeunixnano': [1000, 1100, 1200, 1300],
                'estimation': [0.5, 0.5, 0.7, 0.7],
            }
        )
        return ToolTracesSampler(test=traces)

    def test_tool_in_all_sessions_removed_by_session_id(self,
                                                       multi_trace_sampler: ToolTracesSampler) -> None:
        '''tool_A есть во всех сессиях — удаляется по session_id.

        Даже если в sess1 он встретился не во всех trace_id.
        '''
        result = compute_tool_metrics(
            sampler=multi_trace_sampler,
            scorer=None,
            data_type='test',
            exec_env='default',
            metric_column='estimation',
            metric_agg='single_mean',
            overall_metrics={'single_mean_estimation': 1.0},
            calc_metric_fn=calc_metric_from_data,
        )

        assert 'tool_A' not in result['span_name'].values
        # tool_B только в sess2, router только в sess1 — остаются
        assert set(result['span_name'].unique()) == {'router', 'tool_B'}

    def test_tool_session_id_filter_takes_full_session_value(self,
                                                            multi_trace_sampler: ToolTracesSampler) -> None:
        '''Router встретился только в sess1 (с estimation=0.5).

        Несмотря на то что в sess1 два trace_id,
        метрика берётся по единственному y-значению сессии.
        '''
        result = compute_tool_metrics(
            sampler=multi_trace_sampler,
            scorer=None,
            data_type='test',
            exec_env='default',
            metric_column='estimation',
            metric_agg='single_mean',
            overall_metrics={'single_mean_estimation': 1.0},
            calc_metric_fn=calc_metric_from_data,
        )

        router_row = result[result['span_name'] == 'router']
        assert len(router_row) == 1
        assert abs(router_row['metric_value'].iloc[0] - 0.5) < 1e-9

    def test_y_index_matches_session_ids(self,
                                        multi_trace_sampler: ToolTracesSampler) -> None:
        '''После _preproc в y ровно по одному значению на сессию, индекс — session_id.'''
        y = multi_trace_sampler.test['y']
        assert y.index.name == 'session_id'
        assert list(y.index) == ['sess1', 'sess2']
        assert list(y['estimation']) == [0.5, 0.7]
