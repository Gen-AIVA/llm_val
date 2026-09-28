from copy import deepcopy
from unittest.mock import MagicMock

import pandas as pd
import pytest

from llm_val.val_test.model_quality.valtest_metric_by_thema_agent import (
    calculate_metric_by_thema_agent,
    calculate_p_value_by_thema_agent,
    valtest_metric_by_thema_agent,
)


@pytest.fixture()
def mock_scorer():
    scorer = MagicMock()

    def calc(sampler, data_type, metric_name):
        data = getattr(sampler, data_type)['y']
        return {
            'single_mean_score': data['score'].mean(),
            'single_mean_secondary': data.get('secondary', data['score']).mean(),
        }

    scorer.calc.side_effect = calc
    return scorer


@pytest.fixture()
def above_threshold_p_value():
    return {'A': {'single_mean_score': {'greater': 0.001, 'less': 0.999}}}


class TestCalculateMetricByThemaAgent:
    def test_returns_main_metric_per_scenario(self, mock_scorer):
        X = pd.DataFrame({'scenario': ['A', 'A', 'B']}, index=[10, 11, 20])
        y = pd.DataFrame({'score': [0.1, 0.2, 0.3]}, index=[10, 11, 20])

        result = calculate_metric_by_thema_agent(X=X,
                                                 y=y,
                                                 scorer=mock_scorer,
                                                 main_metric='single_mean_score')

        assert isinstance(result, dict)
        assert set(result.keys()) == {'A', 'B'}
        assert result['A'] == pytest.approx(0.15)
        assert result['B'] == pytest.approx(0.3)
        assert mock_scorer.calc.call_count == 2

    def test_handles_none_scenario(self, mock_scorer):
        X = pd.DataFrame({'scenario': [None, 'A', 'A']}, index=[10, 11, 12])
        y = pd.DataFrame({'score': [0.1, 0.2, 0.3]}, index=[10, 11, 12])

        result = calculate_metric_by_thema_agent(X=X,
                                                 y=y,
                                                 scorer=mock_scorer,
                                                 main_metric='single_mean_score')

        assert None in result
        assert 'A' in result

    def test_missing_main_metric_raises_error(self):
        scorer = MagicMock()
        scorer.calc.return_value = {'single_mean_other': 0.7}
        X = pd.DataFrame({'scenario': ['A']})
        y = pd.DataFrame({'score': [0.7]})

        with pytest.raises(ValueError, match='Ключевая метрика'):
            calculate_metric_by_thema_agent(X=X,
                                            y=y,
                                            scorer=scorer,
                                            main_metric='single_mean_score')


class TestCalculatePValueByThemaAgent:
    def test_returns_per_scenario_p_values_from_main_column(self):
        X = pd.DataFrame({'scenario': ['A'] * 20 + ['B'] * 20})
        y = pd.DataFrame({
            'score': [0.86, 0.88, 0.9, 0.92, 0.94] * 4
            + [0.06, 0.08, 0.1, 0.12, 0.14] * 4,
            'secondary': [0.1] * 40,
        })

        result = calculate_p_value_by_thema_agent(X=X,
                                                  y=y,
                                                  business_metric_threshold=0.5,
                                                  main_metric='single_mean_score',
                                                  main_column='score')

        assert set(result.keys()) == {'A', 'B'}
        assert result['A']['single_mean_score']['greater'] < 0.05
        assert result['B']['single_mean_score']['less'] < 0.05

    def test_uses_main_metric_as_default_main_column(self):
        X = pd.DataFrame({'scenario': ['A'] * 20})
        y = pd.DataFrame({'single_mean_score': [0.86, 0.88, 0.9, 0.92, 0.94] * 4})

        result = calculate_p_value_by_thema_agent(X=X,
                                                  y=y,
                                                  business_metric_threshold=0.5,
                                                  main_metric='single_mean_score')

        assert result['A']['single_mean_score']['greater'] < 0.05

    def test_uses_metric_agg_prefix_to_resolve_main_column(self):
        X = pd.DataFrame({'scenario': ['A'] * 20})
        y = pd.DataFrame({'score': [0.86, 0.88, 0.9, 0.92, 0.94] * 4})

        result = calculate_p_value_by_thema_agent(X=X,
                                                  y=y,
                                                  business_metric_threshold=0.5,
                                                  main_metric='single_mean_score',
                                                  metric_agg='single_mean')

        assert result['A']['single_mean_score']['greater'] < 0.05

    def test_resolves_notebook_estimation_column_without_main_column(self):
        X = pd.DataFrame({'scenario': ['A'] * 20})
        y = pd.DataFrame({'estimation': [0.86, 0.88, 0.9, 0.92, 0.94] * 4})

        result = calculate_p_value_by_thema_agent(X=X,
                                                  y=y,
                                                  business_metric_threshold=0.5,
                                                  main_metric='single_mean_estimation',
                                                  metric_agg='single_mean')

        assert result['A']['single_mean_estimation']['greater'] < 0.05

    def test_uses_single_y_column_as_fallback(self):
        X = pd.DataFrame({'scenario': ['A'] * 20})
        y = pd.DataFrame({'custom_score': [0.86, 0.88, 0.9, 0.92, 0.94] * 4})

        result = calculate_p_value_by_thema_agent(X=X,
                                                  y=y,
                                                  business_metric_threshold=0.5,
                                                  main_metric='quality_metric',
                                                  metric_agg='custom')

        assert result['A']['quality_metric']['greater'] < 0.05

    def test_ambiguous_main_column_raises_error(self):
        X = pd.DataFrame({'scenario': ['A']})
        y = pd.DataFrame({'first': [0.9], 'second': [0.8]})

        with pytest.raises(ValueError, match='Не удалось определить колонку'):
            calculate_p_value_by_thema_agent(X=X,
                                             y=y,
                                             business_metric_threshold=0.5,
                                             main_metric='single_mean_score',
                                             metric_agg='single_mean')

    def test_bootstrap_p_values_are_reproducible(self):
        X = pd.DataFrame({'scenario': ['A'] * 20})
        y = pd.DataFrame({'score': [0.86, 0.88, 0.9, 0.92, 0.94] * 4})
        kwargs = dict(X=X,
                      y=y,
                      business_metric_threshold=0.5,
                      main_metric='single_mean_score',
                      main_column='score',
                      bootstrap=True,
                      bootstrap_iterations=500,
                      random_state=123)

        first = calculate_p_value_by_thema_agent(**kwargs)
        second = calculate_p_value_by_thema_agent(**kwargs)

        assert first == second
        assert first['A']['single_mean_score']['greater'] < 0.05

    def test_bootstrap_uses_custom_metric_fn(self):
        X = pd.DataFrame({'scenario': ['A'] * 20})
        y = pd.DataFrame({'score': [0.1] * 19 + [1.0]})

        mean_result = calculate_p_value_by_thema_agent(X=X,
                                                       y=y,
                                                       business_metric_threshold=0.5,
                                                       main_metric='single_mean_score',
                                                       main_column='score',
                                                       bootstrap=True,
                                                       bootstrap_iterations=500,
                                                       random_state=42)
        max_result = calculate_p_value_by_thema_agent(X=X,
                                                      y=y,
                                                      business_metric_threshold=0.5,
                                                      main_metric='single_mean_score',
                                                      main_column='score',
                                                      bootstrap=True,
                                                      bootstrap_iterations=500,
                                                      random_state=42,
                                                      metric_fn=max)

        assert mean_result['A']['single_mean_score']['greater'] > 0.05
        assert max_result['A']['single_mean_score']['greater'] < 0.05

    def test_metric_fn_without_bootstrap_is_ignored(self):
        X = pd.DataFrame({'scenario': ['A'] * 20})
        y = pd.DataFrame({'score': [0.86, 0.88, 0.9, 0.92, 0.94] * 4})

        result = calculate_p_value_by_thema_agent(X=X,
                                                  y=y,
                                                  business_metric_threshold=0.5,
                                                  main_metric='single_mean_score',
                                                  main_column='score',
                                                  metric_fn=max)

        assert result['A']['single_mean_score']['greater'] < 0.05


class TestValtestMetricByThemaAgent:
    def test_output_format_with_precomputed(self, above_threshold_p_value):
        metric_value_by_scenario = {'A': 0.9}

        test_res = valtest_metric_by_thema_agent(sampler=MagicMock(),
                                                 scorer=MagicMock(),
                                                 main_metric='single_mean_score',
                                                 business_metric_threshold=0.5,
                                                 metric_value_by_scenario=metric_value_by_scenario,
                                                 p_value_by_scenario=above_threshold_p_value)

        assert isinstance(test_res, dict)
        assert set(test_res.keys()) == {'report', 'precomputed'}
        assert isinstance(test_res['report'], dict)
        assert isinstance(test_res['precomputed'], dict)

    def test_output_format_with_sampler(self, mock_scorer):
        sampler = MagicMock()
        sampler.test = {
            'X': pd.DataFrame({'scenario': ['A'] * 4 + ['B'] * 4}),
            'y': pd.DataFrame({'score': [0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4]}),
        }

        test_res = valtest_metric_by_thema_agent(sampler=sampler,
                                                 scorer=mock_scorer,
                                                 main_metric='single_mean_score',
                                                 main_column='score',
                                                 business_metric_threshold=0.5)

        assert isinstance(test_res, dict)
        assert set(test_res.keys()) == {'report', 'precomputed'}
        assert 'metric_value_by_scenario' in test_res['precomputed']
        assert 'p_value_by_scenario' in test_res['precomputed']

    def test_precomputed_passthrough_does_not_call_scorer(self, mock_scorer, above_threshold_p_value):
        metric_value_by_scenario = {'A': 0.9}

        valtest_metric_by_thema_agent(sampler=MagicMock(),
                                      scorer=mock_scorer,
                                      main_metric='single_mean_score',
                                      business_metric_threshold=0.5,
                                      metric_value_by_scenario=metric_value_by_scenario,
                                      p_value_by_scenario=above_threshold_p_value)

        mock_scorer.calc.assert_not_called()

    def test_does_not_mutate_precomputed(self, above_threshold_p_value):
        metric_value_by_scenario = {'A': 0.9}
        original_metric_value = deepcopy(metric_value_by_scenario)
        original_p_value = deepcopy(above_threshold_p_value)

        valtest_metric_by_thema_agent(sampler=MagicMock(),
                                      scorer=MagicMock(),
                                      main_metric='single_mean_score',
                                      business_metric_threshold=0.5,
                                      metric_value_by_scenario=metric_value_by_scenario,
                                      p_value_by_scenario=above_threshold_p_value)

        assert metric_value_by_scenario == original_metric_value
        assert above_threshold_p_value == original_p_value

    def test_is_info_gray(self, above_threshold_p_value):
        test_res = valtest_metric_by_thema_agent(sampler=MagicMock(),
                                                 scorer=MagicMock(),
                                                 main_metric='single_mean_score',
                                                 business_metric_threshold=0.5,
                                                 is_info=True,
                                                 metric_value_by_scenario={'A': 0.9},
                                                 p_value_by_scenario=above_threshold_p_value)

        assert test_res['report']['semaphore'] == 'gray'

    def test_auto_is_info_none_scenario(self, mock_scorer):
        sampler = MagicMock()
        sampler.test = {
            'X': pd.DataFrame({'scenario': [None, None]}),
            'y': pd.DataFrame({'score': [0.1, 0.2]}),
        }

        test_res = valtest_metric_by_thema_agent(sampler=sampler,
                                                 scorer=mock_scorer,
                                                 main_metric='single_mean_score',
                                                 main_column='score',
                                                 business_metric_threshold=0.5)

        assert test_res['report']['semaphore'] == 'gray'

    def test_other_scenarios_do_not_affect_final_semaphore(self, mock_scorer):
        sampler = MagicMock()
        main_values = [0.86, 0.88, 0.9, 0.92, 0.94] * 4
        other_values = [0.06, 0.08, 0.1, 0.12, 0.14] * 4
        sampler.test = {
            'X': pd.DataFrame({'scenario': ['A'] * 20 + ['другое'] * 20}),
            'y': pd.DataFrame({'score': main_values + other_values}),
        }

        test_res = valtest_metric_by_thema_agent(sampler=sampler,
                                                 scorer=mock_scorer,
                                                 main_metric='single_mean_score',
                                                 main_column='score',
                                                 business_metric_threshold=0.5)

        main_df = test_res['report']['result_dataframes'][0]
        assert test_res['report']['semaphore'] == 'green'
        assert main_df.loc[('другое', 'single_mean_score'), 'Результат теста'] == 'red'

    def test_stat_significant_above_threshold_green(self, mock_scorer):
        sampler = MagicMock()
        sampler.test = {
            'X': pd.DataFrame({'scenario': ['A'] * 20}),
            'y': pd.DataFrame({'score': [0.86, 0.88, 0.9, 0.92, 0.94] * 4}),
        }

        test_res = valtest_metric_by_thema_agent(sampler=sampler,
                                                 scorer=mock_scorer,
                                                 main_metric='single_mean_score',
                                                 main_column='score',
                                                 business_metric_threshold=0.5)

        assert test_res['report']['semaphore'] == 'green'

    def test_stat_significant_below_threshold_red(self, mock_scorer):
        sampler = MagicMock()
        sampler.test = {
            'X': pd.DataFrame({'scenario': ['A'] * 20}),
            'y': pd.DataFrame({'score': [0.06, 0.08, 0.1, 0.12, 0.14] * 4}),
        }

        test_res = valtest_metric_by_thema_agent(sampler=sampler,
                                                 scorer=mock_scorer,
                                                 main_metric='single_mean_score',
                                                 main_column='score',
                                                 business_metric_threshold=0.5)

        assert test_res['report']['semaphore'] == 'red'

    def test_not_stat_significant_above_threshold_yellow(self, mock_scorer):
        sampler = MagicMock()
        sampler.test = {
            'X': pd.DataFrame({'scenario': ['A', 'A']}),
            'y': pd.DataFrame({'score': [0.4, 0.6]}),
        }

        test_res = valtest_metric_by_thema_agent(sampler=sampler,
                                                 scorer=mock_scorer,
                                                 main_metric='single_mean_score',
                                                 main_column='score',
                                                 business_metric_threshold=0.5)

        assert test_res['report']['semaphore'] == 'yellow'

    def test_secondary_metric_is_not_in_main_report(self, mock_scorer):
        sampler = MagicMock()
        sampler.test = {
            'X': pd.DataFrame({'scenario': ['A'] * 20}),
            'y': pd.DataFrame({
                'score': [0.86, 0.88, 0.9, 0.92, 0.94] * 4,
                'secondary': [0.06, 0.08, 0.1, 0.12, 0.14] * 4,
            }),
        }

        test_res = valtest_metric_by_thema_agent(sampler=sampler,
                                                 scorer=mock_scorer,
                                                 main_metric='single_mean_score',
                                                 main_column='score',
                                                 business_metric_threshold=0.5)
        main_df = test_res['report']['result_dataframes'][0]

        assert ('A', 'single_mean_secondary') not in main_df.index
        assert ('A', 'single_mean_score') in main_df.index

    def test_report_table_has_methodology_columns(self, above_threshold_p_value):
        test_res = valtest_metric_by_thema_agent(sampler=MagicMock(),
                                                 scorer=MagicMock(),
                                                 main_metric='single_mean_score',
                                                 business_metric_threshold=0.5,
                                                 metric_value_by_scenario={'A': 0.9},
                                                 p_value_by_scenario=above_threshold_p_value)
        report_df = test_res['report']['result_dataframes'][0]

        assert list(report_df.index.names) == ['Тематика', 'Наименование критерия']
        assert list(report_df.columns) == ['Ключевая метрика качества', 'Результат теста']
        assert report_df.index[-1] == ('Общий итог', '')
        assert report_df.iloc[-1]['Ключевая метрика качества'] == ''
        assert report_df.iloc[-1]['Результат теста'] == test_res['report']['semaphore']

    def test_single_observation_stat_test_gray(self, mock_scorer):
        sampler = MagicMock()
        sampler.test = {
            'X': pd.DataFrame({'scenario': ['A']}),
            'y': pd.DataFrame({'score': [0.9]}),
        }

        test_res = valtest_metric_by_thema_agent(sampler=sampler,
                                                 scorer=mock_scorer,
                                                 main_metric='single_mean_score',
                                                 main_column='score',
                                                 business_metric_threshold=0.5)

        assert test_res['report']['semaphore'] == 'gray'

    def test_exec_env_sberds(self, mock_scorer):
        sampler = {
            'test': {
                'X': pd.DataFrame({'scenario': ['A'] * 4 + ['B'] * 4}),
                'y': pd.DataFrame({'score': [0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4]}),
            }
        }

        test_res = valtest_metric_by_thema_agent(sampler=sampler,
                                                 scorer=mock_scorer,
                                                 main_metric='single_mean_score',
                                                 main_column='score',
                                                 business_metric_threshold=0.5,
                                                 exec_env='sberds')

        assert isinstance(test_res, dict)
        assert set(test_res.keys()) == {'report', 'precomputed'}
