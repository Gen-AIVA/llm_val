import pytest
import numpy as np
import pandas as pd

from llm_val.val_test.model_quality.valtest_metric_agent import (
    valtest_metric_agent,
    report_valtest_metric_agent,
    calc_metric
)


@pytest.fixture()
def sample_metric_value():
    return {'multicol_mean': 0.7, 'single_mean_col1': 0.8, 'single_mean_col2': 0.6}


@pytest.fixture()
def sberds_sampler():
    return {
        'test': {
            'y': {
                'col1': {0: 0.8, 1: 0.6, 2: 0.9},
                'col2': {0: 0.5, 1: 0.7, 2: 0.4}
            }
        }
    }


class TestCalcMetric:

    def test_single_mean(self):
        data = pd.DataFrame({'col1': [0.8, 0.6, 0.9], 'col2': [0.5, 0.7, 0.4]})
        result = calc_metric(data, metric_name='single_mean')
        assert 'single_mean_col1' in result
        assert 'single_mean_col2' in result
        assert np.isclose(result['single_mean_col1'], np.mean([0.8, 0.6, 0.9]))
        assert np.isclose(result['single_mean_col2'], np.mean([0.5, 0.7, 0.4]))

    def test_multicol_mean(self):
        data = pd.DataFrame({'col1': [0.8, 0.6, 0.9], 'col2': [0.5, 0.7, 0.4]})
        result = calc_metric(data, metric_name='multicol_mean')
        assert 'multicol_mean' in result
        assert np.isclose(result['multicol_mean'], np.mean([0.8, 0.6, 0.9, 0.5, 0.7, 0.4]))

    def test_all_metrics(self):
        data = pd.DataFrame({'col1': [0.5, 0.5]})
        result = calc_metric(data)
        assert 'single_mean_col1' in result
        assert 'multicol_mean' in result


class TestReportValtestMetricAgent:

    def test_green_semaphore(self, sample_metric_value):
        report = report_valtest_metric_agent(
            metric_value=sample_metric_value,
            main_metric='multicol_mean')
        assert report['semaphore'] == 'green'
        assert isinstance(report['result_plots'], list)
        assert isinstance(report['result_dataframes'], list)

    def test_red_semaphore(self):
        metric_value = {'multicol_mean': 0.1, 'single_mean_col1': 0.05}
        report = report_valtest_metric_agent(
            metric_value=metric_value,
            main_metric='multicol_mean')
        assert report['semaphore'] == 'red'

    def test_info_gray(self, sample_metric_value):
        report = report_valtest_metric_agent(
            metric_value=sample_metric_value,
            main_metric='multicol_mean',
            is_info=True)
        assert report['semaphore'] == 'gray'


class TestValtestMetricAgent:

    def test_output_format_with_precomputed(self, sample_metric_value):
        test_res = valtest_metric_agent(
            sampler=None,
            scorer=None,
            main_metric='multicol_mean',
            metric_value=sample_metric_value)
        assert isinstance(test_res, dict)
        assert set(test_res.keys()) == {'report', 'precomputed'}
        report = test_res['report']
        assert isinstance(report, dict)
        precomputed = test_res['precomputed']
        assert isinstance(precomputed, dict)

    def test_precomputed_contains_metric_value(self, sample_metric_value):
        test_res = valtest_metric_agent(
            sampler=None,
            scorer=None,
            main_metric='multicol_mean',
            metric_value=sample_metric_value)
        assert 'metric_value' in test_res['precomputed']
        assert test_res['precomputed']['metric_value'] == sample_metric_value

    def test_sberds_exec_env(self, sberds_sampler):
        test_res = valtest_metric_agent(
            sampler=sberds_sampler,
            scorer=None,
            main_metric='multicol_mean',
            metric_agg='multicol_mean',
            exec_env='sberds')
        assert isinstance(test_res, dict)
        assert set(test_res.keys()) == {'report', 'precomputed'}
        assert 'metric_value' in test_res['precomputed']
        assert 'multicol_mean' in test_res['precomputed']['metric_value']

    def test_sberds_single_mean(self, sberds_sampler):
        test_res = valtest_metric_agent(
            sampler=sberds_sampler,
            scorer=None,
            main_metric='single_mean_col1',
            metric_agg='single_mean',
            exec_env='sberds')
        precomputed = test_res['precomputed']['metric_value']
        assert 'single_mean_col1' in precomputed
        assert 'single_mean_col2' in precomputed

    def test_info_mode(self, sample_metric_value):
        test_res = valtest_metric_agent(
            sampler=None,
            scorer=None,
            main_metric='multicol_mean',
            is_info=True,
            metric_value=sample_metric_value)
        assert test_res['report']['semaphore'] == 'gray'

    def test_custom_calc_metric_fn(self, sberds_sampler):
        def custom_fn(sampler, data_type, metric_agg):
            return {'custom_metric': 0.42}

        test_res = valtest_metric_agent(
            sampler=sberds_sampler,
            scorer=None,
            main_metric='custom_metric',
            exec_env='sberds',
            calc_metric_fn=custom_fn)
        assert test_res['precomputed']['metric_value'] == {'custom_metric': 0.42}

    def test_precomputed_does_not_mutate_input(self, sample_metric_value):
        original = sample_metric_value.copy()
        valtest_metric_agent(
            sampler=None,
            scorer=None,
            main_metric='multicol_mean',
            metric_value=sample_metric_value)
        assert sample_metric_value == original
