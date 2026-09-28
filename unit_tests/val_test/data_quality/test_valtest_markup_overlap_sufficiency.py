import pytest
import pandas as pd

from unittest.mock import MagicMock

from llm_val.val_test.data_quality.valtest_markup_overlap_sufficiency import (
    report_valtest_markup_overlap_sufficiency,
    valtest_markup_overlap_sufficiency,
    get_overlap_statistics,
    resolve_threshold
)


class TestGetOverlapStatistics:

    def test_equal_overlap(self):
        x_data = pd.DataFrame({
            'assessor_id': [1, 2, 3, 1, 2, 3],
            'session_id': [1, 1, 1, 2, 2, 2],
            'quality_metric': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        })
        result = get_overlap_statistics(x_data, ['quality_metric'])
        assert result == {'quality_metric': 3}

    def test_unequal_overlap(self):
        x_data = pd.DataFrame({
            'assessor_id': [1, 2, 3, 1, 2],
            'session_id': [1, 1, 1, 2, 2],
            'quality_metric': [0.5, 0.6, 0.7, 0.8, 0.9]
        })
        result = get_overlap_statistics(x_data, ['quality_metric'])
        assert result == {'quality_metric': 2}

    def test_multiple_metrics(self):
        x_data = pd.DataFrame({
            'assessor_id': [1, 2, 1, 2],
            'session_id': [1, 1, 2, 2],
            'metric_a_metric': [0.5, 0.6, 0.7, 0.8],
            'metric_b_metric': [0.1, 0.2, 0.3, 0.4]
        })
        result = get_overlap_statistics(
            x_data, ['metric_a_metric', 'metric_b_metric'])
        assert result == {'metric_a_metric': 2, 'metric_b_metric': 2}

    def test_nan_in_metric_reduces_overlap(self):
        x_data = pd.DataFrame({
            'assessor_id': [1, 2, 3, 1, 2, 3],
            'session_id': [1, 1, 1, 2, 2, 2],
            'quality_metric': [0.5, 0.6, None, 0.8, 0.9, 1.0]
        })
        result = get_overlap_statistics(x_data, ['quality_metric'])
        assert result == {'quality_metric': 2}


class TestResolveThreshold:

    def test_auto_from_data_binary(self):
        x_data = pd.DataFrame({
            'quality_metric': [0, 1, 0, 1, 1, 0]
        })
        result = resolve_threshold(None, x_data, ['quality_metric'])
        assert result == {'quality_metric': 3}

    def test_auto_from_data_ordinal(self):
        x_data = pd.DataFrame({
            'quality_metric': [1, 2, 3, 4, 5]
        })
        result = resolve_threshold(None, x_data, ['quality_metric'])
        assert result == {'quality_metric': 6}

    def test_auto_ignores_nan(self):
        x_data = pd.DataFrame({
            'quality_metric': [0, 1, None, 0, 1]
        })
        result = resolve_threshold(None, x_data, ['quality_metric'])
        assert result == {'quality_metric': 3}

    def test_n_categories_int(self):
        x_data = pd.DataFrame({'quality_metric': [0, 1]})
        result = resolve_threshold(5, x_data, ['quality_metric'])
        assert result == {'quality_metric': 6}

    def test_n_categories_dict(self):
        x_data = pd.DataFrame({
            'metric_a_metric': [0, 1],
            'metric_b_metric': [0, 1]
        })
        n_cat = {'metric_a_metric': 3, 'metric_b_metric': 5}
        result = resolve_threshold(
            n_cat, x_data, ['metric_a_metric', 'metric_b_metric'])
        assert result == {'metric_a_metric': 4, 'metric_b_metric': 6}


class TestReportValtestOverlap:

    def test_green_semaphore(self):
        stats = {'quality_metric': 3}
        result = report_valtest_markup_overlap_sufficiency(
            stats, green_semaphore_threshold=3)
        assert result['semaphore'] == 'green'
        assert isinstance(result['result_dataframes'], list)
        assert len(result['result_dataframes']) == 1

    def test_red_semaphore(self):
        stats = {'quality_metric': 2}
        result = report_valtest_markup_overlap_sufficiency(
            stats, green_semaphore_threshold=3)
        assert result['semaphore'] == 'red'

    def test_justified_becomes_yellow(self):
        stats = {'quality_metric': 2}
        result = report_valtest_markup_overlap_sufficiency(
            stats, green_semaphore_threshold=3, is_justified=True)
        assert result['semaphore'] == 'yellow'

    def test_info_gray(self):
        stats = {'quality_metric': 2}
        result = report_valtest_markup_overlap_sufficiency(
            stats, green_semaphore_threshold=3, is_info=True)
        assert result['semaphore'] == 'gray'

    def test_result_structure(self):
        stats = {'quality_metric': 5}
        result = report_valtest_markup_overlap_sufficiency(stats)
        assert 'semaphore' in result
        assert 'result_plots' in result
        assert 'result_dataframes' in result
        assert isinstance(result['result_plots'], list)


class TestValtestOverlap:

    def _make_x_data(self):
        return pd.DataFrame({
            'assessor_id': [1, 2, 3, 1, 2, 3],
            'session_id': [1, 1, 1, 2, 2, 2],
            'quality_metric': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        })

    def test_output_format_default(self):
        sampler = MagicMock()
        sampler.test = {'X': self._make_x_data()}
        result = valtest_markup_overlap_sufficiency(sampler, exec_env='default')
        assert isinstance(result, dict)
        assert set(result.keys()) == {'report', 'precomputed'}
        assert isinstance(result['report'], dict)
        assert isinstance(result['precomputed'], dict)

    def test_output_format_sberds(self):
        x_dict = self._make_x_data().to_dict()
        sampler = {'test': {'X': x_dict}}
        result = valtest_markup_overlap_sufficiency(sampler, exec_env='sberds')
        assert isinstance(result, dict)
        assert set(result.keys()) == {'report', 'precomputed'}
        assert isinstance(result['report'], dict)
        assert isinstance(result['precomputed'], dict)

    def test_precomputed_skips_calculation(self):
        sampler = MagicMock()
        sampler.test = {'X': self._make_x_data()}
        precomputed_stats = {'quality_metric': 5}
        result = valtest_markup_overlap_sufficiency(
            sampler, overlap_statistics=precomputed_stats)
        assert result['precomputed']['overlap_statistics'] == precomputed_stats

    def test_auto_detect_metric_columns(self):
        sampler = MagicMock()
        sampler.test = {'X': self._make_x_data()}
        result = valtest_markup_overlap_sufficiency(sampler)
        assert 'quality_metric' in result['precomputed']['overlap_statistics']

    def test_explicit_metric_columns(self):
        sampler = MagicMock()
        sampler.test = {'X': self._make_x_data()}
        result = valtest_markup_overlap_sufficiency(
            sampler, metric_columns=['quality_metric'])
        assert 'quality_metric' in result['precomputed']['overlap_statistics']

    def test_n_plus_1_threshold_green(self):
        x_data = pd.DataFrame({
            'assessor_id': [1, 2, 3, 1, 2, 3],
            'session_id': [1, 1, 1, 2, 2, 2],
            'quality_metric': [0, 1, 0, 1, 0, 1]
        })
        sampler = MagicMock()
        sampler.test = {'X': x_data}
        result = valtest_markup_overlap_sufficiency(sampler)
        assert result['report']['semaphore'] == 'green'

    def test_n_plus_1_threshold_red(self):
        x_data = pd.DataFrame({
            'assessor_id': [1, 2, 1, 2],
            'session_id': [1, 1, 2, 2],
            'quality_metric': [0, 1, 0, 1]
        })
        sampler = MagicMock()
        sampler.test = {'X': x_data}
        result = valtest_markup_overlap_sufficiency(sampler)
        assert result['report']['semaphore'] == 'red'

    def test_n_plus_1_threshold_justified_yellow(self):
        x_data = pd.DataFrame({
            'assessor_id': [1, 2, 1, 2],
            'session_id': [1, 1, 2, 2],
            'quality_metric': [0, 1, 0, 1]
        })
        sampler = MagicMock()
        sampler.test = {'X': x_data}
        result = valtest_markup_overlap_sufficiency(sampler, is_justified=True)
        assert result['report']['semaphore'] == 'yellow'

    def test_n_categories_overrides_auto(self):
        x_data = pd.DataFrame({
            'assessor_id': [1, 2, 3, 1, 2, 3],
            'session_id': [1, 1, 1, 2, 2, 2],
            'quality_metric': [0, 1, 0, 1, 0, 1]
        })
        sampler = MagicMock()
        sampler.test = {'X': x_data}
        result = valtest_markup_overlap_sufficiency(sampler, n_categories=5)
        assert result['report']['semaphore'] == 'red'
