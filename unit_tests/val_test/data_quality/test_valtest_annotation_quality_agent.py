import pytest
import numpy as np
import pandas as pd

from unittest.mock import MagicMock

from llm_val.val_test.data_quality.valtest_annotation_quality_agent import (
    report_valtest_annotation_quality_agent,
    valtest_annotation_quality_agent,
    classify_annotation_type,
    select_agreement_check,
    compute_agreement_stats,
    compute_criterion_value,
    encode_nominal_labels,
    krippendorff_alpha,
    spearman_pair_correlation,
    cohen_kappa_pair,
    fleiss_kappa_calc,
)


def _make_binary_data():
    '''
    Создание тестового датасета: 3 асессора, 4 сессии, бинарная метрика
    '''
    return pd.DataFrame({
        'assessor_id': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        'session_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'quality_metric': [1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1]
    })


def _make_ordinal_data():
    '''
    Создание тестового датасета с порядковой метрикой
    '''
    return pd.DataFrame({
        'assessor_id': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'session_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'score_metric': [3, 4, 3, 5, 5, 4, 2, 2, 3]
    })


def _make_binary_partial_overlap():
    '''
    Бинарная метрика, где перекрытие меньше числа асессоров
    '''
    return pd.DataFrame({
        'assessor_id': [1, 2, 1, 3, 2, 3],
        'session_id': [1, 1, 2, 2, 3, 3],
        'label_metric': [1, 1, 0, 0, 1, 0]
    })


class TestClassifyAnnotationType:

    def test_binary_numeric(self):
        col = pd.Series([0, 1, 0, 1, 0])
        assert classify_annotation_type(col) == 'binary'

    def test_binary_strings(self):
        col = pd.Series(['yes', 'no', 'yes', 'no'])
        assert classify_annotation_type(col) == 'binary'

    def test_ordinal(self):
        col = pd.Series([1, 2, 3, 4, 5])
        assert classify_annotation_type(col) == 'ordinal'

    def test_ordinal_with_nan(self):
        col = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        assert classify_annotation_type(col) == 'ordinal'

    def test_nominal_strings(self):
        col = pd.Series(['a', 'b', 'c', 'a', 'b'])
        assert classify_annotation_type(col) == 'nominal'


class TestSelectAgreementCheck:

    def test_ordinal_to_spearman(self):
        data = _make_ordinal_data()
        check = select_agreement_check(data, 'score_metric', 3)
        assert check == 'Попарный коэф. Спирмена'

    def test_binary_full_overlap_to_fleiss(self):
        data = _make_binary_data()
        check = select_agreement_check(data, 'quality_metric', 3)
        assert check == 'Каппа Флейса'

    def test_binary_partial_overlap_to_cohen(self):
        data = _make_binary_partial_overlap()
        check = select_agreement_check(data, 'label_metric', 3)
        assert check == 'Каппа Коэна'

    def test_nominal_to_krippendorff(self):
        data = pd.DataFrame({
            'assessor_id': [1, 2, 3, 1, 2, 3],
            'session_id': [1, 1, 1, 2, 2, 2],
            'cat_metric': ['a', 'b', 'c', 'a', 'a', 'b']
        })
        check = select_agreement_check(data, 'cat_metric', 3)
        assert check == 'Альфа Криппендорфа'


class TestEncodeNominalLabels:

    def test_string_labels_encoded_to_int(self):
        data = pd.DataFrame({
            'assessor_id': [1, 2],
            'session_id': [1, 1],
            'cat_metric': ['yes', 'no']
        })
        encoded = encode_nominal_labels(data, 'cat_metric')
        assert set(encoded['cat_metric'].dropna().unique()) == {0, 1}

    def test_nan_preserved(self):
        data = pd.DataFrame({
            'assessor_id': [1, 2],
            'session_id': [1, 1],
            'cat_metric': ['yes', np.nan]
        })
        encoded = encode_nominal_labels(data, 'cat_metric')
        assert encoded['cat_metric'].isna().sum() == 1


class TestReportValtestAnnotation:

    def test_all_nan_gray(self):
        stats = [{'criterion': 'a_metric',
                  'check': 'Попарный коэф. Спирмена', 'value': np.nan}]
        result = report_valtest_annotation_quality_agent(stats)
        assert result['semaphore'] == 'gray'

    def test_info_mode_gray(self):
        stats = [{'criterion': 'a_metric',
                  'check': 'Попарный коэф. Спирмена', 'value': 0.8}]
        result = report_valtest_annotation_quality_agent(stats, is_info=True)
        assert result['semaphore'] == 'gray'

    def test_result_structure(self):
        stats = [{'criterion': 'a_metric',
                  'check': 'Альфа Криппендорфа', 'value': 0.3}]
        result = report_valtest_annotation_quality_agent(stats)
        assert set(result.keys()) == {'semaphore', 'result_plots',
                                      'result_dataframes'}
        assert isinstance(result['result_plots'], list)
        assert isinstance(result['result_dataframes'], list)

    def test_report_columns(self):
        stats = [{'criterion': 'a_metric',
                  'check': 'Попарный коэф. Спирмена', 'value': 0.8}]
        result = report_valtest_annotation_quality_agent(stats)
        cols = list(result['result_dataframes'][0].columns)
        assert cols == ['Наименование проверки', 'Согласие', 'Светофор']

    def test_multi_criteria_check_names(self):
        stats = [{'criterion': 'a_metric',
                  'check': 'Попарный коэф. Спирмена', 'value': 0.8},
                 {'criterion': 'b_metric',
                  'check': 'Альфа Криппендорфа', 'value': 0.5}]
        result = report_valtest_annotation_quality_agent(stats)
        names = list(result['result_dataframes'][0]['Наименование проверки'])
        assert names == ['Попарный коэф. Спирмена (a_metric)',
                         'Альфа Криппендорфа (b_metric)']

    def test_green_semaphore(self):
        stats = [{'criterion': 'a_metric',
                  'check': 'Попарный коэф. Спирмена', 'value': 0.8}]
        result = report_valtest_annotation_quality_agent(stats)
        assert result['semaphore'] == 'green'

    def test_red_semaphore(self):
        stats = [{'criterion': 'a_metric',
                  'check': 'Альфа Криппендорфа', 'value': 0.05}]
        result = report_valtest_annotation_quality_agent(stats)
        assert result['semaphore'] == 'red'

    def test_worst_semaphore_across_criteria(self):
        stats = [{'criterion': 'a_metric',
                  'check': 'Попарный коэф. Спирмена', 'value': 0.9},
                 {'criterion': 'b_metric',
                  'check': 'Альфа Криппендорфа', 'value': 0.05}]
        result = report_valtest_annotation_quality_agent(stats)
        assert result['semaphore'] == 'red'

    def test_nan_criterion_ignored_in_final(self):
        stats = [{'criterion': 'a_metric',
                  'check': 'Попарный коэф. Спирмена', 'value': 0.9},
                 {'criterion': 'b_metric',
                  'check': 'Альфа Криппендорфа', 'value': np.nan}]
        result = report_valtest_annotation_quality_agent(stats)
        assert result['semaphore'] == 'green'


class TestKrippendorffAlpha:

    def test_perfect_agreement(self):
        df = pd.DataFrame({
            'assessor_id': [1, 2, 3, 1, 2, 3],
            'session_id': [1, 1, 1, 2, 2, 2],
            'cat_metric': ['x', 'x', 'x', 'y', 'y', 'y']
        })
        result = krippendorff_alpha(df, 'cat_metric')
        assert result == pytest.approx(1.0, abs=0.01)

    def test_no_agreement(self):
        df = pd.DataFrame({
            'assessor_id': [1, 2, 1, 2],
            'session_id': [1, 1, 2, 2],
            'cat_metric': ['a', 'b', 'b', 'a']
        })
        result = krippendorff_alpha(df, 'cat_metric')
        assert result < 0.5

    def test_single_value_returns_nan(self):
        df = pd.DataFrame({
            'assessor_id': [1, 2],
            'session_id': [1, 1],
            'cat_metric': ['a', 'a']
        })
        assert np.isnan(krippendorff_alpha(df, 'cat_metric'))


class TestSpearmanPairCorrelation:

    def test_perfect_correlation(self):
        df = pd.DataFrame({
            'assessor_id': [1, 2, 1, 2, 1, 2],
            'session_id': [1, 1, 2, 2, 3, 3],
            'score_metric': [1, 1, 2, 2, 3, 3]
        })
        result = spearman_pair_correlation(df, 'score_metric', min_periods=1)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_negative_correlation_not_clipped(self):
        df = pd.DataFrame({
            'assessor_id': [1, 2, 1, 2, 1, 2],
            'session_id': [1, 1, 2, 2, 3, 3],
            'score_metric': [1, 3, 2, 2, 3, 1]
        })
        result = spearman_pair_correlation(df, 'score_metric', min_periods=1)
        assert result < 0


class TestCohenKappaPair:

    def test_perfect_agreement(self):
        df = pd.DataFrame({
            'assessor_id': [1, 2, 1, 2, 1, 2],
            'session_id': [1, 1, 2, 2, 3, 3],
            'label_metric': [0, 0, 1, 1, 0, 0]
        })
        result = cohen_kappa_pair(df, 'label_metric')
        assert result == pytest.approx(1.0, abs=0.01)

    def test_string_binary_labels(self):
        df = pd.DataFrame({
            'assessor_id': [1, 2, 1, 2, 1, 2],
            'session_id': [1, 1, 2, 2, 3, 3],
            'label_metric': ['no', 'no', 'yes', 'yes', 'no', 'no']
        })
        result = cohen_kappa_pair(df, 'label_metric')
        assert result == pytest.approx(1.0, abs=0.01)


class TestFleissKappaCalc:

    def test_perfect_agreement(self):
        df = pd.DataFrame({
            'assessor_id': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'session_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'label_metric': [1, 1, 1, 0, 0, 0, 1, 1, 1]
        })
        result = fleiss_kappa_calc(df, 'label_metric')
        assert result == pytest.approx(1.0, abs=0.01)


class TestComputeAgreementStats:

    def test_duplicate_rows_not_averaged(self):
        df = pd.DataFrame({
            'assessor_id': [1, 1, 2, 3, 1, 2, 3],
            'session_id': [1, 1, 1, 1, 2, 2, 2],
            'quality_metric': [1, 1, 1, 0, 1, 0, 1]
        })
        sampler = MagicMock()
        sampler.test = {'X': df}
        stats = compute_agreement_stats(
            sampler, 'test', 1, ['quality_metric'], 'default')
        assert not np.isnan(stats[0]['value'])

    def test_multiple_criteria(self):
        df = _make_binary_data().copy()
        df['score_metric'] = [3, 4, 3, 5, 5, 4, 2, 2, 3, 4, 4, 5]
        sampler = MagicMock()
        sampler.test = {'X': df}
        stats = compute_agreement_stats(
            sampler, 'test', 1, ['quality_metric', 'score_metric'], 'default')
        assert len(stats) == 2
        checks = {stat['criterion']: stat['check'] for stat in stats}
        assert checks['quality_metric'] == 'Каппа Флейса'
        assert checks['score_metric'] == 'Попарный коэф. Спирмена'


class TestValtestAnnotation:

    def test_output_format_default(self):
        sampler = MagicMock()
        sampler.test = {'X': _make_binary_data()}
        result = valtest_annotation_quality_agent(sampler, exec_env='default')
        assert isinstance(result, dict)
        assert set(result.keys()) == {'report', 'precomputed'}
        assert isinstance(result['report'], dict)
        assert isinstance(result['precomputed'], dict)

    def test_output_format_sberds(self):
        data = _make_binary_data().to_dict()
        sampler = {'test': {'X': data}}
        result = valtest_annotation_quality_agent(sampler, exec_env='sberds')
        assert isinstance(result, dict)
        assert set(result.keys()) == {'report', 'precomputed'}
        assert isinstance(result['report'], dict)
        assert isinstance(result['precomputed'], dict)

    def test_precomputed_used_without_sampler(self):
        stats = [{'criterion': 'quality_metric',
                  'check': 'Каппа Флейса', 'value': 0.85}]
        result = valtest_annotation_quality_agent(
            sampler=None, criteria_stats=stats)
        assert result['precomputed']['criteria_stats'] == stats
        assert result['report']['semaphore'] == 'green'

    def test_precomputed_keys(self):
        sampler = MagicMock()
        sampler.test = {'X': _make_binary_data()}
        result = valtest_annotation_quality_agent(sampler)
        assert set(result['precomputed'].keys()) == {'criteria_stats'}
