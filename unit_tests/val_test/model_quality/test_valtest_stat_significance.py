from unittest.mock import MagicMock, patch

import joblib
import matplotlib.figure
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import f1_score

from llm_val.sampler.BinaryClassSampler import BinaryClassSampler
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.sampler.MultiCriterionSampler import MultiCriterionSampler
from llm_val.scorer.BinaryClassScorer import BinaryClassScorer
from llm_val.val_test.model_quality.valtest_stat_significance import valtest_stat_significance


DATA_PATH = 'data/data_mmlu.pkl'
DATA_PATH_1 = 'data/data_stat_mmlu_1.pkl'
DATA_PATH_2 = 'data/data_stat_mmlu_2.pkl'
DATA_PATH_3 = 'data/data_stat_mmlu_3.pkl'
DATA_PATH_4 = 'data/data_stat_multicriterion.pkl'
DATA_PATH_5 = 'data/data_stat_binary_class.pkl'


@pytest.fixture()
def create_sampler_1sample(request):
    data = joblib.load(DATA_PATH)
    sampler = MMLUSampler(test=data)
    res = valtest_stat_significance(sampler,
                                    scorer=None,
                                    metric_agg='single_mean',
                                    data_type='test')
    return res


@pytest.fixture()
def create_sampler_wilcoxon(request):
    data = joblib.load(DATA_PATH)
    data1 = joblib.load(DATA_PATH_1)
    sampler = MMLUSampler(test=data)
    sampler1 = MMLUSampler(test=data1)
    extra_samplers = {'GigaPlus': sampler1}
    res = valtest_stat_significance(sampler,
                                    scorer=None,
                                    metric_agg='single_mean',
                                    data_type='test',
                                    extra_samplers=extra_samplers,
                                    same_questions=True)
    return res


@pytest.fixture()
def create_sampler_friedman(request):
    data = joblib.load(DATA_PATH)
    data1 = joblib.load(DATA_PATH_1)
    data2 = joblib.load(DATA_PATH_2)
    sampler = MMLUSampler(test=data)
    sampler1 = MMLUSampler(test=data1)
    sampler2 = MMLUSampler(test=data2)
    extra_samplers = {'GigaPlus': sampler1, 'GigaPro': sampler2}
    res = valtest_stat_significance(sampler,
                                    scorer=None,
                                    metric_agg='single_mean',
                                    data_type='test',
                                    extra_samplers=extra_samplers,
                                    same_questions=True)
    return res


@pytest.fixture()
def create_sampler_welch(request):
    data = joblib.load(DATA_PATH)
    data3 = joblib.load(DATA_PATH_3)
    sampler = MMLUSampler(test=data)
    sampler3 = MMLUSampler(test=data3)
    extra_samplers = {'GigaPro': sampler3}
    res = valtest_stat_significance(sampler,
                                    scorer=None,
                                    metric_agg='single_mean',
                                    data_type='test',
                                    extra_samplers=extra_samplers,
                                    same_questions=False)
    return res


@pytest.fixture()
def create_sampler_bootstrap(request):
    data = joblib.load(DATA_PATH)
    sampler = MMLUSampler(test=data)
    res = valtest_stat_significance(sampler,
                                    scorer=None,
                                    metric_agg='single_mean',
                                    data_type='test',
                                    main_column='metric_value',
                                    bootstrap=True,
                                    bootstrap_iterations=300,
                                    bootstrap_ci_type='percentile',
                                    alternative='greater',
                                    random_state=42)
    return res


@pytest.fixture()
def sample_data():
    np.random.seed(42)
    data = np.random.normal(loc=0.3, scale=0.1, size=100)
    return data


@pytest.fixture()
def create_sampler_bootstrap_with_comparison(sample_data):
    data_mmlu = joblib.load(DATA_PATH)
    sampler = MMLUSampler(data_mmlu)
    sampler.test['y'] = pd.DataFrame({'metric': sample_data})

    our_result = valtest_stat_significance(sampler,
                                           scorer=None,
                                           metric_agg='single_mean',
                                           data_type='test',
                                           main_column='metric',
                                           bootstrap=True,
                                           bootstrap_iterations=1000,
                                           bootstrap_ci_type='percentile',
                                           alternative='greater',
                                           random_state=42)

    value_threshold = 0.5
    observed_mean = np.mean(sample_data)
    centered_sample = sample_data - observed_mean + value_threshold
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(1000):
        boot_sample = rng.choice(centered_sample, size=len(centered_sample), replace=True)
        boot_means.append(np.mean(boot_sample))
    boot_means = np.array(boot_means)
    library_p_value = np.sum(boot_means >= observed_mean) / 1000
    bootstrap_ci = stats.bootstrap((sample_data,),
                                   lambda x: np.mean(x),
                                   n_resamples=1000,
                                   confidence_level=0.95,
                                   method='percentile',
                                   random_state=42)
    bootstrap_ci_99 = stats.bootstrap((sample_data,),
                                      lambda x: np.mean(x),
                                      n_resamples=1000,
                                      confidence_level=0.99,
                                      method='percentile',
                                      random_state=42)
    return {'our_result': our_result,
            'library_p_value': library_p_value,
            'library_ci': (bootstrap_ci.confidence_interval.low,
                           bootstrap_ci.confidence_interval.high),
            'library_ci_99': (bootstrap_ci_99.confidence_interval.low,
                              bootstrap_ci_99.confidence_interval.high),
            'sample_data': sample_data,
            'observed_mean': observed_mean,
            'value_threshold': value_threshold}


@pytest.fixture()
def sampler_with_nans():
    data_multicriterion = joblib.load(DATA_PATH_4)
    return MultiCriterionSampler(data_multicriterion)


@pytest.fixture()
def binary_scorer_setup():
    data_binary_class = joblib.load(DATA_PATH_5)
    sampler = BinaryClassSampler(data_binary_class)
    metric_config = {'f1_score': {'call': f1_score, 'is_singlecol': False}}
    scorer = BinaryClassScorer(metric_config)
    baseline_result = scorer.calc(sampler=sampler,
                                  data_type='test',
                                  metric_name='f1_score')
    expected_f1 = float(list(baseline_result.values())[0])
    simple_mean = sampler.test['y']['metric_value'].dropna().mean()
    return sampler, scorer, expected_f1, simple_mean


def test_valtest_stat_significance_report(create_sampler_1sample):
    test_res = create_sampler_1sample
    report = test_res['report']
    assert report['semaphore'] == 'red'


def test_valtest_stat_significance_1sample_precomputed(create_sampler_1sample):
    test_res = create_sampler_1sample
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'p_value'}
    assert precomputed['p_value'] > 0.3


def test_valtest_stat_significance_wilcoxon_precomputed(create_sampler_wilcoxon):
    test_res = create_sampler_wilcoxon
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'p_value'}
    assert precomputed['p_value'] > 0.9


def test_valtest_stat_significance_friedman_precomputed(create_sampler_friedman):
    test_res = create_sampler_friedman
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'p_value'}
    assert precomputed['p_value'] > 0.9


def test_valtest_stat_significance_welch_precomputed(create_sampler_welch):
    test_res = create_sampler_welch
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'p_value'}
    assert precomputed['p_value'] > 0.7
    assert precomputed['p_value'] < 0.9


def test_valtest_stat_significance_bootstrap(create_sampler_bootstrap):
    test_res = create_sampler_bootstrap
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert 'p_value' in precomputed
    assert 'ci_low' in precomputed
    assert 'ci_high' in precomputed
    assert 'ci_low_99' in precomputed
    assert 'ci_high_99' in precomputed
    assert 0 <= precomputed['p_value'] <= 1
    assert precomputed['ci_low'] <= precomputed['ci_high']
    assert precomputed['ci_low_99'] <= precomputed['ci_high_99']
    assert precomputed['ci_low_99'] <= precomputed['ci_low']
    assert precomputed['ci_high_99'] >= precomputed['ci_high']


def test_valtest_stat_significance_bootstrap_plot(create_sampler_bootstrap):
    test_res = create_sampler_bootstrap
    report = test_res['report']
    assert 'result_plots' in report
    assert isinstance(report['result_plots'], list)
    assert len(report['result_plots']) > 0
    fig = report['result_plots'][0]
    assert isinstance(fig, matplotlib.figure.Figure)


def test_bootstrap_comparison(create_sampler_bootstrap_with_comparison):
    result = create_sampler_bootstrap_with_comparison
    our_precomputed = result['our_result']['precomputed']

    p_value_diff = abs(our_precomputed['p_value'] - result['library_p_value'])
    assert p_value_diff < 0.05, f"p-value слишком сильно отличаются: {p_value_diff}"

    ci_diff_low = abs(our_precomputed['ci_low'] - result['library_ci'][0])
    ci_diff_high = abs(our_precomputed['ci_high'] - result['library_ci'][1])
    assert ci_diff_low < 0.05, f"Нижняя граница CI (0.95) слишком сильно отличается: {ci_diff_low}"
    assert ci_diff_high < 0.05, f"Верхняя граница CI (0.95) слишком сильно отличается: {ci_diff_high}"

    ci_diff_low_99 = abs(our_precomputed['ci_low_99'] - result['library_ci_99'][0])
    ci_diff_high_99 = abs(our_precomputed['ci_high_99'] - result['library_ci_99'][1])
    assert ci_diff_low_99 < 0.05, f"Нижняя граница CI (0.99) слишком сильно отличается: {ci_diff_low_99}"
    assert ci_diff_high_99 < 0.05, f"Верхняя граница CI (0.99) слишком сильно отличается: {ci_diff_high_99}"

    assert our_precomputed['ci_low'] <= result['observed_mean'] <= our_precomputed['ci_high']
    assert our_precomputed['ci_low_99'] <= result['observed_mean'] <= our_precomputed['ci_high_99']

    if result['observed_mean'] < result['value_threshold']:
        assert our_precomputed['p_value'] > 0.5
    assert 0 <= our_precomputed['p_value'] <= 1
    assert our_precomputed['ci_low'] <= our_precomputed['ci_high']
    assert our_precomputed['ci_low_99'] <= our_precomputed['ci_high_99']


@patch('llm_val.val_test.model_quality.valtest_stat_significance.ttest_1samp')
def test_nan_handling_without_bootstrap(mock_ttest, sampler_with_nans):
    mock_ttest_instance = MagicMock()
    mock_ttest_instance.pvalue = 0.5
    mock_ttest.return_value = mock_ttest_instance

    original_df = sampler_with_nans.test['y']
    expected_len = len(original_df[original_df['Корректность навигации'].notna()])

    res = valtest_stat_significance(sampler_with_nans,
                                    scorer=None,
                                    metric_agg='single_mean',
                                    data_type='test',
                                    main_column='Корректность навигации')

    call_args = mock_ttest.call_args
    passed_data = call_args[0][0]

    assert len(passed_data) == expected_len, f"Длина не совпадает: ожидалось {expected_len}, получено {len(passed_data)}"
    assert not passed_data.isna().any(), 'В переданных данных остались NaN'
    assert res['precomputed']['p_value'] == 0.5


@patch('llm_val.val_test.model_quality.valtest_stat_significance.bootstrap_confidence_interval')
@patch('llm_val.val_test.model_quality.valtest_stat_significance.bootstrap_p_value')
def test_nan_handling_with_bootstrap(mock_p_value, mock_ci, sampler_with_nans):
    mock_p_value.return_value = 0.05
    mock_ci.return_value = (0.4, 0.6)

    original_df = sampler_with_nans.test['y']
    expected_len = len(original_df[original_df['Корректность навигации'].notna()])

    res = valtest_stat_significance(sampler_with_nans,
                                    scorer=None,
                                    metric_agg='single_mean',
                                    data_type='test',
                                    main_column='Корректность навигации',
                                    bootstrap=True,
                                    bootstrap_iterations=300,
                                    bootstrap_ci_type='percentile',
                                    alternative='greater',
                                    random_state=42)

    call_args = mock_p_value.call_args
    passed_sample = call_args[0][0]

    assert len(passed_sample) == expected_len, f"Длина не совпадает: ожидалось {expected_len}, получено {len(passed_sample)}"
    assert not np.isnan(passed_sample).any(), 'В переданных данных остались NaN'
    assert res['precomputed']['p_value'] == 0.05


def test_real_scorer_is_used_in_bootstrap(binary_scorer_setup):
    sampler, scorer, expected_f1, simple_mean = binary_scorer_setup

    res = valtest_stat_significance(sampler,
                                    scorer=scorer,
                                    metric_agg='f1_score',
                                    data_type='test',
                                    main_column='metric_value',
                                    bootstrap=True,
                                    bootstrap_iterations=500,
                                    bootstrap_ci_type='percentile',
                                    alternative='greater',
                                    random_state=42)

    precomputed = res['precomputed']
    ci_low = precomputed['ci_low']
    ci_high = precomputed['ci_high']
    ci_center = (ci_low + ci_high) / 2

    assert abs(ci_center - expected_f1) < 0.05, (
        f"Центр CI ({ci_center:.4f}) не совпадает с ожидаемым F1 ({expected_f1:.4f}). "
        "Возможно, функция игнорирует скорер и использует np.mean."
    )


@patch('llm_val.val_test.model_quality.valtest_stat_significance.bootstrap_p_value')
@patch('llm_val.val_test.model_quality.valtest_stat_significance.bootstrap_confidence_interval')
def test_strategy_is_rigorous_for_single_mean(mock_ci, mock_p_value):
    mock_p_value.return_value = 0.05
    mock_ci.return_value = (0.5, 0.7)

    data = joblib.load(DATA_PATH)
    sampler = MMLUSampler(test=data)

    valtest_stat_significance(sampler,
                              scorer=None,
                              metric_agg='single_mean',
                              data_type='test',
                              main_column='metric_value',
                              bootstrap=True,
                              bootstrap_iterations=10,
                              random_state=42)

    assert mock_p_value.called, 'bootstrap_p_value не был вызван'

    call_kwargs = mock_p_value.call_args.kwargs
    assert call_kwargs.get('strategy') == 'rigorous', (
        f"Ожидалась стратегия 'rigorous' для single_mean, "
        f"но получена '{call_kwargs.get('strategy')}'"
    )


@pytest.mark.parametrize(
    'scorer, metric_agg',
    [
        (BinaryClassScorer({'f1_score': {'call': f1_score, 'is_singlecol': False}}),
         'f1_score'),
        (None,
         'custom_average')
    ]
)
@patch('llm_val.val_test.model_quality.valtest_stat_significance.bootstrap_p_value')
@patch('llm_val.val_test.model_quality.valtest_stat_significance.bootstrap_confidence_interval')
def test_strategy_is_empirical_for_both_branches(mock_ci, mock_p_value, scorer, metric_agg):
    mock_p_value.return_value = 0.05
    mock_ci.return_value = (0.5, 0.7)

    data_binary_class = joblib.load(DATA_PATH_5)
    sampler = BinaryClassSampler(data_binary_class)

    valtest_stat_significance(sampler,
                              scorer=scorer,
                              metric_agg=metric_agg,
                              data_type='test',
                              main_column='metric_value',
                              bootstrap=True,
                              bootstrap_iterations=10,
                              random_state=42)

    assert mock_p_value.called, f"bootstrap_p_value не был вызван"
    call_kwargs = mock_p_value.call_args.kwargs

    assert call_kwargs.get('strategy') == 'empirical', (
        f"Ожидалась стратегия 'empirical'', "
        f"но в функцию была передана '{call_kwargs.get('strategy')}'."
    )