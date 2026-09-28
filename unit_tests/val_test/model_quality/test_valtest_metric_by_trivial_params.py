import pytest
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from pathlib import Path
from gigachat import GigaChat

from llm_val.utils.metrics import METRICS
from llm_val.sampler.TracesSampler2 import TracesSampler2
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.val_test.model_quality.valtest_metric_by_trivial_params import (
    valtest_metric_by_trivial_params
)

import llm_val.val_test.model_quality.valtest_metric_by_trivial_params as vt

DATA_PATH_1 = 'data/data_traces_short.parquet'
DATA_PATH_2 = 'data/data_ft_full.parquet'
DATA_PATH_3 = 'data/data_ft_full_embeds.npy'

PARAMS = [('bootstrap', 10, True, False, True),
          ('bootstrap', 'fd', True, False, True),
          ('welch', 10, False, False, True),
          ('MW', 10, False, False, False),
          ('welch', 'fd', False, True, True),
          ('MW', 'fd', False, True, False),]

@pytest.fixture(params=PARAMS)
def create_scorer_sampler(request):
    stattest_method, bins, context_test, is_info, greater_is_better = request.param
    data_traces2 = pd.read_parquet(DATA_PATH_1)
    data_ft_full = pd.read_parquet(DATA_PATH_2)
    sampler_traces2 = TracesSampler2(data_ft_full, data_traces2)
    scorer = UniversalScorer(metrics=METRICS)
    embeddings = np.load(DATA_PATH_3)
    res = valtest_metric_by_trivial_params(sampler = sampler_traces2,
                                           scorer = scorer,
                                           embedding_client=None,
                                           data_type ='test',
                                           context_test = context_test,
                                           bins = bins,
                                           stattest_method = stattest_method,
                                           is_info = is_info,
                                           greater_is_better = greater_is_better,
                                           embeddings = embeddings)
    return res, is_info, greater_is_better, bins


def test_valtest_metric_report(create_scorer_sampler):
    test_res, is_info, greater_is_better, bins = create_scorer_sampler
    report = test_res['report']
    if is_info:
        true_color = 'gray'
    elif greater_is_better or bins == 10:
        true_color = 'green'
    else:
        true_color = 'red'
    assert report['semaphore'] == true_color


def test_valtest_metric_precomputed(create_scorer_sampler):
    test_res, is_info, greater_is_better, bins = create_scorer_sampler
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'est_by_len_result', 'est_by_cluster_result', 'est_by_context_result',
                                       'embeddings', 'len_context', 'len_questions'}


class FakeSampler:
    def __init__(self, test=None, **kwargs):
        self.test = test


class FakeScorer:
    def calc(self, sampler, data_type='test', metric_agg='single_mean'):
        y = sampler.test['y']
        return {'q': float(np.asarray(y, dtype=float).ravel().mean())}


@pytest.fixture
def scorer(monkeypatch):
    monkeypatch.setattr(vt, "Sampler", FakeSampler)
    return FakeScorer()


def _decreasing_length_metric_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    lens = pd.Series(range(1, n + 1))
    y = pd.Series(np.clip(1.0 - np.arange(n) / n + rng.normal(0, 0.02, n), 0, 1))
    return lens, y


def _three_blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = [np.array([10., 0, 0, 0, 0]),
               np.array([0, 10., 0, 0, 0]),
               np.array([0, 0, 10., 0, 0])]
    embs, ys, qs = [], [], []
    for ci, c in enumerate(centers):
        for j in range(20):
            embs.append(c + rng.normal(0, 0.3, 5))
            base = 0.9 if ci < 2 else 0.3
            ys.append(float(np.clip(base + rng.normal(0, 0.02), 0, 1)))
            qs.append(f'c{ci}_q{j}')
    return np.array(embs), pd.Series(ys), qs


def test_enforce_min_bin_count_keeps_dense_bins():
    edges = vt.enforce_min_bin_count(list(range(100)), min_count=1, bins=10)
    assert len(edges) == 11


def test_enforce_min_bin_count_merges_sparse_tails():
    dense = vt.enforce_min_bin_count(list(range(100)), min_count=1, bins=10)
    sparse = vt.enforce_min_bin_count(list(range(100)), min_count=40, bins=10)
    assert len(sparse) < len(dense)
    assert len(sparse) >= 3


def test_enforce_min_bin_count_never_below_two_bins():
    edges = vt.enforce_min_bin_count(list(range(100)), min_count=10 ** 6, bins=10)
    assert len(edges) == 3


def test_get_representative_samples_picks_centroid_closest():
    emb = np.array([[0., 0.], [1., 0.], [2., 0.],
                    [10., 5.], [11., 5.], [12., 5.]])
    clusters = np.array([0, 0, 0, 1, 1, 1])
    questions = ['c0_a', 'c0_mid', 'c0_b', 'c1_a', 'c1_mid', 'c1_b']
    reps = vt.get_representative_samples(emb, clusters, questions, n_samples=1)
    assert reps == [['c0_mid'], ['c1_mid']]


def test_get_representative_samples_membership_and_count():
    emb = np.array([[0., 0.], [1., 0.], [2., 0.],
                    [10., 5.], [11., 5.], [12., 5.]])
    clusters = np.array([0, 0, 0, 1, 1, 1])
    questions = ['c0_a', 'c0_mid', 'c0_b', 'c1_a', 'c1_mid', 'c1_b']
    reps = vt.get_representative_samples(emb, clusters, questions, n_samples=3)
    assert [len(r) for r in reps] == [3, 3]
    assert set(reps[0]) <= {'c0_a', 'c0_mid', 'c0_b'}
    assert set(reps[1]) <= {'c1_a', 'c1_mid', 'c1_b'}


def test_get_representative_samples_caps_at_cluster_size():
    emb = np.array([[0., 0.], [0., 1.], [9., 9.]])
    clusters = np.array([0, 0, 1])
    reps = vt.get_representative_samples(emb, clusters, ['a', 'b', 'solo'], n_samples=5)
    assert reps[1] == ['solo']


def test_calculate_quantile_default_indices():
    vals = list(range(100))
    assert vt.calculate_quantile(vals, 0.95) == (2, 97)
    assert vt.calculate_quantile(vals, 0.90) == (4, 95)


def test_calculate_quantile_wider_for_higher_confidence():
    vals = list(range(1000))
    lo95, hi95 = vt.calculate_quantile(vals, 0.95)
    lo80, hi80 = vt.calculate_quantile(vals, 0.80)
    assert lo95 < lo80 and hi95 > hi80


def test_calc_conf_interval_estimation_rejects_bad_quantile():
    data = pd.DataFrame({'m': [0, 1, 1]})
    with pytest.raises(ValueError):
        vt.calc_conf_interval_estimation(data, {'m': 0.5}, 1.0)
    with pytest.raises(ValueError):
        vt.calc_conf_interval_estimation(data, {'m': 0.5}, 0.0)


def testdialogue_to_text_format():
    dlg = [(1, 'Hi', 'Hello'), (1, 'Bye', 'Goodbye')]
    assert vt.dialogue_to_text(dlg) == 'USER: Hi\nASSISTANT:Hello\nUSER: Bye\nASSISTANT:Goodbye\n'


def testmean_dialogue_question_lengths_default_counter():
    d = [(0, 'abcd', 'x'), (0, 'abcdabcd', 'y')]
    assert vt.mean_dialogue_question_lengths([d]) == [6.0]


def testmean_dialogue_question_lengths_custom_counter():
    wc = lambda texts: [len(t.split()) for t in texts]
    d = [(0, 'one two', 'a'), (0, 'three', 'b')]
    assert vt.mean_dialogue_question_lengths([d], wc) == [1.5]


def testmean_dialogue_question_lengths_ignores_answers():
    d = [(0, 'abcd', 'answer' * 100)]
    assert vt.mean_dialogue_question_lengths([d]) == [4.0]


def testmean_dialogue_question_lengths_one_value_per_dialogue():
    out = vt.mean_dialogue_question_lengths([[(0, 'a', 'x')],
                                              [(1, 'bb', 'y'), (1, 'cc', 'z')]])
    assert len(out) == 2


def _raw_len_df():
    return pd.DataFrame({
        'bin_metric': [0.123456, np.nan, np.array([0.7])],
        'other_metric': [0.5, 0.6, 0.7],
        'bin_size': [10, 20, 30],
        'other_size': [90, 80, 70],
        'p-value': [0.04999, np.nan, 0.5],
        'diff_ratio': [1.2345, 0.5, np.nan],
        'business_thr_p_val': [0.01, np.nan, 0.2],
        'business_thr_ratio': [2.0, np.nan, 1.5],
        'bounds': ['1-2', '2-3', '3-4'],
    })


def test_format_df_non_cluster_formatting():
    out = vt.format_df(_raw_len_df())
    assert out['bin_metric'].tolist() == ['0.123', 'NA', '0.700']
    assert out['p-value'].tolist() == ['0.050', 'NA', '0.500']
    assert out['diff_ratio'].tolist() == ['1.23', '0.50', 'NA']
    assert out['business_thr_ratio'].tolist() == ['2.00', 'NA', '1.50']


def test_format_df_does_not_mutate_input():
    df = _raw_len_df()
    vt.format_df(df)
    assert isinstance(df['bin_metric'].iloc[2], np.ndarray)
    assert np.isnan(df['bin_metric'].iloc[1])
    assert df['p-value'].iloc[0] == 0.04999


def test_calc_one_sided_test_bootstrap_path(scorer):
    a = pd.Series([0.0, 0.1, 0.05, 0.0, 0.1, 0.05, 0.0, 0.1])
    b = pd.Series([0.9, 1.0, 0.95, 0.9, 1.0, 0.95, 0.9, 1.0])
    p = vt.calc_one_sided_test(a, b, scorer, stattest_method='bootstrap',
                               n_bootstrap=200, greater_is_better=True)
    assert p == pytest.approx(0.0)


def test_calc_pr_bootstrap_small_when_data1_worse(scorer):
    low = pd.Series([0.0, 0.1, 0.05, 0.0, 0.1, 0.05, 0.0, 0.1])
    high = pd.Series([0.9, 1.0, 0.95, 0.9, 1.0, 0.95, 0.9, 1.0])
    assert vt.calc_pr_bootstrap(low, high, scorer, n_bootstrap=200, greater_is_better=True) == pytest.approx(0.0)


def test_calc_pr_bootstrap_large_when_data1_better(scorer):
    low = pd.Series([0.0, 0.1, 0.05, 0.0, 0.1, 0.05, 0.0, 0.1])
    high = pd.Series([0.9, 1.0, 0.95, 0.9, 1.0, 0.95, 0.9, 1.0])
    assert vt.calc_pr_bootstrap(high, low, scorer, n_bootstrap=200, greater_is_better=True) == pytest.approx(1.0)


def test_calc_pr_bootstrap_greater_is_better_flips(scorer):
    low = pd.Series([0.0, 0.1, 0.05, 0.0, 0.1, 0.05, 0.0, 0.1])
    high = pd.Series([0.9, 1.0, 0.95, 0.9, 1.0, 0.95, 0.9, 1.0])
    assert vt.calc_pr_bootstrap(low, high, scorer, n_bootstrap=200, greater_is_better=True) == pytest.approx(0.0)
    assert vt.calc_pr_bootstrap(low, high, scorer, n_bootstrap=200, greater_is_better=False) == pytest.approx(1.0)


def test_bca_pvalue_small_when_metric_below_threshold(scorer):
    y = pd.Series([0.2, 0.3, 0.25, 0.35, 0.28, 0.22, 0.31, 0.27, 0.24, 0.29])
    p = vt.bca_pvalue(y, scorer, business_threshold=0.8, n_bootstrap=500,
                      random_state=1, greater_is_better=True)
    assert 0.0 <= p <= 1.0
    assert p < 0.05


def test_bca_pvalue_large_when_metric_above_threshold(scorer):
    y = pd.Series([0.2, 0.3, 0.25, 0.35, 0.28, 0.22, 0.31, 0.27, 0.24, 0.29])
    p = vt.bca_pvalue(y, scorer, business_threshold=0.05, n_bootstrap=500,
                      random_state=1, greater_is_better=True)
    assert p > 0.5


def test_bca_pvalue_deterministic(scorer):
    y = pd.Series([0.2, 0.3, 0.25, 0.35, 0.28, 0.22, 0.31, 0.27, 0.24, 0.29])
    p1 = vt.bca_pvalue(y, scorer, 0.5, n_bootstrap=300, random_state=5)
    p2 = vt.bca_pvalue(y, scorer, 0.5, n_bootstrap=300, random_state=5)
    assert p1 == p2


def test_est_len_stat_structure_and_partition(scorer):
    lens, y = _decreasing_length_metric_data()
    df = vt.est_len_stat(lens, y, scorer, min_count=20, bins=5,
                         stattest_method='welch', greater_is_better=True)
    assert list(df.columns) == ['bin_metric', 'other_metric', 'bin_size', 'other_size',
                                'p-value', 'diff_ratio', 'business_thr_p_val',
                                'business_thr_ratio', 'bounds']
    assert ((df['bin_size'] + df['other_size']) == len(y)).all()
    assert df['business_thr_p_val'].isna().all()
    assert df['business_thr_ratio'].isna().all()


def test_est_len_stat_detects_length_trend(scorer):
    lens, y = _decreasing_length_metric_data()
    df = vt.est_len_stat(lens, y, scorer, min_count=20, bins=5,
                         stattest_method='welch', greater_is_better=True)
    assert df['bin_metric'].iloc[0] > df['bin_metric'].iloc[-1]
    assert df['diff_ratio'].iloc[0] > 1 > df['diff_ratio'].iloc[-1]
    assert df['p-value'].iloc[-1] < 0.05


def test_cluster_stat_kmeans_finds_weak_cluster(scorer):
    embs, y, qs = _three_blobs()
    df = vt.cluster_stat(embs, y, qs, scorer, stattest_method='welch', greater_is_better=True)
    assert list(df.columns) == ['cluster_metric', 'other_metric', 'cluster_size',
                                'other_size', 'p-value', 'diff_ratio', 'sample_questions']
    assert ((df['cluster_size'] + df['other_size']) == len(y)).all()
    assert df['cluster_size'].sum() == len(y)
    assert df['sample_questions'].apply(lambda x: isinstance(x, list)).all()
    worst = df.sort_values('diff_ratio').iloc[0]
    assert worst['diff_ratio'] < 0.9
    assert worst['p-value'] < 0.05


def test_calc_conf_interval_by_bootstrap_ordered_and_deterministic(scorer):
    g = pd.DataFrame({'q': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    a = vt.calc_conf_interval_by_bootstrap(g, scorer, random_state=7)
    b = vt.calc_conf_interval_by_bootstrap(g, scorer, random_state=7)
    assert a == b
    lo, hi = a['q']
    assert lo <= hi
    assert 0.1 <= lo and hi <= 0.6


def _report_len_df():
    return pd.DataFrame({'bin_metric': [0.8], 'other_metric': [0.6], 'p-value': [0.2],
                         'diff_ratio': [1.3], 'business_thr_p_val': [0.1],
                         'business_thr_ratio': [1.6], 'bounds': ['1-2']})


def _report_cluster_df():
    return pd.DataFrame({'cluster_metric': [0.7], 'other_metric': [0.6], 'p-value': [0.3],
                         'diff_ratio': [1.1], 'sample_questions': [['q']]})


def test_report_info_test_is_gray():
    r = vt.report_valtest_metric_by_trivial_params(_report_len_df(), _report_cluster_df(),
                                                   None, is_info=True, fig='FIG')
    assert r['semaphore'] == 'gray'
    assert r['result_plots'] == 'FIG'
    assert len(r['result_dataframes']) == 3
    assert r['result_dataframes'][2] is None


def test_report_includes_context_dataframe_when_provided(monkeypatch):
    monkeypatch.setattr(vt, 'semaphore_by_threshold_for_lentest', lambda *a, **k: 'green')
    r = vt.report_valtest_metric_by_trivial_params(_report_len_df(), _report_cluster_df(),
                                                   _report_len_df(), is_info=False)
    assert r['result_dataframes'][2] is not None