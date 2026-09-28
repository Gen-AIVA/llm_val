import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

from llm_val.sampler.Sampler import Sampler
from llm_val.val_test.model_quality.valtest_metric_by_dialog_len import (
    valtest_metric_by_dialog_len,
    compute_subsample_stats,
)
from llm_val.utils.report_helper import dialog_len_semaphore

N_OBJECTS = 120


def build_dialogue(n_turns):
    return [(f'q{idx}', 'вопрос', 'ответ') for idx in range(n_turns)]


def make_sampler_data(degrade_long=False):
    rng = np.random.default_rng(42)
    session_ids = [f's{idx}' for idx in range(N_OBJECTS)]
    turn_counts = rng.integers(1, 6, size=N_OBJECTS)
    dialogues = [build_dialogue(int(count)) for count in turn_counts]
    tokens_per_session = rng.integers(50, 500, size=N_OBJECTS)
    metric = rng.uniform(0.7, 1.0, size=N_OBJECTS)
    if degrade_long:
        longest = np.argsort(tokens_per_session)[-max(50, N_OBJECTS // 10):]
        metric[longest] = 0.1
    data_x = pd.DataFrame({'session_id': session_ids, 'dialogue': dialogues})
    target = pd.DataFrame({'metric_value': metric})
    traces = pd.DataFrame(
        {'session_id': session_ids,
         'span_id': [f'span{idx}' for idx in range(N_OBJECTS)],
         'llm_total_tokens': tokens_per_session})
    return {'X': data_x, 'y': target, 'traces': traces}


def make_sampler(degrade_long=False):
    return Sampler(test=make_sampler_data(degrade_long))


def test_compute_subsample_stats_keys_and_size():
    rng = np.random.default_rng(0)
    lengths = pd.Series(np.arange(N_OBJECTS, dtype=float))
    metric = pd.Series(rng.uniform(0.0, 1.0, size=N_OBJECTS))
    stats = compute_subsample_stats(lengths, metric)
    assert set(stats.keys()) == {'p_value', 'subset_mean', 'rest_mean'}
    longest_index = lengths.nlargest(max(50, int(np.ceil(0.1 * N_OBJECTS)))).index
    rest_index = lengths.index.difference(longest_index)
    assert stats['rest_mean'] == float(metric.loc[rest_index].mean())


def test_compute_subsample_stats_min_fifty():
    rng = np.random.default_rng(1)
    lengths = pd.Series(np.arange(80, dtype=float))
    metric = pd.Series(rng.uniform(0.0, 1.0, size=80))
    stats = compute_subsample_stats(lengths, metric)
    assert not np.isnan(stats['subset_mean'])
    assert len(metric.loc[lengths.nlargest(50).index]) == 50


def test_dialog_len_semaphore_red():
    stats = {'p_value': 0.001, 'subset_mean': 0.5, 'rest_mean': 1.0}
    assert dialog_len_semaphore(stats) == 'red'


def test_dialog_len_semaphore_green_high_pvalue():
    stats = {'p_value': 0.5, 'subset_mean': 0.5, 'rest_mean': 1.0}
    assert dialog_len_semaphore(stats) == 'green'


def test_dialog_len_semaphore_green_small_drop():
    stats = {'p_value': 0.001, 'subset_mean': 0.95, 'rest_mean': 1.0}
    assert dialog_len_semaphore(stats) == 'green'


def test_dialog_len_semaphore_green_on_nan_pvalue():
    stats = {'p_value': np.nan, 'subset_mean': 0.5, 'rest_mean': 1.0}
    assert dialog_len_semaphore(stats) == 'green'


def test_dialog_len_semaphore_red_lower_is_better():
    stats = {'p_value': 0.001, 'subset_mean': 1.5, 'rest_mean': 1.0}
    assert dialog_len_semaphore(stats, greater_is_better=False) == 'red'


def test_dialog_len_semaphore_green_lower_is_better_improvement():
    stats = {'p_value': 0.001, 'subset_mean': 0.5, 'rest_mean': 1.0}
    assert dialog_len_semaphore(stats, greater_is_better=False) == 'green'


def test_compute_subsample_stats_alternative_lower_is_better():
    lengths = pd.Series(np.arange(100, dtype=float))
    metric = pd.Series(np.r_[np.linspace(0.1, 0.3, 50), np.linspace(0.8, 1.0, 50)])
    stats_default = compute_subsample_stats(lengths, metric, greater_is_better=True)
    stats_lower = compute_subsample_stats(lengths, metric, greater_is_better=False)
    assert stats_default['p_value'] > 0.5
    assert stats_lower['p_value'] < 0.05


def test_valtest_output_contract():
    sampler = make_sampler()
    test_res = valtest_metric_by_dialog_len(sampler, data_type='test')
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}
    report = test_res['report']
    assert isinstance(report, dict)
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert report['semaphore'] in ['gray', 'green', 'red']
    assert len(report['result_plots']) > 0
    assert len(report['result_dataframes']) > 0


def test_valtest_precomputed_keys():
    sampler = make_sampler()
    test_res = valtest_metric_by_dialog_len(sampler, data_type='test')
    precomputed = test_res['precomputed']
    assert set(precomputed.keys()) == {'stats_traces', 'stats_dialogue'}


def test_valtest_detects_degradation():
    sampler = make_sampler(degrade_long=True)
    test_res = valtest_metric_by_dialog_len(sampler, data_type='test')
    assert test_res['report']['semaphore'] == 'red'


def test_valtest_is_info_returns_gray():
    sampler = make_sampler(degrade_long=True)
    test_res = valtest_metric_by_dialog_len(sampler, data_type='test', is_info=True)
    assert test_res['report']['semaphore'] == 'gray'


def test_valtest_uses_precomputed_stats():
    sampler = make_sampler(degrade_long=True)
    good_stats = {'p_value': 0.9, 'subset_mean': 0.9, 'rest_mean': 0.9}
    test_res = valtest_metric_by_dialog_len(sampler, data_type='test',
                                            stats_traces=good_stats,
                                            stats_dialogue=good_stats)
    assert test_res['report']['semaphore'] == 'green'
    assert test_res['precomputed']['stats_traces'] == good_stats


def test_valtest_full_precomputed_skips_sampler_access():
    good_stats = {'p_value': 0.9, 'subset_mean': 0.9, 'rest_mean': 0.9}
    test_res = valtest_metric_by_dialog_len(sampler=None,
                                            data_type='test',
                                            stats_traces=good_stats,
                                            stats_dialogue=good_stats)
    assert test_res['report']['semaphore'] == 'green'


def test_valtest_sberds_exec_env():
    data = make_sampler_data()
    test_res = valtest_metric_by_dialog_len({'test': data}, data_type='test',
                                            exec_env='sberds')
    assert set(test_res.keys()) == {'report', 'precomputed'}
    assert test_res['report']['semaphore'] in ['gray', 'green', 'red']


def test_valtest_main_column_mean_over_y():
    data = make_sampler_data()
    data['y'] = pd.DataFrame({'metric_a': data['y']['metric_value'],
                              'metric_b': data['y']['metric_value']})
    sampler = Sampler(test=data)
    test_res = valtest_metric_by_dialog_len(sampler, data_type='test')
    assert test_res['report']['semaphore'] in ['gray', 'green', 'red']
