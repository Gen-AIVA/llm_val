from functools import partial
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import f1_score

from llm_val.sampler.BinaryClassSampler import BinaryClassSampler
from llm_val.sampler.NerSampler import NerSampler
from llm_val.scorer.BinaryClassScorer import BinaryClassScorer
from llm_val.scorer.NerScorer import NerScorer
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.utils.metrics import METRICS
from llm_val.utils.ner_metrics import NER_METRICS, ner_metrics_exact_match_macro_f1, ner_metrics_exact_match_micro_f1
from llm_val.val_test.model_quality.stat_significance_v4 import _p_value_color
from llm_val.val_test.model_quality.valtest_stat_significance import valtest_stat_significance


@pytest.fixture(autouse=True)
def close_figures():
    """Close plots after each test."""
    yield
    plt.close('all')


def _binary_data():
    truth = np.tile([0, 1], 80)
    a = truth.copy()
    a[:23] ^= 1
    b = truth.copy()
    b[16:49] ^= 1
    ids = np.arange(len(truth))
    return (
        pd.DataFrame({'session_id': ids, 'true_answer': truth, 'model_answer': a}),
        pd.DataFrame({'session_id': ids, 'true_answer': truth, 'model_answer': b}),
    )


def _sampler(frame, x=None):
    return SimpleNamespace(test={'y': frame, 'X': x})


def _comparison(a, **kwargs):
    kwargs.setdefault('bootstrap_iterations', 40)
    result = valtest_stat_significance(_sampler(a), **kwargs)
    assert result['report']['semaphore'] == 'gray'
    return result['precomputed']['comparisons'][0]


@pytest.mark.parametrize(
    ('metric', 'denominator', 'numerator'),
    [
        ('accuracy', 160, 137),
        ('precision', 80, 69),
        ('recall', 80, 69),
    ],
)
def test_exact_threshold(metric, denominator, numerator):
    """Use the success count and denominator specific to each metric."""
    a, _ = _binary_data()
    if metric == 'precision':
        denominator = int(a.model_answer.sum())
    result = _comparison(a, metric_agg=metric, value_threshold=0.7, p_value=0.99)
    expected = stats.binomtest(numerator, denominator, 0.7, alternative='greater')
    assert result['method_id'] == 'binomial'
    assert result['estimate'] == pytest.approx(numerator / denominator)
    assert result['p_value'] == pytest.approx(expected.pvalue)


@pytest.mark.parametrize('metric', ['accuracy', 'recall'])
def test_exact_paired_alignment(metric):
    """Compute discordances after matching stable IDs, regardless of row order."""
    a, b = _binary_data()
    result = _comparison(
        a, metric_agg=metric, same_questions=True, extra_samplers={'B': _sampler(b.sample(frac=1, random_state=19))}
    )
    correct_a = a.true_answer == a.model_answer
    correct_b = b.true_answer == b.model_answer
    if metric == 'recall':
        correct_a = correct_a[a.true_answer == 1]
        correct_b = correct_b[b.true_answer == 1]
    n10 = int((correct_a & ~correct_b).sum())
    n01 = int((~correct_a & correct_b).sum())
    assert result['method_id'] == 'mcnemar'
    assert result['p_value'] == pytest.approx(stats.binomtest(n10, n10 + n01, alternative='greater').pvalue)


@pytest.mark.parametrize('metric', ['accuracy', 'precision', 'recall'])
def test_exact_independent(metric):
    """Use the contingency table for the requested metric and direction."""
    a, b = _binary_data()
    b = b.iloc[10:]
    result = _comparison(a, metric_agg=metric, extra_samplers={'B': _sampler(b)}, alternative='less')
    table = []
    for data in (a, b):
        if metric == 'precision':
            data = data[data.model_answer == 1]
        elif metric == 'recall':
            data = data[data.true_answer == 1]
        successes = int((data.true_answer == data.model_answer).sum())
        table.append([successes, len(data) - successes])
    assert result['method_id'] == 'fisher'
    assert result['p_value'] == pytest.approx(stats.fisher_exact(table, alternative='less').pvalue)


def _gradient_and_counts(data, metric):
    truth, predicted = data.true_answer, data.model_answer
    counts = np.column_stack(
        ((truth == 1) & (predicted == 1), (truth == 0) & (predicted == 1), (truth == 1) & (predicted == 0))
    ).astype(float)
    tp, fp, fn = counts.sum(axis=0)
    if metric == 'precision':
        denominator = tp + fp
        return tp / denominator, np.array([fp, -tp, 0]) / denominator**2, counts
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator, np.array([2 * (fp + fn), -2 * tp, -2 * tp]) / denominator**2, counts


@pytest.mark.parametrize(
    ('metric', 'mode'),
    [
        ('f1', 'threshold'),
        ('f1', 'paired'),
        ('f1', 'independent'),
        ('precision', 'paired'),
    ],
)
def test_delta_covariance(metric, mode):
    """Cross-check the influence-function calculation with covariance of totals."""
    a, b = _binary_data()
    value_a, gradient_a, counts_a = _gradient_and_counts(a, metric)
    if mode == 'threshold':
        result = _comparison(a, metric_agg=metric, value_threshold=0.7)
        estimate, reference = value_a, 0.7
        variance = gradient_a @ (len(a) * np.cov(counts_a, rowvar=False)) @ gradient_a
    else:
        if mode == 'independent':
            b = b.iloc[12:]
        value_b, gradient_b, counts_b = _gradient_and_counts(b, metric)
        estimate, reference = value_a - value_b, 0.02
        result = _comparison(
            a,
            metric_agg=metric,
            same_questions=mode == 'paired',
            extra_samplers={'B': _sampler(b)},
            comparison_margin=reference,
        )
        if mode == 'paired':
            gradient = np.r_[gradient_a, -gradient_b]
            variance = gradient @ (len(a) * np.cov(np.column_stack((counts_a, counts_b)), rowvar=False)) @ gradient
        else:
            variance = sum(
                gradient @ (len(counts) * np.cov(counts, rowvar=False)) @ gradient
                for gradient, counts in ((gradient_a, counts_a), (gradient_b, counts_b))
            )
    assert result['method_id'] == 'delta'
    assert result['estimate'] == pytest.approx(estimate)
    assert result['p_value'] == pytest.approx(stats.norm.sf((estimate - reference) / np.sqrt(variance)))


@pytest.mark.parametrize(('size', 'available'), [(99, False), (100, False), (101, True)])
def test_observation_boundary(size, available):
    """The threshold is strict and the semaphore is informational in every case."""
    a, _ = _binary_data()
    row = _comparison(a.iloc[:size], metric_agg='accuracy')
    assert (row['p_value'] is not None) is available
    assert (row['interval_95'] is not None) is available
    if not available:
        assert 'N > 100' in row['comment']


def test_missing_counter_and_pair_ids():
    """Undefined values and invalid pairing get distinct outcomes."""
    a, b = _binary_data()
    a['model_answer'] = 0
    row = _comparison(a, metric_agg='precision')
    assert row['p_value'] is None
    assert 'Нулевой знаменатель' in row['comment']
    with pytest.raises(ValueError, match='Наборы идентификаторов'):
        _comparison(a, metric_agg='accuracy', same_questions=True, extra_samplers={'B': _sampler(b.iloc[1:])})
    with pytest.raises(ValueError, match='уникальным'):
        _comparison(
            a, metric_agg='accuracy', same_questions=True, extra_samplers={'B': _sampler(b.assign(session_id=0))}
        )


def test_clustered_counts_fall_back_to_bootstrap():
    """Resample documents when their entities are clustered."""
    counts = pd.DataFrame({'TP': np.tile([2, 3], 65), 'FP': np.tile([0, 2], 65), 'FN': 1})
    row = _comparison(counts, metric_agg='recall', value_threshold=0.7, bootstrap=True)
    assert row['method_id'] == 'bootstrap'
    assert row['estimate'] == pytest.approx(2.5 / 3.5)
    assert 'бутстрэп по объектам' in row['comment']


@pytest.mark.parametrize('true_positives', [1, 2])
def test_bootstrap_cannot_estimate_precision_from_one_informative_object(true_positives):
    """Keep exact inference where available and report why bootstrap is unavailable."""
    counts = pd.DataFrame({'TP': [true_positives] + [0] * 100, 'FP': 0, 'FN': 0})
    result = valtest_stat_significance(
        _sampler(counts),
        metric_agg='precision',
        value_threshold=0.9,
        bootstrap=true_positives == 1,
    )
    assert result['report']['semaphore'] == 'gray'
    rows = result['precomputed']['comparisons']
    boot = rows[-1]
    assert boot['method_id'] == 'bootstrap'
    assert boot['metric_a'] == 1.0
    assert boot['p_value'] is None
    assert boot['interval_95'] is None
    assert 'двух объектов' in boot['comment']
    if true_positives == 1:
        exact = rows[0]
        assert exact['method_id'] == 'binomial'
        assert exact['p_value'] == pytest.approx(0.9)
        assert exact['interval_95'] == pytest.approx((0.025, 1.0))
    else:
        assert len(rows) == 1


def test_bootstrap_does_not_use_only_defined_resamples():
    """Undefined resamples leave the empirical tail unavailable."""
    counts = pd.DataFrame({'TP': [1, 0] + [0] * 99, 'FP': [0, 1] + [0] * 99, 'FN': 0})
    row = _comparison(counts, metric_agg='precision', methods=['bootstrap'], random_state=7)
    assert row['p_value'] is None
    assert row['interval_95'] is None
    assert 'неопределённую метрику' in row['comment']


def test_exact_comparison_keeps_p_value_when_bootstrap_interval_is_unavailable():
    """A sparse confidence interval does not invalidate exact inference."""
    a = pd.DataFrame({'TP': [1] + [0] * 100, 'FP': 0, 'FN': 0})
    b = pd.DataFrame({'TP': 0, 'FP': [1] + [0] * 100, 'FN': 0})
    row = _comparison(a, metric_agg='precision', extra_samplers={'B': _sampler(b)})
    assert row['method_id'] == 'fisher'
    assert row['p_value'] == pytest.approx(0.5)
    assert row['interval_95'] is None
    assert 'двух объектов' in row['comment']


def test_mean_methods_and_retained_plots():
    """Report agreed methods and analytic and resampled intervals in one result."""
    a = pd.DataFrame({'score': np.linspace(0.3, 0.9, 125)})
    result = valtest_stat_significance(
        _sampler(a), main_column='score', bootstrap=True, bootstrap_iterations=40, value_threshold=0.55
    )
    student, bootstrap = result['precomputed']['comparisons']
    assert student['method_id'] == 'student'
    assert student['p_value'] == pytest.approx(stats.ttest_1samp(a.score, 0.55, alternative='greater').pvalue)
    assert bootstrap['method_id'] == 'bootstrap'
    assert len(result['report']['result_dataframes'][0]) == 2
    assert isinstance(result['report']['result_plots'][0], matplotlib.figure.Figure)


def test_constant_mean_at_threshold_has_no_significant_improvement():
    """Rounding of a constant sample must not produce a false rejection."""
    a = pd.DataFrame({'score': np.full(150, 0.7)})
    result = valtest_stat_significance(
        _sampler(a), main_column='score', value_threshold=0.7, bootstrap=True,
        bootstrap_iterations=20, significance_level=0.05,
    )
    student, bootstrap = result['precomputed']['comparisons']
    assert student['method_id'] == 'student'
    assert student['p_value'] is None
    assert 'нулевой стандартной ошибке' in student['comment']
    assert bootstrap['p_value'] == 1.0
    assert 'не отклоняется' in bootstrap['conclusion']
    assert result['report']['semaphore'] == 'gray'


def test_bootstrap_precision_difference_at_threshold_counts_as_tie():
    """Treat 0.8 − 0.7 as the specified margin of 0.1 within floating-point error."""
    a = pd.DataFrame({'tp': [8] * 125, 'fp': [2] * 125, 'fn': [0] * 125})
    b = a.assign(tp=7, fp=3)
    row = _comparison(
        a, metric_agg='precision', extra_samplers={'B': _sampler(b)}, comparison_margin=0.1,
        bootstrap_iterations=20, significance_level=0.05,
    )
    assert row['method_id'] == 'bootstrap'
    assert row['estimate'] == pytest.approx(0.1)
    assert row['p_value'] == 1.0
    assert 'не отклоняется' in row['conclusion']


def test_mean_comparison_methods():
    """Use the configured margin consistently in Wilcoxon and Welch."""
    ids = np.arange(125)
    a = pd.DataFrame({'session_id': ids, 'score': np.linspace(0.3, 0.9, len(ids))})
    b = pd.DataFrame({'session_id': ids, 'score': np.linspace(0.2, 0.87, len(ids))})
    margin = 0.025
    for paired in (True, False):
        result = _comparison(
            a, main_column='score', same_questions=paired, comparison_margin=margin, extra_samplers={'B': _sampler(b)}
        )
        expected = (
            stats.wilcoxon(a.score - b.score - margin, alternative='less')
            if paired
            else stats.ttest_ind(a.score - margin, b.score, equal_var=False, alternative='less')
        )
        assert result['method_id'] == ('wilcoxon' if paired else 'welch')
        assert result['p_value'] == pytest.approx(expected.pvalue)
        assert result['alternative'] == 'A − B < 0.025'


@pytest.mark.parametrize(('paired', 'default_method'), [(False, 'welch'), (True, 'wilcoxon')])
def test_bootstrap_alternative_does_not_depend_on_other_methods(paired, default_method):
    """Keep the same mean-score hypothesis in one- and two-method reports."""
    a = pd.DataFrame({'session_id': np.arange(125), 'score': np.linspace(0.75, 0.85, 125)})
    b = a.assign(score=np.linspace(0.60, 0.70, len(a)))
    params = {
        'main_column': 'score', 'same_questions': paired, 'extra_samplers': {'B': _sampler(b)},
        'bootstrap_iterations': 20, 'random_state': 42,
    }
    only = _comparison(a, methods=['bootstrap'], **params)
    result = valtest_stat_significance(_sampler(a), methods=[default_method, 'bootstrap'], **params)
    _, together = result['precomputed']['comparisons']
    assert only['alternative'] == together['alternative'] == 'A − B < 0'
    assert only['p_value'] == together['p_value'] == 1.0


def test_wilcoxon_treats_pairs_at_margin_as_ties():
    """A rounding error in every paired difference is compatible with H0."""
    a = pd.DataFrame({'session_id': np.arange(125), 'score': 0.8})
    b = a.assign(score=0.7)
    row = _comparison(
        a, main_column='score', extra_samplers={'B': _sampler(b)}, same_questions=True,
        comparison_margin=0.1, alternative='greater', bootstrap_iterations=20,
    )
    assert row['method_id'] == 'wilcoxon'
    assert row['p_value'] == 1.0


def test_paired_effect_uses_signed_difference_for_unsigned_scores():
    """Keep the effect, confidence interval and p-value consistent for uint scores."""
    a = pd.DataFrame({'session_id': np.arange(125), 'score': np.zeros(125, dtype=np.uint8)})
    b = a.assign(score=np.ones(len(a), dtype=np.uint8))
    result = valtest_stat_significance(
        _sampler(a), main_column='score', extra_samplers={'B': _sampler(b)}, same_questions=True,
        methods=['wilcoxon', 'bootstrap'], bootstrap_iterations=20,
    )
    for row in result['precomputed']['comparisons']:
        assert row['estimate'] == -1.0
        assert row['interval_95'] == (-1.0, -1.0)
        assert row['p_value'] < 0.05


def test_large_common_offset_is_removed_before_mean_comparison():
    """Keep a difference of one at a common score level of 10**16."""
    a = pd.DataFrame({'score': np.repeat([1e16, 1e16 + 2], 64)})
    b = a.assign(score=1e16)
    result = valtest_stat_significance(
        _sampler(a), main_column='score', extra_samplers={'B': _sampler(b)}, alternative='greater',
        methods=['welch', 'bootstrap'], bootstrap_iterations=20,
    )
    for row in result['precomputed']['comparisons']:
        assert row['estimate'] == 1.0
        assert row['interval_95'][0] > 0
        assert row['p_value'] < 0.05


@pytest.mark.parametrize('compare_versions', [False, True])
def test_repeated_objects_are_rejected_in_independent_samples(compare_versions):
    """Repeated scores for ten IDs cannot be counted as 150 independent observations."""
    a = pd.DataFrame({'session_id': np.repeat(np.arange(10), 15), 'metric_value': np.tile([0, 1, 1], 50)})
    if compare_versions:
        kwargs = {'extra_samplers': {'B': _sampler(a)}}
        a = a.assign(session_id=np.arange(len(a)))
    else:
        kwargs = {'id_column': 'session_id'}
    with pytest.raises(ValueError, match="'session_id' должен быть уникальным"):
        _comparison(a, metric_agg='accuracy', **kwargs)


def test_named_index_and_explicit_identifier_are_treated_as_ids():
    """Check named object indices and exclude an ID column from a mean score."""
    a = pd.DataFrame({'object_key': np.arange(128), 'score': np.linspace(0, 1, 128)})
    row = _comparison(a, id_column='object_key')
    assert row['estimate'] == pytest.approx(0.5)
    repeating = a.assign(object_key=np.arange(len(a)) // 8).set_index('object_key')
    with pytest.raises(ValueError, match="'object_key' должен быть уникальным"):
        _comparison(repeating)


@pytest.mark.parametrize('bootstrap', [False, True])
def test_saved_mean_p_value_preserves_the_legacy_report(bootstrap):
    """A legacy scalar can be displayed even with the source sampler present."""
    a = pd.DataFrame({'score': np.linspace(0, 1, 125)})
    result = valtest_stat_significance(
        _sampler(a), main_column='score', value_threshold=0.65, p_value=0.001, bootstrap=bootstrap,
        bootstrap_iterations=20, random_state=42,
    )
    assert result['precomputed']['p_value'] == 0.001
    assert result['report']['semaphore'] == 'gray'
    assert bool(result['report']['result_plots']) is bootstrap


def test_saved_v4_comparisons_are_reused_with_sampler():
    """Recreate the statistical table and the figure from the cached comparisons."""
    a = pd.DataFrame({'score': np.linspace(0.3, 0.9, 125)})
    original = valtest_stat_significance(_sampler(a), main_column='score', significance_level=0.05)
    saved = original['precomputed']
    restored = valtest_stat_significance(SimpleNamespace(), main_column='score', significance_level=0.05, **saved)
    assert restored['precomputed'] == saved
    assert restored['report']['semaphore'] == 'gray'
    assert len(restored['report']['result_plots']) == len(original['report']['result_plots'])
    pd.testing.assert_frame_equal(
        restored['report']['result_dataframes'][0], original['report']['result_dataframes'][0],
    )
    changed = valtest_stat_significance(_sampler(a), main_column='score', significance_level=0.05,
                                        value_threshold=0.9, **saved)
    assert changed['precomputed']['p_value'] > 0.95


def test_custom_metric_paired_bootstrap():
    """Preserve the paired effect and resample both versions together."""
    ids = np.arange(125)
    a = pd.DataFrame({'session_id': ids, 'score': np.linspace(0.1, 0.8, len(ids))})
    b = a.assign(score=a.score - 0.1).sample(frac=1, random_state=3)
    row = _comparison(
        a,
        metric_agg='custom_mean',
        calc_metric_fn=lambda data: data.score.mean(),
        same_questions=True,
        comparison_margin=0.05,
        extra_samplers={'B': _sampler(b)},
    )
    assert row['method_id'] == 'bootstrap'
    assert row['estimate'] == pytest.approx(0.1)
    assert row['interval_95'] == pytest.approx((0.1, 0.1))
    assert row['p_value'] == 0


def test_stage_call_contract_and_additional_thresholds():
    """Keep the normal Sampler call and expose individual metrics in the report."""
    a, _ = _binary_data()
    sampler = BinaryClassSampler(a.rename(columns={'true_answer': 'label', 'session_id': 'text'}))
    result = valtest_stat_significance(
        sampler,
        None,
        'accuracy_score',
        value_threshold=0.75,
        metric_thresholds={'precision': 0.7, 'recall': 0.7, 'f1': 0.7},
    )
    assert result['report']['semaphore'] == 'gray'
    assert [item['metric'] for item in result['precomputed']['comparisons']] == [
        'accuracy_score',
        'precision',
        'recall',
        'f1',
    ]


def test_additional_threshold_uses_its_own_alternative():
    """A lower-tail primary hypothesis leaves the Recall threshold upper-tailed."""
    data = pd.DataFrame(
        {'score': [2, 3, 4, 3, 5] * 24, 'true_answer': [0, 0, 1, 1, 1] * 24, 'model_answer': [0, 1, 1, 1, 0] * 24}
    )
    result = valtest_stat_significance(
        _sampler(data),
        main_column='score',
        value_threshold=4.5,
        alternative='less',
        metric_thresholds={'recall': 0.82},
    )
    primary, recall = result['precomputed']['comparisons']
    assert primary['alternative'] == 'A < 4.5'
    assert recall['alternative'] == 'A > 0.82'
    assert recall['p_value'] == pytest.approx(stats.binomtest(48, 72, 0.82, alternative='greater').pvalue)


def test_additional_threshold_does_not_replace_primary_threshold():
    """Report two thresholds for one metric and keep the primary result first."""
    a, _ = _binary_data()
    result = valtest_stat_significance(
        _sampler(a), metric_agg='f1', value_threshold=0.7, metric_thresholds={'F1': 0.55}
    )
    primary, additional = result['precomputed']['comparisons']
    assert (primary['reference'], additional['reference']) == (0.7, 0.55)
    assert result['precomputed']['p_value'] == primary['p_value']


def test_additional_threshold_reuses_same_scorer_formula():
    """Keep a custom F1 definition on both thresholds, including alias keys."""
    data = pd.DataFrame({'text': np.arange(120), 'label': [0] * 90 + [1] * 30, 'model_answer': [0] * 110 + [1] * 10})
    scorer = BinaryClassScorer({'f1_score': {'call': partial(f1_score, average='macro'), 'is_singlecol': False}})
    result = valtest_stat_significance(
        BinaryClassSampler(data),
        scorer,
        'f1_score',
        main_column='metric_value',
        value_threshold=0.6,
        metric_thresholds={'F1': 0.65},
        bootstrap_iterations=40,
        random_state=7,
    )
    primary, additional = result['precomputed']['comparisons']
    assert (primary['reference'], additional['reference']) == (0.6, 0.65)
    assert primary['estimate'] == pytest.approx(0.7)
    assert additional['estimate'] == pytest.approx(0.7)
    assert primary['method_id'] == additional['method_id'] == 'bootstrap'


def test_additional_accuracy_uses_binary_scores_with_mean_scorer():
    """A primary scorer column is independent from the additional Accuracy column."""
    data = pd.DataFrame({'latency': [1, 2, 3] * 40, 'metric_value': [1, 1, 0] * 40})
    result = valtest_stat_significance(
        _sampler(data),
        UniversalScorer(METRICS),
        main_metric='single_mean_latency',
        metric_thresholds={'accuracy': 0.6},
        bootstrap_iterations=40,
    )
    mean, accuracy = result['precomputed']['comparisons']
    assert mean['estimate'] == 2
    assert accuracy['estimate'] == pytest.approx(2 / 3)
    assert accuracy['p_value'] == pytest.approx(stats.binomtest(80, 120, 0.6, alternative='greater').pvalue)


def test_second_threshold_matches_explicit_primary_metric_kind():
    """Preserve the selected Accuracy column across thresholds."""
    data = pd.DataFrame({'correctness': [1] * 108 + [0] * 12, 'metric_value': [1] * 24 + [0] * 96})
    result = valtest_stat_significance(
        _sampler(data),
        metric_kind='accuracy',
        main_column='correctness',
        metric_thresholds={'accuracy': 0.85},
    )
    primary, additional = result['precomputed']['comparisons']
    assert primary['estimate'] == additional['estimate'] == pytest.approx(0.9)
    assert additional['p_value'] == pytest.approx(stats.binomtest(108, 120, 0.85, alternative='greater').pvalue)


def test_explicit_primary_kind_does_not_override_distinct_additional_metric():
    """An F1 threshold stays F1 when the primary call explicitly selects Accuracy."""
    a, _ = _binary_data()
    result = valtest_stat_significance(
        _sampler(a),
        metric_agg='f1_score',
        metric_kind='accuracy',
        metric_thresholds={'f1': 0.7},
    )
    primary, additional = result['precomputed']['comparisons']
    assert primary['method_id'] == 'binomial'
    assert additional['method_id'] == 'delta'
    assert additional['estimate'] == pytest.approx(f1_score(a.true_answer, a.model_answer))


def test_second_threshold_accepts_primary_custom_metric_key():
    """Allow a second business threshold for a metric defined by the caller."""
    data = pd.DataFrame({'score': np.linspace(0.3, 0.9, 120)})
    result = valtest_stat_significance(
        _sampler(data),
        metric_agg='custom_score',
        calc_metric_fn=lambda rows: rows['score'].mean(),
        value_threshold=0.5,
        metric_thresholds={'custom_score': 0.55},
        bootstrap_iterations=40,
    )
    primary, additional = result['precomputed']['comparisons']
    assert (primary['reference'], additional['reference']) == (0.5, 0.55)
    assert primary['estimate'] == additional['estimate'] == pytest.approx(0.6)


@pytest.mark.parametrize('methods', [['binomial', 'delta'], ['delta']])
def test_methods_are_selected_per_hypothesis(methods):
    """Use a compatible configured method or the default for each hypothesis."""
    a, _ = _binary_data()
    result = valtest_stat_significance(
        _sampler(a), metric_agg='precision', metric_thresholds={'f1': 0.55}, methods=methods
    )
    assert [row['method_id'] for row in result['precomputed']['comparisons']] == ['binomial', 'delta']


def test_report_justifies_consistent_conclusions_for_one_effect():
    """Show why two methods support the conclusion for the same hypothesis."""
    data = pd.DataFrame({'score': np.tile([0.8, 0.9], 60)})
    result = valtest_stat_significance(
        _sampler(data),
        main_column='score',
        value_threshold=0.7,
        bootstrap=True,
        bootstrap_iterations=40,
        significance_level=0.05,
    )
    primary, secondary = result['precomputed']['comparisons']
    assert primary['effect'] == secondary['effect']
    assert 'Оба метода отклоняют H₀' in primary['justification']
    assert result['report']['result_dataframes'][0].at[0, 'Обоснование'] == primary['justification']


def test_report_separates_wilcoxon_and_bootstrap_effects():
    """Distinguish the median paired difference from the mean paired difference."""
    a = pd.DataFrame({'session_id': range(120), 'score': [0.52] * 105 + [0.10] * 15})
    b = a.assign(score=0.5)
    result = valtest_stat_significance(
        _sampler(a),
        main_column='score',
        extra_samplers={'B': _sampler(b)},
        same_questions=True,
        methods=['wilcoxon', 'bootstrap'],
        bootstrap_iterations=80,
        significance_level=0.05,
    )
    median, mean = result['precomputed']['comparisons']
    assert median['estimate'] == pytest.approx(0.02)
    assert mean['estimate'] == pytest.approx(-0.0325)
    assert median['effect'] == 'Медиана парных разностей'
    assert mean['effect'] == 'Средняя парная разность'
    assert 'разные величины' in median['justification']
    assert result['report']['result_dataframes'][0]['Оцениваемая величина'].tolist() == [
        median['effect'],
        mean['effect'],
    ]


@pytest.mark.parametrize(
    ('key', 'score', 'estimate', 'method'),
    [
        ('f1_score', f1_score, 0.5, 'delta'),
        ('f1_score', partial(f1_score, average='macro'), 0.7, 'bootstrap'),
        ('F1', partial(f1_score, average='macro'), 0.7, 'bootstrap'),
    ],
)
def test_uses_configured_scorer_for_metric_alias(key, score, estimate, method):
    """Distinguish the canonical binary F1 callable from macro F1 under the same key."""
    data = pd.DataFrame({'text': np.arange(120), 'label': [0] * 90 + [1] * 30, 'model_answer': [0] * 110 + [1] * 10})
    sampler = BinaryClassSampler(data)
    scorer = BinaryClassScorer({'f1_score': {'call': score, 'is_singlecol': False}})
    result = valtest_stat_significance(
        sampler, scorer, key, main_column='metric_value', value_threshold=0.6, bootstrap_iterations=40, random_state=7
    )
    (row,) = result['precomputed']['comparisons']
    assert row['estimate'] == pytest.approx(estimate)
    assert row['method_id'] == method


def test_multi_column_mean_preserves_complete_cases():
    """Retain the previous row average when multiple scores are supplied."""
    a = pd.DataFrame({'a': np.linspace(0.2, 0.8, 125), 'b': np.linspace(0.6, 0.9, 125)})
    a.loc[0, 'a'] = np.nan
    result = _comparison(a)
    assert result['n_a'] == 124
    assert result['estimate'] == pytest.approx(a.dropna().mean(axis=1).mean())


def test_standard_mean_respects_scorer_output_selection():
    """Use the selected criterion with the usual Student method for a mean."""
    data = pd.DataFrame({'primary': [1] * 110 + [0] * 10, 'secondary': [0] * 110 + [1] * 10})
    scorer = UniversalScorer(METRICS)
    row = _comparison(data, scorer=scorer, main_metric='single_mean_primary', value_threshold=0.8)
    assert row['method_id'] == 'student'
    assert row['estimate'] == pytest.approx(110 / 120)
    assert row['p_value'] == pytest.approx(stats.ttest_1samp(data['primary'], 0.8, alternative='greater').pvalue)


def _ner_data():
    return pd.DataFrame(
        {
            'id': np.arange(130),
            'text': np.arange(130).astype(str),
            'target': [{'NAME': ['Jane Doe', 'Jane Doe'], 'PLACE': ['Rome']} for _ in range(130)],
            'extracted': [
                {
                    'NAME': ['Jane Doe', 'Jane Doe'] if index % 3 else ['Jane Doe'],
                    'PLACE': ['Milan'] if index % 2 else ['Rome'],
                }
                for index in range(130)
            ],
        }
    )


def test_ner_native_f1_paired():
    """Use native document IDs and entity matches with multiplicity."""
    data = _ner_data()
    baseline = data.copy()
    baseline['extracted'] = [{'NAME': ['Joe'], 'PLACE': ['Rome']} for _ in range(len(data))]
    entity_types = ['NAME', 'PLACE']
    sampler = NerSampler(data, entity_types)
    other = NerSampler(baseline.sample(frac=1, random_state=10), entity_types)
    result = valtest_stat_significance(sampler, metric_agg='f1', extra_samplers={'B': other}, same_questions=True)
    row = result['precomputed']['comparisons'][0]
    expected = [ner_metrics_exact_match_micro_f1(frame, entity_types)['micro_f1'] for frame in (data, baseline)]
    assert row['method_id'] == 'delta'
    assert row['estimate'] == pytest.approx(expected[0] - expected[1])
    assert row['n_a'] == 130


@pytest.mark.parametrize('key', ['ner_exact_match_micro_f1', 'ner_partial_match_micro_f1'])
def test_ner_registered_micro_f1_uses_delta_method(key):
    """Resolve the library's scorer key and match the configured entity comparison."""
    data = _ner_data()
    sampler = NerSampler(data, ['NAME', 'PLACE'])
    scorer = NerScorer({key: NER_METRICS[key]})
    result = valtest_stat_significance(sampler, scorer, key)
    (row,) = result['precomputed']['comparisons']
    assert row['method_id'] == 'delta'
    assert row['metric_a'] == pytest.approx(scorer.calc(sampler)['micro_f1'])


def test_ner_custom_macro_f1_scoring():
    """Resample the native scorer's requested aggregation, including its entity types."""
    data = _ner_data()
    entity_types = ['NAME', 'PLACE']
    scorer = NerScorer({'ner': {'call': ner_metrics_exact_match_macro_f1, 'is_singlecol': False}})
    result = valtest_stat_significance(
        NerSampler(data, entity_types), scorer, 'ner', main_metric='macro_f1', bootstrap_iterations=15
    )
    row = result['precomputed']['comparisons'][0]
    assert row['method_id'] == 'bootstrap'
    assert row['estimate'] == pytest.approx(ner_metrics_exact_match_macro_f1(data, entity_types)['macro_f1'])


def test_version_and_threshold_report_has_excel_cell_types():
    """Allow several hypotheses with missing values in the same report table."""
    a, b = _binary_data()
    a['model_answer'] = 0
    result = valtest_stat_significance(
        _sampler(a),
        metric_agg='accuracy',
        extra_samplers={'B': _sampler(b)},
        same_questions=True,
        metric_thresholds={'precision': 0.7},
        bootstrap_iterations=15,
        significance_level=0.05,
    )
    report = result['report']
    assert set(report) == {'semaphore', 'result_plots', 'result_dataframes'}
    table = report['result_dataframes'][0]
    assert isinstance(table.at[0, '95% ДИ'], str)
    assert isinstance(table.at[0, 'Вывод'], str)
    assert table.at[1, 'p-value'] is None
    assert table.at[1, 'N B'] is None
    assert 'Нулевой знаменатель' in table.at[1, 'Комментарий']


def test_exact_comparison_with_nonzero_margin_uses_object_bootstrap():
    """A Fisher or McNemar equality test cannot establish a nonzero difference margin."""
    a, b = _binary_data()
    row = _comparison(
        a, metric_agg='accuracy', extra_samplers={'B': _sampler(b)}, same_questions=True, comparison_margin=0.05
    )
    assert row['method_id'] == 'bootstrap'
    assert row['reference'] == 0.05
    assert row['alternative'] == 'A − B > 0.05'


@pytest.mark.parametrize(
    ('p_value', 'expected'),
    [
        (0.0, 'green'),
        (np.nextafter(0.01, 0), 'green'),
        (0.01, 'yellow'),
        (np.nextafter(0.05, 0), 'yellow'),
        (0.05, 'yellow'),
        (np.nextafter(0.05, 1), 'red'),
        (1.0, 'red'),
        (None, 'gray'),
    ],
)
def test_semaphore_increment_boundaries(p_value, expected):
    """Apply the disjoint v4 bands, including their exact boundaries."""
    assert _p_value_color(p_value, (0.01, 0.05)) == expected


def test_semaphore_mode_keeps_the_same_p_values():
    """Switch presentation and use the primary result for the overall color."""
    a, _ = _binary_data()
    kwargs = {'metric_agg': 'accuracy', 'value_threshold': 0.7, 'metric_thresholds': {'recall': 0.98}}
    info = valtest_stat_significance(_sampler(a), **kwargs)
    rated = valtest_stat_significance(_sampler(a), is_info=False, **kwargs)
    assert info['report']['semaphore'] == 'gray'
    assert rated['report']['semaphore'] == 'green'
    assert rated['precomputed'] == info['precomputed']
    rated_table = rated['report']['result_dataframes'][0]
    assert rated_table['Результат теста'].tolist() == ['green', 'red']
    pd.testing.assert_frame_equal(rated_table.drop(columns='Результат теста'), info['report']['result_dataframes'][0])


def test_rated_result_is_gray_if_primary_p_value_is_unavailable():
    """Keep the size rule and record a gray row in ordinary mode."""
    a, _ = _binary_data()
    rated = valtest_stat_significance(_sampler(a.iloc[:100]), metric_agg='accuracy', is_info=False)
    assert rated['precomputed']['p_value'] is None
    assert rated['report']['semaphore'] == 'gray'
    assert rated['report']['result_dataframes'][0]['Результат теста'].tolist() == ['gray']


def test_rated_result_accepts_existing_custom_thresholds():
    """Reuse the threshold parameter without affecting calculated significance."""
    a, _ = _binary_data()
    rated = valtest_stat_significance(
        _sampler(a),
        metric_agg='accuracy',
        value_threshold=0.7,
        is_info=False,
        semaphore_threshold=(0.000001, 0.01),
    )
    assert rated['report']['semaphore'] == 'yellow'
