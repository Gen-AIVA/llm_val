from types import ModuleType, SimpleNamespace
import sys

import pandas as pd
import pytest

from llm_val.val_test.data_quality.annotation_agreement_v4 import (
    ALPHA,
    CLASS_AGREEMENT,
    COHEN,
    FLEISS,
    GWET,
    _color,
    _rate_band,
)
from llm_val.val_test.data_quality.valtest_annotation_quality_agent import valtest_annotation_quality_agent


COLUMN = 'quality_metric'
THRESHOLDS = {'coefficient': (0.4, 0.7), 'class': (0.4, 0.8)}


def _tables(ratings, final=None, *, users=None, initial_count=None):
    records = []
    for session, votes in enumerate(ratings):
        ids = range(len(votes)) if users is None else users[session]
        for index, vote in enumerate(votes):
            row = {'session_id': session, 'assessor_id': ids[index], COLUMN: vote}
            if initial_count is not None:
                row['is_initial'] = index < initial_count
            records.append(row)
    y = None if final is None else pd.DataFrame({COLUMN: final}, index=pd.Index(range(len(final)), name='session_id'))
    return {'X': pd.DataFrame(records), 'y': y}


def _run(tables, **kwargs):
    return valtest_annotation_quality_agent(SimpleNamespace(test=tables), **kwargs)


def _rows(result, check=None):
    return [row for row in result['precomputed']['criteria_stats'] if check is None or row['check'] == check]


@pytest.mark.parametrize(
    ('ratings', 'final', 'assessors', 'expected_method', 'expected'),
    [
        ([[0, 0], [1, 1], [1, 0], [0, 0]], [0, 1, 1, 0], None, COHEN, 0.5),
        ([[0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0]], [0, 1, 1, 0], None, FLEISS, 1 / 3),
        ([[0, 0], [1, 1, 1], [1, 0]], [0, 1, 1], [[1, 2], [2, 3, 4], [3, 4]], ALPHA, 0.5),
        ([[0, 0], [1, 0], [1, 1]], [0, 1, 1], [[1, 2], [2, 3], [1, 3]], ALPHA, 4 / 9),
    ],
)
def test_nominal_selection_and_exact_values(ratings, final, assessors, expected_method, expected):
    """Calculate a single coefficient using the true overlap structure."""
    result = _run(_tables(ratings, final, users=assessors))
    (row,) = [row for row in _rows(result) if row['primary']]
    assert row['check'] == expected_method
    assert row['value'] == pytest.approx(expected)
    assert result['report']['semaphore'] == 'gray'


def test_ordinal_cohen_uses_declared_order_and_weights():
    """Compare weighted discordances with their chance expectation by hand."""
    ratings = [['low', 'low'], ['medium', 'high'], ['high', 'high'], ['low', 'high']]
    options = {COLUMN: {'scale': 'ordinal', 'categories': ['low', 'medium', 'high'], 'weights': 'quadratic'}}
    (row,) = _rows(_run(_tables(ratings), criteria_specs=options))
    observed = (0 + 1 + 0 + 4) / 4
    expected = (2 * 3 * 4 + 1 * 1 + 1 * 3 * 1 + 1 * 1 * 4) / 16
    assert row['check'] == f'{COHEN} (quadratic)'
    assert row['value'] == pytest.approx(1 - observed / expected)


@pytest.mark.parametrize(('scale', 'expected'), [('ordinal', 2 / 7), ('interval', 4 / 29)])
def test_non_nominal_krippendorff(scale, expected):
    """Verify ordinal and interval distances on the same incomplete ratings."""
    ratings = [[0, 0], [1, 4, 4], [0, 4]]
    result = _run(_tables(ratings), criteria_specs={COLUMN: {'scale': scale}})
    (row,) = _rows(result)
    assert row['check'] == f'{ALPHA} ({scale})'
    assert row['value'] == pytest.approx(expected)


@pytest.mark.parametrize(
    ('count', 'expected'),
    [
        (0, 'absent'),
        (1, 'extreme'),
        (2, 'imbalanced'),
        (9, 'imbalanced'),
        (10, 'standard'),
        (190, 'standard'),
        (191, 'imbalanced'),
        (198, 'imbalanced'),
        (199, 'extreme'),
        (200, 'single'),
    ],
)
def test_target_rate_boundaries(count, expected):
    """Respect every inclusive boundary in the increment."""
    assert _rate_band(count, 200) == expected


def test_imbalanced_ac1_and_rare_class_drive_ordinary_mode():
    """Compute AC1 from per-object marginal frequencies and keep class effects separate."""
    sampler = _tables([[0, 0]] * 23 + [[1, 1], [0, 1]], [0] * 24 + [1])
    result = _run(sampler, is_info=False)
    (ac1,) = _rows(result, GWET)
    majority, rare = sorted(_rows(result, CLASS_AGREEMENT), key=lambda row: row['label'])
    chance = 2 * 0.06 * 0.94
    assert ac1['value'] == pytest.approx((0.96 - chance) / (1 - chance))
    assert rare['value'] == 0
    assert rare['target_rate'] == 0.04
    assert rare['primary']
    assert majority['value'] == 1
    assert not majority['primary']
    assert result['report']['semaphore'] == 'red'
    assert _run(sampler)['report']['semaphore'] == 'gray'


def test_imbalanced_ac1_does_not_weight_objects_by_number_of_ratings():
    """Use the mean distribution per object under incomplete overlap."""
    result = _run(_tables([[0, 0]] * 24 + [[0, 1, 1]], [0] * 24 + [1]))
    (row,) = _rows(result, GWET)
    marginal = (2 / 3) / 25
    chance = 2 * marginal * (1 - marginal)
    observed = (24 + 1 / 3) / 25
    assert row['value'] == pytest.approx((observed - chance) / (1 - chance))


def test_final_labels_are_carried_through_sberds_mapping():
    """Accept a keyed y serialized by DataFrame.to_dict."""
    tables = _tables([[0, 0]] * 24 + [[0, 1]], [0] * 24 + [1])
    payload = {name: table.to_dict() for name, table in tables.items()}
    native = _run(tables)['precomputed']
    serialized = valtest_annotation_quality_agent({'test': payload}, exec_env='sberds')['precomputed']
    assert serialized == native


def test_final_label_type_is_aligned_without_creating_extra_categories():
    """Keep FastTrack-style integer finals on the original string scale."""
    tables = _tables([['0', '0']] * 24 + [['0', '1']], [0] * 24 + [1])
    rates = _rows(_run(tables), 'Target rate')
    assert {row['label']: row['target_rate'] for row in rates} == {'0': 0.96, '1': 0.04}
    assert 'Типы финальных меток' in rates[0]['comment']


@pytest.mark.parametrize(
    ('original_votes', 'final', 'initial_count', 'expected'),
    [
        ([[0, 1, 0], [0, 0], [1, 1]], [0, 0, 1], 2, 0.5),
        ([[0, 1, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1]], [0, 0, 0, 1], None, 5 / 9),
    ],
)
def test_class_agreement_examples(original_votes, final, initial_count, expected):
    """Retain the two coefficients from the increment in the imbalanced branch."""
    # Padding preserves class-0 agreement and pushes the two examples into the imbalance branch.
    padding = 47 if initial_count == 2 else 71
    votes = original_votes + [[1, 1, 1] if initial_count is None else [1, 1]] * padding
    result = _run(_tables(votes, final + [1] * padding, initial_count=initial_count))
    classes = {row['label']: row['value'] for row in _rows(result, CLASS_AGREEMENT)}
    assert classes == pytest.approx({'0': expected, '1': 1.0})


def test_extreme_excludes_both_tails_and_retains_non_extreme_class():
    """Calculate within-class agreement only for eligible classes."""
    final = [0] * 95 + [1] * 4 + [2] + [0] * 100
    result = _run(_tables([[label, label] for label in final], final))
    rows = {row['label']: row for row in _rows(result, CLASS_AGREEMENT)}
    assert rows['2']['target_rate'] == 0.005
    assert rows['2']['value'] is None
    assert rows['0']['target_rate'] == 0.975
    assert rows['0']['value'] == 1
    assert rows['1']['primary']
    assert rows['1']['value'] == 1
    assert not _rows(result, GWET)


def test_absent_class_is_undefined():
    """Differentiate an absent class from a class with zero agreement."""
    result = _run(_tables([[0, 0], [0, 1]], [0, 0]), criteria_specs={COLUMN: {'categories': [0, 1]}})
    rows = {row['label']: row for row in _rows(result, CLASS_AGREEMENT)}
    assert rows['1']['value'] is None
    assert rows['1']['target_rate'] == 0
    assert rows['0']['value'] == 0.5
    assert rows['0']['target_rate'] == 1


def test_novel_adjudication_changes_main_overlap():
    """Ignore repeated adjudication labels and count novel ones jointly."""
    ratings = [[0, 1, 0], [0, 0], [1, 1], [1, 1]]
    paired = _run(_tables(ratings, [0, 0, 1, 1], initial_count=2))
    assert _rows(paired, COHEN)[0]['n_assessors'] == 2
    ratings[0][2] = 2
    partial = _run(_tables(ratings, [0, 0, 1, 1], initial_count=2))
    (main,) = _rows(partial, ALPHA)
    assert main['overlap'] == 'неполное'
    assert main['n_assessors'] == 3
    assert main['max_overlap'] == 2
    assert 'Новых меток арбитров: 1' in main['comment']


def test_novel_adjudication_never_enters_class_agreement():
    """Compute class-level agreement using the initial pair only."""
    result = _run(_tables([[0, 0]] * 24 + [[1, 1, 2]], [0] * 24 + [1], initial_count=2))
    (rare,) = [row for row in _rows(result, CLASS_AGREEMENT) if row['label'] == '1']
    assert rare['value'] == 1
    assert _rows(result, GWET)[0]['value'] < 1


def test_dynamic_order_and_unavailable_metadata():
    """Respect an explicit rating order and expose missing dynamic metadata."""
    tables = _tables([[0, 1, 0], [0, 0], [1, 1]], [0, 0, 1])
    unmarked = _run(tables, dynamic_overlap=True, is_info=False)
    assert unmarked['report']['semaphore'] == 'gray'
    assert 'порядок' in _rows(unmarked)[0]['comment']
    ratings = tables['X'].assign(sequence=tables['X']['assessor_id']).sample(frac=1, random_state=3)
    ordered = _run(dict(tables, X=ratings), rating_order_column='sequence')
    assert _rows(ordered, COHEN)
    conventional = ratings.rename(columns={'sequence': 'rating_order'})
    assert _rows(_run(dict(tables, X=conventional)), COHEN)


def test_dynamic_flags_can_mark_more_than_two_independent_ratings():
    """Respect each explicit primary decision, even with three initial raters."""
    ratings = [[0, 0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0]]
    result = _run(_tables(ratings, [0, 1, 1, 0], initial_count=3))
    (row,) = _rows(result, FLEISS)
    assert row['value'] == pytest.approx(1 / 3)
    assert row['max_overlap'] == 3


def test_ordered_adjudicator_cannot_replace_a_missing_initial_rating():
    """Do not promote rating number three when the second score is missing."""
    tables = _tables([[0, None, 1], [0, 0], [1, 1]], [0, 0, 1])
    tables['X']['rating_order'] = tables['X']['assessor_id']
    result = _run(tables, is_info=False)
    assert result['report']['semaphore'] == 'gray'
    assert 'менее чем двумя независимыми оценками: 1 из 3' in _rows(result)[0]['comment']


@pytest.mark.parametrize(
    ('ratings', 'users', 'message'),
    [
        ([[0, 1], [0]], None, 'менее чем двумя'),
        ([[0, 1], [0, 1]], [[0, 0], [0, 1]], 'несколько разных оценок'),
    ],
)
def test_invalid_ratings_report_reason(ratings, users, message):
    """Provide a gray row when overlap or assessor identity cannot be established."""
    result = _run(_tables(ratings, [0, 1], users=users), is_info=False)
    assert result['report']['semaphore'] == 'gray'
    assert message in _rows(result)[0]['comment']


def test_multilabel_per_label_and_jaccard():
    """Distinguish valid empty sets from missing labels."""
    ratings = [[[], []], [['x'], ['x', 'y']], [['y'], ['y']]]
    tables = _tables(ratings, [[], ['x'], ['y']])
    by_label = _run(tables, criteria_specs={COLUMN: {'scale': 'multilabel'}})
    assert {row['label'] for row in _rows(by_label) if row['primary']} == {'x', 'y'}
    by_set = _run(tables, criteria_specs={COLUMN: {'scale': 'multilabel', 'multilabel_method': 'jaccard'}})
    (row,) = _rows(by_set)
    assert row['check'] == f'{ALPHA} (Jaccard)'
    assert row['value'] == pytest.approx(1 - 0.5 / 2.3)


def test_jaccard_uses_ratings_and_checks_the_declared_categories():
    """The set distance ignores final labels and honors the agreed domain."""
    tables = _tables([[[], ['x']], [['x'], ['x']]], [['x'], ['x']])
    tables['y'].index = pd.Index([0, 0], name='session_id')
    options = {'scale': 'multilabel', 'multilabel_method': 'jaccard', 'categories': ['x']}
    assert _rows(_run(tables, criteria_specs={COLUMN: options}), f'{ALPHA} (Jaccard)')
    options['categories'] = ['y']
    (bad,) = _rows(_run(tables, criteria_specs={COLUMN: options}))
    assert 'вне заданного набора' in bad['comment']


def test_multilabel_adjudicator_novelty_is_rechecked_for_each_label():
    """A new label set can repeat a vote for x while introducing one for y."""
    ratings = [[['x'], [], ['x', 'y']], [['x'], ['x']], [['y'], ['y']], [[], []]]
    result = _run(_tables(ratings, [['x'], ['x'], ['y'], []], initial_count=2))
    main = {row['label']: row for row in _rows(result) if row['primary']}
    assert main['x']['check'] == COHEN
    assert main['x']['n_assessors'] == 2
    assert main['y']['check'] == ALPHA
    assert main['y']['n_assessors'] == 3


@pytest.fixture
def dawid_skene_inputs(monkeypatch):
    """Record votes and return a deterministic decision for each task."""
    crowdkit = ModuleType('crowdkit')
    module = ModuleType('crowdkit.aggregation')
    inputs = []

    class DawidSkeneFixture:
        def __init__(self, n_iter=100):
            self.n_iter = n_iter

        def fit_predict(self, ratings):
            assert list(ratings) == ['task', 'worker', 'label']
            inputs.append(ratings.copy())
            return ratings.groupby('task')['label'].last()

    module.DawidSkene = DawidSkeneFixture
    monkeypatch.setitem(sys.modules, 'crowdkit', crowdkit)
    monkeypatch.setitem(sys.modules, 'crowdkit.aggregation', module)
    return inputs


def test_dawid_skene_receives_all_ratings_when_final_is_missing(dawid_skene_inputs):
    """Arbitration affects final labels while agreement uses the initial ratings."""
    votes = [['no', 'yes', 'no'], ['yes', 'no', 'yes'], ['yes', 'yes'], ['no', 'no']]
    result = _run(_tables(votes, initial_count=2))
    (ratings,) = dawid_skene_inputs
    assert len(ratings) == 10
    assert ratings.loc[ratings.worker == 2, 'task'].tolist() == [0, 1]
    assert set(ratings['label']) == {0, 1}
    (row,) = _rows(result, COHEN)
    assert row['value'] == pytest.approx(0.0)
    assert {row['label']: row['target_rate'] for row in _rows(result, 'Target rate')} == {'no': 0.5, 'yes': 0.5}
    assert 'Дэвидом–Скином' in _rows(result, 'Target rate')[0]['comment']


@pytest.mark.parametrize(
    ('first', 'second', 'expected'),
    [
        ('low', 'high', ['low', 'high']),
        (0.25, 0.75, [0.25, 0.75]),
        (['x'], ['x', 'y'], [frozenset({'x'}), frozenset({'x', 'y'})]),
        ('0', '1', [0, 1]),
        ('01', '1', ['01', '1']),
    ],
)
def test_fasttrack_sampler_preserves_rating_categories(dawid_skene_inputs, first, second, expected):
    """Decode the aggregated class without coercing the original scale to integers."""
    from llm_val.sampler.FastTrackSampler import FastTrackSampler

    wide = pd.DataFrame(
        {
            'session_id': [17, 25],
            'output_answer': ['', ''],
            f'{COLUMN}_asessor_1': [first, second],
            f'{COLUMN}_asessor_2': [first, second],
        }
    )
    sampler = FastTrackSampler(wide)
    assert sampler.test['y'][COLUMN].to_dict() == {17: expected[0], 25: expected[1]}
    assert sampler.test['y'].index.name == 'session_id'
    (ratings,) = dawid_skene_inputs
    assert set(ratings['label']) == {0, 1}


def test_fasttrack_sampler_keeps_first_criterion_sessions(dawid_skene_inputs):
    """Extra criteria cannot add missing primary scores to an existing basket."""
    from llm_val.sampler.FastTrackSampler import FastTrackSampler

    wide = pd.DataFrame(
        {
            'session_id': [17, 25, 49],
            'output_answer': ['', '', ''],
            'final_metric_asessor_1': [1, 1, None],
            'final_metric_asessor_2': [1, 1, None],
            f'{COLUMN}_asessor_1': [0, 1, 1],
            f'{COLUMN}_asessor_2': [0, 1, 1],
        }
    )
    scores = FastTrackSampler(wide).test['y']
    assert scores.index.tolist() == [17, 25]
    assert scores['final_metric'].tolist() == [1, 1]
    assert scores[COLUMN].tolist() == [0, 1]


def test_multilabel_final_label_sources_use_same_aggregation(dawid_skene_inputs):
    """The fallback aggregates the same sets as the stock sampler."""
    from llm_val.sampler.FastTrackSampler import FastTrackSampler

    votes = [['x'], ['y'], ['x', 'y']]
    wide = pd.DataFrame(
        {
            'session_id': [0, 1, 2],
            'output_answer': ['', '', ''],
            f'{COLUMN}_asessor_1': votes,
            f'{COLUMN}_asessor_2': votes,
        }
    )
    sampler = FastTrackSampler(wide)
    saved = _run(sampler.test)
    calculated = _run({'X': sampler.test['X']})

    sampler_input, fallback_input = dawid_skene_inputs
    pd.testing.assert_frame_equal(sampler_input.reset_index(drop=True), fallback_input.reset_index(drop=True))
    assert [(row['label'], row['target_rate']) for row in _rows(saved, 'Target rate')] == [
        (row['label'], row['target_rate']) for row in _rows(calculated, 'Target rate')
    ]


@pytest.mark.parametrize(
    ('check', 'value', 'color'),
    [
        (COHEN, 0.4, 'red'),
        (COHEN, 0.7, 'yellow'),
        (COHEN, 0.71, 'green'),
        (GWET, 0.4, 'red'),
        (GWET, 0.8, 'yellow'),
        (GWET, 0.81, 'green'),
        (CLASS_AGREEMENT, None, 'gray'),
    ],
)
def test_color_boundaries(check, value, color):
    """Use a red inclusive lower bound and a strict green bound."""
    assert _color(value, check, THRESHOLDS) == color


@pytest.mark.parametrize(
    ('votes', 'final', 'value', 'color'),
    [
        ([[0, 0, 0]] * 2 + [[0, 0, 1]] * 2 + [[1, 1, 1]], [0, 0, 0, 0, 1], 0.4, 'red'),
        ([[0, 0, 0]] * 3 + [[0, 1, 1]] + [[1, 1, 1]], [0, 0, 0, 1, 1], 0.7, 'yellow'),
    ],
)
def test_exact_color_boundaries_after_fleiss_arithmetic(votes, final, value, color):
    """Rounding in a computed coefficient cannot move an exact boundary."""
    result = _run(_tables(votes, final), is_info=False)
    assert _rows(result, FLEISS)[0]['value'] == pytest.approx(value)
    assert result['report']['semaphore'] == color
