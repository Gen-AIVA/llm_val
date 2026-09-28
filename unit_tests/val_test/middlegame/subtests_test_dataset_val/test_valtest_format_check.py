import re
import uuid

import pytest
import numpy as np
import pandas as pd

from llm_val.sampler.Sampler import Sampler
import llm_val.utils.dataset_type_match as dt
from llm_val.sampler.TracesSampler2 import TracesSampler2
from llm_val.val_test.middlegame.subtests_test_dataset_val.valtest_format_check import valtest_format_check

DATA_PATH_1 = 'data/data_traces_short.parquet'
DATA_PATH_2 = 'data/data_ft_full.parquet'

PARAMS = [(['input_query_count', 'scenario', 'assessor_id', 'reference_answer', 'solution_version'], False),
          (['scenario'], False),
          ([], False),
          (['input_query_count', 'scenario', 'assessor_id', 'reference_answer', 'solution_version'], True),
          ([], True)]

@pytest.fixture(params=PARAMS)
def create_sampler(request):
    optional_columns, is_info = request.param
    data_traces2 = pd.read_parquet(DATA_PATH_1)
    data_ft_full = pd.read_parquet(DATA_PATH_2)
    sampler_traces2 = TracesSampler2(data_ft_full, data_traces2)
    res = valtest_format_check(sampler = sampler_traces2,
                               optional_columns = optional_columns,
                               is_info = is_info)
    return res, is_info, optional_columns


def test_valtest_metric_report(create_sampler):
    test_res, is_info, optional_columns = create_sampler
    report = test_res['report']
    if is_info:
        true_color = 'gray'
    elif len(optional_columns) > 1:
        true_color = 'green'
    else:
        true_color = 'red'
    assert report['semaphore'] == true_color


def test_valtest_metric_precomputed(create_sampler):
    test_res, is_info, optional_columns = create_sampler
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'violation'}


class SimpleSampler(Sampler):
    def __init__(self, test):
        test = self._preproc(test)
        super().__init__(test=test) 
    def _preproc(self, var):
        result = {'X' : var, 'y' : pd.Series(np.zeros(var.shape[0]))}
        return result


def test_x_types_positive_no_violations():
    df = pd.DataFrame({
        'scenario':            ['rag', 'qa', 'rag'],
        'session_id':          [1, 2, 3],
        'input_query':         ['hi', 'how are you', 'q'],
        'faithfulness_metric': [0.9, 0.5, None],
    })
    sampler = SimpleSampler(df)
    violations = valtest_format_check(sampler)['precomputed']['violation']
    assert violations['type'] == {}
    assert violations['NA'] == {}


def test_x_types_negative_violations():
    df = pd.DataFrame({
        'scenario':            ['rag', None, 'qa'],
        'session_id':          [1, 'bad', 2.5],
        'faithfulness_metric': [0.9, 'x', 0.3],
    })
    sampler = SimpleSampler(df)
    violations = valtest_format_check(sampler)['precomputed']['violation']
    assert violations['NA']['scenario'] == [1]
    assert violations['type']['session_id'] == [1, 2]
    assert violations['type']['faithfulness_metric'] == [1]


def test_x_types_unknown_column_ignored():
    df = pd.DataFrame({
        'scenario':    ['rag'],
        'unknown_col': [object()],
    })
    sampler = SimpleSampler(df)
    violations = valtest_format_check(sampler)['precomputed']['violation']
    assert 'unknown_col' not in violations['NA']
    assert 'unknown_col' not in violations['type']


def test_x_types_pattern_none_allowed():
    df = pd.DataFrame({
        'scenario':        ['rag', 'qa'],
        'accuracy_metric': [None, None],
    })
    sampler = SimpleSampler(df)
    violations = valtest_format_check(sampler)['precomputed']['violation']
    assert 'accuracy_metric' not in violations['NA']
    assert 'accuracy_metric' not in violations['type']


def test_x_types_exact_none_forbidden():
    df = pd.DataFrame({'scenario': [None]})
    sampler = SimpleSampler(df)
    violations = valtest_format_check(sampler)['precomputed']['violation']
    assert violations['NA']['scenario'] == [0]


def test_x_types_session_id_accepts_uuid_string():
    df = pd.DataFrame({'session_id': ['550e8400-e29b-41d4-a716-446655440000']})
    sampler = SimpleSampler(df)
    violations = valtest_format_check(sampler)['precomputed']['violation']
    assert 'session_id' not in violations['NA']
    assert 'session_id' not in violations['type']


def test_x_types_session_id_accepts_uuid_object():
    df = pd.DataFrame({'session_id': [uuid.uuid4()]})
    sampler = SimpleSampler(df)
    violations = valtest_format_check(sampler)['precomputed']['violation']
    assert 'session_id' not in violations['NA']
    assert 'session_id' not in violations['type']


def test_x_types_int_as_float_accepted():
    df = pd.DataFrame({'session_id': [1.0, 2.0, 3.0]})
    sampler = SimpleSampler(df)
    violations = valtest_format_check(sampler)['precomputed']['violation']
    assert 'session_id' not in violations['NA']
    assert 'session_id' not in violations['type']


def test_x_types_nonintegral_float_rejected():
    df = pd.DataFrame({'session_id': [1.5]})
    sampler = SimpleSampler(df)
    violations = valtest_format_check(sampler)['precomputed']['violation']
    assert violations['type']['session_id'] == [0]


def test_dialogue_types_positive_no_violations():
    df = pd.DataFrame({
        'session_id': [1, 2],
        'dialogue':   [[('q1', 'a1', 'r1')],
                       [('q1', 'a1', 'r1'), ('q2', 'a2', 'r2')]],
    })
    sampler = SimpleSampler(df)
    violations = valtest_format_check(sampler)['precomputed']['violation']
    assert violations['type'] == {}
    assert violations['NA'] == {}


def test_dialogue_types_negative_violations():
    df = pd.DataFrame({
        'session_id': [1, 2, 3],
        'dialogue':   [[('q', 'a')],
                       None,
                       'not a list'],
    })
    sampler = SimpleSampler(df)
    violations = valtest_format_check(sampler)['precomputed']['violation']
    assert violations['NA']['dialogue'] == [1]
    assert violations['type']['dialogue'] == [0, 2]


def test_dialogue_types_turn_length_checked():
    df = pd.DataFrame({
        'session_id': [1, 2],
        'dialogue':   [[('q', 'a')],
                       [('q', 'a', 'r')]],
    })
    sampler = SimpleSampler(df)
    violations = valtest_format_check(sampler)['precomputed']['violation']
    assert violations['type']['dialogue'] == [0]


def test_dialogue_types_metric_pattern_none_allowed():
    df = pd.DataFrame({
        'session_id':       [1],
        'dialogue':         [[('q', 'a', 'r')]],
        'coherence_metric': [None],
    })
    sampler = SimpleSampler(df)
    violations = valtest_format_check(sampler)['precomputed']['violation']
    assert 'coherence_metric' not in violations['NA']
    assert 'coherence_metric' not in violations['type']


def test_string_accepts_str():
    assert dt.STRING.check('hello')


def test_string_rejects_int():
    assert not dt.STRING.check(1)


def test_string_rejects_none():
    assert not dt.STRING.check(None)
 

def test_int_accepts_int():
    assert dt.INT.check(42)


def test_int_accepts_integral_float():
    assert dt.INT.check(3.0)


def test_int_rejects_nonintegral_float():
    assert not dt.INT.check(2.5)


def test_int_rejects_bool():
    assert not dt.INT.check(True)


def test_int_rejects_np_bool():
    assert not dt.INT.check(np.bool_(True))


def test_int_rejects_string():
    assert not dt.INT.check('1')


def test_float_accepts_float():
    assert dt.FLOAT.check(0.5)


def test_float_accepts_int():
    assert dt.FLOAT.check(1)


def test_float_rejects_bool():
    assert not dt.FLOAT.check(False)


def test_float_rejects_string():
    assert not dt.FLOAT.check('1.0')


def test_uuid_accepts_uuid_object():
    assert dt.UUID.check(uuid.uuid4())


def test_uuid_accepts_uuid_string():
    assert dt.UUID.check('550e8400-e29b-41d4-a716-446655440000')


def test_uuid_rejects_bad_string():
    assert not dt.UUID.check('not-a-uuid')


def test_uuid_rejects_int():
    assert not dt.UUID.check(123)


def test_union_accepts_first_match():
    u = dt.Union(dt.UUID, dt.INT)
    assert u.check(5)


def test_union_accepts_second_match():
    u = dt.Union(dt.UUID, dt.INT)
    assert u.check('550e8400-e29b-41d4-a716-446655440000')


def test_union_rejects_no_match():
    u = dt.Union(dt.UUID, dt.INT)
    assert not u.check('plain string')


def test_listof_accepts_list():
    assert dt.ListOf().check([1, 2, 3])


def test_listof_accepts_tuple():
    assert dt.ListOf().check((1, 2))


def test_listof_rejects_string():
    assert not dt.ListOf().check('abc')


def test_listof_rejects_int():
    assert not dt.ListOf().check(42)


def test_listof_length_exact_match():
    assert dt.ListOf(length=3).check([1, 2, 3])


def test_listof_length_mismatch():
    assert not dt.ListOf(length=3).check([1, 2])


def test_listof_nested_all_valid():
    spec = dt.ListOf(dt.ListOf(length=3))
    assert spec.check([('a', 'b', 'c'), ('d', 'e', 'f')])


def test_listof_nested_inner_length_violation():
    spec = dt.ListOf(dt.ListOf(length=3))
    assert not spec.check([('a', 'b')])


def test_pattern_to_regex_matches_metric():
    regex = dt._pattern_to_regex('[metric_name]_metric')
    assert regex.match('faithfulness_metric')
    assert regex.match('accuracy_metric')


def test_pattern_to_regex_no_suffix_leak():
    regex = dt._pattern_to_regex('[metric_name]_metric')
    assert not regex.match('accuracy_metric_raw')


def test_pattern_to_regex_double_placeholder():
    regex = dt._pattern_to_regex('[module_name]_[metric_name]_metric')
    assert regex.match('rag_faithfulness_metric')
    assert not regex.match('rag_faithfulness_metric_extra')


def test_compile_schema_separates_exact_and_patterns():
    schema = {'scenario': dt.STRING, '[metric_name]_metric': dt.FLOAT}
    exact, patterns = dt._compile_schema(schema)
    assert 'scenario' in exact
    assert '[metric_name]_metric' not in exact
    assert len(patterns) == 1


def test_compile_schema_all_exact():
    schema = {'a': dt.STRING, 'b': dt.INT}
    exact, patterns = dt._compile_schema(schema)
    assert len(exact) == 2
    assert patterns == []


def test_resolve_spec_exact_found():
    exact = {'scenario': dt.STRING}
    spec, allow_none = dt._resolve_spec('scenario', exact, [])
    assert spec is dt.STRING
    assert allow_none is False


def test_resolve_spec_exact_priority_over_pattern():
    exact = {'scenario': dt.STRING}
    patterns = [(re.compile(r'(.+)$'), dt.FLOAT)]
    spec, allow_none = dt._resolve_spec('scenario', exact, patterns)
    assert spec is dt.STRING
    assert allow_none is False


def test_resolve_spec_pattern_match():
    exact = {}
    patterns = [(re.compile(r'(.+)_metric$'), dt.FLOAT)]
    spec, allow_none = dt._resolve_spec('accuracy_metric', exact, patterns)
    assert spec is dt.FLOAT
    assert allow_none is True


def test_resolve_spec_unknown_returns_none():
    spec, allow_none = dt._resolve_spec('unknown', {}, [])
    assert spec is None
    assert allow_none is False