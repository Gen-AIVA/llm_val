import pytest
import joblib
import pandas as pd

from llm_val.sampler.DialogueSampler import DialogueSampler
from llm_val.val_test.specification_chat.valtest_dialogue_length import valtest_dialogue_length

DATA_PATH = 'data/joy_dataset.pkl'

TEST_PARAMS = [(valtest_dialogue_length, {'data_type': 'test'})]


@pytest.fixture()
def create_sampler(request):
    data = joblib.load(DATA_PATH)
    sampler = DialogueSampler(test=data)
    return sampler

    
@pytest.fixture(params=TEST_PARAMS)
def create_test(request, create_sampler):
    sampler = create_sampler
    test, params = request.param
    test_res = test(sampler=sampler, **params)
    return test_res


def test_valtest_dict(create_test):
    test_res = create_test
    assert isinstance(test_res, dict)


def test_valtest_keys(create_test):
    test_res = create_test
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_report(create_test):
    test_res = create_test
    report = test_res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)


def test_valtest_borders(create_sampler):
    sampler = create_sampler
    precomputed_right = {'avg_length': 5,
                         'dialogue_lengths': pd.Series([5])}
    res = valtest_dialogue_length(sampler=sampler,
                                  conversational=True,
                                  precomputed=precomputed_right)
    assert res['report']['semaphore'] == 'yellow'
    precomputed_left = {'avg_length': 3,
                        'dialogue_lengths': pd.Series([3])}
    res = valtest_dialogue_length(sampler=sampler,
                                  conversational=True,
                                  precomputed=precomputed_left)
    assert res['report']['semaphore'] == 'yellow'
    res = valtest_dialogue_length(sampler=sampler,
                                  conversational=False,
                                  precomputed=precomputed_right)
    assert res['report']['semaphore'] == 'yellow'
    res = valtest_dialogue_length(sampler=sampler,
                                  conversational=False,
                                  precomputed=precomputed_left)
    assert res['report']['semaphore'] == 'yellow'
