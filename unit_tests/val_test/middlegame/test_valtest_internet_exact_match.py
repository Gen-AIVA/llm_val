import pytest
import joblib
import pandas as pd

from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.val_test.middlegame.valtest_internet_exact_match import valtest_internet_exact_match

DATA_PATH = 'data/data_mmlu.pkl'

@pytest.fixture()
def create_sampler():
    data = joblib.load(DATA_PATH)
    sampler = MMLUSampler(test=data)
    res = valtest_internet_exact_match(sampler=sampler,
                                       data_type='test')
    return res


def test_valtest_internet_exact_match_calc(create_sampler):
    test_res = create_sampler
    assert isinstance(test_res, dict)


def test_valtest_internet_exact_match_keys(create_sampler):
    test_res = create_sampler
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_internet_exact_match_report(create_sampler):
    test_res = create_sampler
    report = test_res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)
    assert report['semaphore'] == 'yellow'


def test_valtest_internet_exact_match_precomputed(create_sampler):
    test_res = create_sampler
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'found_frac_without_answers', 'found_frac_with_answers', 'failed_questions', 'failed_pairs'}
    assert precomputed['found_frac_without_answers'] == 0.2
    assert precomputed['found_frac_with_answers'] == 0.0
