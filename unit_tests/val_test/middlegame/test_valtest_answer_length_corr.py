import pytest
import joblib
import pandas as pd

from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.val_test.middlegame.valtest_answer_length_corr import valtest_answer_length_corr

DATA_PATH = 'data/data_mmlu.pkl'

TEST_PARAMS = [(valtest_answer_length_corr, {'data_type': 'test',
                                             'semaphore_threshold_spear': 0.3,
                                             'semaphore_threshold_pval': 0.05})]


@pytest.fixture()
def create_sampler(request):
    data = joblib.load(DATA_PATH)
    sampler = MMLUSampler(test=data)
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


def test_valtest_annotation_quality_report(create_test):
    test_res = create_test
    report = test_res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)


def test_valtest_annotation_quality_precomputed(create_test):
    test_res = create_test
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'spearman_stat', 'spearman_pval', 'is_correct'}
