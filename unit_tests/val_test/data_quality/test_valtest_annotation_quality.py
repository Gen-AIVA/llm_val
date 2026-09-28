import pytest
import joblib
import pandas as pd

from llm_val.sampler.LongAnswerSampler import LongAnswerSampler
from llm_val.val_test.data_quality.valtest_annotation_quality import valtest_annotation_quality

DATA_PATH = 'data/data_longanswer.pkl'

TEST_PARAMS = [(valtest_annotation_quality, {'data_type': 'test'})]


@pytest.fixture()
def create_sampler(request):
    data = joblib.load(DATA_PATH)
    sampler = LongAnswerSampler(test=data)
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
    assert set(precomputed.keys()) == {'spearman_stat', 'krippendorf_stat'}