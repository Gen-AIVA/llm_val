import pytest
import joblib
import pandas as pd

from llm_val.utils.metrics import METRICS
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.sampler.RuBQSampler import RuBQSampler
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.sampler.LongAnswerSampler import LongAnswerSampler


from llm_val.val_test.model_quality.valtest_metric import valtest_metric
from llm_val.val_test.model_quality.valtest_answer_unique import valtest_answer_unique
from llm_val.val_test.model_quality.valtest_metric_by_thema import valtest_metric_by_thema
from llm_val.val_test.model_quality.valtest_stat_significance import valtest_stat_significance
from llm_val.val_test.middlegame.valtest_metric_sens_by_size import valtest_metric_sens_by_size
from llm_val.val_test.data_quality.valtest_complex_data_quality import valtest_complex_data_quality
from llm_val.val_test.stability.valtest_metric_stability import valtest_metric_stability
from llm_val.val_test.model_quality.valtest_metric_by_answer_size import valtest_metric_by_answer_size
from llm_val.val_test.specification_extract.valtest_needle_test import valtest_needle_test

DATA_PATH_RUBQ = 'data/data_rubq.pkl'
DATA_PATH_MMLU = 'data/data_mmlu.pkl'
DATA_PATH_LA = 'data/data_longanswer.pkl'

SAMPLER_PARAMS = [(RuBQSampler, DATA_PATH_RUBQ),
                  (MMLUSampler, DATA_PATH_MMLU),
                  (LongAnswerSampler, DATA_PATH_LA)]

SCORER_PARAMS = [(UniversalScorer, METRICS)]

TEST_PARAMS = [(valtest_metric, {'data_type': 'test', 'main_metric': 'multicol_mean', 'metric_agg': 'multicol_mean'}),
               (valtest_metric_by_thema, {'data_type': 'test', 'main_metric': 'multicol_mean', 'metric_agg': 'multicol_mean'}),
               (valtest_stat_significance, {'data_type': 'test'}), 
               (valtest_answer_unique, {'data_type': 'test'}),
               (valtest_metric_sens_by_size, {'data_type': 'test'}),
               (valtest_metric_stability, {'main_metric': 'multicol_mean', 'metric_agg': 'multicol_mean', 'data_types': ['test', 'test']}),
               (valtest_complex_data_quality, {'data_types': ['test', 'test']}), 
               (valtest_metric_by_answer_size, {'data_type': 'test'})]


@pytest.fixture(params=SAMPLER_PARAMS)
def create_sampler(request):
    sampler_class, data_path = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    return sampler


@pytest.fixture(params=SCORER_PARAMS)
def create_scorer(request):
    scorer_class, metrics = request.param
    scorer = scorer_class(metrics=metrics)
    return scorer


@pytest.fixture(params=TEST_PARAMS)
def create_test(request, create_sampler, create_scorer):
    sampler = create_sampler
    scorer = create_scorer
    test, params = request.param
    test_res = test(sampler=sampler, scorer=scorer, **params)
    return test_res


def test_valtest_dict(create_test):
    test_res = create_test
    assert isinstance(test_res, dict)


def test_valtest_keys(create_test):
    test_res = create_test
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_metric_report(create_test):
    test_res = create_test
    report = test_res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)


def test_valtest_metric_precomputed(create_test):
    test_res = create_test
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)