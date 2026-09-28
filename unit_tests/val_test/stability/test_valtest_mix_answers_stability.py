import pytest
import joblib
import pandas as pd

from llm_val.utils.metrics import METRICS
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.model.MultiThreadGigaModel import MultiThreadGigaModel

from llm_val.val_test.stability.valtest_mix_answers_stability import valtest_mix_answers_stability

DATA_SHORT_PATH = 'data/data_mmlu.pkl'
DATA_LONG_PATH = 'data/data_mmlu_long.pkl'


@pytest.fixture()
def create_scorer_model(request, api_single_param):
    scorer = UniversalScorer(metrics=METRICS)
    model_params = {'n_jobs': 8}
    model_params.update(api_single_param)
    model = MultiThreadGigaModel(**model_params)
    return scorer, model


def test_valtest_mix_answers_stability_res(create_scorer_model):
    scorer, model = create_scorer_model
    data = joblib.load(DATA_SHORT_PATH)
    sampler = MMLUSampler(test=data)
    test_res = valtest_mix_answers_stability(sampler,
                                             scorer,
                                             model,
                                             'test',
                                             bootstrap_iter=100)
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}
    
    
def test_valtest_mix_answers_stability_report(create_scorer_model):
    scorer, model = create_scorer_model
    data = joblib.load(DATA_SHORT_PATH)
    sampler = MMLUSampler(test=data)
    test_res = valtest_mix_answers_stability(sampler,
                                             scorer,
                                             model,
                                             'test',
                                             bootstrap_iter=100)    
    report = test_res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    
    
def test_valtest_mix_answers_stability_precomputed(create_scorer_model):
    scorer, model = create_scorer_model
    data = joblib.load(DATA_SHORT_PATH)
    sampler = MMLUSampler(test=data)
    test_res = valtest_mix_answers_stability(sampler,
                                             scorer,
                                             model,
                                             'test',
                                             bootstrap_iter=100)
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert 'mix_answers_metric_mean' in precomputed
    assert 'mix_answers_metric_std' in precomputed


def test_valtest_mix_answers_stability_big_std(create_scorer_model):
    scorer, model = create_scorer_model
    data = joblib.load(DATA_SHORT_PATH)
    sampler = MMLUSampler(test=data)
    test_res = valtest_mix_answers_stability(sampler,
                                             scorer,
                                             model,
                                             'test',
                                             bootstrap_iter=100)
    report = test_res['report']
    assert report['semaphore'] == 'yellow'

    
def test_valtest_mix_answers_stability_small_std(create_scorer_model):
    scorer, model = create_scorer_model
    data = joblib.load(DATA_LONG_PATH)[:50]
    sampler = MMLUSampler(test=data)
    test_res = valtest_mix_answers_stability(sampler,
                                             scorer,
                                             model,
                                             'test',
                                             bootstrap_iter=100)
    report = test_res['report']
    assert report['semaphore'] == 'green'
