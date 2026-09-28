import pytest
import joblib
import pandas as pd

from llm_val.utils.metrics import METRICS
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.model.MultiThreadGigaModel import MultiThreadGigaModel

from llm_val.val_test.specification_qa_kd.valtest_rubq_from_mmlu_metric import valtest_rubq_from_mmlu_metric

DATA_PATH = 'data/data_mmlu.pkl'


def rewrite_question(question_list):
    question_list = [x for x in question_list]
    return question_list


@pytest.fixture()
def create_scorer_sampler_model(request, api_single_param):
    data = joblib.load(DATA_PATH)
    sampler = MMLUSampler(test=data)
    scorer = UniversalScorer(metrics=METRICS)
    model_params = {'n_jobs': 8}
    model_params.update(api_single_param)
    model = MultiThreadGigaModel(**model_params)
    res = valtest_rubq_from_mmlu_metric(sampler, scorer, model, rewrite_question, 'test')       
    return res


def test_valtest_rubq_from_mmlu_metric_calc(create_scorer_sampler_model):
    test_res = create_scorer_sampler_model
    assert isinstance(test_res, dict)


def test_valtest_rubq_from_mmlu_metric_keys(create_scorer_sampler_model):
    test_res = create_scorer_sampler_model
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_rubq_from_mmlu_metric_report(create_scorer_sampler_model):
    test_res = create_scorer_sampler_model
    report = test_res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)
    assert report['semaphore'] == 'gray'


def test_valtest_rubq_from_mmlu_metric_precomputed(create_scorer_sampler_model):
    test_res = create_scorer_sampler_model
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'metric_value_rubq_from_mmlu'}
    