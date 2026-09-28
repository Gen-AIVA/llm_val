import pytest
import joblib
import pandas as pd

from llm_val.model.MultiThreadGigaModel import MultiThreadGigaModel
from llm_val.val_test.specification_chat.valtest_length_self_chat import valtest_length_self_chat

MODEL = [(MultiThreadGigaModel, {'n_jobs': 8})]


@pytest.fixture(params=MODEL)
def create_model(request, api_param):
    model_class, params = request.param
    params.update(api_param)
    return model_class(model='GigaChat:latest', **params)


def test_valtest_length_self_chat_result(create_model):
    model = create_model
    res = valtest_length_self_chat(model=model,
                                   generated_length=7,
                                   context=3,
                                   n_samples=5,
                                   seed=42,
                                   computed=None)
    assert isinstance(res, dict)
    assert set(res.keys()) == {'report', 'precomputed'}


def test_valtest_length_self_chat_report(create_model):
    model = create_model
    res = valtest_length_self_chat(model=model,
                                   generated_length=7,
                                   context=3,
                                   n_samples=5,
                                   seed=42,
                                   computed=None)
    report = res['report']
    assert isinstance(report, dict)
    assert set(report.keys()) == {'semaphore', 'result_dataframes', 'result_plots'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)
    assert report['semaphore'] == 'gray'


def test_valtest_length_self_chat_precomputed(create_model):
    model = create_model
    res = valtest_length_self_chat(model=model,
                                   generated_length=7,
                                   context=3,
                                   n_samples=5,
                                   seed=42,
                                   computed=None)
    precomputed = res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'count_unique', 'count_repeat', 'cycle_starts'}
    assert isinstance(precomputed['count_unique'], list)
    assert isinstance(precomputed['count_repeat'], list)
    assert isinstance(precomputed['cycle_starts'], list)