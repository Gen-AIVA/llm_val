import pytest
import joblib
import pandas as pd

from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.sampler.RuBQSampler import RuBQSampler
from llm_val.sampler.LongAnswerSampler import LongAnswerSampler

from llm_val.val_test.model_quality.valtest_answer_unique import valtest_answer_unique

DATA_PATH_MMLU = 'data/data_mmlu.pkl'
DATA_PATH_LA = 'data/data_longanswer.pkl'
DATA_PATH_RUBQ = 'data/data_rubq.pkl'

PARAMS = [(MMLUSampler, DATA_PATH_MMLU),
          (RuBQSampler, DATA_PATH_RUBQ),
          (LongAnswerSampler, DATA_PATH_LA)]


@pytest.fixture(params=PARAMS)
def create_sampler(request):
    sampler_class, data_path = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    return sampler


def test_valtest_answer_unique(create_sampler):
    sampler = create_sampler
    res = valtest_answer_unique(sampler, 
                                data_type='test',
                                is_long_answer=True,
                                meteor_threshold=0.9,
                                semaphore_threshold=(0.01, 0.1),
                                is_info=False,
                                usr_value=None)
    assert res['report']['semaphore'] == 'green'
    assert isinstance(res['precomputed'], dict)
    assert 'usr_value' in res['precomputed']


def test_valtest_answer_unique_info(create_sampler):
    sampler = create_sampler
    res = valtest_answer_unique(sampler, 
                                data_type='test',
                                is_long_answer=True,
                                meteor_threshold=0.9,
                                semaphore_threshold=(0.01, 0.1),
                                is_info=True,
                                usr_value=None)
    assert res['report']['semaphore'] == 'gray'
    assert isinstance(res['precomputed'], dict)
    assert 'usr_value' in res['precomputed']


def test_valtest_answer_unique_mmlu():
    data = joblib.load(DATA_PATH_MMLU)
    sampler = MMLUSampler(test=data)
    res = valtest_answer_unique(sampler, 
                                data_type='test',
                                is_long_answer=False,
                                meteor_threshold=0.9,
                                semaphore_threshold=(0.01, 0.1),
                                is_info=False,
                                usr_value=None)
    assert res['report']['semaphore'] == 'green'
    assert res['precomputed']['usr_value'] == 0.6


def test_valtest_answer_unique_rubq():
    data = joblib.load(DATA_PATH_RUBQ)
    sampler = RuBQSampler(test=data)
    res = valtest_answer_unique(sampler, 
                                data_type='test',
                                is_long_answer=True,
                                meteor_threshold=0.9,
                                semaphore_threshold=(0.01, 0.1),
                                is_info=False,
                                usr_value=None)
    assert res['report']['semaphore'] == 'green'
    assert res['precomputed']['usr_value'] == 1.0


def test_valtest_answer_unique_la():
    data = joblib.load(DATA_PATH_LA)
    sampler = LongAnswerSampler(test=data)
    res = valtest_answer_unique(sampler,
                                data_type='test',
                                is_long_answer=True,
                                meteor_threshold=0.9,
                                semaphore_threshold=(0.01, 0.1),
                                is_info=False,
                                usr_value=None)
    assert res['report']['semaphore'] == 'green'
    assert res['precomputed']['usr_value'] == 1.0
