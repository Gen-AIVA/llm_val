import pytest
import joblib
import pandas as pd
from datetime import datetime
from nltk import word_tokenize

from llm_val.sampler.AgentMemorySampler import AgentMemorySampler

from llm_val.val_test.agent_memory.valtest_summarization_quality import valtest_summarization_quality

DATA_PATH_LA = 'data/LongAnswerExample.pkl'
PARAMS = [(AgentMemorySampler, DATA_PATH_LA)]


@pytest.fixture(params=PARAMS)
def create_scorer_sampler(request):
    sampler_class, data_path = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(data=data)

    res = valtest_summarization_quality(sampler=sampler,
                                        tokenizer=word_tokenize,
                                        func_llm_score=None,
                                        model_judge=None)

    return res


def test_valtest_summarization_quality_report(create_scorer_sampler):
    test_res = create_scorer_sampler
    assert isinstance(test_res, dict)

    report = test_res['report']
    true_color = 'gray'

    assert isinstance(report, dict)
    assert report['semaphore'] == true_color


def test_valtest_summarization_quality_precomputed(create_scorer_sampler):
    test_res = create_scorer_sampler
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'summarization_quality'}