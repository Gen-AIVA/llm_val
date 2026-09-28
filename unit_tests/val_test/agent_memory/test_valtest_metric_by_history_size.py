import pytest
import joblib
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer

from llm_val.utils.metrics import METRICS
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.sampler.AgentMemorySampler import AgentMemorySampler

from llm_val.val_test.agent_memory.valtest_metric_by_history_size import valtest_metric_by_history_size

DATA_PATH_LA = 'data/LongAnswerExample.pkl'
PARAMS = [(AgentMemorySampler, DATA_PATH_LA)]


@pytest.fixture(params=PARAMS)
def create_scorer_sampler(request):
    sampler_class, data_path = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(data=data)

    scorer = UniversalScorer(metrics=METRICS)
    rubert = SentenceTransformer('cointegrated/rubert-tiny2')

    res = valtest_metric_by_history_size(sampler,
                                         scorer,
                                         rubert)
    
    return res

def test_valtest_metric_by_history_size_report(create_scorer_sampler):
    test_res = create_scorer_sampler
    assert isinstance(test_res, dict)

    report = test_res['report']
    true_color = 'gray'
    
    assert isinstance(report, dict)
    assert report['semaphore'] == true_color


def test_valtest_metric_by_history_size_precomputed(create_scorer_sampler):
    test_res = create_scorer_sampler
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'correlation_results', 'bins_results'}