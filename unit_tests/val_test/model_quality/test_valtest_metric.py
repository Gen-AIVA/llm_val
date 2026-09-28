import pytest
import joblib
import pandas as pd

from llm_val.utils.metrics import METRICS
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.val_test.model_quality.valtest_metric import valtest_metric

DATA_PATH = 'data/data_mmlu.pkl'


@pytest.fixture()
def create_scorer_sampler():
    data = joblib.load(DATA_PATH)
    sampler = MMLUSampler(test=data)
    scorer = UniversalScorer(metrics=METRICS)
    res = valtest_metric(sampler=sampler, 
                         scorer=scorer, 
                         metric_agg='multicol_mean', 
                         main_metric='multicol_mean', 
                         data_type='test')
    return res


def test_valtest_metric_report(create_scorer_sampler):
    test_res = create_scorer_sampler
    report = test_res['report']
    assert report['semaphore'] == 'green'


def test_valtest_metric_precomputed(create_scorer_sampler):
    test_res = create_scorer_sampler
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'metric_value'}
    assert precomputed['metric_value']['multicol_mean'] == 0.6
