import pytest
import joblib

from llm_val.scorer.Scorer import Scorer
from llm_val.utils.metrics import METRICS
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.sampler.RuBQSampler import RuBQSampler
from llm_val.scorer.UniversalScorer import UniversalScorer

SCORER_LIST = [(UniversalScorer, MMLUSampler, METRICS, 'test', 'data/data_mmlu.pkl'),
               (UniversalScorer, RuBQSampler, METRICS, 'test', 'data/data_rubq.pkl')]


@pytest.mark.parametrize('scorer_param', SCORER_LIST)
def test_create_scorer(scorer_param):
    scorer_class, _, metrics, _, _ = scorer_param
    scorer = scorer_class(metrics)
    assert isinstance(scorer, Scorer)


@pytest.mark.parametrize('scorer_param', SCORER_LIST)
def test_scorer_calc(scorer_param):
    scorer_class, sampler_class, metrics, data_type, data_path = scorer_param
    data = joblib.load(data_path)
    sampler = sampler_class(data)
    scorer = scorer_class(metrics)
    metrics_val = scorer.calc(sampler=sampler,
                              data_type=data_type)
    assert isinstance(metrics_val, dict)
    assert len(metrics_val) != 0
    assert len(metrics_val) == len(metrics)
