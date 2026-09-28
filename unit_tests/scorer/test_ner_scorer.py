import pytest
import joblib

from llm_val.scorer.NerScorer import NerScorer
from llm_val.sampler.NerSampler import NerSampler
from llm_val.utils.ner_metrics import NER_METRICS

DATA_PATH = 'data/data_ner.pkl'
ENTITY_TYPES = ['Drugname', 'Drugclass', 'Drugform', 'DI', 'ADR', 'Finding']


@pytest.fixture()
def create_scorer():
    metrics = {'ner_exact_match_macro_f1': NER_METRICS['ner_exact_match_macro_f1']}
    scorer = NerScorer(metrics)
    return scorer


@pytest.fixture()
def create_sampler():
    data = joblib.load(DATA_PATH)
    sampler = NerSampler(test=data, entity_types=ENTITY_TYPES)
    return sampler


def test_create_ner_scorer(create_scorer):
    scorer = create_scorer
    assert isinstance(scorer, NerScorer)
    assert len(scorer.metrics) == 1


def test_ner_scorer_calc(create_scorer, create_sampler):
    scorer = create_scorer
    sampler = create_sampler
    metrics_val = scorer.calc(sampler=sampler, data_type='test')
    assert isinstance(metrics_val, dict)
    assert len(metrics_val) == len(ENTITY_TYPES) + 1
    assert 'macro_f1' in metrics_val
    assert isinstance(metrics_val['macro_f1'], float)
    assert 0 <= metrics_val['macro_f1'] <= 1


def test_ner_scorer_calc_with_metric_name(create_scorer, create_sampler):
    scorer = create_scorer
    sampler = create_sampler
    metrics_val = scorer.calc(sampler=sampler, data_type='test')
    assert isinstance(metrics_val, dict)
    assert len(metrics_val) == len(ENTITY_TYPES) + 1