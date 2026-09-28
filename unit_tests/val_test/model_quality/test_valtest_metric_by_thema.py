import pytest
import joblib
import pandas as pd

from llm_val.utils.metrics import METRICS
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.sampler.LongAnswerSampler import LongAnswerSampler


from llm_val.val_test.model_quality.valtest_metric_by_thema import valtest_metric_by_thema

DATA_PATH_MMLU = 'data/data_mmlu.pkl'
DATA_PATH_LA = 'data/data_longanswer.pkl'

PARAMS = [(MMLUSampler, DATA_PATH_MMLU, True),
          (LongAnswerSampler, DATA_PATH_LA, False)]

@pytest.fixture(params=PARAMS)
def create_scorer_sampler(request):
    sampler_class, data_path, is_info = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    scorer = UniversalScorer(metrics=METRICS)
    res = valtest_metric_by_thema(sampler=sampler, 
                                  scorer=scorer, 
                                  metric_agg='multicol_mean',
                                  main_metric='multicol_mean',
                                  data_type='test')
    return res, is_info


def test_valtest_metric_report(create_scorer_sampler):
    test_res, is_info = create_scorer_sampler
    color = 'gray' if is_info else 'green'
    report = test_res['report']
    assert report['semaphore'] == color


def test_valtest_metric_precomputed(create_scorer_sampler):
    test_res, is_info = create_scorer_sampler
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'metric_value_by_thema'}
    if not is_info:
        assert precomputed['metric_value_by_thema']['java']['multicol_mean'] == 3.0
