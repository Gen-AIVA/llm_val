import pytest
import joblib
import pandas as pd

from llm_val.utils.metrics import METRICS
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.sampler.RuBQSampler import RuBQSampler
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.sampler.LongAnswerSampler import LongAnswerSampler
from llm_val.val_test.model_quality.valtest_metric_by_answer_size import valtest_metric_by_answer_size

DATA_PATH_MMLU = 'data/data_mmlu.pkl'
DATA_PATH_LA = 'data/data_longanswer.pkl'
DATA_PATH_RUBQ = 'data/data_rubq.pkl'

PARAMS = [(MMLUSampler, DATA_PATH_MMLU, True),
          (MMLUSampler, DATA_PATH_MMLU, False),
          (LongAnswerSampler, DATA_PATH_LA, True),
          (LongAnswerSampler, DATA_PATH_LA, False),
          (RuBQSampler, DATA_PATH_RUBQ, True),
          (RuBQSampler, DATA_PATH_RUBQ, False),]


@pytest.fixture(params=PARAMS)
def create_scorer_sampler(request):
    sampler_class, data_path, is_info = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    scorer = UniversalScorer(metrics=METRICS)
    res = valtest_metric_by_answer_size(sampler=sampler,
                                        scorer=scorer,
                                        data_type='test',
                                        bins='freedman', 
                                        metric_agg='single_mean', 
                                        n_bootstrap=100, 
                                        quantile=0.95,
                                        semaphore_threshold=(0.05, 0.1), 
                                        is_info=is_info)
    return res, is_info


def test_valtest_metric_report(create_scorer_sampler):
    test_res, is_info = create_scorer_sampler
    report = test_res['report']
    true_color = 'gray' if is_info else 'red'
    assert report['semaphore'] == true_color


def test_valtest_metric_precomputed(create_scorer_sampler):
    test_res, is_info = create_scorer_sampler
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'stat_value', 'p_value'}