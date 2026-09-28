import pytest
import joblib
import pandas as pd

from llm_val.utils.metrics import METRICS
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.sampler.RuBQSampler import RuBQSampler
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.sampler.LongAnswerSampler import LongAnswerSampler
from llm_val.val_test.model_quality.valtest_metric_by_token_size import valtest_metric_by_token_size

CONST_TOKEN_COUNT = 228

class ModelTokenCount:
    def __init__(self, model_name: str):
        self.model_name = model_name
    def get_token_count(self, text: str) -> int:
        if self.model_name == 'ConstTokenCountModel':
            return CONST_TOKEN_COUNT
        elif self.model_name == 'LenTokenCountModel':
            return len(text)
        else:
            return 0


DATA_PATH_MMLU = 'data/data_mmlu.pkl'
DATA_PATH_LA = 'data/data_longanswer.pkl'
DATA_PATH_RUBQ = 'data/data_rubq.pkl'

PARAMS = [(MMLUSampler, DATA_PATH_MMLU, ModelTokenCount(model_name='ConstTokenCountModel'), 0.05, 'CONST'),
          (LongAnswerSampler, DATA_PATH_LA, ModelTokenCount(model_name='ConstTokenCountModel'), 0.05, 'CONST'),
          (RuBQSampler, DATA_PATH_RUBQ, ModelTokenCount(model_name='ConstTokenCountModel'), 0.05, 'CONST'),
          (MMLUSampler, DATA_PATH_MMLU, ModelTokenCount(model_name='LenTokenCountModel'), 0.95, 'CONST'),
          (MMLUSampler, DATA_PATH_MMLU, ModelTokenCount(model_name='LenTokenCountModel'), 0.001, 'NOT_CONST'),
        ]


@pytest.fixture(params=PARAMS)
def create_scorer_sampler(request):
    sampler_class, data_path, model, min_share, distribution_type = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    scorer = UniversalScorer(metrics=METRICS)
    res = valtest_metric_by_token_size(sampler=sampler,
                                       scorer=scorer,
                                       data_type='test',
                                       token_counter_model=model,
                                       min_bin_size=min_share,
                                       bin_algo='merge',
                                       metric_agg='single_mean', 
                                       n_bootstrap=100, 
                                       quantile=0.95
                                       )
    return res, distribution_type


def test_valtest_metric_report(create_scorer_sampler):
    test_res, distribution_type = create_scorer_sampler
    report = test_res['report']
    true_color = 'gray'
    assert report['semaphore'] == true_color
    if distribution_type is 'CONST':
        assert report['result_dataframes'][0].shape[0] == 1
    else:
        assert report['result_dataframes'][0].shape[0] > 1


def test_valtest_metric_precomputed(create_scorer_sampler):
    test_res, distribution_type = create_scorer_sampler
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'token_and_bins'}