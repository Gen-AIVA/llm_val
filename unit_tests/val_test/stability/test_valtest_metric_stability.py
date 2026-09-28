import joblib
import pytest

from llm_val.utils.metrics import METRICS
from llm_val.sampler.RuBQSampler import RuBQSampler
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.val_test.stability.valtest_metric_stability import valtest_metric_stability


DATA_PATH_MMLU = 'data/data_mmlu.pkl'
DATA_PATH_RUBQ = 'data/data_rubq.pkl'
PARAMS = [(RuBQSampler, DATA_PATH_RUBQ, 'multicol_mean', 'multicol_mean', ['test', 'test'], True),
          (MMLUSampler, DATA_PATH_MMLU, 'multicol_mean', 'multicol_mean', ['test', 'test'], False)]


@pytest.fixture(params=PARAMS)
def create_scorer_sampler(request):
    sampler_class, data_path, main_metric, metric_agg, data_types, is_info = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    scorer = UniversalScorer(metrics=METRICS)
    res = valtest_metric_stability(sampler=sampler,
                                   scorer=scorer,
                                   main_metric=main_metric,
                                   metric_agg=metric_agg,
                                   data_types=data_types,
                                   is_info=is_info)
    return res, is_info


def test_valtest_metric_stability_report(create_scorer_sampler):
    test_res, is_info = create_scorer_sampler
    color = 'gray' if is_info else 'green'
    report = test_res['report']
    assert report['semaphore'] == color
    result_df = report['result_dataframes'][0]
    assert result_df['Абсолютная разница'][0] == result_df['Относительная разница, %'][0]


def test_valtest_metric_stability_precomputed(create_scorer_sampler):
    test_res, is_info = create_scorer_sampler
    precomputed = test_res['precomputed']
    assert set(precomputed.keys()) == {'base_metric_value', 'test_metric_value'}