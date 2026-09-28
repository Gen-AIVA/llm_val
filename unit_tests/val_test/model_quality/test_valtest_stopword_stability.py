import pytest
import joblib
import pandas as pd
import typing as tp

from llm_val.utils.metrics import METRICS
from llm_val.sampler.RuBQSampler import RuBQSampler
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.val_test.model_quality.valtest_stopword_stability import valtest_metric_aug_stability

DATA_PATH_MMLU = 'data/data_mmlu.pkl'
DATA_PATH_RUBQ = 'data/data_rubq.pkl'

PARAMS = [(RuBQSampler, DATA_PATH_RUBQ),
          (MMLUSampler, DATA_PATH_MMLU)]

class SimpleModel:
    def predict(self, questions: tp.List[str]) -> tp.List[str]:
        return ['A'] * len(questions)

class ExampleModelWrapper:
    def __init__(self, model: tp.Any):
        self.model = model


    def predict(self, questions: tp.List[str]) -> tp.List[str]:
        return self.model.predict(questions)


@pytest.fixture(params=PARAMS)
def create_sampler_scorer_model(request):
    sampler_class, data_path = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    scorer = UniversalScorer(metrics=METRICS)
    model = ExampleModelWrapper(SimpleModel())
    return sampler, scorer, model


def test_valtest_metric_aug_stability_format(create_sampler_scorer_model):
    sampler, scorer, model = create_sampler_scorer_model
    test_res = valtest_metric_aug_stability(sampler, scorer, model)
    assert isinstance(test_res, dict)
    assert 'report' in test_res
    assert 'precomputed' in test_res


def test_valtest_metric_aug_stability_report(create_sampler_scorer_model):
    sampler, scorer, model = create_sampler_scorer_model
    test_res = valtest_metric_aug_stability(sampler, scorer, model)
    report = test_res['report']
    assert isinstance(report, dict)
    assert 'semaphore' in report
    assert 'result_plots' in report
    assert 'result_dataframes' in report
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)
    assert not report['result_dataframes'][0].empty


def test_valtest_metric_aug_stability_precomputed(create_sampler_scorer_model):
    sampler, scorer, model = create_sampler_scorer_model
    test_res = valtest_metric_aug_stability(sampler, scorer, model)
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert 'metric_baseline' in precomputed
    assert 'metrics_augmented' in precomputed
    assert isinstance(precomputed['metric_baseline'], float)
    assert isinstance(precomputed['metrics_augmented'], dict)
    assert all(isinstance(value, float) for value in precomputed['metrics_augmented'].values())


def test_valtest_metric_aug_stability_semaphore(create_sampler_scorer_model):
    sampler, scorer, model = create_sampler_scorer_model
    test_res = valtest_metric_aug_stability(sampler, scorer, model)
    report = test_res['report']
    assert report['semaphore'] in ['red']


def test_valtest_metric_aug_stability_baseline_value(create_sampler_scorer_model):
    sampler, scorer, model = create_sampler_scorer_model
    test_res = valtest_metric_aug_stability(sampler, scorer, model)
    precomputed = test_res['precomputed']
    assert 0 <= precomputed['metric_baseline'] <= 1


def test_valtest_metric_aug_stability_augmented_values(create_sampler_scorer_model):
    sampler, scorer, model = create_sampler_scorer_model
    test_res = valtest_metric_aug_stability(sampler, scorer, model)
    precomputed = test_res['precomputed']
    assert all(0 <= value <= 1 for value in precomputed['metrics_augmented'].values())
