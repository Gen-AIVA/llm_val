import pytest
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from llm_val.utils.metrics import METRICS
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.sampler.LongAnswerSampler import LongAnswerSampler
from llm_val.model.MultiThreadGigaModel import MultiThreadGigaModel
from llm_val.val_test.stability_without_target.valtest_global_drift_stability import valtest_global_drift_stability

DATA_PATH_LA = 'data/LongAnswerExample.pkl'
PARAMS = [(MultiThreadGigaModel, {'n_jobs': 8})]


def long_answer_binarizer(y_data: pd.DataFrame) -> pd.Series:
    return pd.Series((y_data.sum(1) > 15) * 1)


@pytest.fixture(params=PARAMS)
def create_sampler_scorer_model(request, api_single_param):
    model_class, model_params = request.param
    model_params.update(api_single_param)
    scorer = UniversalScorer(metrics=METRICS)
    model = model_class(model='GigaChat:latest', **model_params)
    data_la = joblib.load(DATA_PATH_LA)
    sampler_la = LongAnswerSampler(data_la)
    sampler_la.test['X']['question'] = data_la['question']
    X_train, X_test, y_train, y_test = train_test_split(sampler_la.test['X'],
                                                        sampler_la.test['y'],
                                                        test_size=0.5,
                                                        random_state=42)
    sampler_la.train = {'X': X_train, 'y': y_train}
    sampler_la.test = {'X': X_test, 'y': y_test}
    test_res = valtest_global_drift_stability(sampler=sampler_la,
                                              scorer=scorer,
                                              main_metric='single_mean_metric_value',
                                              model=model,
                                              n_chunks=4,
                                              top_distance_features=[5, 10],
                                              p_value=0.9,
                                              corr_threshold=0.1,
                                              metric_binarizer=long_answer_binarizer)
    return test_res


def test_valtest_global_drift_stability(create_sampler_scorer_model):
    test_res = create_sampler_scorer_model
    assert test_res['report']['semaphore'] == 'green'


def test_valtest_global_drift_stability_dict(create_sampler_scorer_model):
    test_res = create_sampler_scorer_model
    assert isinstance(test_res, dict)


def test_valtest_global_drift_stability_keys(create_sampler_scorer_model):
    test_res = create_sampler_scorer_model
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_global_drift_stability_report(create_sampler_scorer_model):
    test_res = create_sampler_scorer_model
    report = test_res['report']
    report_df = report['result_dataframes'][0]
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report_df, pd.DataFrame)


def test_valtest_global_drift_stability_precomputed(create_sampler_scorer_model):
    test_res = create_sampler_scorer_model
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'metric_value_estimate', 'metric_value'}
