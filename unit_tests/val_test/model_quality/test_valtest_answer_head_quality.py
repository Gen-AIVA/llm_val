import pytest
import joblib
import pandas as pd

from llm_val.sampler.LongAnswerSampler import LongAnswerSampler
from llm_val.val_test.model_quality.valtest_answer_head_quality import valtest_answer_head_quality


@pytest.fixture()
def create_test():
    data = joblib.load('data/LongAnswerExample.pkl')
    sampler = LongAnswerSampler(data)
    sampler.test['X']['true_answer'] = data.groupby(by=['id_task'])['true_answer'].first()
    sampler.test['X']['true_answer'] = sampler.test['X']['true_answer'].apply(lambda x: [x])
    res = valtest_answer_head_quality(sampler=sampler,
                                      data_type='test',
                                      metric_name='bleu',
                                      n_head_elements=3,
                                      use_sentences=True,
                                      step=1,
                                      semaphore_threshold=0.1)
    return sampler, res


def test_valtest_answer_head_quality_dict(create_test):
    _, test_res = create_test
    assert isinstance(test_res, dict)


def test_valtest_answer_head_quality_keys(create_test):
    _, test_res = create_test
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_answer_head_quality_metric_report(create_test):
    _, test_res = create_test
    report = test_res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)


def test_valtest_answer_head_quality_metric_precomputed(create_test):
    _, test_res = create_test
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)


def test_valtest_answer_head_quality(create_test):
    sampler, _ = create_test
    for metric in ['bleu', 'meteor', 'rouge']:
        res = valtest_answer_head_quality(sampler=sampler,
                                          data_type='test',
                                          metric_name=metric,
                                          n_head_elements=3,
                                          use_sentences=True,
                                          step=1,
                                          semaphore_threshold=0.1)
        assert res['report']['semaphore'] == 'green'
        assert 'head_metrics' in res['precomputed']
        assert res['report']['result_dataframes'][0].values[0][1] > 0.1
        res = valtest_answer_head_quality(sampler=sampler,
                                          data_type='test',
                                          metric_name=metric,
                                          n_head_elements=40,
                                          use_sentences=False,
                                          step=5,
                                          semaphore_threshold=0.1)
        assert res['report']['semaphore'] == 'green'
        assert 'head_metrics' in res['precomputed']
        assert res['report']['result_dataframes'][0].values[0][1] > 0.1
