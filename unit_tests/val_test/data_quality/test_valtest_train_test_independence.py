import pytest
import joblib
import pandas as pd

from llm_val.utils.metrics import METRICS
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.sampler.RuBQSampler import RuBQSampler
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.sampler.LongAnswerSampler import LongAnswerSampler
from llm_val.model.MultiThreadGigaModel import MultiThreadGigaModel

from llm_val.val_test.data_quality.valtest_train_test_independence import valtest_train_test_independence

DATA_PATH_MMLU = 'data/data_mmlu.pkl'
DATA_PATH_LA = 'data/data_longanswer.pkl'
DATA_PATH_RUBQ = 'data/data_rubq.pkl'

PARAMS = [(MMLUSampler, DATA_PATH_MMLU, MultiThreadGigaModel, {'n_jobs': 8}),
          (RuBQSampler, DATA_PATH_RUBQ, MultiThreadGigaModel, {'n_jobs': 8}),
          (LongAnswerSampler, DATA_PATH_LA, MultiThreadGigaModel, {'n_jobs': 8})]


@pytest.fixture(params=PARAMS)
def create_sampler_scorer_model(request, api_single_param):
    sampler_class, data_path, model_class, model_params = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    scorer = UniversalScorer(metrics=METRICS)
    model_params.update(api_single_param)
    model = model_class(model='GigaChat:latest', **model_params)
    return sampler, scorer, model


def test_valtest_train_test_independence_key(create_sampler_scorer_model):
    sampler, scorer, model = create_sampler_scorer_model
    res = valtest_train_test_independence(sampler=sampler,
                                          scorer=scorer,
                                          model=model,
                                          test_data_type='test',
                                          is_mask_option=False,
                                          is_split_gini_quantile=False,
                                          mask_word_iterations=5,
                                          mask_word_semaphore_threshold=(0.5, 0.7),
                                          is_info=False,
                                          mask_option_frac=None,
                                          mask_word_frac=None,
                                          split_gini_quantile=None)
    assert res['report']['semaphore'] == 'green'
    assert isinstance(res['precomputed'], dict)
    assert 'mask_word_frac' in res['precomputed']
    
    
def test_valtest_train_test_independence_key(create_sampler_scorer_model):
    sampler, scorer, model = create_sampler_scorer_model
    res = valtest_train_test_independence(sampler=sampler,
                                          scorer=scorer,
                                          model=model,
                                          test_data_type='test',
                                          is_mask_option=False,
                                          is_split_gini_quantile=False,
                                          mask_word_iterations=1,
                                          mask_word_semaphore_threshold=(0.5, 0.7),
                                          is_info=True,
                                          mask_option_frac=None,
                                          mask_word_frac=None,
                                          split_gini_quantile=None)
    assert res['report']['semaphore'] == 'gray'
    assert isinstance(res['precomputed'], dict)
    assert 'mask_word_frac' in res['precomputed']
    
    
def test_valtest_train_test_independence_mmlu(create_sampler_scorer_model):
    _, _, model = create_sampler_scorer_model
    data = joblib.load(DATA_PATH_MMLU)
    sampler_mmlu = MMLUSampler(test=data)
    scorer = UniversalScorer(metrics=METRICS)
    res = valtest_train_test_independence(sampler=sampler_mmlu,
                                          scorer=scorer,
                                          model=model,
                                          test_data_type='test',
                                          is_mask_option=True,
                                          is_split_gini_quantile=False,
                                          mask_word_iterations=5,
                                          mask_option_semaphore_threshold=(0.01, 0.05),
                                          mask_word_semaphore_threshold=(0.5, 0.7),
                                          is_info=False,
                                          mask_option_frac=None,
                                          mask_word_frac=None,
                                          split_gini_quantile=None)
    assert res['report']['semaphore'] == 'green'
    assert 'mask_option_frac' in res['precomputed']
    
    
def test_valtest_train_test_independence_empty(create_sampler_scorer_model):
    sampler, scorer, model = create_sampler_scorer_model
    res = valtest_train_test_independence(sampler=sampler,
                                          scorer=scorer,
                                          model=model,
                                          test_data_type='test',
                                          is_mask_option=False,
                                          is_mask_word=False,
                                          is_split_gini_quantile=False,
                                          is_info=False,
                                          mask_option_frac=None,
                                          mask_word_frac=None,
                                          split_gini_quantile=None)
    assert res['report']['semaphore'] == 'gray'
    assert isinstance(res['precomputed'], dict)
    assert res['report']['result_dataframes'][0].iloc[0]['Проверка'] == 'Проверки не проводились'
