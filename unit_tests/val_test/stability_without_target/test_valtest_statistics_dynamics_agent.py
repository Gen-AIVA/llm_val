import pytest
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from llm_val.utils.api import GigaChatAPI
from llm_val.sampler.DGR_MMLUSampler import DGR_MMLUSampler
from llm_val.sampler.LongAnswerSampler import LongAnswerSampler
from llm_val.val_test.stability_without_target.valtest_statistics_dynamics_agent import valtest_statistics_dynamics_agent

DATA_PATH_LA = 'data/LongAnswerExample.pkl'
DATA_MMLU_DGR = 'data/dgr_data_mmlu.parquet'

@pytest.fixture()
def create_embedding_model(request, api_single_param):
    embedding_model = GigaChatAPI(**api_single_param)
    return embedding_model


def test_valtest_statistics_dynamics_agent_la(create_embedding_model):
    embedding_model = create_embedding_model
    data_la = joblib.load(DATA_PATH_LA)
    sampler_la = LongAnswerSampler(data_la)
    sampler_la.test['X']['question'] = data_la['question']
    sampler_la.test['y'] = pd.DataFrame({'metric_value': (sampler_la.test['y'].sum(1) > 15) * 1})
    sampler_la.train = {'X': [], 'y': []}
    sampler_la.train['X'], sampler_la.test['X'], \
        sampler_la.train['y'], sampler_la.test['y'] = train_test_split(sampler_la.test['X'],
                                                                       sampler_la.test['y'],
                                                                       test_size=0.5,
                                                                       random_state=45)
    
    test_res = valtest_statistics_dynamics_agent(sampler=sampler_la,
                                                 metric_name='cos_sim',
                                                 embedding_model=embedding_model,
                                                 data_types=['train', 'test'],
                                                 is_discrete=False,
                                                 similarity_threshold=0.9)
    
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}
    
    report = test_res['report']
    precomputed = test_res['precomputed']
    
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'new_query_rate', 'new_answer_rate'}
    
    report_df = report['result_dataframes'][0]
    assert isinstance(report_df, pd.DataFrame)
    assert set(report_df.columns) == {'Доля новых запросов', 'Доля новых ответов'}


def test_valtest_statistics_dynamics_agent_mmlu(create_embedding_model):
    data_dgr = pd.read_parquet(DATA_MMLU_DGR)
    sampler_mmlu_dgr = DGR_MMLUSampler(data_dgr)
    sampler_mmlu_dgr.test['X']['model_answer'] = np.random.choice(['A', 'B', 'C', 'D'], sampler_mmlu_dgr.test['X'].shape[0])
    sampler_mmlu_dgr.test['y'] = (sampler_mmlu_dgr.test['X']['model_answer'] == sampler_mmlu_dgr.test['X']['true_answer'])*1.0
    sampler_mmlu_dgr.train = {'X': [], 'y': []}
    sampler_mmlu_dgr.train['X'], sampler_mmlu_dgr.test['X'], \
        sampler_mmlu_dgr.train['y'], sampler_mmlu_dgr.test['y'] = train_test_split(sampler_mmlu_dgr.test['X'],
                                                                                   sampler_mmlu_dgr.test['y'],
                                                                                   test_size=0.5,
                                                                                   random_state=45)
    
    test_res = valtest_statistics_dynamics_agent(sampler=sampler_mmlu_dgr,
                                                 metric_name='bleu',
                                                 data_types=['train', 'test'],
                                                 is_discrete=True,
                                                 similarity_threshold=0.9)
    
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}
    
    report = test_res['report']
    precomputed = test_res['precomputed']
    
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'new_query_rate', 'new_answer_rate'}


def test_valtest_statistics_dynamics_agent_text_metric(create_embedding_model):
    data_la = joblib.load(DATA_PATH_LA)
    sampler_la = LongAnswerSampler(data_la)
    sampler_la.test['X']['question'] = data_la['question']
    sampler_la.test['y'] = pd.DataFrame({'metric_value': (sampler_la.test['y'].sum(1) > 15) * 1})
    sampler_la.train = {'X': [], 'y': []}
    sampler_la.train['X'], sampler_la.test['X'], \
        sampler_la.train['y'], sampler_la.test['y'] = train_test_split(sampler_la.test['X'],
                                                                       sampler_la.test['y'],
                                                                       test_size=0.5,
                                                                       random_state=45)
    
    test_res = valtest_statistics_dynamics_agent(sampler=sampler_la,
                                                 metric_name='rouge',
                                                 embedding_model=None,
                                                 data_types=['train', 'test'],
                                                 is_discrete=False,
                                                 similarity_threshold=0.9)
    
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_statistics_dynamics_agent_bert_score(create_embedding_model):
    data_la = joblib.load(DATA_PATH_LA)
    sampler_la = LongAnswerSampler(data_la)
    sampler_la.test['X']['question'] = data_la['question']
    sampler_la.test['y'] = pd.DataFrame({'metric_value': (sampler_la.test['y'].sum(1) > 15) * 1})
    sampler_la.train = {'X': [], 'y': []}
    sampler_la.train['X'], sampler_la.test['X'], \
        sampler_la.train['y'], sampler_la.test['y'] = train_test_split(sampler_la.test['X'],
                                                                       sampler_la.test['y'],
                                                                       test_size=0.5,
                                                                       random_state=45)
    
    test_res = valtest_statistics_dynamics_agent(sampler=sampler_la,
                                                 metric_name='bert_score',
                                                 embedding_model=None,
                                                 data_types=['train', 'test'],
                                                 is_discrete=False,
                                                 similarity_threshold=0.8)
    
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}
