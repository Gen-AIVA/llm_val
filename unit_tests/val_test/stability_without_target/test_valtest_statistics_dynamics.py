import pytest
import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from llm_val.sampler.DGR_MMLUSampler import DGR_MMLUSampler
from llm_val.sampler.LongAnswerSampler import LongAnswerSampler
from llm_val.val_test.stability_without_target.valtest_statistics_dynamics import valtest_statistics_dynamics

DATA_PATH_LA = 'data/LongAnswerExample.pkl'
DATA_MMLU_DGR = 'data/dgr_data_mmlu.parquet'

@pytest.fixture()
def create_model():
    model = SentenceTransformer('cointegrated/rubert-tiny')
    return model


def test_valtest_statistics_dynamics_la(create_model):
    model = create_model
    columns_list = ['Корреляция с ключевой метрикой качества на train', 'Корреляция с ключевой метрикой качества на test', \
                    'Корреляция с ключевой метрикой качества на val', 'PSI ответов на test', 'PSI запросов на test', \
                    'PSI ответов на val', 'PSI запросов на val', 'Светофор по запросам', 'Светофор по ответам']
    index_list = ['Доля стоп-слов', 'Число токенов', 'Частота токенов', 'Доля новых токенов']
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
    sampler_la.val = {'X': [], 'y': []}
    sampler_la.test['X'], sampler_la.val['X'], \
        sampler_la.test['y'], sampler_la.val['y'] = train_test_split(sampler_la.test['X'],
                                                                     sampler_la.test['y'],
                                                                     test_size=0.5,
                                                                     random_state=45)
    test_res = valtest_statistics_dynamics(sampler=sampler_la,
                                           tokenizer=model,
                                           stop_words_list=stopwords.words('russian'),
                                           data_types=('train', 'test', 'val'))
    report = test_res['report']
    report_df = report['result_dataframes'][0]
    precomputed = test_res['precomputed']
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report_df, pd.DataFrame)
    assert isinstance(precomputed, dict)
    assert set(report_df.columns) == set(columns_list)
    assert set(report_df.index) == set(index_list)
    assert set(precomputed['df_statistics'].columns) == (set(columns_list) - set(['Светофор по запросам', 'Светофор по ответам']))
    assert set(precomputed['df_statistics'].index) == set(index_list)


def test_valtest_statistics_dynamics_mmlu(create_model):
    model = create_model
    columns_list = ['PSI ответов на test', 'PSI запросов на test', 'Светофор по запросам', 'Светофор по ответам']
    index_list = ['Доля стоп-слов', 'Число токенов', 'Частота токенов', 'Доля новых токенов', 'PSI предсказаний ответов LLM']
    data_dgr = pd.read_parquet(DATA_MMLU_DGR)
    sampler_mmlu_dgr = DGR_MMLUSampler(data_dgr)
    sampler_mmlu_dgr.test['X']['model_answer'] = np.random.choice(['A', 'B', 'C', 'D'], sampler_mmlu_dgr.test['X'].shape)
    sampler_mmlu_dgr.test['y'] = (sampler_mmlu_dgr.test['X']['model_answer'] == sampler_mmlu_dgr.test['X']['true_answer'])*1.0
    sampler_mmlu_dgr.train = {'X': [], 'y': []}
    sampler_mmlu_dgr.train['X'], sampler_mmlu_dgr.test['X'], \
        sampler_mmlu_dgr.train['y'], sampler_mmlu_dgr.test['y'] = train_test_split(sampler_mmlu_dgr.test['X'],
                                                                                   sampler_mmlu_dgr.test['y'],
                                                                                   test_size=0.5,
                                                                                   random_state=45)
    test_res = valtest_statistics_dynamics(sampler=sampler_mmlu_dgr,
                                           tokenizer=model,
                                           stop_words_list=stopwords.words('russian'),
                                           data_types=('train', 'test'),
                                           is_discrete=True)
    report = test_res['report']
    report_df = report['result_dataframes'][0]
    precomputed = test_res['precomputed']
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report_df, pd.DataFrame)
    assert isinstance(precomputed, dict)
    assert set(report_df.columns) == set(columns_list)
    assert set(report_df.index) == set(index_list)
    assert set(precomputed['df_statistics'].columns) == (set(columns_list) - set(['Светофор по запросам', 'Светофор по ответам']))
    assert set(precomputed['df_statistics'].index) == set(index_list)
    