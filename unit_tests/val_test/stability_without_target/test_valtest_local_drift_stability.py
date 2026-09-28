import faiss
import pytest
import joblib
import numpy as np
import typing as tp
import pandas as pd
from sklearn.model_selection import train_test_split

from llm_val.utils.metrics import METRICS
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.sampler.LongAnswerSampler import LongAnswerSampler
from llm_val.model.MultiThreadGigaModel import MultiThreadGigaModel
from llm_val.val_test.stability_without_target.valtest_local_drift_stability import valtest_local_drift_stability

DATA_PATH_LA2 = 'data/LongAnswerExample.pkl'
PARAMS = [(MultiThreadGigaModel, {'n_jobs': 8})]

class ANN():
    def __init__(self,
                 n_partitions: int=10,
                 nprobe: int=2):
        self.n_partitions = n_partitions
        self.nprobe = nprobe
        self.index = None
        
    def create_index(self,
                     embedding: tp.Any,
                     exact: bool=False):
        '''
        Создает в классе индекс для дальнейшего поиска по данным эмбеддингам
        Parameters:
            embedding: Набор эмбеддингов для создания индекса
            exact: Если True, то запускается точный поиск (kNN), рекомендуется для небольших наборов (меньше 10к наблюдений) 
        '''
        embedding = np.array(embedding).astype(np.float32)
        dimension = embedding.shape[1]
        db_vectors = embedding.copy().astype(np.float32)
        faiss.normalize_L2(db_vectors)
        self.exact = exact
        if self.exact:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        else:
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer,
                                            dimension,
                                            self.n_partitions,
                                            faiss.METRIC_INNER_PRODUCT)
            self.index.train(db_vectors)
        self.index.add_with_ids(db_vectors, np.arange(len(db_vectors)))
    
    def search_query(self,
                     query_vector: tp.Any,
                     k: int) -> tp.List[tp.Dict[str, tp.Any]]:
        '''
        Выполняет поиск запроса в индексе
        Parameters:
            query_vector: Вектор эмбеддинга запроса
            k: Число получаемых наиболее близких запросов
        Return: 
            Список словарей с индексами наиболее близких запросов и значениями близости
        '''
        query_vector = np.array(query_vector).astype(np.float32).reshape(1, -1)
        self.index.nprobe = self.nprobe
        if self.index is None:
            return []
        faiss.normalize_L2(query_vector)
        similarities, similarities_ids = self.index.search(query_vector, k)
        return [{'id': similarities_ids[0][i], 'similarity': similarities[0][i]} for i in range(len(similarities_ids[0]))]


def long_answer_binarizer(y_data: pd.DataFrame) -> pd.Series:
    return pd.Series((y_data.sum(1) > 15) * 1)


@pytest.fixture(params=PARAMS)
def create_sampler_scorer_model(request, api_single_param):
    model_class, model_params = request.param
    model_params.update(api_single_param)
    scorer = UniversalScorer(metrics=METRICS)
    model = model_class(model='GigaChat:latest', **model_params)
    data_la = joblib.load(DATA_PATH_LA2)
    sampler_la = LongAnswerSampler(data_la)
    sampler_la.test['X']['question'] = data_la['question']
    X_train, X_test, y_train, y_test = train_test_split(sampler_la.test['X'],
                                                        sampler_la.test['y'],
                                                        test_size=0.5,
                                                        random_state=42)
    sampler_la.train = {'X': X_train, 'y': y_train}
    sampler_la.test = {'X': X_test, 'y': y_test}
    ann = ANN(1)
    config = {'create_index': {'exact': True}, 'search_query': {}}
    test_res = valtest_local_drift_stability(sampler=sampler_la,
                                             scorer=scorer,
                                             main_metric='single_mean_metric_value',
                                             model=model,
                                             ann=ann,
                                             ann_config=config,
                                             n_closest=4,
                                             metric_binarizer=long_answer_binarizer,
                                             metric_agg='single_mean',
                                             data_types=('train', 'test'))
    return test_res


def test_valtest_local_drift_stability(create_sampler_scorer_model):
    test_res = create_sampler_scorer_model
    assert test_res['report']['semaphore'] == 'green'


def test_valtest_local_drift_stability_dict(create_sampler_scorer_model):
    test_res = create_sampler_scorer_model
    assert isinstance(test_res, dict)


def test_valtest_local_drift_stability_keys(create_sampler_scorer_model):
    test_res = create_sampler_scorer_model
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_local_drift_stability_report(create_sampler_scorer_model):
    test_res = create_sampler_scorer_model
    report = test_res['report']
    report_df = report['result_dataframes'][0]
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report_df, pd.DataFrame)


def test_valtest_local_drift_stability_precomputed(create_sampler_scorer_model):
    test_res = create_sampler_scorer_model
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'metric_value', 'metric_value_estimate', 'reliability'}

