import sklearn
import collections
import numpy as np
import typing as tp
import pandas as pd
from copy import deepcopy

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.autometrics import metrics2func_agent
from llm_val.utils.report_helper import semaphore_by_threshold

                                         
def report_valtest_statistics_dynamics_agent(new_query_rate: float,
                                             new_answer_rate: float,
                                             semaphore_threshold: tp.Tuple[float, float]=(0.05, 0.5),
                                             greater_is_better: bool=False,
                                             is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        semaphore_threshold: Граница светофора - максимально допустимая доля отличающихся ответов
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: Если True, то тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_table = pd.DataFrame({'Доля новых запросов': [new_query_rate],
                              'Доля новых ответов': [new_answer_rate]})
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else semaphore_by_threshold(new_answer_rate, 
                                                                          threshold=semaphore_threshold, 
                                                                          greater_is_better=greater_is_better)
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_statistics_dynamics_agent(sampler: Sampler,
                                      metric_name: str='rouge',
                                      embedding_model: tp.Optional[tp.Any]=None,
                                      data_types: tp.List[str]=['train', 'test'],
                                      is_discrete: bool=False,
                                      similarity_threshold: float=0.9,
                                      semaphore_threshold: tp.Tuple[float, float]=(0.05, 0.5),
                                      greater_is_better: bool=False,
                                      is_info: bool=False,
                                      **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на анализ динамики стандартных статистик для запроса и ответа LLM
    Parameters:
        sampler: Объект класса Sampler
        metric_name: метрика для расчета близости между текстами. Для расчета доступны метрики 'cos_sim', 'bleu', 'rouge', 'meteor' и 'bert_score'
        embedding_model: Класс-обертка над эмбеддинг моделью. Модель обязана иметь метод get_embedding.
        data_types: Типы данных, на которых сравниваются метрики (в хронологическом порядке формирования выборок)
        is_discrete: Булевский флаг, если True, то cчитается, что ответы модели находятся в дискретной области,
            и проводится анализ динамики распределения предсказаний моделей во времени по PSI
        similarity_threshold: Граница признания семантических дубликатов по метрике metric_name.
        semaphore_threshold: Граница светофора - максимально допустимая доля отличающихся ответов
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    data_train = getattr(sampler, data_types[0])
    data_test = getattr(sampler, data_types[1])
    
    train_query, train_answer = data_train['X']['question'].to_list(), data_train['X']['model_answer'].to_list()
    test_query, test_answer = data_test['X']['question'].to_list(), data_test['X']['model_answer'].to_list()
    
    query_similarity_table = calculate_similarity_table(train_query, test_query, metric_name, embedding_model)
    query_duplicates = query_similarity_table[query_similarity_table['similarity'] >= similarity_threshold]
    query_duplicate_rate = len(set(query_duplicates['idx_2'])) / len(test_query)
    new_query_rate = 1 - query_duplicate_rate

    if is_discrete:
        train_ans_dict, test_ans_dict = collections.Counter(train_answer), collections.Counter(test_answer)
        train_ans_freq, test_ans_freq = get_token_freq(train_ans_dict, test_ans_dict)
        new_answer_rate = psi(train_ans_freq, test_ans_freq)
    else:
        sim_train_answer = list(np.array(train_answer)[list(set(query_duplicates['idx_1']))])
        sim_test_answer = list(np.array(test_answer)[list(set(query_duplicates['idx_2']))])
        if len(sim_test_answer) > 0:
            answer_similarity_table = calculate_similarity_table(sim_train_answer, sim_test_answer, metric_name, embedding_model)
            answer_duplicates = answer_similarity_table[answer_similarity_table['similarity'] >= similarity_threshold]
            answer_duplicate_rate = len(set(answer_duplicates['idx_2'])) / len(sim_test_answer)
            new_answer_rate = 1 - answer_duplicate_rate
        else:
            new_answer_rate = 0

            
    precomputed = {'new_query_rate': deepcopy(new_query_rate),
                   'new_answer_rate': deepcopy(new_answer_rate)}

    report = report_valtest_statistics_dynamics_agent(new_query_rate,
                                                      new_answer_rate,
                                                      semaphore_threshold,
                                                      greater_is_better,
                                                      is_info)
    return {'report': report,
            'precomputed': precomputed}


def calculate_similarity_table(documents_1: tp.List[str],
                               documents_2: tp.List[str],
                               metric_name: str='rouge',
                               embedding_model: tp.Optional[tp.Any]=None) -> pd.DataFrame:
    '''
    Функция создания таблицы семантической схожести между тексами. 
    Таблица схожести имеет 3 колонки:
    idx_1 - индекс предложения из documents_1
    idx_2 - индекс предложения из documents_2
    similarity - семантическое сходство между соответствующими текстами
    Parameters:
        documents_1: первый список текстов
        documents_2: второй список текстов
        metric_name: метрика для расчета близости между текстами. Для расчета доступны метрики 'cos_sim', 'bleu', 'rouge', 'meteor' и 'bert_score'
        embedding_model: Класс-обертка над эмбеддинг моделью. Модель обязана иметь метод get_embedding.
    Return:
        Таблицу семантической схожести между всеми парами текстов.
    '''
    if metric_name == 'cos_sim':
        embeddings_dict_1 = embedding_model.get_embedding(documents_1)
        embeddings_dict_2 = embedding_model.get_embedding(documents_2)
        
        embeddings_1 = [emb['embedding'] for emb in embeddings_dict_1]
        embeddings_2 = [emb['embedding'] for emb in embeddings_dict_2]
        
        similarity_values = sklearn.metrics.pairwise.cosine_similarity(embeddings_1, embeddings_2)
        idx_1, idx_2 = np.meshgrid(np.arange(similarity_values.shape[0]), np.arange(similarity_values.shape[1]), indexing='ij')
        sim_list = np.column_stack([idx_1.ravel(), idx_2.ravel(), similarity_values.ravel()])
    else:
        sim_list = []
        for i, document_1 in enumerate(documents_1):
            for j, document_2 in enumerate(documents_2):
                similarity = metrics2func_agent[metric_name]([document_1],document_2)
                sim_list.append([i, j, similarity])

    similarity_table = pd.DataFrame(sim_list, columns = ['idx_1', 'idx_2', 'similarity']).sort_values(by='similarity', ascending=False)
    similarity_table['idx_1'] = similarity_table['idx_1'].astype(int)
    similarity_table['idx_2'] = similarity_table['idx_2'].astype(int)

    return similarity_table.reset_index(drop=True)


def get_token_freq(tokens_dict_train: tp.Dict[int, int],
                   tokens_dict_test: tp.Dict[int, int]) -> tp.Tuple[float, float]:
    '''
    Считает доли каждого токена и долю новых на test по словарям встречаемости
    Parameters:
        tokens_dict_train: Список строк в обучающем наборе
        tokens_dict_test: Список строк в тестовом наборе 
    Return:
        Частота каждого токена в обучающем и тестовом наборе строк
    '''
    unite = set(tokens_dict_train.keys()) | set(tokens_dict_test.keys())
    train_token_freq, test_token_freq = [], []
    for key in list(unite):
        train_token_freq.append(tokens_dict_train[key])
        test_token_freq.append(tokens_dict_test[key])
    train_token_freq, test_token_freq = [np.array(tf) / np.sum(tf) for tf in [train_token_freq, test_token_freq]]
    except_keys = list(set(tokens_dict_test.keys()) - set(tokens_dict_train.keys()))
    return train_token_freq, test_token_freq


def psi(train: tp.Iterable[float],
        test: tp.Iterable[float]) -> float:
    '''
    Считает PSI 
    Parameters:
        train: Первая выборка
        test: Вторая выборка
    Return:
        Значение PSI
    '''
    EPS = 1e-14
    train = np.array(train)
    test = np.array(test)
    train = (train + EPS) / np.sum(train + EPS)
    test = (test + EPS) / np.sum(test + EPS)
    return np.sum((train - test) * np.log(train / test))

