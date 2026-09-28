import os
import time
import numpy as np
import pandas as pd
import typing as tp
from tqdm import tqdm
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
from langchain_chroma import Chroma
from langchain_community.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters.character import CharacterTextSplitter

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.Embedder.Embedder import Embedder
from llm_val.ragexecutor.RAGExecutor import RAGExecutor
from llm_val.utils.metrics_rag import cosine_distance, eval_function


def report_valtest_rag_quality_by_length(stats_df1: pd.DataFrame,
                                         stats_df2: pd.DataFrame,
                                         stats_df3: pd.DataFrame,
                                         figure: tp.Any) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        stats_df1: Таблица результатов теста в зависимости от размера пассажей
        stats_df2: Таблица результатов теста в зависимости от величины перекрытий
        stats_df3: Таблица результатов теста в зависимости от размера пассажей и величины перекрытий
        figure: График зависимости метрики качества от числа взятых начальных лексических единиц
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_plots'] = [figure]
    res_dict['result_dataframes'] = [stats_df1, stats_df2, stats_df3]
    return res_dict


def valtest_rag_quality_by_length(sampler: Sampler,
                                  executor: tp.Any,
                                  data_type: str,
                                  test_by: str,
                                  use_llm: bool=False,
                                  eval_metric: str='MeanCosDist',
                                  cos_dist_threshold: float=0.25,
                                  **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на анализ влияния размера пассажей и величины перекрытий на качество работы RAG-системы
    Parameters:
        sampler: Объект класса Sampler
        executor: Объект класса RAGExecutor
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        test_by: Название сущности, по которой будет проводиться оценка близости top-K релевантных пассажей
            Hint: При наличии в данных правильных (ground truth) ответов в параметр `test_by` стоит передавать значение `true_answer`.
                  При отсутствии такого ответа в данных в параметр `test_by` можно передавать любое значение из данных, 
                  по которому будет проводиться тест
        use_llm: Флаг, отвечающий за то, используется ли ЛЛМ в процессе проведения теста.
            Hint: Если False, то тест проводится с учетом косинусных расстояний между top-K релевантными для вопроса пассажами и
                  тем, что указано в параметре `test_by`. 
                  Если True, то тест проводится с учетом косинусных расстояний между ответом ЛЛМ на основе top-K релевантных пассажей и
                  тем, что указано в параметре `test_by`.
        eval_metric: Название метрики, согласно которой будет оцениваться качество RAG-системы
            Hint: На данный момент реализованы метрики `MeanCosDist` и `AccuracyCosDist`
        cos_dist_threshold: Порог значения косинусного расстояния, при превышении которого набор релевантных пассажей для query не будет считаться релевантным.
            Hint: Используется только при eval_metric = `AccuracyCosDist`
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    df = deepcopy(getattr(sampler, data_type)['X'])
    params = executor.get_parameters()
    queries = {}
    queries['systems'] = df['system'].values.tolist()
    queries['questions'] = df['question'].values.tolist()
    queries['true_answers'] = df[test_by].values.tolist()
    tqdm.pandas(desc='Calculating true answers embeddings')
    queries['true_answers_embeddings'] = df[test_by].progress_apply(lambda x: executor.embed_text(x)).values.tolist()
    baseline_score = get_baseline_score(executor=executor,
                                        queries=queries,
                                        use_llm=use_llm,
                                        topK=params['topK'],
                                        eval_metric=eval_metric,
                                        cos_dist_threshold=cos_dist_threshold)
    stats_df1 = get_stats_by_chunk_size(executor=executor,
                                        queries=queries,
                                        chunk_size_list=params['chunk_size_list'],
                                        chunk_overlap=params['chunk_overlap'],
                                        use_llm=use_llm,
                                        topK=params['topK'],
                                        eval_metric=eval_metric,
                                        cos_dist_threshold=cos_dist_threshold)
    stats_df2 = get_stats_by_chunk_overlap(executor=executor,
                                           queries=queries,
                                           chunk_size=params['chunk_size'],
                                           chunk_overlap_list=params['chunk_overlap_list'],
                                           use_llm=use_llm,
                                           topK=params['topK'],
                                           eval_metric=eval_metric,
                                           cos_dist_threshold=cos_dist_threshold)
    stats_df3 = get_stats_by_chunk_size_and_overlap(executor=executor,
                                                    queries=queries,
                                                    chunk_size_list=params['chunk_size_list'],
                                                    chunk_overlap_list=params['chunk_overlap_list'],
                                                    use_llm=use_llm,
                                                    topK=params['topK'],
                                                    eval_metric=eval_metric,
                                                    cos_dist_threshold=cos_dist_threshold)
    figure = plot_rag_quality_metrics(stats_df1=stats_df1, 
                                      stats_df2=stats_df2, 
                                      stats_df3=stats_df3, 
                                      eval_metric=eval_metric,
                                      chunk_size=params['chunk_size'], 
                                      chunk_overlap=params['chunk_overlap'], 
                                      topK=params['topK'], 
                                      baseline_score=baseline_score)
    report = report_valtest_rag_quality_by_length(stats_df1=stats_df1,
                                                  stats_df2=stats_df2,
                                                  stats_df3=stats_df3,
                                                  figure=figure)
    precomputed = {'baseline_score': baseline_score}
    return {'report': report,
            'precomputed': precomputed}


def get_baseline_score(executor: tp.Any,
                       queries: tp.Dict[str, tp.List],
                       use_llm: bool=False,
                       topK: int=4,
                       eval_metric: str='MeanCosDist',
                       cos_dist_threshold: float=0.25) -> float:
    '''
    Функция для расчета baseline метрики качества
    Parameters:
        executor: Объект класса RAGExecutor
        queries: Словарь, включающий в себя списки questions, true_answers и true_answers_embeddings
        use_llm: Флаг, отвечающий за то, используется ли ЛЛМ в процессе проведения теста.
            Hint: Если False, то тест проводится с учетом косинусных расстояний между top-K релевантными для вопроса пассажами и
                  тем, что указано в параметре `test_by`. 
                  Если True, то тест проводится с учетом косинусных расстояний между ответом ЛЛМ на основе top-K релевантных пассажей и
                  тем, что указано в параметре `test_by`.
        topK: Число релевантных пассажей, на которых будет строиться RAG-система
        eval_metric: Название метрики, согласно которой будет оцениваться качество RAG-системы
        cos_dist_threshold: Порог значения косинусного расстояния, при превышении которого набор релевантных пассажей для query не будет считаться релевантным.
            Hint: Используется только при eval_metric = `AccuracyCosDist`
    Return:
        Значение baseline метрики качества
    '''
    if use_llm:
        questions_answers = executor.get_llm_answers(baseline=True,
                                                     questions=queries['questions'],
                                                     systems=queries['systems'],
                                                     topK=topK)
        questions_answers = get_true_answers_distances_to_llm_answers(queries=queries,
                                                                      questions_answers=questions_answers)
        topK_cosine_distances = [value['cosine_distance'] for key, value in questions_answers.items()]
    else:
        passages = executor.get_passages_with_embeddings(baseline=True)
        topK_passages = executor.get_topK_passages(baseline=True,
                                                   questions=queries['questions'],
                                                   topK=topK)
        topK_passages = get_true_answers_distances_to_topK_passages(queries=queries,
                                                                    topK_passages=topK_passages)
        topK_cosine_distances = [value['mean_cosine_distance'] for key, value in topK_passages.items()]
    eval_metric_value = eval_function[eval_metric](topK_cosine_distances, cos_dist_threshold)
    return eval_metric_value

    
def get_stats_by_chunk_size(executor: tp.Any,
                            queries: tp.Dict[str, tp.List[tp.Any]],
                            chunk_size_list: tp.List[int],
                            chunk_overlap: int,
                            use_llm: bool=False,
                            topK: int=4,
                            eval_metric: str='MeanCosDist',
                            cos_dist_threshold: float=0.25) -> pd.DataFrame:
    '''
    Функция для проведения части теста, в которой оценивается качество RAG-системы в зависимости от размера пассажей
    Parameters:
        executor: Объект класса RAGExecutor
        queries: Словарь, включающий в себя списки questions, true_answers и true_answers_embeddings
        chunk_size_list: Список размеров пассажей, по которым будут проводиться итерации в тесте
        chunk_overlap: Величина перекрытий между пассажами для теста с нефиксированным размером пассажей
        use_llm: Флаг, отвечающий за то, используется ли ЛЛМ в процессе проведения теста.
            Hint: Если False, то тест проводится с учетом косинусных расстояний между top-K релевантными для вопроса пассажами и
                  тем, что указано в параметре `test_by`. 
                  Если True, то тест проводится с учетом косинусных расстояний между ответом ЛЛМ на основе top-K релевантных пассажей и
                  тем, что указано в параметре `test_by`.
        topK: Число релевантных пассажей, на которых будет строиться RAG-система
        eval_metric: Название метрики, согласно которой будет оцениваться качество RAG-системы
        cos_dist_threshold: Порог значения косинусного расстояния, при превышении которого набор релевантных пассажей для query не будет считаться релевантным.
            Hint: Используется только при eval_metric = `AccuracyCosDist`
    Return:
        Таблица с результатами части теста, в которой оценивается качество RAG-системы в зависимости от размера пассажей
    '''
    stats_df = init_stats_df('chunk size', eval_metric)
    for chunk_size in tqdm(chunk_size_list, desc='Iterating Through Chunks Size'):
        passages = executor.get_passages_with_embeddings(baseline=False,
                                                         chunk_size=chunk_size,
                                                         chunk_overlap=chunk_overlap)
        chunks_len_list = [len(passage) for passage in passages['documents']]
        if use_llm:
            questions_answers = executor.get_llm_answers(baseline=False,
                                                         questions=queries['questions'],
                                                         systems=queries['systems'],
                                                         topK=topK,
                                                         chunk_size=chunk_size,
                                                         chunk_overlap=chunk_overlap)
            questions_answers = get_true_answers_distances_to_llm_answers(queries=queries,
                                                                          questions_answers=questions_answers)
            topK_cosine_distances = [value['cosine_distance'] for key, value in questions_answers.items()]
        else:
            topK_passages = executor.get_topK_passages(baseline=False,
                                                       questions=queries['questions'],
                                                       topK=topK,
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
            topK_passages = get_true_answers_distances_to_topK_passages(queries=queries,
                                                                        topK_passages=topK_passages)
            topK_cosine_distances = [value['mean_cosine_distance'] for key, value in topK_passages.items()]
        eval_metric_value = eval_function[eval_metric](topK_cosine_distances, cos_dist_threshold)
        stats_df = update_stats_df(stats_df, eval_metric_value, topK, chunk_size, chunk_overlap, chunks_len_list)
    return stats_df


def get_stats_by_chunk_overlap(executor: tp.Any,
                               queries: tp.Dict[str, tp.List[tp.Any]],
                               chunk_size: int,
                               chunk_overlap_list: tp.List[int],
                               use_llm: bool=False,
                               topK: int=4,
                               eval_metric: str='MeanCosDist',
                               cos_dist_threshold: float=0.25) -> pd.DataFrame:
    '''
    Функция для проведения части теста, в которой оценивается качество RAG-системы в зависимости от величины перекрытий между пассажами
    Parameters:
        executor: Объект класса RAGExecutor
        queries: Словарь, включающий в себя списки questions, true_answers и true_answers_embeddings
        chunk_size: Размер пассажей для теста с нефиксированной величиной перекрытий между пассажами
        chunk_overlap_list: Список величин перекрытий между пассажами, по которым будут проводиться итерации в тесте
        use_llm: Флаг, отвечающий за то, используется ли ЛЛМ в процессе проведения теста.
            Hint: Если False, то тест проводится с учетом косинусных расстояний между top-K релевантными для вопроса пассажами и
                  тем, что указано в параметре `test_by`. 
                  Если True, то тест проводится с учетом косинусных расстояний между ответом ЛЛМ на основе top-K релевантных пассажей и
                  тем, что указано в параметре `test_by`.
        topK: Число релевантных пассажей, на которых будет строиться RAG-система
        eval_metric: Название метрики, согласно которой будет оцениваться качество RAG-системы
        cos_dist_threshold: Порог значения косинусного расстояния, при превышении которого набор релевантных пассажей для query не будет считаться релевантным.
            Hint: Используется только при eval_metric = `AccuracyCosDist`
    Return:
        Таблица с результатами части теста, в которой оценивается качество RAG-системы в зависимости от величины перекрытий между пассажами
    '''
    stats_df = init_stats_df('chunk overlap', eval_metric)
    for chunk_overlap in tqdm(chunk_overlap_list, desc='Iterating Through Chunks Overlap'):
        passages = executor.get_passages_with_embeddings(baseline=False,
                                                         chunk_size=chunk_size,
                                                         chunk_overlap=chunk_overlap)
        chunks_len_list = [len(passage) for passage in passages['documents']]
        if use_llm:
            questions_answers = executor.get_llm_answers(baseline=False,
                                                         questions=queries['questions'],
                                                         systems=queries['systems'],
                                                         topK=topK,
                                                         chunk_size=chunk_size,
                                                         chunk_overlap=chunk_overlap)
            questions_answers = get_true_answers_distances_to_llm_answers(queries=queries,
                                                                          questions_answers=questions_answers)
            topK_cosine_distances = [value['cosine_distance'] for key, value in questions_answers.items()]
        else:
            topK_passages = executor.get_topK_passages(baseline=False,
                                                       questions=queries['questions'],
                                                       topK=topK,
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
            topK_passages = get_true_answers_distances_to_topK_passages(queries=queries,
                                                                        topK_passages=topK_passages)
            topK_cosine_distances = [value['mean_cosine_distance'] for key, value in topK_passages.items()]
        eval_metric_value = eval_function[eval_metric](topK_cosine_distances, cos_dist_threshold)
        stats_df = update_stats_df(stats_df, eval_metric_value, topK, chunk_overlap, chunk_size, chunks_len_list)
    return stats_df


def get_stats_by_chunk_size_and_overlap(executor: tp.Any,
                                        queries: tp.Dict[str, tp.List[tp.Any]],
                                        chunk_size_list: tp.List[int],
                                        chunk_overlap_list: tp.List[int],
                                        use_llm: bool=False,
                                        topK: int=4,
                                        eval_metric: str='MeanCosDist',
                                        cos_dist_threshold: float=0.25) -> pd.DataFrame:
    '''
    Функция для проведения части теста, в которой оценивается качество RAG-системы в зависимости от размера пассажей и величины перекрытий между пассажами
    Parameters:
        executor: Объект класса RAGExecutor
        queries: Словарь, включающий в себя списки questions, true_answers и true_answers_embeddings
        chunk_size_list: Список размеров пассажей, по которым будут проводиться итерации в тесте
        chunk_overlap_list: Список величин перекрытий между пассажами, по которым будут проводиться итерации в тесте
        use_llm: Флаг, отвечающий за то, используется ли ЛЛМ в процессе проведения теста.
            Hint: Если False, то тест проводится с учетом косинусных расстояний между top-K релевантными для вопроса пассажами и
                  тем, что указано в параметре `test_by`. 
                  Если True, то тест проводится с учетом косинусных расстояний между ответом ЛЛМ на основе top-K релевантных пассажей и
                  тем, что указано в параметре `test_by`.
        topK: Число релевантных пассажей, на которых будет строиться RAG-система
        eval_metric: Название метрики, согласно которой будет оцениваться качество RAG-системы
        cos_dist_threshold: Порог значения косинусного расстояния, при превышении которого набор релевантных пассажей для query не будет считаться релевантным.
            Hint: Используется только при eval_metric = `AccuracyCosDist`
    Return:
        Таблица с результатами части теста, в которой оценивается качество RAG-системы в зависимости от размера пассажей и величины перекрытий между пассажами
    '''
    stats_df = init_stats_df('both', eval_metric, chunk_size_list, chunk_overlap_list)
    iter_list = [(chunk_size, chunk_overlap) for chunk_size in chunk_size_list for chunk_overlap in chunk_overlap_list]
    for chunk_size, chunk_overlap in tqdm(iter_list, desc='Iterating Through Chunks Size & Chunks Overlap'):
        passages = executor.get_passages_with_embeddings(baseline=False,
                                                         chunk_size=chunk_size,
                                                         chunk_overlap=chunk_overlap)
        chunks_len_list = [len(passage) for passage in passages['documents']]
        if use_llm:
            questions_answers = executor.get_llm_answers(baseline=False,
                                                         questions=queries['questions'],
                                                         systems=queries['systems'],
                                                         topK=topK,
                                                         chunk_size=chunk_size,
                                                         chunk_overlap=chunk_overlap)
            questions_answers = get_true_answers_distances_to_llm_answers(queries=queries,
                                                                          questions_answers=questions_answers)
            topK_cosine_distances = [value['cosine_distance'] for key, value in questions_answers.items()]
        else:
            topK_passages = executor.get_topK_passages(baseline=False,
                                                       questions=queries['questions'],
                                                       topK=topK,
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
            topK_passages = get_true_answers_distances_to_topK_passages(queries=queries,
                                                                        topK_passages=topK_passages)
            topK_cosine_distances = [value['mean_cosine_distance'] for key, value in topK_passages.items()]
        eval_metric_value = eval_function[eval_metric](topK_cosine_distances, cos_dist_threshold)
        stats_df = update_stats_df(stats_df, eval_metric_value, topK, [chunk_size, chunk_overlap])
    return stats_df


def get_true_answers_distances_to_topK_passages(queries: tp.Dict[str, tp.List[tp.Any]],
                                                topK_passages: tp.Dict[str, tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Функция для получения косинусных расстояний и средних косинусных расстояний между true_answers и top-K релевантными пассажами
    Parameters:
        queries: Словарь, включающий в себя списки questions, true_answers и true_answers_embeddings
        topK_passages: Словарь top-K пассажей с их эмбеддингами
    Return:
        Словарь top-K пассажей с их эмбеддингами, косинусными расстояниями и средними косинусными расстояниями
    '''
    for question, true_answer_embedding in zip(queries['questions'], queries['true_answers_embeddings']):
        topK_passages[question]['cosine_distances'] = [cosine_distance(true_answer_embedding, passage_embedding) for 
                                                       passage_embedding in topK_passages[question]['embeddings']]
        topK_passages[question]['mean_cosine_distance'] = np.mean(topK_passages[question]['cosine_distances'])
    return topK_passages


def get_true_answers_distances_to_llm_answers(queries: tp.Dict[str, tp.List[tp.Any]],
                                              questions_answers: tp.Dict[str, tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Функция для получения косинусных расстояний между true_answers и llm_answers
    Parameters:
        queries: Словарь, включающий в себя списки questions, true_answers и true_answers_embeddings
        questions_answers: Словарь с ответами ЛЛМ на вопросы и их эмбеддингами
    Return:
        Словарь llm_answers с их эмбеддингами и косинусными расстояниями до true_answers
    '''
    for question, true_answer_embedding in zip(queries['questions'], queries['true_answers_embeddings']):
        questions_answers[question]['cosine_distance'] = cosine_distance(true_answer_embedding, 
                                                                         questions_answers[question]['llm_answer_embedding'])
    return questions_answers


def init_stats_df(iter_through: str, 
                  eval_metric: str, 
                  chunk_size_list: tp.Optional[tp.List[int]]=None, 
                  chunk_overlap_list: tp.Optional[tp.List[int]]=None) -> pd.DataFrame:
    '''
    Вспомогательная функция для инициализации таблиц со статистиками по тесту
    Parameters:
        iter_through: По какому списку в части теста проводятся итерации. Возможные варианты: `chunk size`, `chunk overlap`, `both`
        eval_metric: Название метрики, согласно которой будет оцениваться качество RAG-системы
        chunk_size_list: Список размеров пассажей, по которым будут проводиться итерации в тесте
        chunk_overlap_list: Список величин перекрытий между пассажами, по которым будут проводиться итерации в тесте
    Return:
        Пустая таблица для заполнения статистиками по тесту
    '''
    name_dict = {'chunk size': 'Размер пассажей',
                 'chunk overlap': 'Величина перекрытий'}
    inv_name_dict = {'chunk size': 'Величина перекрытий',
                     'chunk overlap': 'Размер пассажей'}
    if iter_through in name_dict:
        stats_list = [name_dict[iter_through], 
                     'Медианный размер пассажей',
                     '2.5%-й квантиль размера пассажей',
                     '97.5%-й квантиль размера пассажей',
                     inv_name_dict[iter_through],
                     'Число релевантных пассажей',
                     eval_metric]
        stats_df = pd.DataFrame(columns=stats_list).set_index(stats_list[0])
    elif iter_through == 'both':
        idx = 'Размер пассажей'
        col = 'Величина перекрытий'
        idx_list = [[idx]*len(chunk_size_list), chunk_size_list]
        col_list = [[col]*len(chunk_overlap_list), chunk_overlap_list]
        stats_df = pd.DataFrame(index=idx_list, columns=col_list)
    else:
        stats_df = pd.DataFrame()
    return stats_df


def update_stats_df(stats_df: pd.DataFrame, 
                    eval_metric_value: float,
                    topK: int,
                    iter_size: tp.Union[tp.List[int], int], 
                    non_iter_size: tp.Optional[int]=None, 
                    chunks_len_list: tp.Optional[tp.List[int]]=None) -> pd.DataFrame:
    '''
    Вспомогательная функция для заполнения таблиц со статистиками по тесту
    
    Parameters:
        stats_df: Частично заполненная или незаполненная таблица со статистиками
        eval_metric_value: Значение метрики, согласно которой оценивается качество RAG-системы
        topK: Число релевантных пассажей, на которых строится RAG-система
        iter_size: Список (в случае итерации и по размеру пассажей, и по величине перекрытий между пассажами) или одно значение 
            (в случае итерации только по размеру пассажей или по величине перекрытий между пассажами), соответствующее величине
            переменной с именем iter_through из функции `init_stats_df`
        non_iter_size: Значение (в случае итерации только по размеру пассажей или по величине перекрытий между пассажами) или None
            (в случае итерации и по размеру пассажей, и по величине перекрытий между пассажами), соответствующее величине
            обратной переменной к переменной с именем iter_through из функции `init_stats_df`
        chunks_len_list: Список с длинами всех пассажей после нарезки
    Return:
        Заполненная таблица со статистиками по тесту
    '''
    if isinstance(iter_size, int):
        stats_list = list(stats_df.columns)
        stats_df.loc[iter_size, stats_list[0]] = np.median(chunks_len_list)
        stats_df.loc[iter_size, stats_list[1]] = np.quantile(chunks_len_list, 0.025)
        stats_df.loc[iter_size, stats_list[2]] = np.quantile(chunks_len_list, 0.975)
        stats_df.loc[iter_size, stats_list[3]] = non_iter_size
        stats_df.loc[iter_size, stats_list[4]] = topK
        stats_df.loc[iter_size, stats_list[5]] = eval_metric_value
    else:
        stats_df.loc[('Размер пассажей', iter_size[0]), ('Величина перекрытий', iter_size[1])] = eval_metric_value
    return stats_df


def plot_rag_quality_metrics(stats_df1: pd.DataFrame, 
                             stats_df2: pd.DataFrame, 
                             stats_df3: pd.DataFrame, 
                             eval_metric: str, 
                             chunk_size: int, 
                             chunk_overlap: int, 
                             topK: int,
                             baseline_score: float) -> plt.figure:
    '''
    Вспомогательная функция для вывода результирующих графиков со статистиками по тесту
    Parameters:
        stats_df1: Таблица со статистиками по тесту, в котором итерация происходит по размерам чанков
        stats_df2: Таблица со статистиками по тесту, в котором итерация происходит по величинам перекрытий между пассажами
        stats_df3: Таблица со статистиками по тесту, в котором итерация происходит по размерам чанков и величинам перекрытий между пассажами
        eval_metric: Название метрики, согласно которой будет оцениваться качество RAG-системы
        chunk_size: Фиксированное значение размера чанков для теста, в котором итерация происходит по величинам перекрытий между пассажами
        chunk_overlap: Фиксированное значение величины перекрытий между пассажами для теста, в котором итерация происходит по размерам пассажей
        topK: Число релевантных пассажей, на которых будет строиться RAG-система
        baseline_score: Метрика качества RAG в случае baseline модели
    Return:
        Визуализация результатов теста
    '''
    stats_df3_ = stats_df3.copy()
    stats_df3_.index = stats_df3_.index.droplevel(0)
    stats_df3_.columns = stats_df3_.columns.droplevel(0)
    ymin = min([min(stats_df1[eval_metric]), min(stats_df2[eval_metric])]) * 0.9
    ymax = max([max(stats_df1[eval_metric]), max(stats_df2[eval_metric])]) * 1.1
    fig, ax = plt.subplots(1,3, figsize=(20,4))
    ax[0].bar(stats_df1.index.astype(int).astype(str), stats_df1[eval_metric], color='forestgreen', alpha=0.4)
    ax[0].axhline(baseline_score, color='red', ls='dashed', label='Baseline')
    ax[0].set_ylim(ymin, ymax)
    ax[0].set_title(f'В разрезе размера пассажей\nВеличина перекрытий = {chunk_overlap}\nЧисло релевантных пассажей = {topK}')
    ax[0].set_xlabel('Размер пассажей')
    ax[0].set_ylabel(eval_metric)
    ax[0].legend()
    ax[1].bar(stats_df2.index.astype(int).astype(str), stats_df2[eval_metric], color='forestgreen', alpha=0.4)
    ax[1].axhline(baseline_score, color='red', ls='dashed', label='Baseline')
    ax[1].set_ylim(ymin, ymax)
    ax[1].set_title(f'В разрезе величины перекрытий\nРазмер пассажей = {chunk_size}\nЧисло релевантных пассажей = {topK}')
    ax[1].set_xlabel('Величина перекрытий')
    ax[1].set_ylabel(eval_metric)
    ax[1].legend()
    sns.heatmap(stats_df3_.astype(float), cmap='PiYG', ax=ax[2], annot=True, fmt='.4g')
    ax[2].set_title(f'В разрезе размера пассажей\nВ разрезе величины перекрытий\nЧисло релевантных пассажей = {topK}')
    ax[2].set_ylabel('Размер пассажей')
    ax[2].set_xlabel('Величина перекрытий')
    fig.suptitle(eval_metric, fontsize=15, fontweight='bold', y=1.15)
    return fig