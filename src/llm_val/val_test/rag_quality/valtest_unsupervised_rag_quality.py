import os
import re
import numpy as np
import pandas as pd
import typing as tp

from llm_val.ragexecutor.RAGExecutor import RAGExecutor
from llm_val.utils.report_helper import semaphore_by_threshold


def report_valtest_unsupervised_rag_quality(hit_rate_1: float,
                                            hit_rate_5: float,
                                            average_place: float,
                                            semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]]={'hit_rate_1': (0.3, 0.4),
                                                                                                       'hit_rate_5': (0.5, 0.7)},
                                            is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        hit_rate_1: Значение метрики Hit Rate @ 1
        hit_rate_5: Значение метрики Hit Rate @ 5
        average_place: Значение информативной метрики Average Place
        semaphore_sthreshold: Словарь Tuple с порогами для метрик Hit Rate; 
                              Должен иметь ключи hit_rate_1 и hit_rate_5
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_dict = {}
    hit_rate_1_semaphore = semaphore_by_threshold(hit_rate_1, 
                                                  threshold=semaphore_threshold['hit_rate_1'],
                                                  greater_is_better=True,
                                                  left_border_is_bigger=True,
                                                  right_border_is_bigger=True)
    hit_rate_5_semaphore = semaphore_by_threshold(hit_rate_5, 
                                                  threshold=semaphore_threshold['hit_rate_5'],
                                                  greater_is_better=True,
                                                  left_border_is_bigger=True,
                                                  right_border_is_bigger=True)
    
    semaphore = 'red' if 'red' in [hit_rate_1, hit_rate_5] else 'yellow' if all([col == 'yellow' for col in [hit_rate_1, hit_rate_5]]) else 'green'
    metric_names = ['hit_rate_1', 'hit_rate_5', 'average_place']
    metric_values = [hit_rate_1, hit_rate_5, average_place]
    metric_semaphores = [hit_rate_1_semaphore, hit_rate_5_semaphore, 'gray']
    res_df = pd.DataFrame({'Метрика': metric_names,
                           'Значение': metric_values,
                           'Светофор': metric_semaphores})        
    res_dict['semaphore'] = 'gray' if is_info else semaphore
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [res_df]
    return res_dict


def valtest_unsupervised_rag_quality(executor: RAGExecutor,
                                     question_generation_model: tp.Any,
                                     semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]]={'hit_rate_1': (0.3, 0.4),
                                                                                                'hit_rate_5': (0.5, 0.7)},
                                     critic_model: tp.Optional[tp.Any]=None,
                                     hit_rate_relevance_threshold: int=0,
                                     is_info: bool=False,
                                     hit_rate_1: tp.Optional[float]=None,
                                     hit_rate_5: tp.Optional[float]=None,
                                     average_place: tp.Optional[float]=None,
                                     **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на анализ влияния размера пассажей и величины перекрытий на качество работы RAG-системы
    Parameters:
        executor: Объект класса RAGExecutor
        question_generation_model: Объект модели для генерации вопросов по чанкам
        semaphore_threshold: Словарь с порогами светофоров для каждой метрики;
                             Должен содержать ключи hit_rate_1 и hit_rate_5
        critic_model: Опциональная модель-критик, которая будет оценивать релевантность чанка для ответа на вопрос
        hit_rate_relevance_threshold: Порог релевантности, начиная с которого вопросы будут учитываться при подсчете Hit Rate
        is_info: True, если тест информативный
        hit_rate_1: Опциональное значение метрики Hit Rate @ 1
        hit_rate_5: Опциональное значение метрики Hit Rate @ 5
        average_place: Опциональное значение метрики Average Place
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if not hit_rate_1 or not hit_rate_5 or not average_place:
        params = executor.get_parameters()
        passages = executor.get_passages_with_embeddings()['documents']
        question_generation_queries = [f'\n\nКонтекст: {p} \nТвой ответ:::' for p in passages]
        generated_questions = question_generation_model.predict(question_generation_queries)
        queries = [parse_question(q) for q in generated_questions]
        if hit_rate_relevance_threshold > 0 and critic_model:
            relevance = [get_relevance(queries[idx], passages[idx], critic_model) for idx in range(len(queries))]
            queries = pd.Series(queries).loc[pd.Series(relevance) >= hit_rate_relevance_threshold].to_list()
            passages = pd.Series(passages).loc[pd.Series(relevance) >= hit_rate_relevance_threshold].to_list()
        if not hit_rate_1:
            hit_rate_1 = get_hit_rate_at_K_score(executor=executor,
                                                 queries=queries,
                                                 passages=passages,
                                                 topK=1)
        if not hit_rate_5:
            hit_rate_5 = get_hit_rate_at_K_score(executor=executor,
                                                 queries=queries,
                                                 passages=passages,
                                                 topK=5)
        if not average_place:
            average_place = get_average_place_score(executor=executor,
                                                    queries=queries,
                                                    passages=passages,
                                                    topK=5)
    report = report_valtest_unsupervised_rag_quality(hit_rate_1,
                                                     hit_rate_5,
                                                     average_place,
                                                     semaphore_threshold=semaphore_threshold,
                                                     is_info=is_info)
    precomputed = {'hit_rate_1': hit_rate_1,
                   'hit_rate_5': hit_rate_5,
                   'average_place': average_place}
    return {'report': report,
            'precomputed': precomputed}


def get_hit_rate_at_K_score(executor: tp.Any,
                            passages: tp.List[str],
                            queries: tp.List[str],
                            topK: int=1) -> float:
    '''
    Функция для расчета Hit Rate @ K метрики качества
    Parameters:
        executor: Объект класса RAGExecutor
        passages: Пассажи, которые должны быть в top-K релевантных пассажей для заданных вопросов        
        queries: Словарь, включающий в себя списки questions, true_answers и true_answers_embeddings
        topK: Число релевантных пассажей, на которых будет строиться RAG-система
    Return:
        Значение метрики Hit Rate @ K 
    '''
    topK_passages = executor.get_topK_passages(baseline=True,
                                               questions=queries,
                                               topK=topK)
    hit_rate_at_K = sum([int(passages[idx] in topK_passages[query]['documents']) for idx, query in enumerate(queries)]) / len(passages)
    return hit_rate_at_K


def get_average_place_score(executor: tp.Any,
                            passages: tp.List[str],
                            queries: tp.List[str],
                            topK: int=1) -> float:
    '''
    Функция для расчета average place метрики
    Parameters:
        executor: Объект класса RAGExecutor
        passages: Пассажи, которые должны быть в top-K релевантных пассажей для заданных вопросов
        queries: Словарь, включающий в себя списки questions, true_answers и true_answers_embeddings
        topK: Число релевантных пассажей, на которых будет строиться RAG-система
    Return:
        Значение метрики average place
    '''
    topK_passages = executor.get_topK_passages(baseline=True,
                                               questions=queries,
                                               topK=topK)
    average_place = sum([np.argmax([passages[idx] == p for p in topK_passages[query]['documents']]) + 1 for idx, query in enumerate(queries)]) / len(passages)
    return average_place
    
    
def get_relevance(question: str,
                  passage: str,
                  model: tp.Any) -> int:
    '''
    Функция для расчета релевантности пассажа для ответа на вопрос.
    Parameters:
        question: Вопрос
        passage: Пассаж
        model: Инстанс модели-критика
    Return:
        Оценка релевантности от 1 до 5
    '''
    query = '\n\nКонтекст:' + passage + '\nВопрос:' + question + '\nТвой ответ:::'
    response = parse_relevance(model.predict([query])[0])
    return response


def parse_question(response: str) -> str:
    '''
    Функция-парсер ответа модели которая генерирует вопрос
    Parameters:
        response: Ответ модели
    Return:
        Сгенерированный вопрос
    '''
    question_start = response.find('Вопрос:')
    question_end = response.find('\nОтвет:')
    return response[question_start + len('Вопрос:'): question_end]


def parse_relevance(response: str) -> str:
    '''
    Функция-парсер ответа модели которая оценивает релевантность
    Parameters:
        response: Сырой ответ модели
    Return:
        Релевантность
    '''
    relevance_start = response.find('Оценка:')
    int_match = re.search(r'\b\d+\b', response[relevance_start: -1])
    relevance = int(int_match.group()) if int_match else 0
    return relevance