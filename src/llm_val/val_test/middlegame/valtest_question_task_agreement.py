import re
import numpy as np
import pandas as pd
import typing as tp
from copy import deepcopy
from tqdm.auto import tqdm
from functools import partial

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold

def report_valtest_question_task_agreement(incorrect_queries: pd.DataFrame,
                                           total_queries: int,
                                           semaphore_threshold: tp.Tuple[float]=(0.0, 0.05),
                                           is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        incorrect_queries: Запросы, в которых GC нашел ошибку
        total_queries: Общее количество запросов на проверке
        semaphore_threshold: Пороги доли ошибочных запросов для светофоров
        is_info: Если True, то тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    incorrect_counts = incorrect_queries.shape[0]
    error_counts = pd.Series(sum(incorrect_queries['errors'].to_list(), ())).value_counts()
    error_counts.loc['Всего ошибок'] = error_counts.sum()
    error_counts.loc['Запросов с ошибками'] = incorrect_counts
    errors_share = incorrect_counts / total_queries
    error_counts.loc['Доля запросов с ошибками'] = errors_share
    error_df = error_counts.reset_index()
    error_df.columns = ['Тип ошибки', 'Значение']
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else semaphore_by_threshold(errors_share,
                                                                          threshold=semaphore_threshold,
                                                                          greater_is_better=False)
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [error_df]
    return res_dict


def valtest_question_task_agreement(sampler: Sampler,
                                    task_type: str,
                                    model: tp.Any,
                                    system_prompt: str,
                                    answer_parser: tp.Callable[str, tp.Tuple[str]],
                                    data_type: str='test',
                                    ok_answer_key: str='NONE',
                                    semaphore_threshold: tp.Tuple[float]=(0.0, 0.05),
                                    verbose_predictions: bool=False,
                                    is_info: bool=False,
                                    **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на соответствие вопросов в датасете его назначению (RuBQ, LongAnswer)
    Parameters:
        sampler: Объект класса Sampler
        task_type: Формат тестовой выборки, должен быть из ["rubq", "longanswer"]
        model: Инстанс модели, например, MultiThreadGigaModel
        system_prompt: Системный промпт для GC, который будет оценивать ошибки в формировании датасета
        answer_parser: Функция-парсер ответов GC
        data_type: Тип данных, вопросы которого проверяются в тесте. Hint: название атрибута в sampler'е
        ok_answer_key: Код, который должен выдать GC, если все ок
        semaphore_threshold: Пороги доли ошибочных запросов для светофоров
        verbose_predictions: tqdm'ить ли предсказания GC
        is_info: Если True, то тест информативный
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    tqdm.pandas()
        
    data = deepcopy(getattr(sampler, data_type)['X'])
    if task_type.lower() == 'longanswer':
        if verbose_predictions:
            print('Creating LongAnswer answers...')
            data['true_answer'] = data['question'].progress_apply(generate_answer, model=model) 
        else:
            data['true_answer'] = data['question'].apply(generate_answer, model=model)
    
    query = data.apply(lambda x: system_prompt + f'\nQUESTION: {x["question"]} \nANSWER: {x["true_answer"]}', axis=1)
    if verbose_predictions:
        print('Checking errors in queries...')
        errors = query.progress_apply(generate_answer, model=model).apply(answer_parser, ok_answer_key=ok_answer_key) 
    else:
        errors = query.apply(generate_answer, model=model).apply(answer_parser, ok_answer_key=ok_answer_key)
    
    data['errors'] = errors
    incorrect_queries = data.loc[data['errors'] != tuple([ok_answer_key])][['question', 'true_answer', 'errors']]
    total_queries = data.shape[0]
    report = report_valtest_question_task_agreement(incorrect_queries,
                                                    total_queries=total_queries,
                                                    semaphore_threshold=semaphore_threshold,
                                                    is_info=is_info)
    precomputed = {'incorrect_queries': incorrect_queries}
    return {'report': report,
            'precomputed': precomputed}

def generate_answer(query: str,
                    model: tp.Any) -> str:
    '''
    Генератор ответов GС
    Parameters:
        query: Запрос в GC
        model: Инстанс GC
    Return:
        Ответ GC
    '''
    model_answer = model.predict([query])[0]
    return model_answer 