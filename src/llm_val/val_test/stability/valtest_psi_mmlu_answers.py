import numpy as np
import typing as tp
import pandas as pd
from copy import deepcopy
from collections import Counter

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold_without_red


def report_valtest_psi_mmlu_answers(answer_psi_values: tp.Dict[str, float],
                                    semaphore_threshold: float=0.2,
                                    is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        answer_psi_values: Словарь с посчитанными PSI для каждого варианта ответов
        semaphore_threshold: Жёлтый порог PSI для вариантов ответов
        is_info: Если True, то тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    psi_dataframe = pd.DataFrame.from_dict(answer_psi_values, orient='index').reset_index()
    psi_dataframe.columns = ['answer', 'psi_value']
    result_dataframes = [psi_dataframe]
    final_semaphore = semaphore_by_threshold_without_red(answer_psi_values['SUM'], 
                                                         threshold=semaphore_threshold, 
                                                         greater_is_better=False,
                                                         border_is_bigger=True)
    
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else final_semaphore
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = result_dataframes
    return res_dict


def valtest_psi_mmlu_answers(sampler: Sampler,
                             data_type: str='test',
                             semaphore_threshold: float=0.2,
                             is_info: bool=False,
                             answer_psi_values: tp.Optional[tp.Dict[str, float]]=None,
                             **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на соответствие распределений вариантов истинных и предсказанных моделью ответов
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет PSI. Hint: название атрибута в sampler'е    
        semaphore_threshold: Жёлтый порог PSI для варианта ответа
        is_info: Если True, то тест информативный
        answer_psi_values: Опциональный Dict значений PSI по вариантам ответов и их суммой с ключом 'SUM'
            Если answer_psi_values не None, то подсчет PSI не производится, и на его основе выставляется светофор
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if answer_psi_values is None:
        data = getattr(sampler, data_type)['X']
        answer_options = sorted(set(data['answer_options'].apply(lambda x: list(x.keys())).sum()))
        true_answer = data['true_answer']
        model_answer = data['model_answer']
        answer_psi_values = caclulate_psi_answers(true_answer,
                                                  model_answer, 
                                                  answer_options)
    
    precomputed = {'answer_psi_values': answer_psi_values}
    report = report_valtest_psi_mmlu_answers(answer_psi_values,
                                             semaphore_threshold,
                                             is_info)
    
    return {'report': report,
            'precomputed': precomputed}


def caclulate_psi_answers(true_answer: tp.Iterable[str],
                          model_answer: tp.Iterable[str],
                          answer_options: tp.Iterable[str]) -> tp.Dict[str, float]:
    '''
    Функция для подсчета значения PSI по каждому варианту ответа
    Parameters:
        true_answer: Верные ответы на вопросы
        model_answer: Ответы модели на вопросы
        answer_options: Все возможные варианты ответов
    Return:
        Dataframe со значениями PSI для каждого варианта ответов
    '''
    true_answer_counts = Counter(true_answer)
    pred_answer_counts = Counter(model_answer)
    answer_psi_values = {}
    psi_sum = 0
    
    for answer in answer_options:
        true_share = true_answer_counts.get(answer, 0) / len(true_answer)
        pred_share = pred_answer_counts.get(answer, 0) / len(model_answer)
        
        if pred_share == 0 and true_share == 0:
            answer_psi = 0
        elif pred_share == 0 or true_share == 0:
            continue
        else:
            answer_psi = (true_share - pred_share) * np.log(true_share / pred_share)
        
        psi_sum += answer_psi
        answer_psi_values[answer] = answer_psi
    
    answer_psi_values['SUM'] = psi_sum
    return answer_psi_values