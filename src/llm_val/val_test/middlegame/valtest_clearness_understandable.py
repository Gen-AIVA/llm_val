import re
import numpy as np
import typing as tp
import pandas as pd
from tqdm import tqdm
from joblib import load
from copy import deepcopy
from collections import defaultdict

from llm_val.sampler.Sampler import Sampler
from llm_val.model.GigaModel import GigaModel
from llm_val.utils.report_helper import semaphore_by_threshold, proportion_semaphore


def report_valtest_clearness_understandable(result_df: tp.Dict[str, float],
                                            criterions: tp.List[str]=['Логика и ясность', 'Полнота', 'Конкретность', 'Адекватность'],
                                            promt_assessment_semaphore_threshold: tp.Tuple[float, float]=(2.5, 4),
                                            freq_threshold: tp.Tuple[float, float]=(0.2, 0.4),
                                            is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        result_df: Словарь с посчитанными результатами теста на четкость и понятность вопроса
        criterions: Названия критериев, в соответствии с которыми LLM будет оценивать промт
        promt_assessment_semaphore_threshold: Границы светофора для отдельных вопросов (промтов)
        freq_threshold: Доля допустимых желтых и красных светофоров
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    result_df['semaphore'] = result_df['general_assessments'].apply(
        lambda x: semaphore_by_threshold(x,
                                         threshold=promt_assessment_semaphore_threshold,
                                         left_border_is_bigger=True,
                                         right_border_is_bigger=True))
    criterions_dict = dict(zip(range(len(criterions)), criterions))
    result_df = result_df.rename(columns=criterions_dict)
    result_df.loc[result_df['general_assessments'].isna(), 'semaphore'] = 'gray'
    color = proportion_semaphore(result_df['semaphore'],
                                 freq_threshold)    
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [result_df]
    return res_dict


def valtest_clearness_understandable(sampler: Sampler,
                                     model: GigaModel,
                                     surround_symbols: tp.Tuple[str, str]=('(((', ')))'),
                                     criterions: tp.List[str]=['Логика и ясность', 'Полнота', 'Конкретность', 'Адекватность'],
                                     promt_assessment_semaphore_threshold: tp.Tuple[float, float]=(2.5, 4),
                                     freq_threshold: tp.Tuple[float, float]=(0.2, 0.4),
                                     is_info: bool=False,
                                     data_clearness_understandable_dict: tp.Optional[dict]=None,
                                     **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест проверки понятности и четкости вопроса: вычисляется оценка с помощью LLM
    Parameters:
        sampler: Объект класса Sampler
        model: LLM, которая в соответствии с инструкцией оценивает промт. 
            Требования к модели описаны в описании теста в ноутбуке 5.2
        surround_symbols: Символы, которые должны окружать числовую оценку LLM для удобства ее парсинга
        criterions: Названия критериев, в соответствии с которыми LLM будет оценивать промт
        promt_assessment_semaphore_threshold: Границы светофора для отдельных вопросов (промтов)
        freq_threshold: Доля допустимых желтых и красных светофоров
        is_info: True, если тест информативный
        data_clearness_understandable_dict: Опциональный словарь с предпосчитанным результатом теста. Если не None, то тест не запускается, и на основе переданного аргумента выставляется светофор
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if data_clearness_understandable_dict is None:
        df_assessments = calculate_assessment(model=model,
                                              sampler=sampler,
                                              surround_symbols=surround_symbols)
    else:
        df_assessments = data_clearness_understandable_dict
    precomputed = {'data_clearness_understandable': deepcopy(df_assessments)}
    report = report_valtest_clearness_understandable(df_assessments,
                                                     criterions,
                                                     promt_assessment_semaphore_threshold,
                                                     freq_threshold,
                                                     is_info=is_info)
    return {'report': report,
            'precomputed': precomputed}


def get_assessment(answer: str,
                   surround_symbols: tp.Tuple[str, str]=('(((', ')))')) -> str:
    '''
    Функция, которая принимает ответ LLM на вход
    и достает оттуда оценку, выделенную символами
    Parameters:
        answer: Оценка промта, полученная от LLM, с комментариями
        surround_symbols: Символы, внутри которых находится численная оценка промта
    Return:
        Численная оценка промта в формате str
    '''
    step = len(surround_symbols[0])
    grade_value = answer[answer.find(surround_symbols[0]) + step: answer.find(surround_symbols[1])]
    return grade_value

    
def calculate_assessment(model: GigaModel,
                         sampler: Sampler,
                         surround_symbols: tp.Tuple[str, str]=('(((', ')))')) -> pd.DataFrame:
    '''
    Функция для расчета оценок для списка вопросов
    Parameters:
        model: LLM, которая в соответствии с инструкцией оценивает промт. 
            Требования к модели описаны в описании теста в ноутбуке 5.2
        sampler: Объект класса Sampler
        surround_symbols: символы, которые должны окружать числовую оценку LLM для удобства ее парсинга
    Return:
        Датафрейм, в котором каждому вопросу в соответствие поставлены численные оценки LLM и ответы LLM с комментарием
    '''
    questions = sampler.test['X']['question']
    answers = model.predict(questions)
    assessments = defaultdict(list)
    for j in answers:
        for i, answer in tqdm(enumerate(answers[j])):
            assessment = get_assessment(answer,
                                        surround_symbols)
            flag_not_generate = True
            n_attempts = 0
            while flag_not_generate:
                if n_attempts == 5:
                    assessment = None
                    print('Не получилось сгенерировать оценку с помощью LLM, попробуйте поработать с промтом или системным промтом')
                    break
                if not assessment.replace('.', '').isdigit():  
                    print('Оценка для примера не сгенерировалась, пробуем еще')
                    answer = model.predict([questions[i]])
                    assessment = get_assessment(answer,
                                                surround_symbols)
                    n_attempts += 1
                else:
                    flag_not_generate = False
                    answers[j][i] = answer
            assessments[j].append(assessment)
    general_assessments = []
    for i, answer in enumerate(answers[0]):
        values = []
        for j in assessments:
            if assessments[j][i]:
                values.append(float(assessments[j][i]))
            else:
                values.append(np.nan)
        mean_i = np.nanmean(values) 
        general_assessments.append(mean_i)
    result_df = pd.concat([pd.DataFrame({'question': questions,
                                         'general_assessments': general_assessments}), 
                           pd.DataFrame(answers)], axis=1)
    return result_df
