import numpy as np
import pandas as pd
import typing as tp
from copy import deepcopy
from functools import partial
from nltk import word_tokenize
from nltk.translate import meteor_score

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold


def report_valtest_answer_unique(usr_value: float,
                                 data_type: str='test',
                                 semaphore_threshold: tp.Tuple[float, float]=(0.01, 0.1),
                                 is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        usr_value: Доля уникальных ответов модели 
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold: Кортеж с границами светофора. Границы в кортеже указываются в порядке возрастания
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    color = 'gray' if is_info else semaphore_by_threshold(usr_value,
                                                          threshold=semaphore_threshold,
                                                          greater_is_better=True)
    res_table = pd.DataFrame({f'Наименование критерия на {data_type}': ['Доля уникальных значений (USR)'], 
                              'Значение критерия': [usr_value], 
                              'Результат теста': [color]})
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_answer_unique(sampler: Sampler,
                          data_type: str='test',
                          is_long_answer: bool=True,
                          meteor_threshold: float=0.9,
                          semaphore_threshold: tp.Tuple[float, float]=(0.01, 0.1),
                          sample_size: float=1.0,
                          seed: int=42,
                          is_info: bool=False,
                          usr_value: tp.Optional[float]=None,
                          **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на уникальность ответа модели
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        is_long_answer: Булевый флаг. True, если результатом работы является развернутое описание 
        meteor_threshold: Трешхолд по метрике METEOR. Если значение METEOR больше данного трешхолда, то ответ считается не уникальным
        semaphore_threshold: Кортеж с границами светофора. Границы в кортеже указываются в порядке возрастания
        sample_size: Доля выборки, которая будет использована в тесте
        seed: Начальное значение датчика случайных чисел
        is_info: True, если тест информативный
        usr_value: Опциональное значение метрики. 
            Если не None, то подсчет доли уникальных ответов не производится, и на основе переданного аргумента выставляется светофор.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    rng = np.random.default_rng(seed=seed)
    if usr_value is None:
        data = getattr(sampler, data_type)['X']
        data_sample = data.sample(frac=sample_size,
                                  random_state=rng)
        model_answer = list(data_sample['model_answer'].values)
        usr_value = calculate_unique(model_answer=model_answer,
                                     is_long_answer=is_long_answer,
                                     meteor_threshold=meteor_threshold)
    report = report_valtest_answer_unique(usr_value=usr_value,
                                          data_type=data_type,
                                          semaphore_threshold=semaphore_threshold,
                                          is_info=is_info)
    precomputed = {'usr_value': usr_value}
    return {'report': report,
            'precomputed': precomputed}


def calculate_unique(model_answer: tp.List[str],
                     is_long_answer: bool=True,
                     meteor_threshold: float=0.9) -> float:
    '''
    Подсчет доли уникальных значений
    Parameters:
        model_answer: Список с ответами моделей
        is_long_answer: Булевый флаг. True, если результатом работы является развернутое описание 
        meteor_threshold: Трешхолд по метрике METEOR. Если значение METEOR больше данного трешхолда, то ответ считается не уникальным
    Return:
        Доля уникальных значений
    '''
    model_answer = deepcopy(model_answer)
    unique_answers = [model_answer.pop(0)]
    unique_func = partial(is_unique_long_answer, 
                          meteor_threshold=meteor_threshold) if is_long_answer else is_unique_answer
    for ans in model_answer:
        if unique_func(ref=unique_answers, pred=ans):
            unique_answers.append(ans)
    return len(unique_answers) / (len(model_answer) + 1)


def is_unique_answer(ref: tp.List[str],
                     pred: str) -> bool:
    '''
    Считает, является ли предложенный ответ уникальным по сравнению с ответами из списка по полному соовпадению
    Parameters:
        ref: Список строк, содержащий тексты для сравнения
        pred: Строка, содержащая текст для проверки на уникальность
    Return:
        True, если предложенный ответ уникален, иначе False
    '''
    ref = [i.lower() for i in ref]
    return not (pred.lower() in ref)


def is_unique_long_answer(ref: tp.List[str],
                          pred: str,
                          meteor_threshold: float) -> bool:
    '''
    Считает, является ли предложенный ответ уникальным по сравнению с ответами из списка с использованием метрики METEOR.
    Parameters:
        ref: Список строк, содержащий тексты для сравнения
        pred: Строка, содержащая текст для проверки на уникальность
    Return:
        True, если предложенный ответ уникален, иначе False
    '''
    return not calculate_meteor(ref, pred) > meteor_threshold


def calculate_meteor(ref: tp.List[str], pred: str) -> float:
    '''
    Считает метрику METEOR
    Parameters:
        ref: Список строк, содержащий тексты для сравнения
        pred: Строка, содержащая текст для проверки на уникальность
    Return:
        Значение метрики METEOR
    '''
    return meteor_score.meteor_score([word_tokenize(i) for i in ref], word_tokenize(pred))
