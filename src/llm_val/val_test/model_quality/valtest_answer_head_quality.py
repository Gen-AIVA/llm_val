import re
import numpy as np
import pandas as pd
import typing as tp
from copy import deepcopy
from string import punctuation
import matplotlib.pyplot as plt
from nltk import word_tokenize, sent_tokenize

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.autometrics import metrics2func
from llm_val.utils.report_helper import semaphore_by_threshold_without_red


def report_valtest_head_quality(metric_value: float,
                                metric_name: str,
                                n_head_elements: int,
                                figure: tp.Optional[tp.Any]=None,
                                data_type: str='test',
                                semaphore_threshold: float=0.1,
                                is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        metric_value: Значение метрики при использовании желаемой начальной части ответа
        metric_name: Название интересующей метрики
        n_head_elements: Желаемая длина ответа модели в лексических единицах
        figure: График зависимости метрики качества от числа взятых начальных лексических единиц
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold: Кортеж с границами светофора. Границы в кортеже указываются в порядке возрастания
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    color = 'gray' if is_info else semaphore_by_threshold_without_red(metric_value,
                                                                      threshold=semaphore_threshold,
                                                                      greater_is_better=True)
    res_table = pd.DataFrame({'Метрика': [metric_name],
                              f'Значение метрики на первых {n_head_elements} лексических единицах на {data_type}': [metric_value], 
                              'Результат теста': [color]})
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = [figure]
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_answer_head_quality(sampler: Sampler,
                                data_type: str='test',
                                metric_name: str='meteor',
                                n_head_elements: int=5,
                                semaphore_threshold: float=0.2,
                                step: int=5,
                                use_sentences: bool=True,
                                is_info: bool=False,
                                head_metrics: tp.Optional[tp.Dict[int, float]]=None,
                                **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на анализ ключевой метрики при формировании финального ответа на первых нескольких словах/предложениях 
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        metric_name: Название интересующей метрики
        n_head_elements: Желаемая длина ответа модели в лексических единицах
        semaphore_threshold: Минимальный допустимый порог по метрике качества
        step: Размер шага в лексических единицах для построения сетки точек длин ответов, 
            в которых будет подсчитываться среднее значение метрики
        use_sentences: True, если в качестве лексической единицы используется предложение (иначе — слово)
        is_info: True, если тест информативный
        head_metrics: Опциональная пара списков значений метрики. 
            Если не None, то подсчет метрики не производится, и на основе переданного аргумента выставляется светофор.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if head_metrics is None:
        data = deepcopy(getattr(sampler, data_type)['X'])
        head_metrics = get_head_metrics(data,
                                        metric_name,
                                        n_head_elements,
                                        use_sentences,
                                        step)
    figure = plot_head_metrics_result(head_metrics,
                                      metric_name,
                                      semaphore_threshold,
                                      n_head_elements,
                                      use_sentences)
    report = report_valtest_head_quality(head_metrics[n_head_elements],
                                         metric_name,
                                         n_head_elements,
                                         figure,
                                         data_type,
                                         semaphore_threshold,
                                         is_info)
    precomputed = {'head_metrics': head_metrics}
    return {'report': report,
            'precomputed': precomputed}


def get_head_metrics(data: pd.DataFrame,
                     metric_name: str='meteor',
                     n_head_elements: int=5,
                     use_sentences: bool=True,
                     step: int=5) -> tp.Dict[int, float]:
    '''
    Функция для посчета метрик качества в зависимости от числа взятых первых лексических единиц
    Parameters:
        data: pd.DataFrame со столбцами model_answer (ответ модели) и true_answer (ground truth)
        metric_name: Название интересующей метрики
        n_head_elements: Желаемая длина ответа модели в лексических единицах
        use_sentences: True, если в качестве лексической единицы используется предложение (иначе — слово)
        step: Размер шага в лексических единицах для построения сетки точек длин ответов, 
            в которых будет подсчитываться среднее значение метрики
    Return: 
        Словарь пар {число взятых лексических единиц : значение метрики качества}
    '''
    metrics_dict = {}
    data['true_answer'] = data['true_answer'].apply(lambda x: [delete_punct(s) for s in x])
    if use_sentences:
        data['model_answer'] = data['model_answer'].apply(lambda x: [delete_punct(s) for s in sent_tokenize(x)])
    else:
        data['model_answer'] = data['model_answer'].apply(lambda x: word_tokenize(delete_punct(x)))
    answer_length_space = np.arange(1, data['model_answer'].apply(len).max()+step, step)
    answer_length_space = np.insert(answer_length_space, np.searchsorted(answer_length_space, n_head_elements), n_head_elements)
    for answer_length in answer_length_space:
        metrics_dict[answer_length] = data.apply(lambda x: metrics2func[metric_name](x['true_answer'],
                                                                                     ' '.join(x['model_answer'][:answer_length])), axis=1).mean()
    return metrics_dict


def plot_head_metrics_result(metrics_dict: pd.DataFrame,
                             metric_name: str,
                             semaphore_threshold: float,
                             n_head_elements: int,
                             use_sentences: bool) -> tp.Any:
    '''
    Строит график зависимости метрики качества от числа взятых начальных лексических единиц
    Parameters:
        metrics_dict: Словарь пар {число взятых лексических единиц : значение метрики качества}
        metric_name: Название интересующей метрики
        semaphore_threshold: Минимальный допустимый порог по метрике качества
        n_head_elements: Желаемая длина ответа модели в лексических единицах
        use_sentences: True, если в качестве лексической единицы используется предложение (иначе — слово)
    Return:
        График метрики качества 
    '''
    element = 'предложений' if use_sentences else 'слов'
    fig = plt.figure(figsize=(9, 7))
    plt.plot(metrics_dict.keys(), metrics_dict.values(), marker='o')
    plt.xlabel(f'Число первых {element} взятых из ответа модели')
    plt.ylabel(metric_name)
    plt.axhline(y=semaphore_threshold, color='red', label='Допустимый уровень качества')
    plt.axvline(x=n_head_elements, color='black', label=f'Желаемое количество {element} в ответе')
    plt.legend()
    plt.title(f'Зависимость метрики качества от числа взятых {element}')
    return fig


def delete_punct(text_sample: str) -> str:
    '''
    Удаляет все символы пунктуации из строки
    Parameters:
        text_sample: Строка
    Return:
        Строка без символов пунктуации
    '''
    return re.sub(f"[{re.escape(punctuation)}]", "", text_sample)
