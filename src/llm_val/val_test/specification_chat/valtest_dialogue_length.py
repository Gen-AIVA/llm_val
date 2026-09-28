import numpy as np
import typing as tp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold

def report_valtest_dialogue_length(fig: tp.Any,
                                   avg_length: float,
                                   conversational: bool,
                                   semaphore_threshold_converse: tp.Tuple[float, float]=(3, 5),
                                   semaphore_threshold_answer: tp.Tuple[float, float]=(3, 5),
                                   is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а для теста на среднюю длину диалога
    Parameters:
        fig: График длин диалогов
        avg_length: Средняя длина диалога
        conversational: True, если формат диалога - разговор, False, если формат диалога - краткий ответ
        semaphore_threshold_converse: Границы светофора для средней длины диалога в формате разговора
        semaphore_threshold_answer: Границы светофора для средней длины диалога в формате краткого ответа
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    if conversational:
        color = semaphore_by_threshold(avg_length,
                                       semaphore_threshold_converse,
                                       greater_is_better=True,
                                       left_border_is_bigger=True,
                                       right_border_is_bigger=False)
    else:
        color = semaphore_by_threshold(avg_length,
                                       semaphore_threshold_answer,
                                       greater_is_better=False,
                                       left_border_is_bigger=True,
                                       right_border_is_bigger=False)
    color = 'gray' if is_info else color
    res_table = pd.DataFrame({'Метрика': ['Средняя длина диалога'],
                              'Значение': [avg_length],
                              'Результат теста': color})
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = [fig]
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_dialogue_length(sampler: Sampler,
                            data_type: str='test',
                            semaphore_threshold_converse: tp.Tuple[float, float]=(3, 5),
                            semaphore_threshold_answer: tp.Tuple[float, float]=(3, 5),
                            conversational: bool=False,
                            is_info: bool=False,
                            precomputed: tp.Optional[tp.Dict[str, tp.Any]]=None,
                            **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на оценку длины диалога
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold_converse: Границы светофора для средней длины диалога в формате разговора
        semaphore_threshold_answer: Границы светофора для средней длины диалога в формате краткого ответа
        conversational: True, если формат диалога - разговор, False, если формат диалога - краткий ответ
        is_info: True, если тест информативный
        precomputed: Опциональное значение посчитанных длин диалогов и средней длины диалога.
            Если не None, то подсчет метрики не производится
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if precomputed is None:
        data = getattr(sampler, data_type)['X']
        dialogue_lengths = data.groupby('dialogue_id').size() / 2
        avg_length = dialogue_lengths.mean()
        precomputed = {'avg_length': avg_length,
                       'dialogue_lengths': dialogue_lengths}
    fig = plot_lengths(precomputed['dialogue_lengths'])
    report = report_valtest_dialogue_length(fig,
                                            precomputed['avg_length'],
                                            conversational,
                                            semaphore_threshold_converse,
                                            semaphore_threshold_answer,
                                            is_info)
    return {'report': report,
            'precomputed': precomputed}


def plot_lengths(dialogue_lengths: pd.Series,
                 n_bins: int=100) -> tp.Any:
    '''
    Отрисовка графика длин диалогов
    Parameters:
        dialogue_lengths: Series с длинами диалогов
        n_bins: Количество бинов гистограммы
    Return:
        График, показывающий длины диалогов
    '''
    fig, ax = plt.subplots(figsize=(10, 8))
    bins = np.linspace(dialogue_lengths.min(),
                       dialogue_lengths.max(),
                       n_bins)
    sns.histplot(dialogue_lengths.tolist(), bins=bins, kde=True, ax=ax)
    ax.set_xlim(left=0, right=50)
    ax.set_title('Распределение длин диалогов')
    ax.set_xlabel('Длина диалога')
    ax.set_ylabel('Частота')
    ax.grid()
    return fig