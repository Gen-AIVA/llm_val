import numpy as np
import pandas as pd
import typing as tp
from copy import deepcopy

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.psi_calculate import psi
from llm_val.utils.report_helper import semaphore_by_threshold

def report_valtest_topic_proportion(result_df: pd.DataFrame,
                                    is_info: bool,
                                    psi_value: float,
                                    thresholds: tp.Tuple[float, float]) -> tp.Dict[str, tp.Any]:
    '''
    Создает отчет о результатах валидационного теста.
    Parameters:
        result_df: DataFrame с результатами теста
        is_info: True, если тест информативный
        psi_value: Значение PSI
        thresholds: Кортеж для определения результата на основе PSI
    Returns:
        Словарь в соответствии с концепцией llm_val
    '''
    color = 'gray' if is_info else semaphore_by_threshold(psi_value, 
                                                          thresholds, 
                                                          greater_is_better=False)
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [result_df]
    return res_dict


def valtest_topic_proportion(sampler: Sampler,
                             apriori_dict: tp.Dict[str, float],
                             is_info: bool=False,
                             thresholds: tp.Tuple[float, float]=(0.2, 0.4),
                             **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Валидационный тест для проверки соответствия доли тем тестовых вопросов априорному распределению.
    Parameters:
        sampler: Объект класса Sampler с заполненной колонкой 'thema'
        apriori_dict: Словарь с наименованием темы и ее априорной долей
        is_info: True, если тест информативный
        thresholds: Кортеж для определения результата на основе PSI
    Returns:
        Словарь в соответствии с концепцией llm_val
    '''
    data = deepcopy(getattr(sampler, 'test')['X'])

    theme_counts = data['thema'].value_counts(normalize=True)

    themes = list(apriori_dict.keys())
    prior_distribution = np.array([apriori_dict[theme] for theme in themes])
    real_distribution = np.array([theme_counts.get(theme, 0) for theme in themes])

    psi_value = psi(prior_distribution, real_distribution)

    result_df = pd.DataFrame({'Тематика': themes,
                              'Реальное распределение': real_distribution,
                              'Априорное распределение': prior_distribution})

    report = report_valtest_topic_proportion(result_df, 
                                             is_info, 
                                             psi_value, 
                                             thresholds)

    precomputed = {'psi_value': psi_value}

    return {'report': report,
            'precomputed': precomputed}
