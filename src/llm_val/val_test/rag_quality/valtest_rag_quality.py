import numpy as np
import pandas as pd
import typing as tp
from tqdm import tqdm
from scipy import stats
from copy import deepcopy

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.metrics_rag import eval_function_ranking
from llm_val.utils.report_helper import semaphore_by_threshold

def report_valtest_rag_quality(avg_metrics: tp.Dict[str, float],
                               confidence_intervals: tp.Dict[str, str],
                               metric_list: tp.List[str],
                               key_metric: str,
                               max_value: float,
                               thresholds: tp.Tuple[float, float],
                               is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание отчета на основе рассчитанных метрик и доверительных интервалов.
    Parameters:
        avg_metrics: Словарь со средними значениями метрик.
        confidence_intervals: Словарь с доверительными интервалами для каждой метрики.
        metric_list: Cписок метрик для расчета.
        key_metric: Ключевая метрика для определения цвета светофора.
        max_value: Максимальное значение целевой метрики (необходимо для получения светофора).
        thresholds: Кортеж из двух значений (нижний порог, верхний порог) для определения цвета светофора.
        is_info: True, если тест информативный.
    Return:
        Отчет в соответствии с концепцией llm_val.
    '''
    threshold_for_semaphore = (thresholds[0] * max_value, thresholds[1] * max_value)
    color = 'gray' if is_info else semaphore_by_threshold(avg_metrics[key_metric],
                                                          threshold_for_semaphore, 
                                                          greater_is_better=True)
    res_table = pd.DataFrame({'Название метрики': metric_list,
                              'Значение метрики': [avg_metrics[metric] for metric in metric_list],
                              'Доверительный интервал': [confidence_intervals[metric] for metric in metric_list],
                              'Результат теста': [color if metric == key_metric else 'gray' for metric in metric_list]})
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_rag_quality(sampler: Sampler,
                        data_type: str='test',
                        topk: int=3,
                        metric_list: tp.List[str]=['nDCG@k', 
                                                   'MAP@k', 
                                                   'MP@k', 
                                                   'AUC@k', 
                                                   'ERR@k', 
                                                   'MRR@k', 
                                                   'HitRate@k'],
                        key_metric: str='HitRate@k',
                        confidence: float=0.95,
                        max_value: float=1.0,
                        thresholds: tp.Tuple[float, float]=(0.2, 0.4),
                        is_info: bool=False,
                        **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на анализ качества RAG-системы.
    Parameters:
        sampler: объект класса Sampler.
        data_type: Тип данных, на которых производится расчет метрики.
        topk: Количество элементов для расчета метрик.
        metric_list: Cписок метрик для расчета. Hint: ключи словаря eval_function_ranking в модуле llm_val.utils.metrics_rag.
        key_metric: Ключевая метрика для определения цвета светофора (обязательно является элементом списка metric_list).
        confidence: Уровень доверия для расчета доверительного интервала.
        max_value: Максимальное значение целевой метрики (необходимо для получения светофора).
        thresholds: Кортеж из двух значений (нижний порог, верхний порог) для определения цвета светофора.
        is_info: True, если тест информативный.
    Return:
        Словарь в соответствии с концепцией llm_val.
    '''
    df_X = deepcopy(getattr(sampler, data_type)['X'])
    df_y = deepcopy(getattr(sampler, data_type)['y'])

    all_metrics = []
    for idx, row in df_X.iterrows():
        context = row['context']
        passage_accuracy = df_y.loc[idx, 'criterion_passage_accuracy']
        metrics = calculate_metrics_for_query(passage_accuracy, 
                                              topk,
                                              metric_list)
        all_metrics.append(metrics)

    avg_metrics = {metric: np.mean([m[metric] for m in all_metrics]) for metric in metric_list}
    confidence_intervals = {}
    for metric in metric_list:
        interval = calculate_confidence_interval([m[metric] for m in all_metrics], 
                                                 confidence)
        if interval is None:
            confidence_intervals[metric] = 'Недостаточно данных'
        else:
            confidence_intervals[metric] = f'{interval[0]} - {interval[1]}'

    report = report_valtest_rag_quality(avg_metrics, 
                                        confidence_intervals, 
                                        metric_list, 
                                        key_metric, 
                                        max_value, 
                                        thresholds,
                                        is_info)

    precomputed = {'baseline_score': avg_metrics}
    return {'report': report, 
            'precomputed': precomputed}


def calculate_metrics_for_query(passage_accuracy: tp.List[int], 
                                topk: int, 
                                metric_list: tp.List[str]) -> tp.Dict[str, float]:
    '''
    Рассчитывает метрики для одного запроса.
    Parameters:
        passage_accuracy: Список с оценками релевантности пассажей.
        topk: Количество элементов для расчета метрик.
        metric_list: Cписок метрик для расчета. Hint: ключи словаря eval_function_ranking в модуле llm_val.utils.metrics_rag.
    Returns:
        Словарь с рассчитанными метриками.
    '''
    metrics = {}
    for metric in metric_list:
        metrics[metric] = eval_function_ranking[metric](passage_accuracy, topk)
    return metrics


def calculate_confidence_interval(metrics: tp.List[float], 
                                  confidence: float=0.95) -> tp.Optional[tp.Tuple[float, float]]:
    '''
    Рассчитывает доверительный интервал для метрик.
    Parameters:
        metrics: Список значений метрик.
        confidence: Уровень доверия.
    Returns:
        Доверительный интервал или None, если расчет невозможен.
    '''
    result = None
    if len(metrics) >= 2:
        n = len(metrics)
        mean = np.mean(metrics)
        z = stats.norm.ppf((1 + confidence) / 2)
        factor = z * np.sqrt((mean * (1 - mean) + z**2 / (4 * n)) / n)
        denominator = 1 + z**2 / n
        lower = ((mean + z**2 / (2 * n) - factor) / denominator)
        upper = ((mean + z**2 / (2 * n) + factor) / denominator)
        result = (round(lower, 3), round(upper, 3))
    return result

