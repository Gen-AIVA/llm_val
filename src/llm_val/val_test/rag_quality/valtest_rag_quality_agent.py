import numpy as np
import pandas as pd
import typing as tp
from copy import deepcopy

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.metrics_rag import eval_function_ranking
from llm_val.utils.report_helper import (worst_semaphore, 
                                         semaphore_by_threshold)


def report_valtest_rag_quality_agent(avg_metrics: tp.Dict[str, float],
                                     metric_list: tp.List[str],
                                     thresholds: tp.Dict[str, float],
                                     default_thresholds: tp.Tuple[float, float],
                                     groups: tp.Optional[tp.List[str]] = None,
                                     is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание отчета на основе рассчитанных метрик и доверительных интервалов.
    Parameters:
        avg_metrics: Словарь со средними значениями метрик.
        metric_list: Cписок метрик для расчета.
        thresholds: Словарь с порогами для передаваемых метрик;
            Ключ - название метрики, значение - кортеж из двух значений (нижний порог, верхний порог) для определения цвета светофора.
        default_thresholds: Кортеж из двух значений (нижний порог, верхний порог) для определения цвета светофора;
            Используется как дефолтное значение в случае отсутсвия порогов в thresholds для метрик из metric_list.
        groups: список тематик. Если None, группировка не применяется.
        is_info: True, если тест информативный.
    Return:
        Отчет в соответствии с концепцией llm_val.
    '''
    color_list = []
    rows = []
    groups = groups if groups else [None]
    
    for metric in metric_list:
        thresholds_metric = thresholds.get(metric, default_thresholds)
        
        for group in groups:
            key = f"{metric}_{group}" if group is not None else metric
            val = avg_metrics[key]
            color = 'gray' if is_info else semaphore_by_threshold(val, 
                                                                  thresholds_metric, 
                                                                  greater_is_better=True)
            color_list.append(color)
            
            row = {
                'Название метрики': metric,
                'Значение метрики': val,
                'Результат теста': color
            }
            if group is not None:
                row['Название тематики'] = group
            rows.append(row)
            
    columns = ['Название метрики', 'Название тематики', 'Значение метрики', 'Результат теста'] if groups[0] is not None else ['Название метрики', 'Значение метрики', 'Результат теста']
    res_table = pd.DataFrame(rows, columns=columns)

        
    final_color = 'gray' if is_info else worst_semaphore(color_list)
    
    res_dict = {}
    res_dict['semaphore'] = final_color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_rag_quality_agent(sampler: Sampler,
                              data_type: str='test',
                              topk: int=3,
                              metric_list: tp.List[str]=['nDCG@k', 
                                                         'MAP@k', 
                                                         'MP@k', 
                                                         'AUC@k', 
                                                         'ERR@k', 
                                                         'MRR@k', 
                                                         'HitRate@k',
                                                         'Recall@k'],
                              thresholds: tp.Dict[str, float]={'nDCG@k': (0.5, 0.8), 
                                                               'MAP@k': (0.5, 0.8),
                                                               'MP@k': (0.5, 0.8),
                                                               'AUC@k': (0.5, 0.8),
                                                               'ERR@k': (0.5, 0.8),
                                                               'MRR@k': (0.5, 0.8),
                                                               'HitRate@k': (0.5, 0.8),
                                                               'Recall@k': (0.5, 0.8)},
                              default_thresholds: tp.Tuple[float, float]=(0.5, 0.8),
                              is_info: bool=False,
                              avg_metrics: tp.Optional[tp.Dict[str, float]] = None,
                              exec_env: tp.Literal['default', 'sberds']='default',
                              **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на анализ качества RAG-системы.
    Parameters:
        sampler: Объект класса Sampler.
        data_type: Тип данных, на которых производится расчет метрики.
        topk: Количество элементов для расчета метрик.
        metric_list: Cписок метрик для расчета. Hint: ключи словаря eval_function_ranking в модуле llm_val.utils.metrics_rag.
        thresholds: Словарь с порогами для передаваемых метрик;
            Ключ - название метрики, значение - кортеж из двух значений (нижний порог, верхний порог) для определения цвета светофора.
        default_thresholds: Кортеж из двух значений (нижний порог, верхний порог) для определения цвета светофора;
            Используется как дефолтное значение в случае отсутсвия порогов в thresholds для метрик из metric_list.
        is_info: True, если тест информативный.
        avg_metrics: Опциональный словарь с предпосчитанными результатами теста.
            Если не None, то тест не запускается, и на основе переданного аргумента выставляется светофор.
        exec_env: Среда выполнения ('default' или 'sberds')
    Return:
        Словарь в соответствии с концепцией llm_val.
    '''
    if avg_metrics is None:
        data_X = sampler[data_type]['X'] if exec_env == 'sberds' else getattr(sampler, data_type)['X']
        data_y = sampler[data_type]['y'] if exec_env == 'sberds' else getattr(sampler, data_type)['y']
        groups = None
        
        if 'scenario' in data_X.columns:
            avg_metrics = {}
            groups = data_X['scenario'].unique().tolist()
            for thema in groups:
                thema_data_X = data_X[data_X['scenario'] == thema]
                all_metrics = []
                for idx, row in thema_data_X.iterrows():
                    passage_accuracy = data_y.loc[idx, 'passage_accuracy_metric']
                    metrics = calculate_metrics_for_query(passage_accuracy, 
                                                          topk,
                                                          metric_list)
                    all_metrics.append(metrics)
                for metric in metric_list:
                    avg_val = np.mean([m[metric] for m in all_metrics])
                    avg_metrics[f"{metric}_{thema}"] = avg_val
        else:
            all_metrics = []
            for idx, row in data_X.iterrows():
                passage_accuracy = data_y.loc[idx, 'passage_accuracy_metric']
                metrics = calculate_metrics_for_query(passage_accuracy, 
                                                      topk,
                                                      metric_list)
                all_metrics.append(metrics)
        
            avg_metrics = {metric: np.mean([m[metric] for m in all_metrics]) for metric in metric_list}
    
    
    report = report_valtest_rag_quality_agent(avg_metrics,
                                              metric_list,
                                              thresholds,
                                              default_thresholds,
                                              groups,
                                              is_info)

    precomputed = {'avg_metrics': deepcopy(avg_metrics)}
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

