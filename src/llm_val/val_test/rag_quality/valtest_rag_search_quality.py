import numpy as np
import typing as tp
import pandas as pd
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.binarization_algos import iterative_binning
from llm_val.utils.local_rag_qulity_aggregation import calculate_local_rag_quality

def report_valtest_rag_search_quality(correlation_results: tp.Dict, 
                                      res_plot: plt.Figure) -> tp.Dict[str, tp.Any]:
    '''
    Генерирует отчет с результатами корреляции и графиком.
    Parameters:
        correlation_results: Словарь с результатами корреляции.
        res_plot: Объект Figure с графиком.
    Return:
        Отчет в соответствии с концепцией llm_val.
    '''
    res_table = pd.DataFrame(list(correlation_results.items()),
                             columns=['Наименование метрики', 'Корреляция'])
    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_plots'] = [res_plot]
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_rag_search_quality(sampler: Sampler, 
                               topk: int,
                               data_type: str='test',
                               is_discrete: bool=True,
                               calc_local_rag_quality_func: tp.Callable=calculate_local_rag_quality,
                               min_bin_size: float=0.05,
                               **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Выполняет тест качества RAG и генерирует отчет.
    Parameters:
        sampler: Объект класса Sampler.
        topk: Максимальное значение top-k для расчета корреляции.
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        is_discrete: Параметр, определяющий тип множества значений локального качества RAG. True, если локальное качество RAG измеримо дискретно.
        calc_local_rag_quality_func: Функция агрегации оценок релевантности пассажей в топе. Hint: примеры функций есть в utils.local_rag_qulity_aggregation.
        min_bin_size: Минимальный размер бина (в долях от общего числа данных).
    Return:
        Словарь в соответствии с концепцией llm_val.
    '''
    df = deepcopy(getattr(sampler, data_type)['y'])
    correlation_results = calculate_spearman_correlation(df, 
                                                         topk,
                                                         calc_local_rag_quality_func)
    res_plot = plot_boxplot(df,
                            topk,
                            is_discrete,
                            calc_local_rag_quality_func,
                            min_bin_size)
    
    report = report_valtest_rag_search_quality(correlation_results,
                                               res_plot)
    return {'report': report, 
            'precomputed': None}


def calculate_spearman_correlation(df: pd.DataFrame, 
                                   topk: int,
                                   calc_local_rag_quality_func: tp.Callable=calculate_local_rag_quality) -> tp.Dict:
    '''
    Рассчитывает корреляцию Спирмена между локальным качеством RAG и точностью LLM.
    Parameters:
        df: DataFrame с данными.
        topk: Максимальное значение top-k для расчета корреляции.
        calc_local_rag_quality_func: Функция агрегации оценок релевантности пассажей в топе. Hint: примеры функций есть в utils.local_rag_qulity_aggregation.
    Return:
        Словарь с корреляциями для каждого значения top-k (по убыванию).
    '''
    results = {}
    for current_topk in range(topk, 0, -1):
        local_qualities = df['criterion_passage_accuracy'].apply(lambda x: calc_local_rag_quality_func(x[:current_topk]))
        correlation, _ = spearmanr(local_qualities, df['criterion_accuracy'])
        results[f'HitRate@{current_topk}'] = correlation
    return results


def discretize_continuous_rag_quality(series: pd.Series, 
                                      topk: int, 
                                      is_discrete: bool=True,
                                      calc_local_rag_quality_func: tp.Callable=calculate_local_rag_quality,
                                      min_bin_size: float=0.05) -> pd.Series:
    '''
    Дискретизирует непрерывное локальное качество RAG по указанному алгоритму.
    Parameters:
        series: Список значений локального качества RAG.
        topk: Значение top-k для расчета локального качества RAG.
        is_discrete: Параметр, определяющий тип множества значений локального качества RAG. True, если локальное качество RAG измеримо дискретно.
        calc_local_rag_quality_func: Функция агрегации оценок релевантности пассажей в топе.
        min_bin_size: Минимальный размер бина (в долях от общего числа данных).
    Return:
        Дискретизированные или исходные значения в формате pd.Series.
    '''
    local_qualities = series.apply(lambda x: calc_local_rag_quality_func(x[:topk]))    
    if is_discrete:
        result = local_qualities
    else:
        value_counts = local_qualities.value_counts(normalize=True)
        frequent_values = value_counts[value_counts >= min_bin_size].index
        
        frequent_data = local_qualities[local_qualities.isin(frequent_values)]
        rare_data = local_qualities[~local_qualities.isin(frequent_values)]
        
        bins = iterative_binning(rare_data, 
                                 min_bin_size)
        
        bin_labels = []
        for i, bin_data in enumerate(bins):
            if len(bin_data) > 0:
                bin_labels.extend([f'Bin {i+1}'] * len(bin_data))
        
        for i, value in enumerate(frequent_values):
            bin_labels.extend([f'Frequent {i+1}'] * len(frequent_data[frequent_data == value]))
        result = pd.Series(bin_labels, index=local_qualities.index)
        value_counts = result.value_counts(normalize=True)
    return result


def plot_boxplot(df: pd.DataFrame, 
                 topk: int,
                 is_discrete: bool=True,
                 calc_local_rag_quality_func: tp.Callable=calculate_local_rag_quality,
                 min_bin_size: float=0.05) -> plt.Figure:
    '''
    Строит график "ящик с усами" для визуализации зависимости между локальным качеством RAG и точностью LLM.
    Parameters:
        df: DataFrame с данными.
        topk: Значение top-k для расчета локального качества RAG.
        is_discrete: Параметр, определяющий тип множества значений локального качества RAG. True, если локальное качество RAG измеримо дискретно.
        calc_local_rag_quality_func: Функция агрегации оценок релевантности пассажей в топе.
        min_bin_size: Минимальный размер бина (в долях от общего числа данных).
    Return:
        Объект Figure с графиком.
    '''
    local_qualities = discretize_continuous_rag_quality(df['criterion_passage_accuracy'],
                                                        topk,
                                                        is_discrete,
                                                        calc_local_rag_quality_func,
                                                        min_bin_size)
    df['local_rag_quality'] = local_qualities
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if pd.api.types.is_categorical_dtype(local_qualities) or pd.api.types.is_integer_dtype(local_qualities):
        sns.boxplot(x='local_rag_quality', y='criterion_accuracy', data=df, ax=ax, order=sorted(local_qualities.unique()))
    else:
        sns.boxplot(x='local_rag_quality', y='criterion_accuracy', data=df, ax=ax)
        sns.swarmplot(x='local_rag_quality', y='criterion_accuracy', data=df, ax=ax, color=".25")
    
    ax.set_title(f'Точность LLM vs Локальное качество RAG (Top-{topk})')
    ax.set_xlabel('Локальное качество RAG')
    ax.set_ylabel('Точность LLM') 
    return fig