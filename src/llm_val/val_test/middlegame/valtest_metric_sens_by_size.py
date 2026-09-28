import math
import random
import numpy as np
import pandas as pd
import typing as tp
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt

from llm_val.sampler.Sampler import Sampler


def report_valtest_metric_sens_by_size(fig: tp.Any,
                                       cur_n_sample: int,
                                       target_metric: float=0.7,
                                       quantile: float=0.95) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        fig: График
        cur_n_sample: Текущий размер выборки
        target_metric: Целевое значение метрики
        quantile: Квантиль доверительного интервала
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_table = pd.DataFrame({'Текущий размер выборки': [cur_n_sample],
                              'Целевое значение метрики': [target_metric],
                              'Квантиль доверительного  интервала': [quantile]})
    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_plots'] = [fig]
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_metric_sens_by_size(sampler: Sampler,
                                data_type: str='test',
                                metric: str='accuracy',
                                dict_of_binary_freqs: tp.Optional[tp.Dict]=None,
                                n_gened_set: tp.Optional[int]=None,
                                n_border: tp.Tuple[int, int]=(2, 10_000),
                                target_metric: float=0.7,
                                quantile: float=0.95,
                                **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на чувствительность метрики относительно размера тестового сета 
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        metric: Целевая метрика. Поддерживаются значения 'accuracy', 'f1-score', 'precision', 'recall'
        dict_of_binary_freqs: Необязательный словарь, в котором должны быть ключи 'TP', 'FP', 'FN' и 'TN', 
            сумма значений которых должна равняться 1. Не требуется при оценке accuracy.
        n_gened_set: Число генерируемых выборок с долями dict_of_binary_freqs для каждого размера выборки
        n_border: Левая и правая граница размеров выборки
        target_metric: Желаемое значение метрики
        quantile: Квантиль доверительного интервала; должен находиться в интервале (0; 1)
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    l_borders = []
    r_borders = []
    n_samples_list = []
    cur_n_sample = len(getattr(sampler, data_type)['X'])
    sample_size_list = list(set(np.round(np.log(np.arange(*n_border)) * 10)))
    for i in tqdm(sample_size_list):
        c = round(math.exp(i/10))
        if metric in {'f1-score', 'precision', 'recall'}:
            left, right = get_confidence_interval_by_size_estimated(dict_of_binary_freqs,
                                                                    n_gened_set,
                                                                    metric,
                                                                    c,
                                                                    target_metric,
                                                                    quantile)            
        else:
            left, right = get_confidence_interval_by_size_exact(c, 
                                                                target_metric, 
                                                                quantile)
        l_borders.append(left)
        r_borders.append(right)
        n_samples_list.append(c)
    fig = plot_metric_sens_by_size(l_borders=l_borders,
                                   r_borders=r_borders,
                                   n_samples_list=n_samples_list,
                                   cur_n_sample=cur_n_sample,
                                   target_metric=target_metric)
    report = report_valtest_metric_sens_by_size(fig=fig,
                                                cur_n_sample=cur_n_sample,
                                                target_metric=target_metric,
                                                quantile=quantile)
    precomputed = {}
    return {'report': report,
            'precomputed': precomputed}
    

def get_confidence_interval_by_size_exact(n_sample: int=300,
                                          target_metric: float=0.7, 
                                          quantile: float=0.95) -> tp.Tuple[float, float]:
    '''
    Подсчет границ доверительного интервала по формуле Уилсона
    Parameters:
        n_sample: Размер выборки
        target_metric: Желаемое качество метрики
        quantile: Квантиль доверительного интервала; должен находиться в интервале (0; 1)
    Return:
        Левая и правая граница доверительного интервала
    '''
    t = norm.ppf((1 - quantile) / 2 + quantile)
    first = n_sample / (t ** 2 + n_sample)
    second = target_metric + (t ** 2) / (2 * n_sample)
    third = t * math.sqrt(target_metric * (1 - target_metric) / n_sample + (t / (2 * n_sample)) ** 2)
    return first * (second - third), first * (second + third)


def get_confidence_interval_by_size_estimated(dict_of_binary_freqs: tp.Dict,
                                              n_gened_set: int,
                                              metric: str='f1-score',
                                              n_sample: int=300,
                                              target_metric: float=0.7, 
                                              quantile: float=0.95) -> tp.Tuple[float, float]:
    '''
    Подсчет границ доверительного интервала генерацией выборок с известными долями значений бинарной матрицы
    Parameters:
        dict_of_binary_freqs: Словарь, в котором должны быть ключи 'TP', 'FP', 'FN' и 'TN', 
            сумма значений которых должна равняться 1
        n_gened_set: Число генерируемых выборок с долями dict_of_binary_freqs для каждого размера выборки
        metric: Целевая метрика. Поддерживаются значения 'accuracy', 'f1-score', 'precision', 'recall'
        n_sample: Размер выборки
        target_metric: Желаемое качество метрики
        quantile: Квантиль доверительного интервала; должен находиться в интервале (0; 1)
    Return:
        Левая и правая граница доверительного интервала
    '''
    all_options = ['TP', 'FP', 'FN', 'TN']
    probs_list = [dict_of_binary_freqs[i] for i in all_options]
    total_metrics = {'recall': [], 
                     'precision': [], 
                     'f1-score': []}
    for i in range(n_gened_set):
        random.seed(a=i, version=2)
        gened_set = random.choices(all_options, weights=probs_list, k=n_sample)
        tp, fp, fn = gened_set.count('TP'), gened_set.count('FP'), gened_set.count('FN')       
        if (tp > 0 or fp > 0):
            pr = tp / (tp + fp)
            total_metrics['precision'].append(pr)
        else:
            pr = None
        if (tp > 0 or fn > 0):
            rec = tp / (tp + fn)
            total_metrics['recall'].append(rec)
        else:
            rec = None
        if (pr and rec):
            f1 = 2*pr*rec/(pr+rec)
            total_metrics['f1-score'].append(f1)
    left_border = np.quantile(total_metrics[metric], (1 - quantile) / 2)
    right_border = np.quantile(total_metrics[metric], (1 - quantile) / 2 + quantile)
    return left_border, right_border


def plot_metric_sens_by_size(l_borders: tp.List[float],
                             r_borders: tp.List[float],
                             n_samples_list: tp.List[int],
                             cur_n_sample: int,
                             target_metric: float=0.7) -> tp.Any:
    '''
    Отрисовка графика зависимости доверительного интервала от размера выборки
    Parameters:
        l_borders: Список с левыми значениями доверительного интервала
        r_borders: Список с правыми значениями доверительного интервала
        n_samples_list: Список с значениями размера выборки для построения доверительного интервала
        cur_n_sample: Текущий размер выборки
        target_metric: Целевое значение метрики
    Return:
        График зависимости доверительного интервала от размера выборки
    '''
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.scatter(x=n_samples_list, y=l_borders, c='blue', alpha = 0.8, s = 10)
    ax.scatter(x=n_samples_list, y=r_borders, c='blue', alpha = 0.8, s = 10)
    ax.fill_between(x=n_samples_list, y1=l_borders, y2=r_borders,  color='green', alpha=0.3)
    ax.axhline(y=target_metric, color='black', linestyle='-', label = 'Желаемое значение метрики качества')
    ax.axvline(x=cur_n_sample, color='red', linestyle='--', label = 'Текущий размер выборки')
    ax.set_xscale('log')
    ax.set_xlabel('Размер выборки')
    ax.set_ylabel('Границы качества')
    ax.set_title(f'Зависимость доверительного интервала от размера выборки')
    ax.legend()
    return fig