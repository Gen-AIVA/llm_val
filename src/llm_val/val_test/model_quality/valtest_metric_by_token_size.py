import numpy as np
import pandas as pd
import typing as tp
import seaborn as sns
from tqdm import tqdm
from copy import deepcopy
from typing import Literal
from scipy.stats import norm
import matplotlib.pyplot as plt
from collections import defaultdict

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.utils.binarization_algos import iterative_binning, bin_by_spec_with_merge


def report_valtest_metric_by_token_size(fig: tp.Any,
                                        data_type: str='test',
                                        bins_stat: tp.Optional[tp.Dict[int, tp.Dict[str, tp.Any]]]=None,
                                        bins_bound: tp.Optional[tp.List[str]]=None):
    '''
    Создание report'а
    Parameters:
        fig: Фигура с отрисованными boxplot'ами
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        bins_stat: Dict с средними значениями метрик и доверительными интервалами в разрезе бинов;
        bins_bound: List с границами бинов.
    Return:
        report в соответствии с концепцией llm_val
    '''
    result_dataframes = []
    bins_dict = defaultdict(lambda: defaultdict(dict))
    color = 'gray'

    if (not bins_stat is None) and (not bins_bound is None):
        for k, v in bins_stat.items():
            for stat_name, stat_value in v.items():
                for metric_name, metric_value in stat_value.items():
                    bins_dict[metric_name][bins_bound[k]][stat_name] = metric_value
        for k, v in bins_dict.items():
            tmp_df = pd.DataFrame(v).T
            tmp_df['mean'] = tmp_df['mean'].apply(lambda x: f'{x:.3f}')
            tmp_df['conf_int'] = tmp_df['conf_int'].apply(lambda x: f'[{x[0]:.3f}; {x[1]:.3f}]')
            tmp_df.columns = [f'Среднее значение метрики {k}', f'Доверительный интервал метрики {k}']
            
            result_dataframes.append(tmp_df)
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = [fig]
    res_dict['result_dataframes'] = result_dataframes
    return res_dict


def valtest_metric_by_token_size(sampler: Sampler,
                                 scorer: Scorer,
                                 data_type: str='test',
                                 token_counter_model: tp.Optional[tp.Any]=None,
                                 min_bin_size: float=0.05,
                                 bin_algo: Literal['iter', 'merge'] = 'iter',
                                 metric_agg: str='single_mean',
                                 n_bootstrap: int=100,
                                 quantile: float=0.95,
                                 random_state: int=73,
                                 **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на анализ динамики ключевой метрики качества модели в зависимости от длины запроса
    Parameters:
        sampler: Объект класса Sampler
        scorer: Объект класса Scorer
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е    
        token_counter_model: Модель подсчета токенов. Должна иметь метод get_token_count(v: str)
        min_bin_size: Минимальная доля наблюдений, содержащаяся в одном бакете.
        bin_algo: Алгоритм формирования бинов. По умолчанию: 'iter'.
                  - 'iter': итеративный алгоритм (рекомендуется для равномерного распределения)
                  - 'merge': алгоритм слияния (эффективен для skewed распределений)
                  Подробнее см. utils.binarization_algos
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        n_bootstrap: Количество итераций бутстрапирования
        quantile: Квантиль доверительного интервала. Должен быть в диапазоне (0; 1)
        random_state: Рандом стейт
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    bin_map = {}
    bins_bound = []
    bins_stat, bins_val = None, None
    data = getattr(sampler, data_type)
    questions = data['X']['raw_prompt']
    if token_counter_model is None:
        len_questions = questions.apply(lambda x: estimate_tokens(x))
    else:
        len_questions = questions.apply(lambda x: token_counter_model.get_token_count(x))

    if bin_algo == 'iter':
        bins = iterative_binning(len_questions, min_bin_size=min_bin_size)
        bin_map = {val: i for i, bin in enumerate(bins) for val in bin}
        bins_val = len_questions.map(bin_map).values
        bins = [(min(bin), max(bin)) for bin in bins if bins]
    else:
        bins, bins_val = bin_by_spec_with_merge(len_questions, min_bin_size=min_bin_size)

    for idx, (lo, hi) in enumerate(bins):
        if lo == hi:
            bins_bound.append(f"{lo}")
        elif idx < len(bins) - 1:
            bins_bound.append(f"[{lo}–{hi+1})")
        else:
            bins_bound.append(f"[{lo}–{hi}]")

    bins_stat = calc_conf_interval(data['y'],
                                   bins_val,
                                   scorer,
                                   metric_agg,
                                   n_bootstrap,
                                   quantile,
                                   random_state)
    fig = plot_boxplot(data['y'], bins_val, bins_bound)
    
    report = report_valtest_metric_by_token_size(fig=fig,
                                                 data_type=data_type,
                                                 bins_stat=bins_stat,
                                                 bins_bound=bins_bound)
    tokens_to_bins_precomputed = pd.DataFrame(len_questions.values, 
                                              columns=['token_size'], 
                                              index=len_questions.index)
    tokens_to_bins_precomputed['bins'] = bins_val
    precomputed = {'token_and_bins': tokens_to_bins_precomputed}
    return {'report': report,
            'precomputed': precomputed}


def calc_conf_interval(data: pd.DataFrame,
                       bins: tp.List[int],
                       scorer: Scorer,
                       metric_agg: str='single_mean',
                       n_bootstrap: int=100,
                       quantile: float=0.95,
                       random_state: int=73) -> tp.Dict[str, tp.Any]:
    '''
    Подсчет доверительных интервалов для каждого бина
    Parameters:
        data: Данные
        bins: Бины для каждого вопроса
        scorer: Объект класса Scorer
        metric_agg: Название метрики агрегации
        n_bootstrap: Количество итераций бутстрапирования
        quantile: Квантиль доверительного интервала. Должен быть в диапазоне (0; 1)
        random_state: Рандом стейт
    Return:
        Два dict'а с значениями статистики и p-value для каждой метрики
    '''
    stat_bins = {}
    grouped = data.groupby(by=bins)
    for i, group in grouped:
        mean_value = calc_metric_values(group,
                                        scorer,
                                        metric_agg)
        stat_bins[i] = {'mean': mean_value}
        if ((data > 1) | (data < 0)).any().any():
            stat_bins[i]['conf_int'] = calc_conf_interval_by_bootstrap(group,
                                                                       scorer,
                                                                       metric_agg,
                                                                       n_bootstrap,
                                                                       quantile,
                                                                       random_state)
        else:
            stat_bins[i]['conf_int'] = calc_conf_interval_estimation(group,
                                                                     mean_value,
                                                                     quantile)
    return stat_bins


def calc_metric_values(data: pd.Series, 
                       scorer: Scorer,
                       metric_agg: str='single_mean') -> tp.Dict[str, float]:
    '''
    Подсчет метрики качества
    Parameters:
        data: Series с значениями метрики качества в разрезе вопросов
        scorer: Объект класса Scorer
        metric_agg: Название метрики агрегации
    Return:
        Dict с подсчитанными метриками
    '''
    sampler = Sampler(test={'X': None, 'y': pd.DataFrame(data)})
    return scorer.calc(sampler,
                       'test',
                       metric_agg)


def calc_conf_interval_by_bootstrap(data: pd.DataFrame,
                                    scorer: Scorer,
                                    metric_agg: str='single_mean',
                                    n_bootstrap: int=100,
                                    quantile: float=0.95,
                                    random_state: int=73) -> tp.Dict[str, tp.Tuple[float]]:
    '''
    Подсчет доверительного интервала бутстрапированием
    Parameters:
        data: DataFrame с значениями метрик качества в разрезе вопросов
        scorer: Объект класса Scorer
        metric_agg: Название метрики агрегации
        n_bootstrap: Количество итераций бутстрапирования
        quantile: Квантиль доверительного интервала. Должен быть в диапазоне (0; 1)
        random_state: Рандом стейт
    Return:
        Словарь с доверительными интервалами метрик
    '''
    metric_values = []
    rng = np.random.default_rng(random_state)
    for _ in tqdm(range(n_bootstrap)):
        sample = data.sample(frac=1, 
                             random_state=rng,
                             replace=True)
        sampler = Sampler(test={'X': None, 'y': pd.DataFrame(sample)})
        metric_values.append(scorer.calc(sampler,
                                         'test',
                                         metric_agg))
    metric_values = {j: sorted([values[j] for values in metric_values]) for j in metric_values[0].keys()}
    return {k: calculate_quantile(metric_values[k]) for k in metric_values.keys()}


def calc_conf_interval_estimation(data: pd.DataFrame, 
                                  mean_value: tp.Dict[str, float],
                                  quantile: float=0.95) -> tp.Dict[str, tp.Tuple[float, float]]:
    '''
    Подсчет доверительного интервала
    Parameters:
        data: DataFrame с значениями метрик качества в разрезе вопросов
        mean_value: Словарь со средними значениями метрик 
        quantile: Квантиль доверительного интервала. Должен быть в диапазоне (0; 1)
    Return:
        Словарь с доверительными интервалами метрик
    '''
    mean_value = pd.DataFrame(mean_value, index=[0])
    n_sample = len(data)
    if (quantile >= 1) or (quantile <= 0):
        raise ValueError('quantile должен быть в диапазоне (0; 1)')
    t = norm.ppf((1 - quantile) / 2 + quantile)
    first = n_sample / (t ** 2 + n_sample)
    second = mean_value + (t ** 2) / (2 * n_sample)
    third = t * np.sqrt(mean_value * (1 - mean_value) / n_sample + (t / (2 * n_sample)) ** 2)
    left = (first * (second - third)).to_dict('records')[0]
    right = (first * (second + third)).to_dict('records')[0]
    return {k: (left[k], right[k]) for k in left.keys()} 


def calculate_quantile(metric_values: tp.List[float],
                       quantile: float=0.95) -> tp.Tuple[float, float]:
    '''
    Подсчет границ доверительного интервала
    Parameters:
        metric_values: Список значений метрик
        quantile: Квантиль доверительного интервала. Должен быть в диапазоне (0; 1)
    Return:
        Левую и правую границы доверительного интервала
    '''
    left = (1 - quantile) / 2
    right = quantile + left
    left_bound = int(left * len(metric_values))
    right_bound = int(right * len(metric_values))
    return metric_values[left_bound], metric_values[right_bound]


def plot_boxplot(data: pd.DataFrame,
                 bins_val: tp.List[int],
                 bins_bound: tp.List[str]):
    '''
    Отрисовка boxplot'ов
    Parameters:
        data: Данные
        bins_val: Список с номерами бинов
        bins_bound: Границы бинов
    Return:
        Фигура с отрисованными boxplot'ами
    '''
    plot_df = data.copy()
    plot_df['bin_label'] = [bins_bound[i] for i in bins_val]
    n_fig = len(data.columns)
    fig, ax = plt.subplots(n_fig, figsize=(20, 10*n_fig), squeeze=False)
    for i, col_name in enumerate(data.columns):
        sns.boxplot(data=plot_df, x=col_name, y='bin_label', ax=ax[i][0], order=bins_bound)
        ax[i][0].set_xlabel('Значение метрики', fontsize=16)
        ax[i][0].set_ylabel('Длина запроса в токенах', fontsize=16)
        ax[i][0].tick_params(labelsize=12)
    return fig


def estimate_tokens(text: str) -> int:
    '''
    Эмпирическая оценка (1 токен ~ 4 символа для русского)
    Parameters:
        text: Входной текст
    Return:
        Приблизительное число токенов
    '''
    return len(text.encode('utf-8')) // 4 + 1