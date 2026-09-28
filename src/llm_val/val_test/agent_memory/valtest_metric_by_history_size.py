import numpy as np
import pandas as pd
import typing as tp
from tqdm import tqdm
import seaborn as sns
from copy import deepcopy
from scipy.stats import norm
from physt import h1, binnings
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import defaultdict

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler

def report_valtest_metric_by_history_size(corr_results: tp.defaultdict[str, tp.defaultdict[str, np.float64,]],
                                          fig_results: tp.Any) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        corr_results: Значения статистик и p-value 
        fig_results: Словарь с отрисованными boxplot'ами
    Return:
        report в соответствии с концепцией llm_val
    '''
    corr_df = get_corr_df(corr_results)

    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_plots'] = fig_results
    res_dict['result_dataframes'] = [corr_df]
    
    return res_dict


def valtest_metric_by_history_size(sampler: Sampler,
                                   scorer: Scorer,
                                   tokenizer: tp.Any,
                                   data_type: str='test',
                                   bins: tp.Union[str, int]=20,
                                   metric_agg: str='single_mean',
                                   n_bootstrap: int=100,
                                   quantile: float=0.95,
                                   random_state: int=666) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на наличие зависимости между качеством ответа агента и длиной цепочки общения агента.
    При подсчёте корреляции используются метки качества итогового ответа.
    Parameters:
        sampler: Объект класса Sampler
        scorer: Объект класса Scorer
        tokenizer: Модель-токенизатор с методом tokenize, возвращающим словарь с ключом 'input_ids'
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        bins: Название способа бинаризации или количество бинов
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        n_bootstrap: Количество итераций бутстрапирования
        quantile: Квантиль доверительного интервала. Должен быть в диапазоне (0; 1)
        random_state: Рандом стейт
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    data = deepcopy(getattr(sampler, data_type))
    
    data['X']['n_tokens'] = (data['X']['question'] + data['X']['model_answer'] + data['X']['reasoning']).apply(lambda x: len(tokenizer.tokenize(x)['input_ids']))
    data_grouped = get_grouped_data(data)

    corr_results = calc_corr(data_grouped)
    bins_results = calc_bins_stat(data_grouped,
                                  scorer,
                                  bins,
                                  metric_agg,
                                  n_bootstrap,
                                  quantile,
                                  random_state)
    
    fig_results = plot_boxplots(data_grouped, 
                                bins_results)
    
    report = report_valtest_metric_by_history_size(corr_results,
                                                   fig_results)

    precomputed = {'correlation_results': corr_results,
                   'bins_results': bins_results}
    
    return {'report': report,
            'precomputed': precomputed}


def get_grouped_data(data: tp.Dict[str, pd.DataFrame]) -> pd.DataFrame:
    '''
    Агрегирует данные в разрезе диалогов request_id. Для каждого диалога суммируется количество токенов и количество сообщений.
    Метка качества берется из последнего взаимодействия (итоговый ответ).
    Parameters:
        data: Релевантная выборка из сэмплера. Содержит ключи 'X' и 'y'
    Return:
        Агрегированные данные
    '''
    tmp = pd.concat([data['X'], data['y']], axis=1)
    tmp.sort_values(by=['request_id', 'order_idx'], ascending=True, inplace=True)
    
    agg_dict = {}
    agg_dict['n_tokens'] = ('n_tokens', 'sum')
    agg_dict['n_messages'] = ('n_tokens', 'size')
    for col in data['y'].columns:
        agg_dict[col] = (col, 'last')
    
    data_grouped = tmp.groupby(by='request_id', as_index=False).agg(**agg_dict)

    return data_grouped


def calc_corr(data_grouped: pd.DataFrame) -> tp.defaultdict[str, tp.defaultdict[str, tp.Dict[str, np.float64]]]:
    '''
    Подсчет корреляции Спирмена между метриками качества и длинами истории общения с агентом
    Parameters:
        data_grouped: Агрегированные по диалогам данные: длина истории общения и итоговая метрика
    Return:
        Значения коэффициентов корреляции и p-value для каждой метрики
    '''
    corr_results = defaultdict(
        lambda: defaultdict(
            lambda: {'correlation': None, 'p_value': None}
        )
    )
    
    metric_cols = [col for col in data_grouped.columns if col not in ['request_id', 'n_tokens', 'n_messages']]	
    for metric_name in metric_cols:
        for history_len_measure in ['n_tokens', 'n_messages']:
            corr_results[metric_name][history_len_measure]['correlation'], \
            corr_results[metric_name][history_len_measure]['p_value'] = spearmanr(
                data_grouped[history_len_measure], data_grouped[metric_name]
            )
    
    return corr_results


def calc_bins_stat(data_grouped: pd.DataFrame,
                   scorer: Scorer,
                   bins: tp.Union[str, int]=20,
                   metric_agg: str='single_mean',
                   n_bootstrap: int=100,
                   quantile: float=0.95,
                   random_state: int=666) -> tp.Tuple[tp.List, tp.List, tp.Dict]:
    '''
    Подсчет бинов и статистик по ним
    Parameters:
        data_grouped: Агрегированные по диалогам данные: длина истории общения и итоговая метрика
        scorer: Объект класса Scorer
        bins: Название способа бинаризации или количество бинов
        metric_agg: Название метрики агрегации
        n_bootstrap: Количество итераций бутстрапирования
        quantile: Квантиль доверительного интервала. Должен быть в диапазоне (0; 1)
        random_state: Рандом стейт
    Return:
        Словарь, в котором для каждого критерия приводятся:
            - bins_val: List с номерами бинов;
            - bins_bound: List с границами бинов;
            - bins_stat: Dict с средними значениями метрик и доверительными интервалами в разрезе бинов.
    '''
    bins_results = defaultdict(
        lambda: defaultdict(
            lambda: {'bins_val': None, 'bins_bound': None, 'bins_stat': None}
        )
    )

    metric_cols = [col for col in data_grouped.columns if col not in ['request_id', 'n_tokens', 'n_messages']]
    for metric_name in metric_cols:
        for history_len_measure in ['n_tokens', 'n_messages']:
            hist = h1(data_grouped[history_len_measure], bins)
    
            bins_results[metric_name][history_len_measure]['bins_val'] = [hist.find_bin(i) for i in data_grouped[history_len_measure].values]
            bins_results[metric_name][history_len_measure]['bins_bound'] = [f'{np.floor(i[0]):.0f}-{np.floor(i[1]):.0f}' for i in hist.bins]
            bins_results[metric_name][history_len_measure]['bins_stat'] = calc_conf_interval(data_grouped[metric_name],
                                                                                             bins_results[metric_name][history_len_measure]['bins_val'],
                                                                                             scorer,
                                                                                             metric_agg,
                                                                                             n_bootstrap,
                                                                                             quantile,
                                                                                             random_state)
    return bins_results


def calc_conf_interval(y: pd.DataFrame,
                       bins: tp.List[int],
                       scorer: Scorer,
                       metric_agg: str='single_mean',
                       n_bootstrap: int=100,
                       quantile: float=0.95,
                       random_state: int=666) -> tp.Dict[str, tp.Any]:
    '''
    Подсчет доверительных интервалов для каждого бина
    Parameters:
        y: Значения метрик качества в разрезе вопросов
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
    grouped = y.groupby(by=bins)
    
    for i, group in grouped:
        mean_value = calc_metric_values(group,
                                        scorer,
                                        metric_agg)
        stat_bins[i] = {'mean': mean_value}
        
        if ((y > 1) | (y < 0)).any().any():
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


def calc_metric_values(y: pd.DataFrame, 
                       scorer: Scorer,
                       metric_agg: str='single_mean') -> tp.Dict[str, float]:
    '''
    Подсчет метрики качества
    Parameters:
        y: Значения метрик качества в разрезе вопросов 
        scorer: Объект класса Scorer
        metric_agg: Название метрики агрегации
    Return:
        Dict с подсчитанными метриками
    '''
    sampler = Sampler(test={'X': None, 'y': pd.DataFrame(y)})
    
    return scorer.calc(sampler,
                       'test',
                       metric_agg)


def calc_conf_interval_by_bootstrap(y: pd.DataFrame,
                                    scorer: Scorer,
                                    metric_agg: str='single_mean',
                                    n_bootstrap: int=100,
                                    quantile: float=0.95,
                                    random_state: int=666) -> tp.Dict[str, tp.Tuple[float]]:
    '''
    Подсчет доверительного интервала бутстрапированием
    Parameters:
        y: Значения метрик качества в разрезе вопросов 
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
        sample = y.sample(frac=1, random_state=rng, replace=True)
        sampler = Sampler(test={'X': None, 'y': pd.DataFrame(sample)})

        metric_value = scorer.calc(sampler, 'test', metric_agg)
        metric_values.append(metric_value)
        
    metric_values = {j: sorted([values[j] for values in metric_values]) for j in metric_values[0].keys()}
    
    return {k: calculate_quantile(metric_values[k]) for k in metric_values.keys()}


def calc_conf_interval_estimation(y: pd.DataFrame, 
                                  mean_value: tp.Dict[str, float],
                                  quantile: float=0.95) -> tp.Dict[str, tp.Tuple[float, float]]:
    '''
    Подсчет доверительного интервала
    Parameters:
        y: Значения метрики качества в разрезе вопросов 
        mean_value: Словарь со средними значениями метрик 
        quantile: Квантиль доверительного интервала. Должен быть в диапазоне (0; 1)
    Return:
        Словарь с доверительными интервалами метрик
    '''
    mean_value = pd.DataFrame(mean_value, index=[0])
    n_sample = len(y)
    
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
        metric_values: Значения метрик
        quantile: Квантиль доверительного интервала. Должен быть в диапазоне (0; 1)
    Return:
        Левая и правая границы доверительного интервала
    '''
    left = (1 - quantile) / 2
    right = quantile + left
    left_bound = int(left * len(metric_values))
    right_bound = int(right * len(metric_values))
    
    return metric_values[left_bound], metric_values[right_bound]


def plot_boxplots(data_grouped: pd.DataFrame,
                  bins_results: tp.Dict[str, [tp.List, tp.List, tp.Dict]]) -> tp.Any:
    '''
    Отрисовка boxplot'ов
    Parameters:
        data_grouped: Агрегированные по диалогам данные: длина истории общения и итоговая метрика
        bins_results: Характеристики бинов 
    Return:
        Фигуры с отрисованными boxplot'ами
    '''  
    metric_cols = [col for col in data_grouped.columns if col not in ['request_id', 'n_tokens', 'n_messages']]

    n_fig = len(metric_cols) * 2
    fig, ax = plt.subplots(n_fig, figsize=(20, 10*n_fig), squeeze=False)

    n_ax = 0
    for metric_name in metric_cols:
        for history_len_measure in bins_results[metric_name].keys():
            bins_bound = bins_results[metric_name][history_len_measure]['bins_bound']
            bins_val = bins_results[metric_name][history_len_measure]['bins_val']
            tmp = [bins_bound[i] for i in bins_val]

            sns.boxplot(data_grouped, x=data_grouped[metric_name], y=tmp, ax=ax[n_ax][0], order=bins_bound)
            ax[n_ax][0].set_xlabel(f'Значение метрики {metric_name}', fontsize=16)
            ax[n_ax][0].set_ylabel(f'Длина истории диалога ({history_len_measure})', fontsize=16)
            ax[n_ax][0].tick_params(labelsize=12)
            n_ax += 1
                       
    return fig


def get_corr_df(corr_res: tp.defaultdict[str, tp.defaultdict[str, tp.Dict[str, np.float64]]]) -> pd.DataFrame:
    '''
    Преобразование словаря в датафрейм
    Parameters:
        corr_res: Словарь с подсчитанными корреляциями между метриками качества и длинами истории общения с агентом
    Return:
        Датафрейм с подсчитанными корреляциями между метриками качества и длинами истории общения с агентом
    '''
    corr_df = pd.DataFrame.from_dict(
        {
            (metric, measure): vals
            for metric, inner_dict in corr_res.items()
            for measure, vals in inner_dict.items()
        },
        orient='index',
        columns=['correlation', 'p_value']
    )
    corr_df.index.names = ['metric_name', 'history_len']

    return corr_df