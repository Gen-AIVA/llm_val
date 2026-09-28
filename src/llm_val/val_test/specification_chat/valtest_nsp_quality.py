import os
import re
import random
import numpy as np
import typing as tp
import scipy.sparse
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.parse_dialogues import parse_html_dialogue

def report_valtest_nsp_quality(statistics: tp.Dict[str, tp.Any], 
                               nsp_prob_fig: tp.Optional[tp.Any]=None) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        statistics: Словарь с посчитанными вероятностями NSP и значениями из доверительного интервала
            между человеком и LLM и диалогов между людьми
        nsp_prob_fig: График с распределениями вероятностей пар с правильными (предложения из одного диалога следуют друг за другом) и неправильными предложениями
    Return: 
        report в соответствии с концепцией llm_val
    '''
    corr = statistics['correct']
    incorr = statistics['incorrect']
    corr_lb, corr_rb = corr['prob_conf'][0], corr['prob_conf'][-1]
    incorr_lb, incorr_rb = incorr['prob_conf'][0], incorr['prob_conf'][-1]
    corr_conf = f'[{corr_lb}; {corr_rb}]'
    incorr_conf = f'[{incorr_lb}; {incorr_rb}]'
    result = pd.DataFrame({'Наименование класса': ['Правильная следующая фраза', 'Неправильная следующая фраза'],
                           'Среднее значение вероятности NSP': [np.mean(corr['prob_list']), np.mean(incorr['prob_list'])],
                           'Доверительный интервал': [corr_conf, incorr_conf]})
    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_dataframes'] = [result]
    res_dict['result_plots'] = [nsp_prob_fig]
    return res_dict


def valtest_nsp_quality(sampler: Sampler,
                        data_type: str='test',
                        seed: int=42,
                        quantile: float=0.95,
                        use_precomputed_emb: bool=True,
                        model: tp.Optional[tp.Any]=None,
                        toloka_emb: tp.Optional[str]=None,
                        statistics: tp.Optional[tp.Dict[str, tp.Any]]=None,
                        **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на оценку качества решения задачи NSP в диалоге
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на котором проводится анализ. Hint: название атрибута в sampler'е
        seed: Начальное значение датчика случайной величины
        quantile: Квантиль доверительного интервала
        use_precomputed_emb: Флаг, отвечающий за то, будут ли использоваться предпосчитанные эмбеддинги, хранящиеся в sampler'е
        model: Опциональная обёртка модели для получения эмбеддингов предложений
        toloka_emb: Опциональное значение пути к предпосчитанным эмбеддингам для датасета Toloka Persona Chat Rus
        statistics: Опциональное значение статистик для правильных (предложения из одного диалога следуют друг за другом) и неправильных пар предложений.
            Если не None, то подсчет метрики не производится
    Return:
        Словарь с концепцией llm_val
    '''
    if statistics is None:
        statistics = get_nsp_statistics(sampler,
                                        data_type,
                                        seed,
                                        quantile,
                                        use_precomputed_emb,
                                        model,
                                        toloka_emb)
    nsp_prob_fig = plot_nsp_probabilities(statistics, 
                                          quantile)
    report = report_valtest_nsp_quality(statistics, 
                                        nsp_prob_fig)
    return {'report': report,
            'precomputed': statistics}


def get_nsp_statistics(sampler: Sampler,
                       data_type: str='test',
                       seed: int=42,
                       quantile: float=0.95,
                       use_precomputed_emb: bool=True,
                       model: tp.Optional[tp.Any]=None,
                       toloka_emb: tp.Optional[str]=None) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Any]:
    '''
    Функция, возвращающая словарь с посчитанными оценками вероятностей того, что предложения следуют друг за другом, и значениями вероятностей,
        попавших в доверительный интервал
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на котором проводится анализ. Hint: название атрибута в sampler'е
        seed: Начальное значение датчика случайной величины
        quantile: Квантиль доверительного интервала
        use_precomputed_emb: Флаг, отвечающий за то, будут ли использоваться предпосчитанные эмбеддинги, хранящиеся в sampler'е
        model: Опциональная обёртка модели для получения эмбеддингов предложений
        toloka_emb: Опциональное значение пути к предпосчитанным эмбеддингам для датасета Toloka Persona Chat Rus
    Return:
        Словарь с посчитанными оценками вероятностей и значениями вероятностей, попавших в доверительный интервал
    '''
    X = getattr(sampler, data_type)['X']
    rng = np.random.default_rng(seed=seed)
    if use_precomputed_emb:
        llm_emb = X.groupby('dialogue_id')['embedding'].apply(list).to_list()
    else:
        texts = X.groupby('dialogue_id')['content'].apply(list).to_list()
        llm_emb = [model.encode(x) for x in tqdm(texts)]
    if toloka_emb is None:
        toloka = pd.read_pickle(os.environ['TOLOKA_PATH'])
        toloka_texts = toloka['dialogue'].to_list()
        toloka_texts = [parse_html_dialogue(dialogue) for dialogue in toloka_texts]
        human_emb = [model.encode(x) for x in tqdm(toloka_texts)]
    else:
        toloka = pd.read_pickle(toloka_emb)
        human_emb = toloka['embeddings'].to_list()
    train, test = get_train_test_data(llm_emb, 
                                      human_emb, 
                                      rng)
    statistics = calc_nsp_statistics(train,     
                                     test, 
                                     quantile, 
                                     seed)
    return statistics


def get_pairs_df(dataset: tp.List[tp.Any],
                 rng: np.random.Generator) -> tp.Dict[str, tp.Any]:
    '''
    Функция, возвращающая словарь с парами предложений, которые могут либо следовать друг за другом, либо случайно взяты из разных диалогов,
        а также метками того, действительно ли предложения в диалоге следуют друг за другом
    Parameters:
        dataset: Список с эмбеддингами фраз диалогов
        rng: Датчик случайной величины
    Return:
        Cловарь с парами предложений, которые могут либо следовать друг за другом, либо случайно взяты из разных диалогов,
            а также метками того, действительно ли предложения в диалоге следуют друг за другом
    '''
    correct = []
    incorrect = []
    for dialogue in dataset:
        for i in range(len(dialogue) - 1):
            correct.append(np.concatenate([dialogue[i], dialogue[i+1]]))
            random_ids = rng.integers(0, len(dataset), 2)
            txt1 = rng.choice(dataset[random_ids[0]])
            txt2 = rng.choice(dataset[random_ids[1]])
            incorrect.append(np.concatenate([txt1, txt2]))
    ones = np.array([1 for _ in range(len(correct))])
    zeroes = np.array([0 for _ in range(len(incorrect))]) 
    return {'texts': np.concatenate([correct, incorrect]),
            'label': np.concatenate([ones, zeroes])}
    

def get_train_test_data(llm_emb: tp.List[np.array], 
                        human_emb: tp.List[np.array],
                        rng: np.random.Generator) -> tp.Tuple[tp.List[str], tp.Any]:
    '''
    Функция, формирующая обучающую выборку (диалоги людей из датасета Toloka Persona Chat Rus) и тестовую (диалоги с LLM)
    Parameters:
        llm_emb: Список с эмбеддингами фраз диалогов с LLM 
        human_emb: Список с эмбеддингами фраз диалогов людей
        rng: Датчик случайной величины
    Return:
        Обучающая и тестовая выборки в виде словарей, в которых хранятся эмбеддинги предложений и метки того, действительно ли они следуют
            друг за другом в диалоге
    '''
    train = get_pairs_df(human_emb, 
                         rng)
    test = get_pairs_df(llm_emb, 
                        rng)
    return train, test


def get_conf_interval(values: tp.List[float], 
                      quantile: float) -> tp.List[float]:
    '''
    Подсчет доверительного интервала
    Parameters:
        values: Список значений метрик
        quantile: Квантиль доверительного интервала
    Return:
        Список значений метрик, попадающих в доверительный интервал
    '''
    values = sorted(values)
    left = (1 - quantile) / 2
    right = quantile + left
    left_bound = int(left * len(values))
    right_bound = int(right * len(values))
    return values[left_bound:right_bound]

def calc_nsp_statistics(train: tp.Dict[str, tp.Any],
                        test: tp.Dict[str, tp.Any],
                        quantile: float=0.95,
                        seed: int=42) -> tp.Dict[str, tp.Any]:
    '''
    Функция для подсчета оценки вероятностей того, следуют ли фразы друг за другом в диалоге
    Parameters:
        train: Словарь с обучающей выборкой
        test: Словарь с тестовой выборкой
        quantile: Квантиль доверительного интервала
        seed: Начальное значение датчика случайной величины
    Return:
        Словарь с посчитанными оценками вероятностей для корректных пар (предложения из одного диалога следуют друг за другом) и некорректных пар 
            и значениями вероятностей попавших в доверительный интервал
    '''
    bin_model = CatBoostClassifier(random_seed=seed, 
                                   iterations=1000, 
                                   eval_metric='AUC')
    X_train = train['texts']
    y_train = train['label']
    X_test = test['texts']
    y_test = test['label']
    bin_model.fit(X_train, 
                  y_train, 
                  verbose=False)
    y_preds = bin_model.predict_proba(X_test)[:, 1]
    pred_df = pd.DataFrame({'proba': y_preds, 'label': y_test})
    correct_values = pred_df[pred_df['label'] == 1]['proba'].to_list()
    incorrect_values = pred_df[pred_df['label'] == 0]['proba'].to_list()
    result = {}
    result['correct'] = {'prob_list': correct_values,
                         'prob_conf': get_conf_interval(correct_values, quantile)}
    result['incorrect'] = {'prob_list': incorrect_values,
                           'prob_conf': get_conf_interval(incorrect_values, quantile)}
    return result


def plot_nsp_probabilities(statistics: tp.Dict[str, tp.Any], 
                           quantile: float=0.95, 
                           n_bins: int=100) -> tp.Any:
    '''
    Функция построения графика распределения плотностей вероятностей для корректных (предложения следуют друг за другом) 
        и некорректных пар предложений
    Parameters:
        statistics: Словарь с посчитанными оценками вероятностей, а также значениями, попавшими в доверительный интервал
        quantile: Квантиль доверительного интервала
        n_bins: Число бинов
    Return:
        График функции распределения плотностей вероятностей для корректных (предложения из одного диалога следуют друг за другом) и некорректных пар предложений
    '''
    correct = statistics['correct']
    incorrect = statistics['incorrect']
    left_bound_corr = min(correct['prob_conf'])
    right_bound_corr = max(correct['prob_conf'])
    left_bound_incorr = min(incorrect['prob_conf'])
    right_bound_incorr = max(incorrect['prob_conf'])
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_title(f'NSP probabilities', fontsize=16)
    ax.set_xlabel('Metric values', fontsize=14)
    ax.set_ylabel('% of evidence', fontsize=14)
    bins = np.linspace(min(min(correct['prob_list']), min(incorrect['prob_list'])), 
                       max(max(correct['prob_list']), max(incorrect['prob_list'])),
                       n_bins)
    y, _, _ = ax.hist(correct['prob_list'], weights=np.ones_like(correct['prob_list'])/len(correct['prob_list']),
                      label='Correct continuation', bins=bins, alpha=0.5, color='orange')
    plot_conf_height = np.max(y) * 0.05
    ax.plot([left_bound_corr, left_bound_corr, right_bound_corr, right_bound_corr],
            [0, plot_conf_height, plot_conf_height, 0],
            color='blue', linestyle='dashed',
            label=f'Correct continuation {quantile} confidence interval')
    y, _, _ = ax.hist(incorrect['prob_list'], weights=np.ones_like(incorrect['prob_list'])/len(incorrect['prob_list']),
                      label='Incorrect continuation', bins=bins, alpha=0.5, color='blue')
    plot_conf_height = np.max(y) * 0.05
    ax.plot([left_bound_incorr, left_bound_incorr, right_bound_incorr, right_bound_incorr],
            [0, plot_conf_height, plot_conf_height, 0],
            color='red', linestyle='dashed',
            label=f'Incorrect continuation {quantile} confidence interval')
    ax.legend()
    return fig
