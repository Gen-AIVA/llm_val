import os
import torch
import numpy as np
import typing as tp
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.parse_dialogues import parse_html_dialogue

def report_valtest_text_coherence(statistics: tp.Dict[str, tp.Dict[str, tp.Any]],
                                  dialogue_plot: tp.Optional[tp.Any]=None,
                                  toloka_plot: tp.Optional[tp.Any]=None,
                                  llm_toloka_plot: tp.Optional[tp.Any]=None) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        statistics: Словарь с посчитанными косинусными близостями для изначальных диалогов и диалогов, полученных
            с помощью сэмплирования случайных фраз
        dialogue_plot: График для диалогов между человеком и LLM
        toloka_plot: График для диалогов между людьми
        llm_toloka_plot: График с распределениями косинусных близостей для диалогов между человеком и LLM и диалогов между людьми
    Return:
        report в соответствии с концепцией llm_val
    '''
    dialogue_real_list = statistics['llm']['real_cosine_list']
    toloka_real_list = statistics['human']['real_cosine_list']
    dialogue_rand_list = statistics['llm']['rand_cosine_list']
    toloka_rand_list = statistics['human']['rand_cosine_list']
    result = pd.DataFrame({'Наименование датасета': ['Диалоги человека с LLM', 
                                                     'Случайные фразы из диалогов человека с LLM', 
                                                     'Toloka Persona Chat Rus',
                                                     'Случайные фразы из Toloka'],
                           'Среднее значение cosine sim': [np.mean(dialogue_real_list),
                                                           np.mean(dialogue_rand_list),
                                                           np.mean(toloka_real_list),
                                                           np.mean(toloka_rand_list)],
                           'Доверительный интервал': [f'[{np.min(dialogue_real_list)}, {np.max(dialogue_real_list)}]',
                                                      f'[{np.min(dialogue_rand_list)}, {np.max(dialogue_rand_list)}]',
                                                      f'[{np.min(toloka_real_list)}, {np.max(toloka_real_list)}]',
                                                      f'[{np.min(toloka_rand_list)}, {np.max(toloka_rand_list)}]'],
                            'Количество наблюдений (диалогов)': [len(dialogue_real_list),
                                                                 len(dialogue_rand_list),
                                                                 len(toloka_real_list),
                                                                 len(toloka_rand_list)]})
    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_dataframes'] = [result]
    res_dict['result_plots'] = [dialogue_plot, toloka_plot, llm_toloka_plot]
    return res_dict


def valtest_text_coherence(sampler: Sampler,
                           data_type: str='test',
                           quantile: float=0.95,
                           n_samples: int=25,
                           seed: int=42,
                           use_precomputed_emb: bool=False,
                           model: tp.Optional[tp.Any]=None,
                           toloka_embeddings: tp.Optional[str]=None,
                           statistics: tp.Optional[tp.Dict[str, tp.Any]]=None,
                           **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест оценки связности диалога (информативный)
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на котором проводится анализ. Hint: название атрибута в sampler'е
        quantile: Квантиль доверительного интервала
        n_samples: Количество сэмплирований случайных диалогов в тесте
        seed: Начальное значение датчика случайной величины
        use_precomputed_emb: Флаг, отвечающий за то, будут ли в тесте использоваться предпосчитанные эмбеддинги из сэмплера или будут посчитаны новые
        model: Опциональный параметр - обёртка модели получения эмбеддингов
        toloka_embeddings: Путь к pickle-файлу с предпосчитанными эмбеддингами для датасета Toloka Persona Chat Rus
        statistics: Опциональное значение статистик для диалогов между человеком и LLM и диалогов между людьми.
            Если не None, то подсчет метрики не производится
    Return:
        Словарь с концепцией llm_val
    '''
    rng = np.random.default_rng(seed=seed)
    if statistics is None:
        dialogue_cosine_stats = get_mean_cosine_stats(sampler,
                                                      rng,
                                                      data_type, 
                                                      quantile,
                                                      n_samples,
                                                      use_precomputed_emb,
                                                      model)
        toloka_cosine_stats = get_toloka_cosine_stats(rng,
                                                      quantile,
                                                      n_samples,
                                                      toloka_embeddings,
                                                      model)
        statistics = {'llm': dialogue_cosine_stats,
                      'human': toloka_cosine_stats}
    llm_human_plot = plot_real_random_cosines(statistics['llm']['real_cosine_list'],
                                              statistics['llm']['rand_cosine_list'],
                                              statistics['llm']['real_cosine_conf'],
                                              statistics['llm']['rand_cosine_conf'],
                                              quantile,
                                              dialogue_type='LLM')
    toloka_plot = plot_real_random_cosines(statistics['human']['real_cosine_list'],
                                           statistics['human']['rand_cosine_list'],
                                           statistics['human']['real_cosine_conf'],
                                           statistics['human']['rand_cosine_conf'],
                                           quantile,
                                           dialogue_type='Toloka')
    llm_toloka_plot = plot_real_random_cosines(statistics['llm']['real_cosine_list'],
                                               statistics['human']['real_cosine_list'],
                                               statistics['llm']['real_cosine_conf'],
                                               statistics['human']['real_cosine_conf'],
                                               quantile,
                                               dialogue_type='LLM and Human',
                                               labels=('LLM', 'Human'))
    report = report_valtest_text_coherence(statistics, 
                                           llm_human_plot,
                                           toloka_plot,
                                           llm_toloka_plot)
    precomputed = statistics
    return {'report': report,
            'precomputed': statistics}


def get_mean_cosine_stats(sampler: Sampler,
                          rng: np.random.Generator,
                          data_type: str='test',
                          quantile: float=0.95,
                          n_samples: int=25,
                          use_precomputed_emb: bool=False,
                          model: tp.Optional[tp.Any]=None) -> tp.Dict[str, tp.Any]:
    '''
    Подсчет усреднённых косинусных близостей для всех диалогов в датасете и диалогов, 
        полученных сэмплированием случайных фраз
    Parameters:
        sampler: Объект класса sampler
        rng: Датчик случайной величины
        data_type: Тип данных, на котором проводится анализ. Hint: название атрибута в sampler'е
        quantile: Квантиль доверительного интервала
        n_samples: Количество сэмплирований случайных диалогов в тесте
        use_precomputed_emb: Флаг, отвечающий за то, будут ли в тесте использоваться предпосчитанные эмбеддинги из сэмплера или будут посчитаны новые
        model: Опционально обёртка для модели получения эмбеддингов
    Return:
        Словарь с посчитанными усреднёнными косинусными близостями для исходного датасета и датасета,
            полученного путём сэмплирования
    '''
    X = getattr(sampler, data_type)['X']
    if use_precomputed_emb:
        embeddings = X.groupby('dialogue_id')['embedding'].apply(list)
    else:
        dialogues = group_dialogues(X)
        embeddings = calc_embeddings(model, 
                                     dialogues)
    mean_cosine_list = calc_mean_cosines(embeddings)
    rand_cosine_list = calc_rand_cosines(embeddings, 
                                         rng, 
                                         n_samples)
    return {'real_cosine_list': mean_cosine_list,
            'real_cosine_conf': get_conf_int(mean_cosine_list),
            'rand_cosine_list': rand_cosine_list,
            'rand_cosine_conf': get_conf_int(rand_cosine_list)}


def get_toloka_cosine_stats(rng: np.random.Generator,
                            quantile: float=0.95,
                            n_samples: int=25,
                            toloka_embeddings: tp.Optional[str]=None,
                            model: tp.Optional[tp.Any]=None) -> tp.Dict[str, tp.Any]:
    '''
    Функция подсчета усреднённых косинусных близостей на исходном датасете и датасете, полученном сэмплированием
    Parameters:
        rng: Датчик случайной величины
        quantile: Квантиль доверительного интервала
        n_samples: Число сэмплирований исходного датасета
        toloka_embeddings: Путь к pickle-файлу с предпосчитанными эмбеддингами для датасета Toloka Persona Chat Rus
        model: Опционально обёртка для модели получения эмбеддингов
    Return:
        Словарь со списками посчитанных средних косинусных близостей
    '''
    if toloka_embeddings is None:
        toloka_df = pd.read_pickle(os.environ['TOLOKA_PATH'])
        toloka_dialogues = toloka_df['dialogue'].to_list()
        toloka_dialogues = [parse_html_dialogue(dial) for dial in toloka_dialogues]
        embeddings = calc_embeddings(model, 
                                     toloka_dialogues)
    else:
        emb_df = pd.read_pickle(toloka_embeddings)
        embeddings = emb_df['embeddings'].to_list()
    mean_cosine_list = calc_mean_cosines(embeddings)
    rand_cosine_list = calc_rand_cosines(embeddings, 
                                         rng, 
                                         n_samples)
    return {'real_cosine_list': mean_cosine_list,
            'real_cosine_conf': get_conf_int(mean_cosine_list),
            'rand_cosine_list': rand_cosine_list,
            'rand_cosine_conf': get_conf_int(rand_cosine_list)}


def calc_mean_cosines(embeddings: tp.List[np.array]) -> tp.List[float]:
    '''
    Функция для подсчета косинусного расстояния по каждому диалогу
    Parameters:
        embeddings: Список с эмбеддингами фраз диалогов
    Return:
        Список с усреднённым косинусным расстоянием для каждого диалога
    '''
    mean_cosine_list = []
    for dialogue in embeddings:
        cos_sim = cosine_similarity(dialogue, dense_output=True).diagonal(offset=1)
        mean_cosine_list.append(np.mean(cos_sim))
    return mean_cosine_list


def calc_rand_cosines(dialogues: tp.List[np.array],
                      rng: np.random.Generator,
                      n_samples: int=25) -> tp.List[float]:
    '''
    Подсчет косинусных близостей для вырванных из контекста диалогов. Формируется список
        заданного размера, в котором указано среднее косинусное расстояние для сэмпла,
        аналогичного по размерам исходным данным
    Parameters:
        dialogues: Список с эмбеддингами диалогов
        rng: Датчик случайной величины
        n_samples: Число, сколько раз будет происходить сэмплирование
    Return:
        Список размера n_samples с макро усреднениями косинусных расстояний для каждого сэмпла
    '''
    cosine_samples = []
    for i in range(n_samples):
        rand_dialogues = []
        for j, dialogue in enumerate(dialogues):
            rand_dialogue = []
            for k in range(len(dialogue)):
                dialogue = dialogues[rng.integers(len(dialogues))]
                phrase = dialogue[rng.integers(len(dialogue))]
                rand_dialogue.append(phrase)
            rand_dialogues.append(rand_dialogue)
        cosine_samples.extend(calc_mean_cosines(rand_dialogues))
    return cosine_samples


def calc_embeddings(model: tp.Any, 
                    dialogues: tp.List[tp.List[str]]) -> tp.List[np.array]:
    '''
    Функция для подсчета эмбеддингов диалогов
    Parameters:
        model: Обёртка для модели получения эмбеддингов
        dialogues: Список с диалогами
    Return:
        Список с эмбеддингами диалогов
    '''
    embeddings = []
    for dialogue in tqdm(dialogues):
        embeddings.append(model.encode(dialogue))
    return embeddings


def group_dialogues(X: pd.DataFrame) -> tp.List[tp.List[str]]:
    '''
    Получение списка с диалогами, представленными в виде списков
    Parameters:
        X: DataFrame с диалогами и id-шниками диалогов
    Return:
        Список с диалогами, представленнми в виде списков
    '''
    group_df = X.groupby('dialogue_id')['content'].apply(list).reset_index()
    dialogues = group_df['content'].to_list()
    return dialogues


def get_conf_int(values: tp.List[float],
                 quantile: float=0.95) -> tp.List[float]:
    '''
    Подсчет доверительного интервала и получение списка метрик, попавших в этот доверительный интервал
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


def plot_real_random_cosines(real_cosines: tp.List[float],
                             rand_cosines: tp.List[float],
                             conf_real_cosines: tp.List[float],
                             conf_rand_cosines: tp.List[float],
                             quantile: float=0.95,
                             n_bins: int=100,
                             dialogue_type: str='LLM',
                             labels: tp.Tuple[str, str]=('Real', 'Random sample')) -> tp.Any:
    '''
    Построение графика распределения частот и доверительного интервала средних косинусных близостей диалогов
    Parameters:
        real_cosines: Средние косинусные близости, полученные на исходном датасете
        rand_cosines: Средние косинусные близости, полученные на сэмплировании случайных фраз
        conf_real_cosines: Значения косинусных близостей из исходного датасета, попавшие в доверительный интервал
        conf_rand_cosines: Значения косинусных близостей сэмплированного датасета, попавшие в доверительный интервал
        quantile: Квантиль доверительного интервала
        n_bins: Число бинов 
        dialogue_type: Строка с типом диалога, для которого строится график
        labels: Названия датасетов
    Return:
        График распределения частот
    '''
    left_bound_real = conf_real_cosines[0]
    right_bound_real = conf_real_cosines[-1]
    left_bound_rand = conf_rand_cosines[0]
    right_bound_rand = conf_rand_cosines[-1]
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(f'Cosine similarity for {dialogue_type}', fontsize=16)
    ax.set_xlabel('Metric values', fontsize=14)
    ax.set_ylabel('% of evidence', fontsize=14)
    bins = np.linspace(min(min(real_cosines), min(rand_cosines)), 
                       max(max(real_cosines), max(rand_cosines)),
                       n_bins)
    y, _, _ = ax.hist(real_cosines, weights=np.ones_like(real_cosines)/len(real_cosines),
                      label=f'{labels[0]} cosine', bins=bins, alpha=0.5, color='blue')
    plot_conf_height = np.max(y) * 0.05
    ax.plot([left_bound_real, left_bound_real, right_bound_real, right_bound_real],
            [0, plot_conf_height, plot_conf_height, 0],
            color='red', linestyle='dashed',
            label=f'{labels[0]} {quantile} confidence interval')
    y, _, _ = ax.hist(rand_cosines, weights=np.ones_like(rand_cosines)/len(rand_cosines),
                      label=f'{labels[1]} cosine', bins=bins, alpha=0.5, color='orange')
    plot_conf_height = np.max(y) * 0.05
    ax.plot([left_bound_rand, left_bound_rand, right_bound_rand, right_bound_rand],
            [0, plot_conf_height, plot_conf_height, 0],
            color='blue', linestyle='dashed',
            label=f'{labels[1]} {quantile} confidence interval')
    ax.legend()
    return fig
        