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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from llm_val.sampler.Sampler import Sampler
from llm_val.sampler.DialogueSampler import DialogueSampler
from llm_val.utils.parse_dialogues import parse_html_dialogue

def report_valtest_generated_recognition_analysis(statistics: tp.Dict[str, tp.Dict[str, tp.Any]],
                                                  gini_fig: tp.Optional[tp.Any]=None,
                                                  prob_fig: tp.Optional[tp.Any]=None,
                                                  toloka_fig: tp.Optional[tp.Any]=None) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        statistics: Словарь с посчитанными фактическими Gini и квантилями для диалогов
            между человеком и LLM и диалогов между людьми
        gini_fig: Построенные графики с гистограммой плотности распределения Gini и
            функцией распределения метрики Gini
        prob_fig: Построенный график плотности распределения вероятностей того, относится ли фраза к человеку или к LLM
        toloka_fig: Построенный график плотности распределения вероятностей того, относится ли фраза к диалогу между людьми
            либо к диалогу между человеком и LLM
    Return: 
        report в соответствии с концепцией llm_val
    '''
    result = pd.DataFrame({'Проверка': ['Распознавание сгенерированных текстов', 
                                        'Сравнение диалога между человека с LLM и человека с человеком',
                                        'Сравнение фраз людей и LLM из RuATD'], 
                           'Данные': ['LLM', 'Toloka Persona Chat + LLM', 'RuATD'],
                           'Фактический gini': [statistics['human_llm']['gini_fact'], 
                                                statistics['human_human']['gini_fact'],
                                                np.round(0.45812000091286853, 5)],
                           'Квантиль': [statistics['human_human']['gini_quantile'],
                                        statistics['human_llm']['gini_quantile'],
                                        0.275]})
    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_dataframes'] = [result]
    res_dict['result_plots'] = [gini_fig, prob_fig, toloka_fig]
    return res_dict
 
    
def valtest_generated_recognition_analysis(sampler: Sampler,
                                           data_type: str='test', 
                                           vectorizer_min_df: float=0.005,
                                           vectorizer_max_df: float=0.99,
                                           n_iter: int=200,
                                           seed: int=42,
                                           toloka_size: float=1.0,
                                           test_size: float=0.4,
                                           statistics: tp.Optional[tp.Dict[str, tp.Any]]=None,
                                           **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест анализа распознавания сгенерированных текстов
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на котором проводится анализ
        vectorizer_min_df: Параметр min_df для TfidfVectorizer в эксперименте о случайности разбиения выборок
        vectorizer_max_df: Параметр max_df для TfidfVectorizer в эксперименте о случайности разбиения выборок
        n_iter: Число перемешиваний целевой переменной
        seed: Стартовое значение датчика случайной величины
        toloka_size: Доля датасета Toloka Persona Chat Rus, которая будет использована в тесте
        test_size: Доля тестовой выборки. NB: Результат для RuATD предпосчитан с использованием доли из дефолтного значения параметра 
        statistics: Опциональное значение статистик для диалогов между человека и LLM и диалогов между людьми.
            Если не None, то подсчет метрики не производится
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if statistics is None:
        llm_gini_stats, gini_fig, proba_fig = get_recognition_gini_stats(sampler,
                                                                         data_type, 
                                                                         vectorizer_min_df,
                                                                         vectorizer_max_df,
                                                                         n_iter,
                                                                         test_size,
                                                                         seed)
        toloka_gini_stats, toloka_gini_fig = get_toloka_gini_stats(sampler,
                                                                   data_type,
                                                                   vectorizer_min_df,
                                                                   vectorizer_max_df,
                                                                   n_iter,
                                                                   toloka_size,
                                                                   test_size,
                                                                   seed)
        statistics = {}
        statistics['human_llm'] = llm_gini_stats
        statistics['human_human'] = toloka_gini_stats
    report = report_valtest_generated_recognition_analysis(statistics, 
                                                           gini_fig, 
                                                           proba_fig, 
                                                           toloka_gini_fig)
    precomputed = statistics
    return {'report': report,
            'precomputed': precomputed}

    
def preprocess_texts(texts: pd.DataFrame,  
                     vectorizer_min_df: float=0.005, 
                     vectorizer_max_df: float=0.99) -> tp.Tuple[scipy.sparse.spmatrix, np.array]:
    '''
    Функция для очистки текстов, получения tf-idf матрицы текстов диалога, а также кодирования ролей участников диалога
    Parameters:
        texts: DataFrame с диалогами
        vectorizer_min_df: Параметр min_df для TfidfVectorizer в эксперименте о случайности разбиения выборок
        vectorizer_max_df: Параметр max_df для TfidfVectorizer в эксперименте о случайности разбиения выборок
    Return:
        Sparse-matrix с tf-idf векторизацией очищенных текстов и array с метками ролей 
    '''
    texts['content'] = texts['content'].apply(lambda x: x.lower())
    texts['content'] = texts['content'].apply(lambda x: re.sub(r'[^\w\s]','',x))
    X = texts['content']
    y = texts['role']
    vectorizer = TfidfVectorizer(min_df=vectorizer_min_df,
                                 max_df=vectorizer_max_df)
    encoder = LabelEncoder()
    X = vectorizer.fit_transform(X)
    y = encoder.fit_transform(y)
    return X, y


def get_gini_score(X: scipy.sparse.spmatrix, 
                   y: np.array, 
                   test_size: float=0.4,
                   seed: int=42) -> float:
    '''
    Подсчет Gini методом перекрестной проверки (кросс-валидации)
    Parameters:
        X: Tf-idf матрица текстов диалогов
        y: Вектор c метками ролей участников диалогов
        test_size: Доля тестовой выборки
        seed: Стартовое значение датчика случайной величины
    Return:
        Значение Gini, подсчитанное через кросс-валидацию
    '''
    model = LogisticRegression(max_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    model.fit(X_train, y_train)
    gini_score = 2 * roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) - 1
    return gini_score


def get_shuffle_gini_list(X: scipy.sparse.spmatrix, 
                          y: np.array, 
                          n_iter: int=200, 
                          test_size: float=0.4) -> tp.List[float]:
    '''
    Подсчет Gini на перемешанных выборках
    Parameters:
        X: Tf-idf матрица текстов диалогов
        y: Вектор с метками ролей участников диалогов
        n_iter: Число итераций (выборок)
        test_size: Объем тестовой выборки
    Return:
        Список метрик Gini для каждой выборки
    '''
    gini_scores = []
    for seed in tqdm(range(n_iter)):
        gini = get_gini_score(X, 
                              y, 
                              test_size, 
                              seed)
        gini_scores.append(gini)
    gini_scores = sorted(gini_scores)
    return gini_scores


def plot_gini_hist_dist(gini_fact: float,
                        gini_random_list: tp.List[float], 
                        quantile: float) -> tp.Any:
    '''
    Построение графика плотности и функции распределения Gini
    Parameters:
        gini_fact: Gini, посчитанный на исходном разбиении данных
        gini_random_list: Список Gini, посчитанных на рандомном разбиении
        quantile: Квантиль фактического Gini
    Return:
        График плотности и функции распределения Gini модели предсказания роли участника диалога
    '''
    fig, ax = plt.subplots(2, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.05)
    y_lim_min = -0.05
    y_lim_max = 1.05
    weights = np.ones(len(gini_random_list)) / len(gini_random_list)
    ax[0].grid(True, linestyle=':')
    y, _, _ = ax[0].hist(gini_random_list,
                         weights=weights,
                         bins=100,
                         alpha=0.5,
                         color='Orange')
    ax[0].set_ylim(0, y.max() * 1.05)
    ax[0].plot([gini_fact] * 2, [0, y.max() * 1.05], color='green', linestyle='--')
    ax[0].set_ylabel('Probability')
    ax[0].set_xticks([])
    ax[1].grid(True, linestyle=':')
    ax[1].set_ylim(y_lim_min, y_lim_max)
    ax[1].set_xlabel('Gini')
    ax[1].set_ylabel('Probability')
    ax[1].plot(gini_random_list, np.linspace(0, 1, len(gini_random_list)))
    ax[1].plot([gini_fact] * 2, [y_lim_min, y_lim_max], color='green', linestyle='--')
    ax[1].scatter(gini_fact, quantile, color='green', marker='x', s=100)
    return fig


def get_ai_probabilities(X: scipy.sparse.spmatrix, 
                         y: np.array,
                         test_size: float=0.4,
                         seed: int=42) -> tp.Dict['str', tp.List[tp.Any]]:
    '''
    Функция для нахождения оценки вероятностей того, к какому классу относится диалог
    Parameters:
        X: Tf-idf матрица фраз из диалогов
        y: Вектор с истинными метками
    Return:
        Словарь, в котором хранятся вероятности и истинные метки класса
    '''
    model = LogisticRegression(max_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    probas = {'proba': y_proba,
              'labels': y_test} 
    return probas


def plot_density_of_proba(probabilities: tp.Dict[str, tp.Any]) -> tp.Any:
    '''
    Функция для построения графика плотности вероятности того, является ли фраза сгенерирована ИИ
    Parameters:
        probabilities: Словарь с вероятностями и истинными метками класса
    Return:
        График плотности вероятности того, что фраза сгенерирована ИИ
    '''
    fig, ax = plt.subplots(figsize = (8, 8))
    probas = probabilities['proba']
    labels = probabilities['labels']
    prob_df = pd.DataFrame({'prob': probas, 'label': labels})
    human_preds = prob_df[prob_df['label'] == 0]['prob']
    llm_preds = prob_df[prob_df['label'] == 1]['prob']
    sns.histplot(human_preds, kde=True, alpha=0.5, color='#8DB4E3', stat='proportion',
                 label=f'Human phrases, mean prob - {np.round(np.mean(human_preds), 2)}', bins=20)
    sns.histplot(llm_preds, kde=True, alpha=0.5, color='#FF7F50', stat='proportion',
                 label=f'LLM phrases, mean prob - {np.round(np.mean(llm_preds), 2)}', bins=20)
    ax.set_title('Distribution of probabilities "Is phrase generated by AI?"')
    ax.set_xlabel('Probability')
    ax.legend()
    return fig


def get_recognition_gini_stats(sampler: Sampler,
                               data_type: str='test',
                               vectorizer_min_df: float=0.005,
                               vectorizer_max_df: float=0.99,
                               n_iter: int=200,
                               test_size: float=0.4,
                               seed: int=42) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Any]:
    '''
    Функция для получения статистик теста на распознавание сгенерированных текстов и графика Gini
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на котором проводится анализ
        vectorizer_min_df: Параметр min_df для TfidfVectorizer в эксперименте о случайности разбиения выборок
        vectorizer_max_df: Параметр max_df для TfidfVectorizer в эксперименте о случайности разбиения выборок
        n_iter: Число перемешиваний целевой переменной
        test_size: Доля тестовой выборки
        seed: Стартовое значение датчика случайной величины
    Return:
        Словарь с посчитанными статистиками и график плотности и функции распределения Gini
    '''
    texts = getattr(sampler, data_type)['X']
    labels = getattr(sampler, data_type)['y']
    texts_df = pd.merge(texts, labels)
    X, y = preprocess_texts(texts_df, 
                            vectorizer_min_df, 
                            vectorizer_max_df)
    res = {}
    res['gini_fact'] = get_gini_score(X, 
                                      y, 
                                      test_size, 
                                      seed)
    res['gini_shuffle_list'] = get_shuffle_gini_list(X, y, n_iter, test_size)
    res['gini_quantile'] = sum(res['gini_shuffle_list'] <= res['gini_fact']) / len(res['gini_shuffle_list'])
    gini_plot = plot_gini_hist_dist(res['gini_fact'], res['gini_shuffle_list'], res['gini_quantile'])
    probabilities = get_ai_probabilities(X, 
                                         y, 
                                         test_size, 
                                         seed)
    prob_plot = plot_density_of_proba(probabilities)
    return res, gini_plot, prob_plot


def create_toloka_df(toloka_size: float=1.0,
                     seed: int=42) -> pd.DataFrame:
    '''
    Функция, которая получает диалоги Toloka, парсит их и создаёт DialogueSampler
    Parameters:
        toloka_size: Объем датасета Toloka Persona Chat Rus, которая будет использоваться для теста
        seed: Случайное значение датчика случайной величины
    Return:
        DialogueSampler с датасетом Toloka Persona Chat Rus
    '''
    toloka_path = os.environ['TOLOKA_PATH']
    toloka_df = pd.read_pickle(toloka_path)
    if toloka_size < 1:
        toloka_df = toloka_df.sample(frac=toloka_size, random_state=seed)
    toloka_dialogues = toloka_df['dialogue'].to_list()
    toloka_dialogues = [parse_html_dialogue(dial) for dial in toloka_dialogues]
    phrases = []
    for dialogue in toloka_dialogues:
        for phrase in dialogue:
            phrases.append(phrase)
    res_df = pd.DataFrame()
    res_df['content'] = phrases
    res_df['label'] = 'human_human'
    return res_df
    
    
def get_toloka_gini_stats(sampler: Sampler,
                          data_type: str='test', 
                          vectorizer_min_df: float=0.005,                    
                          vectorizer_max_df: float=0.99,
                          n_iter: int=200,
                          toloka_size: float=1.0,
                          test_size: float=0.4,
                          seed: int=42):
    '''
    Функция для получения статистик теста на различие диалогов между человеком и человеком и 
        диалогов между человеком и LLM
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на котором проводится анализ
        vectorizer_min_df: Параметр min_df для TfidfVectorizer в эксперименте о случайности разбиения выборок
        vectorizer_max_df: Параметр max_df для TfidfVectorizer в эксперименте о случайности разбиения выборок
        n_iter: Число перемешиваний целевой переменной
        toloka_size: Доля датасета Toloka Persona Chat Rus, которая будет использоваться для теста
        test_size: Доля тестовой выборки
        seed: Стартовое значение датчика случайной величины
    Return:
        Словарь с посчитанными статистиками
    '''
    toloka_df = create_toloka_df(toloka_size, seed)
    user_texts = getattr(sampler, data_type)['X']
    user_labels = getattr(sampler, data_type)['y']
    user_texts = user_texts.merge(user_labels)
    user_df = pd.DataFrame()
    user_df['content'] = user_texts[user_texts['role'] == 'user']['content']
    user_df['role'] = 'human_llm'
    full_texts = pd.concat([user_df, toloka_df])
    X, y = preprocess_texts(full_texts, 
                            vectorizer_min_df,
                            vectorizer_max_df)
    res = {}
    res['gini_fact'] = get_gini_score(X, y, test_size, seed)
    res['gini_shuffle_list'] = get_shuffle_gini_list(X, y, n_iter, test_size)
    res['gini_quantile'] = sum(res['gini_shuffle_list'] <= res['gini_fact']) / len(res['gini_shuffle_list'])
    probs = get_ai_probabilities(X, 
                                 y, 
                                 test_size, 
                                 seed)
    prob_plot = plot_density_of_proba(probs)
    return res, prob_plot
