import pymorphy3
import collections
import numpy as np
import typing as tp
import pandas as pd
from scipy import stats
from itertools import chain
import matplotlib.pyplot as plt

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold, semaphore_by_threshold_without_red
                                         

def report_valtest_statistics_dynamics(df_statistics: pd.DataFrame,
                                       figs: tp.List[tp.Any],
                                       semaphore_threshold: tp.Tuple[float, float]=(0.5, 1.0),
                                       semaphore_threshold_new_tokens: tp.Tuple[float, float]=(0.003, 0.005),
                                       red_semaphores_answers_threshold: tp.Tuple[int, int]=(1, 3),
                                       yellow_semaphores_answers_threshold: int=3,
                                       yellow_semaphores_queries_threshold: int=3,
                                       greater_is_better: bool=False,
                                       is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        df_statistics: pd.DataFrame с подсчитанными статистиками
        figs: Список графиков
        semaphore_threshold: Желтый и красный порог светофора по PSI для предсказаний ответов, стоп-слов, числа и частот токенов
        semaphore_threshold_new_tokens: Желтый и красный порог светофора по относительному увеличению числа новых токенов 
        red_semaphores_answers_threshold: Желтый и красный порог итогового светофора по числу красных светофоров для статистик ответов 
        yellow_semaphores_answers_threshold: Желтый порог итогового светофора по числу желтых светофоров для статистик ответов
        yellow_semaphores_queries_threshold: Желтый порог итогового светофора по числу красных светофоров для статистик запросов
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: Если True, то тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    semaphores = pd.DataFrame(columns=df_statistics.index)
    df_stats = df_statistics.loc[:, ~df_statistics.columns.str.startswith('Корреляция')]
    color = 'green'
    col_names = ['Частота токенов', 'Число токенов', 'PSI предсказаний ответов LLM']
    indexes = set(df_stats.index) & set(col_names)
    if df_stats.shape[1] > 1:
        for idx in indexes:
            semaphores[idx] = [semaphore_by_threshold(val,
                                                      semaphore_threshold,
                                                      greater_is_better,
                                                      False,
                                                      False)
                                                      if not pd.isnull(val) else 'green'
                                                      for val in df_stats.loc[idx, :]]
        semaphores['Доля стоп-слов'] = [semaphore_by_threshold_without_red(val,
                                                                           semaphore_threshold[0],
                                                                           greater_is_better,
                                                                           False)
                                                                           if not pd.isnull(val) else 'green'
                                                                           for val in df_stats.loc['Доля стоп-слов', :]]
        semaphores['Доля новых токенов'] = [semaphore_by_threshold(val,
                                                                   semaphore_threshold_new_tokens,
                                                                   greater_is_better,
                                                                   False,
                                                                   False)
                                                                   if not pd.isnull(val) else 'green'
                                                                   for val in df_stats.loc['Доля новых токенов', :]]
        red_semaphores_queries = (semaphores.iloc[0, :] == 'red').sum()
        red_semaphores_answers = (semaphores.iloc[-1, :] == 'red').sum()
        yellow_semaphores_answers = (semaphores.iloc[-1, :] == 'yellow').sum()
        if red_semaphores_answers >= red_semaphores_answers_threshold[1]:
            color = 'red'
        elif red_semaphores_answers >= red_semaphores_answers_threshold[0] \
            or yellow_semaphores_answers >= yellow_semaphores_answers_threshold \
            or red_semaphores_queries >= yellow_semaphores_queries_threshold:
            color = 'yellow'
        else:
            color = 'green'
    indexes = list(set(semaphores.columns) & set(['Доля новых токенов', 'Доля стоп-слов'] + col_names))
    df_statistics.loc[indexes, 'Светофор по запросам'] = semaphores.loc[0, indexes].values
    df_statistics.loc[indexes, 'Светофор по ответам'] = semaphores.loc[:, indexes].iloc[-1, :].values
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else color
    res_dict['result_plots'] = figs
    res_dict['result_dataframes'] = [df_statistics]
    return res_dict


def valtest_statistics_dynamics(sampler: Sampler,
                                tokenizer: tp.Any,
                                stop_words_list: tp.List[str],
                                data_types: tp.List[str]=['train', 'test'],
                                is_discrete: bool=False,
                                semaphore_threshold: tp.Tuple[float, float]=(0.5, 1.0),
                                semaphore_threshold_new_tokens: tp.Tuple[float, float]=(0.003, 0.005),
                                red_semaphores_answers_threshold: tp.Tuple[int, int]=(1, 3),
                                yellow_semaphores_answers_threshold: int=3,
                                yellow_semaphores_queries_threshold: int=3,
                                greater_is_better: bool=False,
                                is_info: bool=False,
                                df_statistics: tp.Optional[tp.Dict[str, float]]=None,
                                **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на анализ динамики стандартных статистик для запроса и ответа LLM
    Parameters:
        sampler: Объект класса Sampler
        tokenizer: Модель-токенизатор с методом tokenize, возвращающим словарь с ключом 'input_ids'
        stop_words_list: Список стоп-слов
        data_types: Типы данных, на которых сравниваются метрики (в хронологическом порядке формирования выборок)
        is_discrete: Булевский флаг, если True, то cчитается, что ответы модели находятся в дискретной области,
            и проводится анализ динамики распределения предсказаний моделей во времени по PSI
        semaphore_threshold: Желтый и красный порог светофора по PSI для предсказаний ответов, стоп-слов, числа и частот токенов
        semaphore_threshold_new_tokens: Желтый и красный порог светофора по относительному увеличению числа новых токенов 
        red_semaphores_answers_threshold: Желтый и красный порог итогового светофора по числу красных светофоров для статистик ответов 
        yellow_semaphores_answers_threshold: Желтый порог итогового светофора по числу желтых светофоров для статистик ответов
        yellow_semaphores_queries_threshold: Желтый порог итогового светофора по числу красных светофоров для статистик запросов
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
        df_statistics: Опциональный pd.DataFrame со значениями PSI статистик на ответах и запросах
            Если не None, то подсчет метрики не производится и на ее основе выставляется светофор
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if df_statistics is None:
        data_train = getattr(sampler, data_types[0])
        train_query, train_answer = data_train['X']['question'].to_list(), data_train['X']['model_answer'].to_list()
        stop_train_a = count_stop_words(train_answer, stop_words_list)            
        train_answer_tokens_dict = get_token_dict(train_answer, tokenizer)
        count_train_q = get_tokens_count(train_query, tokenizer)
        count_train_a = get_tokens_count(train_answer, tokenizer)
        df_statistics = pd.DataFrame()
        if not is_discrete and 'y' in data_train:
            quality_metrics = data_train['y'].T.values.reshape(-1)
            corr_res = [stats.spearmanr(stat, quality_metrics).statistic for stat in [stop_train_a, count_train_a]]
            df_statistics = pd.DataFrame([{'Доля стоп-слов': corr_res[0], 'Число токенов': corr_res[1]}], index=[f'Корреляция с ключевой метрикой качества на {data_types[0]}'])
        figs = []
        if len(data_types) > 1:
            stop_train_q = count_stop_words(train_query, stop_words_list)
            train_query_tokens_dict = get_token_dict(train_query, tokenizer)
            for sampler_idx in range(1, len(data_types)):
                cur_dt = data_types[sampler_idx]
                data_test = getattr(sampler, cur_dt)
                test_query, test_answer = data_test['X']['question'].to_list(), data_test['X']['model_answer'].to_list()
                stop_test_q, stop_test_a = count_stop_words(test_query, stop_words_list), count_stop_words(test_answer, stop_words_list)
                stop_q = get_psi_statistic(stop_train_q, stop_test_q)
                stop_a = get_psi_statistic(stop_train_a, stop_test_a)
                count_test_a = get_tokens_count(test_answer, tokenizer)
                freq_train_q, freq_test_q, _ = get_token_freq(train_query_tokens_dict, get_token_dict(test_query, tokenizer))
                freq_train_a, freq_test_a, new_tok_a = get_token_freq(train_answer_tokens_dict, get_token_dict(test_answer, tokenizer))
                freq_q = psi(freq_train_q, freq_test_q)
                freq_a = psi(freq_train_a, freq_test_a)
                count_q = get_psi_statistic(count_train_q, get_tokens_count(test_query, tokenizer))
                count_a = get_psi_statistic(count_train_a, count_test_a)
                df_statistics_upd = pd.DataFrame({'Доля стоп-слов': [stop_q, stop_a],
                                                  'Частота токенов': [freq_q, freq_a],
                                                  'Число токенов': [count_q, count_a],
                                                  'Доля новых токенов': [None, new_tok_a]},
                                                 index=[f'PSI запросов на {cur_dt}', f'PSI ответов на {cur_dt}'])
                if is_discrete:
                    train_ans_dict, test_ans_dict = collections.Counter(train_answer), collections.Counter(test_answer)
                    train_ans_freq, test_ans_freq, _ = get_token_freq(train_ans_dict, test_ans_dict)
                    df_statistics_upd['PSI предсказаний ответов LLM'] = [None, psi(train_ans_freq, test_ans_freq)]
                elif 'y' in data_test:
                    quality_metrics = data_test['y'].T.values.reshape(-1)
                    corr_res = [stats.spearmanr(stat, quality_metrics).statistic for stat in [stop_test_a, count_test_a]]
                    df_statistics_upd.loc[f'Корреляция с ключевой метрикой качества на {cur_dt}'] = [corr_res[0], None, corr_res[1], None]
                df_statistics = pd.concat((df_statistics, df_statistics_upd))
            df_statistics = df_statistics.T
            if len(data_types) > 2:
                df_fig = df_statistics
                if not is_discrete:
                    df_fig = df_statistics.drop(df_statistics.columns[list(range(0, df_statistics.shape[1], 3))], axis=1)
                figs = plot_statistics_dynamics(df_fig, is_discrete)
    precomputed = {'df_statistics': df_statistics.copy(deep=True)}
    report = report_valtest_statistics_dynamics(df_statistics,
                                                figs,
                                                semaphore_threshold,
                                                semaphore_threshold_new_tokens,
                                                red_semaphores_answers_threshold,
                                                yellow_semaphores_answers_threshold,
                                                yellow_semaphores_queries_threshold,
                                                greater_is_better,
                                                is_info)
    return {'report': report,
            'precomputed': precomputed}


def get_psi_statistic(train: tp.Iterable[float],
                      test: tp.Iterable[float]) -> float:
    '''
    Выполняет биннинг входных массивов и считает PSI
    Parameters:
        train: Первая выборка
        test: Вторая выборка
    Return:
        Значение PSI
    '''
    col_name = 'data'
    train = pd.DataFrame({col_name: train})
    test = pd.DataFrame({col_name: test})
    num_bins = int(np.sqrt(len(train)) + 1)
    train['bin'], bins = pd.qcut(train[col_name], num_bins, duplicates='drop', retbins=True)
    return psi(np.histogram(train[col_name], bins=bins)[0], np.histogram(test[col_name], bins=bins)[0])


def psi(train: tp.Iterable[float],
        test: tp.Iterable[float]) -> float:
    '''
    Считает PSI 
    Parameters:
        train: Первая выборка
        test: Вторая выборка
    Return:
        Значение PSI
    '''
    EPS = 1e-14
    train = np.array(train)
    test = np.array(test)
    train = (train + EPS) / np.sum(train + EPS)
    test = (test + EPS) / np.sum(test + EPS)
    return np.sum((train - test) * np.log(train / test))


def lemmatize(text: tp.List[str]) -> tp.List[str]:
    '''
    Выполняет операцию лемматизации для списка слов
    Parameters:
        text: Список слов
    Return:
        Список слов в нормальной форме
    '''
    morph = pymorphy3.MorphAnalyzer()
    return [morph.parse(word)[0].normal_form for word in text.split()]


def count_stop_words(sentences: tp.List[str],
                     stop_words: tp.List[str]) -> tp.List[float]:
    '''
    Считает долю стоп-слов в каждом предложении из списка
    Parameters:
        sentences: Список предложений
        stop_words: Список стоп-слов
    Return:
        Список значений доли стоп-слов в каждом из предложений 
    '''
    stop_words_freq  = []
    for sentence in sentences:
        lem_sentence = lemmatize(sentence)
        words_count = len(lem_sentence)
        stop_words_count = len([word for word in lem_sentence if word in stop_words])
        stop_words_freq.append(stop_words_count / words_count)
    return stop_words_freq


def get_tokens_count(sentences: tp.List[str],
                     tokenizer: tp.Any) -> tp.List[int]:
    '''
    Подсчитывает число токенов в каждом предложении
    Parameters:
        sentences: Список предложений
        tokenizer: Модель-токенизатор с методом tokenize, возвращающим словарь с ключом 'input_ids'
    Return:
        Список числа токенов в каждом предложении
    '''
    return [len(tokenizer.tokenize(sentence)['input_ids']) for sentence in sentences] 


def get_token_dict(sentences: tp.List[str],
                   tokenizer: tp.Any) -> tp.Dict[int, int]:
    '''
    Считает число вхождений токенов в список строк
    Parameters:
        sentences: Список строк
        tokenizer: Модель-токенизатор с методом tokenize, возвращающим словарь с ключом 'input_ids'
    Return:
        Словарь пар {Номер токена: Число вхождений во все строки}
    '''
    sent_list = [list(tokenizer.tokenize(sent)['input_ids'].numpy()[:, 1]) for sent in sentences]
    return collections.Counter(list(chain.from_iterable(sent_list)))

     
def get_token_freq(tokens_dict_train: tp.Dict[int, int],
                   tokens_dict_test: tp.Dict[int, int]) -> tp.Tuple[float, float, float]:
    '''
    Считает доли каждого токена и долю новых на test по словарям встречаемости
    Parameters:
        tokens_dict_train: Список строк в обучающем наборе
        tokens_dict_test: Список строк в тестовом наборе 
    Return:
        Частота каждого токена в обучающем и тестовом наборе строк, доля новых токенов в тестовой выборке относительно обучающей
    '''
    unite = set(tokens_dict_train.keys()) | set(tokens_dict_test.keys())
    train_token_freq, test_token_freq = [], []
    for key in list(unite):
        train_token_freq.append(tokens_dict_train[key])
        test_token_freq.append(tokens_dict_test[key])
    train_token_freq, test_token_freq = [np.array(tf) / np.sum(tf) for tf in [train_token_freq, test_token_freq]]
    except_keys = list(set(tokens_dict_test.keys()) - set(tokens_dict_train.keys()))
    return train_token_freq, test_token_freq, len(except_keys) / len(tokens_dict_test)


def plot_statistics_dynamics(df_statistics: pd.DataFrame,
                             is_discrete: bool=False) -> tp.List[tp.Any]:
    '''
    Строит графики динамики посчитанных статистик на OOS и нескольких OOT
    Parameters:
        df_statistics: pd.DataFrame с подсчитанными статистиками
        is_discrete: Булевский флаг, если True, то cчитается, что ответы модели находятся в дискретной области,
            и строится только график для PSI предсказаний
    Return:
        График динамики посчитанных статистик на OOS и нескольких OOT
    '''
    indexes = df_statistics.index
    if is_discrete:
        indexes = ['PSI предсказаний ответов LLM']
        fig, axs = plt.subplots()
        ax_indexes = [axs]
    else:
        fig, axs = plt.subplots(len(indexes) // 2, 2, figsize=(8, len(indexes)*2))
        ax_indexes = [axs[idx % 2, idx // 2] for idx in range(len(indexes))]  
        fig.tight_layout(pad=2)
    for ax, statistic in zip(ax_indexes, indexes):
        if not df_statistics.loc[statistic, :].iloc[::2].isnull().all():
            ax.plot(df_statistics.loc[statistic, :].iloc[::2].values, label='Запросы', c='blue')
        ax.plot(df_statistics.loc[statistic, :].iloc[1::2].values, label='Ответы', c='orange')
        ax.set_title(statistic, fontsize=12)
        ax.grid(True)
        ax.set_ylabel(statistic, fontsize=10)
        ax.set_xticks(np.arange((df_statistics.shape[1]) // 2))
        ax.set_xticklabels(map(lambda x: x.split('на ')[1], df_statistics.loc[statistic, :].iloc[1::2].index))
        ax.legend()
    return fig
