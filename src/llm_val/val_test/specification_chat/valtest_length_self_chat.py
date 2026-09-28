import os
import time
import numpy as np
import typing as tp
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from urllib.error import HTTPError

from llm_val.utils.parse_dialogues import parse_html_dialogue

def report_valtest_length_self_chat(computed: tp.Dict[str, tp.Any],
                                    unique_dist_plot: tp.Any,
                                    cycle_start_plot: tp.Any,
                                    size: int) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        computed: Словарь с посчитанным количеством уникальных и повторяющихся фраз для каждого диалога, 
            а также с номерами фраз, с которых начинается зацикливание
        unique_dist_plot: Гистограмма распределения числа уникальных значений
        cycle_start_plot: Гистограмма распределения номеров предложений, с которых началось зацикливание
        size: Размер сгенерированных диалогов
    Return:
        report в соответствии с концепцией llm_val
    '''
    total = len(computed['count_unique'])
    repeats_cnt = sum(computed['count_repeat'])
    cycles_cnt = len(computed['cycle_starts'])
    result = pd.DataFrame({'Общее количество диалогов': [total],
                           'Диалоги с зацикливанием': [cycles_cnt],
                           'Общее число фраз в диалогах': [size * total],
                           'Средняя длина диалога': [size - repeats_cnt / total],
                           'Среднее количество повторений': [repeats_cnt / total]})
    report = {}
    report['semaphore'] = 'gray'
    report['result_dataframes'] = [result]
    report['result_plots'] = [unique_dist_plot, cycle_start_plot]
    return report


def valtest_length_self_chat(model: tp.Any,
                             generated_length: int=15,
                             context: int=3,
                             n_samples: int=1000,
                             seed: int=42,
                             time_step: tp.Optional[int]=None,
                             computed: tp.Optional[tp.Dict[str, tp.Any]]=None,
                             **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест оценка продолжительности самодиалога (Информативный тест)
    Parameters:
        model: Обёртка для модели, у которой есть метод predict
        generated_length: Длина генерируемого диалога
        context: Количество предложений из датасета Toloka Persona Chat Rus, которые берутся для затравки
        n_samples: Количество сгенерированных диалогов
        seed: Датчик случайной величины
        time_step: Опциональный параметр. Время между запросами к модели. Если None, то задержки не будет
        computed: Опциональное значение посчитанных количеств уникальных фраз, повторяющихся фраз и позиций, 
            с которых начинаются зацикливания. Если не None, то подсчет метрик не производится
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if computed is None:
        computed = get_generated_dialogues_info(model,
                                                generated_length,
                                                context,
                                                n_samples,
                                                seed,
                                                time_step)
    unique_dist_plot = plot_unique_hist(computed['count_unique'])
    cycle_start_plot = plot_cycle_start(computed['cycle_starts'])
    report = report_valtest_length_self_chat(computed,
                                             unique_dist_plot,
                                             cycle_start_plot,
                                             (generated_length+context))
    res = {}
    res['report'] = report
    res['precomputed'] = computed
    return res


def get_generated_dialogues_info(model: tp.Any,
                                 generated_length: int=15,
                                 context: int=3,
                                 n_samples: int=1000,
                                 seed: int=42,
                                 time_step: tp.Optional[int]=None) -> tp.Dict[str, tp.Any]:
    '''
    Функция для нахождения количества уникальных и повторяющихся предложений в диалогах, а также номеров предложений, 
        с которых началось зацикливание
    Parameters:
        model: Обёртка для модели, у которой есть метод predict
        generated_length: Длина генерируемого диалога
        context: Количество предложений из датасета Toloka Persona Chat Rus, которые берутся для затравки
        n_samples: Количество диалогов, которое будет сгенерировано
        seed: Начальное значение датчика случайной величины
        time_step: Опциональный параметр. Время между запросами к модели. Если None, то задержки не будет
    Return:
        Словарь с посчитанными количествами уникальных и повторяющихся предложений в диалогах, 
            а также номерами предложений, с которых началось зацикливание
    '''
    dialogues = generate_dialogues(model, 
                                   generated_length, 
                                   context, 
                                   n_samples, 
                                   seed,
                                   time_step)
    parsed_dialogues = [parse_dialogue(x) for x in dialogues]
    unique = []
    repeats = []
    cycle_starts = []
    for dialogue in parsed_dialogues:
        generated = dialogue[context:generated_length+context]
        cnt = Counter(generated)
        unique_cnt = len(cnt)
        repeats_cnt = sum(list(cnt.values())) - len(cnt)
        unique.append(unique_cnt)
        repeats.append(repeats_cnt)
        cycle_start = calc_cycle(generated)
        if cycle_start != -1:
            cycle_starts.append(cycle_start)  
    computed = {}
    computed['count_unique'] = unique
    computed['count_repeat'] = repeats
    computed['cycle_starts'] = cycle_starts
    return computed


def calc_cycle(dialogue: tp.List[str]) -> int:
    '''
    Функция для нахождения позиций, с которых начинается зацикливание в диалоге
    Parameters:
        dialogue: Список с фразами диалогов
    Return:
        Номер фразы в диалоге, с которой начинается зацикливание
    '''
    phrases_cnt = Counter(dialogue)
    cycle_start = -1
    for i, phrase in enumerate(dialogue):
        if phrases_cnt[phrase] > 1:
            cycle_start = i+1
            break
    return cycle_start       


def concat_phrases(dialogue: tp.List[str]) -> str:
    '''
    Функция для конкатенации диалогов в промпт для LLM
    Parameters:
        dialogue: Список строк с фразами из диалогов
    Return:
        Строка-промпт для LLM
    '''
    prompt = ''
    for phrase in dialogue:
        prompt += '-' + phrase + '\n '
    return prompt


def generate_dialogues(model: tp.Any,
                       generated_length: int=15,
                       context: int=3,
                       n_samples: int=1000,
                       seed: int=42,
                       time_step: tp.Optional[int]=None) -> tp.List[str]:
    '''
    Функция для генерации диалогов из затравок человеческих диалогов из датасета Toloka Persona Chat Rus
    Parameters:
        model: Обёртка для модели, у которой есть метод predict
        generated_length: Длина генерируемого диалога
        context: Количество предложений из датасета Toloka Persona Chat Rus, которые берутся для затравки
        n_samples: Количество сгенерированных диалогов
        seed: Начальное значение датчика случайной величины
        time_step: Опциональный параметр. Время между запросами к модели. Если None, то задержки не будет
    Return: 
        Список со сгенерированными диалогами
    '''
    toloka_df = pd.read_pickle(os.environ['TOLOKA_PATH'])
    toloka_dialogues = toloka_df.sample(n=n_samples, random_state=seed)['dialogue'].to_list()
    toloka_dialogues = [parse_html_dialogue(x) for x in toloka_dialogues]
    seed_phrases = [dialogue[:context] for dialogue in toloka_dialogues]
    seed_phrases = [concat_phrases(x) for x in seed_phrases]
    for _ in range(generated_length):
        try:
            prediction = model.predict(seed_phrases)
        except HTTPError as err:
            if not time_step is None:
                time.sleep(time_step)
            prediction = model.predict(seed_phrases)
        for i, prompt in enumerate(seed_phrases):
            prompt += prediction[i] + '\n' 
            seed_phrases[i] = prompt
    return seed_phrases


def parse_dialogue(dialogue: str) -> tp.List[str]:
    '''
    Функция, которая убирает лишние знаки пунктуации и парсит сгенерированный диалог
    Parameters:
        dialogue: Строка со сгенерированным диалогом
    Return:
        Список фраз диалогов
    '''
    dialogue = dialogue.replace(' -', '').replace('-', '').split('\n')
    phrases = [phrase.lstrip(' ') for phrase in dialogue[:-1]]
    return phrases


def plot_unique_hist(unique: tp.List[int],
                     generated_length: int=15) -> tp.Any:
    '''
    Функция для построения гистограммы распределения числа уникальных фраз в диалогах
    Parameters:
        unique: Список с количеством уникальных фраз для каждого диалога
        generated_length: Длина генерируемого диалога
    Return:
        Гистограмма распределения числа уникальных фраз в диалогах
    '''
    unique = sorted(unique)
    fig, ax = plt.subplots(figsize=(8, 8))
    bins = range(1, generated_length+2)
    y = sns.histplot(unique, bins=bins, stat='count', linestyle='dashed', color='#8DB4E3')
    labels_rep = [str(v) if v else '' for v in y.containers[0].datavalues]
    mids = [rect.get_x() + rect.get_width() / 2 for rect in y.patches]
    y.set_xticks(mids, labels=range(1, len(mids)+1))
    y.bar_label(y.containers[0], labels=labels_rep)
    ax.set_ylabel('Количество диалогов')
    ax.set_xlabel('Количество уникальных фраз в диалоге')
    ax.set_title('Распределение количества дубликатов внутри одного диалога')
    return fig


def plot_cycle_start(cycle_starts: tp.List[int],
                     generated_length: int=15) -> tp.Any:
    '''
    Функция для посторения гистограммы распределения номеров фраз, с которых начинались зацикливания в диалогах
    Parameters:
        cycle_starts: Список с номерами фраз, с которых начинались зацикливания
        generated_length: Длина генерируемого диалога
    Return:
        Гистограмма распределения номеров предложений, с которых начинаются зацикливания
    '''
    cycle_starts = sorted(cycle_starts)
    fig, ax = plt.subplots(figsize=(8, 8))
    bins = range(1, generated_length+1)
    y = sns.histplot(cycle_starts, bins=bins, stat='count', linestyle='dashed', element='bars', color='#8DB4E3')
    labels_lock = [str(v) if v else '' for v in y.containers[0].datavalues]
    y.bar_label(y.containers[0], labels=labels_lock)
    mids = [rect.get_x() + rect.get_width() / 2 for rect in y.patches]
    y.set_xticks(mids, labels=range(1, len(mids)+1))
    ax.set_xlabel('Порядковый номер фразы, после которой произошло зацикливание в диалоге')
    return fig