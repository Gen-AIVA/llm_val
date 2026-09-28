import numpy as np
import pandas as pd
import typing as tp
import scipy.stats as stats
import matplotlib.pyplot as plt
from numpy.random import Generator, default_rng

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import frequency_semaphore, worst_semaphore


def report_valtest_uniform_distribution_mmlu(figs: tp.List[plt.Figure],
                                             distributions: tp.List[pd.DataFrame],
                                             all_letters: tp.List[str],
                                             semaphore_threshold: tp.Tuple[float, float]=(3.,5.),
                                             is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        figs: Список графиков текущего распределения верных ответов и перемешанного, если было произведено перемешивание
        distributions: Список распределений верных ответов и перемешанных ответов, если было произведено перемешивание
        all_letters: Список всех букв, представленных как варинты ответов в answer_options
        semaphore_threshold: Кортеж с границами светофора. Границы в кортеже указываются в порядке возрастания
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_dict = {}
    real_distribution = distributions[0]
    semaphore_list = frequency_semaphore(values=list(real_distribution['Количество ответов'].values),
                                         threshold=semaphore_threshold)
    color = worst_semaphore(semaphore_list)
    res_dict['semaphore'] = 'gray' if is_info else color
    res_dict['result_plots'] = figs
    res_dict['result_dataframes'] = distributions
    return res_dict


def valtest_uniform_distribution_mmlu(sampler: Sampler,
                                      data_type: str='test',
                                      semaphore_threshold: tp.Tuple[float, float]=(3.,5.),
                                      alpha: float=0.05,
                                      seed: int=42,
                                      is_info: bool=False,
                                      **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на равномерное распределение правильных меток true_answer.
    Если распределение статистически отличается от равномерного, происходит
    перемешивание ответов. Перемешанный датасет со всеми необходимыми колонками
    хранится в precomputed['permutate_data_mmlu']. Также словарь precomputed
    хранит график распределения и стат значения нового перемешенного датасета.
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold: Кортеж с границами светофора. Границы в кортеже указываются в порядке возрастания
        alpha: Выбранный уровень значимости для теста на равномерное распределение
        seed: Сид для перемешивания ответов, если оно необходимо
        is_info: True, если тест информативный
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    X = getattr(sampler, data_type)['X']
    all_letters = get_all_letters(answer_options=list(X['answer_options'].values))
    distribution, fig, chi_2, p_value = get_test_info(true_answers=X['true_answer'], 
                                                      all_letters=all_letters, 
                                                      perm_flag=False)
    precomputed = {'stats': {'chi': chi_2, 'p_value': p_value}, 
                   'distribution_fig': fig}
    distributions, figs = [distribution], [fig]
    if p_value < alpha:
        rng = default_rng(seed)
        permutate_mmlu = get_permutate_data(X, rng)
        permutate_distribution, permutate_fig, chi_2_perm, p_value_perm = get_test_info(true_answers=permutate_mmlu['true_answer'],
                                                                                        all_letters=all_letters,
                                                                                        perm_flag=True)
        figs.append(permutate_fig)
        distributions.append(permutate_distribution)
        precomputed.update({'permutate_data_mmlu': permutate_mmlu,
                            'permutate_stats': {'chi_perm': chi_2_perm, 'p_value_perm': p_value_perm},
                            'permutate_distribution_fig': permutate_fig})

    report = report_valtest_uniform_distribution_mmlu(figs=figs, 
                                                      distributions=distributions,
                                                      all_letters=all_letters,
                                                      semaphore_threshold=semaphore_threshold,
                                                      is_info=is_info)
    return {'report': report,
            'precomputed': precomputed}

def get_test_info(true_answers: pd.Series, 
                  all_letters: tp.List[str], 
                  perm_flag: bool=False) -> tp.Tuple[tp.Dict[str, float], tp.Any, float, float]:
    '''
    Возвращает распределение, график распределения, статистику и p_value для теста Хи-квадрат на равномерное распределение ответов.
    Parameters:
        true_answers: Колонка верных ответов
        all_letters: Список всех букв из пула ответов answer_options
        perm_flag: Флаг, определяющий title для графика распределения ответов в зависимости от того, было перемешивание или нет
    Return:
        Словарь распределения меток классов, круговая диаграмма распределения, значения p_value и статистики Хи квадрат теста.
    '''
    distribution = get_distribution_frame(true_answers=true_answers, 
                                          all_letters=all_letters)
    distribution_dict = get_distribution_dict(distribution=distribution)
    fig = plot_distribution_mmlu(distribution=distribution_dict, 
                                 perm_flag=perm_flag)
    chi_2, p_value = chi_2_stats(distribution=distribution_dict)
    return distribution, fig, chi_2, p_value


def get_distribution_frame(true_answers: pd.Series, 
                           all_letters: tp.List[str]) -> tp.Dict[str, float]:
    '''
    Возвращает датафрейм распределения меток классов, где ключ - буква, значение - число верных ответов под буквой.
    Parameters:
        true_answers: Колонка верных ответов
        all_letters: Список всех букв из пула ответов answer_options
    Return:
        Словарь распределения меток классов.
    '''
    distribution_true_answers = true_answers.value_counts().sort_index().to_dict()
    set_keys = sorted(list(set(all_letters) | set(distribution_true_answers)))
    distribution = {key: distribution_true_answers.get(key, 0) for key in set_keys}
    distribution = pd.DataFrame(distribution.items(), columns = ['Вариант ответа', 'Количество ответов'])
    distribution.set_index('Вариант ответа', inplace=True)
    return distribution

def get_distribution_dict(distribution: pd.DataFrame) -> tp.Dict[str, float]:
    '''
    Возвращает словарь распределения меток классов, где ключ - буква, значение - число верных ответов под буквой
    Parameters:
        distribution: Датафрейм с распределением меток классов
    Return:
        Словарь распределения меток классов
    '''
    return distribution['Количество ответов'].to_dict()


def chi_2_stats(distribution: tp.Dict) -> tp.Tuple[float, float]:
    '''
    Подсчитывает значения статистики и p_value для теста хи-квадрат равномерного распределения.
    Parameters:
        distribution: Словарь из распределения меток классов
    Return:
        Значение статистики и p_value для теста хи-квадрат равномерного распределения
    '''
    observed_values = list(distribution.values())
    expected_values = [sum(observed_values) / len(observed_values)] * len(observed_values)
    chi_2, p_value = stats.chisquare(observed_values, expected_values)
    if np.isnan(p_value):
        p_value = 0.0
    return chi_2, p_value


def get_permutate_data(X: pd.DataFrame, 
                       rng: Generator):
    '''
    Создает датасет с перемешанными ответами.
    Parameters:
        X: Распаршенный исходный датасет из сэмплера
        rng: Генератор случайных чисел
    Return:
        Датасет с перемешанными вариантами ответов
    '''
    permutate_mmlu = X.copy()
    permutate_mmlu[['answer_options', 'true_answer']] = X.apply(lambda row: permutate_row(row, rng), axis=1)
    return permutate_mmlu


def permutate_row(row: pd.Series, 
                  rng: Generator) -> pd.Series:
    '''
    Перемешивает answer_options и переопределяет новую верную метку.
    Parameters:
        row: Одна строка из тестового набора данных
        rng: Генератор случайных чисел
    Return:
        Перемешанный answer_options и новая метка верного ответа
    '''
    new_true_answer = ''
    perm_answer_options = {}
    keys = list(row['answer_options'].keys())
    permutate_keys = rng.choice(keys, size=len(keys), replace=False)
    for (letter, perm_letter) in zip(keys, permutate_keys):
        perm_answer_options.update({letter: row['answer_options'][perm_letter]})
        if perm_letter == row['true_answer']:
            new_true_answer = letter
    return pd.Series((perm_answer_options, new_true_answer))


def plot_distribution_mmlu(distribution: tp.Dict[str, float],
                           perm_flag: bool=False) -> tp.Any:
    '''
    Отрисовка графика распределения истиных меток.
    Parameters:
        distribution: Таблица с колонками меток класса и числом вхождений этой метки в тестовый набор данных
        perm_flag: Флаг, определяющий title для графика распределения ответов в зависимости от того, было перемешивание или нет
    Return:
        Круговой график распределения меток класса
    '''
    title = {False: '', True: 'перемешанных '}[perm_flag]
    fig = plt.figure(figsize=(6, 6))
    plt.pie(list(distribution.values()), labels=list(distribution.keys()), autopct='%1.2f%%')
    plt.title(f'Распределение {title}меток классов')
    plt.text(2.5, 0.2, distribution)
    plt.show()
    return fig


def get_all_letters(answer_options: tp.List[str]):
    '''
    Инициализирует список с ключами, состоящими из всех представленных букв пула ответов.
    Parameters:
        answer_options: Список строк распаршенных ответов из сэмплера
    Return:
        Список всех возможных букв из answer_options, представленных как варианты ответов.
    '''
    all_keys = []
    for answer in answer_options:
        all_keys.extend(answer.keys())
    all_keys = sorted(set(all_keys))
    return all_keys
