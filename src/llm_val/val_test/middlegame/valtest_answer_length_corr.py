import typing as tp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from llm_val.sampler.Sampler import Sampler


def report_valtest_answer_length_corr(fig: tp.Any,
                                      spearman_stat: float,
                                      spearman_pval: float,
                                      semaphore_threshold_spear: float=0.3,
                                      semaphore_threshold_pval: float=0.05,
                                      is_info: bool = False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        spearman_stat: Значение корреляции Спирмена
        spearman_pval: Значение p-value статистики Спирмена
        semaphore_threshold_spear: Границы светофора для статистики Спирмена
        semaphore_threshold_pval: Границы светофора для p-value статистики Спирмена
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    if spearman_stat < semaphore_threshold_spear and spearman_pval < semaphore_threshold_pval:
        color = 'green'
    elif spearman_stat >= semaphore_threshold_spear and spearman_pval >= semaphore_threshold_pval:
        color = 'yellow'
    elif spearman_stat >= semaphore_threshold_spear and spearman_pval < semaphore_threshold_pval:
        color = 'red'
    else:
        color = 'gray'

    res_table = pd.DataFrame({"Статистика": ["Корреляция Спирмена"],
                              "Значение статистики": [spearman_stat],
                              "p-value статистики": [spearman_pval]})
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else color
    res_dict['result_plots'] = [fig]
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_answer_length_corr(sampler: Sampler,
                               data_type: str='train',
                               semaphore_threshold_spear: float=0.3,
                               semaphore_threshold_pval: float=0.05,
                               is_info: bool=False,
                               spearman_stat: tp.Optional[float]=None,
                               spearman_pval: tp.Optional[float]=None,
                               **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на наличие корреляции между корректностью и длиной ответа в словах.
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold_spear: Границы светофора для статистики Спирмена
        semaphore_threshold_pval: Границы светофора для p-value статистики Спирмена
        is_info: True, если тест информативный
        spearman_stat: Преподсчитанное опциональное значение корреляции Спирмена.
            Если не None, то подсчет не производится.
        spearman_pval: Преподсчитанное опциональное значение p-value статистики Спирмена.
            Если не None, то подсчет не производится.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    data = getattr(sampler, data_type)['X'][['answer_options', 'true_answer']]

    df = unravel_mmlu_dataset(data)
    df['answer_length'] = df['answer'].apply(lambda x: len(set(x.split())))

    if spearman_stat is None or spearman_pval is None:
        spearman_stat, spearman_pval = spearmanr(df['answer_length'],
                                                 df['is_correct'])
    fig = plot_lengths(df)
    report = report_valtest_answer_length_corr(fig,
                                               spearman_stat,
                                               spearman_pval,
                                               semaphore_threshold_spear,
                                               semaphore_threshold_pval,
                                               is_info=is_info)
    precomputed = {'spearman_stat': spearman_stat,
                   'spearman_pval': spearman_pval,
                   'is_correct': df['is_correct']}
    return {'report': report,
            'precomputed': precomputed}


def unravel_mmlu_dataset(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Распутывает исходный датасет, превращая каждую строку датасета data в n строк,
    где n - количество вариантов ответов
    Parameters:
        data: Исходный датасет - answer_options, true_answer колонки датасета mmlu
    Return:
        df: pd.DataFrame с двумя колонками: ответ, его корректность
    '''
    unraveled = []
    for _, r in data.iterrows():
        d = r.values[0]
        unraveled.append((d[r.values[1]], 1))
        wrong = [d[k] for k in d if k != r.values[1]]
        unraveled += list(zip(wrong, len(wrong) * [0]))
    return pd.DataFrame(unraveled, columns=['answer', 'is_correct'])


def plot_lengths(df: pd.DataFrame) -> tp.Any:
    '''
    Отрисовка графика длин ответов с изображением верности и длины
    Parameters:
        df: pd.DataFrame
    Return:
        График, показывающий длины ответов и их правильность
    '''
    true_lengths = df[df.is_correct == 1]['answer_length']
    false_lengths = df[df.is_correct == 0]['answer_length']
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.histplot(true_lengths, color='blue', label='Длины правильных ответов', ax=ax, alpha=0.4)
    sns.histplot(false_lengths, color='red', label='Длины неправильных ответов', ax=ax, alpha=0.4)
    ax.legend()
    ax.grid()
    return fig