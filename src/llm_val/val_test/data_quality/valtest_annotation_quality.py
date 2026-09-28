import copy
import numpy as np
import typing as tp
import pandas as pd
import krippendorff

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold, worst_semaphore


def report_valtest_annotation_quality(spearman_stat: float,
                                      krippendorf_stat: float,
                                      semaphore_threshold_spear: tp.Tuple[float, float]=(0.1, 0.6),
                                      semaphore_threshold_krip: tp.Tuple[float, float]=(0., 0.4),
                                      is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        spearman_stat: Значение корреляции Спирмена, range = [-1, 1]
        krippendorf_stat: Значение альфы Криппендорфа, range = [-1, 1]
        semaphore_threshold_spear: Границы светофора для Спирмена.
        semaphore_threshold_krip: Границы светофора для статистики Криппендорфа.
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    color_spearman = 'gray' if np.isnan(spearman_stat) else semaphore_by_threshold(spearman_stat, semaphore_threshold_spear)
    color_krippendorf = 'gray' if np.isnan(krippendorf_stat) else semaphore_by_threshold(krippendorf_stat, semaphore_threshold_krip)
    color = worst_semaphore([color_spearman, color_krippendorf])
    res_table = pd.DataFrame({'Наименование критерия': ["Альфа-версия Криппендорфа",
                                                        "Попарный коэффициент Спирмена"],
                              'Значение': [krippendorf_stat, spearman_stat],
                              'Результат теста': [color_krippendorf, color_spearman]})
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_annotation_quality(sampler: Sampler,
                               data_type: str='train',
                               min_periods: int=1,
                               difference_function: str='ordinal',
                               semaphore_threshold_spear: tp.Tuple[float, float]=(0.1, 0.6),
                               semaphore_threshold_krip: tp.Tuple[float, float]=(0., 0.4),
                               is_info: bool=False,
                               spearman_stat: tp.Optional[float]=None,
                               krippendorf_stat: tp.Optional[float]=None,
                               **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на оценку качества разметки ответов LLM асессорами 
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        min_periods: Минимальное количество не np.nan элементов для подсчета корреляции Спирмена между асессорами
        difference_function: Функция, которая задает метрику на множестве возможных ответов асессоров
            https://en.wikipedia.org/wiki/Level_of_measurement
        semaphore_threshold_spear: Границы светофора для корреляции Спирмена.
        semaphore_threshold_krip: Границы светофора для альфы Криппендорфа.
        is_info: True, если тест информативный
        spearman_stat: Преподсчитанное опциональное значение корреляции Спирмена.
            Если не None, то подсчет статистики не производится, и на основе переданного аргумента выставляется светофор.
        krippendorf_stat: Преподсчитанное опциональное значение альфы Криппендорфа.
            Если не None, то подсчет статистики не производится, и на основе переданного аргумента выставляется светофор.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    X = getattr(sampler, data_type)['X'][getattr(sampler, data_type)['y'].columns]
    if spearman_stat is None:
        spearman_stat = spearman_pair_correlation(X,
                                                  min_periods)
    if krippendorf_stat is None:
        krippendorf_stat = krippendorff_alpha(X,
                                              difference_function)
    report = report_valtest_annotation_quality(spearman_stat,
                                               krippendorf_stat,
                                               semaphore_threshold_spear,
                                               semaphore_threshold_krip,
                                               is_info=False)
    precomputed = {'spearman_stat': spearman_stat,
                   'krippendorf_stat': krippendorf_stat}
    return {'report': report, 
            'precomputed': precomputed}


def krippendorff_alpha(df: pd.DataFrame,
                       difference_function: str) -> float:
    '''
    Вычисление альфы Криппендорфа для датасета longanswer.
    Датасет разбивается на критерии оценивания (questions), для каждого критерия строится reliability matrix,
    у которой индексы - worker, а колонки - id_task. По этой матрице считается альфа Криппендорфа.
    Далее берется среднее всех коэффициентов по каждому критерию.
    Parameters:
        df: pd.DataFrame, содержащий только колонки с критериями, полученный из LongAnswerSampler'а  
    Return:
        Коэффицент альфа Криппендорфа
    '''    
    results = []
    for criteria in df:
        criteria_ratings = []
        for _, row in df.iterrows():
            dict_ratings = row[criteria]
            for worker in dict_ratings:
                criteria_ratings.append((row.name, worker, dict_ratings[worker]))
        ratings_matrix = pd.DataFrame(criteria_ratings,
                                      columns=['id_task', 'worker', 'rating'])
        ratings_matrix = ratings_matrix.pivot(index='worker',
                                              columns='id_task',
                                              values='rating')
        if len(np.unique(ratings_matrix.to_numpy()[~np.isnan(ratings_matrix.to_numpy())])) <= 1:
            continue
        results.append(krippendorff.alpha(reliability_data=ratings_matrix.to_numpy(),
                                          level_of_measurement=difference_function))
    return sum(results) / len(results) if results else np.nan


def merge_dicts(dict_list: tp.List[tp.Dict]) -> tp.Dict:
    '''
    Объединяет несколько словарей в один, ключи которого являются логическим объединением ключей из исходных словарей.
    Так как не каждый словарь содержит все ключи, на месте пропусков будут выставлены np.nan.
    Parameters:
        dict_list: список словарей
    Return:
        Объединенный словарь
    '''  
    for j, d in enumerate(dict_list):
        for k in d:
            d[k] = (d[k], j)
            
    merged_dict = {}            
    n_tasks = len(dict_list)
    for d in dict_list:
        for key, value in d.items():
            if key not in merged_dict:
                merged_dict[key] = np.full(n_tasks, np.nan)
            merged_dict[key][value[1]] = value[0]
    return merged_dict
    

def spearman_pair_correlation(df: pd.DataFrame,
                              min_periods: int) -> float:
    '''
    Вычисление попарного коэффициента корреляции Спирмена для датасета longanswer.
    Датасет разбивается на критерии, для каждого считается попарная корреляция между асессорами,
    далее берется среднее всех корреляций.
    Parameters:
        df: pd.DataFrame, содержащий только колонки с критериями, полученный из LongAnswerSampler'а 
        min_periods: Минимальное количество не np.nan элементов для подсчета корреляции Спирмена между асессорами
    Return:
        Коэффицент Спирмена
    '''   
    cnt = 0
    correlations_sum = 0
    for criteria in df:
        dicts = copy.deepcopy(df[criteria].tolist())
        df_one_criteria_corr = pd.DataFrame(merge_dicts(dicts)).corr(method='spearman', min_periods=min_periods).to_numpy()
        correlations_sum += np.nansum(np.triu(df_one_criteria_corr)) - np.nansum(np.diagonal(df_one_criteria_corr))
        cnt += np.triu(~np.isnan(df_one_criteria_corr)).sum() - np.diagonal(~np.isnan(df_one_criteria_corr)).sum()
    return correlations_sum / cnt if correlations_sum != 0 else np.nan
