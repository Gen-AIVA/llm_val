import typing as tp
import pandas as pd
from copy import deepcopy
from collections import defaultdict 

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import (semaphore_by_threshold,
                                         worst_semaphore,
                                         tricky_semaphore)


def report_valtest_metric_by_thema(metric_value_by_thema: tp.Dict[str, tp.Dict[str, float]],
                                   main_value_by_thema: tp.Dict[str, float],
                                   main_metric: str,
                                   value_counts_by_thema: tp.Optional[pd.Series]=None,
                                   data_type: str='test',
                                   semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]]={'default': (0.4, 0.6)},
                                   main_semaphore_threshold: tp.Tuple[float, float]=(0.4, 0.6),
                                   red_freq_threshold: float=0.2,
                                   yellow_freq_threshold: float=0.2,
                                   greater_is_better: bool=True,
                                   is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        metric_value_by_thema: Значение метрик в разрезе тематик
        value_counts_by_thema: Количество элементов в разрезе тематик
        main_metric: Название ключевой метркики качества. Hint: один из ключей словаря, который выдает scorer
        value_counts_by_thema: Series с количеством элеметов в разрезе тематик. Hint: df['thema'].value_counts()
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold: Словарь с границами светофора, где ключ - название метрики, а значение - кортеж с границами светофора. Границы в кортеже указываются в порядке возрастания.
        main_semaphore_threshold: Границы светофора для ключевой метрики качества
        red_freq_threshold: Количество допустимых красных светофоров (см. таблицу 4.14)
        yellow_freq_threshold: Количество допустимых желтых светофоров (см. таблицу 4.14)
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    global_color_list = {}
    result_dataframes = []
    if value_counts_by_thema is not None:
        info_df = pd.DataFrame(value_counts_by_thema).rename({'thema': 'Количество элементов в разрезе тематик'}, axis='columns')
        result_dataframes.append(info_df)
    for thema, metric_value in metric_value_by_thema.items():
        color_list = {}
        for name, value in metric_value.items():
            color_list[name] = semaphore_by_threshold(value,
                                                      threshold=semaphore_threshold.get(name, semaphore_threshold['default']),
                                                      greater_is_better=greater_is_better)
        color = tricky_semaphore(value=main_value_by_thema[thema],
                                 semaphore_list=list(color_list.values()),
                                 greater_is_better=greater_is_better,
                                 threshold_metric=main_semaphore_threshold,
                                 red_freq_threshold=red_freq_threshold,
                                 yellow_freq_threshold=yellow_freq_threshold)
        global_color_list[thema] = color
        if len(metric_value) > 0:
            res_table = pd.DataFrame([metric_value, color_list], 
                                     index=[f'Значение метрики на {data_type} в разрезе тематики/домена {thema}', 'Результат теста']).T
            result_dataframes.append(res_table)
    result_dataframes.insert(0, pd.DataFrame([main_value_by_thema, global_color_list], 
                                             index=[f'Значение ключевой метрики "{main_metric}" на {data_type} в разрезе тематик', 'Результат теста']).T)
    color = 'gray' if is_info else worst_semaphore(global_color_list.values())
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = result_dataframes
    return res_dict


def valtest_metric_by_thema(sampler: Sampler,
                            scorer: Scorer,
                            main_metric: str,
                            data_type: str='test',
                            metric_agg: str='single_mean',
                            semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]]={'default': (0.4, 0.6)},
                            main_semaphore_threshold: tp.Tuple[float, float]=(0.4, 0.6),
                            red_freq_threshold: float=0.2,
                            yellow_freq_threshold: float=0.2,
                            greater_is_better: bool=True,
                            is_info: bool=False,
                            metric_value_by_thema: tp.Optional[tp.Dict[str, float]]=None,
                            **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест подсчета метрик в разрезе тематик
    Parameters:
        sampler: Объект класса Sampler
        scorer: Объект класса Scorer
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        semaphore_threshold: Словарь с границами светофора, где ключ - название метрики, а значение - кортеж с границами светофора. Границы в кортеже указываются в порядке возрастания.
        main_semaphore_threshold: Границы светофора для ключевой метрики качества
        red_freq_threshold: Количество допустимых красных светофоров (см. таблицу 4.14)
        yellow_freq_threshold: Количество допустимых желтых светофоров (см. таблицу 4.14)
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
        metric_value_by_thema: Опциональное значение метрики. 
            Если не None, то подсчет метрики не производится, и на основе переданного аргумента выставляется светофор.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    value_counts_by_thema = None
    if metric_value_by_thema is None:
        data = getattr(sampler, data_type)
        X, y = data['X'], data['y']
        value_counts_by_thema=X['thema'].value_counts(dropna=False)
        if (len(X['thema'].unique()) <= 1) and (X['thema'].unique()[0] is None):
            is_info = True
        metric_value_by_thema = calculate_metric_by_thema(X=X,
                                                          y=y,
                                                          scorer=scorer,
                                                          metric_agg=metric_agg)
    precomputed = {'metric_value_by_thema': deepcopy(metric_value_by_thema)}
    metric_value_by_thema, main_value_by_thema = get_main_value_by_thema(metric_value_by_thema, main_metric)
    
    report = report_valtest_metric_by_thema(metric_value_by_thema=metric_value_by_thema,
                                            main_value_by_thema=main_value_by_thema,
                                            main_metric=main_metric,
                                            value_counts_by_thema=value_counts_by_thema,
                                            data_type=data_type,
                                            semaphore_threshold=semaphore_threshold,
                                            main_semaphore_threshold=main_semaphore_threshold,
                                            red_freq_threshold=red_freq_threshold,
                                            yellow_freq_threshold=yellow_freq_threshold,
                                            greater_is_better=greater_is_better,
                                            is_info=is_info)
    
    return {'report': report,
            'precomputed': precomputed}


def calculate_metric_by_thema(X: pd.DataFrame,
                              y: pd.DataFrame,
                              scorer: Scorer,
                              metric_agg: str='single_mean') -> tp.Dict[str, tp.Dict[str, float]]:
    '''
    Расчет метрик в разрезе тематик
    Parameters:
        X: DataFrame из sampler'а
        y: DataFrame из sampler'а
        scorer: Объект класса Scorer
        metric_agg: Название агрегации метрики
    Return:
        Словарь c значениями метрик
    '''
    metric_value = {}
    for thema in X['thema'].unique():
        if thema is None:
            X_tmp = X[X['thema'].isnull()].copy()
        else:
            X_tmp = X[X['thema'] == thema].copy()
        y_tmp = y.loc[X_tmp.index]
        sampler = Sampler(test={'X': X_tmp, 'y': y_tmp})
        metric_value[thema] = scorer.calc(sampler=sampler,
                                          data_type='test',
                                          metric_name=metric_agg)
    return metric_value


def get_main_value_by_thema(metric_value_by_thema: tp.Dict[str, tp.Dict[str, float]],
                            main_metric: str) -> tp.Tuple[tp.Dict[str, tp.Dict[str, float]], tp.Dict[str, float]]:
    '''
    Получение ключевой метрики в разрезе тематик
    Parameters:
        metric_value_by_thema: Значение метрик в разрезе тематик. Hint: словарь, где ключ - название темы, значение - словарь, который возвращает scorer
        main_metric: Название ключевой метркики качества. Hint: один из ключей словаря, который выдает scorer
    Return:
        Кортеж из двух элементов: 
            первый - значение метрик в разрезе тематик, но без ключевой метрики;
            второй - значение ключевых метрик в разрезе тематик
    '''
    main_value_by_thema = {}
    metric_value_by_thema = deepcopy(metric_value_by_thema)
    for i in metric_value_by_thema:
        main_value_by_thema[i] = metric_value_by_thema[i].pop(main_metric)
    return metric_value_by_thema, main_value_by_thema
        
    