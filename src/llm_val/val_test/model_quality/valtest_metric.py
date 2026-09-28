import typing as tp
import pandas as pd
from copy import deepcopy

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold, tricky_semaphore


def report_valtest_metric(metric_value: tp.Dict[str, float],
                          main_metric: str,
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
        metric_value: Значение метрик
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold: Словарь с границами светофора, где ключ - название метрики, а значение - кортеж с границами светофора. Границы в кортеже указываются в порядке возрастания
        main_semaphore_threshold: Границы светофора для ключевой метрики качества
        red_freq_threshold: Количество допустимых красных светофоров (см. таблицу 4.14)
        yellow_freq_threshold: Количество допустимых желтых светофоров (см. таблицу 4.14)
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    color_list = {}
    metric_value = deepcopy(metric_value)
    main_value = metric_value.pop(main_metric)
    for name, value in metric_value.items():
        color_list[name] = semaphore_by_threshold(value,
                                                  threshold=semaphore_threshold.get(name, semaphore_threshold['default']),
                                                  greater_is_better=greater_is_better)
    color = 'gray' if is_info else tricky_semaphore(value=main_value,
                                                    semaphore_list=list(color_list.values()),
                                                    greater_is_better=greater_is_better,
                                                    threshold_metric=main_semaphore_threshold,
                                                    red_freq_threshold=red_freq_threshold,
                                                    yellow_freq_threshold=yellow_freq_threshold)
    result_dataframes = []
    if len(color_list) > 0:
        result_dataframes.append(pd.DataFrame([metric_value, color_list], 
                                              index=[f'Значение метрики на {data_type}', 'Результат теста']).T)
    result_dataframes.insert(0, pd.DataFrame({f'Значение ключевой метрики "{main_metric}" на {data_type}': [main_value], 'Результат теста': [color]}))
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = result_dataframes
    return res_dict


def valtest_metric(sampler: Sampler,
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
                   metric_value: tp.Optional[tp.Dict[str, float]]=None,
                   **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест подсчета метрик
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
        greater_is_better: True, если чем больше метрика, тем лучше.
        is_info: True, если тест информативный
        metric_value: Опциональное значение метрики. 
            Если не None, то подсчет метрики не производится, и на основе переданного аргумента выставляется светофор.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if metric_value is None:
        metric_value = scorer.calc(sampler=sampler,
                                   data_type=data_type,
                                   metric_name=metric_agg)
    precomputed = {'metric_value': deepcopy(metric_value)}
    report = report_valtest_metric(metric_value=metric_value,
                                   main_metric=main_metric,
                                   data_type=data_type,
                                   semaphore_threshold=semaphore_threshold,
                                   main_semaphore_threshold=main_semaphore_threshold,
                                   red_freq_threshold=red_freq_threshold,
                                   yellow_freq_threshold=yellow_freq_threshold,
                                   greater_is_better=greater_is_better,
                                   is_info=is_info)
    return {'report': report, 
            'precomputed': precomputed}
