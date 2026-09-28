import typing as tp
import pandas as pd
from copy import deepcopy

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.val_test.model_quality.valtest_metric import valtest_metric
from llm_val.utils.report_helper import semaphore_by_threshold, worst_semaphore


def report_valtest_metric_stability(base_metric_value: tp.Dict[str, float],
                                    test_metric_value: tp.Dict[str, float],
                                    test_color: str,
                                    main_metric: str,
                                    metric_agg: str='single_mean',
                                    data_types: tp.Tuple[str, str]=('train', 'test'),
                                    semaphore_threshold: tp.Tuple[float]=(0.15, 0.25),
                                    greater_is_better: bool=True,
                                    is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        base_metric_value: Словарь со значением метрики на старых данных
        test_metric_value: Словарь со значением метрики на новых данных
        test_color: Светофор по тесту на целевую метрику на out-of-time выборке
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        data_types: Типы данных, на которых сравниваются метрики
        semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: Если True, то тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    result_dataframes = []
    df = pd.DataFrame()
    df[f'Значение метрики на {data_types[0]}'] = pd.Series(base_metric_value)
    df[f'Значение метрики на {data_types[1]}'] = pd.Series(test_metric_value)
    df['Абсолютная разница'] = df[f'Значение метрики на {data_types[1]}'] - df[f'Значение метрики на {data_types[0]}']
    df['Абсолютная разница'] = -df['Абсолютная разница'] if greater_is_better else df['Абсолютная разница']
    df['Относительная разница, %'] = df['Абсолютная разница'] / df[f'Значение метрики на {data_types[0]}'] * 100
    df['Результат теста'] = df.apply(lambda x: 'gray' if is_info else semaphore_by_threshold(x['Абсолютная разница'],
                                                                                             semaphore_threshold,
                                                                                             greater_is_better=False), axis=1)
    df = df.T
    res_table = pd.DataFrame(df.pop(main_metric)).T
    df = df.T
    color_stability = res_table['Результат теста'].values[0]
    if len(df) > 0:
        result_dataframes.append(df)
    color = worst_semaphore([test_color, color_stability])
    result_dataframes.insert(0, res_table)
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = result_dataframes
    return res_dict


def valtest_metric_stability(sampler: Sampler,
                             scorer: Scorer,
                             main_metric: str, 
                             metric_agg: str='single_mean',
                             data_types: tp.Tuple[str, str]=('train', 'test'),
                             semaphore_threshold: tp.Tuple[float]=(0.15, 0.25),
                             greater_is_better: bool=True,
                             is_info: bool=False,
                             base_metric_value: tp.Optional[float]=None,
                             test_metric_value: tp.Optional[float]=None,
                             **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на стабильность ключевой метрики качества
    Parameters:
        sampler: Объект класса Sampler
        scorer: Объект класса Scorer
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        data_types: Типы данных, на которых сравниваются метрики
        semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
        base_metric_value: Опциональное значение метрики на старых данных
        test_metric_value: Опциональное значение метрики на новых данных
            Если base_metric_value и test_metric_value не None, то подсчет метрик не производится 
                и на их основе выставляется светофор
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if base_metric_value is None:
        base_metric_value = scorer.calc(sampler=sampler,
                                        data_type=data_types[0],
                                        metric_name=metric_agg)
    if test_metric_value is None:
        test_metric_value = scorer.calc(sampler=sampler,
                                        data_type=data_types[1],
                                        metric_name=metric_agg)
    precomputed = {'base_metric_value': deepcopy(base_metric_value),
                   'test_metric_value': deepcopy(test_metric_value)}
    test_color = valtest_metric(sampler=sampler,
                                scorer=scorer,
                                main_metric=main_metric,
                                metric_agg=metric_agg,
                                greater_is_better=greater_is_better,
                                metric_value=test_metric_value)['report']['semaphore']
    report = report_valtest_metric_stability(base_metric_value,
                                             test_metric_value,
                                             test_color,
                                             main_metric,
                                             metric_agg,
                                             data_types,
                                             semaphore_threshold,
                                             greater_is_better,
                                             is_info)
    return {'report': report,
            'precomputed': precomputed}