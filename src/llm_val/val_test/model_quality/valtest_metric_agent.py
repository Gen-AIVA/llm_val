import typing as tp
import pandas as pd
from copy import deepcopy

from llm_val.scorer.Scorer import Scorer
from llm_val.utils.metrics import METRICS
from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold, tricky_semaphore_agent


def report_valtest_metric_agent(metric_value: tp.Dict[str, float],
                                main_metric: str,
                                data_type: str='test',
                                semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]]={'default': (0.4, 0.5)},
                                main_semaphore_threshold: tp.Tuple[float, float]=(0.4, 0.6),
                                red_freq_threshold: float=0.2,
                                greater_is_better: bool=True,
                                is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а в соответствии с концепцией llm_val
    Parameters:
        metric_value: Значение метрик
        main_metric: Название ключевой метрики качества.
            Hint: один из ключей словаря, который выдает scorer
        data_type: Тип данных, на которых производится расчет метрики.
            Hint: название атрибута в sampler'е
        semaphore_threshold: Словарь с границами светофора для второстепенных критериев,
            где ключ - название метрики, а значение - кортеж с границами светофора.
            Границы в кортеже указываются в порядке возрастания
        main_semaphore_threshold: Границы светофора для ключевой метрики качества
        red_freq_threshold: Граница по относительной доле красных светофоров
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    color_list = {}
    metric_value = deepcopy(metric_value)
    main_value = metric_value.pop(main_metric)
    for name, value in metric_value.items():
        color_list[name] = semaphore_by_threshold(
            value,
            threshold=semaphore_threshold.get(name, semaphore_threshold['default']),
            greater_is_better=greater_is_better)
    color = ('gray' if is_info
             else tricky_semaphore_agent(
                 value=main_value,
                 semaphore_list=list(color_list.values()),
                 greater_is_better=greater_is_better,
                 threshold_metric=main_semaphore_threshold,
                 red_freq_threshold=red_freq_threshold))
    result_dataframes = []
    if len(color_list) > 0:
        result_dataframes.append(
            pd.DataFrame([metric_value, color_list],
                         index=[f'Значение метрики на {data_type}',
                                'Результат теста']).T)
    result_dataframes.insert(
        0,
        pd.DataFrame({
            f'Значение ключевой метрики "{main_metric}" на {data_type}': [main_value],
            'Результат теста': [color]}))
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = result_dataframes
    return res_dict


def valtest_metric_agent(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                         scorer: tp.Optional[Scorer],
                         main_metric: str,
                         data_type: str='test',
                         metric_agg: str='single_mean',
                         semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]]={'default': (0.4, 0.5)},
                         main_semaphore_threshold: tp.Tuple[float, float]=(0.4, 0.6),
                         red_freq_threshold: float=0.2,
                         greater_is_better: bool=True,
                         exec_env: str='default',
                         calc_metric_fn: tp.Optional[tp.Callable]=None,
                         is_info: bool=False,
                         metric_value: tp.Optional[tp.Dict[str, float]]=None,
                         **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на качество прогноза модели по ключевой метрике качества.
    Parameters:
        sampler: Объект класса Sampler (exec_env='default') или словарь с данными (exec_env='sberds')
        scorer: Объект класса Scorer. Используется при exec_env='default'
        main_metric: Название ключевой метрики качества.
            Hint: один из ключей словаря, который выдает scorer
        data_type: Тип данных, на которых производится расчет метрики.
            Hint: название атрибута в sampler'е
        metric_agg: Название агрегации метрики.
            Hint: один из ключей в словаре metrics в scorer'е
        semaphore_threshold: Словарь с границами светофора для второстепенных критериев,
            где ключ - название метрики, а значение - кортеж с границами светофора.
            Границы в кортеже указываются в порядке возрастания
        main_semaphore_threshold: Границы светофора для ключевой метрики качества
        red_freq_threshold: Граница по относительной доле красных светофоров
        greater_is_better: True, если чем больше метрика, тем лучше
        exec_env: Среда выполнения. 'default' - стандартная, 'sberds' - SberDS
        calc_metric_fn: Функция подсчета метрики для exec_env='sberds'.
            Сигнатура: (sampler, data_type, metric_agg) -> Dict[str, float]
        is_info: True, если тест информативный
        metric_value: Опциональное значение метрики.
            Если не None, то подсчет метрики не производится,
            и на основе переданного аргумента выставляется светофор
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if metric_value is None:
        if exec_env == 'sberds':
            fn = (calc_metric_fn if calc_metric_fn is not None
                  else default_calc_metric_fn)
            metric_value = fn(
                sampler, 
                data_type, 
                metric_agg)
        else:
            metric_value = scorer.calc(
                sampler=sampler,
                data_type=data_type,
                metric_name=metric_agg)
    precomputed = {'metric_value': deepcopy(metric_value)}
    report = report_valtest_metric_agent(
        metric_value=metric_value,
        main_metric=main_metric,
        data_type=data_type,
        semaphore_threshold=semaphore_threshold,
        main_semaphore_threshold=main_semaphore_threshold,
        red_freq_threshold=red_freq_threshold,
        greater_is_better=greater_is_better,
        is_info=is_info)
    return {'report': report,
            'precomputed': precomputed}


def calc_metric(data: pd.DataFrame,
                metric_name: tp.Optional[str]=None,
                metrics: tp.Dict[str, tp.Any]=METRICS) -> tp.Dict[str, float]:
    '''
    Подсчет метрик по данным
    Parameters:
        data: DataFrame с данными для подсчета метрик
        metric_name: Название агрегации метрики
        metrics: Словарь с метриками
    Return:
        Словарь с посчитанными метриками
    '''
    if metric_name is not None:
        metrics = {metric_name: metrics[metric_name]}
    metric_res = {}
    for func_name, func_dict in metrics.items():
        if func_dict['is_singlecol']:
            for name_col in data:
                name = f'{func_name}_{name_col}'
                metric_res[name] = func_dict['call'](data[name_col])
        else:
            metric_res[func_name] = func_dict['call'](data)
    return metric_res


def default_calc_metric_fn(sampler: tp.Dict[str, tp.Any],
                           data_type: str,
                           metric_agg: str) -> tp.Dict[str, float]:
    '''
    Функция подсчета метрики по умолчанию для exec_env='sberds'
    Parameters:
        sampler: Словарь с данными
        data_type: Тип данных
        metric_agg: Название агрегации метрики
    Return:
        Словарь с посчитанными метриками
    '''
    data = pd.DataFrame.from_dict(sampler[data_type]['y'])
    return calc_metric(data, metric_name=metric_agg)
