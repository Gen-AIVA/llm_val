import sberpm
import numpy as np
import pandas as pd
import typing as tp
from sberpm import DataHolder
from sberpm.autoinsights import AutoInsights
from llm_val.sampler.Sampler import Sampler


def report_valtest_loop_duration_analysis(result_dataframe: pd.DataFrame) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'a
    Parameters:
        result_dataframe: датафрейм с рассчитанными по каждой ноде из трейсов статистиками по времени работы
                          и мере зацикленности
    Return: 
        report в соответствии с концепцией llm_val
    '''
    res_dict = {}
    res_dict['semaphore'] = "gray"
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [result_dataframe]
    return res_dict


def valtest_loop_duration_analysis(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                   data_type: str='test',
                                   exec_env: str='default',
                                   loop_time_stats: pd.DataFrame=None,
                                   **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на оценку времени работы и степени зацикленности агента.
    Parameters:
        sampler: Объект класса Sampler
        data_type: Ключ, где лежат трейсы агента
        exec_env: среда выполнения ('default' или 'sberds')
        loop_time_stats: датафрейм с рассчитанными статистиками по времени работы и мере зацикленности графа агента
    Return:
        Словарь в соответствии с концепцией с llm_val
    '''
    if loop_time_stats is None:
        traces = get_data(sampler, data_type, 'traces', exec_env).copy()
        data_holder = DataHolder(data=traces,
                                col_case='trace_id',
                                col_stage='span_name',
                                col_start_time='start_time_ns',
                                col_end_time='end_time_ns',)
        times_statistics = calc_time_statistic(data_holder)
        loop_statistics = calc_loop_statistic(data_holder)
        loop_time_stats = times_statistics.merge(loop_statistics, on='Нода', how='left')
        loop_time_stats = loop_time_stats.rename(columns={
            'В начало':'Зацикленность в начало',
            'В себя':'Зацикленность в себя',
            '«Возврат»':'Зацикленность "Возврат"',
            '«Пинг-Понг»':'Зацикленность "Пинг-Понг"',
            'В произвольную операцию':'Зацикленность в прозвольную ноду',
        })
        metrics_columns = loop_time_stats.select_dtypes(['float64', 'float32']).columns
        loop_time_stats[metrics_columns] = loop_time_stats[metrics_columns].applymap('{:,.2f}'.format)
    report = report_valtest_loop_duration_analysis(loop_time_stats)
    precomputed = {'loop_time_stats': loop_time_stats}
    return {'report': report,
            'precomputed': precomputed}


def calc_time_statistic(data_holder: sberpm.DataHolder) -> pd.DataFrame:
    '''
    Функция для расчета временных статистик для каждой ноды в представленных трейсах агента.
    Parameters:
        data_holder: обертка над датафреймом с трейсами с размеченными колонками идентификатора трейса, 
                     названия ноды, времени ее начала и конца работы
    Return:
        pd.DataFrame с рассчитанными статистиками по времени работы каждой ноды в датафрейме с трейсами
    '''
    stages_info = data_holder.data.groupby(data_holder.col_stage)
    results = []
    for stage, group in stages_info:
        durations = group['duration (synthetic)']
        normal_mask = (durations > durations.quantile(0.05)) & (durations < durations.quantile(0.95))
        mean_without_outliers = durations.loc[normal_mask].mean()
        results.append({'Нода': stage,
                        'Среднее время': round(durations.mean(), 4),
                        'Среднее время без выбросов': round(mean_without_outliers, 4)})
    final_processing_time = pd.DataFrame(results)
    return final_processing_time


def calc_loop_statistic(data_holder: sberpm.DataHolder) -> pd.DataFrame:
    '''
    Функция для расчета статистик с помощью алгоритма sberpm.Autoinsights
    Parameters:
        data_holder: обертка над датафреймом с трейсами с размеченными колонками идентификатора трейса, 
                     названия ноды, времени ее начала и конца работы
    Return:
        pd.DataFrame с рассчитанными значениями меры зацикленности для каждой ноды в датафрейме с трейсами
    '''
    auto_insights = AutoInsights(holder=data_holder)
    auto_insights.apply()
    anomaly = auto_insights.float_insights().reset_index(drop=False)
    anomaly = anomaly[['index', 'Зацикленность']].droplevel(0, axis=1)
    anomaly.columns.name = None
    anomaly = anomaly.rename(columns={'':'Нода',})
    return anomaly


def get_data(sampler_obj: tp.Union[Sampler, tp.Dict],
             data_type: str, 
             key: str, 
             exec_env: str='default') -> tp.Any:
    '''
    Parameters:
        sampler_obj: Объект сэмплера, содержащий данные для извлечения
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        key: Ключ или имя атрибута для извлечения конкретных данных (например, 'X', 'y')
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
    Return: 
        Содержимое sampler_obj из data_type по ключу key
    '''
    res = None
    if exec_env == 'default':
        res = getattr(sampler_obj, data_type)[key]
    elif exec_env == 'sberds':
        res = sampler_obj[data_type][key]
    return res
