import pandas as pd
import typing as tp

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.get_data import get_data
from llm_val.utils.traces import prepare_traces
from llm_val.utils.traces import calc_loop_statistic as calc_loop_statistic
from llm_val.utils.traces import calc_time_statistic as calc_time_statistic


def report_valtest_loop_duration_analysis(result_dataframe: pd.DataFrame) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'a
    Parameters:
        result_dataframe: датафрейм с рассчитанными по каждой ноде из трейсов статистиками по времени работы
                          и мере зацикленности
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_dict: tp.Dict[str, tp.Any] = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [result_dataframe]
    return res_dict


def valtest_loop_duration_analysis(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                   data_type: str='test',
                                   exec_env: str='default',
                                   loop_time_stats: tp.Optional[pd.DataFrame]=None,
                                   **kwargs: tp.Any) -> tp.Dict[str, tp.Any]:
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
        traces = prepare_traces(get_data(sampler, data_type, 'traces', exec_env))
        times_statistics = calc_time_statistic(traces)
        loop_statistics = calc_loop_statistic(traces)
        loop_time_stats = times_statistics.merge(loop_statistics, on='Нода', how='left')
        loop_time_stats = loop_time_stats.rename(columns={'В начало': 'Зацикленность в начало',
                                                         'В себя': 'Зацикленность в себя',
                                                         '«Возврат»': 'Зацикленность "Возврат"',
                                                         '«Пинг-Понг»': 'Зацикленность "Пинг-Понг"',
                                                         'В произвольную операцию': 'Зацикленность в прозвольную ноду'})
        metrics_columns = loop_time_stats.select_dtypes(['float64', 'float32']).columns
        for column in metrics_columns:
            loop_time_stats[column] = loop_time_stats[column].map('{:,.2f}'.format)
    report = report_valtest_loop_duration_analysis(loop_time_stats)
    precomputed = {'loop_time_stats': loop_time_stats}
    return {'report': report,
            'precomputed': precomputed}
