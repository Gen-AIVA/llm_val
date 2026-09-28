import typing as tp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import worst_semaphore, dialog_len_semaphore


def report_valtest_metric_by_dialog_len(stats_traces: tp.Dict[str, float],
                                        stats_dialogue: tp.Dict[str, float],
                                        data_type: str='test',
                                        significance_level: float=0.05,
                                        mean_decrease_threshold: float=0.3,
                                        greater_is_better: bool=True,
                                        is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а в соответствии с концепцией llm_val
    Parameters:
        stats_traces: Статистики проверки по токенам (p_value, subset_mean, rest_mean)
        stats_dialogue: Статистики проверки по репликам (p_value, subset_mean, rest_mean)
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        significance_level: Уровень значимости для выставления светофора
        mean_decrease_threshold: Относительная граница падения средней КМ на подвыборке
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    color_traces = dialog_len_semaphore(stats_traces, significance_level, mean_decrease_threshold, greater_is_better)
    color_dialogue = dialog_len_semaphore(stats_dialogue, significance_level, mean_decrease_threshold, greater_is_better)
    color = 'gray' if is_info else worst_semaphore([color_traces, color_dialogue])
    result_df = pd.DataFrame(
        {'p-value': [stats_traces['p_value'], stats_dialogue['p_value']],
         f'Средняя КМ на подвыборке ({data_type})': [stats_traces['subset_mean'], stats_dialogue['subset_mean']],
         f'Средняя КМ на остальных диалогах ({data_type})': [stats_traces['rest_mean'], stats_dialogue['rest_mean']],
         'Результат теста': [color_traces, color_dialogue]},
        index=['Проверка по длине в токенах', 'Проверка по числу реплик'])
    figure = plot_metric_by_dialog_len(stats_traces, stats_dialogue, data_type)
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = [figure]
    res_dict['result_dataframes'] = [result_df]
    return res_dict


def valtest_metric_by_dialog_len(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                 data_type: str='test',
                                 main_column: tp.Optional[str]=None,
                                 significance_level: float=0.05,
                                 mean_decrease_threshold: float=0.3,
                                 greater_is_better: bool=True,
                                 is_info: bool=False,
                                 stats_traces: tp.Optional[tp.Dict[str, float]]=None,
                                 stats_dialogue: tp.Optional[tp.Dict[str, float]]=None,
                                 exec_env: tp.Literal['default', 'sberds']='default',
                                 **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест динамики ключевой метрики в зависимости от длины истории диалога: сравнивает среднюю КМ на 10% самых длинных диалогов со средней КМ на остальных диалогах с помощью одностороннего двухвыборочного t-теста Уэлча отдельно по длине в токенах и по числу реплик.
    Parameters:
        sampler: Объект класса Sampler (exec_env='default') или словарь с данными (exec_env='sberds')
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        main_column: Название колонки с ключевой метрикой качества в y; если None, усредняются все колонки y
        significance_level: Уровень значимости для выставления светофора
        mean_decrease_threshold: Относительная граница падения средней КМ на подвыборке
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
        stats_traces: Опциональные статистики проверки по токенам из precomputed
        stats_dialogue: Опциональные статистики проверки по репликам из precomputed
        exec_env: Среда выполнения ('default' или 'sberds')
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if (stats_traces is None) or (stats_dialogue is None):
        data = sampler[data_type] if exec_env == 'sberds' else getattr(sampler, data_type)
        if 'session_id' in data['X'].columns:
            data['X'] = data['X'].groupby('session_id', as_index=False).nth(0)
        metric_per_object = data['y'][main_column] if main_column else data['y'].mean(axis=1)
    if stats_traces is None:
        token_lengths = data['traces'].groupby('session_id')['llm_total_tokens'].sum()
        object_token_lengths = data['X']['session_id'].map(token_lengths)
        stats_traces = compute_subsample_stats(object_token_lengths, metric_per_object, greater_is_better)
    if stats_dialogue is None:
        dialogue_lengths = data['X']['dialogue'].apply(len)
        stats_dialogue = compute_subsample_stats(dialogue_lengths, metric_per_object, greater_is_better)
    precomputed = {'stats_traces': stats_traces,
                   'stats_dialogue': stats_dialogue}
    report = report_valtest_metric_by_dialog_len(stats_traces=stats_traces,
                                                 stats_dialogue=stats_dialogue,
                                                 data_type=data_type,
                                                 significance_level=significance_level,
                                                 mean_decrease_threshold=mean_decrease_threshold,
                                                 greater_is_better=greater_is_better,
                                                 is_info=is_info)
    return {'report': report,
            'precomputed': precomputed}


def compute_subsample_stats(lengths: pd.Series,
                            metric_per_object: pd.Series,
                            greater_is_better: bool=True) -> tp.Dict[str, float]:
    '''
    Подсчет статистик проверки: средняя КМ на подвыборке 10% самых длинных диалогов (но не менее 50), средняя КМ на остальных диалогах и p-value одностороннего двухвыборочного t-теста Уэлча на отклонение среднего подвыборки от среднего остальных диалогов в сторону ухудшения
    Parameters:
        lengths: Series с длинами диалогов в разрезе объектов
        metric_per_object: Series со значениями ключевой метрики качества в разрезе объектов
        greater_is_better: True, если чем больше метрика, тем лучше
    Return:
        Словарь с ключами p_value, subset_mean, rest_mean
    '''
    aligned_lengths = lengths.dropna()
    n_total = len(aligned_lengths)
    n_subset = min(max(50, int(np.ceil(0.1 * n_total))), n_total)
    longest_index = aligned_lengths.nlargest(n_subset).index
    rest_index = aligned_lengths.index.difference(longest_index)
    subset_metric = metric_per_object.loc[longest_index]
    rest_metric = metric_per_object.loc[rest_index]
    alternative = 'less' if greater_is_better else 'greater'
    p_value = ttest_ind(subset_metric, rest_metric, equal_var=False, alternative=alternative).pvalue
    return {'p_value': float(p_value),
            'subset_mean': float(subset_metric.mean()),
            'rest_mean': float(rest_metric.mean())}


def plot_metric_by_dialog_len(stats_traces: tp.Dict[str, float],
                              stats_dialogue: tp.Dict[str, float],
                              data_type: str='test') -> tp.Any:
    '''
    Отрисовка сравнения средней КМ на подвыборке и на всем датасете для двух проверок
    Parameters:
        stats_traces: Статистики проверки по токенам
        stats_dialogue: Статистики проверки по репликам
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    Return:
        Фигура с барчартом сравнения средних значений ключевой метрики
    '''
    checks = ['По длине в токенах', 'По числу реплик']
    subset_means = [stats_traces['subset_mean'], stats_dialogue['subset_mean']]
    rest_means = [stats_traces['rest_mean'], stats_dialogue['rest_mean']]
    positions = np.arange(len(checks))
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.bar(positions - 0.2, subset_means, width=0.4, label='Подвыборка самых длинных')
    axis.bar(positions + 0.2, rest_means, width=0.4, label='Остальные диалоги')
    axis.set_xticks(positions)
    axis.set_xticklabels(checks)
    axis.set_title(f'Средняя КМ на подвыборке самых длинных диалогов и на остальных диалогах ({data_type})',
                   fontsize=12, fontweight='bold')
    axis.set_ylabel('Среднее значение ключевой метрики', fontsize=10)
    axis.set_xlabel('Тип проверки длины диалога', fontsize=10)
    axis.legend()
    return fig
