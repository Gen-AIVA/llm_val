import re
import string
import numpy as np
import typing as tp
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations
from collections import OrderedDict

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import shuffle_ci_semaphore, worst_semaphore


def report_valtest_mix_answers_stability(mix_answers_metric_mean: tp.Dict[str, float],
                                         mix_answers_metric_std: pd.Series,
                                         metrics_statistics: tp.Dict[str, tp.Any],
                                         data_type: str,
                                         std_threshold: float,
                                         distrib_fig: tp.Any,
                                         is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        mix_answers_metric_mean: Словарь со средними значениями метрик
        mix_answers_metric_std: pd.Series со значениями стандартного отклонения метрик
        metrics_statistics: словарь со статистиками распределения метрики
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        std_threshold: Порог для светофора по стандартному отклонению
        distrib_fig: График распределения доли успешно восстановленных слов
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    color_list = {i: shuffle_ci_semaphore(mix_answers_metric_mean[i], 
                                          mix_answers_metric_std[i], 
                                          (metrics_statistics['left_interval'].loc[i],
                                           metrics_statistics['right_interval'].loc[i]),
                                          std_threshold) for i in mix_answers_metric_mean.keys()}
    conf_int = {i:f"[{metrics_statistics['left_interval'].loc[i]:.4f}; {metrics_statistics['right_interval'].loc[i]:.4f}]" for i in mix_answers_metric_mean.keys()}
    res_table = pd.DataFrame({f'Значение метрики на {data_type}': mix_answers_metric_mean,
                              f'Стандартное отклонение': mix_answers_metric_std,
                              f'Доверительный интервал': conf_int, 
                              f'Результат теста': color_list})
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else worst_semaphore(color_list.values())
    res_dict['result_plots'] = [distrib_fig]
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_mix_answers_stability(sampler: Sampler,
                                  scorer: Scorer,
                                  model: tp.Any,
                                  data_type: str='test',
                                  metric_agg: str='single_mean',
                                  bootstrap_iter: int=100,
                                  std_threshold: float=0.05,
                                  random_state: int=29,
                                  is_info: bool=False,
                                  mix_answers_metric_mean: tp.Optional[float]=None,
                                  mix_answers_metric_std: tp.Optional[float]=None,
                                  **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на динамику изменения целевой метрики при изменении порядков вариантов ответа
    Parameters:
        sampler: Объект класса Sampler для задачи MMLU
        scorer: Объект класса Scorer
        model: Валидируемая модель; имеет метод predict(список вопросов)
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        bootstrap_iter: Число итераций бутстрапа при построении распределения метрики
        std_threshold: Порог для светофора по стандартному отклонению
        random_state: Рандом стейт ¯\_(ツ)_/¯
        is_info: True, если тест информативный
        mix_answers_metric_mean: Опциональное значение среднего значения целевой метрики
            Если не None и mix_answers_metric_std не None, то подсчет не производится.
        mix_answers_metric_std: Опциональное значение стандартного отклонения целевой метрики
            Если не None и mix_answers_metric_mean не None, то подсчет не производится.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if (mix_answers_metric_mean is None) or (mix_answers_metric_std is None):
        data = getattr(sampler, data_type)['X'].copy()
        data['question_with_context'] = data['context'].fillna('').apply(lambda x: '. '.join(x)) + '.\n' + data['question']
        permuted_data = create_data_with_permutations(data)
        permuted_data = get_answer_from_model(permuted_data, model)
        permuted_data['single_metrics'] = (permuted_data['permut_true_answer'] == permuted_data['permut_model_answer']).astype(int)
        mix_answers_metric_mean = calculate_pointwise_metric(scorer,
                                                             metric_agg,
                                                             permuted_data['single_metrics'])
        metrics_statistics = calculate_metric_statistics(scorer,
                                                         metric_agg,
                                                         permuted_data,
                                                         bootstrap_iter, 
                                                         random_state)
        mix_answers_metric_std = metrics_statistics['distr_std']
        distrib_fig = plot_distribution_hist(metrics_statistics,
                                             mix_answers_metric_mean)
    report = report_valtest_mix_answers_stability(mix_answers_metric_mean,
                                                  mix_answers_metric_std,
                                                  metrics_statistics,
                                                  data_type,
                                                  std_threshold,
                                                  distrib_fig,
                                                  is_info)
    precomputed = {'mix_answers_metric_mean': mix_answers_metric_mean,
                   'mix_answers_metric_std': mix_answers_metric_std}
    return {'report': report,
            'precomputed': precomputed}


def create_data_with_permutations(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Создание датафрейма со всеми возможными перестановками вариантов ответа
    Parameters:
        data: Исходный набор данных
    Return:
        Датафрейм с вопросами со всеми возможными перестановками вариантов ответа
    '''
    data = data.apply(lambda row: permute_answer(row), axis=1).explode(['permut_answer_options',
                                                                        'permut_true_answer'])
    data = data.reset_index().rename(columns = {'index':'QuestionIndex'})
    return data


def get_answer_from_model(permuted_data: pd.DataFrame,
                          model: tp.Any) -> pd.DataFrame:
    '''
    Получение ответа модели для дальнейшего подсчета метрики
    Parameters:
        permuted_data: Датафрейм с вопросами со всеми возможными перестановками вариантов ответа
        model: Валидируемая модель; имеет метод predict(answers: tp.List[str])
    Return:
        Датафрейм с ответами модели
    '''
    full_questions = permuted_data.apply(lambda row: create_full_question(row), axis=1).to_list()
    permuted_data['permut_full_answers'] = model.predict(full_questions)
    permuted_data['permut_model_answer'] = permuted_data.apply(lambda row: find_option_letter(row), axis=1)
    return permuted_data


def create_full_question(row: pd.DataFrame) -> str:
    '''
    Создание вопроса с перемешанными ответами
    Parameters:
        row: Строка из датасета формата MMLU
    Return:
        Вопрос с перемешанными ответами
    '''
    question = f'{row["question_with_context"]}\nВарианты ответов:\n'
    question += '\n'.join(f'{key}) {value}' for key, value in sorted(row['permut_answer_options'].items(), key=lambda x: x[0]))
    question += '\nТвой ответ:\n'
    return question


def find_option_letter(row: pd.DataFrame) -> str:
    '''
    Создание вопроса с перемешанными ответами
    Parameters:
        full_answer: Ответ модели
    Return:
        Вопрос с перемешанными ответами
    '''
    options = ''.join(row['permut_answer_options'].keys())
    regex_for_search = f'[{options}]'
    match = re.search(regex_for_search, row['permut_full_answers'])
    return match.group(0) if match else None


def calculate_pointwise_metric(scorer: Scorer,
                               metric_agg: str,
                               single_metrics: pd.Series) -> tp.Dict[str, float]:
    '''
    Подсчет точечного значения целевой метрики
    Parameters:
        scorer: Объект класса Scorer
        metric_agg: Название метрики
        single_metrics: Колонка с посчитанной верностью ответа в разрезе вопросов
    Return:
        Словарь, где ключ - название метрики, а значение - значение метрики
    '''
    dict_for_metric = {'X': None, 'y': pd.DataFrame({'metric_value': single_metrics})}
    tmp_sampler = Sampler(test=dict_for_metric)
    metric_value_pointwise = scorer.calc(sampler=tmp_sampler,
                                         data_type='test',
                                         metric_name=metric_agg)
    return metric_value_pointwise


def calculate_metric_statistics(scorer: Scorer,
                                metric_agg: str,
                                permuted_data: pd.DataFrame,
                                bootstrap_iter: int=100,
                                random_state: int=29) -> tp.Dict[str, pd.DataFrame]:
    '''
    Подсчет распределения метрики при перемешивании вариантов ответа
    Parameters:
        scorer: Объект класса Scorer
        metric_agg: Название метрики
        permuted_data: Таблица с вопросами с перемешанными вариантами ответа и 
            посчитанной верностью ответа отдельно по вопросам
        bootstrap_iter: Число итераций бутстрапа при построении распределения метрики
        random_state: Рандом стейт ¯\_(ツ)_/¯
    Return:
        Словарь с DataFrame, в котором колонка - это название метрики
    '''
    metric_values = []
    rng = np.random.default_rng(random_state)
    data_group = permuted_data.groupby(by='QuestionIndex')
    for _ in range(bootstrap_iter):
        correct_list = data_group.sample(n=1,
                                         random_state=rng,
                                         replace=True)['single_metrics'].values
        metric_values.append(calculate_pointwise_metric(scorer,
                                                        metric_agg,
                                                        pd.Series(correct_list)))
    metric_values = pd.DataFrame(metric_values)
    distr_std = metric_values.std()
    distr_mean = metric_values.mean()
    left_interval = distr_mean - 3 * distr_std
    right_interval = distr_mean + 3 * distr_std
    return {'total_metric_list': metric_values,
            'distr_std': distr_std,
            'left_interval': left_interval,
            'right_interval': right_interval}


def plot_distribution_hist(metrics_statistics: tp.Dict[str, pd.DataFrame],
                           metric_value_pointwise: tp.Dict[str, float]) -> tp.Any:
    '''
    Построение графика распределения целевой метрики
    Parameters:
        metrics_statistics: словарь со статистиками распределения метрики
        metric_value_pointwise: точечное значение целевой метрики на исходной выборке
    Return:
        График распределения целевой метрики
    '''
    n_fig = len(metric_value_pointwise)
    fig, ax = plt.subplots(n_fig, figsize=(20, 10), squeeze=False)
    for i, name in enumerate(metric_value_pointwise):
        ax[i][0].hist(metrics_statistics['total_metric_list'][name], alpha=0.5, color = 'r') 
        ax[i][0].axvline(x=metric_value_pointwise[name], color='r')
        ax[i][0].legend([f'mean {name}', f' distribution {name}'])
        ax[i][0].axvline(x=metrics_statistics['left_interval'][name], color='b')
        ax[i][0].axvline(x=metrics_statistics['right_interval'][name], color='b')
        ax[i][0].set_title(f'Распределение {name} по результатам бутстрапирования')
        ax[i][0].set_ylabel(f'{name}')
    return fig


def permute_answer(row: pd.DataFrame) -> pd.DataFrame:
    '''
    Создание всевозможных перестановок вариантов ответов в разрезе одного вопроса
    Parameters:
        row: Исходный набор данных (одна строка)
    Return:
        Датафрейм с вопросом со всеми возможными перестановками вариантов ответа
    '''
    options = ''.join(row['answer_options'].keys())
    permut = list(permutations(options, len(options)))
    permut_ans_opt = [dict(zip(i, row['answer_options'].values())) for i in permut]
    norm_to_perm_map = [dict(zip(row['answer_options'], i))[row['true_answer']] for i in permut]
    row['permut_answer_options'] = permut_ans_opt
    row['permut_true_answer'] = norm_to_perm_map
    return row
