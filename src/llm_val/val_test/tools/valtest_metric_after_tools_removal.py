import numpy as np
import pandas as pd
import typing as tp
from tqdm import tqdm
from scipy import stats
from rouge import Rouge
from copy import deepcopy

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.metrics_rag import cosine_distance
from llm_val.utils.report_helper import worst_semaphore
from llm_val.utils.report_helper import semaphore_by_threshold


def report_valtest_metric_after_tools_removal(metric_values: tp.Dict[str, float],
                                              semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]]={'cosine_similarity': (0.75, 0.9),
                                                                                                         'rougeL': (0.5, 0.6),
                                                                                                         'p_value': (0.01, 0.05)},
                                              check_correct_answer: tp.Optional[tp.Callable[[str, tp.List[str]], bool]]=None,
                                              stat_checker: bool=True,
                                              detailed_table: tp.Optional[tp.List[tp.Dict]]=None,
                                              is_info: bool=False,
                                              **kwargs) -> tp.Dict[str, tp.Any]:
    """
    Создание report'а для теста с удалением инструментов
    Parameters:
        metric_values: Словарь с метриками и их значениями
            Для check_correct_answer=True требуется:
                - baseline_mean: среднее базовое качество
                - removal_mean: среднее качество после удаления инструментов
                - p_value_means: p-value для сравнения средних
                - p_value_variance: p-value для сравнения дисперсий
            Для check_correct_answer=False и stat_checker=True требуется:
                - cosine_similarity: косинусная близость
                - rougeL: метрика Rouge-L
                - p_value_cosine_means: p-value для средних косинусной близости
                - p_value_cosine_variance: p-value для дисперсий косинусной близости
                - p_value_rouge_means: p-value для средних Rouge-L
                - p_value_rouge_variance: p-value для дисперсий Rouge-L
            Для check_correct_answer=False и stat_checker=False требуется:
                - cosine_similarity: косинусная близость
                - rougeL: метрика Rouge-L
        semaphore_threshold: Словарь с границами светофора для каждой метрики
            Должен содержать соответствующие ключи в зависимости от режима:
            - Для p-value метрик: 'p_value' с порогами (жёлтый, красный)
            - Для метрик качества: 'cosine_similarity' и 'rougeL' с порогами (жёлтый, красный)
        check_correct_answer: Функция проверки корректности ответа модели
        stat_checker: Параметр для выбора способа выставления светофора, по умолчанию используется статистический подход (светофор по p_value)
        detailed_table: Детальная таблица результатов для режима stat_checker=False
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    """
    result_dataframes = []
    color_list = {}
    semaphore_color = 'gray'
    result_data = []

    if check_correct_answer:
        p_value_means = metric_values.get('p_value_means')
        p_value_variance = metric_values.get('p_value_variance')
        color_list['p_value_means'] = semaphore_by_threshold(value=p_value_means,
                                                             threshold=semaphore_threshold.get('p_value'),
                                                             greater_is_better=False,
                                                             left_border_is_bigger=False,
                                                             right_border_is_bigger=False)
        color_list['p_value_variance'] = semaphore_by_threshold(value=p_value_variance,
                                                                threshold=semaphore_threshold.get('p_value'),
                                                                greater_is_better=False,
                                                                left_border_is_bigger=False,
                                                                right_border_is_bigger=False)
        
        result_data = [{'Метрика': 'Базовое качество', 'Значение': f"{metric_values['baseline_mean']:.3f}"},
                       {'Метрика': 'Качество после удаления', 'Значение': f"{metric_values['removal_mean']:.3f}"},
                       {'Метрика': 'p-value (средние)', 'Значение': f"{p_value_means:.3f}"},
                       {'Метрика': 'p-value (дисперсии)', 'Значение': f"{p_value_variance:.3f}"}]
    else:
        if stat_checker:
            p_value_cosine_means = metric_values.get('p_value_cosine_means')
            p_value_cosine_variance = metric_values.get('p_value_cosine_variance')
            p_value_rouge_means = metric_values.get('p_value_rouge_means')
            p_value_rouge_variance = metric_values.get('p_value_rouge_variance')
            color_list['p_value_cosine_means'] = semaphore_by_threshold(value=p_value_cosine_means,
                                                                        threshold=semaphore_threshold.get('p_value'),
                                                                        greater_is_better=False,
                                                                        left_border_is_bigger=False,
                                                                        right_border_is_bigger=False)
            color_list['p_value_cosine_variance'] = semaphore_by_threshold(value=p_value_cosine_variance,
                                                                           threshold=semaphore_threshold.get('p_value'),
                                                                           greater_is_better=False,
                                                                           left_border_is_bigger=False,
                                                                           right_border_is_bigger=False)
            color_list['p_value_rouge_means'] = semaphore_by_threshold(value=p_value_rouge_means,
                                                                       threshold=semaphore_threshold.get('p_value'),
                                                                       greater_is_better=False,
                                                                       left_border_is_bigger=False,
                                                                       right_border_is_bigger=False)
            color_list['p_value_rouge_variance'] = semaphore_by_threshold(value=p_value_rouge_variance,
                                                                          threshold=semaphore_threshold.get('p_value'),
                                                                          greater_is_better=False,
                                                                          left_border_is_bigger=False,
                                                                          right_border_is_bigger=False)
            
            result_data = [{'Метрика': 'Косинусная близость', 'Значение': f"{metric_values['cosine_similarity']:.3f}"},
                           {'Метрика': 'Rouge-L', 'Значение': f"{metric_values['rougeL']:.3f}"},
                           {'Метрика': 'p-value (cosine средние)', 'Значение': f"{p_value_cosine_means:.3f}"},
                           {'Метрика': 'p-value (cosine дисперсии)', 'Значение': f"{p_value_cosine_variance:.3f}"},
                           {'Метрика': 'p-value (rouge средние)', 'Значение': f"{p_value_rouge_means:.3f}"},
                           {'Метрика': 'p-value (rouge дисперсии)', 'Значение': f"{p_value_rouge_variance:.3f}"}]
        else:
            cosine_similarity = metric_values['cosine_similarity']
            rougeL = metric_values['rougeL']            
            if detailed_table:
                cosine_yellow, cosine_red = semaphore_threshold['cosine_similarity']
                rouge_yellow, rouge_red = semaphore_threshold['rougeL']
                semaphore_colors = []
                for row in detailed_table:
                    mean_cosine = row['mean_cosine']
                    mean_rouge = row['mean_rouge']
                    if mean_cosine > cosine_red and mean_rouge > rouge_red:
                        semaphore_color = 'red'
                    elif mean_cosine > cosine_yellow and mean_rouge > rouge_yellow:
                        semaphore_color = 'yellow'
                    else:
                        semaphore_color = 'green'

                    row['Результат теста'] = semaphore_color
                    semaphore_colors.append(semaphore_color)

                semaphore_color = worst_semaphore(semaphore_colors)
                display_table = []
                for row in detailed_table:
                    display_table.append({'Удаляются': row['Удаляются'],
                                          'Косинусная близость': row['Косинусная близость'],
                                          'ROUGE-L': row['ROUGE-L'],
                                          'Результат теста': row['Результат теста']})
                detailed_df = pd.DataFrame(display_table)
                final_row = pd.DataFrame([{'Удаляются': 'Итог',
                                           'Косинусная близость': '',
                                           'ROUGE-L': '',
                                           'Результат теста': semaphore_color}])
                detailed_df = pd.concat([detailed_df, final_row], ignore_index=True)
                result_dataframes.append(detailed_df)

            result_data = [{'Метрика': 'Средняя косинусная близость', 'Значение': f"{cosine_similarity:.3f}"},
                           {'Метрика': 'Средний Rouge-L', 'Значение': f"{rougeL:.3f}"},
                           {'Метрика': 'Итоговый результат', 'Значение': 'По худшей комбинации'}]
            if is_info:
                semaphore_color = 'gray'
            result_data.append({'Метрика': 'Результат', 'Значение': semaphore_color})
            result_df = pd.DataFrame(result_data)
            result_dataframes.append(result_df)

    if (check_correct_answer or stat_checker):
        if is_info:
            semaphore_color = 'gray'
        else:
            semaphore_color = worst_semaphore(list(color_list.values()))
        
        result_data.append({'Метрика': 'Результат', 'Значение': semaphore_color})
        result_df = pd.DataFrame(result_data)
        result_dataframes.append(result_df)

    res_dict = {}
    res_dict['semaphore'] = semaphore_color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = result_dataframes
    return res_dict


def valtest_metric_after_tools_removal(sampler: Sampler,
                                       model: tp.Any,
                                       removal_mode: str='sequential',
                                       custom_tools_list: tp.Optional[tp.List[tp.List[tp.Any]]]=None,
                                       removal_num: int=2,
                                       get_embed_func: tp.Optional[tp.Callable]=None,
                                       check_correct_answer: tp.Optional[tp.Callable[[str, tp.List[str]], bool]]=None,
                                       data_type: str='test',
                                       semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]]={'cosine_similarity': (0.75, 0.9),
                                                                                                  'rougeL': (0.5, 0.6),
                                                                                                  'p_value': (0.01, 0.05)},
                                       stat_checker: bool=True,
                                       is_info: bool=False,
                                       **kwargs) -> tp.Dict[str, tp.Any]:
    """
    Тест на анализ изменения ответа LLM при удалении инструментов
    Parameters:
        sampler: Объект класса Sampler
        model: Валидируемый агент; имеет метод predict(список вопросов), метод tools со списком инструментов,
                метод get_params, возвращающий текущие параметры агента, и метод set_params, позволяющий задавать параметры агента
        removal_mode: Режим удаления инструментов: 
            - 'sequential' - последовательное удаление по одному
            - 'custom' - использование пользовательского списка инструментов
            - 'random' - случайное удаление инструментов
        custom_tools_list: Пользовательский список наборов инструментов для режима 'custom'
        removal_num: Количество запусков для каждого набора инструментов
        get_embed_func: Функция для получения эмбеддингов
        check_correct_answer: Функция проверки корректности ответа модели
        data_type: Тип данных для расчета метрики
        semaphore_threshold: Словарь с границами светофора для каждой метрики
            Должен содержать соответствующие ключи в зависимости от режима:
            - Для p-value метрик: 'p_value' с порогами (жёлтый, красный)
            - Для метрик качества: 'cosine_similarity' и 'rougeL' с порогами (жёлтый, красный)
        stat_checker: Параметр для выбора способа выставления светофора, по умолчанию используется статистический подход (светофор по p_value)
        is_info: True, если тест информативный
    Returns:
        report в соответствии с концепцией llm_val
    """
    data = deepcopy(getattr(sampler, data_type))
    questions = data['X']['question']
    original_params = model.get_params()
    original_tools = original_params.get('tools', [])
    
    if check_correct_answer:
        metric_values, precomputed = run_removal_with_correct_answer_check(model,
                                                                           questions,
                                                                           original_tools,
                                                                           removal_mode,
                                                                           custom_tools_list,
                                                                           removal_num,
                                                                           check_correct_answer)
        detailed_table = None
    else:
        metric_values, precomputed, detailed_table = run_removal_with_embedding_rouge(model,
                                                                                      questions,
                                                                                      original_tools,
                                                                                      removal_mode,
                                                                                      custom_tools_list,
                                                                                      removal_num,
                                                                                      get_embed_func,
                                                                                      stat_checker,
                                                                                      semaphore_threshold)

    report = report_valtest_metric_after_tools_removal(metric_values=metric_values,
                                                       semaphore_threshold=semaphore_threshold,
                                                       check_correct_answer=check_correct_answer,
                                                       removal_num=removal_num,
                                                       stat_checker=stat_checker,
                                                       is_info=is_info,
                                                       detailed_table=detailed_table)
    
    model.set_params(**original_params)
    return {'report': report, 
            'precomputed': precomputed}


def generate_tools_sets(original_tools: tp.List[tp.Any], 
                        removal_mode: str,
                        custom_tools_list: tp.Optional[tp.List[tp.List[tp.Any]]]=None) -> tp.List[tp.List[tp.Any]]:
    """
    Генерация наборов инструментов для тестирования
    Parameters:
        original_tools: Оригинальный список инструментов
        removal_mode: Режим удаления инструментов
        custom_tools_list: Пользовательский список наборов инструментов
    Returns:
        Список наборов инструментов для тестирования
    """
    tools_sets = []
    if removal_mode == 'custom' and custom_tools_list is not None:
        tools_sets = custom_tools_list
    elif removal_mode == 'sequential':
        for i in range(len(original_tools)):
            tools_set = original_tools.copy()
            del tools_set[i]
            tools_sets.append(tools_set)
    elif removal_mode == 'random':
        n_tools = len(original_tools)
        if n_tools > 1:
            for i in range(1, n_tools):
                tools_set = original_tools.copy()
                indices_to_remove = np.random.choice(range(n_tools), size=i, replace=False)
                for idx in sorted(indices_to_remove, reverse=True):
                    del tools_set[idx]
                tools_sets.append(tools_set)
        else:
            tools_sets.append([])
    return tools_sets


def run_removal_with_correct_answer_check(model: tp.Any, 
                                          questions: tp.List[str], 
                                          original_tools: tp.List[tp.Any],
                                          removal_mode: str,
                                          custom_tools_list: tp.Optional[tp.List[tp.List[tp.Any]]],
                                          removal_num: int,
                                          check_correct_answer: tp.Optional[tp.Callable[[str, tp.List[str]], bool]]) -> tp.Tuple[tp.Dict[str, float], tp.Dict[str, float]]:
    """
    Функция для запуска теста с проверкой корректности ответов с помощью функции check_correct_answer (автометрики)
    Parameters:
        model: Валидируемый агент; имеет метод predict(список вопросов), метод tools со списком инструментов,
                метод get_params, возвращающий текущие параметры агента, и метод set_params, позволяющий задавать параметры агента
        questions: Список вопросов для тестирования
        original_tools: Оригинальный список инструментов модели
        removal_mode: Режим удаления инструментов: 
            - 'sequential' - последовательное удаление по одному
            - 'custom' - использование пользовательского списка инструментов
            - 'random' - случайное удаление инструментов
        custom_tools_list: Пользовательский список наборов инструментов для режима 'custom'
        removal_num: Количество запусков для каждого набора инструментов
        check_correct_answer: Функция проверки корректности ответа модели для подсчета автометрики    
    Returns:
        Кортеж из словарей:
        - metric_values: словарь с метриками и p-значениями статистических тестов
        - precomputed: словарь с предвычисленными основными метриками
    """
    tools_sets = generate_tools_sets(original_tools, 
                                     removal_mode, 
                                     custom_tools_list)
    baseline_metrics = []
    for _ in tqdm(range(removal_num), desc='Baseline runs (full tools)'):
        model.set_params(tools=original_tools)
        baseline_answers = model.predict(questions)
        run_metrics = []
        for i, question in enumerate(questions):
            correctness = float(check_correct_answer(question, [baseline_answers[i]]))
            run_metrics.append(correctness)
        baseline_metrics.append(np.mean(run_metrics))
    
    removal_metrics = []
    
    for tools_set in tqdm(tools_sets, desc='Tools removal sets'):
        set_metrics = []
        for _ in range(removal_num):
            model.set_params(tools=tools_set)
            removal_answers = model.predict(questions)
            
            run_metrics = []
            for i, question in enumerate(questions):
                correctness = float(check_correct_answer(question, [removal_answers[i]]))
                run_metrics.append(correctness)
            
            set_metrics.append(np.mean(run_metrics))
        
        removal_metrics.extend(set_metrics)

    p_value_means, p_value_variance = calculate_p_values(baseline_metrics, 
                                                         removal_metrics)
    metric_values = {'baseline_mean': float(np.mean(baseline_metrics)),
                     'removal_mean': float(np.mean(removal_metrics)),
                     'p_value_means': p_value_means,
                     'p_value_variance': p_value_variance}
    
    precomputed = {'baseline_mean': float(np.mean(baseline_metrics)),
                   'removal_mean': float(np.mean(removal_metrics))}
    return (metric_values, 
            precomputed)


def run_removal_with_embedding_rouge(model: tp.Any,
                                     questions: tp.List[str],
                                     original_tools: tp.List[tp.Any],
                                     removal_mode: str,
                                     custom_tools_list: tp.Optional[tp.List[tp.List[tp.Any]]],
                                     removal_num: int,
                                     get_embed_func: tp.Optional[tp.Callable],
                                     stat_checker: bool=True,
                                     semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]]={'cosine_similarity': (0.75, 0.9),
                                                                                                'rougeL': (0.5, 0.6),
                                                                                                'p_value': (0.01, 0.05)}) -> tp.Tuple[tp.Dict[str, float], tp.Dict[str, float], tp.Optional[tp.List[tp.Dict]]]:
    """
    Функция для запуска теста с проверкой корректности ответов с помощью функции get_embed_func (эмбеддинги)
    Parameters:
        model: Валидируемый агент; имеет метод predict(список вопросов), метод tools со списком инструментов,
               метод get_params возвращающий текущие параметры агента и метод set_params позволяющий задавать параметры агента.
        questions: Список вопросов для тестирования
        original_tools: Оригинальный список инструментов модели
        removal_mode: Режим удаления инструментов: 
            - 'sequential' - последовательное удаление по одному
            - 'custom' - использование пользовательского списка инструментов
            - 'random' - случайное удаление инструментов
        custom_tools_list: Пользовательский список наборов инструментов для режима 'custom'
        removal_num: Количество запусков для каждого набора инструментов
        stat_checker: Параметр для выбора способа выставления светофора, по умолчанию используется статистический подход (светофор по p_value)
        get_embed_func: Функция для получения эмбеддингов
        semaphore_threshold: Словарь с границами светофора для каждой метрики
            Должен содержать соответствующие ключи в зависимости от режима:
            - Для p-value метрик: 'p_value' с порогами (жёлтый, красный)
            - Для метрик качества: 'cosine_similarity' и 'rougeL' с порогами (жёлтый, красный)
    Returns:
        Кортеж из двух словарей:
        - metric_values: словарь с метриками и p-значениями статистических тестов
        - precomputed: словарь с предвычисленными основными метриками
        - detailed_results: детальная таблица результатов для stat_checker=False или None.
    """
    rouge_scorer_obj = Rouge()
    tools_sets = generate_tools_sets(original_tools,
                                     removal_mode, 
                                     custom_tools_list)
    
    model.set_params(tools=original_tools)
    baseline_answers = model.predict(questions)
    baseline_embeddings_raw = [get_embed_func([answer]) for answer in baseline_answers]
    baseline_embeddings = [embed[0][0] for embed in baseline_embeddings_raw]
    baseline_cosine_sims = []
    baseline_rouge_scores = []
    
    for _ in tqdm(range(removal_num), desc='Baseline runs (full tools)'):
        current_answers = model.predict(questions)
        cos_sims = []
        rouge_scores = []
        for i, (base_ans, curr_ans) in enumerate(zip(baseline_answers, current_answers)):
            curr_embed_raw = get_embed_func([curr_ans])
            curr_embed = curr_embed_raw[0][0]
            cos_dist = cosine_distance(baseline_embeddings[i], curr_embed)
            if hasattr(cos_dist, '__len__') and not isinstance(cos_dist, (int, float)):
                cos_dist = np.mean(cos_dist)
            cos_sim = 1 - cos_dist
            if np.isnan(cos_sim) or np.isinf(cos_sim):
                cos_sim = 0.0
            cos_sims.append(float(cos_sim))
            rouge_score_val = rouge_scorer_obj.get_scores(curr_ans, base_ans)[0]['rouge-l']['f']
            rouge_scores.append(rouge_score_val)
        
        baseline_cosine_sims.append(np.mean(cos_sims))
        baseline_rouge_scores.append(np.mean(rouge_scores))
    
    removal_cosine_sims = []
    removal_rouge_scores = []
    detailed_results = [] if not stat_checker else None
    
    for tools_set in tqdm(tools_sets, desc='Tools removal sets'):
        set_cosine_sims = []
        set_rouge_scores = []
        model.set_params(tools=tools_set)
        for _ in range(removal_num):
            removal_answers = model.predict(questions)
            
            cos_sims = []
            rouge_scores = []
            for i, (base_ans, removal_ans) in enumerate(zip(baseline_answers, removal_answers)):
                removal_embed_raw = get_embed_func([removal_ans])
                removal_embed = removal_embed_raw[0][0]
                
                cos_dist = cosine_distance(baseline_embeddings[i], removal_embed)
                if hasattr(cos_dist, '__len__') and not isinstance(cos_dist, (int, float)):
                    cos_dist = np.mean(cos_dist)
                
                cos_sim = 1 - cos_dist
                if np.isnan(cos_sim) or np.isinf(cos_sim):
                    cos_sim = 0.0
                cos_sims.append(float(cos_sim))
                
                rouge_score_val = rouge_scorer_obj.get_scores(removal_ans, base_ans)[0]['rouge-l']['f']
                rouge_scores.append(rouge_score_val)
            
            set_cosine_sims.append(np.mean(cos_sims))
            set_rouge_scores.append(np.mean(rouge_scores))
        
        if not stat_checker:
            mean_cosine = np.mean(set_cosine_sims)
            mean_rouge = np.mean(set_rouge_scores)
            
            removed_count = len(original_tools) - len(tools_set)
            if removed_count == 0:
                removal_description = "Все инструменты"
            else:
                tools_names = [f"Инструмент_{i+1}" for i in range(len(original_tools)) if original_tools[i] not in tools_set]
                removal_description = " и ".join(tools_names)
            
            detailed_results.append({'Удаляются': removal_description,
                                     'Косинусная близость': f"{mean_cosine:.3f}",
                                     'ROUGE-L': f"{mean_rouge:.3f}",
                                     'mean_cosine': mean_cosine,
                                     'mean_rouge': mean_rouge})
            
        removal_cosine_sims.extend(set_cosine_sims)
        removal_rouge_scores.extend(set_rouge_scores)
    
    p_value_cosine_means, p_value_cosine_variance = calculate_p_values(baseline_cosine_sims, 
                                                                       removal_cosine_sims)
    p_value_rouge_means, p_value_rouge_variance = calculate_p_values(baseline_rouge_scores, 
                                                                     removal_rouge_scores)

    metric_values = {'cosine_similarity': float(np.mean(removal_cosine_sims)),
                     'rougeL': float(np.mean(removal_rouge_scores)),
                     'p_value_cosine_means': p_value_cosine_means,
                     'p_value_cosine_variance': p_value_cosine_variance,
                     'p_value_rouge_means': p_value_rouge_means,
                     'p_value_rouge_variance': p_value_rouge_variance}
    
    precomputed = {'cosine_similarity': float(np.mean(removal_cosine_sims)),
                   'rougeL': float(np.mean(removal_rouge_scores))}
    
    return (metric_values, 
            precomputed, 
            detailed_results)


def calculate_p_values(baseline_metrics: tp.List[float], 
                       removal_metrics: tp.List[float]) -> tp.Tuple[float, float]:
    """
    Функция вычисления p-value для сравнения двух распределений
    Parameters:
        baseline_metrics: Метрики базового запуска
        removal_metrics: Метрики запуска с удаленными инструментами
    Returns:
        Кортеж (p_value_means, p_value_variance)
    """
    if np.std(baseline_metrics) <= 1e-10:
        _, p_value_means = stats.ttest_ind(baseline_metrics, removal_metrics)
        p_value_variance = 0.0
    else:
        _, p_value_means = stats.ttest_ind(baseline_metrics, removal_metrics)
        _, p_value_variance = stats.levene(baseline_metrics, removal_metrics)
    
    return (float(p_value_means), 
            float(p_value_variance))