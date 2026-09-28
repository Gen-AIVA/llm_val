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


def report_valtest_metric_after_tools_refine(metric_values: tp.Dict[str, float],
                                             semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]]={'cosine_similarity': (0.75, 0.9),
                                                                                                        'rougeL': (0.5, 0.6),
                                                                                                        'p_value': (0.01, 0.05)},
                                             check_correct_answer: tp.Optional[tp.Callable[[str, tp.List[str]], bool]]=None,
                                             is_info: bool=False,
                                             **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а 
    Parameters:
        metric_values: Словарь с метриками и их значениями
            Для check_correct_answer=True требуется:
                - baseline_mean: среднее базовое качество
                - refine_mean: среднее качество после уточнения описаний
                - p_value_wilcoxon: p-value теста Уилкоксона на разностях
                - mean_difference: средняя разность качества
            Для check_correct_answer=False требуется:
                - cosine_similarity: косинусная близость
                - rougeL: метрика Rouge-L
                - p_value_cosine_wilcoxon: p-value теста Уилкоксона для косинусной близости
                - p_value_rouge_wilcoxon: p-value теста Уилкоксона для Rouge-L
                - mean_cosine_difference: средняя разность косинусной близости
                - mean_rouge_difference: средняя разность Rouge-L
        semaphore_threshold: Словарь с границами светофора для каждой метрики
        check_correct_answer: Функция проверки корректности ответа модели для подсчета автометрики
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    result_dataframes = []
    color_list = {}
    semaphore_color = 'gray'
    result_data = []
    if check_correct_answer:
        p_value_wilcoxon = metric_values.get('p_value_wilcoxon')
        mean_difference = metric_values.get('mean_difference')
        
        color_list['p_value_wilcoxon'] = semaphore_by_threshold(value=p_value_wilcoxon,
                                                                threshold=semaphore_threshold.get('p_value'),
                                                                greater_is_better=False,
                                                                left_border_is_bigger=False,
                                                                right_border_is_bigger=False)
        
        result_data = [{'Метрика': 'Базовое качество', 'Значение': f"{metric_values['baseline_mean']:.3f}"},
                       {'Метрика': 'Качество после уточнения', 'Значение': f"{metric_values['refine_mean']:.3f}"},
                       {'Метрика': 'Средняя разность', 'Значение': f"{mean_difference:.3f}"},
                       {'Метрика': 'p-value (Уилкоксон)', 'Значение': f"{p_value_wilcoxon:.3f}"}]
    else:
        p_value_cosine_wilcoxon = metric_values.get('p_value_cosine_wilcoxon')
        p_value_rouge_wilcoxon = metric_values.get('p_value_rouge_wilcoxon')
        mean_cosine_difference = metric_values.get('mean_cosine_difference')
        mean_rouge_difference = metric_values.get('mean_rouge_difference')
        
        color_list['p_value_cosine_wilcoxon'] = semaphore_by_threshold(value=p_value_cosine_wilcoxon,
                                                                       threshold=semaphore_threshold.get('p_value'),
                                                                       greater_is_better=False,
                                                                       left_border_is_bigger=False,
                                                                       right_border_is_bigger=False)
        color_list['p_value_rouge_wilcoxon'] = semaphore_by_threshold(value=p_value_rouge_wilcoxon,
                                                                      threshold=semaphore_threshold.get('p_value'),
                                                                      greater_is_better=False,
                                                                      left_border_is_bigger=False,
                                                                      right_border_is_bigger=False)
        
        result_data = [{'Метрика': 'Косинусная близость', 'Значение': f"{metric_values['cosine_similarity']:.3f}"},
                       {'Метрика': 'Rouge-L', 'Значение': f"{metric_values['rougeL']:.3f}"},
                       {'Метрика': 'Средняя разность (cosine)', 'Значение': f"{mean_cosine_difference:.3f}"},
                       {'Метрика': 'Средняя разность (rouge)', 'Значение': f"{mean_rouge_difference:.3f}"},
                       {'Метрика': 'p-value (cosine Уилкоксон)', 'Значение': f"{p_value_cosine_wilcoxon:.3f}"},
                       {'Метрика': 'p-value (rouge Уилкоксон)', 'Значение': f"{p_value_rouge_wilcoxon:.3f}"}]

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


def valtest_metric_after_tools_refine(sampler: Sampler,
                                      model: tp.Any,
                                      rephrase_model: tp.Any,
                                      rephrase_tool_description: tp.Callable,
                                      get_embed_func: tp.Optional[tp.Callable]=None,
                                      check_correct_answer: tp.Optional[tp.Callable[[str, tp.List[str]], bool]]=None,
                                      data_type: str='test',
                                      semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]]={'cosine_similarity': (0.75, 0.9), 
                                                                                                 'rougeL': (0.5, 0.6),
                                                                                                 'p_value': (0.01, 0.05)},
                                      is_info: bool=False,
                                      **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на анализ изменения ответа LLM при изменении описаний инструментов
    Parameters:
        sampler: Объект класса Sampler
        model: Валидируемый агент
        rephrase_model: Модель для перефразирования описаний инструментов
        rephrase_tool_description: Функция для перефразирования описания инструмента, ее параметры:
         - rephrase_model: Модель для перефразирования
         - original_description: Оригинальное описание инструмента 
        get_embed_func: Функция для получения эмбеддингов
        check_correct_answer: Функция проверки корректности ответа модели
        data_type: Тип данных, на которых производится расчет метрики
        semaphore_threshold: Границы светофоров
        is_info: True, если тест информативный
    Returns:
        report в соответствии с концепцией llm_val
    '''
    data = deepcopy(getattr(sampler, data_type))
    questions = data['X']['question']
    original_params = model.get_params()
    original_tools = deepcopy(original_params.get('tools', []))
    
    if check_correct_answer:
        metric_values, precomputed = run_with_correct_answer_check_refine(model,
                                                                          rephrase_model,
                                                                          questions,
                                                                          original_tools,
                                                                          check_correct_answer,
                                                                          rephrase_tool_description)
    else:
        metric_values, precomputed = run_with_embedding_rouge_refine(model,
                                                                     rephrase_model,
                                                                     questions,
                                                                     original_tools,
                                                                     get_embed_func,
                                                                     rephrase_tool_description)

    report = report_valtest_metric_after_tools_refine(metric_values=metric_values,
                                                      semaphore_threshold=semaphore_threshold,
                                                      check_correct_answer=check_correct_answer,
                                                      is_info=is_info)
    
    model.set_params(**original_params)
    return {'report': report, 
            'precomputed': precomputed}


def calculate_wilcoxon_test(baseline_metrics: tp.List[float], 
                            refine_metrics: tp.List[float]) -> tp.Tuple[float, float]:
    """
    Функция вычисления теста Уилкоксона на разностях
    Parameters:
        baseline_metrics: Метрики базового запуска
        refine_metrics: Метрики запуска с уточненными описаниями
    Returns:
        Кортеж (p_value, mean_difference)
    """
    differences = [refine - base for base, refine in zip(baseline_metrics, refine_metrics)]
    mean_difference = np.mean(differences)
    _, p_value = stats.wilcoxon(differences)
    return (float(p_value), 
            float(mean_difference))


def run_with_correct_answer_check_refine(model: tp.Any,
                                         rephrase_model: tp.Any,
                                         questions: tp.List[str],
                                         original_tools: tp.List[tp.Dict[str, str]],
                                         check_correct_answer: tp.Optional[tp.Callable[[str, tp.List[str]], bool]],
                                         rephrase_tool_description: tp.Callable) -> tp.Tuple[tp.Dict[str, float], tp.Dict[str, float]]:
    """
    Функция для запуска теста с проверкой корректности ответов с помощью функции check_correct_answer (автометрики)
    Parameters:
        model: Валидируемый агент; имеет метод predict(список вопросов), метод tools со списком инструментов,
               метод get_params возвращающий текущие параметры агента и метод set_params позволяющий задавать параметры агента.
        rephrase_model: Модель для перефразирования описаний инструментов
        questions: Список вопросов для тестирования
        original_tools: Оригинальный список инструментов модели
        check_correct_answer: Функция проверки корректности ответа модели для подсчета автометрики
        rephrase_tool_description: Функция для перефразирования описания инструмента, ее параметры:
         - rephrase_model: Модель для перефразирования
         - original_description: Оригинальное описание инструмента
    Returns:
        Кортеж из двух словарей:
        - metric_values: словарь с метриками и p-значениями статистических тестов
        - precomputed: словарь с предвычисленными основными метриками
    """
    model.set_params(tools=original_tools)
    baseline_answers = model.predict(questions)
    baseline_metrics = []
    for i, question in enumerate(questions):
        correctness = float(check_correct_answer(question, [baseline_answers[i]]))
        baseline_metrics.append(correctness)
    
    refined_tools = deepcopy(original_tools)
    
    for tool in tqdm(refined_tools, desc='Перефразирование описаний инструментов'):
        if 'description' in tool:
            tool['description'] = rephrase_tool_description(rephrase_model, tool['description'])
    
    model.set_params(tools=refined_tools)
    refine_answers = model.predict(questions)
    refine_metrics = []
    for i, question in enumerate(questions):
        correctness = float(check_correct_answer(question, [refine_answers[i]]))
        refine_metrics.append(correctness)

    p_value_wilcoxon, mean_difference = calculate_wilcoxon_test(baseline_metrics, 
                                                                refine_metrics)

    metric_values = {'baseline_mean': float(np.mean(baseline_metrics)),
                     'refine_mean': float(np.mean(refine_metrics)),
                     'p_value_wilcoxon': p_value_wilcoxon,
                     'mean_difference': mean_difference}
    
    precomputed = {'baseline_mean': float(np.mean(baseline_metrics)),
                   'refine_mean': float(np.mean(refine_metrics)),
                   'baseline_answers': baseline_answers,
                   'refine_answers': refine_answers,
                   'refined_tools': refined_tools,
                   'baseline_metrics': baseline_metrics,
                   'refine_metrics': refine_metrics}
    
    return metric_values, precomputed


def run_with_embedding_rouge_refine(model: tp.Any,
                                    rephrase_model: tp.Any,
                                    questions: tp.List[str],
                                    original_tools: tp.List[tp.Dict[str, str]],
                                    get_embed_func: tp.Optional[tp.Callable],
                                    rephrase_tool_description: tp.Callable) -> tp.Tuple[tp.Dict[str, float], tp.Dict[str, float]]:
    """
    Функция для запуска теста с проверкой корректности ответов с помощью функции get_embed_func (эмбеддинги)
    Parameters:
        model: Валидируемый агент; имеет метод predict(список вопросов), метод tools со списком инструментов,
               метод get_params возвращающий текущие параметры агента и метод set_params позволяющий задавать параметры агента.
        rephrase_model: Модель для перефразирования описаний инструментов
        questions: Список вопросов для тестирования
        original_tools: Оригинальный список инструментов модели
        get_embed_func: Функция для получения эмбеддингов
        rephrase_tool_description: Функция для перефразирования описания инструмента, ее параметры:
         - rephrase_model: Модель для перефразирования
         - original_description: Оригинальное описание инструмента
    Returns:
        Кортеж из двух словарей:
         - metric_values: словарь с метриками и p-значениями статистических тестов
         - precomputed: словарь с предвычисленными основными метриками
    """
    rouge_scorer_obj = Rouge()
    model.set_params(tools=original_tools)
    baseline_answers = model.predict(questions)
    baseline_embeddings_raw = [get_embed_func([answer]) for answer in baseline_answers]
    baseline_embeddings = [embed[0][0] for embed in baseline_embeddings_raw]
    refined_tools = deepcopy(original_tools)
    for tool in tqdm(refined_tools, desc='Перефразирование описаний инструментов'):
        if 'description' in tool:
            tool['description'] = rephrase_tool_description(rephrase_model, 
                                                            tool['description'])
    model.set_params(tools=refined_tools)
    refine_answers = model.predict(questions)
    
    baseline_cosine_metrics = []
    baseline_rouge_metrics = []
    refine_cosine_metrics = []
    refine_rouge_metrics = []
    for i, (base_ans, refine_ans) in enumerate(zip(baseline_answers, refine_answers)):
        base_embed_raw = get_embed_func([base_ans])
        base_embed = base_embed_raw[0][0]
        base_cos_dist = cosine_distance(baseline_embeddings[i],
                                        base_embed)
        if hasattr(base_cos_dist, '__len__') and not isinstance(base_cos_dist, (int, float)):
            base_cos_dist = np.mean(base_cos_dist)
        base_cos_sim = 1 - base_cos_dist
        if np.isnan(base_cos_sim) or np.isinf(base_cos_sim):
            base_cos_sim = 0.0
        baseline_cosine_metrics.append(float(base_cos_sim))
        base_rouge_score = rouge_scorer_obj.get_scores(base_ans, base_ans)[0]['rouge-l']['f']
        baseline_rouge_metrics.append(base_rouge_score)
        
        refine_embed_raw = get_embed_func([refine_ans])
        refine_embed = refine_embed_raw[0][0]
        refine_cos_dist = cosine_distance(baseline_embeddings[i], 
                                          refine_embed)
        if hasattr(refine_cos_dist, '__len__') and not isinstance(refine_cos_dist, (int, float)):
            refine_cos_dist = np.mean(refine_cos_dist)
        refine_cos_sim = 1 - refine_cos_dist
        if np.isnan(refine_cos_sim) or np.isinf(refine_cos_sim):
            refine_cos_sim = 0.0
        refine_cosine_metrics.append(float(refine_cos_sim))
        refine_rouge_score = rouge_scorer_obj.get_scores(refine_ans, base_ans)[0]['rouge-l']['f']
        refine_rouge_metrics.append(refine_rouge_score)

    p_value_cosine_wilcoxon, mean_cosine_difference = calculate_wilcoxon_test(baseline_cosine_metrics, 
                                                                              refine_cosine_metrics)
    p_value_rouge_wilcoxon, mean_rouge_difference = calculate_wilcoxon_test(baseline_rouge_metrics, 
                                                                            refine_rouge_metrics)

    metric_values = {'cosine_similarity': float(np.mean(refine_cosine_metrics)),
                     'rougeL': float(np.mean(refine_rouge_metrics)),
                     'p_value_cosine_wilcoxon': p_value_cosine_wilcoxon,
                     'p_value_rouge_wilcoxon': p_value_rouge_wilcoxon,
                     'mean_cosine_difference': mean_cosine_difference,
                     'mean_rouge_difference': mean_rouge_difference}
    
    precomputed = {'cosine_similarity': float(np.mean(refine_cosine_metrics)),
                   'rougeL': float(np.mean(refine_rouge_metrics)),
                   'baseline_answers': baseline_answers,
                   'refine_answers': refine_answers,
                   'refined_tools': refined_tools,
                   'baseline_cosine_metrics': baseline_cosine_metrics,
                   'refine_cosine_metrics': refine_cosine_metrics,
                   'baseline_rouge_metrics': baseline_rouge_metrics,
                   'refine_rouge_metrics': refine_rouge_metrics}
    
    return (metric_values, 
            precomputed)

