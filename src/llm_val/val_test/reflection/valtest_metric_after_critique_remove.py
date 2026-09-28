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
from llm_val.utils.autometrics import bleu_score, meteor_score


def report_valtest_metric_after_critique_removal(metric_values: tp.Dict[str, float],
                                                 semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]]={'cosine_similarity': (0.7, 0.8),
                                                                                                            'rouge': (0.5, 0.6),
                                                                                                            'bleu': (0.5, 0.6),
                                                                                                            'meteor': (0.5, 0.6),
                                                                                                            'p_value': (0.01, 0.05)},
                                                 check_correct_answer: tp.Optional[tp.Callable[[str, tp.List[str]], bool]]=None,
                                                 stat_checker: bool=True,
                                                 is_info: bool=False,
                                                 **kwargs) -> tp.Dict[str, tp.Any]:
    """
    Создание report'а для теста с удалением critique модели
    Parameters:
        metric_values: Словарь с метриками и их значениями
            Для check_correct_answer=True требуется:
                - baseline_mean: среднее базовое качество
                - removal_mean: среднее качество после удаления critique
                - p_value_wilcoxon: p-value теста Уилкоксона
            Для check_correct_answer=False и stat_checker=True требуется:
                - cosine_similarity: косинусная близость
                - p_value_cosine_wilcoxon: p-value теста Уилкоксона для косинусной близости
            Для check_correct_answer=False и stat_checker=False требуется:
                - cosine_similarity: косинусная близость
        semaphore_threshold: Словарь с границами светофора для каждой метрики
        check_correct_answer: Функция проверки корректности ответа модели
        stat_checker: Параметр для выбора способа выставления светофора, по умолчанию используется статистический подход (светофор по p_value)  
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    """
    result_dataframes = []
    color_list = {}
    semaphore_color = 'gray'
    result_data = []

    if check_correct_answer:
        p_value_wilcoxon = metric_values.get('p_value_wilcoxon')
        color_list['p_value_wilcoxon'] = semaphore_by_threshold(value=p_value_wilcoxon,
                                                                threshold=semaphore_threshold.get('p_value'),
                                                                greater_is_better=False,
                                                                left_border_is_bigger=False,
                                                                right_border_is_bigger=False)
        
        result_data = [{'Метрика': 'Базовое качество (с critique)', 'Значение': f"{metric_values['baseline_mean']:.3f}"},
                       {'Метрика': 'Качество после удаления critique', 'Значение': f"{metric_values['removal_mean']:.3f}"},
                       {'Метрика': 'p-value (Уилкоксон)', 'Значение': f"{p_value_wilcoxon:.3f}"}]
    else:
        if stat_checker:
            p_value_cosine_wilcoxon = metric_values.get('p_value_cosine_wilcoxon')
            p_value_rouge_wilcoxon = metric_values.get('p_value_rouge_wilcoxon')
            p_value_bleu_wilcoxon = metric_values.get('p_value_bleu_wilcoxon')
            p_value_meteor_wilcoxon =  metric_values.get('p_value_meteor_wilcoxon')
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
            color_list['p_value_bleu_wilcoxon'] = semaphore_by_threshold(value=p_value_bleu_wilcoxon,
                                                                           threshold=semaphore_threshold.get('p_value'),
                                                                           greater_is_better=False,
                                                                           left_border_is_bigger=False,
                                                                           right_border_is_bigger=False)
            color_list['p_value_meteor_wilcoxon'] = semaphore_by_threshold(value=p_value_meteor_wilcoxon,
                                                                           threshold=semaphore_threshold.get('p_value'),
                                                                           greater_is_better=False,
                                                                           left_border_is_bigger=False,
                                                                           right_border_is_bigger=False)
        
            
            result_data = [{'Метрика': 'Косинусная близость', 'Значение': f"{metric_values['cosine_similarity']:.3f}"},
                           {'Метрика': 'p-value (косинусная близость)', 'Значение': f"{p_value_cosine_wilcoxon:.3f}"},
                           {'Метрика': 'RougeL', 'Значение': f"{metric_values['rouge']:.3f}"},
                           {'Метрика': 'p-value (RougeL)', 'Значение': f"{p_value_rouge_wilcoxon:.3f}"},
                           {'Метрика': 'BLEU', 'Значение': f"{metric_values['bleu']:.3f}"},
                           {'Метрика': 'p-value (BLEU)', 'Значение': f"{p_value_bleu_wilcoxon:.3f}"},
                           {'Метрика': 'METEOR', 'Значение': f"{metric_values['meteor']:.3f}"},
                           {'Метрика': 'p-value (METEOR)', 'Значение': f"{p_value_meteor_wilcoxon:.3f}"}]
        else:
            cosine_similarity = metric_values['cosine_similarity']
            rouge = metric_values.get('rouge')
            bleu = metric_values.get('bleu')
            meteor = metric_values.get('meteor')
            cosine_yellow, cosine_red = semaphore_threshold['cosine_similarity']
            rouge_yellow, rouge_red = semaphore_threshold['rouge']
            bleu_yellow, bleu_red = semaphore_threshold['bleu']
            meteor_yellow, meteor_red = semaphore_threshold['meteor']
            if is_info:
                semaphore_color = 'gray'
            else:
                if cosine_similarity < cosine_yellow or all([rouge < rouge_yellow, bleu < bleu_yellow, meteor < meteor_yellow]):
                    semaphore_color = 'green'
                elif cosine_similarity > cosine_red and any([rouge > rouge_red, bleu > bleu_red, meteor > meteor_red]):
                    semaphore_color = 'red'
                else:
                    semaphore_color = 'yellow'
                
                result_data = [{'Метрика': 'Косинусная близость', 'Значение': f"{cosine_similarity:.3f}"},
                               {'Метрика': 'RougeL', 'Значение': f"{rouge:.3f}"},
                               {'Метрика': 'BLEU', 'Значение': f"{bleu:.3f}"},
                               {'Метрика': 'METEOR', 'Значение': f"{meteor:.3f}"},
                               {'Метрика': 'Результат', 'Значение': semaphore_color}]

    if check_correct_answer or stat_checker:
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


def valtest_metric_after_critique_removal(sampler: Sampler,
                                          model: tp.Any,
                                          get_embed_func: tp.Optional[tp.Callable]=None,
                                          check_correct_answer: tp.Optional[tp.Callable[[str, tp.List[str]], bool]]=None,
                                          data_type: str='test',
                                          semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]]={'cosine_similarity': (0.7, 0.8),
                                                                                                     'rouge': (0.5, 0.6),
                                                                                                     'bleu': (0.5, 0.6),
                                                                                                     'meteor': (0.5, 0.6),
                                                                                                     'p_value': (0.01, 0.05)},
                                          stat_checker: bool=True,
                                          is_info: bool=False,
                                          **kwargs) -> tp.Dict[str, tp.Any]:
    """
    Тест на анализ изменения ответа LLM при удалении critique модели
    Parameters:
        sampler: Объект класса Sampler
        model: Валидируемый агент; имеет метод predict(список вопросов), метод set_params, позволяющий задавать параметры агента
            и метод get_params, возвращающий текущие параметры агента, в частности - critique_model - модель ноды-критика
        get_embed_func: Функция для получения эмбеддингов (для случая без автометрики)
        check_correct_answer: Функция проверки корректности ответа модели (для случая с автометрикой)
        data_type: Тип данных для расчета метрики
        semaphore_threshold: Словарь с границами светофора для каждой метрики
        stat_checker: Параметр для выбора способа выставления светофора, по умолчанию используется статистический подход (светофор по p_value)  
        is_info: True, если тест информативный
    Returns:
        report в соответствии с концепцией llm_val
    """
    data = deepcopy(getattr(sampler, data_type))
    questions = data['X']['question']
    original_params = model.get_params()
    original_critique = original_params.get('critique_model', None)
    
    if check_correct_answer:
        metric_values, precomputed = run_critique_removal_with_correct_answer_check(model,
                                                                                    questions,
                                                                                    original_critique,
                                                                                    check_correct_answer)
    else:
        metric_values, precomputed = run_critique_removal_with_embedding_rouge(model,
                                                                               questions,
                                                                               original_critique,
                                                                               get_embed_func,
                                                                               stat_checker)

    report = report_valtest_metric_after_critique_removal(metric_values=metric_values,
                                                          semaphore_threshold=semaphore_threshold,
                                                          check_correct_answer=check_correct_answer,
                                                          stat_checker=stat_checker,
                                                          is_info=is_info)
    
    model.set_params(**original_params)
    return {'report': report, 
            'precomputed': precomputed}


def run_critique_removal_with_correct_answer_check(model: tp.Any, 
                                                   questions: tp.List[str], 
                                                   original_critique: tp.Any,
                                                   check_correct_answer: tp.Callable[[str, tp.List[str]], bool]) -> tp.Tuple[tp.Dict[str, float], tp.Dict[str, float]]:
    """
    Функция для запуска теста с проверкой корректности ответов с помощью функции check_correct_answer (автометрики)
    Parameters:
        model: Валидируемый агент; имеет метод predict(список вопросов), метод set_params, позволяющий задавать параметры агента
            и метод get_params, возвращающий текущие параметры агента, в частности - critique_model - модель ноды-критика
        questions: Список вопросов для тестирования
        original_critique: Модель для ноды-критика
        check_correct_answer: Функция проверки корректности ответа модели для подсчета автометрики    
    Returns:
        Кортеж из словарей:
        - metric_values: словарь с метриками и p-значениями статистических тестов
        - precomputed: словарь с предвычисленными основными метриками
    """
    model.set_params(critique_model=original_critique)
    baseline_answers = model.predict(questions)
    baseline_metrics = []
    for i, question in enumerate(questions):
        correctness = float(check_correct_answer(question, [baseline_answers[i]]))
        baseline_metrics.append(correctness)
    model.set_params(critique_model=None)
    removal_answers = model.predict(questions)
    removal_metrics = []
    for i, question in enumerate(questions):
        correctness = float(check_correct_answer(question, [removal_answers[i]]))
        removal_metrics.append(correctness)
    p_value_wilcoxon = calculate_wilcoxon_p_value(baseline_metrics, 
                                                  removal_metrics)
    metric_values = {'baseline_mean': float(np.mean(baseline_metrics)),
                     'removal_mean': float(np.mean(removal_metrics)),
                     'p_value_wilcoxon': p_value_wilcoxon}
    precomputed = {'baseline_mean': float(np.mean(baseline_metrics)),
                   'removal_mean': float(np.mean(removal_metrics))}
    return (metric_values, 
            precomputed)


def run_critique_removal_with_embedding_rouge(model: tp.Any,
                                              questions: tp.List[str],
                                              original_critique: tp.Any,
                                              get_embed_func: tp.Callable,
                                              stat_checker: bool) -> tp.Tuple[tp.Dict[str, float], tp.Dict[str, float]]:
    """
    Функция для запуска теста с проверкой корректности ответов с помощью функции get_embed_func (эмбеддинги)
    Parameters:
        model: Валидируемый агент; имеет метод predict(список вопросов), метод set_params, позволяющий задавать параметры агента
            и метод get_params, возвращающий текущие параметры агента, в частности - critique_model - модель ноды-критика
        questions: Список вопросов для тестирования
        original_critique: Модель для ноды-критика
        get_embed_func: Функция для получения эмбеддингов
        stat_checker: Параметр для выбора способа выставления светофора, по умолчанию используется статистический подход (светофор по p_value)  
    Returns:
        Кортеж из словарей:
        - metric_values: словарь с метриками и p-значениями статистических тестов
        - precomputed: словарь с предвычисленными основными метриками
    """
    rouge_scorer_obj = Rouge()
    model.set_params(critique_model=original_critique)
    baseline_answers = model.predict(questions)
    baseline_embeddings_raw = [get_embed_func([answer]) for answer in baseline_answers]
    baseline_embeddings = [embed[0][0] for embed in baseline_embeddings_raw]

    baseline_cosine_sims = []
    baseline_rouge_scores = []
    baseline_bleu_scores = []
    baseline_meteor_scores = []

    for i, base_ans in enumerate(baseline_answers):
        base_embed = baseline_embeddings[i]
        current_answers = model.predict([questions.iloc[i]])
        curr_ans = current_answers[0]
        curr_embed_raw = get_embed_func([curr_ans])
        curr_embed = curr_embed_raw[0][0]
        cos_dist = cosine_distance(base_embed, curr_embed)
        if hasattr(cos_dist, '__len__') and not isinstance(cos_dist, (int, float)):
            cos_dist = np.mean(cos_dist)
        cos_sim = 1 - cos_dist
        if np.isnan(cos_sim) or np.isinf(cos_sim):
            cos_sim = 0.0
        baseline_cosine_sims.append(float(cos_sim))
        rouge_score_val = rouge_scorer_obj.get_scores(curr_ans, base_ans)[0]['rouge-l']['f']
        baseline_rouge_scores.append(rouge_score_val)
        bleu_score_val = bleu_score([base_ans], curr_ans)
        baseline_bleu_scores.append(bleu_score_val)
        meteor_score_val = meteor_score([base_ans], curr_ans)
        baseline_meteor_scores.append(meteor_score_val)

    model.set_params(critique_model=None)
    removal_answers = model.predict(questions)
    removal_cosine_sims = []
    removal_rouge_scores = []
    removal_bleu_scores = []
    removal_meteor_scores = []

    for i, (base_ans, removal_ans) in enumerate(zip(baseline_answers, removal_answers)):
        base_embed = baseline_embeddings[i]
        removal_embed_raw = get_embed_func([removal_ans])
        removal_embed = removal_embed_raw[0][0]
        cos_dist = cosine_distance(base_embed, removal_embed)
        if hasattr(cos_dist, '__len__') and not isinstance(cos_dist, (int, float)):
            cos_dist = np.mean(cos_dist)
        cos_sim = 1 - cos_dist
        if np.isnan(cos_sim) or np.isinf(cos_sim):
            cos_sim = 0.0
        removal_cosine_sims.append(float(cos_sim))
        rouge_score_val = rouge_scorer_obj.get_scores(removal_ans, base_ans)[0]['rouge-l']['f']
        removal_rouge_scores.append(rouge_score_val)
        bleu_score_val = bleu_score([base_ans], removal_ans)
        removal_bleu_scores.append(bleu_score_val)
        meteor_score_val = meteor_score([base_ans], removal_ans)
        removal_meteor_scores.append(meteor_score_val)

    if stat_checker:
        p_value_cosine_wilcoxon = calculate_wilcoxon_p_value(baseline_cosine_sims, 
                                                             removal_cosine_sims)
        p_value_rouge_wilcoxon = calculate_wilcoxon_p_value(baseline_rouge_scores, 
                                                            removal_rouge_scores)
        p_value_bleu_wilcoxon = calculate_wilcoxon_p_value(baseline_bleu_scores, 
                                                           removal_bleu_scores)
        p_value_meteor_wilcoxon = calculate_wilcoxon_p_value(baseline_meteor_scores, 
                                                             removal_meteor_scores)

        metric_values = {'cosine_similarity': float(np.mean(removal_cosine_sims)),
                         'rouge': float(np.mean(removal_rouge_scores)),
                         'bleu': float(np.mean(removal_bleu_scores)),
                         'meteor': float(np.mean(removal_meteor_scores)),
                         'p_value_cosine_wilcoxon': p_value_cosine_wilcoxon,
                         'p_value_rouge_wilcoxon': p_value_rouge_wilcoxon,
                         'p_value_bleu_wilcoxon': p_value_bleu_wilcoxon,
                         'p_value_meteor_wilcoxon': p_value_meteor_wilcoxon}
    else:
        metric_values = {'cosine_similarity': float(np.mean(removal_cosine_sims)),
                         'rouge': float(np.mean(removal_rouge_scores)),
                         'bleu': float(np.mean(removal_bleu_scores)),
                         'meteor': float(np.mean(removal_meteor_scores))}

    precomputed = {'cosine_similarity': float(np.mean(removal_cosine_sims)),
                   'rouge': float(np.mean(removal_rouge_scores)),
                   'bleu': float(np.mean(removal_bleu_scores)),
                   'meteor': float(np.mean(removal_meteor_scores))}

    return (metric_values,
            precomputed)


def calculate_wilcoxon_p_value(baseline_metrics: tp.List[float], 
                               removal_metrics: tp.List[float]) -> float:
    """
    Функция вычисления p-value теста Уилкоксона для парных разностей
    Parameters:
        baseline_metrics: Метрики базового запуска
        removal_metrics: Метрики запуска с удаленными инструментами
    Returns:
        p_value: Значение p_value
    """
    differences = np.array(baseline_metrics) - np.array(removal_metrics)
    if np.all(differences == 0):
        p_value = 1.0
    else:
        _, p_value = stats.wilcoxon(baseline_metrics, removal_metrics)
    return float(p_value)