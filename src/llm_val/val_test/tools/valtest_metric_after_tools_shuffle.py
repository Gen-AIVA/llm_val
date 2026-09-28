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


def report_valtest_metric_after_tools_shuffle(metric_values: tp.Dict[str, float],
                                              semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]]={'cosine_similarity': (0.75, 0.9),
                                                                                                         'rougeL': (0.5, 0.6),
                                                                                                         'p_value': (0.01, 0.05)},
                                              check_correct_answer: tp.Optional[tp.Callable[[str, tp.List[str]], bool]]=None,
                                              stat_checker: bool=True,
                                              is_info: bool=False,
                                              **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а 
    Parameters:
        metric_values: Словарь с метриками и их значениями
            Для check_correct_answer=True требуется:
                - baseline_mean: среднее базовое качество
                - shuffle_mean: среднее качество после перемешивания
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
        check_correct_answer: Функция проверки корректности ответа модели для подсчета автометрики
        stat_checker: Параметр, отвечающий за выбор способа выставления светофора, по дефолту статистический подход (светофор по p_value)
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
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
                       {'Метрика': 'Качество после перемешивания', 'Значение': f"{metric_values['shuffle_mean']:.3f}"},
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
            cosine_yellow, cosine_red = semaphore_threshold['cosine_similarity']
            rouge_yellow, rouge_red = semaphore_threshold['rougeL']
            if cosine_similarity > cosine_red and rougeL > rouge_red:
                semaphore_color = 'red'
            elif cosine_similarity > cosine_yellow and rougeL > rouge_yellow:
                semaphore_color = 'yellow'
            else:
                semaphore_color = 'green'
                    
            result_data = [{'Метрика': 'Косинусная близость', 'Значение': f"{cosine_similarity:.3f}"},
                           {'Метрика': 'Rouge-L', 'Значение': f"{rougeL:.3f}"}]
            
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
    res_dict['semaphore'] =  semaphore_color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = result_dataframes
    return res_dict


def valtest_metric_after_tools_shuffle(sampler: Sampler,
                                       model: tp.Any,
                                       shuffle_num: int=10,
                                       get_embed_func: tp.Optional[tp.Callable]=None,
                                       check_correct_answer: tp.Optional[tp.Callable[[str, tp.List[str]], bool]]=None,
                                       data_type: str='test',
                                       semaphore_threshold: tp.Dict[str, tp.Tuple[float, float]]={'cosine_similarity': (0.75, 0.9),
                                                                                                  'rougeL': (0.5, 0.6),
                                                                                                  'p_value': (0.01, 0.05)},
                                       stat_checker: bool=True,
                                       is_info: bool=False,
                                       **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на анализ изменения ответа LLM при перемешивании списка инструментов
    Parameters:
        sampler: Объект класса Sampler
        model: Валидируемый агент; имеет метод predict(температура, список вопросов), метод tools со списком инструментов, метод get_params, возвращающий текущие параметры агента, и метод set_params, позволяющий задавать параметры агента.
        shuffle_num: Количество случайных перестановок
        get_embed_func: Функция для получения эмбеддингов
        check_correct_answer: Функция проверки корректности ответа модели для подсчета автометрики
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold: Словарь с границами светофора для каждой метрики
            Должен содержать соответствующие ключи в зависимости от режима:
            - Для p-value метрик: 'p_value' с порогами (жёлтый, красный)
            - Для метрик качества: 'cosine_similarity' и 'rougeL' с порогами (жёлтый, красный)
        stat_checker: параметр, отвечающий за выбор способа выставления светофора, по дефолту статистический подход (светофор по p_value)
        is_info: True, если тест информативный
    Returns:
        report в соответствии с концепцией llm_val
    '''
    data = deepcopy(getattr(sampler, data_type))
    questions = data['X']['question']
    original_params = model.get_params()
    original_tools = original_params.get('tools', [])
    
    if check_correct_answer:
        metric_values, precomputed = run_with_correct_answer_check(model,
                                                                   questions,
                                                                   original_tools,
                                                                   shuffle_num,
                                                                   check_correct_answer)
    else:
        metric_values, precomputed = run_with_embedding_rouge(model,
                                                              questions,
                                                              original_tools,
                                                              shuffle_num,
                                                              get_embed_func)

    report = report_valtest_metric_after_tools_shuffle(metric_values=metric_values,
                                                       semaphore_threshold=semaphore_threshold,
                                                       check_correct_answer=check_correct_answer,
                                                       shuffle_num=shuffle_num,
                                                       stat_checker=stat_checker,
                                                       is_info=is_info)
    
    model.set_params(**original_params)
    return {'report': report, 
            'precomputed': precomputed}


def calculate_p_values(baseline_metrics: tp.List[float], 
                       shuffle_metrics: tp.List[float]) -> tp.Tuple[float, float]:
    """
    Функция вычисления p-value для сравнения двух распределений
    Parameters:
        baseline_metrics: Метрики базового запуска
        shuffle_metrics: Метрики запуска с перемешанными инструментами
    Returns:
        Кортеж (p_value_means, p_value_variance)
    """
    if np.std(baseline_metrics) <= 1e-10:
        _, p_value_means = stats.ttest_ind(baseline_metrics, shuffle_metrics)
        p_value_variance = 0.0
    else:
        _, p_value_means = stats.ttest_ind(baseline_metrics, shuffle_metrics)
        _, p_value_variance = stats.levene(baseline_metrics, shuffle_metrics)
    
    return (float(p_value_means), 
            float(p_value_variance))


def run_with_correct_answer_check(model: tp.Any, 
                                  questions: tp.List[str], 
                                  original_tools: tp.List[tp.Any], 
                                  shuffle_num: int, 
                                  check_correct_answer: tp.Optional[tp.Callable[[str, tp.List[str]], bool]]) -> tp.Tuple[tp.Dict[str, float], tp.Dict[str, float]]:
    """
    Функция для запуска теста с проверкой корректности ответов с помощью функции check_correct_answer (автометрики)
    Parameters:
        model: Валидируемый агент; имеет метод predict(список вопросов), метод tools со списком инструментов,
               метод get_params возвращающий текущие параметры агента и метод set_params позволяющий задавать параметры агента.
        questions: Список вопросов для тестирования
        original_tools: Оригинальный список инструментов модели
        shuffle_num: Количество случайных перестановок
        check_correct_answer: Функция проверки корректности ответа модели для подсчета автометрики    
    Returns:
        Кортеж из двух словарей:
        - metric_values: словарь с метриками и p-значениями статистических тестов
        - precomputed: словарь с предвычисленными основными метриками
    """
    baseline_metrics = []
    for _ in tqdm(range(shuffle_num), desc='Baseline runs'):
        baseline_answers = model.predict(questions)
        run_metrics = []
        for i, question in enumerate(questions):
            correctness = float(check_correct_answer(question, [baseline_answers[i]]))
            run_metrics.append(correctness)
        baseline_metrics.append(np.mean(run_metrics))
    
    shuffle_metrics = []
    for _ in tqdm(range(shuffle_num), desc='Tools shuffle'):
        shuffled_tools = deepcopy(original_tools)
        np.random.shuffle(shuffled_tools)
        model.set_params(tools=shuffled_tools)
        shuffle_answers = model.predict(questions)
        
        run_metrics = []
        for i, question in enumerate(questions):
            correctness = float(check_correct_answer(question, [shuffle_answers[i]]))
            run_metrics.append(correctness)
        
        shuffle_metrics.append(np.mean(run_metrics))

    p_value_means, p_value_variance = calculate_p_values(baseline_metrics, 
                                                         shuffle_metrics)

    metric_values = {'baseline_mean': float(np.mean(baseline_metrics)),
                     'shuffle_mean': float(np.mean(shuffle_metrics)),
                     'p_value_means': p_value_means,
                     'p_value_variance': p_value_variance}
    
    precomputed = {'baseline_mean': float(np.mean(baseline_metrics)),
                   'shuffle_mean': float(np.mean(shuffle_metrics))}
    return metric_values, precomputed


def run_with_embedding_rouge(model: tp.Any,
                             questions: tp.List[str],
                             original_tools: tp.List[tp.Any],
                             shuffle_num: int,
                             get_embed_func: tp.Optional[tp.Callable]) -> tp.Tuple[tp.Dict[str, float], tp.Dict[str, float]]:
    """
    Функция для запуска теста с проверкой корректности ответов с помощью функции get_embed_func (эмбеддинги)
    Parameters:
        model: Валидируемый агент; имеет метод predict(список вопросов), метод tools со списком инструментов,
               метод get_params возвращающий текущие параметры агента и метод set_params позволяющий задавать параметры агента.
        questions: Список вопросов для тестирования
        original_tools: Оригинальный список инструментов модели
        shuffle_num: Количество случайных перестановок
        get_embed_func: Функция для получения эмбеддингов
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
    baseline_cosine_sims = []
    baseline_rouge_scores = []
    for _ in tqdm(range(shuffle_num), desc='Baseline runs'):
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
    
    shuffle_cosine_sims = []
    shuffle_rouge_scores = []

    for _ in tqdm(range(shuffle_num), desc='Tools shuffle'):
        shuffled_tools = deepcopy(original_tools)
        np.random.shuffle(shuffled_tools)
        model.set_params(tools=shuffled_tools)
        shuffle_answers = model.predict(questions)
        
        cos_sims = []
        rouge_scores = []
        for i, (base_ans, shuffle_ans) in enumerate(zip(baseline_answers, shuffle_answers)):
            shuffle_embed_raw = get_embed_func([shuffle_ans])
            shuffle_embed = shuffle_embed_raw[0][0]
            
            cos_dist = cosine_distance(baseline_embeddings[i], shuffle_embed)
            if hasattr(cos_dist, '__len__') and not isinstance(cos_dist, (int, float)):
                cos_dist = np.mean(cos_dist)
            
            cos_sim = 1 - cos_dist
            if np.isnan(cos_sim) or np.isinf(cos_sim):
                cos_sim = 0.0
            cos_sims.append(float(cos_sim))
            
            rouge_score_val = rouge_scorer_obj.get_scores(shuffle_ans, base_ans)[0]['rouge-l']['f']
            rouge_scores.append(rouge_score_val)
        
        shuffle_cosine_sims.append(np.mean(cos_sims))
        shuffle_rouge_scores.append(np.mean(rouge_scores))
    
    p_value_cosine_means, p_value_cosine_variance = calculate_p_values(baseline_cosine_sims, 
                                                                       shuffle_cosine_sims)
    p_value_rouge_means, p_value_rouge_variance = calculate_p_values(baseline_rouge_scores, 
                                                                     shuffle_rouge_scores)

    metric_values = {'cosine_similarity': float(np.mean(shuffle_cosine_sims)),
                     'rougeL': float(np.mean(shuffle_rouge_scores)),
                     'p_value_cosine_means': p_value_cosine_means,
                     'p_value_cosine_variance': p_value_cosine_variance,
                     'p_value_rouge_means': p_value_rouge_means,
                     'p_value_rouge_variance': p_value_rouge_variance}
    
    precomputed = {'cosine_similarity': float(np.mean(shuffle_cosine_sims)),
                   'rougeL': float(np.mean(shuffle_rouge_scores))}
    
    return (metric_values, 
            precomputed)
