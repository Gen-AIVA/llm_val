import re
import numpy as np
import pandas as pd
import typing as tp
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics.pairwise import cosine_similarity

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.custom_checkers import check_if_correct_with_options
from llm_val.utils.custom_instructions import mmlu_instructions, rubq_instructions


def report_valtest_confidence_in_incorrect_answers(figure: tp.Any) -> tp.Dict[str, tp.Any]:
    '''
    Формирование отчета для теста на анализ уверенности LLM в ошибочных ответах.
    Parameters:
        figure: График распределения энтропии в зависимости от верного ответа.
    Returns:
        Словарь в соответствии с концепцией llm_val.
    '''
    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_plots'] = [figure]
    res_dict['result_dataframes'] = []
    return res_dict


def valtest_confidence_in_incorrect_answers(sampler: Sampler,
                                            model: tp.Any,
                                            get_embed_func: tp.Callable,
                                            data_type: str='test',
                                            repeat_num: int=3,
                                            high_temp: float=1.5,
                                            check_correct_answer: tp.Optional[tp.Callable[[str, tp.List[str]], bool]]=check_if_correct_with_options,
                                            **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на анализ уверенности LLM в ошибочных ответах.
    Parameters:
        sampler: Объект класса Sampler.
        model: Валидируемая модель; имеет метод predict(температура, список вопросов)
        get_embed_func: Функция для получения эмбеддингов.
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е.
        repeat_num: Количество повторений инференса при повышенной температуре.
        high_temp: Повышенная температура модели.
        check_correct_answer: Функция проверки корректности ответа модели. Используется только для RuBQ
    Returns:
        Словарь в соответствии с концепцией llm_val.
    '''
    data = deepcopy(getattr(sampler, data_type)['X'])
    
    questions = data['question']
    true_answers = data['true_answer']
    model_answers = data['model_answer']
    has_answer_options = 'answer_options' in data
    model_embeds = [get_embed_func([answer]) for answer in model_answers]

    entropies = []
    confidence = []
    correct_labels = []

    for idx, (question, model_answer, model_embed) in enumerate(zip(questions, model_answers, model_embeds)):
        conf = []
        for _ in range(repeat_num):
            formatted_question = format_question(question, 
                                                 idx, 
                                                 mmlu_instructions if has_answer_options else rubq_instructions, 
                                                 data, 
                                                 has_answer_options)           
            high_temp_answer = model.predict(high_temp, 
                                             formatted_question)           
            p = calculate_confidence(high_temp_answer, 
                                     model_answer, 
                                     model_embed, 
                                     get_embed_func, 
                                     has_answer_options)
            conf.append(p)
        
        p_mean = np.mean(conf)            
        confidence.append(p_mean)
        entropy = -1 * p_mean * np.log(p_mean + 1e-7) if p_mean > 1e-7 else 0
        entropies.append(entropy)
        
        if np.isclose(p_mean, 0) or np.isclose(p_mean, 1):
            entropies = [0] * len(entropies)

    if has_answer_options:
        correct_labels = true_answers == model_answer
    else:
        data['is_correct'] = [check_correct_answer(model_answer, true_answer) for model_answer, true_answer in zip(data['model_answer'], data['true_answer'])]
        correct_labels = data['is_correct']

    entropies = np.array(entropies)
    confidence = np.array(confidence)
    correct_labels = np.array(correct_labels)
    
    figure = plot_confidence_quality(entropies, 
                                     correct_labels)
    report = report_valtest_confidence_in_incorrect_answers(figure)
    precomputed = {'entropies': entropies,
                   'correct_labels': correct_labels}
    return {'report': report,
            'precomputed': precomputed}


def plot_confidence_quality(entropies: np.ndarray,
                            correct_labels: np.ndarray) -> Figure:
    '''
    Отрисовка графика распределения энтропии в зависимости от верности ответа.
    Parameters:
        entropies: Массив энтропий.
        correct_labels: Массив меток корректности ответа.
    Returns:
        Объект Figure.
    '''
    fig, ax = plt.subplots(figsize=(9, 7))

    correct_entropies = entropies[correct_labels == 1]
    incorrect_entropies = entropies[correct_labels == 0]

    total_entropy = np.sum(entropies)
    if total_entropy == 0:
        total_entropy = 1

    correct_w = np.ones_like(correct_entropies) / len(entropies)
    incorrect_w = np.ones_like(incorrect_entropies) / len(entropies)

    if len(correct_entropies) > 0 or len(incorrect_entropies) > 0:
        if len(correct_entropies) == 0:
            bins = np.histogram(incorrect_entropies, bins=15)[1]
        elif len(incorrect_entropies) == 0:
            bins = np.histogram(correct_entropies, bins=15)[1]
        else:
            bins = np.histogram(np.hstack((correct_entropies, incorrect_entropies)), bins=15)[1]

        if len(correct_entropies) > 0:
            ax.hist(correct_entropies, weights=correct_w, bins=bins, alpha=0.5, label='Верные ответы', edgecolor='black', linewidth=1.2)
        if len(incorrect_entropies) > 0:
            ax.hist(incorrect_entropies, weights=incorrect_w, bins=bins, alpha=0.5, label='Неверные ответы', edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Энтропия -p*log(p)')
    ax.set_ylabel('Частотность (нормализованная)')
    ax.legend()
    ax.set_title('Распределение энтропии в зависимости от верности ответа')
    return fig


def format_question(question: str, 
                    idx: int, 
                    instructions: tp.Dict[str, str], 
                    data: tp.Any, 
                    has_answer_options: bool) -> str:
    """
    Форматирует вопрос в зависимости от наличия вариантов ответов.
    Parameters:
        question: Текст вопроса.
        idx: Индекс вопроса в данных.
        instructions: Словарь с инструкциями для форматирования вопроса, включая префиксы и постфиксы.
        data: Данные, содержащие контекст или варианты ответов.
        has_answer_options: Флаг, указывающий на наличие вариантов ответов.
    Returns:
        Отформатированный вопрос с учетом наличия дополнительной информации.
    """
    prompt = instructions['instruction'] + instructions['shot']
    if has_answer_options:
        options_prefix, postfix = instructions['options_prefix'], instructions['postfix']
        formatted_question = prompt + question + options_prefix + str(data['answer_options'].iloc[idx]) + postfix
    else:
        context_prefix, postfix = instructions['context_prefix'], instructions['postfix']
        formatted_question = prompt + question + context_prefix + str(data['context'].iloc[idx]) + postfix
    return formatted_question


def calculate_confidence(high_temp_answer: str, 
                         model_answer: str, 
                         model_embed: tp.Any, 
                         get_embed_func: tp.Callable, 
                         has_answer_options: bool) -> float:
    """
    Вычисляет уверенность модели в ответе.
    Parameters:
        high_temp_answer: Ответ модели, полученный с высокой температурой.
        model_answer: Ответ модели, полученный при стандартных параметрах.
        model_embed: Эмбеддинг ответа модели.
        get_embed_func: Функция для получения эмбеддинга текста.
        has_answer_options: Флаг, указывающий на наличие вариантов ответов.
    Returns:
        Уверенность модели в ответе. Если есть варианты ответов, возвращает 1 или 0 в зависимости от совпадения ответов.
            Если вариантов ответов нет, возвращает косинусное сходство между эмбеддингами ответов.
    """
    if has_answer_options:
        high_temp_answer = re.match(r'[a-zA-Z]', high_temp_answer)[0]
        p = 1 if high_temp_answer == model_answer else 0
    else:
        high_temp_embed = get_embed_func([high_temp_answer])
        p = (1 + cosine_similarity(model_embed, high_temp_embed).squeeze()) / 2
    return p
