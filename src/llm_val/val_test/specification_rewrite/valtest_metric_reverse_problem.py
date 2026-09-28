import scipy
import numpy as np
import typing as tp
import pandas as pd
from copy import deepcopy

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.autometrics import metrics2func
from llm_val.utils.report_helper import semaphore_by_threshold


def report_valtest_metric_reverse_problem(forward_problem_metric_value: tp.Dict[str, float],
                                          reverse_problem_metric_value: tp.Dict[str, float],
                                          main_metric: tp.Optional[str]='rouge',
                                          use_cosine_similarity: bool=False,
                                          data_type: str='test',
                                          semaphore_threshold: tp.Tuple[float]=(40, 60),
                                          greater_is_better: bool=True,
                                          is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        forward_problem_metric_value: Словарь со значением ключевой метрики при решении исходной задачи
        reverse_problem_metric_value: Словарь со значением ключевой метрики при решении обратной задачи
        main_metric: Название ключевой метрики качества. Для расчета доступны метрики bleu, rouge и meteor
        use_cosine_similarity: если True, то в качестве ключевой метрики будет использоваться косинусная мера близости между эмбеддингами текстов
        data_type: Тип данных, на котором сравниваются метрики
        semaphore_threshold: Значение желтого и красного порогов для светофора по относительной разнице между ключевыми метриками в процентах
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: Если True, то тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    main_metric = 'cosine_similarity' if use_cosine_similarity else main_metric
    absolute_deviation = reverse_problem_metric_value[main_metric] - forward_problem_metric_value[main_metric]
    absolute_deviation = -absolute_deviation if greater_is_better else absolute_deviation
    relative_deviation = (absolute_deviation / forward_problem_metric_value[main_metric]) * 100
    color = 'gray' if is_info else semaphore_by_threshold(relative_deviation,
                                                          semaphore_threshold,
                                                          greater_is_better=False,
                                                          left_border_is_bigger=True,
                                                          right_border_is_bigger=False)
    res_table = pd.DataFrame({'Метрика': [main_metric],
                             'Значение ключевой метрики при решении прямой задачи': [forward_problem_metric_value[main_metric]],
                             'Значение ключевой метрики при решении обратной задачи': [reverse_problem_metric_value[main_metric]],
                             'Относительное изменение метрики, %': [relative_deviation],
                             'Результат теста': [color]})
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_metric_reverse_problem(sampler: Sampler,
                                   model: tp.Any,
                                   forward_problem_system_prompt: str,
                                   reverse_problem_system_prompt: str,
                                   main_metric: tp.Optional[str]='rouge',
                                   use_cosine_similarity: bool=False,
                                   embedding_model: tp.Optional[tp.Any]=None,
                                   data_type: str='test',
                                   semaphore_threshold: tp.Tuple[float]=(40, 60),
                                   greater_is_better: bool=True,
                                   forward_problem_metric_value: tp.Optional[tp.Dict[str, float]]=None,
                                   reverse_problem_metric_value: tp.Optional[tp.Dict[str, float]]=None,
                                   is_info: bool=False,
                                   **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на расчет относительного изменения ключевой метрики качества при решении обратной задачи
    Parameters:
        sampler: Объект класса Sampler
        model: Валидируемая модель; имеет метод predict(список вопросов)
        forward_problem_system_prompt: Промпт для решения прямой задачи при помощи модели, должен содержать '{source_text}'
        reverse_problem_system_prompt: Промпт для решения обратной задачи при помощи модели, должен содержать '{source_text}'
        main_metric: Название ключевой метрики качества. Для расчета доступны метрики bleu, rouge и meteor
        use_cosine_similarity: если True, то в качестве ключевой метрики будет использоваться косинусная мера близости между эмбеддингами текстов
        embedding_model: Модель для получения эмбеддингов, имеет метод get_embedding(список вопросов)
        data_type: Тип данных, на котором сравниваются метрики
        semaphore_threshold: Значение порогов для светофора по относительной разнице между ключевыми метриками в процентах
        greater_is_better: True, если чем больше метрика, тем лучше
        forward_problem_metric_value: Предрасчитанное значение ключевой метрики при решении прямой задачи. Если не None, то расчет метрики не производится
        reverse_problem_metric_value: Предрасчитанное значение ключевой метрики при решении обратной задачи. Если не None, то расчет метрики не производится
        is_info: True, если тест информативный
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    main_metric = main_metric.lower() if main_metric else main_metric
    if forward_problem_metric_value is None:
        data = getattr(sampler, data_type)['X']
        forward_problem_metric_value = get_main_metric_value(model=model, 
                                                             data=data,
                                                             forward_problem_system_prompt=forward_problem_system_prompt,
                                                             is_forward_problem=True, 
                                                             main_metric=main_metric,
                                                             use_cosine_similarity=use_cosine_similarity,
                                                             embedding_model=embedding_model)
    if reverse_problem_metric_value is None:
        data = getattr(sampler, data_type)['X']
        reverse_problem_metric_value = get_main_metric_value(model=model, 
                                                             data=data,
                                                             reverse_problem_system_prompt=reverse_problem_system_prompt,
                                                             is_forward_problem=False,
                                                             main_metric=main_metric,
                                                             use_cosine_similarity=use_cosine_similarity,
                                                             embedding_model=embedding_model)
    precomputed = {'forward_problem_metric_value': deepcopy(forward_problem_metric_value),
                   'reverse_problem_metric_value': deepcopy(reverse_problem_metric_value)}
    report = report_valtest_metric_reverse_problem(forward_problem_metric_value,
                                                   reverse_problem_metric_value,
                                                   main_metric,
                                                   use_cosine_similarity,
                                                   data_type,
                                                   semaphore_threshold,
                                                   greater_is_better,
                                                   is_info)
    return {'report': report,
            'precomputed': precomputed}


def get_main_metric_value(model: tp.Any,
                          data: pd.DataFrame,
                          forward_problem_system_prompt: tp.Optional[str]=None,
                          reverse_problem_system_prompt: tp.Optional[str]=None, 
                          is_forward_problem: bool=True,
                          main_metric: tp.Optional[str]='rouge',
                          use_cosine_similarity: bool=False,
                          embedding_model: tp.Optional[tp.Any]=None) -> tp.Dict[str, float]:
    '''
    Parameters:
        model: Валидируемая модель; имеет метод predict(список вопросов)
        data: Датафрейм с колонками 'question', 'model_answer', 'true_answer', 'reversed_problem_answer'
        forward_problem_system_prompt: Промпт для решения прямой задачи при помощи модели, должен содержать '{source_text}'
        reverse_problem_system_prompt: Промпт для решения обратной задачи при помощи модели, должен содержать '{source_text}'
        is_forward_problem: True, если хотим посчитать ключевую метрику при решении исходной (прямой) задачи
        main_metric: Название ключевой метрики качества. Для расчета доступны метрики bleu, rouge и meteor
        use_cosine_similarity: если True, то в качестве ключевой метрики будет использоваться косинусная мера близости между эмбеддингами текстов
        embedding_model: Модель для получения эмбеддингов, имеет метод get_embedding(список вопросов)
    Return:
        Словарь со значением ключевой метрики
    '''
    metrics_dict = {}
    if is_forward_problem:
        data['model_answer'] = get_problem_solution(system_prompt=forward_problem_system_prompt, 
                                                    model=model, 
                                                    question=data['question'].tolist())
        if use_cosine_similarity:
            embedding_model_true_answer = embedding_model.get_embedding(data['true_answer'].tolist())
            embedding_model_model_answer = embedding_model.get_embedding(data['model_answer'].tolist())
            metrics_dict['cosine_similarity'] = np.mean([cosine_similarity(a['embedding'], 
                                                                           b['embedding']) 
                                                         for a, b in zip(embedding_model_model_answer, 
                                                                         embedding_model_true_answer)])
        else:
            metrics_dict[main_metric] = data.apply(lambda x: metrics2func[main_metric]([x['true_answer']],
                                                                                        x['model_answer']), axis=1).mean()
    else:
        true_answer = data['true_answer'].tolist()
        data['reversed_problem_solution'] = get_problem_solution(system_prompt=reverse_problem_system_prompt, 
                                                                 model=model,
                                                                 question=true_answer if None not in true_answer else data['model_answer'].tolist())
        if use_cosine_similarity:
            embedding_model_true_answer = embedding_model.get_embedding(true_answer if None not in true_answer else data['model_answer'].tolist())
            embedding_model_reversed_problem_solution = embedding_model.get_embedding(data['reversed_problem_solution'].tolist())
            metrics_dict['cosine_similarity'] = np.mean([cosine_similarity(a['embedding'], 
                                                                           b['embedding']) 
                                                         for a, b in zip(embedding_model_reversed_problem_solution, 
                                                                         embedding_model_true_answer)])
        else:
            metrics_dict[main_metric] = data.apply(lambda x: metrics2func[main_metric]([x['reversed_problem_solution']],
                                                                                        x['question']), axis=1).mean()
    return metrics_dict

def get_problem_solution(system_prompt: str,
                         model: tp.Any,
                         question: tp.List[str]) -> tp.List[str]:
    '''
    Получение решения прямой или обратной задачи при помощи модели
    Parameters:
        system_prompt: Промпт для решения прямой или обратной задачи при помощи модели, должен содержать '{source_text}'
        model: Валидируемая модель; имеет метод predict(список вопросов)
        question: Список текстов для переписывания
    Return:
        Список переписанных текстов
    '''
    question = [system_prompt.format(source_text=q) for q in question]
    problem_solution = model.predict(question)
    return problem_solution


def cosine_similarity(first_emb: tp.List[float], 
                      second_emb: tp.List[float]) -> float:
    '''
    Расчёт косинусной близости между двумя эмбеддингами
    Parameters:
        first_emb: первый эмбеддинг, представляет из себя список чисел формата float
        second_emb: второй эмбеддинг, представляет из себя список чисел формата float
    Return:
        Косинусная близость между векторами a и b
    '''
    cosine_similarity = 1 - scipy.spatial.distance.cosine(first_emb, second_emb)
    return cosine_similarity
