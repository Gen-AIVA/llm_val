import random
import numpy as np
import pandas as pd
import typing as tp

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.paraphraser import batch_parapraser
from llm_val.ragexecutor.RAGExecutor import RAGExecutor
from llm_val.utils.word_augmentations import batch_swap_word
from llm_val.utils.human_augmentations import batch_augmentations
from llm_val.utils.report_helper import semaphore_by_threshold, worst_semaphore

def report_valtest_rag_top_stability(strict_strategy_results: tp.Dict[int, float],
                                     soft_strategy_results: tp.Dict[int, float],
                                     topK: int,
                                     soft_strategy_thresholds: tp.Tuple[float, float]=(0.1, 0.8),
                                     strict_strategy_thresholds: tp.Tuple[float, float]=(0.05, 0.7),
                                     is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        strict_strategy_results: Словарь с результатами строгой стратегии для каждого размера топа
        soft_strategy_results: Словарь с результатами мягкой стратегии для каждого размера топа
        topK: Размер топа, для которого определяется цвет светофора
        soft_strategy_thresholds: Кортеж из двух значений (нижний порог, верхний порог) для определения цвета светофора по мягкой стратегии.
        strict_strategy_thresholds: Кортеж из двух значений (нижний порог, верхний порог) для определения цвета светофора по строгой стратегии.
        is_info: True, если тест информативный.
    Return:
        Отчет в соответствии с концепцией llm_val
    '''
    res_table = pd.DataFrame({'Размер топа': list(strict_strategy_results.keys()),
                              'Строгая стратегия': list(strict_strategy_results.values()),
                              'Мягкая стратегия': list(soft_strategy_results.values()),
                              'Результат теста': 'gray'})
    
    strict_value = strict_strategy_results[topK]
    soft_value = soft_strategy_results[topK]
    if is_info:
        color = 'gray'
    else:
        strict_semaphore = semaphore_by_threshold(value=strict_value,
                                                  threshold=(strict_strategy_thresholds[0], strict_strategy_thresholds[1]),
                                                  greater_is_better=True,
                                                  left_border_is_bigger=False,
                                                  right_border_is_bigger=True)
        soft_semaphore = semaphore_by_threshold(value=soft_value,
                                                threshold=(soft_strategy_thresholds[0], soft_strategy_thresholds[1]),
                                                greater_is_better=True,
                                                left_border_is_bigger=False,
                                                right_border_is_bigger=True)
        color = worst_semaphore([strict_semaphore, 
                                 soft_semaphore])
    res_table.loc[res_table['Размер топа'] == topK, 'Результат теста'] = color
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_rag_top_stability(sampler: Sampler,
                              executor: RAGExecutor,
                              augmentations_list: tp.List[str]=['symbol_swap',
                                                                'butter_finger',
                                                                'human_level_augmentations',
                                                                'insert',
                                                                'deletion',
                                                                'word_swap',
                                                                'paraphrase'],
                              p_augmented: float=0.1,
                              data_type: str='test',
                              seed: int=42,
                              paraphrase_model: tp.Optional[tp.Any]=None,
                              paraphrase_tokenizer: tp.Optional[tp.Any]=None,
                              paraphrase_model_config: tp.Optional[tp.Dict[str, tp.Any]]=None,
                              paraphrase_tokenizer_config: tp.Optional[tp.Dict[str, tp.Any]]=None,
                              soft_strategy_thresholds: tp.Tuple[float, float]=(0.1, 0.8),
                              strict_strategy_thresholds: tp.Tuple[float, float]=(0.05, 0.7),
                              is_info: bool=False,
                              **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на стабильность выбранного ТОП-а при RAG в зависимости от аугментаций текстов
    Parameters:
        sampler: Объект класса Sampler
        executor: Объект класса RAGExecutor
        augmentations_list: Список выполняемых аугментаций
        p_augmented: Доля аугментированных слов (для аугментации методом перестановки слов)
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        seed: Начальное значение датчика случайных чисел
        paraphrase_model: Модель для перефразирования (если используется аугментация 'paraphrase')
        paraphrase_tokenizer: Токенизатор для перефразирования (если используется аугментация 'paraphrase')
        paraphrase_model_config: Конфигурация модели для перефразирования
        paraphrase_tokenizer_config: Конфигурация токенизатора для перефразирования
        soft_strategy_thresholds: Кортеж из двух значений (нижний порог, верхний порог) для определения цвета светофора по мягкой стратегии.
        strict_strategy_thresholds: Кортеж из двух значений (нижний порог, верхний порог) для определения цвета светофора по строгой стратегии.
        is_info: True, если тест информативный.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    X = getattr(sampler, data_type)['X']
    queries = X['question'].values.tolist()
    params = executor.get_parameters()
    topK = params['topK']
    original_tops_max = executor.get_topK_passages(questions=queries, 
                                                   topK=topK)
    original_tops = {k: [results['documents'][:k] for results in original_tops_max.values()] for k in range(1, topK + 1)}

    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    strict_strategy_results = {}
    soft_strategy_results = {}
    augmented_tops_dict = {}
    for aug_name in augmentations_list:
        augmented_queries = queries.copy()
        if aug_name == 'word_swap':
            augmented_queries = batch_swap_word(batch=augmented_queries, rng=rng, p=p_augmented)
        elif aug_name == 'paraphrase':
            augmented_queries = batch_parapraser(batch=augmented_queries,
                                                 rng=rng,
                                                 model=paraphrase_model,
                                                 tokenizer=paraphrase_tokenizer,
                                                 model_config=paraphrase_model_config,
                                                 tokenizer_config=paraphrase_tokenizer_config,
                                                 p=p_augmented)
        else:
            aug_params = {'use_swap': aug_name == 'symbol_swap',
                          'use_bf': aug_name == 'butter_finger',
                          'use_deletion': aug_name == 'deletion',
                          'use_insert': aug_name == 'insert',
                          'use_special_symbol': aug_name == 'human_level_augmentations'}
            augmented_queries = batch_augmentations(batch=augmented_queries,
                                                    rng=rng,
                                                    p=p_augmented,
                                                    **aug_params)

        max_topK_results = executor.get_topK_passages(questions=augmented_queries, topK=topK)
        augmented_tops_dict[aug_name] = max_topK_results

    for k in range(1, topK + 1):
        augmented_tops_list = []
        for aug_name in augmentations_list:
            max_topK_results = augmented_tops_dict[aug_name]
            current_k_results = [results['documents'][:k] for results in max_topK_results.values()]
            augmented_tops_list.append(current_k_results)
        strict_scores = []
        soft_scores = []
        for augmented_tops in augmented_tops_list:
            strict_scores.append(evaluate_tops(original_tops[k], 
                                               augmented_tops, 
                                               strategy='strict'))
            soft_scores.append(evaluate_tops(original_tops[k], 
                                             augmented_tops, 
                                             strategy='soft'))

        strict_strategy_results[k] = np.mean(strict_scores)
        soft_strategy_results[k] = np.mean(soft_scores)
    report = report_valtest_rag_top_stability(strict_strategy_results,
                                              soft_strategy_results,
                                              topK,
                                              soft_strategy_thresholds,
                                              strict_strategy_thresholds,
                                              is_info)
    return {'report': report,
            'precomputed': {'strict_strategy_results': strict_strategy_results,
                            'soft_strategy_results': soft_strategy_results}}


def strict_strategy(original_top: tp.List,
                    modified_top: tp.List) -> float:
    '''
    Функция для реализации строгой стратегии.
    Parameters:
        original_top: Cписок исходных пассажей в топе.
        modified_top: Cписок измененных пассажей в топе.
    Return:
        Если топ полностью совпадает (включая порядок), то ставим 1, иначе 0.
    '''
    return 1 if original_top == modified_top else 0


def soft_strategy(original_top: tp.List,
                  modified_top: tp.List) -> float:
    '''
    Функция для реализации мягкой стратегии.
    Parameters:
        original_top: Cписок исходных пассажей в топе.
        modified_top: Cписок измененных пассажей в топе.
    Return:
        Доля изменённых пассажей в топе.
    '''
    original_set = set(original_top)
    modified_set = set(modified_top)
    if original_set == modified_set:
        result = 1
    else:
        changed_elements = len(original_top) - len(original_set.intersection(modified_set))
        result =  1 - (changed_elements / len(original_top))
    return result


def evaluate_tops(original_tops: tp.List[tp.List], 
                  modified_tops: tp.List[tp.List], 
                  strategy: str='strict') -> float:
    '''
    Оценивает изменения топов по выбранной стратегии.
    Parameters:
        original_tops: Список списков исходных пассажей в топе.
        modified_tops: Список списков измененных пассажей в топе.
        strategy: Возможое значение стратегии оценивания ('strict' или 'soft').
    Return:       
        Среднее значение изменений по всем вопросам.
    '''
    total_score = 0
    for original_top, modified_top in zip(original_tops, modified_tops):
        if strategy == 'strict':
            total_score += strict_strategy(original_top, 
                                           modified_top)
        elif strategy == 'soft':
            total_score += soft_strategy(original_top, 
                                         modified_top)
    return total_score / len(original_tops)
