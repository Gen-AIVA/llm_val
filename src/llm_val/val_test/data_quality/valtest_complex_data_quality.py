import re
import typing as tp
import pandas as pd
from copy import deepcopy
from simhash import SimhashIndex, Simhash

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import (worst_semaphore,
                                         semaphore_by_threshold,
                                         semaphore_by_threshold_without_yellow)


def report_valtest_complex_data_quality(result_dict: tp.Dict[str, float],
                                        data_types: tp.Tuple[str, str]=('train', 'test'),
                                        duplicates_semaphore_threshold: tp.Tuple[float, float]=(0.02, 0.05),
                                        contradictions_semaphore_threshold: float=0.0,
                                        intersections_semaphore_threshold: float=0.0,
                                        is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        result_dict: Словарь с посчитанными результатами теста на качество данных
        data_types: Типы данных, на которых исследуется качество данных
        duplicates_semaphore_threshold: Границы светофора для теста на уникальность ключа
        contradictions_semaphore_threshold: Границы светофора для теста на противоречивость данных
        intersections_semaphore_threshold: Границы светофора для теста на пересечение данных
            для обучения и тестирования
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    contradictions_semaphore = worst_semaphore([semaphore_by_threshold_without_yellow(value,
                                                                                      threshold=contradictions_semaphore_threshold,
                                                                                      greater_is_better=False,
                                                                                      border_is_bigger=False)
                                                for value in [result_dict['contradictions_train'], result_dict['contradictions_test']]])
    duplicates_semaphore = worst_semaphore([semaphore_by_threshold(value,
                                                                   threshold=duplicates_semaphore_threshold,
                                                                   greater_is_better=False,
                                                                   left_border_is_bigger=False,
                                                                   right_border_is_bigger=False)
                                            for value in [result_dict['duplicates_train'], result_dict['duplicates_test']]])
    intersections_semaphore = semaphore_by_threshold_without_yellow(result_dict['intersections'],
                                                                    threshold=intersections_semaphore_threshold,
                                                                    greater_is_better=False,
                                                                    border_is_bigger=False)
    color_list = [duplicates_semaphore, contradictions_semaphore, intersections_semaphore]
    color = 'gray' if is_info else worst_semaphore(color_list)
    result = pd.DataFrame({
        'Проверка': ['Доля дубликатов', 'Доля противоречий', 'Доля пересечений'],
        data_types[0]: [result_dict['duplicates_train'], result_dict['contradictions_train'], result_dict['intersections']],
        data_types[1]: [result_dict['duplicates_test'], result_dict['contradictions_test'], result_dict['intersections']],
        'Результат теста': color_list})
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [result]
    return res_dict


def valtest_complex_data_quality(sampler: Sampler,
                                 data_types: tp.Tuple[str, str]=('train', 'test'),
                                 window_size: int=5,
                                 max_bit_diff: int=2,
                                 duplicates_semaphore_threshold: tp.Tuple[float, float]=(0.02, 0.05),
                                 contradictions_semaphore_threshold: float=0.0,
                                 intersections_semaphore_threshold: float=0.0,
                                 is_info: bool=False,
                                 data_quality_dict: tp.Optional[tp.Dict[str, float]]=None,
                                 **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест проверки качества данных: вычисляются доли дубликатов и противоречий внутри каждой из выборок
        и доля пересечений между выборками
    Parameters:
        sampler: Объект класса Sampler
        data_types: Типы данных, на которых исследуется качество данных
        window_size: Длина n-грамм, на которые будут разбиваться строки с вопросами 
        max_bit_diff: Максимально допустимое отличие между строками в битах, 
            при котором они считаются равными
        duplicates_semaphore_threshold: Границы светофора для теста на уникальность ключа
        contradictions_semaphore_threshold: Границы светофора для теста на противоречивость данных
        intersections_semaphore_threshold: Границы светофора для теста на пересечение данных
        is_info: True, если тест информативный
        data_quality_dict: Опциональный словарь с предпосчитанными результатами теста на качество данных.
            Если не None, то тест не запускается, и на основе переданного аргумента выставляется светофор.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if data_quality_dict is None:
        data_train = getattr(sampler, data_types[0])
        data_test = getattr(sampler, data_types[1])
        X_train, y_train = data_train['X'], data_train['y']
        X_test, y_test = data_test['X'], data_test['y']
        if set(y_train.columns).issubset(set(X_train.columns)):
            data_quality_dict = train_test_quality_checks(X_train['question'],
                                                          X_train[y_test.columns],
                                                          X_test['question'],
                                                          X_test[y_test.columns],
                                                          window_size,
                                                          max_bit_diff)
        else:
            data_quality_dict = train_test_quality_checks(X_train['question'],
                                                          X_train[['true_answer']],
                                                          X_test['question'],
                                                          X_test[['true_answer']],
                                                          window_size,
                                                          max_bit_diff)
    precomputed = {'data_quality_dict': deepcopy(data_quality_dict)}
    report = report_valtest_complex_data_quality(data_quality_dict,
                                                 data_types,
                                                 duplicates_semaphore_threshold,
                                                 contradictions_semaphore_threshold,
                                                 intersections_semaphore_threshold,
                                                 is_info)
    return {'report': report,
            'precomputed': precomputed}


def get_ngrams(text: str,
               window_size: int=5) -> tp.List[str]:
    '''
    Функция подсчета n-грамм
    Parameters:
        text: исходная строка
        window_size: длина n-грамм
    Return:
        Массив n-грамм
    '''
    ngram_pattern = r'.' * window_size
    ngrams = re.findall(rf'(?=({ngram_pattern}))', ' '.join(text.split()))
    return ngrams


def get_hashes(text_data: pd.Series,
                window_size: int=5) -> tp.List[tp.Tuple[int, Simhash]]:
    '''
    Функция для извлечения списка хэшей
    Parameters:
        text_data: Набор текстов 
        window_size: Длина n-грамм, на которые будут разбиваться строки с вопросами 
    Return:
        Список кортежей из индексов и хэшей 
    '''
    hashes = []
    for idx, text in enumerate(text_data):
        hashes.append((str(idx), Simhash(get_ngrams(text, window_size))))
    return hashes


def get_duplicates_and_contradictions(hash_list: tp.List[tp.Tuple[int, Simhash]],
                                      answers: tp.Optional[pd.DataFrame]=None,
                                      max_bit_diff: int=2) -> tp.Tuple[tp.Set[tp.Set[int]], tp.Set[tp.Set[int]]]:
    '''
    Функция для поиска пересечений и противоречий в наборе хэшей 
    Parameters:
        hash_list: Список кортежей из индексов и хэшей вопросов
        answers: Набор корректных ответов на вопросы или метки асессоров 
        max_bit_diff: Максимально допустимое отличие между строками в битах, 
            при котором они считаются равными
    Return:
        Множество индексов дубликатов, множество противоречий --- подмножество индексов 
            дубликтатов, для которых ответы не совпали между собой
    '''
    duplicates, contradictions = [], []
    hash_index = SimhashIndex(hash_list, k=max_bit_diff)
    for i, hash_item in hash_list:
        duplicates_list = hash_index.get_near_dups(hash_item)
        i = int(i)
        if len(duplicates_list) > 1:
            duplicates_subset = []
            for dupl in duplicates_list:
                duplicates_subset.append(int(dupl))
            if answers is not None:
                curr_true = answers.iloc[i, :].values
                dupl_true_list = [answers.iloc[int(d), :].values for d in duplicates_list]
                for idx, dupl_true in enumerate(dupl_true_list):
                    if (dupl_true != curr_true).any():
                        contradictions.append(frozenset([i, int(duplicates_list[idx])]))
            duplicates.append(frozenset(duplicates_subset))
    return set(duplicates), set(contradictions)


def get_intersection(train_hash: tp.List[tp.Tuple[int, Simhash]],
                     test_hash: tp.List[tp.Tuple[int, Simhash]],
                     max_bit_diff: int=2) -> tp.Set[int]:
    '''
    Функция для поиска пересечения между двумя наборами хэшей 
    Parameters:
        train_hash: Список кортежей из индексов и хэшей вопросов из обучающей выборки
        test_hash: Список кортежей из индексов и хэшей вопросов из тестовой выборки
        max_bit_diff: Максимально допустимое отличие между строками в битах, 
            при котором они считаются равными
    Return:
        Множество индексов элементов из второго набора, 
            для которых обнаружено совпадение с некоторым элементом из первого
    '''
    intersections = set()
    train_hash_index = SimhashIndex(train_hash, k=max_bit_diff)
    for idx, hash_item in test_hash:
        if train_hash_index.get_near_dups(hash_item):
            intersections.add(int(idx))
    return intersections


def set_of_sets_size(set_of_sets: tp.Set[tp.Set[int]]) -> int:
    '''
    Подсчет размера объединения нескольких множеств 
    Parameters:
        set_of_sets: Множество множеств 
    Return:
        Размер множества состоящего из объединения элементов всех множеств 
    '''
    return len(frozenset().union(*set_of_sets))


def train_test_quality_checks(train_questions: pd.Series,
                              train_answers: pd.DataFrame,
                              test_questions: pd.Series,
                              test_answers: pd.DataFrame,
                              window_size: int=5,
                              max_bit_diff: int=2
                              ) -> tp.Dict[str, tp.Union[float, tp.Set[int], tp.Set[tp.Set[int]]]]:
    '''
    Подсчитывает доли дубликатов, противоречий и пересечений в обучающей и тестовой выборках с помощью SimHash.
    Parameters:
        train_questions: Набор вопросов из обучающей выборки
        train_answers: Набор ответов из обучающей выборки или метки асессоров 
            (может содержать одну или несколько колонок)
        test_questions: Набор вопросов из тестовой выборки
        test_answers: Набор ответов из тестовой выборки или метки асессоров 
            (может содержать одну или несколько колонок)
        window_size: Длина n-грамм, на которые будут разбиваться строки с вопросами 
        max_bit_diff: Максимально допустимое отличие между строками в битах, 
            при котором они считаются равными
    Return:
        Словарь с долями найденных дубликатов, противоречий и пересечений, 
            а также наборы соответствующих индексов
    '''
    train_questions_hash = get_hashes(train_questions, window_size)
    test_questions_hash = get_hashes(test_questions, window_size)
    duplicates_train, contradictions_train = get_duplicates_and_contradictions(train_questions_hash,
                                                                               train_answers,
                                                                               max_bit_diff)
    duplicates_test, contradictions_test = get_duplicates_and_contradictions(test_questions_hash,
                                                                             test_answers,
                                                                             max_bit_diff)
    intersections = get_intersection(train_questions_hash, test_questions_hash)
    len_train_questions, len_test_questions = len(train_questions), len(test_questions)
    return {'duplicates_train': set_of_sets_size(duplicates_train)/len_train_questions,
            'duplicates_test': set_of_sets_size(duplicates_test)/len_test_questions,
            'contradictions_train': set_of_sets_size(contradictions_train)/len_train_questions,
            'contradictions_test': set_of_sets_size(contradictions_test)/len_test_questions,
            'intersections': len(intersections)/len_train_questions,
            'duplicates_train_set': duplicates_train,
            'contradictions_train_set': contradictions_train,
            'duplicates_test_set': duplicates_test,
            'contradictions_test_set': contradictions_test,
            'intersections_set': intersections}
