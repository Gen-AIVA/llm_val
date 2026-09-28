import re
import os
import spacy
import numpy as np
import pandas as pd
import typing as tp
from copy import deepcopy
from num2words import num2words
from ru_word2number.w2n import word_to_num

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.ordinal_dict import (forms_of_one,
                                        ordinal_to_cardinal,
                                        ordinal_powers_of_ten, 
                                        cardinal_powers_of_ten)
from llm_val.utils.report_helper import (worst_semaphore,
                                         semaphore_by_threshold)

def report_valtest_numbers_numerals_duplicated(amount: float,
                                               duplicates_semaphore_threshold: tp.Tuple[float]=(0.0, 0.05),
                                               is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        amount: Посчитанный результат теста - доля непродублированных числительных и чисел
        duplicates_semaphore_threshold: Границы светофора для теста на дублирование
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    duplicates_semaphore = semaphore_by_threshold(amount,
                                                  threshold=duplicates_semaphore_threshold,
                                                  greater_is_better=False,
                                                  left_border_is_bigger=False,
                                                  right_border_is_bigger=False)

    color = 'gray' if is_info else duplicates_semaphore
    result = pd.DataFrame({'Проверка': ['Доля непродублированных числительных и чисел'], 
                           'Значение': [amount], 
                           'Результат теста': [color]})
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [result]
    return res_dict


def valtest_numbers_numerals_duplicated(sampler: Sampler,
                                        data_type: str='test',
                                        semaphore_threshold: tp.Tuple[str]=(0.0, 0.05),
                                        is_info: bool=False,
                                        not_duplicated_dict: tp.Optional[tp.Dict[str, float]]=None,
                                        **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест проверки RuBQ датасета на то, что числа продублированы числительными,
        а числительные продублированы числом
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых проводится тест. Hint: название атрибута в sampler'е
        semaphore_threshold: Границы выставления светофора по тесту
        is_info: True, если тест информативный
        not_duplicated_dict: Опциальный словарь с предпосчитанной долей непродублированных чисел 
            и числительных. Если не None, то тест не запускается, и  
            на основе переданного аргумента выставляется светофор
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if not_duplicated_dict is None:
        test_data = getattr(sampler, data_type)
        test_answers = test_data['X']['true_answer']
        not_duplicated_dict = get_not_duplicated_nums(test_answers)
        
    report = report_valtest_numbers_numerals_duplicated(not_duplicated_dict['amount'],
                                                        semaphore_threshold,
                                                        is_info)
    return {'report': report,
            'precomputed': not_duplicated_dict}


def get_not_duplicated_nums(data: pd.Series) -> tp.Dict[str, str]:
    '''
    Функция, возвращающая словарь с информацией о продублированных и
        непродублированных числительных и числах
    Parameters:
        data: pd.Series с ответами
    Return:
        Словарь с непродублированными числами, продублированными и 
            доля непродублированных чисел 
    '''
    duplicated = []
    not_duplicated = []
    for i in range(len(data)):
        not_dupl, dupl = get_numbers_numerals_identification(data.iloc[i])
        duplicated.append(dupl)
        not_duplicated.append(not_dupl)
    amount_not_duplicated = np.sum([len(x) for x in not_duplicated])
    amount_duplicated = np.sum([len(x) for x in duplicated])
    amount_numbers_numerals = amount_not_duplicated + amount_duplicated
    not_duplicated_rate = amount_not_duplicated / amount_numbers_numerals
    result = {'not duplicated': not_duplicated,
              'duplicated': duplicated,
              'amount': not_duplicated_rate}
    return result


def get_numbers_numerals_identification(answers: tp.List[str]) -> tp.Tuple[tp.Dict[str, str], tp.Dict[str, str]]:
    '''
    Функция для нахождения непродублированных и продублированных числительных и чисел в ответе
    Parameters:
        answers: Список с ответами
    Return:
        Tuple из двух словарей. Hint: в словаре not_duplicated по ключу хранится числительное или число,
            а по значению то, как оно должно быть продублировано 
    '''
    numbers = set()
    numerals = set()
    for ans in answers:
        cur_numbers = set(find_all_numbers(ans))
        cur_numerals = set(find_all_numerals(ans))
        numbers = numbers.union(cur_numbers)  
        numerals = numerals.union(cur_numerals)
    not_duplicated = {}
    duplicated = {}
    for num in numbers:
        if not num_to_word(num) in numerals:
            not_duplicated[num] = num_to_word(num)
        else:
            duplicated[num_to_word(num)] = num
    for num in numerals:
        if not str(word_to_num(num)) in numbers:
            not_duplicated[num] = str(word_to_num(num))
        else:
            duplicated[num] = str(word_to_num(num))
    return not_duplicated, duplicated
 
    
def remove_spaces_between_digits(input_string: str) -> str:
    '''
    Функция для удаления пробелов в числах
    Parameters:
        input_string: Строка, в которой нужно удалить пробелы между цифрами в числах
    Return:
        Строка с удалёнными пробелами между цифрами в числах
    '''
    return re.sub(r'(?<=\d) (?=\d)', '', input_string)


def split_by_digits(num_str: str) -> tp.Dict[str, str]:
    '''
    Функция для разделения порядков числительного
    Parameters:
        num_str: Строка с числительным
    Return:
        Словарь с порядком и его числом
    '''
    digit = {}
    for suffix in ordinal_powers_of_ten:
        pattern = r'(.*?)(' + suffix + r')$'
        match = re.match(pattern, num_str)
        if match:
            prefix = match.group(1) 
            digit[suffix] = prefix
    return digit


def find_all_numbers(text: str) -> tp.List[str]:
    '''
    Функция для поиска чисел в тексте
    Parameters:
        text: Исходная строка
    Return:
        Список с числами из исходной строки
    '''
    processed_txt = remove_spaces_between_digits(text)
    pattern = r'[-+]?[0-9]*[.,]?[0-9]+(?:[eE][-+]?[0-9]+)??'
    nums = re.findall(pattern, processed_txt)
    return nums


def preprocess_numerals(text: str,
                        nlp: tp.Any) -> str:
    '''
    Функция для препроцессинга строки для поиска числительных: 
        приведение к нижнему регистру, лемматизация, выделение порядка для числительных
    Parameters:
        text: Исходная строка
        nlp: Пайплайн для обработки русского языка из SpaCy
    Return:
        Преобразованная строка
    '''
    text = text.lower()
    text = re.sub(r'\d', '', text)
    text = ' '.join(token.lemma_ for token in nlp(text))
    doc = nlp(text)
    preprocessed = []
    for token in doc:
        order = split_by_digits(token.text)
        if order:
            suffix, prefix = next(iter(order.items()))
            preprocessed.append(' '.join([prefix, suffix]))
        else:
            preprocessed.append(token.text)
    return ' '.join(preprocessed)


def find_all_numerals(text: str) -> tp.List[str]:
    '''
    Функция для нахождения в строке количественных и порядковых числительных
    Parameters:
        text: Исходная строка
    Return:
        Список строк из числительных
    '''
    nlp = spacy.load(os.environ['ru_spacy'])
    text = preprocess_numerals(text, nlp)
    doc = nlp(text)
    numerals = []
    current_number = []
    for token in doc:
        if token.pos_ == 'NUM' or token.text in cardinal_powers_of_ten:
            current_number.append(token.text)
        elif token.text in ordinal_to_cardinal:
            current_number.append(ordinal_to_cardinal[token.text])
        else:
            if current_number:
                numerals.append(' '.join(clear_long_numerals(current_number)))
                current_number = []
    if current_number:
        numerals.append(' '.join(clear_long_numerals(current_number)))
    return numerals        


def num_to_word(num: str) -> str:
    '''
    Функция для перевода числа в слово
    Parameters:
        num: Число
    Return:
        Строка с записью числа в виде слов
    '''
    words = num2words(num, lang = 'ru').split()
    cleared_words = ' '.join(words)
    if len(words) > 1 and words[0] in forms_of_one:
        cleared_words = ' '.join(words[1:])
    return cleared_words


def clear_long_numerals(words: tp.List[str]) -> str:
    '''
    Функция для очистки больших числительных от грамматических форм слова "один" в начале
    Parameters:
        words: Список, в котором записаны слова, из которых состоит числительное
    Return:
        Строка с записью очищенного числительного
    '''
    cleared_words = words
    if len(words) > 1 and words[0] in forms_of_one:
        cleared_words = words[1:]
    return cleared_words
