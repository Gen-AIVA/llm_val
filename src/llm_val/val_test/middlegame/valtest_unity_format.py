import re
import unicodedata
import numpy as np
import pandas as pd
import typing as tp
import matplotlib.pyplot as plt
from collections import defaultdict

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold
from llm_val.utils.speech_quality.spelling import spelling_score_api

def report_valtest_unity_format(potential_errors: pd.DataFrame,
                                statistic_table: tp.Dict[str, float],
                                model: tp.Any=None,
                                threshold: tp.Tuple[float,float]=(0.0, 0.5),
                                score_name: str='Word error rate',
                                is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        potential_errors: Таблица потенциальных ошибок.
        statistic_table: Словарь, ключами которого являются:
                         Num errors - число найденных вопросов с ошибками
                         Error rate - доля найденных вопросов с ошибками
                         Word error rate - wer (если model not None)
                         Levenshtein distance - расстояние Левенштейна (если model not None)
        model: Класс-обертка над API моделью для исправления ошибок, имеет метод get_answer_sage.
               Если передать None, то орфография проверяться не будет.
        threshold: Границы светофора (чем меньше, тем лучше).
                   Левая граница соответсвует допустимой границе ошибок для зеленого светофора,
                   правая граница соответсвует допустимой границе ошибок для желтого светофора.
        score_name: Название метрики, подсчитывающей долю орфографических ошибок, по которой выставляется светофор.
                    Поддерживаемые метрики: 'Word error rate' (по умолчанию), 'Levenshtein distance'
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    value = 1.0 if len(potential_errors) > 0 else 0
    if model is not None:
        error_score = statistic_table[score_name][0]
        value = max(value, error_score)
    color = semaphore_by_threshold(value, 
                                   threshold=threshold, 
                                   greater_is_better=False, 
                                   left_border_is_bigger=False, 
                                   right_border_is_bigger=True)
    res_dict = {}
    res_dict['result_plots'] = []
    res_dict['semaphore'] = 'gray' if is_info else color
    res_dict['result_dataframes'] = [potential_errors, pd.DataFrame(statistic_table)]
    return res_dict


def valtest_unity_format(sampler: Sampler,
                         data_type: str='test',
                         sampler_type: str='MMLU', 
                         model: tp.Any=None,
                         threshold: tp.Tuple[float,float]=(0.0, 0.5),
                         min_symbols: int=5, 
                         min_words: int=3, 
                         min_true_answers: int=2, 
                         languages:  tp.List[str]=['RU', 'EN'], 
                         normalization: bool=True,
                         score_name: str='Word error rate',
                         is_info: bool=False,
                         cache_api: tp.Dict[str, str]={}) -> tp.Dict[str, tp.Any]:
    '''
    Тест на единство формата для сэмплера. 
    Обрабатывание некорректных букв алфавита происходит путем создание таблицы потенциальных ошибок, 
    в которой находятся номера предложений и комментарии к найденным ошибкам.
    Тест включает в себя:
    Для MMLU формата:
    1. Проверку алфавита верных ответов true_answers. Все буквы должны быть из пула верных ответов.
    2. Проверку алфавита пула верных ответов. Пул ответов одинаковый по составу, по длине и представлен по порядку. 
    3. Проверку, что вопрос и ответы не содержат спецсимволов (соблюдение utf-8)
    4. Проверку, что есть вопрос (содержит по крайней мере min_symbols символов без пробелов и число слов больше min_words)
       Либо некорректно сработал сэмплер, либо вопрос отсутствовал изначально.
    5. Проверку, что все представленные ответы не пустые. 
       Либо некорректно сработал сэмплер, либо хотя бы один из ответов отсутствовал изначально.

    Для LongAnswer формата:
    1. Проверку, что вопрос не содержит спецсимволов (соблюдение utf-8)
    2. Проверку, что есть вопрос (содержит по крайней мере min_symbols символов без пробелов и число слов больше min_words)
       Либо некорректно сработал сэмплер, либо вопрос отсутствовал изначально.
    
    Для RuBQ формата:
    1. Проверку, что вопрос не содержит спецсимволов (соблюдение utf-8)
    2. Проверку, что есть вопрос (содержит по крайней мере min_symbols символов без пробелов и число слов больше min_words)
       Либо некорректно сработал сэмплер, либо вопрос отсутствовал изначально.
    3. Проверку, что все представленные ответы в списке не пустые, а также число представленных ответов больше min_true_answers
    Paremeters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        sampler_type: Тип sampler'a (MMLU, RuBQ, LongAnswer)
        model: Класс-обертка над API моделью для исправления ошибок, имеет метод get_answer_sage. 
               Если передать None, то орфография проверяться не будет.
        threshold: Границы светофора (чем меньше, тем лучше).
                   Левая граница соответсвует допустимой границе ошибок для зеленого светофора,
                   правая граница соответсвует допустимой границе ошибок для желтого светофора.
        min_symbols: минимальное число символов в вопросе без пунктуации и пробелов
        min_words:   минимальное число слов в вопросе без пунктуации и пробелов
        min_true_answers: минимальное число слов в предложенном списке ответов (RuBQ формат)
        languages: список названий языков, символы которых могут встречаться в raw_prompt, поддерживается: 
            RU - русский язык
            EN - английский язык
            ZH - китайский язык
            BE - белорусский язык
            KZ - казахский язык
        normalization: True, если используем нормировку в расстоянии Левенштейна, False  в противном случае
        score_name: Название метрики, подсчитывающей долю орфографических ошибок, по которой выставляется светофор.
                    Поддерживаемые метрики: 'Word error rate' (по умолчанию), 'Levenshtein distance'
        is_info: True, если тест информативный
        cache_api: Словарь для кэширования работы модели исправления орфографии. 
                   Если параметр не передать, словарь создастся автоматически и будет возвращен в precomputed.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    X = getattr(sampler, data_type)['X']
    allowed_symbols = allowed_characters(languages=languages)
    if sampler_type == 'MMLU':
        potential_errors = {'Num': [], 
                            'true_answers': [], 
                            'pool_answers': [], 
                            'encoding': [],
                            'question': [],
                            'answers': []
                            }
        answers_letters = search_pool(data=X)
        potential_errors = unity_format_mmlu(data=X, 
                                             answers_letters=answers_letters,
                                             potential_errors=potential_errors,
                                             allowed_symbols=allowed_symbols,
                                             min_symbols=min_symbols,
                                             min_words=min_words)
    elif sampler_type == 'LongAnswer':
        potential_errors = {'Num': [], 
                            'encoding': [],
                            'question': []
                        }
        potential_errors = unity_format_longanswer(data=X, 
                                                   potential_errors=potential_errors,
                                                   allowed_symbols=allowed_symbols,
                                                   min_symbols=min_symbols,
                                                   min_words=min_words)
    elif sampler_type == 'RuBQ':
        potential_errors = {'Num': [], 
                            'encoding': [],
                            'question': [],
                            'true_answers': []
                        }
        potential_errors = unity_format_rubq(data=X, 
                                             potential_errors=potential_errors,
                                             allowed_symbols=allowed_symbols,
                                             min_symbols=min_symbols,
                                             min_words=min_words,
                                             min_true_answers=min_true_answers)
    else:
        raise KeyError(f'Некорректный тип сэмплера: {sampler_type}. Поддерживаются только MMLU, RuBQ, LongAnswer.')

    statistic_table = {'Num errors': [len(potential_errors)], 
                       'Error rate': [len(potential_errors)/len(X)]}
    if model is not None:
        text_list = list(X['raw_prompt'])
        wer, lev_dist = spelling_score_api(text_list=text_list, 
                                           model=model,
                                           normalization=normalization, 
                                           cache_api=cache_api)
        statistic_table.update({'Word error rate': [wer]})
        statistic_table.update({'Levenshtein distance': [lev_dist]})
    report = report_valtest_unity_format(potential_errors=potential_errors,
                                         model=model,
                                         threshold=threshold,
                                         statistic_table=statistic_table,
                                         score_name=score_name,
                                         is_info=is_info)
    precomputed = {'cache_sage_api': cache_api}
    return {'report': report, 
            'precomputed': precomputed}


def unity_format_rubq(data: pd.DataFrame,
                      potential_errors: tp.Dict[str, tp.List[tp.Any]],
                      allowed_symbols:  tp.Set[str],
                      min_symbols: int=5, 
                      min_words: int=3,
                      min_true_answers: int=2) -> pd.DataFrame:
    '''
    Функция глобального поиска ошибок RuBQ сэмплера, описанных в valtest_unity_format, путем заполнения таблицы potential_errors.
    Parameters:
        data: Набор данных RuBQ-формата, хранящийся в сэмплере
        potential_errors: Таблица потенциальных ошибок
        allowed_symbols: Множество допустимых символов в тексе
        min_symbols: минимальное число символов в вопросе без пунктуации и пробелов
        min_words:   минимальное число слов в вопросе без пунктуации и пробелов
        min_true_answers: минимальное число слов в предложенном списке ответов (RuBQ формат)
    Return:
        Таблицу потенциальных ошибок, найденных в датасете.
    '''
    data.apply(lambda row: checking_unity_format_rubq_rules(row=row,
                                                            potential_errors=potential_errors,
                                                            allowed_symbols=allowed_symbols,
                                                            min_symbols=min_symbols,
                                                            min_words=min_words,
                                                            min_true_answers=min_true_answers), axis=1)
    potential_errors = pd.DataFrame(potential_errors)
    potential_errors.set_index('Num', inplace=True)
    return potential_errors


def checking_unity_format_rubq_rules(row: pd.Series,
                                     potential_errors: tp.Dict[str, tp.List[tp.Any]],
                                     allowed_symbols:  tp.Set[str],
                                     min_symbols: int=5,
                                     min_words: int=3, 
                                     min_true_answers: int=2) -> None:
    '''
    Функция локального поиска ошибок одной строки RuBQ сэмплера, описанных в valtest_unity_format. 
    Parameters:
        row: Одна строка данных RuBQ-формата, хранящаяся в сэмплере
        potential_errors: Таблица потенциальных ошибок
        allowed_symbols: Множество допустимых символов в тексе
        min_symbols: минимальное число символов в вопросе без пунктуации и пробелов
        min_words:   минимальное число слов в вопросе без пунктуации и пробелов
        min_true_answers: минимальное число слов в предложенном списке ответов (RuBQ формат)
    Return:
        Ничего не выводит, обновляет словарь потенциальных ошибок.
    '''
    flag, comment = False, ''
    comment_question = 'Отсутствует корректный вопрос.'
    comment_encoding = 'Спецсимолы: {}.'
    comment_true_answers = 'Некорректный true_answer.'
    local_potential_errors = {'Num': [],  
                              'encoding': [],
                              'question': [],
                              'true_answers': []
                              }
    flagQ_nums = len(remove_spaces_and_punctuation(row['question'])) < min_symbols
    flagQ_words = len(remove_spaces(row['question'])) < min_words
    if flagQ_nums or flagQ_words:
        local_potential_errors['question'].append(comment_question)
        flag = True
    flag_encoding, unknown_symbols = is_valid(text=row['raw_prompt'], 
                                             allowed_characters=allowed_symbols)
    if not flag_encoding:
        local_potential_errors['encoding'].append(comment_encoding.format(unknown_symbols))
        flag = True
    
    true_answers_rubq = [remove_spaces_and_punctuation(ans) for ans in row['true_answer']]
    if (len(true_answers_rubq) < min_true_answers) or ('' in true_answers_rubq):
        local_potential_errors['true_answers'].append(comment_true_answers)
        flag = True
    
    if flag is True:
        local_potential_errors['Num'].append(row.name)
        local_potential_errors = dict_values_to_ravel(local_potential_errors)
        potential_errors = merge_2_dicts(potential_errors, local_potential_errors)

    return None


def unity_format_longanswer(data: pd.DataFrame,
                            potential_errors: tp.Dict[str, tp.List[tp.Any]],
                            allowed_symbols:  tp.Set[str],
                            min_symbols: int=5, 
                            min_words: int=3) -> pd.DataFrame:
    '''
    Функция глобального поиска ошибок LongAnswer сэмплера, описанных в valtest_unity_format, путем заполнения таблицы potential_errors.
    Parameters:
        data: Набор данных LongAnswer-формата, хранящийся в сэмплере
        potential_errors: Таблица потенциальных ошибок
        allowed_symbols: Множество допустимых символов в тексе
        min_symbols: минимальное число символов в вопросе без пунктуации и пробелов
        min_words:   минимальное число слов в вопросе без пунктуации и пробелов
    Return:
        Таблицу потенциальных ошибок, найденных в датасете.
    '''
    data.apply(lambda row: checking_unity_format_longanswer_rules(row=row,
                                                                  potential_errors=potential_errors,
                                                                  allowed_symbols=allowed_symbols,
                                                                  min_symbols=min_symbols,
                                                                  min_words=min_words), axis=1)
    potential_errors = pd.DataFrame(potential_errors)
    potential_errors.set_index('Num', inplace=True)
    return potential_errors


def checking_unity_format_longanswer_rules(row: pd.Series,  
                                           potential_errors: tp.Dict[str, tp.List[tp.Any]],
                                           allowed_symbols:  tp.Set[str],
                                           min_symbols: int=5,
                                           min_words: int=3) -> None:
    '''
    Функция локального поиска ошибок одной строки LongAnswer сэмплера, описанных в valtest_unity_format. 
    Parameters:
        row: Одна строка данных LongAnswer-формата, хранящаяся в сэмплере
        potential_errors: Таблица потенциальных ошибок
        allowed_symbols: Множество допустимых символов в тексе
        min_symbols: минимальное число символов в вопросе без пунктуации и пробелов
        min_words:   минимальное число слов в вопросе без пунктуации и пробелов
    Return:
        Ничего не выводит, обновляет словарь потенциальных ошибок.
    '''
    flag, comment = False, ''
    comment_question = 'Отсутствует корректный вопрос.'
    comment_encoding = 'Спецсимволы: {}.'
    local_potential_errors = {'Num': [],  
                              'encoding': [],
                              'question': [],
                              }
    flagQ_nums = len(remove_spaces_and_punctuation(row['question'])) < min_symbols
    flagQ_words = len(remove_spaces(row['question'])) < min_words
    if flagQ_nums or flagQ_words:
        local_potential_errors['question'].append(comment_question)
        flag = True
    flag_encoding, unknown_symbols = is_valid(text=row['raw_prompt'], 
                                             allowed_characters=allowed_symbols)
    if not flag_encoding:
        local_potential_errors['encoding'].append(comment_encoding.format(unknown_symbols))
        flag = True
    
    if flag is True:
        local_potential_errors['Num'].append(row.name)
        local_potential_errors = dict_values_to_ravel(local_potential_errors)
        potential_errors = merge_2_dicts(potential_errors, local_potential_errors)

    return None



def unity_format_mmlu(data: pd.DataFrame, 
                      answers_letters: tp.List[str], 
                      potential_errors: tp.Dict[str, tp.List[tp.Any]],
                      allowed_symbols:  tp.Set[str],
                      min_symbols: int=5, 
                      min_words: int=3) -> pd.DataFrame:
    '''
    Функция глобального поиска ошибок MMLU сэмплера, описанных в valtest_unity_format, путем заполнения таблицы potential_errors.
    Parameters:
        data: Набор данных MMLU-формата, хранящийся в сэмплере
        answers_letters: Список букв пула ответов
        potential_errors: Таблица потенциальных ошибок
        allowed_symbols: Множество допустимых символов в тексе
        min_symbols: минимальное число символов в вопросе без пунктуации и пробелов
        min_words:   минимальное число слов в вопросе без пунктуации и пробелов
    Return:
        Таблицу потенциальных ошибок, найденных в датасете.
    '''
    data.apply(lambda row: checking_unity_format_mmlu_rules(row=row, 
                                                            answers_letters=answers_letters,
                                                            potential_errors=potential_errors,
                                                            allowed_symbols=allowed_symbols,
                                                            min_symbols=min_symbols,
                                                            min_words=min_words), axis=1)
    potential_errors = pd.DataFrame(potential_errors)
    potential_errors.set_index('Num', inplace=True)
    return potential_errors


def checking_unity_format_mmlu_rules(row: pd.Series, 
                                     answers_letters: tp.List[str], 
                                     potential_errors: tp.Dict[str, tp.List[tp.Any]],
                                     allowed_symbols:  tp.Set[str],
                                     min_symbols: int=5,
                                     min_words: int=3) -> None:
    '''
    Функция локального поиска ошибок одной строки MMLU сэмплера, описанных в valtest_unity_format. 
    Parameters:
        row: Одна строка данных MMLU-формата, хранящаяся в сэмплере
        answers_letters: Список букв пула ответов
        potential_errors: Таблица потенциальных ошибок
        allowed_symbols: Множество допустимых символов в тексе
        min_symbols: минимальное число символов в вопросе без пунктуации и пробелов
        min_words:   минимальное число слов в вопросе без пунктуации и пробелов
    Return:
        Ничего не выводит, обновляет словарь потенциальных ошибок.
    '''
    flag, comment = False, ''
    pool_length = len(answers_letters)
    comment_pool_1 = 'Число представленных ответов = {}. Должно быть {}.'
    comment_pool_2 = 'Некорректная буква {} в пуле ответов.'
    comment_true_answers = 'Некорректная буква {} в true_answer.'
    comment_answers = 'Пустой ответ под буквой {}.'
    comment_question = 'Отсутствует корректный вопрос.'
    comment_encoding = 'Спецсимволы: {}.'
    local_potential_errors = {'Num': [], 
                              'true_answers': [], 
                              'pool_answers': [], 
                              'encoding': [],
                              'question': [],
                              'answers': []
                              }

    if len(row['answer_options'].keys()) != pool_length:
        local_potential_errors['pool_answers'].append(comment_pool_1.format(len(row['answer_options'].keys()), pool_length))
        flag = True

    for i, letter_in_answer_options in enumerate(row['answer_options'].keys()):
        if ((i >= pool_length) or (letter_in_answer_options != answers_letters[i])): 
            local_potential_errors['pool_answers'].append(comment_pool_2.format(letter_in_answer_options))
            flag = True
        if len(remove_spaces_and_punctuation(row['answer_options'][letter_in_answer_options])) == 0:
            local_potential_errors['answers'].append(comment_answers.format(letter_in_answer_options))
            flag = True
    
    if row['true_answer'] not in answers_letters:
        local_potential_errors['true_answers'].append(comment_true_answers.format(row['true_answer']))
        flag = True

    flagQ_nums = len(remove_spaces_and_punctuation(row['question'])) < min_symbols
    flagQ_words = len(remove_spaces(row['question'])) < min_words
    if flagQ_nums or flagQ_words:
        local_potential_errors['question'].append(comment_question)
        flag = True

    flag_encoding, unknown_symbols = is_valid(text=row['raw_prompt'], 
                                             allowed_characters=allowed_symbols)
    if not flag_encoding:
        local_potential_errors['encoding'].append(comment_encoding.format(unknown_symbols))
        flag = True
    
    if flag is True:
        local_potential_errors['Num'].append(row.name)
        local_potential_errors = dict_values_to_ravel(local_potential_errors)
        potential_errors = merge_2_dicts(potential_errors, local_potential_errors)
    
    return None


def search_pool(data: pd.DataFrame) -> tp.List[str]:
    '''
    Функция поиска пула ответов на вопросы. 
    Parameters:
        row: Строка датасета
    Return:
        Наиболее распространенный пул ответов (считаем его верным).
    '''
    pool = defaultdict(int)
    for answer_options in data['answer_options']:
        keys = tuple(answer_options.keys())
        pool[keys] += 1
    most_common_pool = list(max(pool, key=pool.get))
    return most_common_pool


def remove_spaces_and_punctuation(text: str) -> str:
    '''
    Удаляет из текста все пробелы и пунктуацию.
    Parameters:
        text: Входной текст
    Return:
        Текст без пробелов и пунктуации.
    '''
    cleaned_text = re.sub(r'[^\w]', '', text)
    return cleaned_text


def remove_spaces(text: str) -> tp.List[str]:
    '''
    Удаляет из текста все пробелы.
    Parameters:
        text: Входной текст
    Return:
        Список слов текста.
    '''
    cleaned_text = re.sub('\s+', ' ', text).strip(' ').split(' ')
    return cleaned_text


def dict_values_to_ravel(dict: tp.Dict[str, tp.List[str]]) -> tp.Dict[str, tp.List[str]]:
    '''
    Делает ключи словаря списками длины 1, склеивая все комментарии ошибок локального словаря для одного запроса.
    Parameters:
        dict: Локальный словарь ошибок для одного запроса
    Return:
        Словарь, ключами которого являются списки единичной длины.
    '''
    dict_to_ravel = {}
    for key, value in dict.items():
        if (len(value) > 0) and (type(value[0]) is str):
            dict_to_ravel[key] = [' '.join(value).strip(' ')]
        else:
            dict_to_ravel[key] = value
    return dict_to_ravel


def merge_2_dicts(base_dict: tp.Dict[str, tp.Any], 
                  local_dict: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    '''
    Функция склейки двух словарей.
    Parameters:
        base_dict: Глобальный словарь, который содержит найденные ошибки
        local_dict: Локальный словарь ошибок, найденный в рамках одной стоки датасета.
    Return:
        Обновленный глобальный словарь ошибок.
    '''
    for key, value in local_dict.items():
        if len(value) > 0:
            base_dict[key] += value
        else:
            base_dict[key] += [None]
    return base_dict


def allowed_characters(languages: tp.List[str] = ['RU', 'EN']) -> tp.Set[str]:
    '''
    Выдает множество допустимых символов, которые могут встречаться в тексте
    Parameters:
        languages: список названий языков, символы которых могут встречаться в raw_prompt, поддерживается: 
            RU - русский язык
            EN - английский язык
            ZH - китайский язык
            BE - белорусский язык
            KZ - казахский язык
    Return:
        Множество допустимых символов
    '''
    PUNCTUATION = [
        "’", "'", '"', "`",   
        "(", ")", "[", "]",   
        "{", "}", "<", ">",   
        ":",                  
        ",",                  
        "‒", "–", "—", "―",   
        "…",                  
        "!",                  
        ".",                  
        "«", "»",             
        "-", "‐", "+",        
        "?",                  
        "‘", "’", "“", "”",   
        ";",                  
        "/",                  
        "⁄",                  
        " ",                  
        "·",                  
        "&",                  
        "@",                  
        "*",                  
        "\\",                 
        "•",                  
        "^",                  
        "$", "€", "£",        
        "†", "‡",             
        "°",                  
        "¬",                  
        "#",                  
        "№",                  
        "%",                  
        "¶",                  
        "′",                  
        "§",                  
        "~",                  
        "¨",                  
        "_",                  
        "\n",                 
        "=",                  
        ">",                 
        "<",                  
        "|"                   
    ]
    PUNCTUATION = set(PUNCTUATION)
    numbers = set(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    extended_punctuation = {
        chr(i) for i in range(0x110000)
        if unicodedata.category(chr(i)).startswith(('L', 'N', 'P', 'Z')) and
           unicodedata.normalize('NFKC', chr(i)) in PUNCTUATION
    } | PUNCTUATION

    russian_alphabet = {
        *map(chr, range(ord('А'), ord('я') + 1)),  
        'Ё', 'ё'                                 
    }

    english_alphabet = {
        *map(chr, range(ord('A'), ord('Z') + 1)),
        *map(chr, range(ord('a'), ord('z') + 1))
    }

    belarusian_alphabet = {
        *map(chr, range(ord('А'), ord('я') + 1)),  
        'Ё', 'ё', 'І', 'і', 'Ў', 'ў'               
    }

    kazakh_alphabet = {
        *map(chr, range(ord('А'), ord('я') + 1)), 
        'Ё', 'ё', 'Ә', 'ә', 'Ғ', 'ғ', 'Қ', 'қ', 'Ң', 'ң', 'Ө', 'ө', 'Ұ', 'ұ', 'Ү', 'ү', 'Һ', 'һ'
    }

    chinese_alphabet = {chr(i) for i in range(0x4E00, 0x9FFF + 1)}
    
    dict_languages = {
        'RU': russian_alphabet,
        'EN': english_alphabet, 
        'ZH': chinese_alphabet, 
        'BE': belarusian_alphabet,
        'KZ': kazakh_alphabet
    }
    
    languages_chars = [dict_languages[language] for language in languages]
    languages_chars = set().union(*languages_chars)
    
    characters = (
        extended_punctuation |
        numbers|
        languages_chars
    )
    
    return characters


def is_valid(text: str, 
             allowed_characters: tp.Set[str]) -> tp.Tuple[bool, tp.List[str]]:
    '''
    Проверка текста на содержание спецсимволов.
    Parameters:
        text: Входной текст
    Return:
        True - если нет спецсимволов в тексте, в противном случае False, и список найденных спецсимволов
    '''
    invalid_chars = [char for char in text if char not in allowed_characters]
    if not invalid_chars:
        return True, invalid_chars
    return False, invalid_chars