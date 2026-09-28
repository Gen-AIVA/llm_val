import re
import sys
import time
import numpy as np
import pandas as pd
import typing as tp
from simhash import Simhash
from tqdm.auto import tqdm as tqdm
from sentence_transformers import util

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold

def report_valtest_duplicates(value: float,
                              threshold: tp.Tuple[float, float]=(0.0, 0.05),
                              is_info: bool=False):
    '''
    Создание report'а
    Parameters:
        value: Доля дубликатов в выборках
        threshold: Граница светофора - допустимая доля дубликатов в выборке.
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_dict = {}
    res_dict['result_plots'] = []
    res_dict['semaphore'] = 'gray' if is_info else semaphore_by_threshold(value, 
                                                                          threshold=threshold, 
                                                                          greater_is_better=False, 
                                                                          left_border_is_bigger=False, 
                                                                          right_border_is_bigger=True)
    res_dict['result_dataframes'] = []
    return res_dict


def valtest_duplicates(sampler: Sampler,
                       sampler_type: str='MMLU',
                       data_type: str='test',
                       model: tp.Any=None,
                       threshold: tp.Tuple[float, float]=(0.0, 0.05),
                       threshold_duplicates: float=0.98,
                       threshold_contradictions: float=0.9875,
                       is_info: bool=False,
                       **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест наличия дубликатов вопросов в test выборке, а также нахождения противоречений для test выборки MMLU формата.
    Для всех типов сэмплера строится таблица расстояний пар между raw_prompt. Если model = None, то в качестве расстояния
    берется SimHash, иначе модель должна иметь метод encode (строить эмбеддинг).
    
    Дополнительные проверки Для MMLU формата: 
        1. Из таблицы дубликатов (по трешхолду считает дубликаты) дополнительно строится таблица противоречий 
           путем сравнения верных ответов у найденных пар. 
        2. Строится таблица расстояний пар между вопросом + верным ответом (по трешхолду считает дубликаты).  
        3. По таблице из пункта 2 аналогично строится таблица противоречий.
    Parameters:
        sampler: Объект класса Sampler
        sampler_type: Тип sampler'a (MMLU, RuBQ, LongAnswer и т.д.)
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        model: Класс-обертка над эмбеддинг моделью. Модель обязана иметь метод encode.
               Если в качестве модели указано None, вызывается функция SimHash.
        threshold: Граница светофора - максимально допустимая доля дубликатов в выборке.
        threshold_duplicates: Граница признания семантических дубликатов.
        treshhold_contradictions: Граница признания дубликатами вопросов для поиска противоречий.
        is_info: True, если тест информативный
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    value = 0.0
    precomputed = {}
    duplicates_tables = []
    X = getattr(sampler, data_type)['X']
    data = list(X['raw_prompt'].values)
    duplicates = duplicate_table(documents_1=data, 
                                 documents_2=data,
                                 model=model,
                                 name_loop='raw_prompt',
                                 are_same_documents_flag=True)
    duplicate_rate = len(duplicates[duplicates['similarity'] >= threshold_duplicates]) / len(data)
    precomputed['Таблица пар расстояний raw_prompt'] = duplicates
    precomputed['Доля дубликатов raw_prompt'] = duplicate_rate
    value = max(value, duplicate_rate)

    if (sampler_type.lower() == 'mmlu') and (model is not None):
        contradictions = contradiction_table(data_1=X, 
                                             data_2=X, 
                                             duplicate_table=duplicates,
                                             threshold_contradictions=threshold_contradictions)
        contradiction_nums = len(contradictions)
        if contradiction_nums > 0: 
            value = 1
        precomputed['Таблица противоречий raw_prompt'] = contradictions
        precomputed['Число противоречий raw_prompt'] = contradiction_nums

        data_qa = list(X.apply(lambda row: row['question'] + ' ' + row['answer_options'][row['true_answer']], axis=1).values)
        duplicates_qa = duplicate_table(documents_1=data_qa, 
                                        documents_2=data_qa,
                                        model=model, 
                                        name_loop='question + true answer',
                                        are_same_documents_flag=True)
        duplicate_rate_qa = len(duplicates_qa[duplicates_qa['similarity'] >= threshold_duplicates])/len(data_qa)
        value = max(value, duplicate_rate_qa)
        precomputed['Таблица пар расстояний question + answer'] = duplicates_qa
        precomputed['Доля дубликатов question + answer'] = duplicate_rate_qa

        contradictions_qa = contradiction_table(data_1=X, 
                                                data_2=X, 
                                                duplicate_table=duplicates_qa,
                                                threshold_contradictions=threshold_contradictions)
        contradiction_qa_nums = len(contradictions_qa)
        if contradiction_qa_nums > 0: 
            value = 1
        precomputed['Таблица противоречий question + answer'] = contradictions_qa
        precomputed['Число противоречий question + answer'] = contradiction_qa_nums
            
    report = report_valtest_duplicates(value=value,
                                       threshold=threshold,
                                       is_info=is_info)
    return {'report': report, 
            'precomputed': precomputed}


def duplicate_table(documents_1: tp.List[str],
                    documents_2: tp.List[str],
                    model: tp.Optional[tp.Any]=None, 
                    name_loop: str='Progress:', 
                    are_same_documents_flag: bool=True) -> pd.DataFrame:
    '''
    Функция создания таблицы семантической схожести (косинусная близость векторов эмбеддингов) между тексами. 
    Если model=None, вычисляется SimHash.
    Таблица схожести имеет 3 колонки:
    idx_1 - индекс предложения из documents_1
    idx_2 - индекс предложения из documents_2
    similarity - семантическое сходство между соответствующими текстами
    Parameters:
        documents_1: первый список текстов
        documents_2: второй список текстов
        model: Класс-обертка над эмбеддинг моделью. Модель обязана иметь метод encode.
               Если в качестве модели указано None, вызывается функция SimHash.
        name_loop: Название колонки, по которой ищутся дубликаты и происходит эмбеддинг кодирование.
                   Данное название будет отображаться в соответствующих прогресс-барах - для кодирования 
                   (если используется модель) и поиска дубликатов.
        are_same_documents_flag: Флаг, который равен True, если первый список текстов совпадает со вторым.
              Нужен для оптимизации перебора всех парных расстояний.
    Return:
        Таблицу расстояний между всеми парами текстов.
    '''
    hist = []
    end_documents_1 = len(documents_1)
    start_documents_2 = 0
    end_documents_2 = len(documents_2)

    if model is not None:
        embeddings_1 = []
        for doc in tqdm(documents_1, desc='Progress encode ' + name_loop + ' in documents: '):
            embeddings_1.append(model.encode(doc))

    if documents_1 == documents_2:
        if model is not None:
            embeddings_2 = embeddings_1
        end_documents_1 -= 1
    else:
        if model is not None:
            embeddings_2 = []
            for doc in tqdm(documents_2, desc='Progress encode ' + name_loop + ' in documents_2: '):
                embeddings_2.append(model.encode(doc))
        are_same_documents_flag = False

    with tqdm(total=end_documents_1, desc='Calculate cos similarity ' + name_loop + ': ', file=sys.stdout, mininterval=0.05) as pbar:
        for i in range(end_documents_1):
            start_documents_2 = i+1 if are_same_documents_flag is True else 0
            for j in range(start_documents_2, end_documents_2):
                if model is not None:
                    similarity = util.cos_sim(embeddings_1[i],embeddings_2[j])[0][0].item()
                else:
                    similarity = calculate_similarity(documents_1[i], documents_2[j])
                hist.append([i, j, similarity])
            pbar.update(1)
            sys.stdout.flush()

    semantic_doubles = pd.DataFrame(hist, columns = ['idx_1', 'idx_2', 'similarity']).sort_values(by='similarity', ascending=False)
    return semantic_doubles.reset_index(drop=True)


def calculate_similarity(text1: str, 
                         text2: str) -> float:
    '''
    Вычисляет схожесть между двумя текстами на основе расстояния Хэмминга их SimHash значений.
    Parameters:
        text1: Первый текст.
        text2: Второй текст.
    Returns:
        Схожесть между текстами от 0 до 1, где 1 - идентичные тексты.
    '''
    hash1 = Simhash(text1)
    hash2 = Simhash(text2)
    hamming_distance = hash1.distance(hash2)
    similarity = 1 - (hamming_distance / 64)
    return similarity


def contradiction_table(data_1: pd.DataFrame, 
                        data_2: pd.DataFrame, 
                        duplicate_table: pd.DataFrame,
                        threshold_contradictions: float) -> pd.DataFrame:
    '''
    Выдает таблицу противоречий для найденной таблицы дубликатов (для MMLU формата).
    Parameters:
        data_1: первый датасет.
        data_2: второй датасет.
        duplicate_table: таблица дубликатов, найденных по первому и второму датасету.
        threshold_contradictions: Граница признания дубликатами вопросов для поиска противоречий.
    Return:
        Таблицу противоречий
    '''
    hist = []
    filtered_duplicate_table = duplicate_table[duplicate_table['similarity'] >= threshold_contradictions]
    for _, row in filtered_duplicate_table.iterrows():
        i, j = row['idx_1'], row['idx_2']
        true_answer_1, true_answer_2 = data_1.loc[i, 'true_answer'], data_2.loc[j, 'true_answer']
        answer_1, answer_2 = data_1.loc[i, 'answer_options'][true_answer_1], data_2.loc[j, 'answer_options'][true_answer_2] 
        if remove_spaces_and_punctuation(answer_1) != remove_spaces_and_punctuation(answer_2):
            hist.append([i, j, data_1.loc[i, 'raw_prompt'], true_answer_1, data_2.loc[i, 'raw_prompt'], true_answer_2])
    return pd.DataFrame(hist, columns = ['idx_1', 'idx_2', 'raw_prompt_1', 'true_answer_1', 'raw_prompt_2', 'true_answer_2'])


def remove_spaces_and_punctuation(text: str) -> tp.List[str]:
    '''
    Удаляет из текста всю пунктуацию и длинные пробелы.
    Parameters:
        text: Входной текст
    Return:
        Текст без пунктуации и длинных пробелов.
    '''
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub('\s+', ' ', cleaned_text).strip()
    return cleaned_text.lower()
