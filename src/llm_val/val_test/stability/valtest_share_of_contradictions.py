import sys
import numpy as np
import pandas as pd
import typing as tp
from copy import deepcopy
from simhash import Simhash
from tqdm.auto import tqdm as tqdm
from sentence_transformers import util

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold


def report_share_of_contradictions(rates: dict,
                                   threshold: tp.Tuple[float, float]=(0.01, 0.05),
                                   is_info: bool=False) -> dict:
    '''
    Создание report'а
    Parameters:
        rates: dict с расчитанными итоговыми статистиками.
        threshold: Граница светофора - максимально допустимая доля противоречий в ответах асессоров.
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_dict = {}
    res_dict['result_plots'] = []
    if len(rates.keys()) != 1 and not is_info:
        value = rates['Доля противоречий меток ассесоров в одних и тех же ответах на те же вопросы'][0]
        res_dict['semaphore'] = semaphore_by_threshold(value, 
                                                       threshold=threshold, 
                                                       greater_is_better=False, 
                                                       left_border_is_bigger=False, 
                                                       right_border_is_bigger=True)
    else: 
        res_dict['semaphore'] = 'gray'
    results = pd.DataFrame(rates)
    res_dict['result_dataframes'] = [results]
    return res_dict


def valtest_share_of_contradictions(sampler: Sampler,
                                    data_types: tp.Tuple[str, str]=('test', 'oot'),
                                    embedding_model: tp.Any=None,
                                    threshold: tp.Tuple[float, float]=(0.01, 0.05),
                                    threshold_duplicates: float=0.98,
                                    is_info: bool=False,
                                    **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на долю противоречий в выборках out-of-sample и out-of-time. 
    Parameters:
        sampler: Объект класса Sampler
        data_types: Тип выборок, на которых производится сравнение ответов асессоров. Hint: название атрибута в sampler'е
        embedding_model: Класс-обертка над эмбеддинг моделью. Модель обязана иметь метод encode.
            Метод encode должен формировать эмбеддинг от текста, близость полученных эмбедингов определяется с помощью косинусного расстояния.
            Если в качестве модели указано None, вызывается функция SimHash и между полученными хэшами расчитывается расстояние Хэмминга.
        threshold: Граница светофора - максимально допустимая доля противоречий в ответах асессоров.
        threshold_duplicates: Граница признания семантических дубликатов по используемой метрике.
        is_info: True, если тест информативный
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    precomputed = {}
    oos = deepcopy(getattr(sampler, data_types[0])['X'])
    oot = deepcopy(getattr(sampler, data_types[1])['X'])
    answer_oos = deepcopy(getattr(sampler, data_types[0])['y'])
    answer_oot = deepcopy(getattr(sampler, data_types[1])['y'])
    data_oos = list(oos['question'].values)
    data_oot = list(oot['question'].values)
    duplicates = duplicate_table(documents_1=data_oos, 
                                 documents_2=data_oot,
                                 embedding_model=embedding_model,
                                 name_loop='question')
    duplicate_rate = len(set(duplicates[duplicates['similarity'] >= threshold_duplicates]['idx_2'])) / len(oot)
    precomputed['Таблица пар расстояний question'] = duplicates
    precomputed['Доля дубликатов question'] = duplicate_rate

    rates = {}
    rates['Доля семантических дубликатов вопросов в OOT относительно OOS'] = [duplicate_rate]
    if set(answer_oos.columns) <= set(oos.columns):
        duplicates_filtered = duplicates[duplicates['similarity']  >= threshold_duplicates]
        
        most_common_questions = find_most_common_questions(duplicates_filtered)
        contradictions = contradiction_table_most_common(answer_oos, 
                                                         answer_oot, 
                                                         most_common_questions) / len(oot)
        contradictions_rate = contradictions.loc[0, 'Средняя ошибка']
        precomputed['Таблица противоречий асессоров по критериям'] = contradictions
        data_qa_oos = list(oos['model_answer'].values)
        data_qa_oot = list(oot['model_answer'].values)
        duplicates_qa = duplicate_table(documents_1=data_qa_oos, 
                                        documents_2=data_qa_oot,
                                        embedding_model=embedding_model, 
                                        name_loop='question + true answer')

        similarity_by_qa = duplicates_qa.merge(duplicates, on = ['idx_1', 'idx_2'], suffixes = ['_q', '_a'])
        duplicates_qa_rate = len(set(similarity_by_qa[(similarity_by_qa['similarity_q'] >= threshold_duplicates) & (similarity_by_qa['similarity_a'] >= threshold_duplicates)]['idx_2'])) / len(oot)
        duplicates_qa_from_filtered = duplicates_qa.merge(duplicates_filtered[['idx_1', 'idx_2']], on = ['idx_1', 'idx_2'], how = 'inner')
        duplicates_qa_from_filtered = duplicates_qa_from_filtered[duplicates_qa_from_filtered['similarity'] >= threshold_duplicates]
        statistics_by_answers, contradictions_qa = contradiction_table_qa(answer_oos, 
                                                                          answer_oot, 
                                                                          duplicates_qa_from_filtered)
        contradictions_qa_rate = 0
        if len(contradictions_qa) != 0:
            contradictions_qa_rate = sum(contradictions_qa.values())/len(contradictions_qa)
        rates['Доля семантических дубликатов ответов LLM на похожие вопросы в OOT относительно OOS'] = [duplicates_qa_rate]
        rates['Доля противоречий меток асессоров в вопросах (допустимы разные ответы)'] = [contradictions_rate]
        rates['Доля противоречий меток ассесоров в одних и тех же ответах на те же вопросы'] = [contradictions_qa_rate]
        precomputed['Таблица противоречий асессоров по критериям для question + answer'] = statistics_by_answers
        precomputed['Словарь противоречий question + answer'] = contradictions_qa

    report = report_share_of_contradictions(rates=rates,
                                            threshold=threshold,
                                            is_info=is_info)
    return {'report': report, 
            'precomputed': precomputed}


def duplicate_table(documents_1: tp.List[str],
                    documents_2: tp.List[str],
                    embedding_model: tp.Optional[tp.Any]=None, 
                    name_loop: str='question') -> pd.DataFrame:
    '''
    Функция создания таблицы семантической схожести (косинусная близость векторов эмбеддингов) между текстами. 
    Если embedding_model=None, вычисляется SimHash.
    Таблица схожести имеет 3 колонки:
        idx_1 - индекс предложения из documents_1;
        idx_2 - индекс предложения из documents_2;
        similarity - семантическое сходство между соответствующими текстами.
    Parameters:
        documents_1: первый список текстов
        documents_2: второй список текстов
        embedding_model: Класс-обертка над эмбеддинг моделью. Модель обязана иметь метод encode.
               Если в качестве модели указано None, вызывается функция SimHash.
        name_loop: Название колонки, по которой ищутся дубликаты и происходит эмбеддинг кодирование.
                   Данное название будет отображаться в соответствующих прогресс-барах - для кодирования 
                   (если используется модель) и поиска дубликатов.
    Return:
        Таблицу расстояний между всеми парами текстов.
    '''
    hist = []
    end_documents_1 = len(documents_1)
    start_documents_2 = 0
    end_documents_2 = len(documents_2)

    if embedding_model is not None:
        embeddings_1 = []
        for doc in tqdm(documents_1, desc='Progress encode ' + name_loop + ' in documents: '):
            embeddings_1.append(embedding_model.encode(doc))

    if documents_1 == documents_2:
        if embedding_model is not None:
            embeddings_2 = embeddings_1
    else:
        if embedding_model is not None:
            embeddings_2 = []
            for doc in tqdm(documents_2, desc='Progress encode ' + name_loop + ' in documents_2: '):
                embeddings_2.append(embedding_model.encode(doc))

    with tqdm(total=end_documents_1, desc='Calculate cos similarity ' + name_loop + ': ', file=sys.stdout, mininterval=0.05) as pbar:
        for i in range(end_documents_1):
            for j in range(end_documents_2):
                if embedding_model is not None:
                    similarity = util.cos_sim(embeddings_1[i],
                                              embeddings_2[j])[0][0].item()
                else:
                    similarity = calculate_similarity(documents_1[i], 
                                                      documents_2[j])
                hist.append([i, j, similarity])
            pbar.update(1)
            sys.stdout.flush()

    semantic_doubles = pd.DataFrame(hist, columns = ['idx_1', 'idx_2', 'similarity']).sort_values(by='similarity', ascending=False)
    return semantic_doubles.reset_index(drop=True)


def find_most_common_questions(duplicate_table: pd.DataFrame) ->  dict:
    '''
    Возвращает к каждому вопросу из oot наиболее схожий вопрос из oos, используя предпосчитанную таблицу близости.
    Parameters:
        duplicate_table: предпосчитанная таблица близости.
    Return:
        Словарь, где ключ - id вопроса из oos, а значение - id вопроса из oot. 
    '''
    uniq_keys = duplicate_table['idx_2'].unique()
    most_common_questions = {}
    for key in uniq_keys:
        best_duplicate = duplicate_table[duplicate_table['idx_2'] == key]['idx_1'].iloc[0]
        most_common_questions[key] = best_duplicate
    return most_common_questions
    
    
def calculate_similarity(text1: str, 
                         text2: str) -> float:
    '''
    Вычисляет схожесть между двумя текстами на основе расстояния Хэмминга их SimHash значений.
    Parameters:
        text1: Первый текст.
        text2: Второй текст.
    Return:
        Схожесть между текстами от 0 до 1, где 1 - идентичные тексты.
    '''
    hash1 = Simhash(text1)
    hash2 = Simhash(text2)
    hamming_distance = hash1.distance(hash2)
    similarity = 1 - (hamming_distance / 64)
    return similarity


def contradiction_table_qa(answer_1: pd.DataFrame,
                           answer_2: pd.DataFrame, 
                           similar_qa: pd.DataFrame) -> (pd.DataFrame, dict):
    '''
    Возвращает таблицу противоречий для найденной таблицы дубликатов (для LongAnswer формата). 
    Для каждого из критериев по которому асессоры оценивали ответ LLM выдается доля расхождений. 
    Также добавляется усредненная доля между всеми критериями.
    Parameters:
        answer_1: ответы асессоров из oos.
        answer_2: ответы асессоров из oot.
        most_common_answers: Пары id-ников похожих question + answer.
    Return: 
        Таблицу с статистикой расхождений по всем критериям и dict, где ключ - номер вопроса, значение - 1 или 0,отсутствие или наличие расхождений
    '''
    oot_idx = set(similar_qa['idx_2'])
    oot_contr = {item: 0 for item in oot_idx}
    columns = list(answer_1.columns) 
    statistics_by_answer = pd.DataFrame(0,index = list(oot_idx), columns = columns)
    for key1, key2 in zip(similar_qa['idx_1'], similar_qa['idx_2']):
        for count, ans in enumerate(columns):
            if answer_2.loc[key2, ans] != answer_1.loc[key1, ans]:
                oot_contr[key2] = 1
                statistics_by_answer.loc[key2, ans] += 1
    return statistics_by_answer, oot_contr


def contradiction_table_most_common(answer_1: pd.DataFrame,
                                    answer_2: pd.DataFrame, 
                                    most_common_answers: dict) -> pd.DataFrame:
    '''
    Возвращает таблицу противоречий для найденной таблицы дубликатов (для LongAnswer формата). 
    Для каждого из критериев, по которому асессоры оценивали ответ LLM, выдается доля расхождений.
    Также добавляется усредненная доля между всеми критериями.
    Parameters:
        answer_1: ответы асессоров из oos.
        answer_2: ответы асессоров из oot.
        most_common_answers: Пары id-ников наиболее схожих вопросов из oos для oot.
    Return: 
        Таблицу с статистикой расхождений по всем критериям
    '''
    columns = list(answer_1.columns) 
    result = pd.DataFrame([[0] * len(columns)],columns = columns)
    for key, value in most_common_answers.items():
        result_row = np.zeros(len(answer_1.columns))
        for count, ans in enumerate(answer_1.columns):
            if answer_2.loc[value, ans] != answer_1.loc[key, ans]:
                result_row[count] += 1
        result.iloc[0] += result_row
    result['Средняя ошибка']  = result.mean(axis = 1)
    return result