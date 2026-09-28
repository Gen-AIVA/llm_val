import os
import re
import nltk
import random
import pymorphy2
import numpy as np
import typing as tp
import pandas as pd
from tqdm.auto import tqdm
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold


def report_valtest_train_test_intersection(similar_texts: pd.DataFrame,
                                           intersection_test_share: float,
                                           semaphore_threshold: tp.Tuple[float, float]=(0.0, 0.05),
                                           is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        similar_texts: Датафрейм со схожими текстами
        intersection_test_share: Доля тестовых текстов, лежащих в обучающей выборке
        semaphore_threshold: Пороги для желтого и красного светофоров для доли пересечений в тестовой выборке
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_dataframe = pd.DataFrame({'Тип сходства': ['Полная подстрока исходного текста', 
                                                   'Полная подстрока предобработанного текста', 
                                                   'Превышен лимит общих слов для исходных текстов',
                                                   'Превышен лимит общих слов для предобработанных текстов'],
              'Количество': [0, 0, 0, 0]}).set_index('Тип сходства')
    sim_counts = similar_texts['similarity_type'].value_counts().to_dict()
    for key in res_dataframe.index:
        res_dataframe.loc[key, 'Количество'] += sim_counts.get(key, 0) 
    
    res_dataframe.loc['Суммарно пересечений'] = res_dataframe['Количество'].sum()
    res_dataframe.loc['Доля test-текстов в train'] = intersection_test_share
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else semaphore_by_threshold(intersection_test_share,
                                                                          threshold=semaphore_threshold,
                                                                          greater_is_better=False,
                                                                          left_border_is_bigger=True,
                                                                          right_border_is_bigger=True)
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [res_dataframe]
    return res_dict


def valtest_train_test_intersection(sampler: Sampler,
                                    path_to_train_files: tp.Union[str, tp.List[str]],
                                    train_reader_fn: tp.Callable[tp.Union[str, tp.List[str]], tp.Tuple[tp.List[str], tp.List[str]]],
                                    data_type: str='test',
                                    semaphore_threshold: tp.Tuple[float, float]=(0.0, 0.05),
                                    use_answers: bool=False,
                                    base_similarity_threshold: float=0.6,
                                    words_in_common_rate_threshold: float=0.5,
                                    preprocessed_words_in_common_rate_threshold: float=0.8,
                                    use_adaptive_thresholding: bool=True,
                                    random_substring_min_length: int=300, 
                                    random_substring_max_length: int=450,
                                    adaptive_threshold_step: float=0.1,
                                    adaptive_thresholding_diff_cutoff: float=0.014,
                                    random_state: int=17,
                                    verbose: bool=False,
                                    similarities: tp.Optional[pd.DataFrame]=None,
                                    similar_texts: tp.Optional[pd.DataFrame]=None,
                                    is_info: bool=False,
                                    **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на пересечение test и train выборок.
    Все параметры, включающие в себя длину чего-либо, имеют длину в словах
    из расчета 1000 символов ~= 150 слов.
    Parameters:
        sampler: Объект класса Sampler
        path_to_train_files: Путь/список путей к файлу/файлам с train-текстами
        train_reader_fn: Функция для чтения текстов из файлов;
                         должна принимать на вход path_to_train_files, 
                         на выход отдавать tuple со списками текстов и уникальных индексов для этих текстов
        data_type: Тип выборки, которая проверяется на пересечение
        semaphore_threshold: Пороги для желтого и красного светофоров для доли пересечений в тестовой выборке
        use_answers: Проверять ли ответы на нахождение в train-выборке?
        base_similarity_threshold: Базовый порог косинусного сходства для признания текстов схожими 
        words_in_common_rate_threshold: Порог доли общих слов для сырых текстов,
                                        после которого тексты считаются схожими
        preprocessed_words_in_common_rate_threshold: Порог доли общих слов для обработанных текстов,
                                                     после которого тексты считаются схожими
        use_adaptive_thresholding: Использовать ли адаптивный подбор порога сходства
        random_substring_min_length: Минимальная длина случайной подстроки при адаптивном подборе порога сходства
        random_substring_max_length: Максимальная длина случайной подстроки при адаптивном подборе порога сходства
        adaptive_threshold_step: Шаг сетки при адаптивном подборе порога сходства
        adaptive_thresholding_diff_cutoff: Разность метрики между соседними уровнями порога 
                                           при адаптивном подборе порога сходства
        random_state: random_state при генерации длин случайных подстрок
        verbose: Показывать ли прогресс прохождения теста?
        similarities: Посчитанные сходства между test и train текстами;
                      для корректной работы необходимо, чтобы они имели shape = (n_test_texts, n_train_texts),
                      а также индексы test/train текстов в наименовании строк и колонок соответственно;
                      если similarities не None, подсчет сходств производиться не будет
        similar_texts: Подобранные пары текстов test-train с выставленным типом сходства;
                       для корректной работы необходимо, чтобы в датафрейме были как минимум индексы для обоих выборок,
                       а также тип сходства similarity_type;
                       если не None, то подбор порога, расчеты сходств и вычисление типов не будет производиться
        is_info: True, если тест информативный
    '''
    data = getattr(sampler, data_type)['X']
    test_texts = data['question'].to_list()
    test_indices = pd.Series(data.index).apply(lambda x: f'Q_{x}').to_list()
    if use_answers:
        test_texts += data['true_answer'].apply(str).to_list()
        test_indices += pd.Series(data.index).apply(lambda x: f'A_{x}').to_list()
      
    train_texts, train_indices = train_reader_fn(path_to_train_files)
    preprocessed_train_texts = [preprocess_ru(txt) for txt in train_texts]
    preprocessed_test_texts = [preprocess_ru(txt) for txt in test_texts]
    
    train_data = pd.DataFrame({'index': train_indices,
                               'raw_text': train_texts,
                               'preprocessed_text': preprocessed_train_texts})
    
    test_data = pd.DataFrame({'index': test_indices,
                              'raw_text': test_texts,
                              'preprocessed_text': preprocessed_test_texts})
    if not similar_texts:
        if use_adaptive_thresholding:
            similarity_threshold = calc_adaptive_threshold(train_data,
                                                           random_substring_min_length=random_substring_min_length,
                                                           random_substring_max_length=random_substring_max_length,
                                                           base_similarity_threshold=base_similarity_threshold,
                                                           adaptive_threshold_step=adaptive_threshold_step,
                                                           adaptive_thresholding_diff_cutoff=adaptive_thresholding_diff_cutoff,
                                                           random_state=random_state,
                                                           verbose=verbose)
        else:
            similarity_threshold = base_similarity_threshold

        if not similarities:
            similarities = pd.DataFrame(calc_similarities(train_data['preprocessed_text'].to_list(),
                                                          test_data['preprocessed_text'].to_list(),
                                                          verbose=verbose))
            similarities.columns = train_data['index']
            similarities.index = test_data['index']

        similar_texts = aggregate_similar_texts(train_data,
                                                test_data,
                                                similarities,
                                                similarity_threshold=similarity_threshold,
                                                words_in_common_rate_threshold=words_in_common_rate_threshold,
                                                preprocessed_words_in_common_rate_threshold=preprocessed_words_in_common_rate_threshold,
                                                verbose=verbose)
    
    intersection_test_share = similar_texts['test_index'].apply(lambda x: x[2:]).unique().shape[0] \
                              / test_data['index'].apply(lambda x: x[2:]).unique().shape[0]
    
    report = report_valtest_train_test_intersection(similar_texts,
                                                    intersection_test_share,
                                                    semaphore_threshold=semaphore_threshold,
                                                    is_info=is_info)
    precomputed = {'similarities': similarities,
                   'similar_texts': similar_texts}
    return {'report': report,
            'precomputed': precomputed}

def calc_adaptive_threshold(train_data: pd.DataFrame,
                            random_substring_min_length: int=300, 
                            random_substring_max_length: int=450,
                            base_similarity_threshold: float=0.6,
                            adaptive_threshold_step: float=0.1,
                            adaptive_thresholding_diff_cutoff: float=0.02,
                            random_state: int=17,
                            verbose: bool=False) -> float:
    '''
    Подсчет адаптивного порога для косинусного сходства.
    Подсчет производится спуском по сетке шага adaptive_threshold_step, начиная с base_similarity_threshold;
    если разность метрики между двумя соседними порогами меньше adaptive_thresholding_diff_cutoff,
    то поиск прекращается и выбирается больший порог из соседних.
    В качестве метрики используется точность подбора совпадений между train выборкой и случайными ее подстроками.
    Parameters:
        train_data: Датафрейм с train индексами и текстами,
        random_substring_min_length: Минимальная длина случайной подстроки в словах (150 слов ~= 1000 символов)
        random_substring_max_length: Максимальная длина случайной подстроки в словах (150 слов ~= 1000 символов)
        base_similarity_threshold: Базовый (максимальный) уровень сходства, с которого будет начинаться поиск
        adaptive_threshold_step: Шаг сетки при адаптивном подборе порога сходства
        adaptive_thresholding_diff_cutoff: Разность метрики между соседними уровнями порога 
                                           при адаптивном подборе порога сходства
        random_state: random_state при генерации длин случайных подстрок
        verbose: Если True, будет выводиться прогресс выполнения
    Return:
        Посчитанный порог для сходства
    '''
    sim_thresholds = np.flip(np.arange(0, base_similarity_threshold, adaptive_threshold_step)) + adaptive_threshold_step
    if verbose: 
        print('Selecting adaptive threshold...')
        iterator_for_metrics = tqdm(sim_thresholds)
        iterator_for_threshold = tqdm(range(1, len(sim_thresholds)))
    else:
        iterator_for_metrics = sim_thresholds
        iterator_for_threshold = range(1, len(sim_thresholds))
        
    random_substrings = train_data['raw_text'].apply(get_random_substring_from_text, 
                                                     min_length=random_substring_min_length, 
                                                     max_length=random_substring_max_length, 
                                                     random_state=random_state).to_list()
    
    similarities = calc_similarities(train_data['preprocessed_text'].to_list(),
                                     random_substrings)
    most_similar = np.argmax(similarities, axis=1) 
    metrics = [sum([(train_data['index'].iloc[idx] == train_data['index'].iloc[most_similar[idx]]) / len(random_substrings) \
               for idx in range(len(similarities)) if similarities[idx][most_similar[idx]] >= thresh]) for thresh in iterator_for_metrics]
    
    for idx in iterator_for_threshold:
        if metrics[idx] - metrics[idx-1] < adaptive_thresholding_diff_cutoff:
            adaptive_threshold = sim_thresholds[idx-1]
            if verbose:
                print(f'Selected similarity threshold = {round(adaptive_threshold, 3)} with score {round(metrics[idx-1], 3)}')
            break
            
    return adaptive_threshold

def aggregate_similar_texts(train_data: pd.DataFrame,
                            test_data: pd.DataFrame,
                            similarities: pd.DataFrame,
                            similarity_threshold: float=0.6,
                            words_in_common_rate_threshold: float=0.5,
                            preprocessed_words_in_common_rate_threshold: float=0.8,
                            verbose: bool=False) -> pd.DataFrame:
    '''
    Агрегатор результатов на основе косинусных сходств
    Parameters:
        train_data: Датафрейм с train-текстами
        test_data: Датафрейм с test-текстами
        similarities: Посчитанные косинусные сходства
        similarity_threshold: Порог сходства, после которого тексты считаются схожими
        words_in_common_rate_threshold: Порог доли общих слов для сырых текстов,
                                        после которого тексты считаются схожими
        preprocessed_words_in_common_rate_threshold: Порог доли общих слов для обработанных текстов,
                                                     после которого тексты считаются схожими
        verbose: Показывать ли прогресс?
    Return:
        Таблица пар схожих текстов с индексами и типом сходства;
        В случае отсутствия похожих текстов функция возвращает пустой dataframe
    '''
    if verbose:
        print('Aggregating similar texts...')
    high_sim_pairs = pd.DataFrame(similarities.stack().loc[lambda x: x >= similarity_threshold].index.to_list(), columns=['test_index', 'train_index'])
    columns_to_save = ['test_index', 'train_index', 'test_text', 'train_text', 'similarity_type']
    if not high_sim_pairs.empty:
        columns_to_add = ['raw_text', 'preprocessed_text']
        new_names = ['text', 'prep_text']
        for col in zip(columns_to_add, new_names):
            high_sim_pairs = attach_column_to_pairs(high_sim_pairs, 
                                                    train_data,
                                                    column_to_attach=col[0],
                                                    new_column_name='train_' + col[1],
                                                    pairs_index_column='train_index',
                                                    data_index_column='index',
                                                    suffixes=('', '_train'))
            high_sim_pairs = attach_column_to_pairs(high_sim_pairs, 
                                                    test_data,
                                                    column_to_attach=col[0],
                                                    new_column_name='test_' + col[1],
                                                    pairs_index_column='test_index',
                                                    data_index_column='index',
                                                    suffixes=('', '_test'))
            
        high_sim_pairs['prep_subset'] = high_sim_pairs.apply(lambda x: is_substring(x['test_prep_text'], x['train_prep_text']), axis=1)
        high_sim_pairs['raw_subset'] = high_sim_pairs.apply(lambda x: is_substring(x['test_text'], x['train_text']), axis=1)
        high_sim_pairs['prep_words_in_common'] = high_sim_pairs.apply(lambda x: True if \
                                                                      count_words_in_common(x['test_prep_text'], x['train_prep_text']) / len(x['test_prep_text'].split()) \
                                                                      >= preprocessed_words_in_common_rate_threshold else False, axis=1)
        high_sim_pairs['raw_words_in_common'] = high_sim_pairs.apply(lambda x: True if \
                                                                      count_words_in_common(x['test_text'], x['train_text']) / len(x['test_text'].split()) \
                                                                      >= words_in_common_rate_threshold else False, axis=1)
        high_sim_pairs['similarity_type'] = high_sim_pairs.apply(lambda x: get_similarity_type(x['raw_subset'],
                                                                                               x['prep_subset'],
                                                                                               x['raw_words_in_common'],
                                                                                               x['prep_words_in_common']), axis=1)
        aggregated_results = high_sim_pairs.loc[high_sim_pairs['similarity_type'] != 'no_similarity'][columns_to_save]
    else:
        aggregated_results = pd.DataFrame({col: [] for col in columns_to_save})
    return aggregated_results

def calc_similarities(train_texts: list[str],
                      test_texts: list[str],
                      verbose: bool=False) -> tp.Iterable[float]:
    '''
    Подсчет cos_sim от tf-idf векторов
    Parameters:
        train_texts: Тексты с обучения
        test_texts: Тестовые тексты
        verbose: Показывать ли прогресс
    Return:
        Матрица косинусных сходств
    '''
    if verbose:
        print('Calculating similarities...')
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)
    similarities = cosine_similarity(test_vectors, train_vectors)
    return similarities

def attach_column_to_pairs(pairs: pd.DataFrame,
                           data: pd.DataFrame,
                           column_to_attach: str,
                           new_column_name: str,
                           pairs_index_column: str,
                           data_index_column: str,
                           suffixes: tp.Tuple[str, str]=('', '_')):
    '''
    Прикрепление колонки к парам и исправление названий во избежание коллизий
    Parameters:
        pairs: Датафрейм с парами
        data: Датафрейм, из которого будет браться колонка
        column_to_attach: Колонка, которая будет прикреплена
        new_column_name: Новое название для колонки
        pairs_index_column: Индекс для матчинга со стороны датафрейма пар
        data_index_column: Индекс для матчинга со стороны второй таблички
        suffixes: Суффиксы для pd.merge
    Return:
        Датафрейм пар с новой колонкой
    '''
    pairs = pd.merge(pairs, data[[data_index_column, column_to_attach]], left_on=pairs_index_column, right_on=data_index_column, how='left', suffixes=suffixes)
    pairs.rename({column_to_attach: new_column_name}, axis=1, inplace=True)
    return pairs

def preprocess_ru(text: str,
                  tokenizer: tp.Callable=word_tokenize) -> str:
    '''
    Препроцессинг текста на русском языке 
    (нормализация, удаление стоп-слов и всех символов, кроме буквенных)
    Parameters:
        text: Текст для обработки
        tokenizer: Токенизатор текста
    Return:
        Нормализованный текст
    '''
    stop_words_en = set(stopwords.words('english'))
    stop_words_ru = set(stopwords.words('russian'))
    stopwords_set = stop_words_en.union(stop_words_ru)
    morph = pymorphy2.MorphAnalyzer()
    tokens = tokenizer(text)
    tokens = [morph.parse(word)[0].normal_form for word in tokens if word.isalpha() and word.lower() not in stopwords_set]
    return ' '.join(tokens)

def get_random_substring_from_text(text: str, 
                                   min_length: int, 
                                   max_length: int,
                                   random_state: int=17) -> str:
    '''
    Получить случайную подстроку из единственного текста
    Parameters:
        text: Текст
        min_length: Минимальная длина подстроки
        max_length: Максимальная длина подстроки
        random_state: random_state при генерации случайной подстроки
    Returns: 
        Cлучайная подстрока
    '''
    random.seed(random_state)
    if len(text) < max_length:
        random_substring = text
    else:
        substring_length = random.randint(min_length, max_length)
        start_index = random.randint(0, len(text) - substring_length)
        end_index = start_index + substring_length
        random_substring = text[start_index:end_index]
    return random_substring


def count_words_in_common(str1: str,
                          str2: str,
                          skip_stopwords: bool=True) -> int:
    '''
    Подсчет количества общих слов в двух строках
    Parameters:
        str1: Первая строка
        str2: Вторая строка
        skip_stopwords: Пропускать ли стоп-слова?
    Return:
        Сумма вхождений общих слов
    '''
    stop_words_en = set(stopwords.words('english'))
    stop_words_ru = set(stopwords.words('russian'))
    stopwords_set = stop_words_en.union(stop_words_ru)
    words1 = str1.split()
    words2 = str2.split()
    common_word_count = 0
    for word in set(words1):
        if skip_stopwords and word in stopwords_set:
            continue
        common_word_count += min(words1.count(word), words2.count(word))
    return common_word_count

def normalize_string(s: str) -> str:
    '''
    Вспомогательная функция для очистки строки
    Parameters: 
        s: Строка для очистки
    Return:
        Строка состоящая только из строчных букв и пробелов
    '''
    s = re.sub(r'[^a-zA-Zа-яА-Я ]+', '', s)
    return s.lower()


def is_substring(possible_substring: str, 
                 source_string: str) -> bool:
    '''
    Проверка на содержание одной строки в другой
    Parameters: 
        possible_substring: Предполагаемая подстрока
        source_string: Строка, в которой будем искать вхождение
    Return:
        Bool - входит ли possible_substring в source_string
    '''
    return normalize_string(possible_substring) in normalize_string(source_string)

def get_similarity_type(raw_subset: bool,
                        prep_subset: bool,
                        raw_words_in_common: bool,
                        prep_words_in_common: bool) -> str:
    '''
    Агрегация типа сходства текстов относительно результатов по проверкам
    Parameters:
        raw_subset: Исходный test-текст является подстрокой исходного train-текста
        prep_subset: Обработанный test-текст является подстрокой обработанного train-текста
        raw_words_in_common: Количество общих слов в исходных test и train текстах выше порога
        prep_words_in_common: Количество общих слов в обработанных test и train текстах выше порога
    Return:
        Наиболее приоритетный тип сходства в соответсвии со следующей иерархией
        raw_subset > prep_subset > raw_words_in_common > prep_words_in_common
        no_similarity, если все флаги - False
    '''
    if raw_subset:
        similarity_type = 'Полная подстрока исходного текста'
    elif prep_subset:
        similarity_type = 'Полная подстрока предобработанного текста'
    elif raw_words_in_common:
        similarity_type = 'Превышен лимит общих слов для исходных текстов'
    elif prep_words_in_common:
        similarity_type = 'Превышен лимит общих слов для предобработанных текстов'
    else: 
        similarity_type = 'no_similarity'
    return similarity_type