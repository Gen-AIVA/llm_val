import re
import numpy as np
import typing as tp
import pandas as pd
from copy import deepcopy
from collections import defaultdict, Counter

from llm_val.sampler.Sampler import Sampler


def report_valtest_conflicting_predictions_analysis(conflicting_predictions_dict: tp.Dict[str, float],
                                                    data_type: str='test') -> tp.Dict[str, tp.Any]:
    '''
    Создание отчета о проверке наличия конфликтующих предсказаний для NER задачи
    Parameters:
        conflicting_predictions_dict: Словарь с результатами проверки наличия конфликтующих предсказаний, содержащий:
            - texts_with_conflicts_ratio: Доля текстов с конфликтующими предсказаниями среди всех текстов
            - avg_class_conflict_diff: Средняя разность между предсказанным количеством строки и количеством ее вхождений в исходный текст для конфликтующего предсказания типа "По классу и строке" в текстах с конфликтующими предсказаниями
            - avg_text_conflict_ratio: Средняя доля строк, признанных конфликтующим предсказанием типа "По тексту и строке", среди всех предсказанных строк в текстах с конфликтующими предсказаниями
            - avg_occurrences: Среднее количество вхождений строк, признанных конфликтующими, в соответствующий им входной текст
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    Return:
        report в соответствии с концепцией llm_val
    '''
    color = 'gray'
    result = pd.DataFrame({
        'Проверка': ['Доля текстов с конфликтующими предсказаниями среди всех текстов',
                     'Средняя разность между предсказанным количеством строки и количеством ее вхождений в исходный текст для конфликтующего предсказания типа "По классу и строке" в текстах с конфликтующими предсказаниями',
                     'Средняя доля строк, признанных конфликтующим предсказанием типа "По тексту и строке", среди всех предсказанных строк в текстах с конфликтующими предсказаниями',
                     'Среднее количество вхождений строк, признанных конфликтующими, в соответствующий им входной текст'],
        f'Значение на {data_type}': [conflicting_predictions_dict['texts_with_conflicts_ratio'],
                                     conflicting_predictions_dict['avg_class_conflict_diff'],
                                     conflicting_predictions_dict['avg_text_conflict_ratio'], 
                                     conflicting_predictions_dict['avg_occurrences']]})
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [result]
    return res_dict


def valtest_conflicting_predictions_analysis(sampler: Sampler,
                                             data_type: str='test',
                                             conflicting_predictions_dict: tp.Optional[tp.Dict[str, float]]=None,
                                             **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест проверки наличия конфликтующих предсказаний для NER задачи
    Parameters:
        sampler: Объект класса Sampler для получения данных
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        conflicting_predictions_dict: Словарь с предпосчитанными результатами проверки конфликтов        
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if conflicting_predictions_dict is None:
        data_test = getattr(sampler, data_type)
        conflicting_predictions = parse_conflicting_predictions(data_test['X'])
        conflicting_predictions_dict = aggregate_conflicting_predictions(conflicting_predictions, 
                                                                         data_test['X'])
    precomputed = {'conflicting_predictions_dict': deepcopy(conflicting_predictions_dict)}
    report = report_valtest_conflicting_predictions_analysis(conflicting_predictions_dict, 
                                                             data_type)
    return {'report': report,
            'precomputed': precomputed}


def find_all_occurrences(sub_string: str,
                         main_string: str) -> tp.List[str]:
    '''
    Поиск всех вхождений подстроки в основной строке
    Parameters:
        sub_string: Искомая подстрока
        main_string: Основная строка, в которой производится поиск
    Return:
        Список всех найденных вхождений подстроки
    '''
    occurrences = re.findall(re.escape(sub_string), main_string)
    return occurrences


def parse_conflicting_predictions(X: pd.DataFrame) -> tp.Dict[str, tp.Any]:
    '''
    Агрегация конфликтующих предсказаний для NER задачи. Проверяет два типа конфликтов:
    1. Конфликты по классу и строке: когда для некоторого класса строка входит в текст больше раз, чем была предсказана
    2. Конфликты по тексту и строке: когда суммарно по всем классам строка была предсказана больше раз, чем входит в текст
    Parameters:
        X: Датафрейм с колонками extracted (извлеченные сущности), text (исходный текст), id
    Return:
        Словарь с информацией о конфликтующих предсказаниях:
            - total: Общее количество конфликтов
            - errors_by_sample_id: Словарь с информацией о конфликтах для каждого текста
    '''
    conflicting_predictions = {
        'total': 0,
        'errors_by_sample_id': defaultdict(list)
    }
    
    extracted = X['extracted']
    texts = X['text']
    sample_ids = X['id']

    for extracted_dict, text, sample_id in zip(extracted, texts, sample_ids):
        extracted_entities = defaultdict(int)
        for entity_type in extracted_dict.keys():
            extracted_values = Counter(extracted_dict[entity_type])
            for entity_text, count in extracted_values.items():
                extracted_entities[entity_text] += count
                number_of_word_occurrences = len(find_all_occurrences(entity_text, text))
                if count < number_of_word_occurrences:
                    conflicting_predictions['total'] += 1
                    conflicting_predictions['errors_by_sample_id'][sample_id].append((entity_text, count, number_of_word_occurrences, entity_type))
            
        for entity_text, count in extracted_entities.items():
            number_of_word_occurrences = len(find_all_occurrences(entity_text, text))
            if count > max(number_of_word_occurrences, 1):
                conflicting_predictions['total'] += 1
                conflicting_predictions['errors_by_sample_id'][sample_id].append((entity_text, extracted_entities[entity_text], number_of_word_occurrences))
                
    return conflicting_predictions


def aggregate_conflicting_predictions(conflicting_predictions: tp.Dict[str, tp.Any], 
                                      X: pd.DataFrame) -> tp.Dict[str, tp.Any]:
    '''
    Агрегация конфликтующих предсказаний для NER задачи.
    Parameters:
        conflicting_predictions: Словарь с информацией о конфликтующих предсказаниях
        X: Датафрейм с данными для тестирования
    Return:
        Словарь с агрегированными метриками:
            - texts_with_conflicts_ratio: Доля текстов с конфликтующими 
              предсказаниями среди всех текстов
            - avg_class_conflict_diff: Средняя разность между предсказанным 
              количеством строки и количеством ее вхождений в исходный текст для 
              конфликтующего предсказания типа "По классу и строке" в текстах с 
              конфликтующими предсказаниями
            - avg_text_conflict_ratio: Средняя доля строк, признанных конфликтующим 
              предсказанием типа "По тексту и строке", среди всех предсказанных 
              строк в текстах с конфликтующими предсказаниями
            - avg_occurrences: Среднее количество вхождений строк, признанных 
              конфликтующими, в соответствующий им входной текст
    '''
    results = {
        'texts_with_conflicts_ratio': 0.0,
        'avg_class_conflict_diff': 0.0,
        'avg_text_conflict_ratio': 0.0,
        'avg_occurrences': 0.0
    }
    
    if not conflicting_predictions['errors_by_sample_id']:
        return results
        
    results['texts_with_conflicts_ratio'] = len(set(conflicting_predictions['errors_by_sample_id'].keys())) / len(X)
    
    class_conflicts_diffs = []
    text_conflicts_ratios = []
    total_occurrences = []
    
    for sample_id, errors in conflicting_predictions['errors_by_sample_id'].items():
        sample_text_conflicts = 0
        for error in errors:
            if len(error) == 4: 
                _, count, occurrences, _ = error
                class_conflicts_diffs.append(count - occurrences)
            else: 
                _, _, occurrences = error
                sample_text_conflicts += 1
            total_occurrences.append(occurrences)
            
        extracted_entities_length = len(X.loc[X['id'] == sample_id, 'flat_extracted_entities'].iloc[0])
        text_conflicts_ratios.append(sample_text_conflicts / extracted_entities_length)
        
    if class_conflicts_diffs:
        results['avg_class_conflict_diff'] = np.mean(class_conflicts_diffs)
    if text_conflicts_ratios:
        results['avg_text_conflict_ratio'] = np.mean(text_conflicts_ratios)
    if total_occurrences:
        results['avg_occurrences'] = np.mean(total_occurrences)
    return results