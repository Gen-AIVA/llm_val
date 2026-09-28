import typing as tp
import pandas as pd
from copy import deepcopy

from llm_val.sampler.Sampler import Sampler


def report_valtest_format_compliance(result_dict: tp.Dict[str, float],
                                     data_type: str='test') -> tp.Dict[str, tp.Any]:
    '''
    Создание отчета о проверке соответствия формата данных для NER задачи
    Parameters:
        result_dict: Словарь с результатами проверки формата данных, содержащий:
            - distorted_predicted_ratio: Доля искажений в извлеченных сущностях
            - distorted_target_ratio: Доля искажений относительно размеченных сущностях  
            - format_mismatch_ratio: Доля несоответствий формату
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    Return:
        report в соответствии с концепцией llm_val
    '''
    color = 'gray'
    result = pd.DataFrame({
        'Проверка': ['Доля искажений текста сущности (извлеченные)', 'Доля искажений текста сущности (размеченные)', 'Доля несоответствий предсказаний заданному формату'],
        f'Значение на {data_type}': [result_dict['distorted_predicted_ratio'], result_dict['distorted_target_ratio'], result_dict['format_mismatch_ratio']]})
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [result]
    return res_dict


def valtest_format_compliance(sampler: Sampler,
                              data_type: str='test',
                              format_compliance_dict: tp.Optional[tp.Dict[str, float]]=None,
                              **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест проверки соответствия формата генераций модели для NER задачи. Проверяет наличие искажений 
    в извлеченных сущностях и соответствие предсказаний заданному формату.
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        format_compliance_dict: Опциональный словарь с предпосчитанными результатами проверки формата        
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if format_compliance_dict is None:
        data_test = getattr(sampler, data_type)
        format_compliance_dict = check_format_compliance(data_test['X'])
    precomputed = {'format_compliance_dict': deepcopy(format_compliance_dict)}
    report = report_valtest_format_compliance(format_compliance_dict,
                                              data_type)
    return {'report': report,
            'precomputed': precomputed}


def check_format_compliance(df: pd.DataFrame) -> tp.Dict[str, float]:
    '''
    Проверяет искажение текста сущностей при извлечении и соответствие предсказаний заданному формату 
    Parameters:
        df: Датафрейм с колонками text, flat_extracted_entities, flat_target_entities, parser_error
    Return:
        Словарь с долями искаженных сущностей среди извлеченных и размеченных, а также долей несоответствий формату
    '''
    results = {
        'distorted_predicted_ratio': 0.0,  
        'distorted_target_ratio': 0.0,  
        'format_mismatch_ratio': 0.0,
    }
    
    total_predicted_distorted = 0
    total_predicted_entities = 0
    total_target_entities = 0
    total_format_mismatch = 0
    
    for _, row in df.iterrows():
        text = row['text']
        predicted_entities = row['flat_extracted_entities']
        target_entities = row['flat_target_entities']
        
        for entity in predicted_entities:
            if entity not in text:
                total_predicted_distorted += 1
        
        total_target_entities += len(target_entities)
        total_predicted_entities += len(predicted_entities)
        total_format_mismatch += int(row['parser_error'])

    if total_predicted_entities > 0:
        results['distorted_predicted_ratio'] = round(total_predicted_distorted / total_predicted_entities, 4)
    if total_target_entities > 0:
        results['distorted_target_ratio'] = round(total_predicted_distorted / total_target_entities, 4)
    results['format_mismatch_ratio'] = round(total_format_mismatch / len(df), 4) 
    return results
