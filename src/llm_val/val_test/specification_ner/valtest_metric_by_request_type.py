import typing as tp
import pandas as pd
from copy import deepcopy

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.ner_metrics import ner_metrics_exact_match_macro_f1


def report_valtest_metric_by_request_type(metrics_by_request_type: tp.Dict[str, float],
                                          main_metric: str,
                                          data_type: str='test') -> tp.Dict[str, tp.Any]:
    '''
    Создание отчета о сравнении метрик качества для разных способов извлечения сущностей в NER задаче
    Parameters:
        metrics_by_request_type: Словарь с метриками качества для разных способов извлечения сущностей
        main_metric: Название основной метрики качества
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    Return:
        report в соответствии с концепцией llm_val
    '''
    color = 'gray'
    result_df = pd.DataFrame({
        'Тип запроса': ['Все классы', 'По одному классу'],
        f'Значение метрики {main_metric} на {data_type}': [metrics_by_request_type['extracted_all_types_metric'], 
                                                           metrics_by_request_type['extracted_by_type_merged_metric']]})

    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [result_df]
    return res_dict


def valtest_metric_by_request_type(sampler: Sampler,
                                   main_metric: str='macro_f1',
                                   data_type: str='test',
                                   metric_function: tp.Callable=ner_metrics_exact_match_macro_f1,
                                   test_size: float=1.0,
                                   metrics_by_request_type: tp.Optional[tp.Dict[str, float]]=None,
                                   **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест сравнения метрик качества для разных способов извлечения сущностей в NER задаче (по одному классу и все за один запрос)
    Parameters:
        sampler: Объект класса Sampler для получения данных
        main_metric: Название основной метрики качества для сравнения
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        metric_function: Функция для подсчета метрик качества
        test_size: Доля данных от общего объема тестовой выборки, которая будет использоваться для расчета метрик 
        metrics_by_request_type: Опциональный словарь с предподсчитанными метриками качества
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if metrics_by_request_type is None:
        data_test = getattr(sampler, data_type)['X']
        if test_size < 1.0:
            data_test = data_test.sample(frac=test_size, random_state=42)
        entity_types = sampler.entity_types
        metrics_by_request_type = calculate_metrics_by_request_type(data_test, 
                                                                    entity_types, 
                                                                    main_metric, 
                                                                    metric_function)
    precomputed = {'metrics_by_request_type': deepcopy(metrics_by_request_type)}
    report = report_valtest_metric_by_request_type(metrics_by_request_type, 
                                                   main_metric, 
                                                   data_type)
    return {'report': report,
            'precomputed': precomputed}


def calculate_metrics_by_request_type(df: pd.DataFrame,
                                      entity_types: tp.List[str], 
                                      main_metric: str,
                                      metric_function: tp.Callable=ner_metrics_exact_match_macro_f1,
                                      **kwargs) -> tp.Dict[str, float]:
    '''
    Вычисляет метрики качества для извлечения сущностей по всем типам и по одному типу за раз
    Parameters:
        df: Датафрейм с предсказаниями модели
        entity_types: Список типов сущностей
        main_metric: Название основной метрики
        metric_function: Функция для подсчета метрик
    Return:
        Словарь с метриками для разных способов извлечения:
            - extracted_all_types_metric: метрика для извлечения всех типов сразу
            - extracted_by_type_merged_metric: метрика для извлечения по одному типу за раз (объединение результатов)
    '''
    extracted_all_types_metric = metric_function(df, 
                                                 entity_types=entity_types, 
                                                 extracted_col_name='extracted_all_types')
    extracted_by_type_merged_metric = metric_function(df,
                                                      entity_types=entity_types, 
                                                      extracted_col_name='extracted_by_type_merged')

    return {'extracted_all_types_metric': extracted_all_types_metric[main_metric],
            'extracted_by_type_merged_metric': extracted_by_type_merged_metric[main_metric]}
