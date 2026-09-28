import typing as tp
import pandas as pd
from copy import deepcopy

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold
from llm_val.utils.ner_metrics import ner_metrics_exact_match_macro_f1


def report_valtest_entity_description_quality_analysis(metrics_by_description_variantion: tp.Dict[str, tp.Dict[str, float]],
                                                       main_metric: str,
                                                       data_type: str='test',
                                                       absolute_improvement_threshold: tp.Tuple[float, float]=(0.01, 0.03), 
                                                       is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание отчета о сравнении метрик качества для разных способов описания сущностей в NER задаче
    Parameters:
        metrics_by_description_variantion: Словарь с метриками качества для разных способов описания сущностей
        main_metric: Название основной метрики качества для сравнения
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        absolute_improvement_threshold: Пороговые значения для абсолютного прироста метрики   
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    variations = ['extracted',
                  'extracted_description_detalization',
                  'extracted_examples_addition', 
                  'extracted_description_detalization_and_examples_addition']
    variation_metrics = []
    base_metric = metrics_by_description_variantion['extracted_metric'][main_metric]
    for variation in variations:
        variation_metric = metrics_by_description_variantion[f'{variation}_metric'][main_metric]
        variation_metrics.append(variation_metric)
    improvements = [variation_metric - base_metric for variation_metric in variation_metrics]
    final_color = 'gray' if is_info else semaphore_by_threshold(max(improvements), 
                                                                absolute_improvement_threshold, 
                                                                greater_is_better=False)
    colors = [semaphore_by_threshold(improvement, absolute_improvement_threshold, greater_is_better=False) for improvement in improvements]
    result_df = pd.DataFrame({
        'Вариант описания': ['Исходное описание', 
                             'Детализированное описание',
                             'Добавление примеров',
                             'Детализированное описание с примерами'],
        f'Значение метрики {main_metric} на {data_type}': variation_metrics,
        'Абсолютный прирост': improvements,
        'Результат теста': colors})
    res_dict = {}
    res_dict['semaphore'] = final_color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [result_df]
    return res_dict
    

def valtest_entity_description_quality_analysis(sampler: Sampler,
                                                main_metric: str='macro_f1',
                                                data_type: str='test',
                                                absolute_improvement_threshold: tp.Tuple[float, float]=(0.01, 0.03),
                                                metric_function: tp.Callable=ner_metrics_exact_match_macro_f1,
                                                is_info: bool=False,
                                                metrics_by_description_variantion: tp.Optional[tp.Dict[str, float]]=None,
                                                **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест проверки качества описания сущностей для NER задачи
    Parameters:
        sampler: Объект класса Sampler для получения данных
        main_metric: Название основной метрики качества
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        absolute_improvement_threshold: Пороговые значения для абсолютного прироста метрики
        metric_function: Функция для подсчета метрик
        is_info: True, если тест информативный
        metrics_by_description_variantion: Опциональный словарь с предподсчитанными метриками для разных способов описания сущностей
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if metrics_by_description_variantion is None:
        data_test = getattr(sampler, data_type)
        entity_types = sampler.entity_types
        metrics_by_description_variantion = calculate_metric_by_description_variantion(data_test['X'],
                                                                                       entity_types, 
                                                                                       metric_function)

    precomputed = {'metrics_by_description_variantion': deepcopy(metrics_by_description_variantion)}
    report = report_valtest_entity_description_quality_analysis(metrics_by_description_variantion,
                                                                main_metric, 
                                                                data_type, 
                                                                absolute_improvement_threshold, 
                                                                is_info)
    return {'report': report,
            'precomputed': precomputed}


def calculate_metric_by_description_variantion(df: pd.DataFrame,
                                               entity_types: tp.List[str], 
                                               metric_function: tp.Callable=ner_metrics_exact_match_macro_f1,
                                               **kwargs) -> tp.Dict[str, tp.Dict[str, float]]:
    '''
    Вычисляет метрики качества для разных вариантов описания сущностей в NER задаче
    Parameters:
        df: Датафрейм с предсказаниями модели и исходными текстами
        entity_types: Список типов сущностей для извлечения
        metric_function: Функция для расчета метрик качества извлечения сущностей
    Return:
        Словарь с метриками качества для разных вариантов описания сущностей:
            - extracted_metric: базовое описание
            - extracted_description_detalization_metric: детализированное описание
            - extracted_examples_addition_metric: описание с примерами
            - extracted_description_detalization_and_examples_addition_metric: детализированное описание с примерами
    '''
    results = {}
    variations = ['extracted',
                  'extracted_description_detalization', 
                  'extracted_examples_addition', 
                  'extracted_description_detalization_and_examples_addition']
    for variation_name in variations:
        results[f'{variation_name}_metric'] = metric_function(df, 
                                                              entity_types=entity_types, 
                                                              extracted_col_name=variation_name)
    return results  
