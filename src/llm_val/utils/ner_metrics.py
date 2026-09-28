import typing as tp
import pandas as pd
from collections import defaultdict, Counter


def split_entities_by_words(dct: tp.Dict[str, tp.List[str]]) -> tp.Dict[str, tp.List[str]]:
    '''
    Разбивает сущности на отдельные слова
    Parameters:
        dct: Словарь, где ключ - название сущности, значение - список строк с сущностями
    Return:
        Словарь с тем же форматом, где строки разбиты на отдельные слова
    '''
    word_lists = {}
    for key in dct:
        word_lists[key] = [word for item in dct[key] for word in item.split()]
    return word_lists


def calculate_f1(tp: int, fp: int, fn: int) -> tp.Tuple[float, float, float]:
    '''
    Вычисляет метрики precision, recall и f1-score на основе количества истинно положительных, 
    ложно положительных и ложно отрицательных предсказаний
    Parameters:
        tp: Количество истинно положительных предсказаний
        fp: Количество ложно положительных предсказаний  
        fn: Количество ложно отрицательных предсказаний
    Return:
        Кортеж из трех значений: precision, recall, f1-score
    '''
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def calculate_metrics(y: pd.DataFrame,
                      entity_types: tp.List[str],
                      extracted_col_name: str='extracted',
                      target_col_name: str='target',
                      metric_type: tp.Literal['micro', 'macro']='macro',
                      exact_match: bool=True,
                      ignore_repetitions: bool=False) -> tp.Dict[str, float]:
    '''
    Вычисляет метрики качества для задачи извлечения именованных сущностей (NER)
    Parameters:
        y: DataFrame, содержащий колонки с извлеченными и целевыми сущностями
        entity_types: Список типов сущностей, для которых нужно посчитать метрики
        extracted_col_name: Название колонки с извлеченными сущностями в формате dict[str, list[str]]
        target_col_name: Название колонки с целевыми сущностями в формате dict[str, list[str]]
        metric_type: Способ агрегации метрик
        exact_match: Флаг для выбора способа сравнения сущностей - целиком или по словам
        ignore_repetitions: Флаг для игнорирования повторяющихся сущностей при подсчете
    Return:
        Словарь с F1-метриками для каждого типа сущностей и общей агрегированной метрикой
    '''
    extracted_entities = y[extracted_col_name]
    target_entities = y[target_col_name]
    overall_metrics = {label: {'tp': 0, 'fp': 0, 'fn': 0} for label in entity_types}

    for extracted, target in zip(extracted_entities, target_entities):
        if not exact_match:
            target = split_entities_by_words(target)
            extracted = split_entities_by_words(extracted)

        if len(target.keys()) != len(extracted.keys()) and not isinstance(target, defaultdict):
            target = defaultdict(list, target)
            extracted = defaultdict(list, extracted)

        for label in entity_types:
            if ignore_repetitions:
                pred_set = set(extracted[label])
                target_set = set(target[label])

                tp = len(pred_set.intersection(target_set))
                fp = len(pred_set.difference(target_set))
                fn = len(target_set.difference(pred_set))
            else:
                pred_counter = Counter(extracted[label])
                target_counter = Counter(target[label])

                tp = sum((pred_counter & target_counter).values())
                fp = sum((pred_counter - target_counter).values())
                fn = sum((target_counter - pred_counter).values())
            
            overall_metrics[label]['tp'] += tp
            overall_metrics[label]['fp'] += fp
            overall_metrics[label]['fn'] += fn

    results = {}
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    f1_scores = []
    for label in entity_types:
        overall_tp += overall_metrics[label]['tp']
        overall_fp += overall_metrics[label]['fp']
        overall_fn += overall_metrics[label]['fn']

        _, _, f1 = calculate_f1(**overall_metrics[label])
        results[f'{label}_f1'] = f1
        f1_scores.append(f1)
        
    if metric_type == 'micro':
        _, _, micro_f1 = calculate_f1(overall_tp, overall_fp, overall_fn)
        results['micro_f1'] = micro_f1
    elif metric_type == 'macro':
        macro_f1 = sum(f1_scores) / len(entity_types)
        results['macro_f1'] = macro_f1
    
    return results


def ner_metrics_exact_match_macro_f1(y: pd.DataFrame, 
                                     entity_types: tp.List[str], **kwargs) -> tp.Dict[str, float]:
    '''
    Вычисляет macro-F1 метрику для полного сопоставления сущностей NER
    Parameters:
        y: DataFrame с колонками extracted и target, содержащими предсказанные и истинные сущности
        entity_types: Список типов сущностей для оценки
    Returns:
        Словарь с macro-F1 метриками для каждого типа сущностей
    '''
    return calculate_metrics(y,
                             exact_match=True,
                             metric_type='macro',
                             entity_types=entity_types,
                             **kwargs)


def ner_metrics_partial_match_macro_f1(y: pd.DataFrame, 
                                       entity_types: tp.List[str], **kwargs) -> tp.Dict[str, float]:
    '''
    Вычисляет macro-F1 метрику для частичного сопоставления сущностей NER
    Parameters:
        y: DataFrame с колонками extracted и target, содержащими предсказанные и истинные сущности
        entity_types: Список типов сущностей для оценки
    Returns:
        Словарь с macro-F1 метриками для каждого типа сущностей
    '''
    return calculate_metrics(y,
                             exact_match=False,
                             metric_type='macro',
                             entity_types=entity_types,
                             **kwargs)


def ner_metrics_exact_match_micro_f1(y: pd.DataFrame, 
                                     entity_types: tp.List[str], **kwargs) -> tp.Dict[str, float]:
    '''
    Вычисляет micro-F1 метрику для полного сопоставления сущностей NER
    Parameters:
        y: DataFrame с колонками extracted и target, содержащими предсказанные и истинные сущности
        entity_types: Список типов сущностей для оценки
    Returns:
        Словарь с micro-F1 метриками для каждого типа сущностей
    '''
    return calculate_metrics(y, 
                             exact_match=True, 
                             metric_type='micro', 
                             entity_types=entity_types, 
                             **kwargs)


def ner_metrics_partial_match_micro_f1(y: pd.DataFrame, 
                                       entity_types: tp.List[str], **kwargs) -> tp.Dict[str, float]:
    '''
    Вычисляет micro-F1 метрику для частичного сопоставления сущностей NER
    Parameters:
        y: DataFrame с колонками extracted и target, содержащими предсказанные и истинные сущности
        entity_types: Список типов сущностей для оценки
    Returns:
        Словарь с micro-F1 метриками для каждого типа сущностей
    '''
    return calculate_metrics(y, 
                             exact_match=False, 
                             metric_type='micro', 
                             entity_types=entity_types, 
                             **kwargs)


NER_METRICS = {'ner_exact_match_macro_f1': {'call': ner_metrics_exact_match_macro_f1, 'is_singlecol': False},
               'ner_exact_match_micro_f1': {'call': ner_metrics_exact_match_micro_f1, 'is_singlecol': False},
               'ner_partial_match_macro_f1': {'call': ner_metrics_partial_match_macro_f1, 'is_singlecol': False},
               'ner_partial_match_micro_f1': {'call': ner_metrics_partial_match_micro_f1, 'is_singlecol': False}}