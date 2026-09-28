import numpy as np
import pandas as pd
import typing as tp
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import defaultdict

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold, quanity_semaphore

        
def report_valtest_error_distribution_analysis(confusion_matrix: tp.Dict[str, tp.Dict[str, float]],
                                               figs: tp.List[plt.Figure],
                                               data_type: str='test',
                                               class_semaphore_threshold: tp.Tuple[float, float]=(0.25, 0.5),
                                               overall_semaphore_threshold: tp.Tuple[int, int]=(1, 2),
                                               is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание отчета о распределении ошибок модели для NER задачи
    Parameters:
        confusion_matrix: Матрица ошибок в формате словаря, где ключ - истинный класс,
                        значение - словарь {предсказанный класс: доля}
        figs: Список графиков для визуализации результатов
        data_type: Тип данных, на которых производится расчет. Hint: название атрибута в sampler'е
        class_semaphore_threshold: Границы светофора для оценки максимальной доли ошибок на уровне класса,
                                где первое значение - граница для зеленого светофора,
                                второе значение - граница для желтого светофора
        overall_semaphore_threshold: Границы для итоговой оценки,
                                где первое значение - минимальное количество красных для красного светофора,
                                второе значение - минимальное количество желтых для желтого светофора
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    semaphores = {}
    aggregated_errors = aggregate_mistaken_extractions(confusion_matrix)

    for true_class, predictions in aggregated_errors.items():
        max_error = predictions['max_error_rate']
        semaphores[true_class] = semaphore_by_threshold(max_error, 
                                                        class_semaphore_threshold, 
                                                        greater_is_better=False)
    overall_semaphore = 'gray' if is_info else quanity_semaphore(semaphores.values(), 
                                                                 overall_semaphore_threshold)

    result = pd.DataFrame({'Класс': list(aggregated_errors.keys()),
                           f'Максимальная доля ошибок на {data_type}': [predictions['max_error_rate'] for predictions in aggregated_errors.values()],
                           'Результат теста': [semaphores[true_class] for true_class in aggregated_errors.keys()]})
    
    report = {}
    report['semaphore'] = overall_semaphore
    report['result_plots'] = figs
    report['result_dataframes'] = [result]
    return report


def valtest_error_distribution_analysis(sampler: Sampler,
                                        data_type: str='test',
                                        class_semaphore_threshold: tp.Tuple[float, float]=(0.25, 0.5),
                                        overall_semaphore_threshold: tp.Tuple[int, int]=(1, 2),
                                        is_info: bool=False,
                                        confusion_matrix: tp.Optional[tp.Dict[str, float]]=None,
                                        **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест проверки распределения ошибок модели для NER задачи. Анализирует случаи, когда модель 
    неправильно определяет тип сущности, путем построения матрицы ошибок между разными типами.
    Parameters:
        sampler: Объект класса Sampler с данными для анализа
        data_type: Тип данных, на которых производится расчет. Hint: название атрибута в sampler'е
        class_semaphore_threshold: Границы светофора для оценки ошибок на уровне отдельных классов
        overall_semaphore_threshold: Границы светофора для итоговой оценки по всем классам
        is_info: True, если тест информативный
        confusion_matrix: Опциональный словарь с предпосчитанной матрицей ошибок между типами сущностей
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if confusion_matrix is None:
        data_test = getattr(sampler, data_type)
        errors = collect_mistaken_extractions(data_test['X'])
        confusion_matrix = build_confusion_matrix(errors)
    figs = [plot_confusion_matrix(confusion_matrix)]
    precomputed = {'confusion_matrix': deepcopy(confusion_matrix)}
    report = report_valtest_error_distribution_analysis(confusion_matrix, 
                                                        figs,
                                                        data_type,
                                                        class_semaphore_threshold=class_semaphore_threshold,
                                                        overall_semaphore_threshold=overall_semaphore_threshold,
                                                        is_info=is_info)
    return {'report': report,
            'precomputed': precomputed}


class MistakenExtractions(tp.TypedDict):
    mistaken_extractions: tp.Dict[str, tp.List[tp.Tuple[str, str, tp.Union[str, int]]]]
    number_of_entities: tp.Dict[str, int]


def collect_mistaken_extractions(X: pd.DataFrame) -> MistakenExtractions:
    '''
    Собирает ошибочно извлеченные сущности из предсказаний NER модели.
    Parameters:
        X: DataFrame с колонками:
            - extracted: извлеченные сущности
            - target: целевые сущности
            - id: идентификаторы сэмплов (опционально)  
    Return:
        Словарь, где ключ - правильный тип сущности, значение - список кортежей:
        (ошибочное_значение, предсказанный_тип, id_сэмпла)
    '''
    extracted_entities = X['extracted']
    target_entities = X['target'] 
    sample_ids = X['id'] if X['id'].isna().sum() == 0 else np.arange(len(X))
    errors = {'mistaken_extractions': defaultdict(list), 'number_of_entities': defaultdict(int)}
    
    for extracted_dict, target_dict, sample_id in zip(extracted_entities, target_entities, sample_ids):        
        for entity_type, values in extracted_dict.items():
            for value in values:
                if value not in target_dict[entity_type]:
                    real_target = next((k for k, v in target_dict.items() if value in v), None)
                    if real_target is not None:
                        errors['mistaken_extractions'][real_target].append((value, entity_type, sample_id))
        
        for entity_type, values in target_dict.items():
            errors['number_of_entities'][entity_type] += len(values)
    return errors


def build_confusion_matrix(errors: MistakenExtractions) -> tp.Dict[str, tp.Dict[str, float]]:
    '''
    Строит матрицу ошибок для неправильно распознанных сущностей.
    Parameters:
        errors: Словарь с ошибочно извлеченными сущностями и количеством сущностей каждого типа        
    Return:
        Матрица ошибок в виде словаря, где первый ключ - правильный тип сущности,
        второй ключ - предсказанный тип, значение - количество или доля ошибок
    '''
    confusion_matrix = defaultdict(lambda: defaultdict(float))
    for entity_type, mistaken_extractions in errors['mistaken_extractions'].items():
        for _, predicted_type, _ in mistaken_extractions:
            confusion_matrix[entity_type][predicted_type] += (1 / errors['number_of_entities'][entity_type])
    return confusion_matrix


def aggregate_mistaken_extractions(confusion_matrix: tp.Dict[str, tp.Dict[str, float]]) -> tp.Dict[str, tp.Dict[str, float]]:
    """
    Вычисляет максимальную долю ошибок для каждого класса сущностей.
    Parameters:
        confusion_matrix: Матрица ошибок в виде словаря словарей        
    Returns:
        Словарь с максимальной долей ошибок для каждого класса
    """
    aggregated_errors = {}
    for actual_type, predictions in confusion_matrix.items():
        if predictions:
            max_error_type = None
            max_error = 0
            for predicted_type, error_rate in predictions.items():
                if error_rate > max_error:
                    max_error = error_rate
                    max_error_type = predicted_type
            if max_error_type:
                aggregated_errors[actual_type] = {'max_error_rate': max_error,
                                                  'confused_with': max_error_type}
    return aggregated_errors


def plot_confusion_matrix(confusion_matrix: tp.Dict[str, tp.Dict[str, float]], 
                          add_description: bool = True) -> tp.Any:
    '''
    Строит матрицу ошибок для неправильно распознанных сущностей.
    Parameters:
        confusion_matrix: Словарь с матрицей ошибок, где ключи - реальные и предсказанные типы сущностей
        add_description: Если True, добавляет подробное описание для каждой ячейки
    Return:
        Объект matplotlib Figure с матрицей ошибок
    '''
    entity_types = sorted(set(confusion_matrix.keys()) | 
                         set(entity_type for types in confusion_matrix.values() for entity_type in types))

    matrix_data = [[confusion_matrix.get(row, {}).get(col, 0) for col in entity_types] 
                   for row in entity_types]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(matrix_data, cmap='tab20b')
    ax.set_xticks(np.arange(len(entity_types)))
    ax.set_yticks(np.arange(len(entity_types)))
    ax.set_xticklabels(entity_types)
    ax.set_yticklabels(entity_types)

    for i in range(len(entity_types)):
        for j in range(len(entity_types)):
            if i != j:
                value = matrix_data[i][j]
                if add_description:
                    ax.text(j, i, f"{value:.2f}\nAct: {entity_types[i]}\nPred: {entity_types[j]}",
                           ha="center", va="center", color="w")
                else:
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="w")             

    ax.set_ylabel('Actual', fontdict=dict(weight='bold'))
    ax.set_xlabel('Predicted', fontdict=dict(weight='bold'))
    ax.set_title('Mistaken recognitions confusion matrix')

    fig.tight_layout()
    plt.show()
    return fig
