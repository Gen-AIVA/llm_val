import numpy as np
import typing as tp
import pandas as pd
from copy import deepcopy

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.val_test.model_quality.valtest_metric import valtest_metric
from llm_val.utils.report_helper import semaphore_by_threshold, worst_semaphore


def report_valtest_local_drift_stability(metric_value: tp.Dict[str, float],
                                         metric_value_estimate: float,
                                         reliability_value: float,
                                         main_metric: str,
                                         test_color: str,
                                         data_types: tp.Tuple[str, str]=('train', 'test'),
                                         semaphore_threshold: tp.Tuple[float]=(0.15, 0.25),
                                         reliability_threshold: float=0.2,
                                         greater_is_better: bool=True,
                                         is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        metric_value: Словарь со значением метрики на старых данных
        metric_value_estimate: Оценка значения метрики на новых данных
        reliability_value: Значение надежности теста
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        test_color: Светофор по тесту на целевую метрику на out-of-time выборке
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        data_types: Типы данных, на которых сравниваются метрики
        semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
        reliability_threshold: Граница надежности, если значение надежности ниже порога, то тест информативный
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: Если True, то тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    metric_value = metric_value[main_metric]
    df = pd.DataFrame({f'Значение метрики на {data_types[0]}': [metric_value],
                       f'Оценённое значение метрики на {data_types[1]}': [metric_value_estimate],
                       'Абсолютная разница': metric_value_estimate - metric_value,
                       'Мера надежности теста': reliability_value})
    df['Абсолютная разница'] = -df['Абсолютная разница'] if greater_is_better else df['Абсолютная разница']
    df['Результат теста'] = df.apply(lambda x: 'gray' if is_info or reliability_value < reliability_threshold
                                     else semaphore_by_threshold(x['Абсолютная разница'],
                                                                 semaphore_threshold,
                                                                 greater_is_better=False), axis=1)
    color = worst_semaphore([test_color, df['Результат теста'].values[0]]) if df['Результат теста'].values[0] != 'gray' else 'gray'
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [df]
    return res_dict


def valtest_local_drift_stability(sampler: Sampler,
                                  scorer: Scorer,
                                  main_metric: str,
                                  model: tp.Any,
                                  ann: tp.Any,
                                  ann_config: tp.Dict[str, tp.Any],
                                  n_closest: int=5,
                                  metric_binarizer: tp.Optional[tp.Callable[[pd.DataFrame], pd.DataFrame]]=None,
                                  metric_agg: str='single_mean',
                                  data_types: tp.Tuple[str, str]=('train', 'test'),
                                  semaphore_threshold: tp.Tuple[float]=(0.15, 0.25),
                                  reliability_threshold: float=0.2,
                                  greater_is_better: bool=True,
                                  is_info: bool=False,
                                  metric_value: tp.Optional[tp.Dict[str, float]]=None,
                                  test_color: tp.Optional[str]=None,
                                  metric_value_estimate: tp.Optional[float]=None,
                                  reliability_value: tp.Optional[float]=None,
                                  **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на анализ качества ответа модели в зависимости от локального drift запросов
    Parameters:
        sampler: Объект класса Sampler
        scorer: Объект класса Scorer
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        model: Валидируемая модель; имеет метод get_embedding(список вопросов)
        ann: Объект класса реализующего интерфейс Approximate Nearest Neighbor, имеет методы create_index и search_query
        ann_config: Словарь с параметрами запуска для ANN
        n_closest: Число выбираемых наиболее близких запросов
        metric_binarizer: Функция для бинаризации набора с метками качества, возвращает набор со столбцом 'metric_value'
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        data_types: Типы данных, на которых сравниваются метрики
        semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
        reliability_threshold: Граница надежности, если значение надежности ниже порога, то тест информативный
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
        metric_value: Опциональный словарь со значением метрики на старых данных
        test_color: Результат теста 2.1 на ключевую метрику качества (опционально)
            Если metric_value и test_color не None,
            то подсчет метрики не производится и на ее основе выставляется светофор
        metric_value_estimate: Опциональное значение метрики на новых данных
        reliability_value: Опциональное значение надежности
            Если metric_value_estimate и reliability_value не None,
            то подсчет метрик не производится и на их основе выставляется светофор
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    oos = getattr(sampler, data_types[0])
    oot = getattr(sampler, data_types[1])
    sampler_copy = deepcopy(sampler)
    X_oos, y_oos = oos['X'], oos['y']
    X_oot, y_oot = oot['X'], oot['y']
    if metric_binarizer is not None:
        y_oos = metric_binarizer(y_oos)
        setattr(sampler_copy, data_types[0], {'X': oos['X'], 'y': pd.DataFrame({'metric_value': y_oos})})
        setattr(sampler_copy, data_types[1], {'X': oos['X'], 'y': pd.DataFrame({'metric_value': metric_binarizer(y_oot)})})
    test_scores, reliability_values = [], []
    if metric_value is None or test_color is None:
        metric_result = valtest_metric(sampler=sampler_copy,
                                       scorer=scorer,
                                       main_metric=main_metric,
                                       metric_agg=metric_agg,
                                       greater_is_better=greater_is_better)
        test_color = metric_result['report']['semaphore']
        metric_value = metric_result['precomputed']['metric_value']
    if metric_value_estimate is None or reliability_value is None:
        ann.create_index(model.get_embedding(X_oos['question'].to_list()),
                         **ann_config['create_index'])
        test_embeddings = model.get_embedding(X_oot['question'].to_list())
        for query_emb in test_embeddings:
            query_res = ann.search_query(query_emb,
                                         n_closest,
                                         **ann_config['search_query'])
            reliability_values.append(np.mean([item['similarity'] for item in query_res]))
            y_oos[y_oos==0] = -1
            test_scores.append(0.5 + sum(item['similarity'] * y_oos.iloc[item['id']] for item in query_res) / (2 * n_closest))
        metric_value_estimate = np.mean(test_scores)
        reliability_value = np.mean(reliability_values)
    precomputed = {'metric_value': metric_value[main_metric],
                   'metric_value_estimate': metric_value_estimate,
                   'reliability': reliability_value}
    report = report_valtest_local_drift_stability(metric_value,
                                                  metric_value_estimate,
                                                  reliability_value,
                                                  main_metric,
                                                  test_color,
                                                  data_types,
                                                  semaphore_threshold,
                                                  reliability_threshold,
                                                  greater_is_better,
                                                  is_info)
    return {'report': report,
            'precomputed': precomputed}
