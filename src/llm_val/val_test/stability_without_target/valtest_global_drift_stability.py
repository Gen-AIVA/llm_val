import numpy as np
import typing as tp
import pandas as pd
from scipy import stats
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_distances

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.val_test.model_quality.valtest_metric import valtest_metric
from llm_val.utils.report_helper import semaphore_by_threshold, worst_semaphore


def report_valtest_global_drift_stability(metric_value: tp.Dict[str, float],
                                          metric_value_estimate: float,
                                          main_metric: str,
                                          test_color: str,
                                          data_types: tp.Tuple[str, str]=('train', 'test'),
                                          semaphore_threshold: tp.Tuple[float, float]=(0.15, 0.25),
                                          greater_is_better: bool=True,
                                          is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        metric_value: Словарь со значением метрики на старых данных
        metric_value_estimate: Оценка значения метрики на новых данных
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        test_color: Светофор по тесту на целевую метрику на out-of-time выборке
        data_types: Типы данных, на которых сравниваются метрики
        semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: Если True, то тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    metric_value = metric_value[main_metric]
    df = pd.DataFrame({f'Значение метрики на {data_types[0]}': [metric_value],
                       f'Оценённое значение метрики на {data_types[1]}': [metric_value_estimate],
                       'Абсолютная разница': metric_value_estimate - metric_value})
    df['Абсолютная разница'] = -df['Абсолютная разница'] if greater_is_better else df['Абсолютная разница']
    df['Результат теста'] = df.apply(lambda x: 'gray' if is_info else semaphore_by_threshold(x['Абсолютная разница'],
                                                                                             semaphore_threshold,
                                                                                             greater_is_better=False), axis=1)
    color = worst_semaphore([test_color, df['Результат теста'].values[0]])
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [df]
    return res_dict


def valtest_global_drift_stability(sampler: Sampler,
                                   scorer: Scorer,
                                   main_metric: str,
                                   model: tp.Any,
                                   n_chunks: int,
                                   distance_func: tp.Callable[[tp.Any, tp.Any], tp.Any]=cosine_distances,
                                   top_distance_features: tp.Optional[tp.List[int]]=[5, 10, 15],
                                   p_value: float=0.05,
                                   corr_threshold: float=0.5,
                                   random_state: int=42,
                                   metric_binarizer: tp.Optional[tp.Callable[[pd.DataFrame], pd.DataFrame]]=None,
                                   metric_agg: str='single_mean',
                                   data_types: tp.Tuple[str, str]=('train', 'test'),
                                   semaphore_threshold: tp.Tuple[float, float]=(0.15, 0.25),
                                   greater_is_better: bool=True,
                                   is_info: bool=False,
                                   metric_value: tp.Optional[tp.Dict[str, float]]=None,
                                   test_color: tp.Optional[str]=None,
                                   metric_value_estimate: tp.Optional[float]=None,
                                   **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на анализ качества ответа модели в зависимости от глобального drift запросов
    Parameters:
        sampler: Объект класса Sampler
        scorer: Объект класса Scorer
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        model: Валидируемая модель; имеет метод get_embedding(список вопросов)
        n_chunks: Число подвыборок, на которые делятся запросы из сэмплера 
        distance_func: Функция для подсчета расстояния между эмбеддингами
        top_distance_features: Список значений числа наибольших расстояний, которые будут использованы для извлечения признаков  
        p_value: Порог значимости корреляции Пирсона
        corr_threshold: Пороговое значение корреляции, выше которого признак считается коррелирующим
        random_state: Начальное состояние датчика случайных чисел
        metric_binarizer: Функция для бинаризации набора с метками качества, возвращает набор со столбцом 'metric_value'
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        data_types: Типы данных, на которых сравниваются метрики
        semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
        metric_value: Опциональный словарь со значением метрики на старых данных
        test_color: Результат теста 2.1 на ключевую метрику качества (опциональный) 
            Если metric_value и test_color не None, то подсчет метрики не производится и на ее основе выставляется светофор
        metric_value_estimate: Опциональное значение метрики на новых данных
            Если не None, то подсчет метрик не производится и на их основе выставляется светофор
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if metric_value_estimate is None:
        sampler_copy = deepcopy(sampler)
        if metric_binarizer is not None:
            train, test = getattr(sampler_copy, data_types[0]), getattr(sampler_copy, data_types[1])
            setattr(sampler_copy, data_types[0], {'X': train['X'], 'y': pd.DataFrame({'metric_value': metric_binarizer(train['y'])})})
            setattr(sampler_copy, data_types[1], {'X': test['X'], 'y': pd.DataFrame({'metric_value': metric_binarizer(test['y'])})})
        data_train = getattr(sampler_copy, data_types[0])
        data_test = getattr(sampler_copy, data_types[1])
        base_X, add_X, base_Y, add_Y = train_test_split(data_train['X'],
                                                        data_train['y'],
                                                        test_size=0.5,
                                                        random_state=random_state)
        base_embeddings = model.get_embedding(base_X['question'].to_list())
        features = []
        test_metric_values = []
        features_getters = [np.mean, np.std, np.max, np.median]
        for num in top_distance_features:
            features_getters.append(lambda x: np.sort(x)[-num:].sum())
        for chunk_X_idx, chunk_Y_idx in zip(np.array_split(add_X.index, n_chunks), np.array_split(add_Y.index, n_chunks)):
            chunk_X, chunk_Y = add_X.loc[chunk_X_idx], add_Y.loc[chunk_Y_idx]
            distances = distance_func(model.get_embedding(chunk_X['question'].to_list()), base_embeddings)
            sampler_copy.test['y'] = pd.DataFrame({'metric_value': chunk_Y.values.reshape(-1)})
            test_metric_value = scorer.calc(sampler=sampler_copy,
                                            data_type='test',
                                            metric_name=metric_agg)
            test_metric_values.append(test_metric_value[main_metric])
            features.append([func(dist) for func in features_getters for dist in [distances, distances.mean(1)]])
        distances_test = distance_func(model.get_embedding(data_test['X']['question'].to_list()), base_embeddings)
        test_features = np.array([func(dist) for func in features_getters for dist in [distances_test, distances_test.mean(1)]])
        features = np.array(features)
        train_features = []
        for idx in range(len(features[0])):
            corr_res = stats.pearsonr(features[:, idx], test_metric_values)
            if corr_res.pvalue < p_value and np.abs(corr_res.statistic) > corr_threshold:
                train_features.append(idx)
        if train_features:
            lin_reg = LinearRegression().fit(features[:, train_features], test_metric_values)
            metric_value_estimate = lin_reg.predict(test_features[train_features].reshape(1, -1))[0]
            metric_value_estimate = np.clip(metric_value_estimate, 0, 1)
        else:
            is_info = True
            metric_value_estimate = np.nan
        if metric_value is None or test_color is None:
            metric_result = valtest_metric(sampler=sampler_copy,
                                           scorer=scorer,
                                           main_metric=main_metric,
                                           data_type=data_types[0],
                                           metric_agg=metric_agg,
                                           greater_is_better=greater_is_better)
            test_color = metric_result['report']['semaphore']
            metric_value = metric_result['precomputed']['metric_value']
    precomputed = {'metric_value': metric_value[main_metric],
                   'metric_value_estimate': metric_value_estimate}
    report = report_valtest_global_drift_stability(metric_value,
                                                   metric_value_estimate,
                                                   main_metric,
                                                   test_color,
                                                   data_types,
                                                   semaphore_threshold,
                                                   greater_is_better,
                                                   is_info)
    return {'report': report,
            'precomputed': precomputed}
