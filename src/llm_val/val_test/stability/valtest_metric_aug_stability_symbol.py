import random
import numpy as np
import typing as tp
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.utils.retries import call_with_retries
from llm_val.utils.custom_checkers import check_if_correct
from llm_val.utils.human_augmentations import batch_augmentations
from llm_val.utils.report_helper import semaphore_by_threshold, worst_semaphore


def report_valtest_metric_aug_stability_symbol(metric_baseline: tp.Dict[str, float],
                                               metrics_augmented: tp.Dict[str, tp.Dict[str, float]],
                                               main_metric: str,
                                               augmentations_list: tp.Optional[tp.List[str]]=None,
                                               semaphore_threshold: tp.Tuple[float, float]=(0.05, 0.10),
                                               greater_is_better: bool=False,
                                               is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а в соответствии с концепцией llm_val
    Parameters:
        metric_baseline: Словарь со значением метрики на данных без изменений
        metrics_augmented: Словарь со словарями значений метрики на аугментированных данных разными методами
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        augmentations_list: Список выполняемых аугментаций
        semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: Если True, то тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    augmentations_list = augmentations_list or get_default_augmentations()
    baseline_value = metric_baseline[main_metric]
    augmented_values = np.array([metrics_augmented[main_metric][name] for name in augmentations_list])
    absolute_difference = baseline_value - augmented_values
    relative_difference = absolute_difference / baseline_value * 100
    semaphore_colors = [semaphore_by_threshold(difference, semaphore_threshold, greater_is_better=greater_is_better)
                        for difference in absolute_difference]
    color = 'gray' if is_info else worst_semaphore(semaphore_colors)
    results = pd.DataFrame({'Тип преобразования': augmentations_list,
                            'Значение ключевой метрики': augmented_values.tolist(),
                            'Абсолютное изменение ключевой метрики': absolute_difference.tolist(),
                            'Относительное изменение ключевой метрики': relative_difference.tolist(),
                            'Результат теста': semaphore_colors})
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [results]
    return res_dict


def valtest_metric_aug_stability_symbol(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                        scorer: Scorer,
                                        model: tp.Any,
                                        main_metric: str,
                                        augmentations_list: tp.Optional[tp.List[str]]=None,
                                        p_augmented: float=0.1,
                                        answer_checker: tp.Optional[tp.Callable[..., float]]=None,
                                        metric_agg: str='single_mean',
                                        data_type: str='test',
                                        semaphore_threshold: tp.Tuple[float, float]=(0.05, 0.10),
                                        greater_is_better: bool=False,
                                        seed: int=42,
                                        timeout: float=60.0,
                                        retry_delay: float=0.1,
                                        max_attempts: int=3,
                                        exec_env: tp.Literal['default', 'sberds']='default',
                                        is_info: bool=False,
                                        metric_baseline: tp.Optional[tp.Dict[str, float]]=None,
                                        metrics_augmented: tp.Optional[tp.Dict[str, tp.Dict[str, float]]]=None,
                                        **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на стабильность прогноза модели в зависимости от символьных аугментаций запросов
    Parameters:
        sampler: Объект класса Sampler (exec_env='default') или словарь с данными (exec_env='sberds')
        scorer: Объект класса Scorer
        model: Валидируемая модель; имеет метод predict(запросы)
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        augmentations_list: Список выполняемых аугментаций 
            доступны: 'human_level_augmentations', 'symbol_swap', 'butter_finger'.
        p_augmented: Вероятность аугментации символа каждой аугментацией
        answer_checker: Функция автооценки ответа с сигнатурой (запрос, ответ модели, эталонный ответ)
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
        greater_is_better: True, если чем больше метрика, тем лучше
        seed: Начальное значение датчика случайных чисел
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
        exec_env: Среда выполнения ('default' или 'sberds')
        is_info: True, если тест информативный
        metric_baseline: Опциональный словарь со значением метрики на данных без изменений из precomputed
        metrics_augmented: Опциональный словарь со словарями значений метрики на аугментированных данных из precomputed
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    augmentations_list = augmentations_list or get_default_augmentations()
    if metric_baseline is None:
        metric_baseline = compute_baseline_metric(sampler, scorer, data_type, metric_agg, exec_env)
    if metrics_augmented is None:
        data = get_sampler_data(sampler, data_type, exec_env)
        checker = resolve_answer_checker(answer_checker)
        metrics_augmented = compute_augmented_metrics(data, model, augmentations_list, checker,
                                                      scorer, metric_agg, seed, p_augmented,
                                                      timeout, retry_delay, max_attempts)
    report = report_valtest_metric_aug_stability_symbol(metric_baseline,
                                                        metrics_augmented,
                                                        main_metric,
                                                        augmentations_list,
                                                        semaphore_threshold,
                                                        greater_is_better,
                                                        is_info)
    precomputed = {'metric_baseline': metric_baseline,
                   'metrics_augmented': metrics_augmented,
                   'augmentations_list': augmentations_list}
    return {'report': report,
            'precomputed': precomputed}


def get_default_augmentations() -> tp.List[str]:
    '''
    Возвращает список аугментаций по умолчанию для теста стабильности на уровне символов
    Return:
        Список символьных аугментаций согласно методике
    '''
    return ['human_level_augmentations', 'symbol_swap', 'butter_finger']


def resolve_answer_checker(answer_checker: tp.Optional[tp.Callable[..., float]]) -> tp.Callable[..., float]:
    '''
    Определяет функцию автооценки, подставляя exact-match по умолчанию
    Parameters:
        answer_checker: Переданная пользователем функция автооценки или None
    Return:
        Функция автооценки ответа
    '''
    checker = answer_checker if answer_checker is not None else default_answer_checker
    return checker


def default_answer_checker(query: tp.Any,
                           model_answer: str,
                           reference_answer: tp.Optional[str]=None) -> float:
    '''
    Автооценка ответа методом exact-match при наличии эталонного ответа
    Parameters:
        query: Запрос, на который получен ответ
        model_answer: Ответ модели
        reference_answer: Эталонный ответ для сравнения
    Return:
        Оценка корректности ответа в диапазоне от 0 до 1
    '''
    return float(check_if_correct(model_answer, reference_answer)) if reference_answer is not None else 0.0


def get_sampler_data(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                     data_type: str,
                     exec_env: tp.Literal['default', 'sberds']) -> pd.DataFrame:
    '''
    Получает данные из sampler с учетом среды выполнения теста
    Parameters:
        sampler: Объект класса Sampler (exec_env='default') или словарь с данными (exec_env='sberds')
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        exec_env: Среда выполнения ('default' или 'sberds')
    Return:
        DataFrame с данными для расчета теста
    '''
    if exec_env == 'sberds':
        sampler_dict = tp.cast(tp.Dict[str, tp.Any], sampler)
        data = sampler_dict[data_type]['X']
    else:
        sampler_object = tp.cast(Sampler, sampler)
        data = getattr(sampler_object, data_type)['X']
    return deepcopy(data)


def compute_baseline_metric(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                            scorer: Scorer,
                            data_type: str,
                            metric_agg: str,
                            exec_env: tp.Literal['default', 'sberds']) -> tp.Dict[str, float]:
    '''
    Считает метрику на исходных данных по объектным значениям метрики из y
    Parameters:
        sampler: Объект класса Sampler (exec_env='default') или словарь с данными (exec_env='sberds')
        scorer: Объект класса Scorer
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        metric_agg: Название агрегации метрики
        exec_env: Среда выполнения ('default' или 'sberds')
    Return:
        Значение метрики качества на исходных данных
    '''
    baseline_sampler = tp.cast(Sampler, sampler) if exec_env == 'default' \
        else Sampler(**{data_type: tp.cast(tp.Dict[str, tp.Any], sampler)[data_type]})
    return scorer.calc(sampler=baseline_sampler,
                       data_type=data_type,
                       metric_name=metric_agg)


def compute_augmented_metrics(data: pd.DataFrame,
                              model: tp.Any,
                              augmentations_list: tp.List[str],
                              answer_checker: tp.Callable[..., float],
                              scorer: Scorer,
                              metric_agg: str,
                              seed: int,
                              p_augmented: float,
                              timeout: float=60.0,
                              retry_delay: float=0.1,
                              max_attempts: int=3) -> tp.Dict[str, tp.Dict[str, float]]:
    '''
    Считает значения метрики на данных, аугментированных каждым методом из списка
    Parameters:
        data: Исследуемый набор данных
        model: Валидируемая модель; имеет метод predict(запросы)
        augmentations_list: Список выполняемых аугментаций
        answer_checker: Функция автооценки ответа
        scorer: Объект класса Scorer
        metric_agg: Название агрегации метрики
        seed: Начальное значение датчика случайных чисел
        p_augmented: Вероятность аугментации символа каждой аугментацией
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
    Return:
        Словарь со словарями значений метрики на аугментированных данных разными методами
    '''
    metrics_augmented = {}
    for aug_name in tqdm(augmentations_list):
        score = compute_augmented_metric_for_method(data, model, aug_name, answer_checker,
                                                    scorer, metric_agg, seed, p_augmented,
                                                    timeout, retry_delay, max_attempts)
        for key, value in score.items():
            sub_dict = metrics_augmented.get(key, {})
            sub_dict[aug_name] = value
            metrics_augmented[key] = sub_dict
    return metrics_augmented


def compute_augmented_metric_for_method(data: pd.DataFrame,
                                        model: tp.Any,
                                        aug_name: str,
                                        answer_checker: tp.Callable[..., float],
                                        scorer: Scorer,
                                        metric_agg: str,
                                        seed: int,
                                        p_augmented: float,
                                        timeout: float=60.0,
                                        retry_delay: float=0.1,
                                        max_attempts: int=3) -> tp.Dict[str, float]:
    '''
    Считает метрику для одной аугментации, пропуская инференс или аугментацию по готовым колонкам
    Parameters:
        data: Исследуемый набор данных
        model: Валидируемая модель; имеет метод predict(запросы)
        aug_name: Название аугментации
        answer_checker: Функция автооценки ответа
        scorer: Объект класса Scorer
        metric_agg: Название агрегации метрики
        seed: Начальное значение датчика случайных чисел
        p_augmented: Вероятность аугментации символа
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
    Return:
        Значение метрики качества для данной аугментации
    '''
    pred_column = 'pred_aug_valtest_metric_aug_stability_symbol_' + aug_name
    aug_column = 'aug_valtest_metric_aug_stability_symbol_' + aug_name
    if pred_column in data.columns:
        metric_values = get_metric_values_from_predictions(data, pred_column, answer_checker)
        metric = calc_metric(metric_values, scorer, metric_agg)
    else:
        augmented_data = data[aug_column].to_list() if aug_column in data.columns else None
        metric = compute_metric(data, model, answer_checker, aug_name, augmented_data,
                                scorer, metric_agg, seed, p_augmented,
                                timeout, retry_delay, max_attempts)
    return metric


def get_metric_values_from_predictions(data: pd.DataFrame,
                                       pred_column: str,
                                       answer_checker: tp.Callable[..., float]) -> tp.List[float]:
    '''
    Считает объектные значения метрики по готовым предсказаниям модели для формата отдельных запросов
    Parameters:
        data: Исследуемый набор данных с колонками input_query и предсказаний модели
        pred_column: Название колонки с предсказаниями модели
        answer_checker: Функция автооценки ответа
    Return:
        Список объектных значений метрики качества
    '''
    queries = data['input_query'].to_list()
    predictions = data[pred_column].to_list()
    references = data['reference_answer'].to_list() if 'reference_answer' in data.columns else [None] * len(queries)
    return [answer_checker(query, prediction, reference)
            for query, prediction, reference in zip(queries, predictions, references)]


def compute_metric(data: pd.DataFrame,
                   model: tp.Any,
                   answer_checker: tp.Callable[..., float],
                   aug_name: tp.Optional[str],
                   augmented_data: tp.Optional[tp.List[str]],
                   scorer: Scorer,
                   metric_agg: str,
                   seed: int,
                   p_augmented: float,
                   timeout: float=60.0,
                   retry_delay: float=0.1,
                   max_attempts: int=3) -> tp.Dict[str, float]:
    '''
    Считает значение метрики качества на исходных или аугментированных запросах
    Parameters:
        data: Исследуемый набор данных
        model: Валидируемая модель; имеет метод predict(запросы)
        answer_checker: Функция автооценки ответа
        aug_name: Название аугментации либо None для исходных данных
        augmented_data: Готовые аугментированные запросы либо None для аугментации внутри
        scorer: Объект класса Scorer
        metric_agg: Название агрегации метрики
        seed: Начальное значение датчика случайных чисел
        p_augmented: Вероятность аугментации символа
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
    Return:
        Значение метрики качества
    '''
    rng = build_rng(seed)
    metric_values = get_queries_metric_values(data, model, answer_checker,
                                              aug_name, augmented_data, rng, p_augmented,
                                              timeout, retry_delay, max_attempts)
    return calc_metric(metric_values, scorer, metric_agg)


def build_rng(seed: int) -> tp.Any:
    '''
    Создает генератор случайных чисел и фиксирует датчики случайных чисел
    Parameters:
        seed: Начальное значение датчика случайных чисел
    Return:
        Генератор случайных чисел numpy
    '''
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def get_queries_metric_values(data: pd.DataFrame,
                              model: tp.Any,
                              answer_checker: tp.Callable[..., float],
                              aug_name: tp.Optional[str],
                              augmented_data: tp.Optional[tp.List[str]],
                              rng: tp.Any,
                              p_augmented: float,
                              timeout: float=60.0,
                              retry_delay: float=0.1,
                              max_attempts: int=3) -> tp.List[float]:
    '''
    Реинференс и автооценка ответов для формата отдельных запросов
    Parameters:
        data: Исследуемый набор данных с колонкой input_query
        model: Валидируемая модель; имеет метод predict(запросы)
        answer_checker: Функция автооценки ответа
        aug_name: Название аугментации либо None для исходных запросов
        augmented_data: Готовые аугментированные запросы либо None для аугментации внутри
        rng: Генератор случайных чисел
        p_augmented: Вероятность аугментации символа
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
    Return:
        Список объектных значений метрики качества
    '''
    queries = data['input_query'].to_list()
    references = data['reference_answer'].to_list() if 'reference_answer' in data.columns else [None] * len(queries)
    augmented_queries = augmented_data if augmented_data is not None \
        else apply_augmentation(queries, aug_name, rng, p_augmented)
    model_answers = call_with_retries(model.predict, timeout, retry_delay, max_attempts, (augmented_queries,))
    return [answer_checker(query, answer, reference)
            for query, answer, reference in zip(augmented_queries, model_answers, references)]


def apply_augmentation(texts: tp.List[str],
                       aug_name: tp.Optional[str],
                       rng: tp.Any,
                       p_augmented: float) -> tp.List[str]:
    '''
    Применяет к набору текстов выбранную символьную аугментацию
    Parameters:
        texts: Набор текстов (строк)
        aug_name: Название аугментации либо None для исходных текстов
        rng: Генератор случайных чисел
        p_augmented: Вероятность аугментации символа
    Return:
        Аугментированный набор текстов либо исходный набор при aug_name None
    '''
    if aug_name is None:
        augmented_texts = list(texts)
    else:
        char_aug_dict = {'human_level_augmentations': ['use_deletion', 'use_swap', 'use_insert',
                                                       'use_bf', 'use_special_symbol'],
                         'symbol_swap': ['use_swap'],
                         'butter_finger': ['use_bf']}
        aug_params = {param: True for param in char_aug_dict[aug_name]}
        augmented_texts = list(batch_augmentations(batch=list(texts), rng=rng, p=p_augmented, **aug_params))
    return augmented_texts


def calc_metric(metric_values: tp.List[float],
                scorer: Scorer,
                metric_agg: str) -> tp.Dict[str, float]:
    '''
    Считает значение метрики качества по объектным значениям метрики
    Parameters:
        metric_values: Список объектных значений метрики качества
        scorer: Объект класса Scorer
        metric_agg: Название агрегации метрики
    Return:
        Значение метрики качества
    '''
    dict_for_scorer = {'X': None, 'y': pd.DataFrame({'metric_value': metric_values})}
    return scorer.calc(sampler=Sampler(test=dict_for_scorer),
                       data_type='test',
                       metric_name=metric_agg)


def get_augmented_data(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                       data_type: str='test',
                       augmentations_list: tp.Optional[tp.List[str]]=None,
                       p_augmented: float=0.1,
                       seed: int=42,
                       exec_env: tp.Literal['default', 'sberds']='default') -> pd.DataFrame:
    '''
    Проводит символьные аугментации запросов всеми методами из списка для проведения реинференса
    Parameters:
        sampler: Объект класса Sampler (exec_env='default') или словарь с данными (exec_env='sberds')
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        augmentations_list: Список выполняемых аугментаций
        p_augmented: Вероятность аугментации символа каждой аугментацией
        seed: Начальное значение датчика случайных чисел
        exec_env: Среда выполнения ('default' или 'sberds')
    Return:
        DataFrame с аугментированными запросами в колонках aug_valtest_metric_aug_stability_symbol_<aug>
    '''
    augmentations_list = augmentations_list or get_default_augmentations()
    data = get_sampler_data(sampler, data_type, exec_env)
    aug_df = pd.DataFrame(index=data.index)
    for aug_name in augmentations_list:
        rng = build_rng(seed)
        aug_df['aug_valtest_metric_aug_stability_symbol_' + aug_name] = build_augmented_column(data, aug_name,
                                                                                               rng, p_augmented)
    return aug_df


def build_augmented_column(data: pd.DataFrame,
                           aug_name: str,
                           rng: tp.Any,
                           p_augmented: float) -> tp.List[str]:
    '''
    Формирует значения одной aug-колонки с аугментированными запросами
    Parameters:
        data: Исследуемый набор данных
        aug_name: Название аугментации
        rng: Генератор случайных чисел
        p_augmented: Вероятность аугментации символа
    Return:
        Список аугментированных запросов
    '''
    return apply_augmentation(data['input_query'].to_list(), aug_name, rng, p_augmented)
