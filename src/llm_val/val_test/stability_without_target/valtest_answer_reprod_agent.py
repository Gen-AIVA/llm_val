import random
import typing as tp
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.retries import call_with_retries
from llm_val.utils.openai_utils import get_openai_embeddings
from llm_val.utils.human_augmentations import batch_augmentations
from llm_val.utils.report_helper import worst_semaphore, semaphore_reproducibility_agent


def report_valtest_answer_reprod_agent(orig_diversity: tp.Dict[str, float],
                                       is_creative: bool,
                                       figs: tp.List[tp.Any],
                                       aug_diversity: tp.Optional[tp.Dict[str, float]]=None,
                                       diversity_threshold_pp: tp.Tuple[float, float]=(0.03, 0.05),
                                       rel_threshold: float=0.5,
                                       is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а в соответствии с концепцией llm_val
    Parameters:
        orig_diversity: Словарь с лексическим и семантическим разнообразием на исходных данных
        is_creative: True для творческой задачи, False для точной
        figs: Список графиков
        aug_diversity: Словарь с разнообразием на аугментированных данных
        diversity_threshold_pp: Нижний и верхний пороги разнообразия в долях
        rel_threshold: Порог относительной разницы
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    creativity = {'Лексическое разнообразие': False,
                  'Семантическое разнообразие': True}
    keys = {'Лексическое разнообразие': 'lexical',
            'Семантическое разнообразие': 'semantic'}
    result_df = build_result_table(orig_diversity,
                                   aug_diversity,
                                   keys)
    for label, metric_key in keys.items():
        result_df.loc[label, 'Результат теста'] = semaphore_reproducibility_agent(
            result_df.loc[label, 'Исходная'],
            result_df.loc[label, 'Относительная разница'],
            creativity[label],
            diversity_threshold_pp,
            rel_threshold)
    main_label = 'Семантическое разнообразие' if is_creative else 'Лексическое разнообразие'
    color = worst_semaphore([result_df.loc[main_label, 'Результат теста']])
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else color
    res_dict['result_plots'] = figs
    res_dict['result_dataframes'] = [result_df]
    return res_dict


def valtest_answer_reprod_agent(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                data_type: str='test',
                                model: tp.Optional[tp.Any]=None,
                                is_long_answer: bool=True,
                                is_creative: bool=False,
                                n_generated_answers: int=10,
                                n_gramms_length: int=3,
                                embedding_client: tp.Optional[tp.Any]=None,
                                embedding_model_name: str='Embeddings',
                                use_augmentations: bool=True,
                                augmentations_list: tp.Optional[tp.List[str]]=None,
                                p_augmented: float=0.1,
                                diversity_threshold_pp: tp.Tuple[float, float]=(0.03, 0.05),
                                rel_threshold: float=0.5,
                                seed: int=42,
                                timeout: float=60.0,
                                retry_delay: float=0.1,
                                max_attempts: int=3,
                                exec_env: tp.Literal['default', 'sberds']='default',
                                is_info: bool=False,
                                orig_diversity: tp.Optional[tp.Dict[str, float]]=None,
                                aug_diversity: tp.Optional[tp.Dict[str, float]]=None,
                                **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест стабильности воспроизведения ответов агента на схожих запросах: оценивает смысловую и языковую стабильность ответов решения путем многократного реинференса отдельных запросов.
    Parameters:
        sampler: Объект класса Sampler (exec_env='default') или словарь с данными (exec_env='sberds')
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        model: Валидируемая языковая модель, имеет метод predict(запросы)
        is_long_answer: True, если ответы развернутые, иначе ожидаются короткие ответы
        is_creative: True для творческой задачи, False для точной
        n_generated_answers: Число повторных генераций ответов
        n_gramms_length: Длина n-грамм для лексического разнообразия
        embedding_client: OpenAI-совместимый клиент с методом embeddings.create. Обязателен при is_long_answer=True
        embedding_model_name: Название эмбеддинг-модели, передаваемое в embeddings.create
        use_augmentations: Флаг включения проверки на аугментированных данных
        augmentations_list: Список символьных аугментаций: 'use_deletion', 'use_swap', 'use_insert', 'use_bf',
            'use_special_symbol', 'use_translit_symbol'. По умолчанию None,
            тогда используется ['use_special_symbol'] для выполнения шага аугментаций из методики без явного указания
        p_augmented: Вероятность аугментации символа
        diversity_threshold_pp: Нижний и верхний пороги разнообразия в долях
        rel_threshold: Порог относительной разницы
        seed: Начальное значение датчика случайных чисел
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
        exec_env: Среда выполнения ('default' или 'sberds')
        is_info: True, если тест информативный
        orig_diversity: Опциональное разнообразие на исходных данных из precomputed
        aug_diversity: Опциональное разнообразие на аугментированных данных из precomputed
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if orig_diversity is None or (use_augmentations and aug_diversity is None):
        data = get_sampler_data(sampler, data_type, exec_env)
        diversity_func = partial(compute_pairwise_diversity,
                                 is_long_answer=is_long_answer,
                                 n_gramms_length=n_gramms_length,
                                 embedding_client=embedding_client,
                                 embedding_model_name=embedding_model_name,
                                 timeout=timeout,
                                 retry_delay=retry_delay,
                                 max_attempts=max_attempts)
        if orig_diversity is None:
            orig_diversity = compute_orig_diversity(data, model, diversity_func, n_generated_answers,
                                                    timeout, retry_delay, max_attempts)
        if use_augmentations and aug_diversity is None:
            aug_diversity = compute_aug_diversity(data, model, diversity_func, n_generated_answers,
                                                  augmentations_list, seed, p_augmented, timeout, retry_delay,
                                                  max_attempts)
    figs = [plot_diversity_metrics(orig_diversity, aug_diversity)]
    precomputed = {'orig_diversity': orig_diversity,
                   'aug_diversity': aug_diversity}
    report = report_valtest_answer_reprod_agent(orig_diversity,
                                                is_creative,
                                                figs,
                                                aug_diversity,
                                                diversity_threshold_pp,
                                                rel_threshold,
                                                is_info)
    return {'report': report,
            'precomputed': precomputed}


def compute_orig_diversity(data: pd.DataFrame,
                           model: tp.Any,
                           diversity_func: tp.Callable[[tp.List[str]], tp.Tuple[float, float]],
                           n_generated_answers: int,
                           timeout: float=60.0,
                           retry_delay: float=0.1,
                           max_attempts: int=3) -> tp.Dict[str, float]:
    '''
    Считает разнообразие на исходных данных, переиспользуя колонки предсказаний при их наличии
    Parameters:
        data: Данные из сэмплера
        model: Языковая модель, имеет метод predict(запросы)
        diversity_func: Функция расчета лексического и семантического разнообразия для списка ответов
        n_generated_answers: Число повторных генераций ответов
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
    Return:
        Словарь со средним лексическим и семантическим разнообразием
    '''
    if find_columns_by_prefix(data, 'pred_valtest_answer_reprod_agent_'):
        predictions = predictions_from_columns(data, 'pred_valtest_answer_reprod_agent_')
    else:
        predictions = get_query_predictions(model, data['input_query'].to_list(), n_generated_answers,
                                            timeout, retry_delay, max_attempts)
    return evaluate_mean_diversity(predictions, diversity_func)


def compute_aug_diversity(data: pd.DataFrame,
                          model: tp.Any,
                          diversity_func: tp.Callable[[tp.List[str]], tp.Tuple[float, float]],
                          n_generated_answers: int,
                          augmentations_list: tp.Optional[tp.List[str]],
                          seed: int,
                          p_augmented: float,
                          timeout: float=60.0,
                          retry_delay: float=0.1,
                          max_attempts: int=3) -> tp.Dict[str, float]:
    '''
    Считает разнообразие на аугментированных данных, переиспользуя предсказания или аугментации при их наличии
    Parameters:
        data: Данные из сэмплера
        model: Языковая модель, имеет метод predict(запросы)
        diversity_func: Функция расчета лексического и семантического разнообразия для списка ответов
        n_generated_answers: Число повторных генераций ответов
        augmentations_list: Список символьных аугментаций: 'use_deletion', 'use_swap', 'use_insert', 'use_bf',
            'use_special_symbol', 'use_translit_symbol'
        seed: Начальное значение датчика случайных чисел
        p_augmented: Вероятность аугментации символа
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
    Return:
        Словарь со средним лексическим и семантическим разнообразием
    '''
    effective_augmentations = augmentations_list if augmentations_list is not None else ['use_special_symbol']
    if find_columns_by_prefix(data, 'pred_aug_valtest_answer_reprod_agent_'):
        predictions = predictions_from_columns(data, 'pred_aug_valtest_answer_reprod_agent_')
    else:
        aug_columns = find_columns_by_prefix(data, 'aug_valtest_answer_reprod_agent_')
        if aug_columns:
            query_variants = values_from_columns(data, 'aug_valtest_answer_reprod_agent_')
            predictions = get_query_variant_predictions(model, query_variants, timeout, retry_delay, max_attempts)
        else:
            queries = get_augmented_queries(data['input_query'].to_list(), effective_augmentations, seed,
                                            p_augmented)
            predictions = get_query_predictions(model, queries, n_generated_answers,
                                                timeout, retry_delay, max_attempts)
    return evaluate_mean_diversity(predictions, diversity_func)


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


def find_columns_by_prefix(data: pd.DataFrame,
                           prefix: str) -> tp.List[str]:
    '''
    Находит колонки данных, имена которых начинаются с заданного префикса
    Parameters:
        data: Данные из сэмплера
        prefix: Префикс имени колонки
    Return:
        Список имен колонок, начинающихся с префикса
    '''
    return [column for column in data.columns if column.startswith(prefix)]


def predictions_from_columns(data: pd.DataFrame,
                             prefix: str) -> tp.List[tp.List[str]]:
    '''
    Восстанавливает предсказания по запросам из колонок с предвычисленными ответами
    Parameters:
        data: Данные из сэмплера с колонками предсказаний prefix + idx
        prefix: Префикс имен колонок предсказаний
    Return:
        Список по запросам, для каждого список генераций ответов
    '''
    predictions = values_from_columns(data, prefix)
    return predictions


def values_from_columns(data: pd.DataFrame,
                        prefix: str) -> tp.List[tp.List[str]]:
    '''
    Восстанавливает значения по строкам из колонок с заданным префиксом
    Parameters:
        data: Данные из сэмплера
        prefix: Префикс имен колонок
    Return:
        Список по строкам, для каждой строки список значений из найденных колонок
    '''
    columns = find_columns_by_prefix(data, prefix)
    values = []
    for _, row in data.iterrows():
        values.append([row[column] for column in columns])
    return values


def build_result_table(orig_diversity: tp.Dict[str, float],
                       aug_diversity: tp.Optional[tp.Dict[str, float]],
                       keys: tp.Dict[str, str]) -> pd.DataFrame:
    '''
    Формирует таблицу результатов теста по примеру оформления из методики
    Parameters:
        orig_diversity: Словарь с разнообразием на исходных данных
        aug_diversity: Словарь с разнообразием на аугментированных данных
        keys: Соответствие названий строк таблицы и ключей словарей разнообразия
    Return:
        Таблица результатов с абсолютной и относительной разницей и светофором
    '''
    rows = []
    for label, metric_key in keys.items():
        orig_value = orig_diversity[metric_key]
        aug_value = aug_diversity[metric_key] if aug_diversity is not None else np.nan
        abs_diff, rel_diff = compute_difference(orig_value, aug_value)
        rows.append({'Метрика': label,
                     'Исходная': orig_value,
                     'Аугментированная': aug_value,
                     'Абсолютная разница': abs_diff,
                     'Относительная разница': rel_diff,
                     'Результат теста': None})
    return pd.DataFrame(rows).set_index('Метрика')


def get_query_predictions(model: tp.Any,
                          queries: tp.List[str],
                          n_generated_answers: int=10,
                          timeout: float=60.0,
                          retry_delay: float=0.1,
                          max_attempts: int=3) -> tp.List[tp.List[str]]:
    '''
    Получает n_generated_answers повторных генераций по каждому запросу
    Parameters:
        model: Языковая модель, имеет метод predict(запросы)
        queries: Список запросов
        n_generated_answers: Число повторных генераций ответов
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
    Return:
        Список по запросам, для каждого список генераций ответов
    '''
    predictions = [[] for _ in queries]
    for _ in tqdm(range(n_generated_answers)):
        answers = call_with_retries(model.predict,
                                    timeout,
                                    retry_delay,
                                    max_attempts,
                                    (queries,))
        for query_idx, answer in enumerate(answers):
            predictions[query_idx].append(answer)
    return predictions


def get_query_variant_predictions(model: tp.Any,
                                  query_variants: tp.List[tp.List[str]],
                                  timeout: float=60.0,
                                  retry_delay: float=0.1,
                                  max_attempts: int=3) -> tp.List[tp.List[str]]:
    '''
    Получает по одному ответу модели на каждый заранее подготовленный вариант запроса
    Parameters:
        model: Языковая модель, имеет метод predict(запросы)
        query_variants: Список вариантов запросов по строкам данных
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
    Return:
        Список по запросам, для каждого список ответов на варианты запроса
    '''
    flat_queries = [query for query_group in query_variants for query in query_group]
    group_counts = [len(query_group) for query_group in query_variants]
    flat_answers = call_with_retries(model.predict,
                                     timeout,
                                     retry_delay,
                                     max_attempts,
                                     (flat_queries,))
    predictions = regroup_by_counts(flat_answers, group_counts)
    return predictions


def regroup_by_counts(flat_answers: tp.List[str],
                      group_counts: tp.List[int]) -> tp.List[tp.List[str]]:
    '''
    Группирует плоский список ответов в соответствии с размерами групп
    Parameters:
        flat_answers: Плоский список ответов модели
        group_counts: Размеры групп
    Return:
        Список ответов модели по группам
    '''
    grouped = []
    offset = 0
    for count in group_counts:
        grouped.append(list(flat_answers[offset:offset + count]))
        offset += count
    return grouped


def evaluate_mean_diversity(predictions: tp.List[tp.List[str]],
                            diversity_func: tp.Callable[[tp.List[str]], tp.Tuple[float, float]]) -> tp.Dict[str, float]:
    '''
    Считает среднее лексическое и семантическое разнообразие по запросам
    Parameters:
        predictions: Список по запросам, для каждого список генераций ответов
        diversity_func: Функция расчета лексического и семантического разнообразия для списка ответов
    Return:
        Словарь со средним лексическим и семантическим разнообразием
    '''
    lexical_values, semantic_values = [], []
    for query_predictions in predictions:
        lexical, semantic = diversity_func(query_predictions)
        lexical_values.append(lexical)
        semantic_values.append(semantic)
    return {'lexical': float(np.mean(lexical_values)),
            'semantic': float(np.mean(semantic_values))}


def compute_pairwise_diversity(sentences: tp.List[str],
                               is_long_answer: bool,
                               n_gramms_length: int,
                               embedding_client: tp.Optional[tp.Any]=None,
                               embedding_model_name: str='Embeddings',
                               timeout: float=60.0,
                               retry_delay: float=0.1,
                               max_attempts: int=3) -> tp.Tuple[float, float]:
    '''
    Считает лексическое и семантическое разнообразие для набора ответов на один запрос.
    Для развернутых ответов лексика - AverageNgrams, семантика - эмбеддинги и косинусная близость.
    Для коротких ответов оба разнообразия оцениваются как exact match.
    Parameters:
        sentences: Список ответов модели на один запрос
        is_long_answer: True, если ответы развернутые
        n_gramms_length: Длина n-грамм для лексического разнообразия
        embedding_client: OpenAI-совместимый клиент с методом embeddings.create. Обязателен при is_long_answer=True
        embedding_model_name: Название эмбеддинг-модели, передаваемое в embeddings.create
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
    Return:
        Пара значений лексического и семантического разнообразия
    '''
    if is_long_answer:
        if embedding_client is None:
            raise ValueError('embedding_client обязателен при is_long_answer=True')
        lexical = mean_pairwise_lingustic_diversity(sentences, n_gramms_length)
        semantic = mean_pairwise_semantic_diversity_emb(sentences,
                                                        embedding_client,
                                                        embedding_model_name,
                                                        timeout,
                                                        retry_delay,
                                                        max_attempts)
    else:
        exact_match_diversity = mean_pairwise_exact_match_diversity(sentences)
        lexical, semantic = exact_match_diversity, exact_match_diversity
    return lexical, semantic


def lines_to_ngrams(lines: tp.List[str],
                    n: int=3) -> tp.List[tp.List[tp.Tuple[str, ...]]]:
    '''
    Создает список n-грамм для каждой строки из поданного списка
    Parameters:
        lines: Список строк
        n: Длина n-грамм
    Return:
        Список списков n-грамм для каждой строки из поданного списка
    '''
    ngrams = []
    for line in lines:
        words = [word for word in line.replace('.', '').replace('\n', '').split(' ') if word != '']
        ngrams.append([tuple(words[i:(i + n)]) for i in range(len(words) - n + 1)])
    return ngrams


def mean_pairwise_lingustic_diversity(sentences: tp.List[str],
                                      n: int) -> float:
    '''
    Считает среднее попарное языковое разнообразие как 1 минус средний коэффициент Жаккара по n-граммам
    Parameters:
        sentences: Список строк
        n: Длина n-грамм
    Return:
        Языковое разнообразие ответов
    '''
    ngram_lists = lines_to_ngrams(sentences, n)
    count = len(sentences)
    similarities = []
    for i in range(count):
        for j in range(i + 1, count):
            unite = set(ngram_lists[i] + ngram_lists[j])
            intersection = set(ngram_lists[i]).intersection(set(ngram_lists[j]))
            sim_value = len(intersection) / len(unite) if len(unite) != 0 else 0
            similarities.append(sim_value)
    return 1 - sum(similarities) / len(similarities) if similarities else 0.0


def mean_pairwise_semantic_diversity_emb(sentences: tp.List[str],
                                         embedding_client: tp.Any,
                                         embedding_model_name: str='Embeddings',
                                         timeout: float=60.0,
                                         retry_delay: float=0.1,
                                         max_attempts: int=3) -> float:
    '''
    Считает среднее попарное смысловое разнообразие как 1 минус средняя косинусная близость эмбеддингов
    Parameters:
        sentences: Список строк
        embedding_client: OpenAI-совместимый клиент с методом embeddings.create
        embedding_model_name: Название эмбеддинг-модели, передаваемое в embeddings.create
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
    Return:
        Смысловое разнообразие ответов
    '''
    embeddings = call_with_retries(get_openai_embeddings,
                                   timeout,
                                   retry_delay,
                                   max_attempts,
                                   (embedding_client, sentences, embedding_model_name))
    similarity = cosine_similarity(embeddings, embeddings)
    count = len(sentences)
    upper_sum = np.triu(similarity, 1).sum()
    return 1 - 2 * upper_sum / (count * (count - 1)) if count > 1 else 0.0


def mean_pairwise_exact_match_diversity(sentences: tp.List[str]) -> float:
    '''
    Считает разнообразие коротких ответов как долю несовпадающих пар (exact match)
    Parameters:
        sentences: Список строк
    Return:
        Разнообразие ответов по exact match
    '''
    count = len(sentences)
    mismatches = []
    for i in range(count):
        for j in range(i + 1, count):
            mismatches.append(0 if sentences[i] == sentences[j] else 1)
    return sum(mismatches) / len(mismatches) if mismatches else 0.0


def compute_difference(orig_value: float,
                       aug_value: float) -> tp.Tuple[float, float]:
    '''
    Считает абсолютную и относительную разницу разнообразия между исходными и аугментированными данными
    Parameters:
        orig_value: Значение разнообразия на исходных данных
        aug_value: Значение разнообразия на аугментированных данных
    Return:
        Пара из абсолютной и относительной разницы
    '''
    if np.isnan(aug_value):
        abs_diff, rel_diff = np.nan, np.nan
    else:
        abs_diff = abs(aug_value - orig_value)
        rel_diff = compute_relative_difference(orig_value, abs_diff)
    return abs_diff, rel_diff


def compute_relative_difference(orig_value: float,
                                abs_diff: float) -> float:
    '''
    Считает относительную разницу с учетом нулевого исходного значения
    Parameters:
        orig_value: Значение разнообразия на исходных данных
        abs_diff: Абсолютная разница разнообразия
    Return:
        Относительная разница разнообразия
    '''
    if orig_value == 0:
        rel_diff = 0.0 if abs_diff == 0 else np.inf
    else:
        rel_diff = abs_diff / orig_value
    return rel_diff


def get_augmented(questions: tp.List[str],
                  augmentations_list: tp.List[str],
                  rng: tp.Any,
                  p_augmented: float=0.1) -> tp.List[str]:
    '''
    Аугментирует список запросов выбранными символьными аугментациями
    Parameters:
        questions: Список аугментируемых запросов
        augmentations_list: Список символьных аугментаций: 'use_deletion', 'use_swap', 'use_insert', 'use_bf',
            'use_special_symbol', 'use_translit_symbol'
        rng: Датчик случайных чисел
        p_augmented: Вероятность аугментации символа
    Return:
        Список аугментированных запросов
    '''
    augmented_questions = list(questions)
    aug_params = {par: True for par in augmentations_list}
    augmented_questions = list(batch_augmentations(batch=augmented_questions, rng=rng, p=p_augmented, **aug_params))
    return augmented_questions


def get_augmented_queries(queries: tp.List[str],
                          augmentations_list: tp.Optional[tp.List[str]]=None,
                          seed: int=42,
                          p_augmented: float=0.1) -> tp.List[str]:
    '''
    Аугментирует список запросов символьными аугментациями с фиксированным seed
    Parameters:
        queries: Список запросов
        augmentations_list: Список символьных аугментаций: 'use_deletion', 'use_swap', 'use_insert', 'use_bf',
            'use_special_symbol', 'use_translit_symbol'
        seed: Начальное значение датчика случайных чисел
        p_augmented: Вероятность аугментации символа
    Return:
        Список аугментированных запросов
    '''
    effective_augmentations = augmentations_list if augmentations_list is not None else ['use_special_symbol']
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    augmented_queries = get_augmented(queries,
                                      effective_augmentations,
                                      rng,
                                      p_augmented)
    return augmented_queries


def get_augmented_data(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                       data_type: str='test',
                       augmentations_list: tp.Optional[tp.List[str]]=None,
                       seed: int=42,
                       p_augmented: float=0.1,
                       n_columns: int=10,
                       exec_env: tp.Literal['default', 'sberds']='default') -> pd.DataFrame:
    '''
    Проводит аугментации запросов для проведения реинференса
    Parameters:
        sampler: Объект класса Sampler (exec_env='default') или словарь с данными (exec_env='sberds')
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        augmentations_list: Список символьных аугментаций: 'use_deletion', 'use_swap', 'use_insert', 'use_bf',
            'use_special_symbol', 'use_translit_symbol'
        seed: Начальное значение датчика случайных чисел
        p_augmented: Вероятность аугментации символа
        n_columns: Количество колонок с аугментированными запросами
        exec_env: Среда выполнения ('default' или 'sberds')
    Return:
        DataFrame с аугментированными запросами в колонках aug_valtest_answer_reprod_agent_<idx>
    '''
    effective_augmentations = augmentations_list if augmentations_list is not None else ['use_special_symbol']
    data = get_sampler_data(sampler, data_type, exec_env)
    augmented_queries = get_augmented_queries(data['input_query'].to_list(),
                                              effective_augmentations,
                                              seed,
                                              p_augmented)
    result_df = pd.DataFrame({'aug_valtest_answer_reprod_agent_' + str(idx): list(augmented_queries)
                              for idx in range(n_columns)})
    return result_df


def plot_diversity_metrics(orig_diversity: tp.Dict[str, float],
                           aug_diversity: tp.Optional[tp.Dict[str, float]]=None,
                           title: tp.Optional[str]=None) -> tp.Any:
    '''
    Строит сравнительную гистограмму лексического и семантического разнообразия на исходных и аугментированных данных
    Parameters:
        orig_diversity: Словарь с разнообразием на исходных данных
        aug_diversity: Словарь с разнообразием на аугментированных данных
        title: Опциональная строка заглавия графика
    Return:
        График гистограммы разнообразия
    '''
    metric_labels = {'lexical': 'Лексическое',
                     'semantic': 'Семантическое'}
    plot_data = build_plot_data(orig_diversity, aug_diversity, metric_labels)
    fig = plt.figure(figsize=(7, 5))
    sns.barplot(data=plot_data,
                x='Метрика',
                y='Разнообразие',
                hue='Тип данных',
                palette='muted')
    plt.ylabel('Разнообразие', fontsize=10)
    plt.title(title or 'Стабильность воспроизведения ответов агента', fontsize=12, fontweight='bold')
    return fig


def build_plot_data(orig_diversity: tp.Dict[str, float],
                    aug_diversity: tp.Optional[tp.Dict[str, float]],
                    metric_labels: tp.Dict[str, str]) -> pd.DataFrame:
    '''
    Формирует длинный DataFrame для построения сравнительной гистограммы разнообразия
    Parameters:
        orig_diversity: Словарь с разнообразием на исходных данных
        aug_diversity: Словарь с разнообразием на аугментированных данных
        metric_labels: Соответствие ключей метрик и их названий для графика
    Return:
        DataFrame с колонками 'Метрика', 'Тип данных' и 'Разнообразие'
    '''
    rows = [{'Метрика': metric_labels[key], 'Тип данных': 'Исходные', 'Разнообразие': value}
            for key, value in orig_diversity.items()]
    if aug_diversity is not None:
        rows.extend([{'Метрика': metric_labels[key], 'Тип данных': 'Аугментированные', 'Разнообразие': value}
                     for key, value in aug_diversity.items()])
    return pd.DataFrame(rows)
