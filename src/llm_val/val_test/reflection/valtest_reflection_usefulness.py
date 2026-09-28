import warnings
import typing as tp

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.get_data import get_data
from llm_val.utils.retries import call_with_retries
from llm_val.utils.openai_utils import get_openai_embeddings


def report_valtest_reflection_usefulness(avg_distances: tp.List[tp.Tuple[str, float]]) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'a
    Parameters:
        avg_distances: Список кортежей, где каждый кортеж содержит название ноды рефлексии и среднее косинусное расстояние
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_dataframes'] = [pd.DataFrame(avg_distances, columns=['Название ноды рефлексии', 'Среднее косинусное расстояние'])]
    res_dict['result_plots'] = []
    return res_dict


def valtest_reflection_usefulness(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                  exec_env: str='default',
                                  reflection_nodes: tp.Optional[tp.List[str]]=None,
                                  embedding_client: tp.Optional[tp.Any]=None,
                                  embedding_model_name: str='EmbeddingsGigaR',
                                  data_type: str='test',
                                  timeout: float=60.0,
                                  retry_delay: float=0.1,
                                  max_attempts: int=3,
                                  avg_distances: tp.Optional[tp.List[tp.Tuple[str, float]]]=None,
                                  **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на оценку полезности рефлексии путем сравнения косинусного расстояния между ответами до и после рефлексии.
    Parameters:
        sampler: Объект класса Sampler (exec_env='default') или словарь с данными (exec_env='sberds')
        exec_env: Среда выполнения ("default" или "sberds")
        reflection_nodes: Опциональный список названий нод рефлексии (span_name), для которых нужно оценить полезность.
                          Должен быть передан, если avg_distances не передан.
        embedding_client: OpenAI-совместимый клиент с методом embeddings.create. Должен быть передан, если avg_distances не передан.
        embedding_model_name: Название эмбеддинг-модели, передаваемое в embeddings.create
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
        avg_distances: Опциональный список готовых средних косинусных расстояний.
               Формат: [(название_ноды, среднее_расстояние), ...]
               Если передан, расчеты не выполняются.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    traces_key = 'traces'
    if reflection_nodes is None and avg_distances is None:
        raise ValueError('Должно быть передано либо reflection_nodes, либо avg_distances.')
    if avg_distances is not None and reflection_nodes is not None:
        warnings.warn('Переданы и reflection_nodes, и avg_distances. Будут использованы avg_distances.',
                      UserWarning, stacklevel=2)

    if avg_distances is not None:
        report = report_valtest_reflection_usefulness(avg_distances)
        precomputed = {'all_cosine_distances_by_node': avg_distances}
    else:
        if embedding_client is None:
            raise ValueError('При передаче reflection_nodes необходимо также передать embedding_client.')
        traces_df = get_data(sampler, data_type, traces_key, exec_env)
        if not isinstance(traces_df, pd.DataFrame) or traces_df.empty:
            raise ValueError('Не удалось получить корректный DataFrame с трейсами.')
        avg_distances_result = process_all_nodes(reflection_nodes,
                                                 traces_df,
                                                 embedding_client,
                                                 embedding_model_name,
                                                 timeout,
                                                 retry_delay,
                                                 max_attempts)

        report = report_valtest_reflection_usefulness(avg_distances_result)
        precomputed = {'all_cosine_distances_by_node': avg_distances_result}
    return {'report': report, 
            'precomputed': precomputed}


def process_all_nodes(reflection_nodes: tp.List[str],
                      traces_df: pd.DataFrame,
                      embedding_client: tp.Any,
                      embedding_model_name: str,
                      timeout: float=60.0,
                      retry_delay: float=0.1,
                      max_attempts: int=3) -> tp.List[tp.Tuple[str, float]]:
    '''
    Обрабатывает все ноды рефлексии и вычисляет средние косинусные расстояния.
    Parameters:
        reflection_nodes: Список названий нод рефлексии для анализа
        traces_df: DataFrame с трейсами
        embedding_client: OpenAI-совместимый клиент с методом embeddings.create
        embedding_model_name: Название эмбеддинг-модели
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
    Return:
        Список кортежей, где каждый кортеж содержит название ноды и среднее косинусное расстояние
    '''
    avg_distances_result = []
    for node_name in reflection_nodes:
        avg_dist = process_single_node(node_name,
                                       traces_df,
                                       embedding_client,
                                       embedding_model_name,
                                       timeout,
                                       retry_delay,
                                       max_attempts)
        if avg_dist is not None:
            avg_distances_result.append((node_name, avg_dist))
    return avg_distances_result


def process_single_node(node_name: str,
                        traces_df: pd.DataFrame,
                        embedding_client: tp.Any,
                        embedding_model_name: str,
                        timeout: float=60.0,
                        retry_delay: float=0.1,
                        max_attempts: int=3) -> tp.Optional[float]:
    '''
    Обрабатывает одну ноду рефлексии и вычисляет среднее косинусное расстояние.

    Логика отбора трейсов для анализа:
        1. Находим все срабатывания ноды рефлексии по span_name.
        2. Для каждой сессии берём первое и последнее срабатывание, фиксируем span_id.
        3. Находим дочерние LLM-спаны (parent_span_id == span_id и aef_kind == 'llm').
        4. Из input_text первого LLM-спана извлекаем ответ агента до рефлексии.
        5. Из output_text последнего LLM-спана извлекаем ответ агента после рефлексии.

    Parameters:
        node_name: Название ноды рефлексии
        traces_df: DataFrame с трейсами
        embedding_client: OpenAI-совместимый клиент с методом embeddings.create
        embedding_model_name: Название эмбеддинг-модели
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
    Return:
        Среднее косинусное расстояние или None, если нода пропущена
    '''
    node_df = traces_df[traces_df['span_name'] == node_name].copy()

    if node_df.empty:
        warnings.warn(f'Для ноды "{node_name}" не найдено ни одного срабатывания. Оценка пропущена.',
                      UserWarning,
                      stacklevel=3)
        avg_dist = None
    else:
        before_reflection_answers, after_reflection_answers = group_node_traces(node_df, traces_df, node_name)
        if not after_reflection_answers:
            warnings.warn(f'Для ноды "{node_name}" не найдено случаев изменения ответа после рефлексии. Оценка пропущена.',
                          UserWarning,
                          stacklevel=3)
            avg_dist = None
        else:
            avg_dist = calculate_distances(before_reflection_answers,
                                           after_reflection_answers,
                                           embedding_client,
                                           embedding_model_name,
                                           timeout,
                                           retry_delay,
                                           max_attempts)

    return avg_dist


def group_node_traces(node_df: pd.DataFrame,
                      traces_df: pd.DataFrame,
                      node_name: str) -> tp.Tuple[tp.List[str], tp.List[str]]:
    '''
    Группирует срабатывания ноды рефлексии по сессиям и извлекает ответы агента
    из дочерних LLM-спанов первого и последнего срабатывания.

    Parameters:
        node_df: DataFrame со срабатываниями конкретной ноды рефлексии
        traces_df: Полный DataFrame с трейсами (для поиска дочерних LLM-спанов)
        node_name: Название ноды рефлексии
    Return:
        Кортеж из двух списков: ответы агента до рефлексии и после рефлексии
    '''
    before_reflection_answers = []
    after_reflection_answers = []

    for session_id, group in node_df.groupby('session_id'):
        if len(group) <= 1:
            continue

        group = group.sort_values('start_time_ns')
        first_row, last_row = group.iloc[0], group.iloc[-1]

        first_children = get_llm_children(traces_df, first_row['span_id'])
        last_children = get_llm_children(traces_df, last_row['span_id'])

        if first_children.empty or last_children.empty:
            raise ValueError(
                f'Не найдено дочерних LLM-спанов для ноды "{node_name}" '
                f'(сессия {session_id}). Проверьте структуру трейсов.'
            )

        if len(first_children) != 1 or len(last_children) != 1:
            raise ValueError(
                f'Ожидался ровно один дочерний LLM-спан для ноды "{node_name}" '
                f'(сессия {session_id}), найдено: first={len(first_children)}, last={len(last_children)}.'
            )

        first_llm = first_children.iloc[0]
        last_llm = last_children.iloc[0]

        text_first = extract_last_assistant_message(first_llm.get('input_text'))
        text_last = extract_last_assistant_message(last_llm.get('output_text'))

        if not text_first or not text_last:
            raise ValueError(
                f'Не удалось извлечь сообщение агента для ноды "{node_name}" '
                f'(сессия {session_id}). Проверьте формат input_text / output_text'
                f'(ожидается список списков сообщений в OpenAI-формате или одно сообщение в OpenAI-формате).'
            )

        before_reflection_answers.append(text_first)
        after_reflection_answers.append(text_last)

    return before_reflection_answers, after_reflection_answers


def get_llm_children(traces_df: pd.DataFrame, parent_span_id: str) -> pd.DataFrame:
    '''
    Находит дочерние LLM-спаны для заданного parent_span_id.
    Parameters:
        traces_df: DataFrame с трейсами
        parent_span_id: Идентификатор родительского спана
    Return:
        DataFrame с дочерними LLM-спанами, отсортированный по start_time_ns
    '''
    return traces_df[(traces_df['parent_span_id'] == parent_span_id) &
                     (traces_df['aef_kind'] == 'llm')].sort_values('start_time_ns')


def extract_last_assistant_message(
        messages: tp.Union[tp.List[tp.List[tp.Dict[str, str]]], tp.Dict[str, str]]
) -> tp.Optional[str]:
    '''
    Извлекает последнее сообщение агента из входных данных.
    Parameters:
        messages: Словарь-сообщение в OpenAI-формате (output_text) или список списков словарей-сообщений в OpenAI-формате (input_text)
    Return:
        Строка с содержимым последнего сообщения агента или None
    '''
    if isinstance(messages, dict):
        message = messages.get('content', '')
    else:
        message = extract_from_message_list(messages)
    return message


def extract_from_message_list(messages: tp.List[tp.List[tp.Dict[str, str]]]) -> tp.Optional[str]:
    '''
    Извлекает последнее сообщение агента из списка попыток.
    Parameters:
        messages: Список попыток, каждая — список сообщений
    Return:
        Строка с содержимым последнего сообщения агента или None
    '''
    message = None
    last_attempt = messages[-1]
    for msg in reversed(last_attempt):
        if msg.get('role') == 'assistant':
            message = msg.get('content', '')
            break
    return message


def calculate_distances(before_reflection_answers: tp.List[str],
                        after_reflection_answers: tp.List[str],
                        embedding_client: tp.Any,
                        embedding_model_name: str,
                        timeout: float=60.0,
                        retry_delay: float=0.1,
                        max_attempts: int=3) -> float:
    '''
    Вычисляет среднее косинусное расстояние между запросами до и после рефлексии.
    Parameters:
        before_reflection_answers: Список запросов до рефлексии
        after_reflection_answers: Список запросов после рефлексии
        embedding_client: OpenAI-совместимый клиент с методом embeddings.create
        embedding_model_name: Название эмбеддинг-модели
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
    Return:
        Среднее косинусное расстояние между запросами
    '''
    emb_first = call_with_retries(get_openai_embeddings,
                                  timeout,
                                  retry_delay,
                                  max_attempts,
                                  (embedding_client, before_reflection_answers,  embedding_model_name))
    emb_last = get_openai_embeddings(embedding_client, after_reflection_answers, embedding_model_name)

    distances = [cosine(u, v) for u, v in zip(emb_first, emb_last)]
    avg_distance = float(np.mean(distances))
    return avg_distance