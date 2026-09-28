import random
import typing as tp
import uuid

import pandas as pd
import pytest
from unittest.mock import MagicMock

from llm_val.sampler.TracesSamplerWithReflection import TracesSamplerWithReflection
from llm_val.val_test.reflection.valtest_reflection_usefulness import valtest_reflection_usefulness


DATA_PATH = 'data/data_traces_reflection.parquet'


def _make_mock_openai_client() -> MagicMock:
    '''
    Создаёт мок OpenAI-совместимого клиента с методом embeddings.create.
    Return:
        Мок клиента, возвращающий предсказуемые эмбеддинги фиксированной размерности
    '''
    mock_client = MagicMock()

    def create_embeddings(model: str, input: tp.List[str]):
        response = MagicMock()
        response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])
                        for _ in input]
        return response

    mock_client.embeddings.create.side_effect = create_embeddings
    return mock_client


@pytest.fixture()
def create_sampler_with_precomputed(request):
    '''
    Фикстура создаёт семплер и запускает функцию с готовыми avg_distances.
    '''
    avg_distances = [('node_A', 0.01), ('node_B', 0.01)]
    data = pd.read_parquet(DATA_PATH)
    sampler = TracesSamplerWithReflection(test=data)
    res = valtest_reflection_usefulness(sampler=sampler,
                                        embedding_client=_make_mock_openai_client(),
                                        avg_distances=avg_distances)
    return res


@pytest.fixture()
def create_sampler(request):
    '''
    Фикстура создаёт семплер и запускает функцию со списком нод для расчёта косинусных расстояний.
    '''
    data = pd.read_parquet(DATA_PATH)
    sampler = TracesSamplerWithReflection(test=data)
    res = valtest_reflection_usefulness(sampler=sampler,
                                        reflection_nodes=['reflection_node_A',
                                                          'reflection_node_B'],
                                        embedding_client=_make_mock_openai_client(),
                                        data_type='test')
    return res


def _make_trace_row(session_id: str,
                    starttime: int,
                    span_id: str,
                    parent_span_id: str,
                    span_name: str,
                    aef_kind: str,
                    input_text: tp.List[tp.List[tp.Dict[str, str]]],
                    output_text: tp.Dict[str, str]) -> dict:
    '''
    Создаёт одну строку мок-трейса.
    Parameters:
        session_id: Идентификатор сессии
        starttime: Время начала в наносекундах
        span_id: Идентификатор спана
        parent_span_id: Идентификатор родительского спана
        span_name: Название ноды
        aef_kind: Тип ноды
        input_text: Список списков сообщений в формате OpenAI
        output_text: Словарь с сообщением агента в формате OpenAI
    Return:
        Словарь с данными одной строки трейса
    '''
    return {
        'trace_id': uuid.uuid4().hex,
        'span_id': span_id,
        'parent_span_id': parent_span_id,
        'span_name': span_name,
        'start_time_ns': starttime,
        'end_time_ns': starttime + random.randint(100000, 1000000),
        'session_id': session_id,
        'aef_kind': aef_kind,
        'input_text': input_text,
        'output_text': output_text,
        'input_query': f'Исходный запрос {session_id}',
        'output_answer': f'Финальный ответ {session_id}',
        'final_metric_assessor_1': 1,
        'final_metric_assessor_2': 1,
        'final_metric_assessor_3': 1,
        'final_metric_assessor_4': 1,
        'final_metric_assessor_5': 1,
        'estimation': 1
    }


def _make_reflection_span_with_child(session_id: str,
                                     node_starttime: int,
                                     node_name: str,
                                     node_span_id: str,
                                     node_aef_kind: str,
                                     child_span_id: str,
                                     child_input_text: tp.List[tp.List[tp.Dict[str, str]]],
                                     child_output_text: tp.Dict[str, str]) -> tp.List[dict]:
    '''
    Создаёт пару: срабатывание ноды рефлексии + дочерний LLM-спан.
    Parameters:
        session_id: Идентификатор сессии
        node_starttime: Время начала срабатывания ноды
        node_name: Название ноды рефлексии
        node_span_id: Идентификатор срабатывания ноды
        node_aef_kind: Тип срабатывания ноды
        child_span_id: Идентификатор дочернего LLM-спана
        child_input_text: input_text дочернего LLM-спана
        child_output_text: output_text дочернего LLM-спана
    Return:
        Список из двух строк: срабатывание ноды и дочерний LLM-спан
    '''
    return [
        _make_trace_row(
            session_id=session_id,
            starttime=node_starttime,
            span_id=node_span_id,
            parent_span_id='root',
            span_name=node_name,
            aef_kind=node_aef_kind,
            input_text=[[{'message': 'Запуск ноды'}]],
            output_text={'output_message': 'Результат ноды'},
        ),
        _make_trace_row(
            session_id=session_id,
            starttime=node_starttime + 100,
            span_id=child_span_id,
            parent_span_id=node_span_id,
            span_name='GigaChat',
            aef_kind='llm',
            input_text=child_input_text,
            output_text=child_output_text,
        ),
    ]


@pytest.fixture
def sampler_no_assistant_msg():
    '''
    Фикстура создаёт мок-трейс с ошибкой: в input_text дочернего LLM-спана
    первого срабатывания нет assistant-сообщения.
    '''
    data = []
    # Первое срабатывание: дочерний LLM-спан без assistant-сообщения в input_text
    data.extend(_make_reflection_span_with_child(session_id='sess_err_2',
                                                 node_starttime=1000,
                                                 node_name='reflection_node_B',
                                                 node_span_id='span_b1',
                                                 node_aef_kind='chain',
                                                 child_span_id='llm_b1',
                                                 child_input_text=[[{'role': 'user', 'content': 'Запрос без ответа ассистента'}]],
                                                 child_output_text={'role': 'assistant', 'content': 'Ответ 1'}))
    # Второе срабатывание: корректное
    data.extend(_make_reflection_span_with_child(session_id='sess_err_2',
                                                 node_starttime=2000,
                                                 node_name='reflection_node_B',
                                                 node_span_id='span_b2',
                                                 node_aef_kind='chain',
                                                 child_span_id='llm_b2',
                                                 child_input_text=[[{'role': 'user', 'content': 'Нормальный запрос'},
                                                                {'role': 'assistant', 'content': 'Нормальный ответ'}]],
                                                 child_output_text={'role': 'assistant', 'content': 'Ответ 2'}))
    df = pd.DataFrame(data)
    sampler = TracesSamplerWithReflection(test=df)
    return sampler, ['reflection_node_B']


@pytest.fixture
def sampler_single_call():
    '''
    Фикстура создаёт мок-трейс с единственным срабатыванием ноды в сессии.
    '''
    data = []
    data.extend(_make_reflection_span_with_child(session_id='sess_single',
                                                 node_starttime=1000,
                                                 node_name='reflection_node_A',
                                                 node_span_id='span_a1',
                                                 node_aef_kind='chain',
                                                 child_span_id='llm_a1',
                                                 child_input_text=[[{'role': 'user', 'content': 'Единственный запрос'},
                                                                {'role': 'assistant', 'content': 'Единственный ответ'}]],
                                                 child_output_text={'role': 'assistant', 'content': 'Ответ'}))
    df = pd.DataFrame(data)
    sampler = TracesSamplerWithReflection(test=df)
    return sampler, ['reflection_node_A']


@pytest.fixture
def sampler_empty_node():
    '''
    Фикстура создаёт мини-датасет, где для анализируемой ноды нет ни одного срабатывания.
    '''
    data = []
    data.extend(_make_reflection_span_with_child(session_id='sess_other',
                                                 node_starttime=1000,
                                                 node_name='other_node',
                                                 node_span_id='span_other',
                                                 node_aef_kind='llm',
                                                 child_span_id='llm_other',
                                                 child_input_text=[[{'role': 'user', 'content': 'Запрос'},
                                                                {'role': 'assistant', 'content': 'Ответ'}]],
                                                 child_output_text={'role': 'assistant', 'content': 'Ответ'}))
    df = pd.DataFrame(data)
    sampler = TracesSamplerWithReflection(test=df)
    return sampler, ['reflection_node_A']


def test_valtest_reflection_usefulness_with_precomputed(create_sampler_with_precomputed):
    '''
    Проверяет корректную работу функции при передаче готовых avg_distances.
    '''
    avg_distances = [('node_A', 0.01), ('node_B', 0.01)]
    result = create_sampler_with_precomputed
    assert 'report' in result
    assert 'precomputed' in result
    assert result['precomputed']['all_cosine_distances_by_node'] == avg_distances
    assert 'result_dataframes' in result['report']
    assert 'result_plots' in result['report']


def test_valtest_reflection_usefulness_correct_params():
    '''
    Проверяет валидацию параметров: предупреждение при передаче обоих параметров
    и ошибку при отсутствии обоих обязательных параметров.
    '''
    data = pd.read_parquet(DATA_PATH)
    sampler = TracesSamplerWithReflection(test=data)
    avg_distances = [('node_A', 0.01)]
    reflection_nodes = ['node_A']
    with pytest.warns(UserWarning, match='Переданы и reflection_nodes, и avg_distances'):
        result = valtest_reflection_usefulness(sampler=sampler,
                                               reflection_nodes=reflection_nodes,
                                               embedding_client=_make_mock_openai_client(),
                                               avg_distances=avg_distances)
    assert result['precomputed']['all_cosine_distances_by_node'] == avg_distances
    with pytest.raises(ValueError, match='Должно быть передано либо reflection_nodes, либо avg_distances'):
        valtest_reflection_usefulness(sampler=sampler,
                                      embedding_client=_make_mock_openai_client())


def test_valtest_reflection_usefulness(create_sampler):
    '''
    Проверяет корректную работу теста.
    '''
    result = create_sampler
    report = result['report']
    precomputed = result['precomputed']
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}
    assert isinstance(report, dict)
    assert isinstance(precomputed, dict)
    assert 'result_dataframes' in report
    assert 'result_plots' in report
    assert isinstance(report['result_dataframes'], list)


def test_reflection_usefulness_fails_on_missing_assistant_msg(sampler_no_assistant_msg):
    '''
    Проверяет, что функция падает с ValueError при отсутствии assistant-сообщения
    в input_text дочернего LLM-спана.
    '''
    sampler, reflection_nodes = sampler_no_assistant_msg
    with pytest.raises(ValueError, match='Не удалось извлечь сообщение агента'):
        valtest_reflection_usefulness(sampler=sampler,
                                      reflection_nodes=reflection_nodes,
                                      embedding_client=_make_mock_openai_client())


def test_reflection_usefulness_warns_on_single_call(sampler_single_call):
    '''
    Проверяет, что функция выдает Warning при сессии с единственным срабатыванием ноды.
    '''
    sampler, reflection_nodes = sampler_single_call
    with pytest.warns(UserWarning, match='не найдено случаев изменения ответа после рефлексии'):
        result = valtest_reflection_usefulness(sampler=sampler,
                                               reflection_nodes=reflection_nodes,
                                               embedding_client=_make_mock_openai_client())
    assert 'reflection_node_A' not in result['precomputed']['all_cosine_distances_by_node']


def test_reflection_usefulness_warns_on_empty_node(sampler_empty_node):
    '''
    Проверяет, что функция выдает Warning при отсутствии срабатываний ноды.
    '''
    sampler, reflection_nodes = sampler_empty_node
    with pytest.warns(UserWarning, match='не найдено ни одного срабатывания'):
        result = valtest_reflection_usefulness(sampler=sampler,
                                               reflection_nodes=reflection_nodes,
                                               embedding_client=_make_mock_openai_client())
    assert 'reflection_node_A' not in result['precomputed']['all_cosine_distances_by_node']