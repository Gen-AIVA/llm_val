import typing as tp
import json
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock

from llm_val.sampler.Sampler import Sampler
from openai import OpenAI

from llm_val.val_test.traces.valtest_incorrect_tool_calls import (
    check_call,
    extract_args,
    build_p3_prompts,
    get_data,
    is_optional,
    is_type_compatible,
    make_prompt_from_input_query,
    valtest_incorrect_tool_calls,
)

_PARQUET_PATH = 'data/data_X_with_traces.parquet'
_DATA_PATH = 'data/data_traces_with_tools.pkl'

SIGNATURES_PARQUET = {
    'get_document_storage_start_date': {
        'document_group_number': tp.Optional[str],
        'form_code': str,
    },
    'get_document_upload_date': {
        'form_code': str,
        'document_group_number': str,
    },
    'map_doc_name_to_code': {
        'doc_name': str,
    },
    'call_rag': {
        'text': str,
    },
}


SIGNATURES_SIMPLE = {
    'angle_photo_classifier': {'img_path': str, 'x_trace_id': str},
    'check_disadvantage': {'img_path': str, 'x_trace_id': str},
    'check_photo_quality': {'img_path': str, 'x_trace_id': str},
    'check_tags': {'comment': str, 'x_trace_id': str},
    'detect_atm': {'img_path': str, 'x_trace_id': str},
}


SIG_OPTIONAL = {
    'search_web': {'query': str, 'top_k': tp.Optional[int]},
}


@pytest.fixture
def sampler() -> Sampler:
    data_path = Path(__file__).parents[2] / 'data' / 'data_traces_with_tools.pkl'
    data = pd.read_pickle(data_path)
    # traces в pickle не содержит session_id, добавляем для совместимости
    traces = data['traces'].copy()
    if 'session_id' not in traces.columns:
        traces['session_id'] = range(len(traces))
    data['traces'] = traces
    return Sampler(test=data)


@pytest.fixture
def parquet_sampler() -> Sampler:
    """Создаёт Sampler на основе данных из data_X_with_traces.parquet."""
    parquet_path = Path(__file__).parents[2] / 'data' / 'data_X_with_traces.parquet'
    df = pd.read_parquet(parquet_path)
    tool_spans = df[df['aef_kind'] == 'tool'].copy()
    tool_spans = tool_spans.rename(columns={'name': 'span_name'})
    traces = tool_spans[['span_name', 'aef_kind', 'input_text', 'session_id']].reset_index(drop=True)
    X = df[['session_id', 'input_query']].drop_duplicates().reset_index(drop=True)
    sampler_obj = Sampler(test={'X': X, 'y': pd.Series(dtype='object'), 'traces': traces})
    return sampler_obj


# ── extract_args (pure, no pickle) ────────────────────────────

_SIG_FLAT = {'flat_tool': {'query': str, 'top_k': int}}


class TestExtractArgs:
    """Проверка extract_args — парсинг и нормализация аргументов.

    extract_args возвращает (args, status, path). При полном совпадении
    сигнатуры вложенный словарь распаковывается (статус 'unwrapped').
    """

    def _check_unwrapped(self, input_text, expected_args,
                         tool_name='flat_tool', expected_path='$'):
        """Проверяет, что extract_args возвращает корректные args (распакованные)."""
        args, status, path = extract_args(input_text, tool_name,
                                            {'flat_tool': {'query': str, 'top_k': int}})
        assert args == expected_args
        return args, status, path

    def _check_raw(self, input_text, expected_args,
                    tool_name='flat_tool', sig=None):
        """Проверяет, что extract_args возвращает args как есть (без распаковки)."""
        if sig is None:
            sig = _SIG_FLAT
        args, status, path = extract_args(input_text, tool_name, sig)
        assert args == expected_args
        return args, status, path

    def test_flat(self):
        """Плоский словарь с полным совпадением сигнатуры — распаковка."""
        self._check_unwrapped(
            '{"query": "hello", "top_k": 5}',
            {'query': 'hello', 'top_k': 5},
        )

    def test_tool_args(self):
        """tool_args с полным покрытием — распаковка."""
        self._check_unwrapped(
            '{"tool_args": {"query": "hello", "top_k": 5}}',
            {'query': 'hello', 'top_k': 5},
            expected_path='$.tool_args',
        )

    def test_tool_input(self):
        """tool_input — распаковка при полном покрытии."""
        self._check_unwrapped(
            '{"tool_input": {"query": "hello", "top_k": 5}}',
            {'query': 'hello', "top_k": 5},
            expected_path='$.tool_input',
        )

    def test_arguments(self):
        """arguments — распаковка при полном покрытии."""
        self._check_unwrapped(
            '{"arguments": {"query": "hello", "top_k": 5}}',
            {'query': 'hello', "top_k": 5},
            expected_path='$.arguments',
        )

    def test_input(self):
        """input — распаковка при полном покрытии."""
        self._check_unwrapped(
            '{"input": {"query": "hello", "top_k": 5}}',
            {'query': 'hello', "top_k": 5},
            expected_path='$.input',
        )

    def test_kwargs(self):
        """kwargs — распаковка при полном покрытии."""
        self._check_unwrapped(
            '{"kwargs": {"query": "hello", "top_k": 5}}',
            {'query': 'hello', "top_k": 5},
            expected_path='$.kwargs',
        )

    def test_non_wrapper_passthrough(self):
        """Неподдерживаемые ключи не распаковываются."""
        self._check_raw(
            '{"nested": {"inner": "val"}}',
            {'nested': {'inner': 'val'}},
        )

    def test_multi_key_no_unwrap(self):
        """Несколько ключей на одном уровне — без сигнатуры."""
        self._check_raw(
            '{"a": 1, "b": 2}',
            {'a': 1, 'b': 2},
        )

    def test_invalid_json(self):
        """Не JSON — возвращаем None."""
        args, _, _ = extract_args('not json', 'flat_tool', _SIG_FLAT)
        assert args is None

    def test_non_dict(self):
        """Список — не словарь, возвращаем None."""
        args, _, _ = extract_args('["list"]', 'flat_tool', _SIG_FLAT)
        assert args is None

    def test_wrapper_with_non_dict_inner(self):
        """wrapper с не-словарем внутри — не распаковывается."""
        self._check_raw('{"tool_args": "not_a_dict"}', {'tool_args': 'not_a_dict'})

    def test_empty_string(self):
        """Пустая строка — не парсится."""
        args, _, _ = extract_args('', 'flat_tool', _SIG_FLAT)
        assert args is None


# ── is_optional ──────────────────────────────────────────────


class TestIsOptional:
    def test_optional(self):
        assert is_optional(tp.Optional[int]) is True

    def test_plain(self):
        assert is_optional(str) is False

    def test_union_non_optional(self):
        assert is_optional(tp.Union[int, str]) is False

    def test_none_type(self):
        assert is_optional(type(None)) is False


# ── is_type_compatible ───────────────────────────────────────


class TestIsTypeCompatible:
    def test_exact(self):
        assert is_type_compatible(5, int) is True

    def test_wrong(self):
        assert is_type_compatible('five', int) is False

    def test_none_optional(self):
        assert is_type_compatible(None, tp.Optional[int]) is True

    def test_value_optional(self):
        assert is_type_compatible(5, tp.Optional[int]) is True

    def test_object_type_always_true(self):
        assert is_type_compatible('anything', object) is True

    def test_none_for_non_optional_is_false(self):
        assert is_type_compatible(None, str) is False

    def test_bool_not_int(self):
        """В текущей реализации bool явно не считается int/float."""
        assert is_type_compatible(True, int) is False


# ── check_call ────────────────────────────────────────────────

SIG_CHECK = {'search_web': {'query': str, 'top_k': int}, 'read_image': {'img_path': str}}


class TestCheckCall:
    def test_correct(self):
        assert check_call({'query': 'hi', 'top_k': 5}, 'search_web', SIG_CHECK) is True

    def test_missing_required(self):
        assert check_call({'top_k': 5}, 'search_web', SIG_CHECK) is False

    def test_wrong_type(self):
        assert check_call({'query': 'hi', 'top_k': 'five'}, 'search_web', SIG_CHECK) is False

    def test_extra_param(self):
        assert check_call({'query': 'hi', 'top_k': 5, 'extra': 'x'}, 'search_web', SIG_CHECK) is False

    def test_unknown_tool(self):
        assert check_call({'foo': 'bar'}, 'nope', SIG_CHECK) is False

    def test_optional_missing_ok(self):
        assert check_call({'query': 'hi'}, 'search_web', SIG_OPTIONAL) is True

    def test_empty_parsed_args_with_required_params(self):
        assert check_call({}, 'search_web', SIG_CHECK) is False

    def test_correct_optional_with_value(self):
        assert check_call({'query': 'hi', 'top_k': 3}, 'search_web', SIG_OPTIONAL) is True

    def test_optional_wrong_type(self):
        assert check_call({'query': 'hi', 'top_k': 'three'}, 'search_web', SIG_OPTIONAL) is False


# ── get_data ───────────────────────────────────────────────────


class TestGetData:
    def test_default(self, sampler):
        df = get_data(sampler, 'test', 'traces')
        assert isinstance(df, pd.DataFrame)

    def test_sberds(self):
        d = {'test': {'traces': pd.DataFrame({'a': [1]})}}
        assert get_data(d, 'test', 'traces', exec_env='sberds')['a'].iloc[0] == 1

    def test_unknown_raises(self, sampler):
        with pytest.raises(ValueError, match='unknown'):
            get_data(sampler, 'test', 'traces', exec_env='unknown')


# ── valtest_incorrect_tool_calls ──────────────────────────


def _make_empty_sampler():
    empty = pd.DataFrame(columns=['span_name', 'aef_kind', 'input_text'])
    return Sampler(test={'X': pd.DataFrame(), 'y': pd.Series(), 'traces': empty})


class TestValtest:
    def test_signatures_none_raises(self, sampler):
        with pytest.raises(ValueError, match='cannot be None'):
            valtest_incorrect_tool_calls(sampler, signatures=None)

    def test_output_structure(self, sampler):
        result = valtest_incorrect_tool_calls(sampler, signatures=SIGNATURES_SIMPLE)
        assert set(result.keys()) == {'report', 'precomputed'}
        assert set(result['precomputed'].keys()) == {'per_tool_stats', 'total_error_rate', 'tool_call_results'}
        assert set(result['report'].keys()) == {'semaphore', 'result_plots', 'result_dataframes'}

    def test_precomputed_through(self):
        """Precomputed stats are returned unchanged, no traces read."""
        pre = pd.DataFrame(
            {
                'tool_name': ['search_web'],
                'total_calls': [10],
                'error_count': [3],
                'error_rate': [0.3],
            }
        )
        empty = pd.DataFrame(
            columns=['trace_id', 'span_name', 'aef_kind', 'input_text', 'start_time_ns', 'end_time_ns']
        )
        s = Sampler(test={'X': pd.DataFrame(), 'y': pd.Series(), 'traces': empty})
        r = valtest_incorrect_tool_calls(s, signatures=SIGNATURES_SIMPLE, per_tool_stats=pre)
        assert r['precomputed']['total_error_rate'] == 0.3
        pd.testing.assert_frame_equal(r['precomputed']['per_tool_stats'], pre)

    def test_no_tool_spans(self):
        empty_traces = pd.DataFrame(
            {
                'span_name': ['llm_call'],
                'aef_kind': ['llm'],
                'input_text': ['{}'],
            }
        )
        s = Sampler(test={'X': pd.DataFrame(), 'y': pd.Series(), 'traces': empty_traces})
        r = valtest_incorrect_tool_calls(s, signatures=SIGNATURES_SIMPLE)
        assert r['precomputed']['total_error_rate'] == 0.0
        # Без tool-спанов светофор становится серым (тест неприменим)
        assert r['report']['semaphore'] == 'gray'

    def test_no_aef_kind_column_fallback_to_all_spans(self):
        """Если колонка aef_kind отсутствует — все спаны считаются tool."""
        traces_no_aef = pd.DataFrame(
            {
                'span_name': ['search_web', 'search_web', 'llm_call'],
                'input_text': [
                    '{"query": "test"}',
                    '{"query": "test", "top_k": 3}',
                    '{}',
                ],
                'session_id': ['s1', 's1', 's2'],
            }
        )
        s = Sampler(test={
            'X': pd.DataFrame({'session_id': ['s1', 's2'], 'input_query': ['hi', 'bye']}),
            'y': pd.Series(),
            'traces': traces_no_aef,
        })
        r = valtest_incorrect_tool_calls(s, signatures=SIG_OPTIONAL)
        assert r['precomputed']['total_error_rate'] > 0
        # Светофор не должен быть gray — тест применим
        assert r['report']['semaphore'] != 'gray'

    def test_semaphore_red(self):
        pre = pd.DataFrame(
            {
                'tool_name': ['angle_photo_classifier'],
                'total_calls': [4],
                'error_count': [4],
                'error_rate': [1.0],
            }
        )
        s = _make_empty_sampler()
        r = valtest_incorrect_tool_calls(s, signatures=SIGNATURES_SIMPLE, per_tool_stats=pre)
        assert r['report']['semaphore'] == 'red'

    def test_semaphore_yellow(self):
        pre = pd.DataFrame(
            {
                'tool_name': ['angle_photo_classifier', 'detect_atm'],
                'total_calls': [10, 10],
                'error_count': [0, 6],
                'error_rate': [0.0, 0.6],
            }
        )
        s = _make_empty_sampler()
        r = valtest_incorrect_tool_calls(s, signatures=SIGNATURES_SIMPLE, per_tool_stats=pre)
        assert r['report']['semaphore'] == 'yellow'

    def test_semaphore_green(self):
        pre = pd.DataFrame(
            {
                'tool_name': ['angle_photo_classifier'],
                'total_calls': [10],
                'error_count': [2],
                'error_rate': [0.2],
            }
        )
        s = _make_empty_sampler()
        r = valtest_incorrect_tool_calls(s, signatures=SIGNATURES_SIMPLE, per_tool_stats=pre)
        assert r['report']['semaphore'] == 'green'

    def test_per_tool_stats_columns(self, sampler):
        r = valtest_incorrect_tool_calls(sampler, signatures=SIGNATURES_SIMPLE)
        cols = set(r['precomputed']['per_tool_stats'].columns)
        assert {'tool_name', 'total_calls', 'error_count', 'error_rate'}.issubset(cols)

    def test_known_tools_covered(self, sampler):
        result = valtest_incorrect_tool_calls(sampler, signatures=SIGNATURES_SIMPLE)
        stats = result['precomputed']['per_tool_stats']
        unknown_tools = set(stats['tool_name']) - set(SIGNATURES_SIMPLE.keys())
        assert unknown_tools == set(), (
            f'Tools in data not covered by SIGNATURES_SIMPLE: {unknown_tools}. '
            f'Add them to the signature dict or they will all be counted as errors.'
        )

    def test_per_tool_stats_valid_ranges(self, sampler):
        result = valtest_incorrect_tool_calls(sampler, signatures=SIGNATURES_SIMPLE)
        stats = result['precomputed']['per_tool_stats']
        assert (stats['error_rate'].between(0.0, 1.0)).all()
        assert (stats['error_count'] <= stats['total_calls']).all()
        assert (stats['error_count'] >= 0).all()
        assert (stats['total_calls'] > 0).all()

    def test_total_rate_consistent(self, sampler):
        result = valtest_incorrect_tool_calls(sampler, signatures=SIGNATURES_SIMPLE)
        stats = result['precomputed']['per_tool_stats']
        expected = stats['error_count'].sum() / stats['total_calls'].sum()
        assert abs(result['precomputed']['total_error_rate'] - expected) < 1e-9

    def test_semaphore_matches_thresholds(self, sampler):
        result = valtest_incorrect_tool_calls(sampler, signatures=SIGNATURES_SIMPLE)
        total_rate = result['precomputed']['total_error_rate']
        stats = result['precomputed']['per_tool_stats']
        semaphore = result['report']['semaphore']
        print(stats)

        if total_rate > 0.30:
            assert semaphore == 'red'
        elif len(stats) > 0 and (stats['error_rate'] > 0.50).any():
            assert semaphore == 'yellow'
        else:
            assert semaphore == 'green'


# ── Parquet-тесты ───────────────────────────────────────────


class TestCheckCallWithParquet:
    """Проверка check_call с реальными аргументами из Parquet и известными сигнатурами."""

    def test_correct_call(self, parquet_sampler):
        """Вызов с правильными аргументами должен вернуть True."""
        # Для map_doc_name_to_code аргументы содержат doc_name
        traces = parquet_sampler.test['traces']
        tool_spans = traces[traces['span_name'] == 'map_doc_name_to_code']
        if len(tool_spans) == 0:
            pytest.skip("No map_doc_name_to_code spans in test data")
        first_row = tool_spans.iloc[0]
        parsed_args, _, _ = extract_args(first_row['input_text'], 'map_doc_name_to_code', SIGNATURES_PARQUET)
        result = check_call(parsed_args, 'map_doc_name_to_code', SIGNATURES_PARQUET)
        assert result is True

    def test_incorrect_call_missing_param(self, parquet_sampler):
        """Отсутствие required параметра -> False."""
        args = {'extra': 'value'}
        result = check_call(args, 'get_document_upload_date', SIGNATURES_PARQUET)
        assert result is False

    def test_incorrect_call_wrong_type(self, parquet_sampler):
        """Неправильный тип параметра -> False."""
        args = {'form_code': 123}  # ожидается строка
        result = check_call(args, 'get_document_upload_date', SIGNATURES_PARQUET)
        assert result is False

    def test_unknown_tool(self):
        """Неизвестный инструмент -> False."""
        args = {'form_code': '0401060'}
        result = check_call(args, 'unknown_tool', SIGNATURES_PARQUET)
        assert result is False

    def test_optional_missing_ok(self):
        """Отсутствие Optional параметра допустимо."""
        args = {'form_code': '0401060'}  # document_group_number опционален
        result = check_call(args, 'get_document_storage_start_date', SIGNATURES_PARQUET)
        assert result is True

    def test_optional_wrong_type(self):
        """Неправильный тип для Optional параметра -> False."""
        args = {'form_code': '0401060', 'document_group_number': 123}
        result = check_call(args, 'get_document_storage_start_date', SIGNATURES_PARQUET)
        assert result is False


class TestMakePromptFromInputQuery:
    """Проверка make_prompt_from_input_query с реальными данными."""

    def test_basic_prompt_format(self, parquet_sampler):
        """Проверяет формат промпта."""
        traces = parquet_sampler.test['traces']
        first_row = traces.iloc[0]
        tool_name = first_row['span_name']
        parsed_args, _, _ = extract_args(first_row['input_text'], first_row['span_name'], SIGNATURES_PARQUET)
        tool_args_json = json.dumps(parsed_args, ensure_ascii=False)
        session_id = first_row['session_id']
        X = parquet_sampler.test['X']
        x_row = X[X['session_id'] == session_id]
        if len(x_row) == 0:
            pytest.skip("No input_query for session_id")
        input_query = str(x_row['input_query'].iloc[0]).strip()

        prompt = make_prompt_from_input_query(
            input_query=input_query,
            tool_name=tool_name,
            tool_args_json=tool_args_json,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert tool_name in prompt
        assert input_query[:50] in prompt or input_query in prompt

    def test_prompt_with_description(self, parquet_sampler):
        """Промпт с tool_description должен содержать блок описания."""
        prompt = make_prompt_from_input_query(
            input_query="test query",
            tool_name="test_tool",
            tool_args_json='{"arg": "value"}',
            tool_description="Этот инструмент ищет данные",
        )
        assert 'Назначение инструмента' in prompt
        assert 'test_tool' in prompt
        assert 'Этот инструмент ищет данные' in prompt

    def test_prompt_without_description(self):
        """Промпт без tool_description не должен содержать блок описания."""
        prompt = make_prompt_from_input_query(
            input_query="test query",
            tool_name="test_tool",
            tool_args_json='{"arg": "value"}',
            tool_description=None,
        )
        assert 'Назначение инструмента' not in prompt
        assert 'Диалог / пользовательский контекст' in prompt
        assert 'Фактические аргументы вызова' in prompt


class TestBuildP3Prompts:
    """Проверка build_p3_prompts с Parquet-данными."""

    def test_returns_dataframe_with_prompt(self, parquet_sampler):
        """Функция возвращает DataFrame с колонкой prompt."""
        traces = parquet_sampler.test['traces']
        tool_spans = traces[traces['span_name'] == 'get_document_upload_date']
        prompts = build_p3_prompts(
            sampler=parquet_sampler,
            tool_spans=tool_spans,
            data_type='test',
            exec_env='default',
            tool_name='get_document_upload_date',
        )
        assert isinstance(prompts, pd.DataFrame)
        assert 'prompt' in prompts.columns
        assert len(prompts) > 0

    def test_empty_after_filtering(self, parquet_sampler):
        """Если после фильтрации нет данных, возвращается пустой DataFrame."""
        empty_spans = pd.DataFrame(columns=['span_name', 'aef_kind', 'input_text', 'session_id'])
        prompts = build_p3_prompts(
            sampler=parquet_sampler,
            tool_spans=empty_spans,
            data_type='test',
            exec_env='default',
            tool_name='any_tool',
        )
        assert isinstance(prompts, pd.DataFrame)
        assert prompts.empty

    def test_input_query_merge(self, parquet_sampler):
        """Проверяет, что input_query успешно присоединён."""
        traces = parquet_sampler.test['traces']
        tool_spans = traces.iloc[:2]
        prompts = build_p3_prompts(
            sampler=parquet_sampler,
            tool_spans=tool_spans,
            data_type='test',
            exec_env='default',
            tool_name='get_document_upload_date',
        )
        assert isinstance(prompts, pd.DataFrame)
        assert 'prompt' in prompts.columns
        assert prompts['prompt'].notna().all()


class TestValtestWithParquet:
    """Тесты valtest_incorrect_tool_calls с Parquet-данными."""

    def test_output_structure(self, parquet_sampler):
        """Проверяет, что основная функция возвращает правильную структуру."""
        mock_client = Mock(spec=OpenAI)

        result = valtest_incorrect_tool_calls(
            sampler=parquet_sampler,
            signatures=SIGNATURES_PARQUET,
            is_info=False,
            client=mock_client,
            model_name='gpt-4o',
        )
        # Проверяем формат
        assert isinstance(result, dict)
        assert set(result.keys()) == {'report', 'precomputed'}

        report = result['report']
        precomputed = result['precomputed']
        assert isinstance(report, dict)
        assert isinstance(precomputed, dict)

        # Проверяем структуру report
        assert 'semaphore' in report
        assert 'result_plots' in report
        assert 'result_dataframes' in report

        # Проверяем структуру precomputed
        assert 'per_tool_stats' in precomputed
        assert 'total_error_rate' in precomputed

        # Проверяем, что пер-tool статистика содержит требуемые колонки
        stats = precomputed['per_tool_stats']
        required_cols = {'tool_name', 'total_calls', 'error_count', 'error_rate'}
        missing = required_cols - set(stats.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_zero_tool_spans(self):
        """Если нет tool-спанов, error_rate = 0, semaphore green."""
        traces = pd.DataFrame(columns=['span_name', 'aef_kind', 'input_text', 'session_id'])
        X = pd.DataFrame(columns=['session_id', 'input_query'])
        empty_sampler = Sampler(test={'X': X, 'y': pd.Series(), 'traces': traces})
        mock_client = Mock(spec=OpenAI)
        result = valtest_incorrect_tool_calls(
            sampler=empty_sampler,
            signatures=SIGNATURES_PARQUET,
            client=mock_client,
            model_name='gpt-4o',
        )
        assert result['precomputed']['total_error_rate'] == 0.0
        # Без tool-спанов светофор становится серым (тест неприменим)
        assert result['report']['semaphore'] == 'gray'

    def test_precomputed_through(self, parquet_sampler):
        """Precomputed-статистика передаётся без пересчёта."""
        pre = pd.DataFrame(
            {
                'tool_name': ['get_document_upload_date', 'call_rag'],
                'total_calls': [10, 5],
                'error_count': [3, 2],
                'error_rate': [0.3, 0.4],
            }
        )
        mock_client = Mock(spec=OpenAI)
        result = valtest_incorrect_tool_calls(
            sampler=parquet_sampler,
            signatures=SIGNATURES_PARQUET,
            client=mock_client,
            model_name='gpt-4o',
            per_tool_stats=pre,
        )
        # total_error_rate = sum(error_count) / sum(total_calls) = 5/15 = 0.3333333333333333
        assert abs(result['precomputed']['total_error_rate'] - 0.3333333333333333) < 1e-9
        pd.testing.assert_frame_equal(result['precomputed']['per_tool_stats'], pre)

    def test_error_rate_calculation(self, parquet_sampler):
        """Проверяет, что error_rate соответствует P2 ошибкам, а P3 не влияет."""
        mock_client = Mock(spec=OpenAI)
        # Вернём ошибки по P3 для всех промптов, но P2 ошибок остаётся теми же.
        traces = parquet_sampler.test['traces']
        result = valtest_incorrect_tool_calls(
            sampler=parquet_sampler,
            signatures=SIGNATURES_PARQUET,
            client=mock_client,
            model_name='gpt-4o',
        )
        # P2 ошибки только у get_document_upload_date (6 из 20), так что общая error_rate = 0.3
        assert result['precomputed']['total_error_rate'] == pytest.approx(0.3, abs=1e-9)
        # Светофор должен быть жёлтым из-за инструмента с error_rate > 0.50 (get_document_upload_date)
        assert result['report']['semaphore'] == 'yellow'

    def test_known_tools_covered(self, parquet_sampler):
        """Проверяет, что все инструменты в данных покрыты SIGNATURES_PARQUET."""
        mock_client = Mock(spec=OpenAI)
        result = valtest_incorrect_tool_calls(
            sampler=parquet_sampler,
            signatures=SIGNATURES_PARQUET,
            is_info=False,
            client=mock_client,
            model_name='gpt-4o',
        )
        stats = result['precomputed']['per_tool_stats']
        unknown_tools = set(stats['tool_name']) - set(SIGNATURES_PARQUET.keys())
        assert unknown_tools == set(), (
            f'Tools in data not covered by SIGNATURES_PARQUET: {unknown_tools}. '
            f'Add them to the signature dict.'
        )
