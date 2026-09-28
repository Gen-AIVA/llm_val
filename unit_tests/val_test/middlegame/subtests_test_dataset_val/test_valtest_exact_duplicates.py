import typing as tp
from unittest.mock import MagicMock

import pytest
import pandas as pd

from llm_val.val_test.middlegame.subtests_test_dataset_val.valtest_exact_duplicates import (
    collect_duplicate_indices,
    compute_duplicates_stats,
    report_valtest_exact_duplicates,
    valtest_exact_duplicates,
)


@pytest.fixture
def default_data_input_query() -> pd.DataFrame:
    '''Создает тестовый датасет с дубликатами в input_query.'''
    return pd.DataFrame({
        'input_query': ['hello', 'world', 'hello', 'foo', 'bar', 'world'],
        'session_id': range(6)
    })


@pytest.fixture
def default_data_input_query_no_duplicates() -> pd.DataFrame:
    '''Создает тестовый датасет без дубликатов в input_query.'''
    return pd.DataFrame({
        'input_query': ['apple', 'banana', 'cherry', 'date'],
        'session_id': range(4)
    })


@pytest.fixture
def default_data_dialogue() -> pd.DataFrame:
    '''Создает тестовый датасет с диалогами.'''
    common_dialogue = [('1', 'hello', ''), ('2', 'world', '')]
    return pd.DataFrame({
        'dialogue': [
            common_dialogue,
            common_dialogue,
            [('3', 'foo', ''), ('4', 'bar', '')]
        ],
        'session_id': range(3)
    })


@pytest.fixture
def empty_data() -> pd.DataFrame:
    '''Создает пустой датасет.'''
    return pd.DataFrame({'input_query': pd.Series(dtype=object),
                         'session_id': pd.Series(dtype=object)})


@pytest.fixture
def mock_sampler_default(default_data_input_query: pd.DataFrame) -> MagicMock:
    '''Создает мок Sampler для режима default.'''
    sampler = MagicMock()
    sampler.test = {'X': default_data_input_query}
    return sampler


@pytest.fixture
def mock_sampler_sberds(default_data_input_query: pd.DataFrame) -> tp.Dict[str, tp.Dict[str, pd.DataFrame]]:
    '''Создает мок Sampler для режима sberds.'''
    return {'test': {'X': default_data_input_query}}


@pytest.fixture
def mock_sampler_dialogue(default_data_dialogue: pd.DataFrame) -> MagicMock:
    '''Создает мок Sampler с датасетом диалогов.'''
    sampler = MagicMock()
    sampler.test = {'X': default_data_dialogue}
    return sampler


@pytest.fixture
def mock_sampler_empty(empty_data: pd.DataFrame) -> MagicMock:
    '''Создает мок Sampler с пустым датасетом.'''
    sampler = MagicMock()
    sampler.test = {'X': empty_data}
    return sampler


def test_valtest_format_default(mock_sampler_default: MagicMock) -> None:
    '''Проверить формат выхода основной функции в режиме default.'''
    test_result = valtest_exact_duplicates(sampler=mock_sampler_default, exec_env='default')

    assert isinstance(test_result, dict)
    assert set(test_result.keys()) == {'report', 'precomputed'}

    report = test_result['report']
    precomputed = test_result['precomputed']
    assert isinstance(report, dict)
    assert isinstance(precomputed, dict)

    assert 'semaphore' in report
    assert 'result_plots' in report
    assert 'result_dataframes' in report
    assert 'duplicates_ratio' in precomputed
    assert 'duplicates' in precomputed
    assert 'duplicate_indices' in precomputed


def test_valtest_format_sberds(mock_sampler_sberds: tp.Dict[str, tp.Dict[str, pd.DataFrame]]) -> None:
    '''Проверить работу с exec_env=sberds.'''
    test_result = valtest_exact_duplicates(sampler=mock_sampler_sberds, exec_env='sberds')

    precomputed = test_result['precomputed']
    assert isinstance(precomputed['duplicates_ratio'], float)
    assert precomputed['duplicates_ratio'] > 0.0
    assert precomputed['duplicates'] == [(0, 2), (1, 5)]
    assert precomputed['duplicate_indices'] == {0, 1, 2, 5}


def test_duplicate_detection_input_query(mock_sampler_default: MagicMock) -> None:
    '''Проверить логику подсчета дубликатов для input_query.'''
    test_result = valtest_exact_duplicates(sampler=mock_sampler_default, exec_env='default')
    precomputed = test_result['precomputed']

    assert precomputed['duplicates_ratio'] == pytest.approx(4 / 6)
    assert precomputed['duplicates'] == [(0, 2), (1, 5)]
    assert precomputed['duplicate_indices'] == {0, 1, 2, 5}

    report = test_result['report']
    comment = report['result_dataframes'][0].iloc[0]['Комментарий']
    assert 'Номера строк с примерами-дубликатами' in comment
    assert '(0, 2)' in comment
    assert '(1, 5)' in comment


def test_duplicate_detection_dialogue(mock_sampler_dialogue: MagicMock) -> None:
    '''Проверить логику подсчета дубликатов для колонки dialogue.'''
    test_result = valtest_exact_duplicates(sampler=mock_sampler_dialogue, exec_env='default')
    precomputed = test_result['precomputed']

    assert precomputed['duplicates_ratio'] == pytest.approx(2 / 3)
    assert precomputed['duplicates'] == [(0, 1)]
    assert precomputed['duplicate_indices'] == {0, 1}


def test_empty_dataset(mock_sampler_empty: MagicMock) -> None:
    '''Проверить обработку пустого датасета.'''
    test_result = valtest_exact_duplicates(sampler=mock_sampler_empty, exec_env='default')
    precomputed = test_result['precomputed']

    assert precomputed['duplicates_ratio'] == 0.0
    assert precomputed['duplicates'] == []
    assert precomputed['duplicate_indices'] == set()


def test_no_duplicates_dataset(default_data_input_query_no_duplicates: pd.DataFrame) -> None:
    '''Проверить датасет без дубликатов.'''
    sampler = MagicMock()
    sampler.test = {'X': default_data_input_query_no_duplicates}

    test_result = valtest_exact_duplicates(sampler=sampler, exec_env='default')
    precomputed = test_result['precomputed']

    assert precomputed['duplicates_ratio'] == 0.0
    assert precomputed['duplicates'] == []
    assert precomputed['duplicate_indices'] == set()
    assert test_result['report']['semaphore'] == 'green'


def test_precomputed_usage(mock_sampler_default: MagicMock) -> None:
    '''Проверить использование предвычисленных значений из data_duplicates_dict.'''
    data_duplicates_dict = {
        'duplicates_ratio': 0.0,
        'duplicates': []
    }

    test_result = valtest_exact_duplicates(
        sampler=mock_sampler_default,
        data_duplicates_dict=data_duplicates_dict,
        exec_env='default'
    )
    precomputed = test_result['precomputed']

    assert precomputed['duplicates_ratio'] == data_duplicates_dict['duplicates_ratio']
    assert precomputed['duplicates'] == data_duplicates_dict['duplicates']
    assert precomputed['duplicate_indices'] == set()
    assert test_result['report']['semaphore'] == 'green'


def test_precomputed_contract_with_duplicate_groups() -> None:
    '''Проверить новый контракт precomputed для полных дубликатов.'''
    data_duplicates_dict = {
        'duplicates_ratio': 0.2,
        'duplicates': [(2, 4)]
    }

    test_result = valtest_exact_duplicates(
        sampler={},
        data_duplicates_dict=data_duplicates_dict
    )

    assert test_result['precomputed']['duplicates'] == [(2, 4)]
    assert test_result['precomputed']['duplicate_indices'] == {2, 4}


def test_report_semaphore_green() -> None:
    '''Проверить выставление зеленого светофора.'''
    report = report_valtest_exact_duplicates(
        duplicates_ratio=0.05,
        duplicate_indices=[(0, 2)],
        threshold=0.1
    )

    assert report['semaphore'] == 'green'
    assert report['result_dataframes'][0].iloc[0]['Результат'] == 'Пройдена'


def test_report_semaphore_yellow() -> None:
    '''Проверить выставление желтого светофора.'''
    report = report_valtest_exact_duplicates(
        duplicates_ratio=0.15,
        duplicate_indices=[(0, 2), (1, 5)],
        threshold=0.1
    )

    assert report['semaphore'] == 'yellow'
    assert report['result_dataframes'][0].iloc[0]['Результат'] == 'Не пройдена'


def test_report_semaphore_info() -> None:
    '''Проверить работу флага is_info.'''
    report = report_valtest_exact_duplicates(
        duplicates_ratio=0.15,
        duplicate_indices=[(0, 2)],
        is_info=True
    )

    assert report['semaphore'] == 'gray'
    assert report['result_dataframes'][0].iloc[0]['Результат'] == 'Пройдена'


def test_report_comment_format() -> None:
    '''Проверить форматирование комментария с индексами дубликатов.'''
    duplicate_indices = [(0, 2), (1, 5, 7)]

    report = report_valtest_exact_duplicates(
        duplicates_ratio=0.1,
        duplicate_indices=duplicate_indices
    )
    comment = report['result_dataframes'][0].iloc[0]['Комментарий']

    assert comment == 'Номера строк с примерами-дубликатами: (0, 2), (1, 5, 7)'


def test_compute_duplicates_stats_with_strings() -> None:
    '''Проверить вспомогательную функцию на серии строк.'''
    series = pd.Series(['a', 'b', 'a', 'c', 'b', 'd'])

    stats = compute_duplicates_stats(series)

    assert stats['duplicates_ratio'] == pytest.approx(4 / 6)
    assert stats['duplicates'] == [(0, 2), (1, 4)]


def test_compute_duplicates_stats_with_lists() -> None:
    '''Проверить вспомогательную функцию на серии списков.'''
    series = pd.Series([
        ['hello', 'world'],
        ['hello', 'world'],
        ['foo', 'bar']
    ])

    stats = compute_duplicates_stats(series)

    assert stats['duplicates_ratio'] == pytest.approx(2 / 3)
    assert stats['duplicates'] == [(0, 1)]


def test_compute_duplicates_stats_normalizes_spaces() -> None:
    '''Проверить нормализацию пробелов при поиске дубликатов.'''
    series = pd.Series(['hello   world', 'hello world', 'other'])

    stats = compute_duplicates_stats(series)

    assert stats['duplicates_ratio'] == pytest.approx(2 / 3)
    assert stats['duplicates'] == [(0, 1)]


def test_compute_duplicates_stats_empty() -> None:
    '''Проверить вспомогательную функцию на пустой серии.'''
    series = pd.Series(dtype=object)

    stats = compute_duplicates_stats(series)

    assert stats['duplicates_ratio'] == 0.0
    assert stats['duplicates'] == []


def test_collect_duplicate_indices() -> None:
    '''Проверить сбор плоского множества индексов из групп дубликатов.'''
    duplicate_indices = collect_duplicate_indices(duplicates=[(0, 2, 3), (1, 4)])

    assert duplicate_indices == {0, 1, 2, 3, 4}
