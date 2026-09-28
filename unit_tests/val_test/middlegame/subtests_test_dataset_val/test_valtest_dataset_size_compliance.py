import pytest
import pandas as pd
from unittest.mock import MagicMock
import typing as tp
from llm_val.sampler.Sampler import Sampler
from llm_val.val_test.middlegame.subtests_test_dataset_val.valtest_dataset_size_compliance import (
    valtest_dataset_size_compliance,
)


@pytest.fixture
def mock_sampler_default() -> MagicMock:
    """Фикстура: mock-объект Sampler для exec_env='default' с 300 строками."""
    sampler = MagicMock()
    sampler.test = {'X': pd.DataFrame({'col': range(300)})}
    return sampler


@pytest.fixture
def mock_sampler_sberds() -> tp.Dict[str, tp.Any]:
    """Фикстура: mock-словарь для exec_env='sberds' с 300 строками."""
    return {'test': {'X': pd.DataFrame({'col': range(300)})}}


@pytest.fixture
def mock_sampler_default_200() -> MagicMock:
    """Фикстура: mock-объект Sampler с 200 строками (меньше порога)."""
    sampler = MagicMock()
    sampler.test = {'X': pd.DataFrame({'col': range(200)})}
    return sampler


@pytest.fixture
def mock_sampler_sberds_200() -> tp.Dict[str, tp.Any]:
    """Фикстура: mock-словарь для exec_env='sberds' с 200 строками."""
    return {'test': {'X': pd.DataFrame({'col': range(200)})}}


def test_output_format(mock_sampler_default: Sampler) -> None:
    """Проверить, что возвращается словарь с ключами 'report' и 'precomputed'."""
    result = valtest_dataset_size_compliance(sampler=mock_sampler_default)

    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}

    report = result['report']
    assert isinstance(report, dict)
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert report['result_plots'] == []

    precomputed = result['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'cleaned_count', 'indices_to_remove'}


def test_default_sufficient(mock_sampler_default: Sampler) -> None:
    """exec_env='default': 300 строк >= 250 → passed."""
    result = valtest_dataset_size_compliance(sampler=mock_sampler_default, agreed_min_count=250)
    assert result['report']['semaphore'] == 'green'
    assert result['precomputed']['cleaned_count'] == 300
    assert result['precomputed']['indices_to_remove'] == set()
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}


def test_default_insufficient(mock_sampler_default_200: Sampler) -> None:
    """exec_env='default': 200 строк < 250 → fail."""
    result = valtest_dataset_size_compliance(sampler=mock_sampler_default_200, agreed_min_count=250)
    assert result['report']['semaphore'] == 'yellow'
    assert result['precomputed']['cleaned_count'] == 200
    assert result['precomputed']['indices_to_remove'] == set()
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}


def test_sberds_sufficient(mock_sampler_sberds: Sampler) -> None:
    """exec_env='sberds': 300 строк >= 250 → passed."""
    result = valtest_dataset_size_compliance(sampler=mock_sampler_sberds, agreed_min_count=250, exec_env='sberds')
    assert result['report']['semaphore'] == 'green'
    assert result['precomputed']['cleaned_count'] == 300
    assert result['precomputed']['indices_to_remove'] == set()
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}


def test_sberds_insufficient(mock_sampler_sberds_200: MagicMock) -> None:
    """exec_env='sberds': 200 строк < 250 → fail."""
    result = valtest_dataset_size_compliance(sampler=mock_sampler_sberds_200, agreed_min_count=250, exec_env='sberds')
    assert result['report']['semaphore'] == 'yellow'
    assert result['precomputed']['cleaned_count'] == 200
    assert result['precomputed']['indices_to_remove'] == set()
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}


def test_with_indices_to_remove(mock_sampler_default: Sampler) -> None:
    """Передача готового indices_to_remove: cleaned_count = total - len(indices_to_remove)."""
    indices_to_remove = {1, 2, 3}
    result = valtest_dataset_size_compliance(
        sampler=mock_sampler_default,
        agreed_min_count=250,
        indices_to_remove=indices_to_remove,
    )
    assert result['precomputed']['cleaned_count'] == 297
    assert result['report']['semaphore'] == 'green'
    assert result['precomputed']['indices_to_remove'] == indices_to_remove
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}


def test_precomputed_with_large_indices_to_remove() -> None:
    """Передача большого набора индексов на удаление."""
    df = pd.DataFrame({'col': range(500)})
    sampler = MagicMock()
    sampler.test = {'X': df}

    result = valtest_dataset_size_compliance(
        sampler=sampler,
        agreed_min_count=250,
        indices_to_remove=set(range(100)),
    )
    assert result['precomputed']['cleaned_count'] == 400
    assert result['report']['semaphore'] == 'green'
    assert result['precomputed']['indices_to_remove'] == set(range(100))
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}


def test_duplicate_outlier_removal() -> None:
    """Индексы дубликатов и выбросов корректно исключаются."""
    df = pd.DataFrame({'col': range(300)})
    sampler = MagicMock()
    sampler.test = {'X': df}
    duplicate_indices = {1, 2}
    outlier_indices = {3}

    result = valtest_dataset_size_compliance(
        sampler=sampler,
        agreed_min_count=250,
        duplicate_indices=duplicate_indices,
        outlier_indices=outlier_indices,
    )
    assert result['precomputed']['cleaned_count'] == 297
    assert result['report']['semaphore'] == 'green'
    assert result['precomputed']['indices_to_remove'] == {1, 2, 3}

    result2 = valtest_dataset_size_compliance(
        sampler=sampler,
        agreed_min_count=298,
        duplicate_indices=duplicate_indices,
        outlier_indices=outlier_indices,
    )
    assert result2['precomputed']['cleaned_count'] == 297
    assert result2['report']['semaphore'] == 'yellow'
    assert result2['precomputed']['indices_to_remove'] == {1, 2, 3}
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}
    assert isinstance(result2, dict)
    assert set(result2.keys()) == {'report', 'precomputed'}


def test_duplicate_outlier_overlap() -> None:
    """Пересекающиеся наборы: {1,2,3} и {3,4,5} → 5 уникальных индексов."""
    df = pd.DataFrame({'col': range(300)})
    sampler = MagicMock()
    sampler.test = {'X': df}
    duplicate_indices = {1, 2, 3}
    outlier_indices = {3, 4, 5}

    result = valtest_dataset_size_compliance(
        sampler=sampler,
        agreed_min_count=250,
        duplicate_indices=duplicate_indices,
        outlier_indices=outlier_indices,
    )
    assert result['precomputed']['cleaned_count'] == 295
    assert result['report']['semaphore'] == 'green'
    assert result['precomputed']['indices_to_remove'] == {1, 2, 3, 4, 5}
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}


def test_semantic_duplicates_removal() -> None:
    """Семантические дубликаты добавляются к исключаемым индексам."""
    df = pd.DataFrame({'col': range(300)})
    sampler = MagicMock()
    sampler.test = {'X': df}
    semantic_duplicate_indices = {10, 11, 12}

    result = valtest_dataset_size_compliance(
        sampler=sampler,
        agreed_min_count=250,
        semantic_duplicate_indices=semantic_duplicate_indices,
    )
    assert result['precomputed']['cleaned_count'] == 297
    assert result['report']['semaphore'] == 'green'
    assert result['precomputed']['indices_to_remove'] == {10, 11, 12}
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}


def test_all_removal_types_combined() -> None:
    """Комбинация дубликатов, выбросов и семантических дубликатов."""
    df = pd.DataFrame({'col': range(300)})
    sampler = MagicMock()
    sampler.test = {'X': df}
    duplicate_indices = {1, 2}
    outlier_indices = {2, 3}
    semantic_duplicate_indices = {3, 4}

    result = valtest_dataset_size_compliance(
        sampler=sampler,
        agreed_min_count=250,
        duplicate_indices=duplicate_indices,
        outlier_indices=outlier_indices,
        semantic_duplicate_indices=semantic_duplicate_indices,
    )
    assert result['precomputed']['cleaned_count'] == 296
    assert result['report']['semaphore'] == 'green'
    assert result['precomputed']['indices_to_remove'] == {1, 2, 3, 4}
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}


def test_indices_to_remove_overrides_all_others(mock_sampler_default: Sampler) -> None:
    """Если передан indices_to_remove, duplicate/outlier/semantic игнорируются."""
    result = valtest_dataset_size_compliance(
        sampler=mock_sampler_default,
        agreed_min_count=250,
        indices_to_remove={1, 2},
        duplicate_indices={3, 4, 5},
        outlier_indices={6, 7},
        semantic_duplicate_indices={8, 9},
    )
    assert result['precomputed']['cleaned_count'] == 298
    assert result['report']['semaphore'] == 'green'
    assert result['precomputed']['indices_to_remove'] == {1, 2}
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}


def test_is_info(mock_sampler_default: Sampler) -> None:
    """is_info=True → светофор всегда серый, precomputed вычислен."""
    result = valtest_dataset_size_compliance(sampler=mock_sampler_default, agreed_min_count=250, is_info=True)
    assert result['report']['semaphore'] == 'gray'
    assert result['precomputed']['cleaned_count'] == 300
    assert result['precomputed']['indices_to_remove'] == set()
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}


def test_exact_boundary() -> None:
    """Ровно 250 строк с порогом 250 → должно проходить."""
    df = pd.DataFrame({'col': range(250)})
    sampler = MagicMock()
    sampler.test = {'X': df}

    result = valtest_dataset_size_compliance(sampler=sampler, agreed_min_count=250)
    assert result['report']['semaphore'] == 'green'
    assert result['precomputed']['cleaned_count'] == 250
    assert result['precomputed']['indices_to_remove'] == set()
    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 'precomputed'}