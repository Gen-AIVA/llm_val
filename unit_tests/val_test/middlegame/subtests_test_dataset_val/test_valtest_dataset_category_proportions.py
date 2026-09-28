import typing as tp
import pytest
import pandas as pd

from unittest.mock import MagicMock
from llm_val.val_test.middlegame.subtests_test_dataset_val.valtest_dataset_category_proportions import (
    valtest_dataset_category_proportions,
)


@pytest.fixture
def perfect_match_data() -> pd.DataFrame:
    """Создает датасет с ровно 50% cat_a и 50% cat_b."""
    return pd.DataFrame({'scenario': ['cat_a'] * 5 + ['cat_b'] * 5})


@pytest.fixture
def biased_data() -> pd.DataFrame:
    """Создает датасет с 80% cat_a и 20% cat_b."""
    return pd.DataFrame({'scenario': ['cat_a'] * 8 + ['cat_b'] * 2})


@pytest.fixture
def extra_category_data() -> pd.DataFrame:
    """Создает датасет с 60% cat_a, 30% cat_b и 10% cat_c."""
    return pd.DataFrame({'scenario': ['cat_a'] * 6 + ['cat_b'] * 3 + ['cat_c'] * 1})


@pytest.fixture
def missing_category_data() -> pd.DataFrame:
    """Создает датасет только с cat_a."""
    return pd.DataFrame({'scenario': ['cat_a'] * 10})


@pytest.fixture
def empty_data() -> pd.DataFrame:
    """Создает пустой датасет."""
    return pd.DataFrame({'scenario': pd.Series(dtype=object)})


@pytest.fixture
def mock_sampler_default(perfect_match_data: pd.DataFrame) -> MagicMock:
    """Мок Sampler для режима default."""
    sampler = MagicMock()
    sampler.test = {'X': perfect_match_data}
    return sampler


@pytest.fixture
def mock_sampler_biased(biased_data: pd.DataFrame) -> MagicMock:
    """Мок Sampler с перекосом."""
    sampler = MagicMock()
    sampler.test = {'X': biased_data}
    return sampler


@pytest.fixture
def mock_sampler_extra(extra_category_data: pd.DataFrame) -> MagicMock:
    """Мок Sampler с лишней категорией."""
    sampler = MagicMock()
    sampler.test = {'X': extra_category_data}
    return sampler


@pytest.fixture
def mock_sampler_missing(missing_category_data: pd.DataFrame) -> MagicMock:
    """Мок Sampler с отсутствующей категорией."""
    sampler = MagicMock()
    sampler.test = {'X': missing_category_data}
    return sampler


@pytest.fixture
def mock_sampler_sberds(perfect_match_data: pd.DataFrame) -> tp.Dict[str, tp.Dict[str, pd.DataFrame]]:
    """Мок Sampler для режима sberds."""
    return {'test': {'X': perfect_match_data}}


def test_valtest_format_default(mock_sampler_default: MagicMock) -> None:
    """Проверка формата выхода основной функции в режиме default."""
    apriori_dict = {'cat_a': 0.5, 'cat_b': 0.5}
    test_res = valtest_dataset_category_proportions(
        sampler=mock_sampler_default,
        apriori_dict=apriori_dict,
        exec_env='default',
    )

    report = test_res['report']
    assert isinstance(report, dict)

    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_format_sberds(mock_sampler_sberds: tp.Dict[str, tp.Dict[str, pd.DataFrame]]) -> None:
    """Проверка формата выхода в режиме sberds."""
    apriori_dict = {'cat_a': 0.5, 'cat_b': 0.5}
    test_res = valtest_dataset_category_proportions(
        sampler=mock_sampler_sberds,
        apriori_dict=apriori_dict,
        exec_env='sberds',
    )

    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_perfect_match(mock_sampler_default: MagicMock) -> None:
    """Проверка: ровное распределение должно давать зелёный."""
    apriori_dict = {'cat_a': 0.5, 'cat_b': 0.5}
    test_res = valtest_dataset_category_proportions(
        sampler=mock_sampler_default,
        apriori_dict=apriori_dict,
        exec_env='default',
    )

    report = test_res['report']
    assert report['semaphore'] == 'green'
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_biased_but_within_threshold(mock_sampler_biased: MagicMock) -> None:
    """Проверка: 80/20 при пороге 0.25 должно быть зелёным."""
    apriori_dict = {'cat_a': 0.8, 'cat_b': 0.2}
    test_res = valtest_dataset_category_proportions(
        sampler=mock_sampler_biased,
        apriori_dict=apriori_dict,
        exec_env='default',
    )

    report = test_res['report']
    assert report['semaphore'] == 'green'
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_biased_above_threshold(mock_sampler_biased: MagicMock) -> None:
    """Проверка: 80/20 при apriori 50/50 должно давать жёлтый (теперь вместо красного)."""
    apriori_dict = {'cat_a': 0.5, 'cat_b': 0.5}
    test_res = valtest_dataset_category_proportions(
        sampler=mock_sampler_biased,
        apriori_dict=apriori_dict,
        exec_env='default',
    )

    report = test_res['report']
    assert report['semaphore'] == 'red'
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_extra_category(mock_sampler_extra: MagicMock) -> None:
    """Проверка: лишняя категория в данных -> жёлтый (теперь вместо красного)."""
    apriori_dict = {'cat_a': 0.6, 'cat_b': 0.4}
    test_res = valtest_dataset_category_proportions(
        sampler=mock_sampler_extra,
        apriori_dict=apriori_dict,
        exec_env='default',
    )

    report = test_res['report']
    assert report['semaphore'] == 'red'
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_missing_category(mock_sampler_missing: MagicMock) -> None:
    """Проверка: отсутствующая категория в данных -> жёлтый (теперь вместо красного)."""
    apriori_dict = {'cat_a': 0.5, 'cat_b': 0.5}
    test_res = valtest_dataset_category_proportions(
        sampler=mock_sampler_missing,
        apriori_dict=apriori_dict,
        exec_env='default',
    )

    report = test_res['report']
    assert report['semaphore'] == 'red'
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_empty_apriori(mock_sampler_default: MagicMock) -> None:
    """Проверка: пустой apriori_dict -> серый."""
    test_res = valtest_dataset_category_proportions(
        sampler=mock_sampler_default,
        apriori_dict={},
        exec_env='default',
    )

    report = test_res['report']
    assert report['semaphore'] == 'gray'
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_precomputed_keys(mock_sampler_default: MagicMock) -> None:
    """Проверка структуры precomputed."""
    apriori_dict = {'cat_a': 0.5, 'cat_b': 0.5}
    test_res = valtest_dataset_category_proportions(
        sampler=mock_sampler_default,
        apriori_dict=apriori_dict,
        exec_env='default',
    )

    precomputed = test_res['precomputed']
    expected_keys = {
        'psi_value',
        'actual_props',
        'extra_categories',
        'missing_categories',
        'max_deviation',
    }
    assert set(precomputed.keys()) == expected_keys
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_precomputed_values(mock_sampler_default: MagicMock) -> None:
    """Проверка значений в precomputed."""
    apriori_dict = {'cat_a': 0.5, 'cat_b': 0.5}
    test_res = valtest_dataset_category_proportions(
        sampler=mock_sampler_default,
        apriori_dict=apriori_dict,
        exec_env='default',
    )

    precomputed = test_res['precomputed']
    assert precomputed['extra_categories'] == []
    assert precomputed['missing_categories'] == []
    assert isinstance(precomputed['psi_value'], float)
    assert isinstance(precomputed['actual_props'], dict)
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_precomputed_pass_through(mock_sampler_default: MagicMock) -> None:
    """Проверка: переданные precomputed не пересчитываются."""
    precomputed_data = {
        'psi_value': 0.05,
        'actual_props': {'cat_a': 0.5, 'cat_b': 0.5},
        'extra_categories': [],
        'missing_categories': [],
    }
    test_res = valtest_dataset_category_proportions(
        sampler=mock_sampler_default,
        apriori_dict={'cat_a': 0.5, 'cat_b': 0.5},
        category_stats=precomputed_data,
        exec_env='default',
    )

    precomputed = test_res['precomputed']
    assert precomputed['psi_value'] == 0.05
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_report_has_dataframes(mock_sampler_default: MagicMock) -> None:
    """Проверка наличия result_dataframes в отчёте."""
    apriori_dict = {'cat_a': 0.5, 'cat_b': 0.5}
    test_res = valtest_dataset_category_proportions(
        sampler=mock_sampler_default,
        apriori_dict=apriori_dict,
        exec_env='default',
    )

    report = test_res['report']
    assert 'result_dataframes' in report
    assert isinstance(report['result_dataframes'], list)
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_report_has_plots(mock_sampler_default: MagicMock) -> None:
    """Проверка наличия result_plots в отчёте."""
    apriori_dict = {'cat_a': 0.5, 'cat_b': 0.5}
    test_res = valtest_dataset_category_proportions(
        sampler=mock_sampler_default,
        apriori_dict=apriori_dict,
        exec_env='default',
    )

    report = test_res['report']
    assert 'result_plots' in report
    assert isinstance(report['result_plots'], list)
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}
