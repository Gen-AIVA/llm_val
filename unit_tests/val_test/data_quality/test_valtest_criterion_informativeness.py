import pytest
import numpy as np
import pandas as pd

from llm_val.sampler.Sampler import Sampler
from llm_val.val_test.data_quality.valtest_criterion_informativeness import (
    valtest_criterion_informativeness,
    compute_informativeness,
    default_measure,
)


@pytest.fixture
def sample_sampler():
    """
    Создаёт Sampler с данными.
    """
    df = pd.DataFrame({
        'session_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'assessor_id': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'],
        'crit1': [1, 2, 3, 2, 3, 4, 1, 2, 3],       # разбросанные оценки
        'crit2': [5, 5, 5, 5, 5, 5, 5, 5, 5],       # константа
        'crit3': [0, 10, 0, 1, 9, 1, 2, 8, 2],      # разброс
        'extra_col': ['x', 'x', 'x', 'y', 'y', 'y', 'z', 'z', 'z']
    })
    data = {'X': df, 'y': pd.DataFrame()}
    sampler = Sampler()
    setattr(sampler, 'test', data)
    return sampler


@pytest.fixture
def sample_sampler_metric():
    """
    Sampler с колонками, содержащими '_metric'.
    """
    df = pd.DataFrame({
        'session_id': [1, 1, 2, 2],
        'assessor_id': ['a', 'b', 'a', 'b'],
        'accuracy_metric': [0.8, 0.9, 0.7, 0.85],
        'fluency_metric': [5, 5, 5, 5],               # константа
        'relevance_metric': [3, 1, 2, 4],              # разброс
        'extra_col': ['x', 'x', 'y', 'y']
    })
    data = {'X': df, 'y': pd.DataFrame()}
    sampler = Sampler()
    setattr(sampler, 'test', data)
    return sampler


@pytest.fixture
def sample_sampler_empty():
    """
    Sampler с пустыми данными (нет числовых критериев).
    """
    df = pd.DataFrame({'dummy': ['a', 'b', 'c']})
    data = {'X': df, 'y': pd.DataFrame()}
    sampler = Sampler()
    setattr(sampler, 'test', data)
    return sampler


def test_compute_informativeness(sample_sampler):
    df = sample_sampler.test['X']
    criterion_columns = ['crit1', 'crit2', 'crit3']
    measures, uninformative_list, share, missing = compute_informativeness(
        df, criterion_columns, default_measure, uninformativeness_threshold=0.05
    )
    assert set(measures.keys()) == {'crit1', 'crit2', 'crit3'}
    assert missing == []
    # crit2 — константа, мера (дисперсия) 0 → неинформативна
    assert 'crit2' in uninformative_list
    # crit1 и crit3 имеют меру >0.05
    assert 'crit1' not in uninformative_list
    assert 'crit3' not in uninformative_list
    # Доля неинформативных = 1/3
    assert share == 1 / 3


def test_compute_informativeness_custom_criteria(sample_sampler):
    df = sample_sampler.test['X']
    criterion_columns = ['crit1', 'crit2']
    measures, uninformative_list, share, missing = compute_informativeness(
        df, criterion_columns, default_measure, uninformativeness_threshold=0.05
    )
    assert set(measures.keys()) == {'crit1', 'crit2'}
    assert missing == []
    assert 'crit2' in uninformative_list
    assert 'crit1' not in uninformative_list
    assert share == 0.5


def test_compute_informativeness_auto_detect(sample_sampler_metric):
    df = sample_sampler_metric.test['X']
    # Не передаём criterion_columns, должна сработать автодетекция по '_metric'
    measures, uninformative_list, share, missing = compute_informativeness(
        df, criterion_columns=None, measure_func=default_measure, uninformativeness_threshold=0.05
    )
    expected = {'accuracy_metric', 'fluency_metric', 'relevance_metric'}
    assert set(measures.keys()) == expected
    # fluency_metric — константа → неинформативна
    assert 'fluency_metric' in uninformative_list
    # accuracy_metric и relevance_metric имеют разброс → информативны
    assert 'accuracy_metric' not in uninformative_list
    assert 'relevance_metric' not in uninformative_list
    assert share == 1 / 3


def test_compute_informativeness_missing_columns(sample_sampler):
    df = sample_sampler.test['X']
    # Передаём колонки, часть из которых отсутствует
    criterion_columns = ['crit1', 'crit2', 'non_existent1', 'non_existent2']
    measures, uninformative_list, share, missing = compute_informativeness(
        df, criterion_columns, default_measure, uninformativeness_threshold=0.05
    )
    # Ожидаем, что в measures будут все 4 колонки, но для отсутствующих np.nan
    assert set(measures.keys()) == {'crit1', 'crit2', 'non_existent1', 'non_existent2'}
    assert set(missing) == {'non_existent1', 'non_existent2'}
    # Отсутствующие попадают в uninformative_list
    assert 'non_existent1' in uninformative_list
    assert 'non_existent2' in uninformative_list
    # Доля считается только по существующим (crit1, crit2)
    # crit2 неинформативен, crit1 информативен → 1 из 2 → 0.5
    assert share == 0.5


def test_valtest_criterion_informativeness(sample_sampler):
    criterion_columns = ['crit1', 'crit2', 'crit3']
    result = valtest_criterion_informativeness(
        sampler=sample_sampler,
        data_type='test',
        criterion_columns=criterion_columns,
        uninformativeness_threshold=0.05,
        is_info=False
    )
    assert 'report' in result
    assert 'precomputed' in result
    report = result['report']
    assert 'semaphore' in report
    assert 'result_plots' in report
    assert 'result_dataframes' in report
    assert isinstance(report['result_dataframes'], list)
    # Доля неинформативных = 1/3 → 0.33 → жёлтый (0 < 0.33 <= 0.5)
    assert report['semaphore'] == 'yellow'
    pre = result['precomputed']
    assert 'measures' in pre
    assert 'criterion_columns' in pre
    assert pre['criterion_columns'] == criterion_columns
    assert set(pre['measures'].keys()) == {'crit1', 'crit2', 'crit3'}


def test_valtest_with_is_info(sample_sampler):
    criterion_columns = ['crit1', 'crit2', 'crit3']
    result = valtest_criterion_informativeness(
        sampler=sample_sampler,
        data_type='test',
        criterion_columns=criterion_columns,
        uninformativeness_threshold=0.05,
        is_info=True
    )
    assert result['report']['semaphore'] == 'gray'


def test_valtest_with_precomputed(sample_sampler):
    criterion_columns = ['crit1', 'crit2', 'crit3']
    # Первый вызов — вычисляем меры
    result1 = valtest_criterion_informativeness(
        sampler=sample_sampler,
        data_type='test',
        criterion_columns=criterion_columns,
        uninformativeness_threshold=0.05,
        is_info=False
    )
    pre = result1['precomputed']
    # Второй вызов с precomputed и другим порогом (0.2)
    # Мера crit1 ≈0.1667 < 0.2 → становится неинформативным
    result2 = valtest_criterion_informativeness(
        sampler=sample_sampler,
        data_type='test',
        criterion_columns=criterion_columns,
        uninformativeness_threshold=0.2,
        is_info=False,
        measures=pre['measures']
    )
    # При пороге 0.05 неинформативна только crit2 → доля 1/3 → жёлтый
    # При пороге 0.2 crit1 также неинформативен → доля 2/3 → красный
    assert result1['report']['semaphore'] == 'yellow'
    assert result2['report']['semaphore'] == 'red'


def test_valtest_auto_detection(sample_sampler_metric):
    # Не передаём criterion_columns, автодетекция по '_metric'
    result = valtest_criterion_informativeness(
        sampler=sample_sampler_metric,
        data_type='test',
        uninformativeness_threshold=0.05,
        is_info=False
    )
    pre = result['precomputed']
    expected = {'accuracy_metric', 'fluency_metric', 'relevance_metric'}
    assert set(pre['measures'].keys()) == expected
    # fluency_metric — константа, её мера = 0
    # остальные >0.05
    # Доля неинформативных = 1/3 → жёлтый
    assert result['report']['semaphore'] == 'yellow'


def test_valtest_sberds_mode(sample_sampler):
    # Подготавливаем данные в формате sberds (словарь)
    data = {
        'test': {
            'X': sample_sampler.test['X'].copy(),
            'y': pd.DataFrame()
        }
    }
    criterion_columns = ['crit1', 'crit2', 'crit3']
    result = valtest_criterion_informativeness(
        sampler=data,
        data_type='test',
        criterion_columns=criterion_columns,
        exec_env='sberds'
    )
    assert 'report' in result
    assert 'precomputed' in result
    assert result['report']['semaphore'] == 'yellow' 