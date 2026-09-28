import typing as tp
import pytest
import numpy as np
import pandas as pd

from unittest.mock import MagicMock
from llm_val.val_test.middlegame.subtests_test_dataset_val.valtest_outliers_by_char_count import (
    valtest_outliers_by_char_count,
    report_valtest_outliers_by_char_count,
    get_worst_outlier_data,
    calculate_char_outlier_stats
)


@pytest.fixture
def default_data_input_query() -> pd.DataFrame:
    '''Создает тестовый датасет с 10 нормальными значениями и 2 выбросами для input_query'''
    normal_values = ['а' * 50 for _ in range(10)]
    outlier_values = ['а' * 500, 'а']
    return pd.DataFrame({'input_query': normal_values + outlier_values})


@pytest.fixture
def default_data_dialogue() -> pd.DataFrame:
    '''
    Создает тестовый датасет с диалогами.
    Диалог хранится как список троек: (query_id, input_query, output_answer).
    '''
    normal_values = [
        [('id1', 'а' * 50, 'б' * 50), ('id2', 'а' * 50, 'б' * 50)] 
        for _ in range(10)
    ]
    outlier_values = [
        [('id3', 'а' * 1000, 'б' * 1000)],  
        [('id4', 'аа', 'бб')]                
    ]
    return pd.DataFrame({'dialogue': normal_values + outlier_values})


@pytest.fixture
def empty_data() -> pd.DataFrame:
    '''Создает пустой датасет'''
    return pd.DataFrame({'input_query': pd.Series(dtype=object)})


@pytest.fixture
def mock_sampler_default(default_data_input_query: pd.DataFrame) -> MagicMock:
    '''Мок Sampler для режима default'''
    sampler = MagicMock()
    sampler.test = {'X': default_data_input_query}
    return sampler


@pytest.fixture
def mock_sampler_sberds(default_data_input_query: pd.DataFrame) -> tp.Dict[str, tp.Dict[str, pd.DataFrame]]:
    '''Мок Sampler для режима sberds'''
    return {'test': {'X': default_data_input_query}}


@pytest.fixture
def mock_sampler_dialogue(default_data_dialogue: pd.DataFrame) -> MagicMock:
    '''Мок Sampler с датасетом диалогов'''
    sampler = MagicMock()
    sampler.test = {'X': default_data_dialogue}
    return sampler


@pytest.fixture
def mock_sampler_empty(empty_data: pd.DataFrame) -> MagicMock:
    '''Мок Sampler с пустым датасетом'''
    sampler = MagicMock()
    sampler.test = {'X': empty_data}
    return sampler


def test_valtest_format_default(mock_sampler_default: MagicMock):
    '''Проверка формата выхода основной функции в режиме default'''
    test_res = valtest_outliers_by_char_count(sampler=mock_sampler_default, exec_env='default')
    
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}
    
    report = test_res['report']
    precomputed = test_res['precomputed']
    assert isinstance(report, dict)
    assert isinstance(precomputed, dict)
    
    assert 'semaphore' in report
    assert 'result_plots' in report
    assert 'result_dataframes' in report
    assert 'input_stats' in precomputed
    assert 'output_stats' not in precomputed


def test_valtest_format_sberds(mock_sampler_sberds: tp.Dict[str, tp.Dict[str, pd.DataFrame]]):
    '''Проверка работы с exec_env=sberds'''
    test_res = valtest_outliers_by_char_count(sampler=mock_sampler_sberds, exec_env='sberds')
    
    assert isinstance(test_res['precomputed']['input_stats']['outlier_ratio'], float)
    assert test_res['precomputed']['input_stats']['outlier_ratio'] > 0.0


def test_outlier_detection_logic_input_query(mock_sampler_default: MagicMock):
    '''Проверка логики подсчета доли выбросов для обычного input_query'''
    test_res = valtest_outliers_by_char_count(sampler=mock_sampler_default, exec_env='default')
    outlier_ratio = test_res['precomputed']['input_stats']['outlier_ratio']
    
    assert outlier_ratio > 0.0
    assert len(test_res['precomputed']['input_stats']['outlier_indices']) > 0


def test_outlier_detection_logic_dialogue(mock_sampler_dialogue: MagicMock):
    '''Проверка логики подсчета выбросов для колонки dialogue (раздельно по инпутам и аутпутам)'''
    test_res = valtest_outliers_by_char_count(sampler=mock_sampler_dialogue, exec_env='default')
    
    input_ratio = test_res['precomputed']['input_stats']['outlier_ratio']
    output_ratio = test_res['precomputed']['output_stats']['outlier_ratio']
    
    assert input_ratio > 0.0
    assert len(test_res['precomputed']['input_stats']['outlier_indices']) > 0
    assert output_ratio > 0.0
    assert len(test_res['precomputed']['output_stats']['outlier_indices']) > 0
    assert 'output_stats' in test_res['precomputed']


def test_empty_dataset(mock_sampler_empty: MagicMock):
    '''Проверка обработки пустого датасета (защита от ZeroDivisionError и IndexError)'''
    test_res = valtest_outliers_by_char_count(sampler=mock_sampler_empty, exec_env='default')
    
    assert test_res['precomputed']['input_stats']['outlier_ratio'] == 0.0
    assert test_res['precomputed']['input_stats']['outlier_indices'] == []


def test_precomputed_usage_input_only(mock_sampler_default: MagicMock):
    '''Проверка использования предвычисленных значений из precomputed для инпутов'''
    precomputed_input = {
        'outlier_ratio': 0.05,
        'outlier_indices': [],
        'char_counts': {0: 50},
        'q1': 50.0,
        'q3': 50.0,
        'iqr': 0.0,
        'lower_bound': 50.0,
        'upper_bound': 50.0
    }
    test_res = valtest_outliers_by_char_count(
        sampler=mock_sampler_default,
        input_stats=precomputed_input,
        exec_env='default'
    )
    
    assert test_res['precomputed']['input_stats']['outlier_ratio'] == 0.05
    assert test_res['report']['semaphore'] == 'green'


def test_precomputed_usage_dialogue(mock_sampler_dialogue: MagicMock):
    '''Проверка использования предвычисленных значений из precomputed для диалогов'''
    precomputed_input = {
        'outlier_ratio': 0.05,
        'outlier_indices': [],
        'char_counts': {0: 100},
        'q1': 100.0,
        'q3': 100.0,
        'iqr': 0.0,
        'lower_bound': 100.0,
        'upper_bound': 100.0
    }
    precomputed_output = {
        'outlier_ratio': 0.2,
        'outlier_indices': [1],
        'char_counts': {0: 100, 1: 5000},
        'q1': 100.0,
        'q3': 100.0,
        'iqr': 0.0,
        'lower_bound': 100.0,
        'upper_bound': 100.0
    }
    test_res = valtest_outliers_by_char_count(
        sampler=mock_sampler_dialogue,
        input_stats=precomputed_input,
        output_stats=precomputed_output,
        exec_env='default'
    )
    
    assert test_res['precomputed']['input_stats']['outlier_ratio'] == 0.05
    assert test_res['precomputed']['output_stats']['outlier_ratio'] == 0.2
    assert test_res['report']['semaphore'] == 'yellow'


def test_report_semaphore_green():
    '''Проверка выставления зеленого светофора'''
    report = report_valtest_outliers_by_char_count(outlier_ratio=0.05, outlier_indices=[], threshold=0.1)
    assert report['semaphore'] == 'green'
    assert report['result_dataframes'][0].iloc[0]['Результат'] == 'Пройдена'


def test_report_semaphore_yellow():
    '''Проверка выставления желтого светофора'''
    report = report_valtest_outliers_by_char_count(outlier_ratio=0.15, outlier_indices=[1, 2], threshold=0.1)
    assert report['semaphore'] == 'yellow'
    assert report['result_dataframes'][0].iloc[0]['Результат'] == 'Не пройдена'


def test_report_semaphore_info():
    '''Проверка работы флага is_info'''
    report = report_valtest_outliers_by_char_count(outlier_ratio=0.05, outlier_indices=[], is_info=True)
    assert report['semaphore'] == 'gray'


def test_helper_get_worst_outlier_data():
    '''Проверка функции определения худшей доли выбросов'''
    worst_ratio, combined_indices = get_worst_outlier_data(0.1, [1], 0.2, [2, 3])
    assert worst_ratio == 0.2
    assert set(combined_indices) == {1, 2, 3}

    worst_ratio_no_out, combined_indices_no_out = get_worst_outlier_data(0.1, [1], None, None)
    assert worst_ratio_no_out == 0.1
    assert combined_indices_no_out == [1]


def test_helper_calculate_char_outlier_stats():
    '''Проверка функции расчета статистик выбросов'''
    char_series = pd.Series([10, 10, 10, 10, 1000])
    stats = calculate_char_outlier_stats(char_series)
    
    assert stats['outlier_ratio'] == 0.2
    assert stats['outlier_indices'] == [4]
    assert stats['q1'] == 10.0
    assert stats['iqr'] == 0.0
    
    empty_stats = calculate_char_outlier_stats(pd.Series(dtype=int))
    assert empty_stats['outlier_ratio'] == 0.0
    assert empty_stats['outlier_indices'] == []
