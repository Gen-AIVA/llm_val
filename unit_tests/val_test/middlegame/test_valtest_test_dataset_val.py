import typing as tp
import pytest
import pandas as pd

from llm_val.val_test.middlegame.valtest_test_dataset_val import (
    valtest_test_dataset_val,
    discover_subtests,
    extract_subtest_info
)


@pytest.fixture
def create_sampler():
    data = pd.DataFrame({'input_query': ['test query']})
    sampler = type('Sampler', (object,), {'test': {'X': data}})
    return sampler


@pytest.fixture
def green_subtest_report():
    return {
        'semaphore': 'green',
        'result_plots': [],
        'result_dataframes': [
            pd.DataFrame({
                'Проверка': ['Успешная проверка'],
                'Результат': ['Пройдена'],
                'Комментарий': ['Ок']
            })
        ]
    }


@pytest.fixture
def yellow_subtest_report():
    return {
        'semaphore': 'yellow',
        'result_plots': [],
        'result_dataframes': [
            pd.DataFrame({
                'Проверка': ['Проверка с предупреждением'],
                'Результат': ['Не пройдена'],
                'Комментарий': ['Есть отклонения']
            })
        ]
    }


def mock_subtest(subtest_report: tp.Dict[str, tp.Any], **kwargs) -> tp.Dict[str, tp.Any]:
    '''Вспомогательная функция-мок для подтеста'''
    return {'report': subtest_report, 'precomputed': {}}


def mock_discover_single(subtest_report: tp.Dict[str, tp.Any], *args, **kwargs) -> tp.Dict[str, tp.Any]:
    '''Мок функции discover_subtests, возвращающий один подтест'''
    return {'mock_subtest': lambda **inner_kwargs: mock_subtest(subtest_report, **inner_kwargs)}


def mock_discover_multiple(green_report: tp.Dict[str, tp.Any],
                           yellow_report: tp.Dict[str, tp.Any],
                           *args, **kwargs) -> tp.Dict[str, tp.Any]:
    '''Мок функции discover_subtests, возвращающий несколько подтестов'''
    return {
        'subtest_green': lambda **inner_kwargs: mock_subtest(green_report, **inner_kwargs),
        'subtest_yellow': lambda **inner_kwargs: mock_subtest(yellow_report, **inner_kwargs)
    }


@pytest.mark.parametrize(
    'is_info, expected_semaphore',
    [
        (False, 'green'),
        (True, 'gray')
    ]
)
def test_valtest_single_subtest_semaphore(create_sampler, green_subtest_report, monkeypatch, is_info, expected_semaphore):
    '''Тест проверки итогового светофора при одном подтесте (green и gray)'''
    monkeypatch.setattr(
        'llm_val.val_test.middlegame.valtest_test_dataset_val.discover_subtests',
        lambda *args, **kwargs: mock_discover_single(green_subtest_report, *args, **kwargs)
    )
    result = valtest_test_dataset_val(create_sampler, is_info=is_info)
    assert result['report']['semaphore'] == expected_semaphore


def test_valtest_yellow_semaphore(create_sampler, green_subtest_report, yellow_subtest_report, monkeypatch):
    '''Тест проверки итогового светофора при нескольких подтестах (yellow)'''
    monkeypatch.setattr(
        'llm_val.val_test.middlegame.valtest_test_dataset_val.discover_subtests',
        lambda *args, **kwargs: mock_discover_multiple(green_subtest_report, yellow_subtest_report, *args, **kwargs)
    )
    result = valtest_test_dataset_val(create_sampler)
    assert result['report']['semaphore'] == 'yellow'


def test_valtest_format(create_sampler, green_subtest_report, monkeypatch):
    '''Тест проверки формата выхода основной функции'''
    def mock_discover_with_precomputed(*args, **kwargs):
        return {'mock_subtest': lambda **inner_kwargs: {'report': green_subtest_report, 'precomputed': {'some_key': 1}}}
    
    monkeypatch.setattr(
        'llm_val.val_test.middlegame.valtest_test_dataset_val.discover_subtests',
        mock_discover_with_precomputed
    )
    test_res = valtest_test_dataset_val(create_sampler)
    
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}
    
    report = test_res['report']
    assert isinstance(report, dict)
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)


def test_discover_subtests_all():
    all_subtests = discover_subtests()
    assert isinstance(all_subtests, dict)
    assert len(all_subtests) > 0
    for func_name, func in all_subtests.items():
        assert callable(func)


def test_discover_subtests_specific():
    target_subtest = 'valtest_outliers_by_char_count'
    filtered_subtests = discover_subtests(subtests=[target_subtest])
    assert isinstance(filtered_subtests, dict)
    assert target_subtest in filtered_subtests
    assert len(filtered_subtests) == 1


def test_extract_subtest_info_full(green_subtest_report):
    name, res, comment = extract_subtest_info(
        subtest_report=green_subtest_report,
        default_name='default'
    )
    assert name == 'Успешная проверка'
    assert res == 'Пройдена'
    assert comment == 'Ок'