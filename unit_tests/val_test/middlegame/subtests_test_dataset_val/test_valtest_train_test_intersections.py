import pytest
import pandas as pd

from llm_val.val_test.middlegame.subtests_test_dataset_val.valtest_train_test_intersections import (
    compute_intersection_stats,
    dialogue_to_text,
    get_duplicate_indices_to_skip,
    valtest_train_test_intersections,
)


def build_sampler(train_data: pd.DataFrame=None,
                  test_data: pd.DataFrame=None):
    '''Создать sampler для exec_env default.'''
    sampler_attrs = {}
    if train_data is not None:
        sampler_attrs['train'] = {'X': train_data}
    if test_data is not None:
        sampler_attrs['test'] = {'X': test_data}
    sampler = type('Sampler', (object,), sampler_attrs)
    return sampler


def get_result_status(test_result: dict) -> str:
    '''Получить статус проверки из результата теста.'''
    result_dataframe = test_result['report']['result_dataframes'][0]
    return str(result_dataframe['Результат'].iloc[0])


def test_valtest_train_test_intersections_output_format() -> None:
    '''Проверить формат выхода основной функции.'''
    train_data = pd.DataFrame({'input_query': ['train']})
    test_data = pd.DataFrame({'input_query': ['test']})
    sampler = build_sampler(train_data=train_data, test_data=test_data)

    test_result = valtest_train_test_intersections(sampler=sampler)

    assert isinstance(test_result, dict)
    assert set(test_result.keys()) == {'report', 'precomputed'}
    assert isinstance(test_result['report'], dict)
    assert isinstance(test_result['precomputed'], dict)


def test_valtest_train_test_intersections_passed_less_than_threshold() -> None:
    '''Проверить статус Пройдена, если пересечений меньше порога.'''
    train_data = pd.DataFrame({'input_query': ['query_0']})
    test_data = pd.DataFrame({'input_query': [f'query_{index}' for index in range(20)]})
    sampler = build_sampler(train_data=train_data, test_data=test_data)

    test_result = valtest_train_test_intersections(sampler=sampler, threshold=0.1)
    stats = test_result['precomputed']['intersections_stats']

    assert test_result['report']['semaphore'] == 'green'
    assert get_result_status(test_result) == 'Пройдена'
    assert stats['intersection_indices'] == [0]
    assert stats['intersection_ratio'] == pytest.approx(0.05)


def test_valtest_train_test_intersections_failed_at_threshold() -> None:
    '''Проверить статус Не пройдена, если доля пересечений равна порогу.'''
    train_data = pd.DataFrame({'input_query': ['query_0']})
    test_data = pd.DataFrame({'input_query': [f'query_{index}' for index in range(10)]})
    sampler = build_sampler(train_data=train_data, test_data=test_data)

    test_result = valtest_train_test_intersections(sampler=sampler, threshold=0.1)
    stats = test_result['precomputed']['intersections_stats']

    assert test_result['report']['semaphore'] == 'yellow'
    assert get_result_status(test_result) == 'Не пройдена'
    assert stats['intersection_indices'] == [0]
    assert stats['intersection_ratio'] == pytest.approx(0.1)


def test_valtest_train_test_intersections_not_applicable_without_validation() -> None:
    '''Проверить статус Не проводилась, если валидационный датасет отсутствует.'''
    test_data = pd.DataFrame({'input_query': ['test']})
    sampler = build_sampler(test_data=test_data)

    test_result = valtest_train_test_intersections(sampler=sampler)
    stats = test_result['precomputed']['intersections_stats']

    assert test_result['report']['semaphore'] == 'gray'
    assert get_result_status(test_result) == 'Не проводилась'
    assert stats['is_applicable'] is False


def test_valtest_train_test_intersections_not_applicable_with_none_validation() -> None:
    '''Проверить статус Не проводилась, если валидационный датасет равен None.'''
    test_data = pd.DataFrame({'input_query': ['test']})
    sampler = type('Sampler', (object,), {'train': None, 'test': {'X': test_data}})

    test_result = valtest_train_test_intersections(sampler=sampler)
    stats = test_result['precomputed']['intersections_stats']

    assert test_result['report']['semaphore'] == 'gray'
    assert get_result_status(test_result) == 'Не проводилась'
    assert stats['is_applicable'] is False


def test_valtest_train_test_intersections_not_applicable_with_none_validation_sberds() -> None:
    '''Проверить статус Не проводилась, если sberds-валидационный датасет равен None.'''
    sampler = {
        'train': None,
        'test': {'X': pd.DataFrame({'input_query': ['test']})}
    }

    test_result = valtest_train_test_intersections(sampler=sampler, exec_env='sberds')
    stats = test_result['precomputed']['intersections_stats']

    assert test_result['report']['semaphore'] == 'gray'
    assert get_result_status(test_result) == 'Не проводилась'
    assert stats['is_applicable'] is False


def test_valtest_train_test_intersections_uses_first_duplicate_example() -> None:
    '''Проверить, что среди дублей используется только пример с минимальным индексом.'''
    train_data = pd.DataFrame({'input_query': ['same']})
    test_data = pd.DataFrame({'input_query': ['other', 'same', 'same', 'third']})
    sampler = build_sampler(train_data=train_data, test_data=test_data)

    test_result = valtest_train_test_intersections(sampler=sampler, duplicates=[(1, 2)])
    stats = test_result['precomputed']['intersections_stats']

    assert stats['checked_test_indices'] == [0, 1, 3]
    assert stats['intersection_indices'] == [1]
    assert stats['intersection_ratio'] == pytest.approx(1 / 3)


def test_valtest_train_test_intersections_dialogue_input_only() -> None:
    '''Проверить сравнение диалогов только по пользовательскому вводу.'''
    train_data = pd.DataFrame(
        {
            'dialogue': [[('id_0', 'question', 'answer')]]
        }
    )
    test_data = pd.DataFrame(
        {
            'dialogue': [
                [('id_1', 'question', 'another answer')],
                [('id_2', 'question', 'answer')]
            ]
        }
    )
    sampler = build_sampler(train_data=train_data, test_data=test_data)

    test_result = valtest_train_test_intersections(sampler=sampler)
    stats = test_result['precomputed']['intersections_stats']

    assert stats['intersection_indices'] == [0, 1]
    assert dialogue_to_text([('technical_id', 'question', 'answer')]) == 'question'


def test_valtest_train_test_intersections_sberds() -> None:
    '''Проверить работу в exec_env sberds.'''
    sampler = {
        'train': {'X': pd.DataFrame({'input_query': ['query_0']})},
        'test': {'X': pd.DataFrame({'input_query': ['query_0', 'query_1']})}
    }

    test_result = valtest_train_test_intersections(sampler=sampler, exec_env='sberds')

    assert test_result['precomputed']['intersections_stats']['intersection_indices'] == [0]


def test_valtest_train_test_intersections_precomputed_is_used() -> None:
    '''Проверить использование предпосчитанных пересечений.'''
    precomputed_intersections = {
        'intersection_ratio': 0.2,
        'intersection_indices': [2],
        'checked_test_indices': [0, 1, 2, 3, 4],
        'is_applicable': True
    }

    test_result = valtest_train_test_intersections(
        sampler={},
        precomputed_intersections=precomputed_intersections
    )

    assert test_result['report']['semaphore'] == 'yellow'
    assert test_result['precomputed']['intersections_stats'] == precomputed_intersections


def test_get_duplicate_indices_to_skip() -> None:
    '''Проверить получение индексов дублей, которые не участвуют в сравнении.'''
    indices_to_skip = get_duplicate_indices_to_skip(duplicates=[(4, 1, 2), (3, 5)])

    assert indices_to_skip == {2, 4, 5}


def test_compute_intersection_stats_with_empty_test() -> None:
    '''Проверить статистики для пустого тестового датасета.'''
    stats = compute_intersection_stats(
        train_texts=pd.Series(['query']),
        test_texts=pd.Series([], dtype='str')
    )

    assert stats['intersection_ratio'] == 0.0
    assert stats['intersection_indices'] == []
    assert stats['checked_test_indices'] == []
