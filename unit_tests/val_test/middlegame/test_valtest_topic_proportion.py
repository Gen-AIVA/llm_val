import pytest
import pandas as pd

from llm_val.val_test.middlegame.valtest_topic_proportion import valtest_topic_proportion


@pytest.fixture
def create_sampler():
    data = pd.DataFrame({
        'thema': ['Тема 1', 'Тема 2', 'Тема 2', 'Тема 3', 'Тема 3', 'Тема 3']
    })
    sampler = type('Sampler', (object,), {'test': {'X': data}})
    return sampler


@pytest.fixture
def create_validator_dict():
    return {
        'Тема 1': 0.1,
        'Тема 2': 0.3,
        'Тема 3': 0.6
    }


@pytest.fixture
def create_empty_sampler():
    data = pd.DataFrame({'thema': []})
    sampler = type('Sampler', (object,), {'test': {'X': data}})
    return sampler


def test_valtest_topic_proportion_green_semaphore(create_sampler, create_validator_dict):
    result = valtest_topic_proportion(create_sampler, create_validator_dict)
    assert result['report']['semaphore'] == 'green'


def test_valtest_topic_proportion_red_semaphore(create_sampler, create_validator_dict):
    create_validator_dict['Тема 1'] = 0.8
    create_validator_dict['Тема 2'] = 0.1
    create_validator_dict['Тема 3'] = 0.1
    result = valtest_topic_proportion(create_sampler, create_validator_dict)
    assert result['report']['semaphore'] == 'red'


def test_valtest_topic_proportion_gray_semaphore(create_sampler, create_validator_dict):
    result = valtest_topic_proportion(create_sampler, create_validator_dict, is_info=True)
    assert result['report']['semaphore'] == 'gray'


def test_valtest_topic_proportion_empty_data(create_empty_sampler, create_validator_dict):
    result = valtest_topic_proportion(create_empty_sampler, create_validator_dict)
    assert result['report']['semaphore'] == 'red'


def test_valtest_topic_proportion_precomputed(create_sampler, create_validator_dict):
    result = valtest_topic_proportion(create_sampler, create_validator_dict)
    precomputed = result['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'psi_value'}
    assert isinstance(precomputed['psi_value'], float)