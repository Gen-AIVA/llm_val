import pytest
import joblib

from llm_val.sampler.Sampler import Sampler
from llm_val.sampler.NerSampler import NerSampler

DATA_PATH = 'data/data_ner.pkl'
ENTITY_TYPES = ['Drugname', 'Drugclass', 'Drugform', 'DI', 'ADR', 'Finding']


@pytest.fixture()
def create_sampler():
    data = joblib.load(DATA_PATH)
    sampler = NerSampler(test=data, entity_types=ENTITY_TYPES)
    metadata = {'data': data}
    return sampler, metadata


def test_create_sampler(create_sampler):
    sampler, _ = create_sampler
    assert isinstance(sampler, Sampler)


def test_getter_sampler(create_sampler):
    sampler, metadata = create_sampler
    assert id(metadata['data']) != id(sampler.test['X'])
    assert len(metadata['data']) == len(sampler.test['X'])
    assert set(sampler.test['X'].columns) == {'id', 'text', 'extracted', 'target', 'parser_error',
                                             'flat_extracted_entities', 'flat_target_entities',
                                             'extracted_all_types', 'extracted_by_type_merged',
                                             'extracted_description_detalization',
                                             'extracted_examples_addition',
                                             'extracted_description_detalization_and_examples_addition'}
    assert set(sampler.test['y'].columns) == {'extracted', 'target'}


def test_entity_types(create_sampler):
    sampler, _ = create_sampler
    assert sampler.entity_types == ENTITY_TYPES
