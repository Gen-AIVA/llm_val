import pytest
import joblib
import numpy as np

from llm_val.sampler.Sampler import Sampler
from llm_val.sampler.DialogueSampler import DialogueSampler

DATA_PATH = './data/joy_dataset.pkl'

PARAMS = [(DialogueSampler, DATA_PATH, {'dialogue_id',
                                        'content',
                                        'embedding',
                                        'pos_id'}, {'dialogue_id', 
                                                    'role',
                                                    'pos_id'})]


@pytest.fixture(params=PARAMS)
def create_sampler(request):
    sampler_class, data_path, X_col, y_col = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    metadata = {'data': data, 'X_columns': X_col, 'y_columns': y_col}
    return sampler, metadata


def test_create_sampler(create_sampler):
    sampler, _ = create_sampler
    assert isinstance(sampler, Sampler)


def test_getter_sampler(create_sampler):
    sampler, metadata = create_sampler
    assert id(metadata['data']) != id(sampler.test['X'])
    assert len(metadata['data']) == np.max(sampler.test['X']['dialogue_id']) + 1    
    if metadata['X_columns'] is not None:
        assert set(sampler.test['X'].columns) == metadata['X_columns']
    if metadata['y_columns'] is not None:
        assert set(sampler.test['y'].columns) == metadata['y_columns']