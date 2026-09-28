import pytest
import joblib

from llm_val.sampler.Sampler import Sampler
from llm_val.sampler.RuBQSampler import RuBQSampler
from llm_val.sampler.MMLUSampler import MMLUSampler

SAMPLERS = [(Sampler, 'data/data_mmlu.pkl', False, None, None),
            (RuBQSampler, 'data/data_rubq.pkl', True, {'system',
                                                       'question',
                                                       'context',
                                                       'thema',
                                                       'true_answer',
                                                       'raw_prompt',
                                                       'model_answer',
                                                       'few_shot'}, {'metric_value'}),
            (MMLUSampler, 'data/data_mmlu.pkl', True, {'system',
                                                       'question',
                                                       'context',
                                                       'thema',
                                                       'answer_options',
                                                       'true_answer',
                                                       'raw_prompt',
                                                       'model_answer',
                                                       'few_shot'}, {'metric_value'})]


@pytest.fixture(params=SAMPLERS)
def create_sampler(request):
    sampler_class, data_path, y_flag, X_col, y_col = request.param
    data = joblib.load(data_path)
    if not y_flag:
        sampler_data = {'X': data, 'y': None}
    else:
        sampler_data = data
    sampler = sampler_class(test=sampler_data)
    metadata = {'data': data, 'X_columns': X_col, 'y_columns': y_col}
    return sampler, metadata


def test_create_sampler(create_sampler):
    sampler, _ = create_sampler
    assert isinstance(sampler, Sampler)


def test_getter_sampler(create_sampler):
    sampler, metadata = create_sampler
    assert id(metadata['data']) != id(sampler.test['X'])
    assert len(metadata['data']) == len(sampler.test['X'])
    if metadata['X_columns'] is not None:
        assert set(sampler.test['X'].columns) == metadata['X_columns']
    if metadata['y_columns'] is not None:
        assert set(sampler.test['y'].columns) == metadata['y_columns']
    

def test_raise_create_sampler():
    data_path = 'data/data_mmlu.pkl'
    sampler_data = joblib.load(data_path)
    with pytest.raises(AttributeError):
        Sampler(test=sampler_data)
    with pytest.raises(AttributeError):
        Sampler(test={'X_train': sampler_data, 'y_train': None})
