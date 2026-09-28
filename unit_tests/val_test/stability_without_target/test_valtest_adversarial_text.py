import pytest
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from llm_val.sampler.OOTMMLUSampler import OOTMMLUSampler
from llm_val.val_test.stability_without_target.valtest_adversarial_text import valtest_adversarial_text, make_adversarial_dataset


DATA_PATH_MMLU = 'data/data_mmlu_long.pkl'
DATASET_PARAMS = [(OOTMMLUSampler, DATA_PATH_MMLU, 0.3, 17, False),
                  (OOTMMLUSampler, DATA_PATH_MMLU, 0.5, 17, True)]
PARAMS = [(OOTMMLUSampler, DATA_PATH_MMLU, (0.4, 0.8), ('test', 'oot'), 20, 0.1, 3, 17, False)]


@pytest.fixture(params=PARAMS)
def create_sampler(request):
    sampler_class, data_path, semaphore_threshold, data_types, catboost_iterations, catboost_learning_rate, catboost_depth, random_state, is_info = request.param
    data = joblib.load(data_path)[:100]
    test_data = data[:50]
    oot_data = data[50:]
    sampler = sampler_class(test=test_data, oot=oot_data)
    res = valtest_adversarial_text(sampler=sampler,
                                   semaphore_threshold=semaphore_threshold,
                                   data_types=data_types,
                                   catboost_iterations=catboost_iterations,
                                   catboost_learning_rate=catboost_learning_rate,
                                   catboost_depth=catboost_depth,
                                   random_state=random_state,
                                   is_info=is_info)
    return res, is_info


def test_valtest_adversarial_text_report(create_sampler):
    test_res, is_info = create_sampler
    color = 'gray' if is_info else 'yellow'
    report = test_res['report']
    assert isinstance(report, dict)
    assert report['semaphore'] == color
    result_df = report['result_dataframes'][0]
    assert result_df['Светофор'][0] == color
    assert result_df['Значение Джини'][0] >= 0.4 and result_df['Значение Джини'][0] < 0.8


def test_valtest_adversarial_text_precomputed(create_sampler):
    test_res, is_info = create_sampler
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'gini_value'}
            
        
@pytest.mark.parametrize('dataset_params', DATASET_PARAMS)
def test_make_adversarial_dataset(dataset_params):
    sampler_class, data_path, test_size, random_state, use_context = dataset_params
    data = joblib.load(data_path)[:100]
    split_idx = np.random.randint(low=10, high=90)
    sampler = sampler_class(test=data[:split_idx], oot=data[split_idx:])
    
    cols_to_use = ['question', 'context'] if use_context else ['question']
    test_texts = sampler.test['X'][cols_to_use]
    oot_texts = sampler.oot['X'][cols_to_use]
    
    if use_context:
        test_texts['context'] = test_texts['context'].apply(lambda x: ' '.join(x))
        oot_texts['context'] = oot_texts['context'].apply(lambda x: ' '.join(x))
    adversarial_dataset = make_adversarial_dataset(test_texts, oot_texts, test_size=test_size, random_state=random_state)
    
    assert isinstance(adversarial_dataset, dict)
    assert set(adversarial_dataset.keys()) == {'train', 'test'}
    
    eps = 1 / data.shape[0] + 1 / (10 * data.shape[0])
    assert np.abs(adversarial_dataset['train'].shape[0] / data.shape[0] - (1 - test_size)) <= eps
    assert np.abs(adversarial_dataset['test'].shape[0] / data.shape[0] - test_size) <= eps
    assert all(adversarial_dataset['train'].isna().sum() == 0)
    assert all(adversarial_dataset['test'].isna().sum() == 0)
    
    if use_context:
        assert list(adversarial_dataset['train'].columns) == ['question', 'context', 'target']
        assert list(adversarial_dataset['test'].columns) == ['question', 'context', 'target']
    else:
        assert list(adversarial_dataset['train'].columns) == ['question', 'target']
        assert list(adversarial_dataset['test'].columns) == ['question', 'target']