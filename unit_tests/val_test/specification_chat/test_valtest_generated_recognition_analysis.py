import pytest
import joblib
import pandas as pd

from llm_val.sampler.DialogueSampler import DialogueSampler
from llm_val.val_test.specification_chat.valtest_generated_recognition_analysis import valtest_generated_recognition_analysis

DATA_PATH = './data/joy_dataset.pkl'

PARAMS = [(DialogueSampler, DATA_PATH)]

@pytest.fixture(params=PARAMS)
def create_sampler(request):
    sampler_class, data_path = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    return sampler


def test_valtest_generated_recognition_analysis_result(create_sampler):
    sampler = create_sampler
    res = valtest_generated_recognition_analysis(sampler=sampler,
                                                 toloka_size=0.01,
                                                 test_size=0.4,
                                                 statistics=None)
    assert isinstance(res['report'], dict)
    assert isinstance(res['precomputed'], dict)
    
    
def test_valtest_generated_recognition_analysis_keys(create_sampler):
    sampler = create_sampler
    res = valtest_generated_recognition_analysis(sampler=sampler,
                                                 toloka_size=0.01,
                                                 test_size=0.4,
                                                 statistics=None)
    assert set(res.keys()) == {'report', 'precomputed'}
    assert set(res['precomputed'].keys()) == {'human_human', 'human_llm'} 
    assert set(res['report'].keys()) == {'semaphore', 'result_dataframes', 'result_plots'}

    
def test_valtest_generated_recognition_analysis_precomputed(create_sampler):
    sampler = create_sampler
    res = valtest_generated_recognition_analysis(sampler=sampler,
                                                 toloka_size=0.01,
                                                 test_size=0.4,
                                                 statistics=None)
    assert isinstance(res['precomputed']['human_human']['gini_fact'], float)
    assert isinstance(res['precomputed']['human_human']['gini_shuffle_list'], list)
    assert isinstance(res['precomputed']['human_llm']['gini_fact'], float)
    assert isinstance(res['precomputed']['human_llm']['gini_shuffle_list'], list)

    
def test_valtest_generated_recognition_analysis_report(create_sampler):
    sampler = create_sampler
    res = valtest_generated_recognition_analysis(sampler=sampler,
                                                 toloka_size=0.01,
                                                 statistics=None)
    assert isinstance(res['report']['result_dataframes'][0], pd.DataFrame)
    assert res['report']['semaphore'] == 'gray'
