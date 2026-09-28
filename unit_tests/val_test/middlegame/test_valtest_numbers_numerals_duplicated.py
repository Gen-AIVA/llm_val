import os
import pytest
import joblib
import numpy as np
import pandas as pd

from llm_val.sampler.RuBQSampler import RuBQSampler
from llm_val.val_test.middlegame.valtest_numbers_numerals_duplicated import valtest_numbers_numerals_duplicated

DATA_PATH_RUBQ = 'data/data_rubq.pkl'

PARAMS = [(RuBQSampler, DATA_PATH_RUBQ)]

@pytest.fixture(params=PARAMS)
def create_sampler(request):
    sampler_class, data_path = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    return sampler


def test_valtest_numbers_numerals_duplicated(create_sampler):
    sampler = create_sampler
    res = valtest_numbers_numerals_duplicated(sampler=sampler,
                                              semaphore_threshold=(0.0, 0.05),
                                              is_info=False,
                                              not_duplicated_dict=None)
    assert res['report']['semaphore'] == 'red'

    
def test_valtest_numbers_numerals_duplicated_is_info(create_sampler):
    sampler = create_sampler
    res = valtest_numbers_numerals_duplicated(sampler=sampler,
                                              semaphore_threshold=(0.0, 0.05),
                                              is_info=True,
                                              not_duplicated_dict=None)
    assert res['report']['semaphore'] == 'gray'
    assert isinstance(res['precomputed'], dict)
    
    
def test_valtest_numbers_numerals_duplicated_keys(create_sampler):
    sampler = create_sampler
    res = valtest_numbers_numerals_duplicated(sampler=sampler,
                                              semaphore_threshold=(0.0, 0.05),
                                              is_info=False,
                                              not_duplicated_dict=None)
    assert set(res.keys()) == {'report', 'precomputed'}
    
    
def test_valtess_numbers_numerals_duplicated_report(create_sampler):
    sampler = create_sampler
    res = valtest_numbers_numerals_duplicated(sampler=sampler,
                                              semaphore_threshold=(0.0, 0.05),
                                              is_info=False,
                                              not_duplicated_dict=None)
    report = res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)
    assert report['semaphore'] == 'red'
    
    
def test_valtest_numbers_numerals_duplicated_precomputed(create_sampler):
    sampler = create_sampler
    res = valtest_numbers_numerals_duplicated(sampler=sampler,
                                              semaphore_threshold=(0.0, 0.05),
                                              is_info=False,
                                              not_duplicated_dict=None)
    assert isinstance(res['precomputed'], dict)
    assert 'amount' in res['precomputed']
    assert np.allclose(0.8571428571428571, res['precomputed']['amount'], 0.1) == True
    