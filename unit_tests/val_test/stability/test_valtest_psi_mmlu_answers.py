import joblib
import pytest
import pandas as pd
from copy import deepcopy

from llm_val.utils.metrics import METRICS
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.val_test.stability.valtest_psi_mmlu_answers import valtest_psi_mmlu_answers

DATA_PATH_MMLU = 'data/data_mmlu.pkl'
PARAMS = [(MMLUSampler, DATA_PATH_MMLU, 'test', 0.2, False, None),
          (MMLUSampler, DATA_PATH_MMLU, 'test', 0.2, False, {'A': 0.3, 'B': 0.1, 'SUM': 0.4})]

@pytest.fixture(params=PARAMS)
def create_sampler(request):
    sampler_class, data_path, data_type, semaphore_treshold, is_info, precomputed_psi = request.param
    has_precomputed_psi = False
    if precomputed_psi is not None:
        has_precomputed_psi = True
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    res = valtest_psi_mmlu_answers(sampler=sampler,
                                   data_type=data_type,
                                   semaphore_treshold=semaphore_treshold,
                                   is_info=is_info,
                                   answer_psi_values=precomputed_psi)
        
    return res, is_info, has_precomputed_psi

def test_valtest_psi_mmlu_answers_res(create_sampler):
    test_res, is_info, has_precomputed_psi = create_sampler
    assert isinstance(test_res, dict)
    assert set(test_res.keys()) == {'report', 'precomputed'}

def test_valtest_psi_mmlu_answers_report(create_sampler):
    test_res, is_info, has_precomputed_psi = create_sampler
    color = 'gray' if is_info else 'green' if not has_precomputed_psi else 'yellow'
    report = test_res['report']
    result_df = report['result_dataframes'][0]
    assert isinstance(report, dict)
    assert report['semaphore'] == color
    
    if has_precomputed_psi:
        assert result_df['psi_value'].to_list() == [0.3, 0.1, 0.4]
    else:
        assert result_df['psi_value'].to_list()[-1] == 0.0
        
    
def test_valtest_metric_precomputed(create_sampler):
    test_res, is_info, has_precomputed_psi = create_sampler
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'answer_psi_values'}
    
    precomputed_copy = deepcopy(precomputed)
    psi_sum = precomputed_copy['answer_psi_values'].pop('SUM')
    assert psi_sum == sum(precomputed_copy['answer_psi_values'].values())
    
    if has_precomputed_psi:
        assert precomputed['answer_psi_values']['SUM'] == 0.4
    else:
        assert precomputed['answer_psi_values']['SUM'] == 0.0
    