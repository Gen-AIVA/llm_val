import pytest
import joblib
import pandas as pd

from llm_val.sampler.NerSampler import NerSampler
from llm_val.val_test.specification_ner.valtest_entity_description_quality_analysis import valtest_entity_description_quality_analysis

DATA_PATH = 'data/data_ner.pkl'
ENTITY_TYPES = ['Drugname', 'Drugclass', 'Drugform', 'DI', 'ADR', 'Finding']


@pytest.fixture()
def create_test():
    data = joblib.load(DATA_PATH)
    sampler = NerSampler(test=data, entity_types=ENTITY_TYPES)
    res = valtest_entity_description_quality_analysis(sampler)       
    return res


def test_valtest_entity_description_quality_analysis_calc(create_test):
    test_res = create_test
    assert isinstance(test_res, dict)


def test_valtest_entity_description_quality_analysis_keys(create_test):
    test_res = create_test
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_entity_description_quality_analysis_report(create_test):
    test_res = create_test
    report = test_res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)


def test_valtest_entity_description_quality_analysis_precomputed(create_test):
    test_res = create_test
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'metrics_by_description_variantion'}
    
    metrics = precomputed['metrics_by_description_variantion']
    variations = ['extracted', 'extracted_description_detalization', 
                 'extracted_examples_addition', 'extracted_description_detalization_and_examples_addition']
    
    for variation in variations:
        metric_key = f'{variation}_metric'
        assert metric_key in metrics
        assert isinstance(metrics[metric_key], dict)
        assert 'macro_f1' in metrics[metric_key]
        assert isinstance(metrics[metric_key]['macro_f1'], float)
        assert 0 <= metrics[metric_key]['macro_f1'] <= 1