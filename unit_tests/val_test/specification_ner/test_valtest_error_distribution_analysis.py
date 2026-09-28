import pytest
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from llm_val.sampler.NerSampler import NerSampler
from llm_val.val_test.specification_ner.valtest_error_distribution_analysis import valtest_error_distribution_analysis

DATA_PATH = 'data/data_ner.pkl'
ENTITY_TYPES = ['Drugname', 'Drugclass', 'Drugform', 'DI', 'ADR', 'Finding']


@pytest.fixture(scope="session")
def create_test():
    data = joblib.load(DATA_PATH)
    sampler = NerSampler(test=data, entity_types=ENTITY_TYPES)
    res = valtest_error_distribution_analysis(sampler)       
    return res


def test_valtest_error_distribution_analysis_calc(create_test):
    test_res = create_test
    assert isinstance(test_res, dict)


def test_valtest_error_distribution_analysis_keys(create_test):
    test_res = create_test
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_error_distribution_analysis_report(create_test):
    test_res = create_test
    report = test_res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_plots'][0], plt.Figure)
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)
    assert report['semaphore'] in ['red', 'yellow', 'green', 'gray']


def test_valtest_error_distribution_analysis_precomputed(create_test):
    test_res = create_test
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'confusion_matrix'}
    
    confusion_matrix = precomputed['confusion_matrix']
    assert isinstance(confusion_matrix, dict)
    
    for true_class, predictions in confusion_matrix.items():
        assert true_class in ENTITY_TYPES
        assert isinstance(predictions, dict)
        for pred_class, value in predictions.items():
            assert pred_class in ENTITY_TYPES
            assert isinstance(value, float)
            assert 0 <= value <= 1 