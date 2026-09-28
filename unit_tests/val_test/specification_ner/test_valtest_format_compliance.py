import pytest
import joblib
import pandas as pd

from llm_val.sampler.NerSampler import NerSampler
from llm_val.val_test.specification_ner.valtest_format_compliance import valtest_format_compliance

DATA_PATH = 'data/data_ner.pkl'
ENTITY_TYPES = ['Drugname', 'Drugclass', 'Drugform', 'DI', 'ADR', 'Finding']


@pytest.fixture()
def create_test():
    data = joblib.load(DATA_PATH)
    sampler = NerSampler(test=data, entity_types=ENTITY_TYPES)
    res = valtest_format_compliance(sampler)       
    return res


def test_valtest_format_compliance_calc(create_test):
    test_res = create_test
    assert isinstance(test_res, dict)


def test_valtest_format_compliance_keys(create_test):
    test_res = create_test
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_format_compliance_report(create_test):
    test_res = create_test
    report = test_res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)
    assert report['semaphore'] == 'gray'


def test_valtest_format_compliance_precomputed(create_test):
    test_res = create_test
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'format_compliance_dict'}
    assert set(precomputed['format_compliance_dict'].keys()) == {'distorted_predicted_ratio', 
                                                                 'distorted_target_ratio',
                                                                 'format_mismatch_ratio'}