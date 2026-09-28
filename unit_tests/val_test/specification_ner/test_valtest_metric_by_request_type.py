import pytest
import joblib
import pandas as pd

from llm_val.sampler.NerSampler import NerSampler
from llm_val.val_test.specification_ner.valtest_metric_by_request_type import valtest_metric_by_request_type

DATA_PATH = 'data/data_ner.pkl'
ENTITY_TYPES = ['Drugname', 'Drugclass', 'Drugform', 'DI', 'ADR', 'Finding']


@pytest.fixture()
def create_test():
    data = joblib.load(DATA_PATH)
    sampler = NerSampler(test=data, entity_types=ENTITY_TYPES)
    res = valtest_metric_by_request_type(sampler)       
    return res


def test_valtest_metric_by_request_type_calc(create_test):
    test_res = create_test
    assert isinstance(test_res, dict)


def test_valtest_metric_by_request_type_keys(create_test):
    test_res = create_test
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_metric_by_request_type_report(create_test):
    test_res = create_test
    report = test_res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)
    assert report['semaphore'] == 'gray'


def test_valtest_metric_by_request_type_precomputed(create_test):
    test_res = create_test
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'metrics_by_request_type'}
    assert set(precomputed['metrics_by_request_type'].keys()) == {'extracted_all_types_metric',
                                                                  'extracted_by_type_merged_metric'}