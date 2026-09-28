import pytest
import joblib
import pandas as pd

from llm_val.sampler.NerSampler import NerSampler
from llm_val.val_test.specification_ner.valtest_conflicting_predictions_analysis import valtest_conflicting_predictions_analysis

DATA_PATH = 'data/data_ner.pkl'
ENTITY_TYPES = ['Drugname', 'Drugclass', 'Drugform', 'DI', 'ADR', 'Finding']


@pytest.fixture()
def create_test():
    data = joblib.load(DATA_PATH)
    sampler = NerSampler(test=data, entity_types=ENTITY_TYPES)
    res = valtest_conflicting_predictions_analysis(sampler)       
    return res


def test_valtest_conflicting_predictions_analysis_calc(create_test):
    test_res = create_test
    assert isinstance(test_res, dict)


def test_valtest_conflicting_predictions_analysis_keys(create_test):
    test_res = create_test
    assert set(test_res.keys()) == {'report', 'precomputed'}


def test_valtest_conflicting_predictions_analysis_report(create_test):
    test_res = create_test
    report = test_res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert isinstance(report['result_dataframes'][0], pd.DataFrame)


def test_valtest_conflicting_predictions_analysis_precomputed(create_test):
    test_res = create_test
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'conflicting_predictions_dict'}
    assert set(precomputed['conflicting_predictions_dict'].keys()) == {'texts_with_conflicts_ratio',
                                                                       'avg_class_conflict_diff',
                                                                       'avg_text_conflict_ratio',
                                                                       'avg_occurrences'}