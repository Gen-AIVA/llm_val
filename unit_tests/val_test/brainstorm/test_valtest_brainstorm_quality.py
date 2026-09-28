import pytest
import pandas as pd

from llm_val.utils.Embedder.E5Embedder import E5Embedder as e5e
from llm_val.utils.TokenizerWrapper import TokenizerWrapper as towr
from llm_val.sampler.BrainStormSampler import BrainStormSampler as bss
from llm_val.utils.Embedder.SBERTEmbedder import SBERTEmbedder as sberte
from llm_val.val_test.brainstorm.valtest_brainstorm_quality import valtest_brainstorm_quality as vbq

GLOBAL_CONFIG = 'data/brainstorm/configs/global_config.json'
FULL_DS_TASK_CONFIG = 'data/brainstorm/configs/full_task_config.json'
NO_SCORE_DS_TASK_CONFIG = 'data/brainstorm/configs/no_score_task_config.json'
NO_THEME_DS_TASK_CONFIG = 'data/brainstorm/configs/no_theme_task_config.json'
SIMPLE_DS_TASK_CONFIG = 'data/brainstorm/configs/simple_task_config.json'
FULL_DS = 'data/brainstorm/datasets/micro_full_ds.json'
NO_SCORE_DS = 'data/brainstorm/datasets/micro_no_score_ds.json'
NO_THEME_DS = 'data/brainstorm/datasets/micro_no_theme_ds.json'
SIMPLE_DS = 'data/brainstorm/datasets/micro_simple_ds.json'
E5_PATH = 'data/intfloat_multilingual-e5-large'
SBERT_PATH = 'data/sbert_large_mt_nlu_ru'
ENCODE_CONFIG = {
    'padding': True,
    'truncation': True,
    'max_length': 512,
    'return_tensors': 'pt'
}


@pytest.fixture()
def full_bss():
    bss_ = bss(GLOBAL_CONFIG, FULL_DS)
    return bss_


@pytest.fixture()
def no_score_bss():
    bss_ = bss(GLOBAL_CONFIG, NO_SCORE_DS)
    return bss_


@pytest.fixture()
def no_theme_bss():
    bss_ = bss(GLOBAL_CONFIG, NO_THEME_DS)
    return bss_


@pytest.fixture()
def simple_bss():
    bss_ = bss(GLOBAL_CONFIG, SIMPLE_DS)
    return bss_


@pytest.fixture()
def models():
    e5_towr_ = towr(E5_PATH)
    sbert_towr_ = towr(SBERT_PATH)
    e5e_ = e5e(E5_PATH, ENCODE_CONFIG)
    sberte_ = sberte(SBERT_PATH, ENCODE_CONFIG)
    models = {'e5': {'model': e5e_, 'tokenizer': e5_towr_},
              'sbert': {'model': sberte_, 'tokenizer':  sbert_towr_}}
    return models


def test_valtest_brainstorm_quality_full_ds(full_bss, models):
    test_result = vbq(full_bss,
                      GLOBAL_CONFIG,
                      FULL_DS_TASK_CONFIG,
                      models,
                      True)
    assert isinstance(test_result, dict)
    assert 'report' in test_result.keys()
    assert 'precomputed' in test_result.keys()
    assert isinstance(test_result['report'], dict)
    assert {'semaphore',
            'result_plots',
            'result_dataframes'} == set(list(test_result['report'].keys()))
    assert {'all_raw_data',
            'info_strings',
            'all_plots',
            'full_table'} == set(list(test_result['precomputed'].keys()))
    assert test_result['report']['semaphore'] == 'gray'
    assert isinstance(test_result['report']['result_plots'], list)
    assert len(test_result['report']['result_plots'][0]) == 2
    assert isinstance(test_result['report']['result_dataframes'], pd.DataFrame)
    assert test_result['report']['result_dataframes'].shape[0] == len(test_result['report']['result_plots'])


def test_valtest_brainstorm_quality_no_score(no_score_bss, models):
    test_result = vbq(no_score_bss,
                      GLOBAL_CONFIG,
                      NO_SCORE_DS_TASK_CONFIG,
                      models,
                      True)
    assert isinstance(test_result, dict)
    assert 'report' in test_result.keys()
    assert 'precomputed' in test_result.keys()
    assert isinstance(test_result['report'], dict)
    assert set(['semaphore',
                'result_plots',
                'result_dataframes']) == set(list(test_result['report'].keys()))
    assert {'all_raw_data',
            'info_strings',
            'all_plots',
            'full_table'} == set(list(test_result['precomputed'].keys()))
    assert test_result['report']['semaphore'] == 'gray'
    assert isinstance(test_result['report']['result_plots'], list)
    assert len(test_result['report']['result_plots'][0]) == 1
    assert isinstance(test_result['report']['result_dataframes'], pd.DataFrame)
    assert test_result['report']['result_dataframes'].shape[0] == len(test_result['report']['result_plots'])


def test_valtest_brainstorm_quality_no_theme(no_theme_bss, models):
    test_result = vbq(no_theme_bss,
                      GLOBAL_CONFIG,
                      NO_THEME_DS_TASK_CONFIG,
                      models,
                      True)
    assert isinstance(test_result, dict)
    assert 'report' in test_result.keys()
    assert 'precomputed' in test_result.keys()
    assert isinstance(test_result['report'], dict)
    assert {'semaphore',
            'result_plots',
            'result_dataframes'} == set(list(test_result['report'].keys()))
    assert {'all_raw_data',
            'info_strings',
            'all_plots',
            'full_table'} == set(list(test_result['precomputed'].keys()))
    assert test_result['report']['semaphore'] == 'gray'
    assert isinstance(test_result['report']['result_plots'], list)
    assert len(test_result['report']['result_plots'][0]) == 1
    assert isinstance(test_result['report']['result_dataframes'], pd.DataFrame)
    assert test_result['report']['result_dataframes'].shape[0] == len(test_result['report']['result_plots'])


def test_valtest_brainstorm_quality_simple(simple_bss, models):
    test_result = vbq(simple_bss,
                      GLOBAL_CONFIG,
                      SIMPLE_DS_TASK_CONFIG,
                      models,
                      True)
    assert isinstance(test_result, dict)
    assert 'report' in test_result.keys()
    assert 'precomputed' in test_result.keys()
    assert isinstance(test_result['report'], dict)
    assert {'semaphore',
            'result_plots',
            'result_dataframes'} == set(list(test_result['report'].keys()))
    assert {'all_raw_data',
            'info_strings',
            'all_plots',
            'full_table'} == set(list(test_result['precomputed'].keys()))
    assert test_result['report']['semaphore'] == 'gray'
    assert isinstance(test_result['report']['result_plots'], list)
    assert len(test_result['report']['result_plots'][0]) == 1
    assert isinstance(test_result['report']['result_dataframes'], pd.DataFrame)
    assert test_result['report']['result_dataframes'].shape[0] == len(test_result['report']['result_plots'])
