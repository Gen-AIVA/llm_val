import pytest
import pandas as pd

from llm_val.sampler.BrainStormSampler import BrainStormSampler as bss

GLOBAL_CONFIG = 'data/brainstorm/configs/global_config.json'
FULL_DS = 'data/brainstorm/datasets/micro_full_ds.json'
NO_SCORE_DS = 'data/brainstorm/datasets/micro_no_score_ds.json'
NO_THEME_DS = 'data/brainstorm/datasets/micro_no_theme_ds.json'
SIMPLE_DS = 'data/brainstorm/datasets/micro_simple_ds.json'


@pytest.fixture()
def all_cases_dict():
    case_dict = {'full': FULL_DS, 'no_score': NO_SCORE_DS,
                 'no_theme': NO_THEME_DS, 'simple': SIMPLE_DS}
    result_dicts = dict()
    for key, value in case_dict.items():
        data = pd.read_json(value)
        data = data.to_dict(orient='list')
        cur_dict = {'X': dict(), 'y': dict()}
        cur_dict['X']['task'] = data['task']
        cur_dict['X']['ideas'] = data['ideas']
        has_score = 'score' in data.keys()
        has_theme = 'theme' in data.keys()
        if has_score:
            cur_dict['X']['score'] = data['score']
        else:
            pass
        if has_theme:
            cur_dict['X']['theme'] = data['theme']
        else:
            pass
        result_dicts[key] = cur_dict
    return result_dicts


@pytest.fixture()
def bss_json():
    bss_json = dict()
    bss_json['full'] = bss(GLOBAL_CONFIG, FULL_DS)
    bss_json['no_score'] = bss(GLOBAL_CONFIG, NO_SCORE_DS)
    bss_json['no_theme'] = bss(GLOBAL_CONFIG, NO_THEME_DS)
    bss_json['simple'] = bss(GLOBAL_CONFIG, SIMPLE_DS)
    return bss_json


def test_brainstorm_full_dict(all_cases_dict):
    bss_ = bss(GLOBAL_CONFIG,
               test=all_cases_dict['full'])
    data_table = bss_.test['X']
    na_contains = dict(data_table.isna().any())
    assert isinstance(data_table, pd.DataFrame)
    assert {'system',
            'context',
            'true_answer',
            'question',
            'model_answer',
            'thema',
            'score'} == set(bss_.test['X'].columns.tolist())
    assert na_contains['system'] == True
    assert na_contains['context'] == True
    assert na_contains['true_answer'] == True
    assert na_contains['question'] == False
    assert na_contains['model_answer'] == False
    assert na_contains['thema'] == False
    assert na_contains['score'] == False


def test_brainstorm_no_score_dict(all_cases_dict):
    bss_ = bss(GLOBAL_CONFIG,
               test=all_cases_dict['no_score'])
    data_table = bss_.test['X']
    na_contains = dict(data_table.isna().any())
    assert isinstance(data_table, pd.DataFrame)
    assert {'system',
            'context',
            'true_answer',
            'question',
            'model_answer',
            'thema',
            'score'} == set(bss_.test['X'].columns.tolist())
    assert na_contains['system'] == True
    assert na_contains['context'] == True
    assert na_contains['true_answer'] == True
    assert na_contains['question'] == False
    assert na_contains['model_answer'] == False
    assert na_contains['thema'] == False
    assert na_contains['score'] == True


def test_brainstorm_no_theme_dict(all_cases_dict):
    bss_ = bss(GLOBAL_CONFIG,
               test=all_cases_dict['no_theme'])
    data_table = bss_.test['X']
    na_contains = dict(data_table.isna().any())
    assert isinstance(data_table, pd.DataFrame)
    assert {'system',
            'context',
            'true_answer',
            'question',
            'model_answer',
            'thema',
            'score'} == set(bss_.test['X'].columns.tolist())
    assert na_contains['system'] == True
    assert na_contains['context'] == True
    assert na_contains['true_answer'] == True
    assert na_contains['question'] == False
    assert na_contains['model_answer'] == False
    assert na_contains['thema'] == True
    assert na_contains['score'] == False


def test_brainstorm_simple_dict(all_cases_dict):
    bss_ = bss(GLOBAL_CONFIG,
               test=all_cases_dict['simple'])
    data_table = bss_.test['X']
    na_contains = dict(data_table.isna().any())
    assert isinstance(data_table, pd.DataFrame)
    assert {'system',
            'context',
            'true_answer',
            'question',
            'model_answer',
            'thema',
            'score'} == set(bss_.test['X'].columns.tolist())
    assert na_contains['system'] == True
    assert na_contains['context'] == True
    assert na_contains['true_answer'] == True
    assert na_contains['question'] == False
    assert na_contains['model_answer'] == False
    assert na_contains['thema'] == True
    assert na_contains['score'] == True


def test_brainstorm_sampler_full_json(bss_json):
    bss_ = bss_json['full']
    data_table = bss_.test['X']
    na_contains = dict(data_table.isna().any())
    assert isinstance(data_table, pd.DataFrame)
    assert {'system',
            'context',
            'true_answer',
            'question',
            'model_answer',
            'thema',
            'score'} == set(bss_.test['X'].columns.tolist())
    assert na_contains['system'] == True
    assert na_contains['context'] == True
    assert na_contains['true_answer'] == True
    assert na_contains['question'] == False
    assert na_contains['model_answer'] == False
    assert na_contains['thema'] == False
    assert na_contains['score'] == False


def test_brainstorm_sampler_no_score_json(bss_json):
    bss_ = bss_json['no_score']
    data_table = bss_.test['X']
    na_contains = dict(data_table.isna().any())
    assert isinstance(data_table, pd.DataFrame)
    assert {'system',
            'context',
            'true_answer',
            'question',
            'model_answer',
            'thema',
            'score'} == set(bss_.test['X'].columns.tolist())
    assert na_contains['system'] == True
    assert na_contains['context'] == True
    assert na_contains['true_answer'] == True
    assert na_contains['question'] == False
    assert na_contains['model_answer'] == False
    assert na_contains['thema'] == False
    assert na_contains['score'] == True


def test_brainstorm_sampler_no_theme_json(bss_json):
    bss_ = bss_json['no_theme']
    data_table = bss_.test['X']
    na_contains = dict(data_table.isna().any())
    assert isinstance(data_table, pd.DataFrame)
    assert {'system',
            'context',
            'true_answer',
            'question',
            'model_answer',
            'thema',
            'score'} == set(bss_.test['X'].columns.tolist())
    assert na_contains['system'] == True
    assert na_contains['context'] == True
    assert na_contains['true_answer'] == True
    assert na_contains['question'] == False
    assert na_contains['model_answer'] == False
    assert na_contains['thema'] == True
    assert na_contains['score'] == False


def test_brainstorm_sampler_simple_json(bss_json):
    bss_ = bss_json['simple']
    data_table = bss_.test['X']
    na_contains = dict(data_table.isna().any())
    assert isinstance(data_table, pd.DataFrame)
    assert {'system',
            'context',
            'true_answer',
            'question',
            'model_answer',
            'thema',
            'score'} == set(bss_.test['X'].columns.tolist())
    assert na_contains['system'] == True
    assert na_contains['context'] == True
    assert na_contains['true_answer'] == True
    assert na_contains['question'] == False
    assert na_contains['model_answer'] == False
    assert na_contains['thema'] == True
    assert na_contains['score'] == True
