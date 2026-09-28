import pytest
import joblib
import pandas as pd

from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.sampler.RuBQSampler import RuBQSampler
from llm_val.sampler.LongAnswerSampler import LongAnswerSampler

from llm_val.val_test.data_quality.valtest_complex_data_quality import valtest_complex_data_quality

DATA_PATH_MMLU = 'data/data_mmlu.pkl'
DATA_PATH_RUBQ = 'data/data_rubq.pkl'
DATA_PATH_LA = 'data/data_longanswer.pkl'

PARAMS = [(MMLUSampler, DATA_PATH_MMLU),
          (RuBQSampler, DATA_PATH_RUBQ)]


@pytest.fixture(params=PARAMS)
def create_sampler(request):
    sampler_class, data_path = request.param
    data = joblib.load(data_path)
    sampler = sampler_class(test=data)
    return sampler


def test_valtest_complex_data_quality(create_sampler):
    sampler = create_sampler
    res = valtest_complex_data_quality(
        sampler=sampler, data_types=('test', 'test'))
    res_dict = res['precomputed']['data_quality_dict']
    assert res['report']['semaphore'] == 'red'
    assert res_dict.keys() == set(['duplicates_train', 'duplicates_test',
                                   'contradictions_train', 'contradictions_test', 'intersections',
                                   'duplicates_train_set', 'duplicates_test_set', 'contradictions_train_set',
                                   'contradictions_test_set', 'intersections_set'])
    assert res_dict['intersections'] == 1.0
    assert res_dict['intersections_set'] == set(range(5))
    assert res_dict['duplicates_train'] == 0.0
    assert res_dict['duplicates_test'] == 0.0
    assert res_dict['duplicates_test_set'] == set()
    assert res_dict['duplicates_train_set'] == set()

def test_valtest_complex_data_quality_duplication_rate(create_sampler):
    sampler = create_sampler
    sampler.test['X'] = pd.concat((sampler.test['X'], sampler.test['X'])).reset_index(drop=True)
    res = valtest_complex_data_quality(sampler=sampler, data_types=('test', 'test'))
    res_dict = res['precomputed']['data_quality_dict']
    assert res_dict['duplicates_train'] == 1.0
    assert res_dict['duplicates_test'] == 1.0
    assert res_dict['duplicates_train_set'] == set([frozenset([0, 5]), frozenset([1, 6]),
                                                    frozenset([2, 7]), frozenset([3, 8]),
                                                    frozenset([4, 9])])


def test_valtest_complex_data_quality_contradiction_rate(create_sampler):
    sampler = create_sampler
    sampler.test['X'] = pd.concat((sampler.test['X'], sampler.test['X'])).reset_index(drop=True)
    data_la1 = joblib.load('data/LongAnswerExample.pkl')
    sampler.test['X']['true_answer'] = data_la1['true_answer'][:sampler.test['X'].shape[0]]
    res = valtest_complex_data_quality(sampler=sampler, data_types=('test', 'test'))
    res_dict = res['precomputed']['data_quality_dict']
    assert res_dict['contradictions_train'] == 1.0
    assert res_dict['contradictions_test'] == 1.0
    assert res_dict['contradictions_train_set'] == set([frozenset([0, 5]), frozenset([1, 6]),
                                                    frozenset([2, 7]), frozenset([3, 8]),
                                                    frozenset([4, 9])])


def test_valtest_complex_data_qualiity_close_questions(create_sampler):
    sampler = create_sampler
    sampler.test['X'] = sampler.test['X'][:5]
    sampler.test['X']['question'][0] = 'Ты GIGACHAT?'
    sampler.test['X']['question'][1] = 'Ты GIGACHAT!'
    sampler.test['X']['question'][2] = 'Ты gIGACHAT'
    sampler.test['X']['question'][3] = 'Ты    \n GIGACHAT'
    sampler.test['X']['question'][4] = 'Ты GIGACHAT'
    res = valtest_complex_data_quality(sampler=sampler, data_types=('test', 'test'))
    res_dict = res['precomputed']['data_quality_dict']
    assert res_dict['duplicates_test'] == 0.4
    assert res_dict['contradictions_test'] == 0.4
    assert res_dict['contradictions_test_set'] == set([frozenset([3, 4])])
    sampler.test['X']['true_answer'][4] = sampler.test['X']['true_answer'][1]
    res = valtest_complex_data_quality(sampler=sampler, data_types=('test', 'test'), max_bit_diff=4)
    res_dict = res['precomputed']['data_quality_dict']
    assert res_dict['contradictions_test'] == 0.6
    assert res_dict['contradictions_test_set'] == set([frozenset([1, 3]), frozenset([3, 4])])
    assert res_dict['duplicates_train'] == 0.6
    assert res_dict['duplicates_train_set'] == set([frozenset([1, 3, 4])])
    res = valtest_complex_data_quality(sampler=sampler, data_types=('test', 'test'), max_bit_diff=20)
    res_dict = res['precomputed']['data_quality_dict']
    assert res_dict['contradictions_test'] == 1.0
    assert res_dict['duplicates_train'] == 1.0


def test_valtest_complex_data_quality_la_sampler():
    data = joblib.load(DATA_PATH_LA)
    sampler = LongAnswerSampler(test=data)
    res = valtest_complex_data_quality(sampler=sampler, data_types=('test', 'test'))
    precomputed = res['precomputed']
    res_dict = precomputed['data_quality_dict']
    assert res['report']['semaphore'] == 'red'
    assert res_dict.keys() == set(['duplicates_train', 'duplicates_test',
                                   'contradictions_train', 'contradictions_test', 'intersections',
                                   'duplicates_train_set', 'duplicates_test_set', 'contradictions_train_set',
                                   'contradictions_test_set', 'intersections_set'])
    assert res_dict['intersections'] == 1.0
    assert res_dict['intersections_set'] == set(range(5))
    assert res_dict['duplicates_train'] == 0.0
    assert res_dict['duplicates_test'] == 0.0
    assert res_dict['duplicates_test_set'] == set()
    assert res_dict['duplicates_train_set'] == set()


def test_valtest_complex_data_quality_la_sampler_contradiction():
    data = joblib.load(DATA_PATH_LA)
    sampler = LongAnswerSampler(test=data)
    sampler.test['X'] = pd.concat((sampler.test['X'], sampler.test['X']))
    sampler.test['X']['Вы согласны с замечаниями и проблемами, описанными в CodeReview?'][0] = {
        'Агулов Андрей Владимирович ': 333.0}
    res = valtest_complex_data_quality(sampler=sampler, data_types=('test', 'test'))
    res_dict = res['precomputed']['data_quality_dict']
    assert res_dict['contradictions_test'] == 2/sampler.test['X'].shape[0]
    assert res_dict['contradictions_train_set'] == set([frozenset([0, 5])])
    assert res_dict['duplicates_test'] == 1.0
    assert res_dict['duplicates_train'] == 1.0
    assert res_dict['duplicates_train_set'] == set([frozenset([0, 5]), frozenset([1, 6]),
                                                    frozenset([2, 7]), frozenset([3, 8]),
                                                    frozenset([4, 9])])
