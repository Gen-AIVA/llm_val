import re
import pytest
import joblib
import pandas as pd

from llm_val.utils.metrics import METRICS
from llm_val.sampler.RuBQSampler import RuBQSampler
from llm_val.sampler.MMLUSampler import MMLUSampler
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.model.MultiThreadGigaModel import MultiThreadGigaModel

from llm_val.val_test.stability.valtest_mix_fewshot_stability import valtest_mix_fewshot_stability, permute_fewshot, flatten_list

DATA_PATH_RUBQ = 'data/data_rubq.pkl'
DATA_PATH_MMLU = 'data/MMLU_with_fewshots.pkl'
PARAMS = [(RuBQSampler, DATA_PATH_RUBQ, 'rubq', 'single_mean_metric_value', 'single_mean', 'test', (0.01, 0.02), False, 1, None, False),
          (MMLUSampler, DATA_PATH_MMLU, 'mmlu', 'single_mean_metric_value', 'single_mean', 'test', (0.01, 0.02), True, 1, 1, False)]
PERMUTE_PARAMS = [(DATA_PATH_MMLU, False, 2, None),
                  (DATA_PATH_MMLU, True, 2, 2)]

@pytest.fixture(params=PARAMS)
def create_sampler_scorer_model(request, api_single_param):
    sampler_cls, data_path, task_type, main_metric, metric_agg, data_type, semaphore_threshold, \
    permute_answer_options, shot_permut_count, answers_permut_per_shot, is_info = request.param
    scorer = UniversalScorer(metrics=METRICS)
    data = joblib.load(data_path)
    sampler = sampler_cls(test=data)
    model_params = {'n_jobs': 8}
    model_params.update(api_single_param)
    model = MultiThreadGigaModel(**model_params)
    res = valtest_mix_fewshot_stability(sampler,
                                        scorer,
                                        model,
                                        task_type=task_type,
                                        main_metric=main_metric,
                                        data_type=data_type,
                                        metric_agg=metric_agg,
                                        semaphore_threshold=semaphore_threshold,
                                        permute_answer_options=permute_answer_options,
                                        query_parser=query_parser,
                                        shot_permut_count=shot_permut_count,
                                        answers_permut_per_shot=answers_permut_per_shot,
                                        is_info=is_info)    
    return res, is_info


def test_valtest_mix_fewshot_stability_res(create_sampler_scorer_model):
    res, is_info = create_sampler_scorer_model
    assert isinstance(res, dict)
    assert set(res.keys()) == {'report', 'precomputed'}
    
    
def test_valtest_mix_fewshot_stability_report(create_sampler_scorer_model):
    res, is_info = create_sampler_scorer_model
    report = res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}
    assert list(report['result_dataframes'][0].columns) == ['Относительное изменение метрики', 'Светофор']
    
    
def test_valtest_mix_fewshot_stability_precomputed(create_sampler_scorer_model):
    res, is_info = create_sampler_scorer_model
    precomputed = res['precomputed']
    assert isinstance(precomputed, dict)
    assert 'base_metric_value' in precomputed
    assert 'mix_few_shot_metric_value' in precomputed

@pytest.mark.parametrize('permute_params', PERMUTE_PARAMS)
def test_permute_fewshot(permute_params):
    data_path, permute_answer_options, shot_permut_count, answers_permut_per_shot = permute_params
    data = joblib.load(data_path)
    initial_questions_count = data.shape[0]
    final_count = initial_questions_count * shot_permut_count * answers_permut_per_shot if permute_answer_options else initial_questions_count * shot_permut_count 
    data['few_shot_permutations'] = data['few_shot'].apply(permute_fewshot, permute_answer_options=permute_answer_options, 
                                                           shot_permut_count=shot_permut_count, 
                                                           answers_permut_per_shot=answers_permut_per_shot,
                                                           query_parser=query_parser)
    if permute_answer_options:
        data = data.explode('few_shot_permutations').explode('few_shot_permutations')
    else:
        data = data.explode('few_shot_permutations')
        data['few_shot_permutations'] = data['few_shot_permutations'].apply(flatten_list)
    assert data.shape[0] == final_count
    assert all(data['few_shot_permutations'].apply(lambda x: len(x) % 2 == 0) == True)
    

def query_parser(query: str):
    question_start = query.find('Вопрос:') + len('Вопрос:')
    question_end = query.find('Варианты ответов:')
    question = query[question_start: question_end]
    answer_options = re.findall(r'\[([A-Z])\](.+)', query)
    return {'question': question,
            'answer_options': answer_options,
            'question_prefix': 'Вопрос:',
            'answer_options_prefix': 'Варианты ответов:'}