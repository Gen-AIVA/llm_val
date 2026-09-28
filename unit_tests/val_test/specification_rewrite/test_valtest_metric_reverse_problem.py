import pytest
import joblib

from llm_val.utils.api import GigaChatAPI
from llm_val.sampler.Sampler import Sampler
from llm_val.sampler.RewriteSampler import RewriteSampler
from llm_val.model.MultiThreadGigaModel import MultiThreadGigaModel
from llm_val.val_test.specification_rewrite.valtest_metric_reverse_problem import valtest_metric_reverse_problem


DATA_PATH = 'data/flores-devtest.pkl'


@pytest.fixture()
def create_model_embedding_model_prompts(request, api_single_param):
    embedding_model = GigaChatAPI(**api_single_param)
    model_params = {'n_jobs': 8}
    model_params.update(api_single_param)
    model = MultiThreadGigaModel(**model_params)
    forward_problem_system_prompt = '''
    Тебе будет дан текст на языке хинди. Тебе необходимо перевести его на русский язык. Не пиши ничего, кроме переведённого текста.
    Текст на хинди: {source_text}
    '''
    reverse_problem_system_prompt = '''
    Тебе будет дан текст на русском языке. Тебе необходимо перевести его на хинди. Не пиши ничего, кроме переведённого текста.
    Текст на русском: {source_text}
    '''
    return embedding_model, model, forward_problem_system_prompt, reverse_problem_system_prompt


def test_valtest_metric_reverse_problem_res(create_model_embedding_model_prompts):
    embedding_model, model, forward_problem_system_prompt, reverse_problem_system_prompt = create_model_embedding_model_prompts
    data_rewrite = joblib.load(DATA_PATH)
    sampler = RewriteSampler(data_rewrite)
    
    res = valtest_metric_reverse_problem(sampler=sampler,
                                         model=model,
                                         main_metric='rouge',
                                         use_cosine_similarity=False,
                                         embedding_model=embedding_model,
                                         data_type='test',
                                         semaphore_threshold=(40, 60),
                                         greater_is_better=False,
                                         is_info=False,
                                         forward_problem_system_prompt=forward_problem_system_prompt,
                                         reverse_problem_system_prompt=reverse_problem_system_prompt)
    assert isinstance(res, dict)
    assert set(res.keys()) == {'report', 'precomputed'}

def test_valtest_metric_reverse_problem_report(create_model_embedding_model_prompts):
    embedding_model, model, forward_problem_system_prompt, reverse_problem_system_prompt = create_model_embedding_model_prompts
    data_rewrite = joblib.load(DATA_PATH)
    sampler = RewriteSampler(data_rewrite)
    
    res = valtest_metric_reverse_problem(sampler=sampler,
                                         model=model,
                                         main_metric='rouge',
                                         use_cosine_similarity=False,
                                         embedding_model=embedding_model,
                                         data_type='test',
                                         semaphore_threshold=(40, 60),
                                         greater_is_better=False,
                                         is_info=False,
                                         forward_problem_system_prompt=forward_problem_system_prompt,
                                         reverse_problem_system_prompt=reverse_problem_system_prompt)
    
    report = res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}

def test_valtest_metric_reverse_problem_precomputed(create_model_embedding_model_prompts):
    embedding_model, model, forward_problem_system_prompt, reverse_problem_system_prompt = create_model_embedding_model_prompts
    data_rewrite = joblib.load(DATA_PATH)
    sampler = RewriteSampler(data_rewrite)
    
    res = valtest_metric_reverse_problem(sampler=sampler,
                                         model=model,
                                         main_metric='rouge',
                                         use_cosine_similarity=False,
                                         embedding_model=embedding_model,
                                         data_type='test',
                                         semaphore_threshold=(40, 60),
                                         greater_is_better=False,
                                         is_info=False,
                                         forward_problem_system_prompt=forward_problem_system_prompt,
                                         reverse_problem_system_prompt=reverse_problem_system_prompt)
    
    precomputed = res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'forward_problem_metric_value', 'reverse_problem_metric_value'}

def test_valtest_metric_reverse_problem_rouge_value(create_model_embedding_model_prompts):
    embedding_model, model, forward_problem_system_prompt, reverse_problem_system_prompt = create_model_embedding_model_prompts
    data_rewrite = joblib.load(DATA_PATH)
    sampler = RewriteSampler(data_rewrite)

    
    res = valtest_metric_reverse_problem(sampler=sampler,
                                         model=model,
                                         main_metric='rouge',
                                         use_cosine_similarity=False,
                                         embedding_model=embedding_model,
                                         data_type='test',
                                         semaphore_threshold=(40, 60),
                                         greater_is_better=False,
                                         is_info=False,
                                         forward_problem_system_prompt=forward_problem_system_prompt,
                                         reverse_problem_system_prompt=reverse_problem_system_prompt)
    assert 0 <= res['precomputed']['forward_problem_metric_value']['rouge'] <= 1
    assert 0 <= res['precomputed']['reverse_problem_metric_value']['rouge'] <= 1

def test_valtest_metric_reverse_problem_bleu_value(create_model_embedding_model_prompts):
    embedding_model, model, forward_problem_system_prompt, reverse_problem_system_prompt = create_model_embedding_model_prompts
    data_rewrite = joblib.load(DATA_PATH)
    sampler = RewriteSampler(data_rewrite)

    
    res = valtest_metric_reverse_problem(sampler=sampler,
                                         model=model,
                                         main_metric='bleu',
                                         use_cosine_similarity=False,
                                         embedding_model=embedding_model,
                                         data_type='test',
                                         semaphore_threshold=(40, 60),
                                         greater_is_better=False,
                                         is_info=False,
                                         forward_problem_system_prompt=forward_problem_system_prompt,
                                         reverse_problem_system_prompt=reverse_problem_system_prompt)
    assert 0 <= res['precomputed']['forward_problem_metric_value']['bleu'] <= 1
    assert 0 <= res['precomputed']['reverse_problem_metric_value']['bleu'] <= 1


def test_valtest_metric_reverse_problem_meteor_value(create_model_embedding_model_prompts):
    embedding_model, model, forward_problem_system_prompt, reverse_problem_system_prompt = create_model_embedding_model_prompts
    data_rewrite = joblib.load(DATA_PATH)
    sampler = RewriteSampler(data_rewrite)

    
    res = valtest_metric_reverse_problem(sampler=sampler,
                                         model=model,
                                         main_metric='meteor',
                                         use_cosine_similarity=False,
                                         embedding_model=embedding_model,
                                         data_type='test',
                                         semaphore_threshold=(40, 60),
                                         greater_is_better=False,
                                         is_info=False,
                                         forward_problem_system_prompt=forward_problem_system_prompt,
                                         reverse_problem_system_prompt=reverse_problem_system_prompt)
    assert 0 <= res['precomputed']['forward_problem_metric_value']['meteor'] <= 1
    assert 0 <= res['precomputed']['reverse_problem_metric_value']['meteor'] <= 1

def test_valtest_metric_reverse_problem_cosine_similarity_value(create_model_embedding_model_prompts):
    embedding_model, model, forward_problem_system_prompt, reverse_problem_system_prompt = create_model_embedding_model_prompts
    data_rewrite = joblib.load(DATA_PATH)
    sampler = RewriteSampler(data_rewrite)

    res = valtest_metric_reverse_problem(sampler=sampler,
                                         model=model,
                                         main_metric='cosine_similarity',
                                         use_cosine_similarity=True,
                                         embedding_model=embedding_model,
                                         data_type='test',
                                         semaphore_threshold=(40, 60),
                                         greater_is_better=False,
                                         is_info=False,
                                         forward_problem_system_prompt=forward_problem_system_prompt,
                                         reverse_problem_system_prompt=reverse_problem_system_prompt)
    assert -1 <= res['precomputed']['forward_problem_metric_value']['cosine_similarity'] <= 1
    assert -1 <= res['precomputed']['reverse_problem_metric_value']['cosine_similarity'] <= 1                 
