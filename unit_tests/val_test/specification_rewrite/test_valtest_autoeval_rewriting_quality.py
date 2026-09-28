import pytest
import joblib

from llm_val.utils.api import GigaChatAPI
from llm_val.sampler.RewriteSampler import RewriteSampler
from llm_val.val_test.specification_rewrite.valtest_autoeval_rewriting_quality import valtest_autoeval_rewriting_quality

DATA_PATH = 'data/ru_paraphraser.pkl'


@pytest.fixture()
def create_embedding_model(request, api_single_param):
    embedding_model = GigaChatAPI(**api_single_param)
    return embedding_model


def test_valtest_autoeval_rewriting_quality_res(create_embedding_model):
    embedding_model = create_embedding_model
    data_paraphrase= joblib.load(DATA_PATH)
    sampler_paraphrase = RewriteSampler(data_paraphrase)
    res = valtest_autoeval_rewriting_quality(sampler=sampler_paraphrase,
                                             embedding_model=embedding_model,
                                             semantic_sim_func='cosine_similarity',
                                             lexical_diff_func='levenshtein_distance',
                                             data_type='test')
    assert isinstance(res, dict)
    assert set(res.keys()) == {'report', 'precomputed'}


def test_valtest_autoeval_rewriting_quality_report(create_embedding_model):
    embedding_model = create_embedding_model
    data_paraphrase= joblib.load(DATA_PATH)
    sampler_paraphrase = RewriteSampler(data_paraphrase)
    res = valtest_autoeval_rewriting_quality(sampler=sampler_paraphrase,
                                             embedding_model=embedding_model,
                                             semantic_sim_func='cosine_similarity',
                                             lexical_diff_func='levenshtein_distance',
                                             data_type='test')
    report = res['report']
    assert set(report.keys()) == {'semaphore', 'result_plots', 'result_dataframes'}


def test_valtest_autoeval_rewriting_quality_precomputed(create_embedding_model):
    embedding_model = create_embedding_model
    data_paraphrase= joblib.load(DATA_PATH)
    sampler_paraphrase = RewriteSampler(data_paraphrase)
    res = valtest_autoeval_rewriting_quality(sampler=sampler_paraphrase,
                                             embedding_model=embedding_model,
                                             semantic_sim_func='cosine_similarity',
                                             lexical_diff_func='levenshtein_distance',
                                             data_type='test')
    precomputed = res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'cosine_similarity', 'levenshtein_distance', 'rewrite_quality_values'}              
