import pytest
import numpy as np
import pandas as pd
from types import MappingProxyType

from llm_val.utils.brainstorm_metrics import (modified_cosine_similarity,
                                              modified_euclidean_similarity)
from llm_val.utils.Embedder.E5Embedder import E5Embedder
from llm_val.utils.TokenizerWrapper import TokenizerWrapper
from llm_val.utils.Embedder.SBERTEmbedder import SBERTEmbedder
from llm_val.sampler.BrainStormSampler import BrainStormSamplerUpgraded
from llm_val.val_test.brainstorm.valtest_bstorm_best_ideas_nb import valtest_bstorm_best_ideas_nb

TASK_CONFIG_PATH = 'data/brainstorm/configs/pytest_cfg_quality_stability_by_n_ideas.json'
DS_PATH = 'data/brainstorm/datasets/pytest_ds_quality_stability_by_n_ideas.json'
E5_PATH = 'data/intfloat_multilingual-e5-large'
SBERT_PATH = 'data/sbert_large_mt_nlu_ru'
MODEL_CONFIG = MappingProxyType({'padding': True, 'truncation': True, 'max_length': 512, 'return_tensors': 'pt'})


@pytest.fixture()
def model_storage():
    e5_model = E5Embedder(E5_PATH, MODEL_CONFIG)
    e5_tokenizer = TokenizerWrapper(E5_PATH)
    sbert_model = SBERTEmbedder(SBERT_PATH, MODEL_CONFIG)
    sbert_tokenizer = TokenizerWrapper(SBERT_PATH)

    return {'e5': {'model': e5_model, 'tokenizer': e5_tokenizer},
            'sbert': {'model': sbert_model, 'tokenizer': sbert_tokenizer}}


@pytest.fixture()
def bs_sampler():
    data_df = pd.read_json(DS_PATH)
    shuffled_df = data_df.sample(frac=1, random_state=42)
    train_prop = 0.7
    train_df = shuffled_df[:int(train_prop * len(shuffled_df))]
    test_df = shuffled_df[int(train_prop * len(shuffled_df)):]

    return BrainStormSamplerUpgraded(train=train_df, test=test_df)


def test_valtest_bstorm_best_ideas_nb(bs_sampler, model_storage):
    test_results = valtest_bstorm_best_ideas_nb(sampler=bs_sampler,
                                                task_config_path=TASK_CONFIG_PATH,
                                                model_storage=model_storage,
                                                max_ideas=6,
                                                nb_of_scores=3,
                                                seed=42)
    assert isinstance(test_results, dict)
    assert 'report' in test_results.keys()
    assert 'precomputed' in test_results.keys()
    assert isinstance(test_results['report'], dict)
    assert {'semaphore', 'result_plots', 'result_dataframes'} == set(test_results['report'].keys())
    assert {'cached_embeds', 'dist_binning_thresholds', 'seed'} == set(test_results['precomputed'].keys())
    assert test_results['report']['semaphore'] == 'gray'
    assert isinstance(test_results['report']['result_plots'], list)
    assert len(test_results['report']['result_plots'][0]) == 2
    assert isinstance(test_results['report']['result_dataframes'][0], pd.DataFrame)
    assert len(test_results['report']['result_dataframes']) == len(test_results['report']['result_plots'])


def test_valtest_bstorm_best_ideas_nb_modified_cosine():
    vec_a = np.array([1, 0, 0])
    vec_b = np.array([0, 1, 0])
    vec_c = np.array([-1, 0, 0])

    assert modified_cosine_similarity(vec_a, vec_a) == 1.
    assert modified_cosine_similarity(vec_a, vec_b) == 0.5
    assert modified_cosine_similarity(vec_a, vec_c) == 0.


def test_valtest_bstorm_best_ideas_nb_modified_euclidean():
    vec_a = np.array([1, 0, 0])
    vec_b = np.array([0, 1, 0])
    vec_c = np.array([-1, 0, 0])

    assert modified_euclidean_similarity(vec_a, vec_a) == 1.
    assert np.allclose(modified_euclidean_similarity(vec_a, vec_b), 0.29289, atol=1e-6)
    assert modified_euclidean_similarity(vec_a, vec_c) == 0.
