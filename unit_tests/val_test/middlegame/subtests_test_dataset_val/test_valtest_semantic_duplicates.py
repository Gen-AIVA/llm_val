import pytest
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from gigachat import GigaChat

from llm_val.utils.metrics import METRICS
from llm_val.sampler.TracesSampler2 import TracesSampler2
from llm_val.scorer.UniversalScorer import UniversalScorer
from llm_val.val_test.middlegame.subtests_test_dataset_val.valtest_exact_duplicates import valtest_exact_duplicates
from llm_val.val_test.middlegame.subtests_test_dataset_val.valtest_semantic_duplicates import valtest_semantic_duplicates

DATA_PATH_1 = 'data/data_traces_short.parquet'
DATA_PATH_2 = 'data/data_ft_full.parquet'
DATA_PATH_3 = 'data/data_ft_full_embeds.npy'

PARAMS = [(0.95, True),
          (0.5, True),
          (0.95, False),
          (0.5, False),]

@pytest.fixture(params=PARAMS)
def create_scorer_sampler(request):
    threshold, is_info = request.param
    data_traces2 = pd.read_parquet(DATA_PATH_1)
    data_ft_full = pd.read_parquet(DATA_PATH_2)
    embeddings = np.load(DATA_PATH_3)
    sampler_traces2 = TracesSampler2(data_ft_full, data_traces2)
    scorer = UniversalScorer(metrics=METRICS)
    exact_duplicates = valtest_exact_duplicates(sampler_traces2)['precomputed']['duplicates']
    res = valtest_semantic_duplicates(sampler = sampler_traces2,
                                      exact_duplicates=exact_duplicates,
                                      data_type ='test',
                                      threshold = threshold,
                                      is_info = is_info,
                                      embeddings = embeddings)
    return res, is_info, threshold


def test_valtest_metric_report(create_scorer_sampler):
    test_res, is_info, threshold = create_scorer_sampler
    report = test_res['report']
    if is_info:
        true_color = 'gray'
    elif threshold > 0.05:
        true_color = 'green'
    else:
        true_color = 'yellow'
    assert report['semaphore'] == true_color


def test_valtest_metric_precomputed(create_scorer_sampler):
    test_res, is_info, greater_is_better = create_scorer_sampler
    precomputed = test_res['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'embeddings', 'similarity', 'duplicate_rate', 'semantic_duplicate_indices', 'semantic_duplicate_pairs'}