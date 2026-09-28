import pytest
import pandas as pd

from llm_val.sampler.RAGSampler import RAGSampler
from llm_val.ragexecutor.RAGExecutor import RAGExecutor
from llm_val.val_test.rag_quality.valtest_rag_quality_by_length import valtest_rag_quality_by_length

DATASET_PATH = 'data/data_rag.pkl'
E5_PATH = 'data/intfloat_multilingual-e5-large'
ENCODE_CONFIG = {
    'max_length': 512,
    'padding': True,
    'truncation': True,
    'return_tensors': 'pt'
}
TEXT_PROCESS_CONFIG = {
    'use_text_cleaner': True,
    'use_lemmatization': True,
    'language': 'russian',
    'stopwords_path': '../examples/data/brainstorm/ru_stop_words.txt'
}
DOCUMENTS_PATH = '../unit_tests/data/sample_text_rag.txt'
VECTORSTORE_PATH = '../unit_tests/data/rag_vectorstore'
TOPK = 3
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOPK_LIST = [1, 2, 3]
CHUNK_SIZE_LIST = [600, 800, 1000]
CHUNK_OVERLAP_LIST = [50, 100, 150]
DATA_TYPE = 'test'
TEST_BY = 'true_answer'
EVAL_METRIC = 'MeanCosDist'
COS_DIST_THRESHOLD = 0.25


@pytest.fixture()
def ragsampler():
    data_rag = pd.read_pickle(DATASET_PATH)
    ragsampler_ = RAGSampler(data_rag)
    return ragsampler_


@pytest.fixture()
def ragexecutor():
    ragexecutor_ = RAGExecutor(
        embed_model_path=E5_PATH,
        encode_config=ENCODE_CONFIG,
        text_process_config=TEXT_PROCESS_CONFIG,
        documents_paths=DOCUMENTS_PATH,
        vectorstore_path=VECTORSTORE_PATH,
        topK=TOPK,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        topK_list=TOPK_LIST,
        chunk_size_list=CHUNK_SIZE_LIST,
        chunk_overlap_list=CHUNK_OVERLAP_LIST
    )
    return ragexecutor_


def test_valtest_rag_quality_by_length(ragsampler, ragexecutor):
    test_result = valtest_rag_quality_by_length(sampler=ragsampler,
                                                executor=ragexecutor,
                                                data_type=DATA_TYPE,
                                                test_by=TEST_BY,
                                                eval_metric=EVAL_METRIC,
                                                cos_dist_threshold=COS_DIST_THRESHOLD)
    assert isinstance(test_result, dict)
    assert 'report' in test_result.keys()
    assert 'precomputed' in test_result.keys()
    assert isinstance(test_result['report'], dict)
    assert set(['semaphore',
                'result_plots',
                'result_dataframes']) == set(list(test_result['report'].keys()))
    assert set(['baseline_score']) == set(list(test_result['precomputed'].keys()))
    assert test_result['report']['semaphore'] == 'gray'
    assert isinstance(test_result['report']['result_plots'], list)
    assert len(test_result['report']['result_plots']) == 1
    assert isinstance(test_result['report']['result_dataframes'], list)
    assert len(test_result['report']['result_dataframes']) == 3
    assert ['Медианный размер пассажей',
            '2.5%-й квантиль размера пассажей',
            '97.5%-й квантиль размера пассажей',
            'Величина перекрытий',
            'Число релевантных пассажей',
            'MeanCosDist'] == test_result['report']['result_dataframes'][0].columns.tolist()
    assert ['Медианный размер пассажей',
            '2.5%-й квантиль размера пассажей',
            '97.5%-й квантиль размера пассажей',
            'Размер пассажей',
            'Число релевантных пассажей',
            'MeanCosDist'] == test_result['report']['result_dataframes'][1].columns.tolist()
    assert set([('Величина перекрытий', overlap) for overlap in CHUNK_OVERLAP_LIST]) == set(test_result['report']['result_dataframes'][2].columns.tolist())
    assert set([('Размер пассажей', size) for size in CHUNK_SIZE_LIST]) == set(test_result['report']['result_dataframes'][2].index.tolist())