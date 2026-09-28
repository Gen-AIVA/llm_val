import os
import pytest
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from llm_val.sampler.RAGSampler import RAGSampler
from llm_val.ragexecutor.RAGExecutor import RAGExecutor
from llm_val.val_test.rag_quality.valtest_rag_quality_by_k_passages import valtest_rag_quality_by_k_passages

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
TOPK_LIST = [0,1,2,3,4]
CHUNK_SIZE_LIST=[600,800,1000]
CHUNK_OVERLAP_LIST=[50,100,150]
DATA_TYPE = 'test'
TEST_BY = 'true_answer'
EVAL_METRIC = 'MeanCosDist'
model_gc = GigaChat(auth_url='https://sm-auth-sd.prom-88-89-apps.ocp-geo.ocp.sigma.sbrf.ru/api/v2/oauth',
                 credentials='INSERT_YOUR_CLIENT_SECRET_HERE',
                 verify_ssl_certs=False,
                 scope='GIGACHAT_API_CORP')


@pytest.fixture()
def ragsampler():
    data_rag = pd.read_pickle(DATASET_PATH)
    ragsampler_ = RAGSampler(data_rag)
    return ragsampler_


@pytest.fixture()
def ragexecutor():
    ragexecutor_ = RAGExecutor(embed_model_path=E5_PATH,
                               encode_config=ENCODE_CONFIG,
                               text_process_config=TEXT_PROCESS_CONFIG,
                               documents_paths=DOCUMENTS_PATH,
                               vectorstore_path=VECTORSTORE_PATH,
                               topK=TOPK,
                               chunk_size=CHUNK_SIZE,
                               chunk_overlap=CHUNK_OVERLAP,
                               topK_list=TOPK_LIST,
                               chunk_size_list=CHUNK_SIZE_LIST,
                               chunk_overlap_list=CHUNK_OVERLAP_LIST,
                               chat_model=model_gc)
    return ragexecutor_


def test_val_rag_quality_by_k_passages(ragsampler,
                                       ragexecutor):
    test_result = valtest_rag_quality_by_k_passages(sampler=ragsampler,
                                                    executor=ragexecutor,
                                                    data_type=DATA_TYPE,
                                                    test_by=TEST_BY,
                                                    eval_metric=EVAL_METRIC)
    assert isinstance(test_result, dict)
    assert 'report' in test_result.keys()
    assert 'precomputed' in test_result.keys()