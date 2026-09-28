import pytest
import pandas as pd

from llm_val.ragexecutor.RAGExecutor import RAGExecutor
from llm_val.model.MultiThreadGigaModel import MultiThreadGigaModel
from llm_val.val_test.rag_quality.valtest_unsupervised_rag_quality import valtest_unsupervised_rag_quality

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

QUESTION_GENERATION_PROMPT = '''
Твоя задача - написать фактологический вопрос и ответ с учетом контекста.
На твой фактологический вопрос следует ответить конкретной, краткой фактической информацией из контекста.
Твой фактологический вопрос должен быть сформулирован в том же стиле, что и вопросы, которые пользователи могли бы задавать в поисковой системе.
Старайся использовать более простую речь, чем в предоставленных контекстах. Формулируй свои вопросы будто ты обычный пользователь средних лет из России.
Используй простые и просторечные формулировки.
В вашем фактологическом вопросе НЕ ДОЛЖНО упоминаться что-то вроде "согласно отрывку" или "контекст".

Свой ответ оформляй следующим образом:

Output:::
Вопрос: (сгенерированный фактологический вопрос)
Ответ: (Твой ответ на сгенерированный вопрос)

Твой ответ ДОЛЖЕН содержать поля 'Вопрос' и 'Ответ'. Верни только ОДИН вопрос.
'''

CRITIC_PROMPT = '''
Тебе будут даны вопрос и некоторый контекст.
Твоя задача - оценить, насколько подробно и однозначно можно ответить на вопрос при помощи данного контекста.
Дай ответ по шкале от 1 до 5, где 1 означает абсолютное отсутствие возможности ответить на вопрос при помощи контекста, а 5 означает, что контекст дает возможность подробно и однозначно полкостью ответить на вопрос.

Свой ответ оформляй следующим образом:

Answer:::
Обоснование: (Твое обоснование, почему ты даешь такую оценку)
Оценка: (твоя оценка, число от 1 до 5)

Твой ответ ДОЛЖЕН содержать поля 'Обоснование' и 'Оценка'.

'''

PARAMS = [('test', {'hit_rate_1': (0.3, 0.4), 'hit_rate_5': (0.5, 0.7)}, 2, False)]

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


@pytest.fixture()
def question_generation_model(api_single_param):
    model_params = {'n_jobs': 8,
                    'sys_promt': QUESTION_GENERATION_PROMPT}
    model_params.update(api_single_param)
    model = MultiThreadGigaModel(**model_params)
    return model


@pytest.fixture()
def critic_model(api_single_param):
    model_params = {'n_jobs': 8,
                    'sys_promt': CRITIC_PROMPT}
    model_params.update(api_single_param)
    model = MultiThreadGigaModel(**model_params)
    return model


@pytest.fixture(params=PARAMS)
def run_test(request, ragexecutor, question_generation_model, critic_model):
    data_type, semaphore_threshold, hit_rate_relevance_threshold, is_info = request.param
    test_result = valtest_unsupervised_rag_quality(executor=ragexecutor,
                                                   question_generation_model=question_generation_model,
                                                   data_type=data_type,
                                                   semaphore_threshold=semaphore_threshold,
                                                   critic_model=critic_model,
                                                   hit_rate_relevance_threshold=hit_rate_relevance_threshold,
                                                   is_info=is_info)
    return test_result
    

def test_valtest_unsupervised_rag_quality_res(run_test):
    test_result = run_test
    assert isinstance(test_result, dict)
    assert set(test_result.keys()) == {'report', 'precomputed'}
    

def test_valtest_unsupervised_rag_quality_report(run_test):
    test_result = run_test
    report = test_result['report']
    assert isinstance(report, dict)
    assert set(report['result_dataframes'][0].columns) == {'Метрика', 'Значение', 'Светофор'}
    
    
def test_valtest_unsupervised_rag_quality_precomputed(run_test):
    test_result = run_test
    precomputed = test_result['precomputed']
    assert isinstance(precomputed, dict)
    assert set(precomputed.keys()) == {'hit_rate_1', 'hit_rate_5', 'average_place'}