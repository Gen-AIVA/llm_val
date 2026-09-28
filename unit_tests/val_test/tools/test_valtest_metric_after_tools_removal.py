import pytest
import numpy as np
import pandas as pd

from llm_val.val_test.tools.valtest_metric_after_tools_removal import valtest_metric_after_tools_removal

FAKE_QUESTIONS = ["Сколько будет 2 + 2?",
                  "Столица Франции?"]

@pytest.fixture()
def create_sampler():
    class FakeSampler:
        def __init__(self):
            self.test = {'X': {'question': FAKE_QUESTIONS}}
    return FakeSampler()

class FakeAgent:
    def __init__(self, 
                 tools=None):
        self.tools = tools or ["tool1"]
        self.temperature = 1.0

    def predict(self, 
                questions):
        if isinstance(questions, str):
            questions = [questions]
        return ["фиксированный ответ"] * len(questions)

    def get_params(self):
        return {'tools': self.tools.copy(), 'temperature': self.temperature}

    def set_params(self, 
                   **kwargs):
        if 'tools' in kwargs:
            self.tools = kwargs['tools']

def get_fake_embedding(texts):
    if isinstance(texts, str):
        texts = [texts]
    emb = np.ones((len(texts), 1024), dtype=np.float32)
    return [emb]

def test_valtest_basic_run(create_sampler):
    """
    Проверка: функция запускается и возвращает корректную структуру.
    """
    agent = FakeAgent(tools=["search", "math"])

    result = valtest_metric_after_tools_removal(sampler=create_sampler,
                                                model=agent,
                                                get_embed_func=get_fake_embedding,
                                                removal_num=2,
                                                data_type='test')

    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 
                                  'precomputed'}
    
    report = result['report']
    assert isinstance(report, dict)
    assert set(report.keys()) == {'semaphore', 
                                  'result_plots', 
                                  'result_dataframes'}
    
    assert report['semaphore'] in ['green', 'yellow', 'red', 'gray']
    assert isinstance(report['result_dataframes'], list)
    
    precomputed = result['precomputed']
    assert isinstance(precomputed, dict)
    assert 'cosine_similarity' in precomputed
    assert 'rougeL' in precomputed

def test_valtest_with_correctness_check(create_sampler):
    """
    Проверка: режим с check_correct_answer работает.
    """
    def check_answer_by_length(question, answers):
        return len(answers[0]) > 0

    agent = FakeAgent(tools=["math"])
    result = valtest_metric_after_tools_removal(sampler=create_sampler,
                                                model=agent,
                                                removal_num=2,
                                                check_correct_answer=check_answer_by_length,
                                                data_type='test')

    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 
                                  'precomputed'}
    
    report = result['report']
    assert isinstance(report, dict)
    assert set(report.keys()) == {'semaphore', 
                                  'result_plots', 
                                  'result_dataframes'}
    
    assert report['semaphore'] in ['green', 'yellow', 'red', 'gray']
    
    precomputed = result['precomputed']
    assert isinstance(precomputed, dict)
    assert 'baseline_mean' in precomputed
    assert 'removal_mean' in precomputed

def test_valtest_stat_checker_false(create_sampler):
    """
    Проверка: режим с stat_checker=False работает.
    """
    agent = FakeAgent(tools=["search", "math"])

    result = valtest_metric_after_tools_removal(sampler=create_sampler,
                                                model=agent,
                                                get_embed_func=get_fake_embedding,
                                                removal_num=2,
                                                data_type='test',
                                                stat_checker=False)

    assert isinstance(result, dict)
    assert set(result.keys()) == {'report', 
                                  'precomputed'}
    
    report = result['report']
    assert report['semaphore'] in ['green', 'yellow', 'red', 'gray']

def test_valtest_sequential_mode(create_sampler):
    """
    Проверка: режим sequential работает.
    """
    agent = FakeAgent(tools=["search", "math", "translate"])
    result = valtest_metric_after_tools_removal(sampler=create_sampler,
                                                model=agent,
                                                get_embed_func=get_fake_embedding,
                                                removal_mode='sequential',
                                                removal_num=2,
                                                data_type='test',
                                                stat_checker=False)

    assert isinstance(result, dict)
    report = result['report']
    assert len(report['result_dataframes']) >= 2

def test_valtest_custom_mode(create_sampler):
    """
    Проверка: режим custom с пользовательским списком инструментов работает.
    """
    agent = FakeAgent(tools=["search", "math", "translate"])
    custom_tools = [["search", "math"],
                    ["search"]]

    result = valtest_metric_after_tools_removal(sampler=create_sampler,
                                                model=agent,
                                                get_embed_func=get_fake_embedding,
                                                removal_mode='custom',
                                                custom_tools_list=custom_tools,
                                                removal_num=2,
                                                data_type='test',
                                                stat_checker=False)

    assert isinstance(result, dict)
    report = result['report']
    assert report['semaphore'] in ['green', 'yellow', 'red', 'gray']

def test_valtest_random_mode(create_sampler):
    """
    Проверка: режим random работает.
    """
    agent = FakeAgent(tools=["search", "math", "translate"])

    result = valtest_metric_after_tools_removal(sampler=create_sampler,
                                                model=agent,
                                                get_embed_func=get_fake_embedding,
                                                removal_mode='random',
                                                removal_num=2,
                                                data_type='test')

    assert isinstance(result, dict)
    report = result['report']
    assert report['semaphore'] in ['green', 'yellow', 'red', 'gray']