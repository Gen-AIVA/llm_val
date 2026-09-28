import pytest
import numpy as np

from llm_val.val_test.tools.valtest_metric_after_tools_refine import valtest_metric_after_tools_refine

class FakeSampler:
    def __init__(self):
        self.test = {'X': {'question': ["Вопрос 1", "Вопрос 2"]}}

class FakeAgent:
    def __init__(self,
                 tools=None):
        self.tools = tools or [{"name": "test", "description": "описание"}]
        self.answer_count = 0
    
    def predict(self, 
                questions):
        answers = []
        for i in range(len(questions)):
            if self.answer_count == 0:
                answers.append("Ответ базовый")
            else:
                answers.append("Ответ измененный")
        self.answer_count += 1
        return answers
    

    def get_params(self):
        return {'tools': self.tools}
    

    def set_params(self, **kwargs):
        if 'tools' in kwargs:
            self.tools = kwargs['tools']

class FakeRephraseAgent:
    def predict(self, prompts):
        return ["Новое описание инструмента"]


def fake_rephrase(model, description):
    return "Новое: " + description


def fake_embedding(texts):
    if "базовый" in texts[0]:
        return [[[1.0, 2.0, 3.0]]]
    else:
        return [[[1.1, 2.1, 3.1]]]


def test_simple_run():
    """Простой тест - функция должна работать и возвращать результат"""
    sampler = FakeSampler()
    agent = FakeAgent()
    rephrase_agent = FakeRephraseAgent()
    
    result = valtest_metric_after_tools_refine(sampler=sampler,
                                               model=agent,
                                               rephrase_model=rephrase_agent,
                                               rephrase_tool_description=fake_rephrase,
                                               get_embed_func=fake_embedding,
                                               data_type='test')
    
    assert isinstance(result, dict)
    assert result is not None
    assert 'report' in result
    assert 'precomputed' in result
    assert isinstance(result['report'], dict)
    assert 'semaphore' in result['report']
    assert isinstance(result['precomputed'], dict)


def test_tool_description_change():
    """Тест что описания инструментов действительно меняются"""
    def tracking_rephrase(model, description):
        return "ИЗМЕНЕНО"
    
    sampler = FakeSampler()
    agent = FakeAgent(tools=[{"name": "test", "description": "оригинал"}])
    rephrase_agent = FakeRephraseAgent()
    
    result = valtest_metric_after_tools_refine(sampler=sampler,
                                               model=agent,
                                               rephrase_model=rephrase_agent,
                                               rephrase_tool_description=tracking_rephrase,
                                               get_embed_func=fake_embedding,
                                               data_type='test')
    
    assert isinstance(result, dict)
    assert 'precomputed' in result
    refined_tools = result['precomputed']['refined_tools']
    assert refined_tools[0]['description'] == "ИЗМЕНЕНО"