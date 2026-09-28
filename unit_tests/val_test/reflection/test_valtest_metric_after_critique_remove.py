import pytest
import numpy as np
import pandas as pd

from llm_val.val_test.reflection.valtest_metric_after_critique_remove import valtest_metric_after_critique_removal

class FakeSampler:
    def __init__(self):
        self.test = {'X': {'question': pd.Series(["Вопрос 1", "Вопрос 2", "Вопрос 3"])}}

class FakeAgent:
    def __init__(self, critique_model=None):
        self.critique_model = critique_model
        self.answer_count = 0

    def predict(self, questions):
        answers = []
        for i in range(len(questions)):
            if self.critique_model is not None:
                answers.append(f"Точный и детальный ответ на вопрос {i+1} с анализом и улучшениями")
            else:
                answers.append(f"Базовый ответ на вопрос {i+1} без дополнительного анализа")
        self.answer_count += 1
        return answers

    def get_params(self):
        return {'critique_model': self.critique_model}

    def set_params(self, **kwargs):
        if 'critique_model' in kwargs:
            self.critique_model = kwargs['critique_model']

class FakeCritiqueModel:
    def predict(self, prompts):
        return ["Улучшенный ответ после критического анализа и проверки"] * len(prompts)

def fake_check_correct_answer(question, answers):
    """Фейковая функция проверки корректности ответа - возвращает разные значения"""
    answer = answers[0]
    if ("Точный" in answer and "детальный" in answer and
        "анализ" in answer and "улучшениями" in answer):
        return 1.0
    else:
        return 0.6

def fake_embedding(texts):
    """Фейковая функция эмбеддингов - возвращает разные эмбеддинги"""
    text = texts[0]
    if "Точный" in text or "детальный" in text or "улучшениями" in text:
        return [[[1.0, 2.0, 3.0, 0.5, 0.8, 1.2]]]
    else:
        return [[[0.8, 1.8, 2.8, 0.3, 0.6, 0.9]]]

def test_simple_run_with_embedding():
    """Простой тест с функцией эмбеддингов - функция должна работать и возвращать результат"""
    sampler = FakeSampler()
    critique_model = FakeCritiqueModel()
    agent = FakeAgent(critique_model=critique_model)

    result = valtest_metric_after_critique_removal(sampler=sampler,
                                                   model=agent,
                                                   get_embed_func=fake_embedding,
                                                   data_type='test')

    assert isinstance(result, dict)
    assert result is not None
    assert 'report' in result
    assert 'precomputed' in result
    assert isinstance(result['report'], dict)
    assert 'semaphore' in result['report']
    assert isinstance(result['precomputed'], dict)
    assert 'cosine_similarity' in result['precomputed']

def test_simple_run_with_correct_answer_check():
    """Простой тест с функцией проверки корректности - функция должна работать и возвращать результат"""
    sampler = FakeSampler()
    critique_model = FakeCritiqueModel()
    agent = FakeAgent(critique_model=critique_model)

    result = valtest_metric_after_critique_removal(sampler=sampler,
                                                   model=agent,
                                                   check_correct_answer=fake_check_correct_answer,
                                                   data_type='test')

    assert isinstance(result, dict)
    assert result is not None
    assert 'report' in result
    assert 'precomputed' in result
    assert isinstance(result['report'], dict)
    assert 'semaphore' in result['report']
    assert isinstance(result['precomputed'], dict)
    assert 'baseline_mean' in result['precomputed']
    assert 'removal_mean' in result['precomputed']

def test_stat_checker_false_mode():
    """Тест работы в режиме stat_checker=False"""
    sampler = FakeSampler()
    critique_model = FakeCritiqueModel()
    agent = FakeAgent(critique_model=critique_model)

    result = valtest_metric_after_critique_removal(sampler=sampler,
                                                   model=agent,
                                                   get_embed_func=fake_embedding,
                                                   stat_checker=False,
                                                   data_type='test')

    assert isinstance(result, dict)
    assert 'precomputed' in result
    assert 'cosine_similarity' in result['precomputed']
    assert 'semaphore' in result['report']

def test_critique_removal_effect():
    """Тест что удаление critique действительно влияет на ответы"""
    sampler = FakeSampler()
    critique_model = FakeCritiqueModel()
    agent = FakeAgent(critique_model=critique_model)
    original_params = agent.get_params()

    result = valtest_metric_after_critique_removal(sampler=sampler,
                                                   model=agent,
                                                   check_correct_answer=fake_check_correct_answer,
                                                   data_type='test')

    assert agent.get_params() == original_params
    precomputed = result['precomputed']
    assert precomputed['baseline_mean'] != precomputed['removal_mean']

def test_no_critique_model():
    """Тест с агентом у которого изначально нет critique модели"""
    sampler = FakeSampler()
    agent = FakeAgent(critique_model=None)

    result = valtest_metric_after_critique_removal(sampler=sampler,
                                                   model=agent,
                                                   get_embed_func=fake_embedding,
                                                   data_type='test')

    assert isinstance(result, dict)
    assert 'precomputed' in result
    assert 'cosine_similarity' in result['precomputed']

def test_info_mode():
    """Тест информативного режима (is_info=True)"""
    sampler = FakeSampler()
    critique_model = FakeCritiqueModel()
    agent = FakeAgent(critique_model=critique_model)

    result = valtest_metric_after_critique_removal(sampler=sampler,
                                                   model=agent,
                                                   get_embed_func=fake_embedding,
                                                   is_info=True,
                                                   data_type='test')

    assert isinstance(result, dict)
    assert result['report']['semaphore'] == 'gray'
