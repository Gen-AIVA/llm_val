import pytest
from llm_val.model.GigaModel import GigaModel
from llm_val.model.BaseGigaModel import BaseGigaModel
from llm_val.model.MultiThreadGigaModel import MultiThreadGigaModel


MODELS = [(BaseGigaModel, {}), 
          (MultiThreadGigaModel, {'n_jobs': 8})]


@pytest.fixture(params=MODELS)
def create_model(request, api_param):
    model_class, params = request.param
    params.update(api_param)
    return model_class(model='GigaChat:latest', **params)


def test_create_model(create_model):
    assert isinstance(create_model, GigaModel)


def test_model_predict(create_model):
    model = create_model
    messages = ['Сколько в среднем весит Капибара?',
                'Сколько лет живет капибара?',
                'Как звали капибару по имени Боб?',
                'Расскажи анекдот про капибар',
                'Как понять, что перед тобой стоит капибара?',
                'Напиши стихотворение про капибару']
    ans = model.predict(messages)
    assert isinstance(ans, list)
    assert len(ans) != 0
    assert len(ans) == len(messages)


def test_model_predict_str(create_model):
    model = create_model
    message = 'Сколько весит капибара?'
    ans = model.predict(message)
    assert isinstance(ans, list)
    assert len(ans) != 0