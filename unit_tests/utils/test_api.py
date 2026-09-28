import time
import pytest
from llm_val.utils.api import GigaChatAPI
from conftest import RAISE_API_PARAMS


@pytest.fixture()
def create_api(api_param):
    return GigaChatAPI(**api_param)


@pytest.fixture(scope='module')
def get_corpus():
    corpus = ['Привет. Я маленькая капибара', 
              'Перемен - требуют наши сердца', 
              'Расстояние, версты, мили. Нас расставили, рассадили',
              'Грохочет гром. Сверкает молния в ночи']
    return corpus


def test_create_api(create_api):
    assert isinstance(create_api, GigaChatAPI)


@pytest.mark.parametrize('raise_param', RAISE_API_PARAMS)
def test_raise_create_api(raise_param):
    with pytest.raises(ValueError):
        GigaChatAPI(**raise_param)


def test_update_token(create_api):
    AUTH_DATA = 'INSERT_YOUR_CLIENT_SECRET_HERE'
    api = create_api
    old_token = api.token
    old_time = api.token_expires_time
    time.sleep(5)
    api.update_token(AUTH_DATA)
    new_token = api.token
    new_time = api.token_expires_time
    assert old_token != new_token
    assert old_time < new_time


def test_is_token_expires(create_api):
    api = create_api
    print(api.is_token_expires())
    assert not api.is_token_expires()


def test_get_list_models(create_api):
    api = create_api
    models = api.get_list_models()
    assert isinstance(models, list)
    assert len(models) != 0


def test_get_answer_gigachat(create_api):
    api = create_api
    messages = [{'role': 'system', 
                 'content': ''},
                {'role': 'user', 
                 'content': 'Тестовое сообщение. Напиши "капибара", если меня понимаешь.'}]
    answers = api.get_answer_gigachat(messages)
    assert isinstance(answers, list)
    assert len(answers) != 0


def test_get_embedding(create_api, get_corpus):
    api = create_api
    emb = api.get_embedding(corpus=get_corpus)
    assert isinstance(emb, list)
    assert len(emb) != 0


def test_get_token_count(create_api, get_corpus):
    api = create_api
    counts = api.get_token_count(corpus=get_corpus)
    assert isinstance(counts, list)
    assert len(counts) != 0


