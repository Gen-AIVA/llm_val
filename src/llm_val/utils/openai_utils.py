import typing as tp
from openai import OpenAI

import logging
logger = logging.getLogger(__name__)


def get_openai_embeddings(client: tp.Any,
                          corpus: tp.List[str],
                          model: str) -> tp.List[tp.List[float]]:
    '''
    Получает эмбеддинги текстов через OpenAI-совместимый клиент
    Parameters:
        client: OpenAI-совместимый клиент, имеет метод embeddings.create
        corpus: Список текстов для получения эмбеддингов
        model: Название эмбеддинг-модели
    Return:
        Список векторных представлений текстов
    '''
    response = client.embeddings.create(model=model, input=corpus)
    return [item.embedding for item in response.data]


def get_openai_predictions(client: tp.Any,
                           prompts: tp.List[str],
                           model: str,
                           system_prompt: str = "",
                           temperature: float = 0.0,
                           max_tokens: int = 2048) -> tp.List[str]:
    '''
    Получает текстовые предсказания через OpenAI-совместимый клиент
    Parameters:
        client: OpenAI-совместимый клиент, имеет метод chat.completions.create
        prompts: Список пользовательских промптов
        model: Название модели
        system_prompt: Системный промпт (одинаковый для всех вызовов)
        temperature: Температура генерации
        max_tokens: Максимальное количество токенов в ответе
    Return:
        Список текстовых ответов модели
    '''
    results = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        results.append(response.choices[0].message.content or "")
    return results
