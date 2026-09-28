import random
import string
import typing as tp

import numpy as np

from llm_val.utils.openai_utils import get_openai_predictions
from llm_val.utils.retries import call_with_retries


DEFAULT_PARAPHRASE_PROMPT = (
    "Перефразируй следующий текст, полностью сохранив его исходный смысл и факты. "
    "Измени структуру предложений и лексику. "
    "В ответе верни ТОЛЬКО перефразированный текст, без кавычек, пояснений или вводных слов."
)

DEFAULT_SYNONYM_PROMPT = (
    "Подбери синоним к указанному слову в контексте предложения. "
    "Верни СТРОГО ОДНО слово, без кавычек, точек и пояснений."
)

DEFAULT_TRANSLATION_FWD_PROMPT = (
    "Переведи следующий текст на указанный язык. "
    "Верни ТОЛЬКО переведенный текст, без кавычек, пояснений или вводных слов."
)

DEFAULT_TRANSLATION_BWD_PROMPT = (
    "Переведи следующий текст на русский язык. "
    "Сохрани исходный смысл и факты. "
    "Верни ТОЛЬКО переведенный текст, без кавычек, пояснений или исходного текста."
)


def batch_paraphrase_api(batch: tp.List[str], 
                         aug_client: tp.Optional[tp.Any] = None,
                         aug_model_name: tp.Optional[str] = None,
                         system_prompt: tp.Optional[str] = None,
                         max_attempts: int = 3,
                         timeout: float = 60.0,
                         retry_delay: float = 0.1) -> tp.List[str]:
    """
    Parameters:
        batch: Список исходных текстов для перефразирования.
        aug_client: OpenAI-совместимый клиент для генерации аугментаций.
        aug_model_name: Название модели для генерации аугментаций.
        system_prompt: Системный промпт для модели аугментации.
                       Если None, используется DEFAULT_PARAPHRASE_PROMPT.
        max_attempts: Максимальное количество попыток перефразирования для 
                      текстов, которые не удалось изменить.
        timeout: Максимальное время ожидания одного обращения к модели в секундах.
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах.
    Returns:
        Список перефразированных текстов. Если перефразирование не удалось, 
        возвращается исходный текст.
    """
    if not batch:
        return batch
    results = list(batch)
    indices_to_process = list(range(len(batch)))
    actual_prompt = system_prompt if system_prompt is not None else DEFAULT_PARAPHRASE_PROMPT
    for attempt in range(max_attempts):
        if not indices_to_process:
            break
        texts_to_aug = [results[i] for i in indices_to_process]
        prompts = [f"Текст: {text}" for text in texts_to_aug]
        responses = call_with_retries(
            get_openai_predictions,
            timeout=timeout,
            retry_delay=retry_delay,
            max_attempts=3,
            func_args=(aug_client, prompts, aug_model_name),
            func_kwargs={'system_prompt': actual_prompt}
        )
        new_indices_to_process = []
        for i, orig_idx in enumerate(indices_to_process):
            original_text = batch[orig_idx].strip()
            new_text = responses[i].strip()
            
            if new_text != original_text and len(new_text) > 0:
                results[orig_idx] = new_text
            else:
                new_indices_to_process.append(orig_idx)
                
        indices_to_process = new_indices_to_process
    return results


def batch_synonym_api(batch: tp.List[str],
                      aug_client: tp.Optional[tp.Any] = None,
                      aug_model_name: tp.Optional[str] = None,
                      rng: tp.Any = None,
                      word_replace_ratio: float = 0.1,
                      system_prompt: tp.Optional[str] = None,
                      max_attempts: int = 3,
                      timeout: float = 60.0,
                      retry_delay: float = 0.1) -> tp.List[str]:
    """
    Parameters:
        batch: Список исходных текстов для аугментации.
        aug_client: OpenAI-совместимый клиент для генерации аугментаций.
        aug_model_name: Название модели для генерации аугментаций.
        rng: Генератор случайных чисел (например, numpy.random), используемый 
             для выбора индексов слов для замены.
        word_replace_ratio: Доля слов в тексте, которые необходимо заменить 
                            на синонимы.
        system_prompt: Системный промпт для модели аугментации.
                       Если None, используется DEFAULT_SYNONYM_PROMPT.
        max_attempts: Максимальное количество попыток аугментации для текстов, 
                      которые не удалось изменить.
        timeout: Максимальное время ожидания одного обращения к модели в секундах.
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах.
    Returns:
        Список текстов с замененными синонимами. Если аугментация не удалась, 
        возвращается исходный текст.
    """
    if not batch:
        return batch
    results = list(batch)
    indices_to_process = list(range(len(batch)))
    actual_prompt = system_prompt if system_prompt is not None else DEFAULT_SYNONYM_PROMPT
    for attempt in range(max_attempts):
        if not indices_to_process:
            break
        tasks = []
        for orig_idx in indices_to_process:
            text = results[orig_idx]
            words = text.split()
            if not words:
                continue
            num_to_replace = max(1, int(len(words) * word_replace_ratio))
            chosen_indices = rng.choice(
                len(words), 
                size=min(num_to_replace, len(words)), 
                replace=False
            ).tolist()
            for word_idx in chosen_indices:
                exact_word = words[word_idx]
                clean_word = exact_word.strip(string.punctuation)
                if not clean_word:
                    continue
                prompt = (
                    f"Слово: '{clean_word}'\n"
                    f"Предложение: {text}"
                )
                tasks.append((orig_idx, word_idx, exact_word, clean_word, prompt))
        if not tasks:
            break
        final_words = [task[3] for task in tasks]
        is_success = [False] * len(tasks)
        for word_attempt in range(max_attempts):
            indices_to_retry = [i for i, success in enumerate(is_success) if not success]
            if not indices_to_retry:
                break
            retry_prompts = [tasks[i][4] for i in indices_to_retry]
            responses = call_with_retries(
                get_openai_predictions,
                timeout=timeout,
                retry_delay=retry_delay,
                max_attempts=3,
                func_args=(aug_client, retry_prompts, aug_model_name),
                func_kwargs={'system_prompt': actual_prompt}
            )
            for i, response in zip(indices_to_retry, responses):
                raw_response = response.strip()
                word_list = raw_response.split()
                if len(word_list) == 1:
                    candidate = word_list[0]
                    cleaned = candidate.strip('"\':;,.!?()[]{}«»`*~')
                    if cleaned:
                        final_words[i] = cleaned
                        is_success[i] = True
        for i, task in enumerate(tasks):
            orig_idx, word_idx, exact_word, clean_word, _ = task
            best_word = final_words[i]
            words = results[orig_idx].split()
            words[word_idx] = _replace_word_preserving_format(exact_word, clean_word, best_word)
            results[orig_idx] = " ".join(words)
        new_indices_to_process = []
        for orig_idx in indices_to_process:
            if results[orig_idx].strip() == batch[orig_idx].strip():
                new_indices_to_process.append(orig_idx)
        indices_to_process = new_indices_to_process
    return results


def _replace_word_preserving_format(exact_word: str,
                                    clean_word: str,
                                    best_word: str) -> str:
    """
    Заменяет тело слова (clean_word) внутри exact_word на best_word,
    сохраняя пунктуацию по краям и регистр первой буквы исходного слова.
    """
    if not clean_word or not best_word:
        return exact_word

    prefix_len = 0
    while prefix_len < len(exact_word) and exact_word[prefix_len] in string.punctuation:
        prefix_len += 1
    suffix_len = 0
    while suffix_len < len(exact_word) - prefix_len and \
            exact_word[len(exact_word) - 1 - suffix_len] in string.punctuation:
        suffix_len += 1

    prefix = exact_word[:prefix_len]
    suffix = exact_word[len(exact_word) - suffix_len:] if suffix_len > 0 else ""

    first_char = clean_word[0]
    if first_char.isupper():
        best_word = best_word[0].upper() + best_word[1:]
    elif first_char.islower():
        best_word = best_word[0].lower() + best_word[1:]

    return prefix + best_word + suffix


def batch_translation_api(batch: tp.List[str],
                          aug_client: tp.Optional[tp.Any] = None,
                          aug_model_name: tp.Optional[str] = None,
                          target_language: str = 'английский',
                          system_prompt_fwd: tp.Optional[str] = None,
                          system_prompt_bwd: tp.Optional[str] = None,
                          max_attempts: int = 3,
                          timeout: float = 60.0,
                          retry_delay: float = 0.1) -> tp.List[str]:
    """
    Parameters:
        batch: Список исходных текстов на русском языке.
        aug_client: OpenAI-совместимый клиент для генерации аугментаций.
        aug_model_name: Название модели для генерации аугментаций.
        target_language: Целевой язык для промежуточного перевода 
                         (по умолчанию 'английский').
        system_prompt_fwd: Системный промпт для прямого перевода.
                           Если None, используется DEFAULT_TRANSLATION_FWD_PROMPT
                           с подстановкой target_language.
        system_prompt_bwd: Системный промпт для обратного перевода.
                           Если None, используется DEFAULT_TRANSLATION_BWD_PROMPT.
        max_attempts: Максимальное количество попыток обратного перевода для 
                      текстов, которые не удалось изменить.
        timeout: Максимальное время ожидания одного обращения к модели в секундах.
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах.
    Returns:
        Список текстов, полученных в результате обратного перевода. Если 
        перевод не привел к изменениям, возвращается исходный текст.
    """
    if not batch:
        return batch
    results = list(batch)
    indices_to_process = list(range(len(batch)))
    actual_prompt_fwd = system_prompt_fwd if system_prompt_fwd is not None \
        else DEFAULT_TRANSLATION_FWD_PROMPT.replace("указанный язык", target_language)
    actual_prompt_bwd = system_prompt_bwd if system_prompt_bwd is not None else DEFAULT_TRANSLATION_BWD_PROMPT
    for attempt in range(max_attempts):
        if not indices_to_process:
            break
        texts_to_aug = [results[i] for i in indices_to_process]
        prompts_fwd = [f"Текст: {text}" for text in texts_to_aug]
        mid_language_texts = call_with_retries(
            get_openai_predictions,
            timeout=timeout,
            retry_delay=retry_delay,
            max_attempts=3,
            func_args=(aug_client, prompts_fwd, aug_model_name),
            func_kwargs={'system_prompt': actual_prompt_fwd}
        )
        prompts_bwd = [f"Текст: {text}" for text in mid_language_texts]
        back_translated_texts = call_with_retries(
            get_openai_predictions,
            timeout=timeout,
            retry_delay=retry_delay,
            max_attempts=3,
            func_args=(aug_client, prompts_bwd, aug_model_name),
            func_kwargs={'system_prompt': actual_prompt_bwd}
        )
        new_indices_to_process = []
        for i, orig_idx in enumerate(indices_to_process):
            original_text = batch[orig_idx].strip()
            new_text = back_translated_texts[i].strip()
            if new_text != original_text and len(new_text) > 0:
                results[orig_idx] = new_text
            else:
                new_indices_to_process.append(orig_idx)
        indices_to_process = new_indices_to_process
    return results