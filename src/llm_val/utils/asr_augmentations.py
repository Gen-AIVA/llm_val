import re
import random
import pymorphy3
import numpy as np
import typing as tp
from nltk.corpus import stopwords

punkts = {'.', ',', ';', '!', '?', '', '...'}
russian_stopwords = stopwords.words('russian')

def asr_punktuation_augmentation(text: str,
                                 rng: tp.Any,
                                 p: float=0.05) -> str:
    '''
    Функция, добавляющая к строке аугментации пунктуации (удаление знаков препинания, замена знаков препинания)
    Parameters:
        text: входная строка текста
        rng: Датчик случайных чисел
        p: вероятность агументации
    Return:
        Строка с внесенными агументациями
    '''
    sentences = re.split(r'(\.{3}|[\.|!|?|;|,])', text)
    for i in range(len(sentences)):
        if (sentences[i] in (punkts - {''})) and (rng.random() < p):
            sentences[i] = random.sample(sorted(punkts - {sentences[i]}), 1)[0]
    return ''.join(sentences)

def asr_normalization(text: str,
                      rng: tp.Any,
                      p: float=0.05)-> str:
    '''
    Добавление к строке аугментаций с нормализацией
    Parameters:
        text: входная строка текста
        rng: Датчик случайных чисел
        p: вероятность агументации
    Return:
        Строка с внесенными агументациями
    '''
    morph_analyzier = pymorphy3.MorphAnalyzer()
    words = re.split(r'([.!?;,\s])', text)
    for i in range(len(words)):
        if (words[i] not in punkts | {' '}) and (rng.random() < p):
            words[i] = morph_analyzier.parse(words[i])[0].normal_form
    return ''.join(words)

def asr_stopwords_augmentation(text: str,
                               rng: tp.Any,
                               p: float=0.05) -> str:
    '''
    Добавление к строке аугментаций c удалением небольших слов, которые могут быть незамечены asr
    Parameters:
        text: входная строка текста
        rng: Датчик случайных чисел
        p: вероятность агументации
    Return:
        Строка с внесенными агументациями
    '''
    words = re.split(r'([.!?;,\s])', text)
    for i in range(len(words)):
        if (words[i] not in punkts | {' '}) and (words[i] in russian_stopwords) and (len(words[i]) < 4):
            if (rng.random() < p):
                words[i] = ''
    return ''.join(words)

def asr_augmentations(text: str,
                      rng: tp.Any,
                      p: float=0.05,
                      use_puktuation_augmentation: bool=False,
                      use_normalization: bool=False,
                      use_stopwords_augmentation: bool=False) -> str:
    '''
    Применение к тексту сразу нескольких аугментаций
    Parameters:
        text: входная строка текста
        rng: Датчик случайных чисел
        p: вероятность агументации
        use_puktuation_augmentation: флаг использования аугментаций пункутации
        use_normalization: флаг использования нормализации 
        use_stopwords_augmentation: флаг использования аугментаций по стоп словам
    Return:
        Строка с внесенными агументациями
    '''
    aug_functions = [asr_punktuation_augmentation, asr_normalization, asr_stopwords_augmentation]
    flag_list =  [use_puktuation_augmentation, use_normalization, use_stopwords_augmentation]
    aug_functions = [func for idx, func in enumerate(aug_functions) if flag_list[idx]]
    for aug_function in aug_functions:
        text = aug_function(text, rng, p)
    return text

def batch_asr_augmentations(batch: tp.List[str],
                            rng: tp.Any,
                            p: float=0.05,
                            use_puktuation_augmentation: bool=True,
                            use_normalization: bool=True,
                            use_stopwords_augmentation: bool=True) -> tp.List[str]:
    '''
    Список предложений
    Parameters:
        batch: Набор предложений (строк)
        rng: Датчик случайных чисел
        p: вероятность агументации
    Return:
        Список с предложениями, к которым применены аугментации
    '''
    augmented_batch = [asr_augmentations(text,
                                         rng,
                                         p,
                                         use_puktuation_augmentation,
                                         use_normalization,
                                         use_stopwords_augmentation) for text in batch]
    return augmented_batch