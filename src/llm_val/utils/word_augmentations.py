import string
import typing as tp
from nltk.tokenize import word_tokenize

from llm_val.utils.human_augmentations import symbol_swap


def swap_word(text: str,
              rng: tp.Any,
              p: float=0.1) -> str:
    '''
    Аугментация строки методом перестановки соседних слов 
    Parameters:
        text: Исходная строка
        rng: Датчик случайных чисел
        p: Вероятность перестановки слова
    Return:
        Аугментированная строка (в которой некоторые слова переставлены)
    '''
    words = word_tokenize(text)
    words = [word for word in words if word != ""]
    aug_idx = []
    pass_fine = 0
    for i, w in enumerate(words[:-1]):
        if rng.random() + pass_fine < p:
            if i-1 in aug_idx or w in string.punctuation or words[i+1] in string.punctuation:
                pass_fine = -1
            else:
                aug_idx.append(i)
                pass_fine = 0
    for i in aug_idx:
        words = symbol_swap(words, i)
    return ' '.join(words)


def batch_swap_word(batch: tp.List[str],
                    rng: tp.Any,
                    p: float=0.1):
    '''
    Функция для аугментации набора строк
    Parameters:
        batch: Набор строк
        rng: Датчик случайных чисел
        p: Вероятность перестановки слова
    Return:
        Аугментированный набор строк
    '''
    rngs = rng.spawn(len(batch))
    return [swap_word(text, rngs[i], p) for i, text in enumerate(batch)]
