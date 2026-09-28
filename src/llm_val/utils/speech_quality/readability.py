import re
import numpy as np
import typing as tp
import pandas as pd
import seaborn as sns
from string import punctuation
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from nltk.tokenize import sent_tokenize, word_tokenize


def readability_COLE_index(text: str) -> float:
    '''
    Подсчитывает индекс читаемости Колман — Лиау (Coleman–Liau index) текста
    Parameters: 
        text: Строка
    Return:
        Значение метрики читаемости
    '''
    sentences = sent_tokenize(text)
    words = word_tokenize(re.sub(f'[{re.escape(punctuation)}]', ' ', text))
    clear_text = ''.join(words).lower()
    n_words = len(words) if len(words) > 1 else 1
    n_sentences = len(sentences) if len(sentences) > 1 else 1
    n_letters = len(clear_text)
    avg_words_per_sentence = n_words / float(n_sentences)
    avg_letter_per_word = n_letters / float(n_words)
    COLE = (0.03993 * avg_letter_per_word - 0.4976 / avg_words_per_sentence) * 100  - 5.747
    normalized_COLE = max(0, min(COLE, 16)) / 16 * 100
    return normalized_COLE


def corpus_readability_COLE_index(text_list: tp.List[str]) -> tp.List[float]:
    '''
    Подсчитывает метрику читаемости текста для набора строк и усредняет их
    Parameters:
        text_list: Список строк
    Return:
        Список со списками значений метрик 
    '''
    return [readability_COLE_index(text) for text in text_list]


def plot_readability_confusion_matrix(prompt_readability: tp.List[float],
                                      answer_readability: tp.List[float]) -> tp.Any:
    '''
    Строит матрицу зависимости метрики читаемости ответа модели от значений запроса
    Parameters:
        prompt_readability: Подсчитанная метрика читаемости для запросов 
        answer_readability: Подсчитанная метрика читаемости ответа модели 
    Return:
        График зависимости метрики читаемости ответа модели от значений запроса
    '''
    def round_col(x):
        if x < 10:
            out = '< 10'
        elif x < 20:
            out = '10-20'
        elif x < 30:
            out =  '20-30'
        elif x <= 40:
            out = '30-40'
        else:
            out = '> 40'
        return out 
    
    prompt_readability = np.array([round_col(x) for x in prompt_readability])
    answer_readability = np.array([round_col(x) for x in answer_readability])
    labels = ['< 10', '10-20', '20-30', '30-40', '> 40']
    cm = confusion_matrix(prompt_readability, answer_readability, labels=labels)
    cm = np.multiply(cm, (1 / (cm + 1e-16).sum(axis=0))) * 100
    cm = cm[::-1, :]
    fig = plt.figure(figsize=(9, 7))
    lables_x = [f'{l}\n N = {sum(prompt_readability == str(l))}' for l in labels]
    readbility_heatmap = sns.heatmap(data=cm,
                                     xticklabels=lables_x,
                                     yticklabels=labels[::-1],
                                     vmin=0,
                                     vmax=100,
                                     annot=True,
                                     fmt='.2f',
                                     linewidths=0.5,
                                     cmap=sns.color_palette('rocket_r', as_cmap=True))
    readbility_heatmap.set(xlabel='Prompt \n 1 - COLE', ylabel='Answer \n 1 - COLE')
    plt.title('Readability')
    return fig
