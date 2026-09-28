import re
import typing as tp
from nltk.tokenize import sent_tokenize

def get_sentences_repeatness_ratio(texts: tp.List[str]) -> float:
    '''
    Посчитывает долю предложений с повторами фраз
    Parameters:
        texts: Список текстов
    Return: Доля предложений с повторами фраз
    '''
    text_reps = 0
    num_sentences = 0
    for text in texts:
        splitted = sent_tokenize(text, 'russian')
        num_sentences += len(splitted)
        for sent in splitted:
            match = re.search(r'(\b\w+(?:\s+\w+)+)\s+\1', sent)
            if match:
                text_reps += 1
    return text_reps / num_sentences
