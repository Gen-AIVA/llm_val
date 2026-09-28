import typing as tp
from rouge import Rouge
from bert_score import score
from nltk import word_tokenize
from nltk.translate import meteor
from nltk.translate.bleu_score import (SmoothingFunction,
                                       sentence_bleu)


def rouge_score(references: tp.Iterable[str],
                predictions: str,
                tokenizer: tp.Callable[[str], tp.List[str]]=word_tokenize) -> float:
    '''
    Функция для посчета метрики rouge при наличии нескольких референсных ответов
    Parameters:
        references: Набор корректных ответов
        predictions: Набор ответов модели
        tokenizer: Токенизатор, очищающий строку и разбивающий ее на слова
    Return:
        Максимальное значение метрики rouge при попарном сранении референса и ответа модели
    '''
    references = [' '.join(tokenizer(r)) for r in references]
    predictions = ' '.join(tokenizer(predictions))
    return max(Rouge().get_scores(reference, predictions)[0]['rouge-1']['f'] for reference in references)


def bleu_score(references: tp.Iterable[str],
               predictions: str,
               tokenizer: tp.Callable[[str], tp.List[str]]=word_tokenize) -> float:
    '''
    Функция для посчета метрики bleu при наличии нескольких референсных ответов
    Parameters:
        references: Набор корректных ответов
        predictions: Набор ответов модели
        tokenizer: Токенизатор, очищающий строку и разбивающий ее на слова
    Return:
        Значение метрики bleu при попарном сранении референса и ответа модели
    '''
    references = [' '.join(tokenizer(r)) for r in references]
    predictions = ' '.join(tokenizer(predictions))
    return sentence_bleu(references, predictions, smoothing_function=SmoothingFunction().method1)


def meteor_score(references: tp.Iterable[str],
                 predictions: str,
                 tokenizer: tp.Callable[[str], tp.List[str]]=word_tokenize) -> float:
    '''
    Функция для посчета метрики meteor при наличии нескольких референсных ответов
    Parameters:
        references: Набор корректных ответов
        predictions: Набор ответов модели
        tokenizer: Токенизатор, очищающий строку и разбивающий ее на слова
    Return:
        Значение метрики meteor при попарном сравнении референса и ответа модели
    '''
    references = [tokenizer(r) for r in references]
    predictions = tokenizer(predictions)
    return meteor(references, predictions)


def bert_score_score(references: tp.Iterable[str],
                     predictions: str,
                     tokenizer: tp.Callable[[str], tp.List[str]]=word_tokenize) -> float:
    '''
    Функция для посчета метрики bert_score
    Parameters:
        references: Набор корректных ответов
        predictions: Набор ответов модели
        tokenizer: Токенизатор, очищающий строку и разбивающий ее на слова
    Return:
        Значение метрики meteor при попарном сравнении референса и ответа модели
    '''
    references = [tokenizer(r) for r in references]
    predictions = tokenizer(predictions)
    return score(references, predictions, lang='ru')[2].item()


metrics2func = {'bleu': bleu_score,
                'rouge': rouge_score,
                'meteor': meteor_score}

metrics2func_agent = {'bleu': bleu_score,
                      'rouge': rouge_score,
                      'meteor': meteor_score,
                      'bert_score': bert_score_score}
