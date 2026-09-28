import torch
import jiwer
import numpy as np
import pandas as pd
import typing as tp
from tqdm import tqdm
import seaborn as sns
from copy import deepcopy

from rouge import Rouge
from bert_score import score
from nltk import word_tokenize
from nltk.translate import meteor

from llm_val.sampler.Sampler import Sampler

def report_valtest_summarization_quality(summarization_quality: pd.DataFrame) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        summarization_quality: Подсчитанные метрики качества суммаризации
    Return:
        report в соответствии с концепцией llm_val
    ''' 
    metric_cols = ['WIL', 'Rouge-L', 'Meteor', 'Bert_score', 'LLM_as_judge_score']
    result_dataframe = pd.DataFrame(summarization_quality[metric_cols].mean(), columns=['Значение'])
    result_dataframe.index.name = 'Наименование метрики'
    result_dataframe = result_dataframe.apply(lambda x: round(x, 2))

    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_dataframes'] = [result_dataframe]
    
    return res_dict


def valtest_summarization_quality(sampler: Sampler,
                                  tokenizer: tp.Any=word_tokenize,
                                  func_llm_score: tp.Optional=None,
                                  model_judge: tp.Optional=None,
                                  data_type: str='test') -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на оценку качества суммаризации истории общения пользователя с агентом. 
    Рассчитываются следующие метрики: WIL, Rouge-l, Meteor, Bert_score и оценка через подход "LLM as judge", сопровождающаяся комментарием.
    Parameters:
        sampler: Объект класса Sampler
        tokenizer: Модель-токенизатор с методом tokenize
        func_llm_score: Функция для получения оценки LLM as judge. Ожидается, что функция:
            - принимает на вход строку с исходной историей, строку с суммаризированной историей и модель для оценки
            - возвращает пару (score: int, comment: str)
        model_judge: Модель для получения оценки LLM as judge имеет метод predict
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    X = deepcopy(getattr(sampler, data_type)['X'][['request_id', 'order_idx', 'question', 'model_answer', 'history_summarization']])
    X.sort_values(by=['request_id', 'order_idx'], ascending=True, inplace=True)
    X.reset_index(drop=True, inplace=True)
    
    X['temp'] = '#Вопрос пользователя:# ' + X['question'] + ' #Ответ помощника:# ' + X['model_answer'] + ' '
    X['history_raw'] = X.groupby(by='request_id')['temp'].transform(
        lambda col: col.cumsum()
    )
    
    agg_dict = {
        'history_raw': ('history_raw', 'last'),
        'history_summarization': ('history_summarization', 'last')
    }
    X_grouped = X.groupby(by='request_id', as_index=False).agg(**agg_dict)
    
    summarization_quality = X_grouped.apply(
        lambda row: get_summarization_quality(*row, tokenizer, func_llm_score, model_judge),
        axis=1,
        result_type='expand'
    )

    report = report_valtest_summarization_quality(summarization_quality)
    precomputed = {'summarization_quality': summarization_quality}
    
    return {'report': report,
            'precomputed': precomputed}


def get_summarization_quality(request_id: int,
                              history_raw: str,
                              history_summarization: str,
                              tokenizer: tp.Any,
                              func_llm_score: tp.Optional=None,
                              model_judge: tp.Optional=None) -> tp.Dict[str, tp.Union[int, float, str]]:
    '''
    Подсчёт метрик качества суммаризации. 
    Parameters:
        request_id: id диалога пользователь-агент
        history_raw: Фактическая история диалога
        history_summarization: Суммаризированная история диалога
        tokenizer: Модель-токенизатор   
        func_llm_score: Функция для получения оценки LLM as judge
        model_judge: Модель для получения оценки llm-as-judge, имеет метод .predict()
    Return:
        Значения метрик и светофор для рассматриваемого request_id
    '''
    transformation = jiwer.Compose([jiwer.ToLowerCase(),
                                    jiwer.RemovePunctuation(),
                                    jiwer.RemoveSpecificWords(['вопрос пользователя', 'ответ помощника']),
                                    jiwer.RemoveMultipleSpaces(),
                                    jiwer.Strip()])
    
    history_raw_preproc, history_summarization_preproc = transformation(history_raw), transformation(history_summarization)

    wil_score = jiwer.wil(history_raw_preproc, history_summarization_preproc)
    rouge_score = Rouge().get_scores(history_raw_preproc, history_summarization_preproc)[0]['rouge-l']['f']
    meteor_score = meteor([tokenizer(history_raw_preproc)], tokenizer(history_summarization_preproc))
    bert_score = score([history_raw_preproc], [history_summarization_preproc], lang='ru')[2].item()
    
    if model_judge is not None:
        llm_as_judge_score, llm_as_judge_comment = func_llm_score(history_raw,
                                                                  history_summarization,
                                                                  model_judge)
    else:
        llm_as_judge_score, llm_as_judge_comment = None, None

    return {
        'request_id': request_id,
        'WIL': wil_score,
        'Rouge-L': rouge_score,
        'Meteor': meteor_score,
        'Bert_score': bert_score,
        'LLM_as_judge_score': llm_as_judge_score,
        'LLM_as_judge_comment': llm_as_judge_comment,
        'semaphore': 'gray'
    }