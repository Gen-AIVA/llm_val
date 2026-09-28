import numpy as np
import typing as tp
import pandas as pd
from copy import deepcopy
from datetime import datetime 

from llm_val.sampler.Sampler import Sampler

class AgentMemorySampler(Sampler):
    '''
    Пример реализации sampler'а для данных, содержащих историю общения пользователя и агента.
    '''
    def __init__(self, 
                 data: pd.DataFrame,
                 metric_names: tp.Optional[tp.List[str]]=None):
        '''
        Конструктор класса.
        Parameters:
            data: Исходные данные
            metric_names: Названия колонок с метриками 
        '''
        preproc_data = self._preproc(data, metric_names)
        super().__init__(test=preproc_data)
        
    def _preproc(self, 
                 data: pd.DataFrame, 
                 metric_names: tp.Optional[tp.List[str]]) -> tp.Dict[str, pd.DataFrame]:
        '''
        Преобразование данных для проведения тестов.
        Parameters:
            data: Исходные данные
            metric_names: Названия колонок с метриками 
        Return:
            X (содержит колонки с информацией о диалоге), y (содержит колонки для подсчёта метрик)
        '''
        n_rows, cols = data.shape[0], data.columns
        X = pd.DataFrame()

        X['question'] = data['question']
        X['model_answer'] = data['answer']

        if 'request_id' not in cols:
            request_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            X['request_id'] = data['question'].apply(lambda x: np.random.choice(request_ids))
        else:
            X['request_id'] = data['request_id']

        if 'order_idx' not in cols:
            X['order_idx'] = X.groupby(by='request_id', as_index=False).cumcount() + 1
        else:
            X['order_idx'] = data['order_idx']
        
        if 'reasoning' not in cols:
            X['reasoning'] = data['question'].apply(lambda x: 'Пример внутренних рассуждений агента')
        else:
            X['reasoning'] = data['reasoning']

        if 'history_summarization' not in cols:
            X['history_summarization'] = data['question'].apply(lambda x: 'Пример суммаризации диалога')
        else:
            X['history_summarization'] = data['history_summarization']
        
        if not metric_names:
            y = pd.DataFrame()
            np.random.seed(666)
            y['main_metric'] = np.random.binomial(1, 0.5, n_rows)
            y['another_metric'] = np.random.binomial(1, 0.5, n_rows)
        else:
            y = deepcopy(pd.DataFrame(data[metric_names]))
        
        return {'X': X, 'y': y}