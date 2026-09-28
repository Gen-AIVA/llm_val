import pandas as pd
import typing as tp
from llm_val.sampler.Sampler import Sampler


class BinaryClassSampler(Sampler):
    '''
    Пример реализации sampler'а для датасетов c бинарной классификацией.
    '''
    def __init__(self,
                 test: pd.DataFrame):
        '''
        test: Тестовый набор данных
        '''
        test = self._preproc(test)
        super().__init__(test=test)


    def _preproc(self, var: pd.DataFrame) -> tp.Dict[str, pd.DataFrame]:
        X = pd.DataFrame()
        y = pd.DataFrame()
        X['system'] = None
        X['question'] = var['text']
        X['raw_prompt'] = None
        y['true_answer'] = var['label']
        y['model_answer'] = var['model_answer']
        y['metric_value'] = (y['true_answer'] == y['model_answer']).astype(int)
        return {'X': X, 'y': y}