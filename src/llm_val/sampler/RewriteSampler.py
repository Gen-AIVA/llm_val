import typing as tp
import pandas as pd
from llm_val.sampler.Sampler import Sampler

class RewriteSampler(Sampler):
    def __init__(self, test: pd.DataFrame):
        '''
        test: Тестовый набор данных
        '''
        test = self._preproc(test)
        super().__init__(test=test)

    
    def _preproc(self, var: pd.DataFrame) -> tp.Dict[str, pd.DataFrame]:
        X = pd.DataFrame()
        X['question'] = var['question']
        X['true_answer'] = var['true_answer'] if 'true_answer' in var.columns else None
        X['model_answer'] = var['model_answer'] if 'model_answer' in var.columns else None
        X['reverse_problem_answer'] = None
        X['context'] = None
        X['thema'] = None
        X['system'] = None
        return {'X': X, 'y': None}
    