import os
import numpy as np
import typing as tp
import pandas as pd
from llm_val.sampler.Sampler import Sampler

class RAGSampler(Sampler):
    def __init__(self, test: pd.DataFrame):
        '''
        test: Тестовый набор данных
        '''
        test = self._preproc(test)
        super().__init__(test=test)

    def _preproc(self, var: pd.DataFrame) -> tp.Dict[str, pd.DataFrame]:
        X, y = pd.DataFrame(), pd.DataFrame()
        X['system'] = var['prompt']
        X['question'] = var['question']
        X['context'] = var['context']
        X['thema'] = var['thema']
        X['true_answer'] = var['true_answer']
        X['model_answer'] = var['model_answer']
        X['accuracy'] = var['accuracy']
        X['completeness'] = var['completeness']
        X['passage_accuracy'] = var['passage_accuracy']
        y['criterion_passage_accuracy'] = X['passage_accuracy'].apply(self.aggregate_scores)
        y['sim_cosine_distance'] = var['sim_cosine_distance']
        y['criterion_accuracy'] = X['accuracy'].apply(lambda x: self.agg_criterion(x))
        y['criterion_completeness'] = X['completeness'].apply(lambda x: self.agg_criterion(x))
        return {'X': X, 'y': y}

    def agg_criterion(self, criterion_dict: tp.Union[tp.Dict[str, float], None]) -> tp.Union[float, int, None]:
        if isinstance(criterion_dict, dict) and len(criterion_dict.values()) > 0:
            values = [value for value in criterion_dict.values()]
            return np.mean(values)
        else:
            return None

    def aggregate_scores(self, passage_accuracy: tp.Dict[tp.Union[str, int], tp.List[int]]) -> tp.List[int]:
        num_passages = len(next(iter(passage_accuracy.values())))
        aggregated = [1] * num_passages

        for scores in passage_accuracy.values():
            for i, score in enumerate(scores):
                if score == 0:
                    aggregated[i] = 0
        return aggregated
