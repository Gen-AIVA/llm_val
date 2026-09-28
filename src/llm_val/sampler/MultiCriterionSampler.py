import typing as tp
import pandas as pd
from llm_val.sampler.Sampler import Sampler


class MultiCriterionSampler(Sampler):
    '''
    Пример реализации sampler'а для датасетов с несколькими критериями оценки.
    '''
    def __init__(self,
                 data: pd.DataFrame):
        test = self._preproc(data)
        super().__init__(test=test)

    def _preproc(self, var: pd.DataFrame) -> tp.Dict[str, pd.DataFrame]:
        CRITERIA_COLS = ['Корректность навигации',
                         'Корректность форматирования',
                         'Корректность математики',
                         'Корректность ответа']
        X = pd.DataFrame()
        y = pd.DataFrame()
        X['system'] = None
        X['question'] = var['Вопрос']
        X['true_answer'] = None
        X['raw_prompt'] = None
        X['model_answer'] = var['Ответ агента']
        for criterion in CRITERIA_COLS:
            y[criterion] = var[criterion]
        y["final_metric"] = y.min(axis=1)
        return {'X': X, 'y': y}