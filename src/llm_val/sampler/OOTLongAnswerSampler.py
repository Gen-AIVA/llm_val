import os
import typing as tp
import pandas as pd
from crowdkit.aggregation import DawidSkene
from llm_val.sampler.Sampler import Sampler


class OOTLongAnswerSampler(Sampler):
    def __init__(self, 
                 test: pd.DataFrame, 
                 oot: pd.DataFrame):
        '''
        test: Тестовый набор данных
        oot: Набор ООТ данных
        '''
        test = self._preproc(test)
        oot = self._preproc(oot)
        super().__init__(test=test, oot=oot)

    
    def _preproc(self, var: pd.DataFrame) -> tp.Dict[str, pd.DataFrame]:
        col_criterion_list = ['Вы согласны с замечаниями и проблемами, описанными в CodeReview?',
                              'Все существенные проблемы перечислены в CodeReview?',
                              'CodeReview не содержит токсичных комментариев (ненормативная лексика, оскорбительные высказывания или намеки)?',
                              'Ответ лаконичен и по существу?',
                              'Ответ структурирован, все параметры раскрыты?',
                              'Как вы оцениваете качество CodeReview?']
        X = pd.DataFrame()
        df = var.groupby('id_task').apply(lambda x: x.iloc[0])[['answer', 'prompt', 'thema']]
        X['system'] = None
        X['question'] = df['prompt']
        X['context'] = None
        X['thema'] = df['thema']
        X['true_answer'] = None
        X['model_answer'] = df['answer']
        criterion_df = self.group_assessor_grade(var, col_criterion_list)
        X = pd.concat([X, criterion_df], axis='columns')
        y = self.create_david_skene(var, col_criterion_list).loc[X.index]
        return {'X': X, 'y': y}


    def group_assessor_grade(self, var: pd.DataFrame, col_list: tp.List[str]):
        return var.groupby('id_task').apply(lambda x: self.create_combined_dicts(x, col_list))


    def create_combined_dicts(self, row, name_list):
        combined = {}
        for val_col in name_list:
            combined[val_col] = dict(zip(row['worker'], row[val_col]))
        return pd.Series(combined)


    def create_david_skene(self, var: pd.DataFrame, col_list: tp.List[str]):
        df = pd.DataFrame()
        var = var.copy()
        var = var.rename({'id_task': 'task'}, axis='columns')
        for name in col_list:
            df_tmp = var[['task', 'worker', name]]
            df_tmp = df_tmp.rename({name: 'label'}, axis='columns')
            model = DawidSkene()
            df[name] = model.fit_predict(df_tmp)
        return df
