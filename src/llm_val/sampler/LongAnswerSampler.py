import os
import typing as tp
import pandas as pd
from crowdkit.aggregation import DawidSkene
from llm_val.sampler.Sampler import Sampler


class LongAnswerSampler(Sampler):
    def __init__(self, test: pd.DataFrame):
        '''
        test: Тестовый набор данных
        '''
        test = self._preproc(test)
        super().__init__(test=test)

    
    def _preproc(self, var: pd.DataFrame) -> tp.Dict[str, pd.DataFrame]:
        col_criterion_list = ['Вы согласны с замечаниями и проблемами, описанными в CodeReview?',
                              'Все существенные проблемы перечислены в CodeReview?',
                              'CodeReview не содержит токсичных комментариев (ненормативная лексика, оскорбительные высказывания или намеки)?',
                              'Ответ лаконичен и по существу?',
                              'Ответ структурирован, все параметры раскрыты?',
                              'Как вы оцениваете качество CodeReview?']
        X = pd.DataFrame()
        df = var.groupby('id_task').apply(lambda x: x.iloc[0])[['answer', 'prompt', 'thema']]
        few_shot = var.groupby('id_task').apply(lambda x: x.iloc[0]).get('few_shot', [None for _ in range(df.shape[0])])
        X['system'] = None
        X['question'] = df['prompt']
        X['context'] = None
        X['thema'] = df['thema']
        X['true_answer'] = None
        X['raw_prompt'] = df.apply(lambda row: self._parse_all_prompt(row), axis=1)
        X['model_answer'] = df['answer']
        X['few_shot'] = few_shot
        criterion_df = self.group_assessor_grade(var, col_criterion_list)
        X = pd.concat([X, criterion_df], axis='columns')
        y = self.create_david_skene(var, col_criterion_list).loc[X.index]
        return {'X': X, 'y': y}

    
    def _parse_all_prompt(self, row: pd.Series):
        '''
        Парсит весь запрос, который целиком будет передаваться в модель.
        Parameters:
            row: Строка датасета
        Return:
            Запрос в модель
        '''
        prompt = row['prompt']
        return prompt


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
