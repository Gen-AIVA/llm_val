import os
import json
import typing as tp
import pandas as pd
from llm_val.sampler.Sampler import Sampler


class TracesSampler2(Sampler):

    def __init__(self, test_base: pd.DataFrame, test_aef: pd.DataFrame):
        '''
        test: Тестовый набор данных
        '''
        test = self._preproc(test_base, test_aef)
        super().__init__(test=test)

    def _preproc(self, var_base: pd.DataFrame, var_aef: pd.DataFrame) -> tp.Dict[str, pd.DataFrame]:
        X = var_base.copy()
        result = {}
        X = X[(~X['query_id'].isna()) & (~X['session_id'].isna())].reset_index(drop = True)
        X['session_id'] = X['query_id']

        result['y'] = pd.DataFrame(X['estimation']).set_index(X['session_id'])

        x_data = X.drop('estimation', axis=1)
        assessor_groups, other_columns = self._split_columns(x_data)
        result['X'] = self._wide_to_long(x_data, assessor_groups, other_columns)

        traces = var_aef.copy()
        traces['start_time_ns'] = pd.to_datetime(traces['starttime'])
        traces['end_time_ns'] = pd.to_datetime(traces['endtime'])
        traces.drop(columns = ['starttime', 'endtime'], inplace = True)
        traces['session_id'] = traces['trace_id']
        traces['llm_prompt_tokens'] = traces['llm_total_token']
        result['traces'] = traces

        return result

    def _split_columns(self,
                    df: pd.DataFrame) -> tp.Tuple[tp.Dict, tp.List[str]]:
        '''
        Разделение колонок на группы асессоров и прочие колонки
        Parameters:
            df: Исходный DataFrame
        Return:
            Кортеж из словаря групп асессоров и списка прочих колонок
        '''
        assessors = {}
        other_columns = []
        for column in df.columns:
            if 'asessor' in column:
                assessor_key = column[column.find('asessor'):]
                assessors[assessor_key] = assessors.get(
                    assessor_key, []) + [column]
            elif column != 'estimation':
                other_columns.append(column)
        return assessors, other_columns

    def _wide_to_long(self,
                      df: pd.DataFrame,
                      assessor_groups: tp.Dict[str, tp.List[str]],
                      other_columns: tp.List[str]) -> pd.DataFrame:
        '''
        Преобразование wide-формата в long-формат
        Parameters:
            df: Исходный DataFrame
            assessor_groups: Словарь с группами колонок по асессорам
            other_columns: Список прочих колонок
        Return:
            DataFrame в long-формате с колонкой assessor_id
        '''
        agg_data = []
        for assessor_key in assessor_groups:
            curr_df = pd.DataFrame()
            for column in assessor_groups[assessor_key]:
                metric_name = column.split('_asessor')[0]
                curr_df[metric_name] = df[column]
            curr_df['assessor_id'] = assessor_key.split('_')[-1]
            agg_data.append(
                pd.concat([curr_df, df[other_columns]], axis=1))
        return pd.concat(agg_data, axis=0).reset_index(drop=True)