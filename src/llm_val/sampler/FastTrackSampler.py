import typing as tp
import pandas as pd

from crowdkit.aggregation import DawidSkene

from llm_val.sampler.Sampler import Sampler


class FastTrackSampler(Sampler):
    '''
    Семплер для данных FastTrack.
    Преобразует wide-формат (assessor_1..N в колонках) в long-формат
    и агрегирует метки асессоров с помощью алгоритма Дэвида-Скина.
    '''
    def __init__(self,
                 test: pd.DataFrame,
                 n_iter: int=100):
        '''
        Parameters:
            test: Тестовый набор данных в wide-формате
            n_iter: Количество итераций Dawid-Skene
        '''
        test = self._preproc(test, n_iter)
        super().__init__(test=test)

    def _preproc(self,
                 df: pd.DataFrame,
                 n_iter: int) -> tp.Dict[str, pd.DataFrame]:
        '''
        Преобразование данных из wide-формата в long-формат
        и агрегация меток асессоров с помощью Dawid-Skene
        Parameters:
            df: Исходный DataFrame в wide-формате
            n_iter: Количество итераций Dawid-Skene
        Return:
            Словарь с ключами X и y
        '''
        df = df.copy()
        df['output_answer'] = df['output_answer'].fillna('').astype(str)
        assessor_groups, other_columns = self._split_columns(df)
        x_data = self._wide_to_long(df, assessor_groups, other_columns)
        metric_columns = [col for col in x_data.columns
                          if col.endswith('_metric')]
        y_data = self._aggregate_dawid_skene(
            x_data, metric_columns, n_iter)
        return {'X': x_data, 'y': y_data}

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

    def _aggregate_dawid_skene(self,
                               x_data: pd.DataFrame,
                               metric_columns: tp.List[str],
                               n_iter: int) -> pd.DataFrame:
        '''
        Агрегация меток асессоров с помощью Dawid-Skene
        Parameters:
            x_data: DataFrame в long-формате
            metric_columns: Список колонок с метриками
            n_iter: Количество итераций Dawid-Skene
        Return:
            DataFrame с агрегированными метками по session_id
        '''
        y_data = pd.DataFrame()
        for metric_col in metric_columns:
            long_df = x_data[['session_id', 'assessor_id', metric_col]].dropna(
                subset=[metric_col])
            long_df = long_df.rename(columns={
                'session_id': 'task',
                'assessor_id': 'worker',
                metric_col: 'label'
            })
            long_df['label'] = long_df['label'].astype(int)
            model = DawidSkene(n_iter=n_iter)
            y_data[metric_col] = model.fit_predict(long_df)
        y_data.index.name = 'session_id'
        return y_data
