import typing as tp
import pandas as pd

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.annotation_aggregation import aggregate_dawid_skene


class FastTrackSampler(Sampler):
    '''
    Семплер для данных FastTrack.
    Преобразует широкий формат с отдельными колонками асессоров в длинный
    и агрегирует метки асессоров с помощью алгоритма Дэвида-Скина.
    '''
    def __init__(self,
                 test: pd.DataFrame,
                 n_iter: int=100):
        '''
        Параметры:
            test: Тестовый набор данных в широком формате
            n_iter: Число итераций алгоритма Дэвида–Скина
        '''
        self._n_iter = n_iter
        test = self._preproc(test, n_iter)
        super().__init__(test=test)

    def _preproc(self,
                 df: pd.DataFrame,
                 n_iter: int) -> tp.Dict[str, pd.DataFrame]:
        '''
        Преобразование данных из широкого формата в длинный
        и агрегация меток асессоров методом Дэвида–Скина
        Параметры:
            df: Исходный DataFrame в широком формате
            n_iter: Число итераций алгоритма Дэвида–Скина
        Результат:
            Словарь с ключами X и y
        '''
        df = df.copy()
        df['output_answer'] = df['output_answer'].fillna('').astype(str)
        assessor_groups, other_columns = self._split_columns(df)
        x_data = self._wide_to_long(df, assessor_groups, other_columns)
        metric_columns = [col for col in x_data.columns
                          if col.endswith('_metric')]
        labels = self._aggregate_dawid_skene(x_data, metric_columns, n_iter)
        y_data = pd.DataFrame()
        for metric_col, scores in labels.items():
            y_data[metric_col] = scores
        y_data.index.name = 'session_id'
        return {'X': x_data, 'y': y_data}

    @property
    def final_labels(self) -> pd.DataFrame:
        """Возвращает финальные метки, включая сессии вне корзины для расчёта качества."""
        x_data, y_data = self.test['X'], self.test['y']
        metrics = [column for column in y_data if column in x_data and column.endswith('_metric')]
        omitted = x_data.loc[~x_data['session_id'].isin(y_data.index), metrics]
        metrics = omitted.columns[omitted.notna().any()].tolist()
        if not metrics or y_data.index.name != 'session_id':
            return y_data
        labels = self._aggregate_dawid_skene(x_data, metrics, getattr(self, '_n_iter', 100))
        additional = pd.DataFrame({
            name: scores.loc[~scores.index.isin(y_data.index)].astype(object) for name, scores in labels.items()
        })
        return pd.concat([y_data.astype(object), additional]).rename_axis('session_id')

    def _split_columns(self,
                       df: pd.DataFrame) -> tp.Tuple[tp.Dict, tp.List[str]]:
        '''
        Разделение колонок на группы асессоров и прочие колонки
        Параметры:
            df: Исходный DataFrame
        Результат:
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
        Преобразование широкого формата в длинный
        Параметры:
            df: Исходный DataFrame
            assessor_groups: Словарь с группами колонок по асессорам
            other_columns: Список прочих колонок
        Результат:
            DataFrame в длинном формате с колонкой assessor_id
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
                               n_iter: int) -> tp.Dict[str, pd.Series]:
        '''
        Агрегация меток асессоров методом Дэвида–Скина
        Параметры:
            x_data: DataFrame в длинном формате
            metric_columns: Список колонок с метриками
            n_iter: Число итераций алгоритма Дэвида–Скина
        Результат:
            Агрегированные по session_id метки каждого критерия
        '''
        final_labels = {}
        for metric_col in metric_columns:
            long_df = x_data[['session_id', 'assessor_id', metric_col]].rename(columns={
                'session_id': 'task',
                'assessor_id': 'worker',
                metric_col: 'label'
            })
            final_labels[metric_col] = aggregate_dawid_skene(long_df, n_iter=n_iter, numeric_strings=True)
        return final_labels
