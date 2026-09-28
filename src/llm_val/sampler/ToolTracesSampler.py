import typing as tp
import pandas as pd

from llm_val.sampler.Sampler import Sampler


class ToolTracesSampler(Sampler):
    '''Сэмплер для данных с трейсами, session_id и aef_kind.

    Принимает DataFrame, в котором есть трейсовые колонки
    (traceid, spanid, name, starttimeunixnano, endtimeunixnano),
    корзинные колонки и колонка session_id.

    В y индекс — session_id, чтобы фильтрация по сессиям
    не зависела от взаимного порядка строк X и y.
    В traces также добавляется колонка aef_kind
    для возможности фильтрации по типу инструмента.
    '''

    def __init__(self, test: pd.DataFrame):
        '''Инициализация сэмплера.

        Parameters:
            test: Тестовый набор данных
        '''
        preprocessed = self._preproc(test)
        super().__init__(test=preprocessed)

    def _preproc(self, var: pd.DataFrame) -> tp.Dict[str, pd.DataFrame]:
        '''Преобразует данные, сохраняя session_id и aef_kind в трейсах.

        Parameters:
            var: Исходный DataFrame с колонками трейсов,
                session_id и оценками
        Return:
            Словарь с ключами X, y и traces
        '''
        traces_columns = [
            'traceid',
            'spanid',
            'name',
            'starttimeunixnano',
            'endtimeunixnano',
        ]
        traces_extra = ['session_id', 'input_text']
        for extra_col in ['aef_kind']:
            if extra_col in var.columns:
                traces_extra.append(extra_col)
        basket_columns = list(set(var.columns) - set(traces_columns))

        X = var[basket_columns].drop_duplicates().reset_index(drop=True).copy()
        result = {}
        result['X'] = X.drop('estimation', axis=1)
        result['y'] = pd.DataFrame(
            {'estimation': X['estimation'].values},
            index=pd.Index(X['session_id'].values, name='session_id'),
        )

        traces = var[traces_columns + traces_extra].copy()
        traces = traces.rename(
            columns={
                'traceid': 'trace_id',
                'spanid': 'span_id',
                'name': 'span_name',
            }
        )
        traces['start_time_ns'] = pd.to_datetime(traces['starttimeunixnano'])
        traces['end_time_ns'] = pd.to_datetime(traces['endtimeunixnano'])
        traces = traces.drop_duplicates(
            subset=['trace_id', 'span_id'],
        ).drop(
            ['starttimeunixnano', 'endtimeunixnano'],
            axis=1,
        )
        result['traces'] = traces

        return result
