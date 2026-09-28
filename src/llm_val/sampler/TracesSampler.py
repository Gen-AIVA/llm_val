import typing as tp
import pandas as pd
from llm_val.sampler.Sampler import Sampler


class TracesSampler(Sampler):

    def __init__(self, test: pd.DataFrame):
        '''
        test: Тестовый набор данных
        '''
        test = self._preproc(test)
        super().__init__(test=test)

    def _preproc(self, var: pd.DataFrame) -> tp.Dict[str, pd.DataFrame]:
        traces_columns = ['session_id', 'traceid', 'spanid', 'name', 'starttimeunixnano', 'endtimeunixnano']
        basket_columns = list(set(var.columns) - set(['name', 'starttimeunixnano', 'endtimeunixnano']))
        X = var[basket_columns].drop_duplicates().reset_index(drop=True).copy()
        result = {}
        result['X'] = X.drop('estimation', axis=1).rename(columns={'traceid': 'trace_id', 'spanid': 'span_id'})
        result['y'] = pd.DataFrame(X['estimation'])
        result['y'].index.name = 'session_id'
        traces = var[traces_columns].copy().rename(columns={'traceid': 'trace_id', 'spanid': 'span_id'})
        traces = traces.rename(columns={'name': 'span_name'})
        traces['start_time_ns'] = pd.to_datetime(traces['starttimeunixnano'])
        traces['end_time_ns'] = pd.to_datetime(traces['endtimeunixnano'])
        traces = traces.drop_duplicates(subset=['trace_id', 'span_id']).drop(['starttimeunixnano', 'endtimeunixnano'], axis=1)
        result['traces'] = traces
        return result