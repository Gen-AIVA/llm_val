import typing as tp
import pandas as pd
from llm_val.sampler.Sampler import Sampler


class TracesSamplerWithReflection(Sampler):
    def __init__(self, test: pd.DataFrame):
        '''
        test: Тестовый набор данных
        '''
        test = self._preproc(test)
        super().__init__(test=test)

    def _preproc(self, var: pd.DataFrame) -> tp.Dict[str, pd.DataFrame]:
        traces_columns = ['trace_id', 'span_id', 'span_name',
                        'start_time_ns', 'end_time_ns',
                        'session_id', 'aef_kind', 'input_text',
                        'output_text', 'parent_span_id']
        
        basket_columns = [col for col in var.columns if col not in traces_columns or col in ('input_text', 'session_id')]

        X = var[basket_columns].drop_duplicates(subset=['session_id']).reset_index(drop=True).copy()
        result = {}
        result['X'] = X.drop('estimation', axis=1) if 'estimation' in X.columns else X
        result['y'] = pd.DataFrame(X['estimation']) if 'estimation' in X.columns else pd.DataFrame()

        traces = var[traces_columns].copy()
        traces['start_time_ns'] = pd.to_datetime(traces['start_time_ns'])
        traces['end_time_ns'] = pd.to_datetime(traces['end_time_ns'])
        traces = traces.drop_duplicates(subset=['trace_id', 'span_id'])
        result['traces'] = traces
        return result