import numpy as np
import pandas as pd
import typing as tp
import warnings

from llm_val.sampler.Sampler import Sampler
from llm_val.scorer.Scorer import Scorer
from llm_val.utils.metrics import METRICS


def report_valtest_tool_metric_dynamics(tool_metrics: pd.DataFrame,
                                        greater_is_better: bool = True) -> tp.Dict[str, tp.Any]:
    '''Словарь в соответствии с концепцией llm_val.

    Parameters:
        tool_metrics: DataFrame с колонками span_name, metric_name,
            metric_value, abs_change, rel_change
        greater_is_better: True, если чем больше метрика, тем лучше.
            Используется для сортировки таблицы: сверху — самые
            "плохие" инструменты (с наименьшим улучшением метрики)
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    result_df = format_result_table(tool_metrics, greater_is_better)
    res_dict = {
        'semaphore': 'gray',
        'result_plots': [],
        'result_dataframes': [result_df],
    }
    return res_dict


def valtest_tool_metric_dynamics(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                 scorer: tp.Optional[Scorer] = None,
                                 data_type: str = 'test',
                                 metric_column: str = 'estimation',
                                 metric_agg: str = 'single_mean',
                                 greater_is_better: bool = True,
                                 exec_env: str = 'default',
                                 calc_metric_fn: tp.Optional[tp.Callable] = None,
                                 precomputed_report: tp.Optional[pd.DataFrame] = None,
                                 **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''Определяет поведение ключевой метрики в зависимости от инструментов.

    Тест работает только с трейсами, имеющими в колонке aef_kind
    значение tool. Инструменты, присутствующие во всех сессиях, удаляются.

    Parameters:
        sampler: Объект класса Sampler с трейсами
        scorer: Объект класса Scorer для подсчёта метрики качества.
            Если None — метрика считается через calc_metric_fn (или calc_metric_from_data)
        data_type: Тип данных, на которых производится расчёт.
            Hint: название атрибута в sampler'е
        metric_column: Название колонки в sampler[data_type]['y'],
            содержащей целевую метрику качества
        metric_agg: Название агрегации метрики.
            Hint: один из ключей в словаре metrics в scorer'е
        greater_is_better: True, если чем больше метрика, тем лучше
        exec_env: Среда выполнения ('default' или 'sberds')
        calc_metric_fn: Функция подсчёта метрики при scorer=None.
            Сигнатура: (data: pd.DataFrame, metric_agg: str) -> Dict[str, float].
            Если None — используется calc_metric_from_data
        precomputed_report: Предподсчитанная таблица результатов.
            Если передана, расчёт не производится
    Return:
        report в соответствии с концепцией llm_val
    '''
    if precomputed_report is not None:
        report = report_valtest_tool_metric_dynamics(
            tool_metrics=precomputed_report,
            greater_is_better=greater_is_better,
        )
        result = {'report': report, 'precomputed': {'result_df': precomputed_report}}
    else:
        overall_metrics = calc_overall_metrics(
            sampler,
            scorer,
            data_type,
            exec_env,
            metric_agg,
            calc_metric_fn,
        )

        tool_metrics_df = compute_tool_metrics(
            sampler=sampler,
            scorer=scorer,
            data_type=data_type,
            exec_env=exec_env,
            metric_column=metric_column,
            metric_agg=metric_agg,
            overall_metrics=overall_metrics,
            greater_is_better=greater_is_better,
            calc_metric_fn=calc_metric_fn,
        )

        report = report_valtest_tool_metric_dynamics(
            tool_metrics=tool_metrics_df,
            greater_is_better=greater_is_better,
        )
        result = {'report': report, 'precomputed': {'result_df': tool_metrics_df}}

    return result


def calc_overall_metrics(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                         scorer: tp.Optional[Scorer],
                         data_type: str,
                         exec_env: str,
                         metric_agg: str,
                         calc_metric_fn: tp.Optional[tp.Callable] = None) -> tp.Dict[str, float]:
    '''Вычисляет метрики качества по всем данным переданного sampler.

    Parameters:
        sampler: Объект класса Sampler с трейсами
        scorer: Объект класса Scorer. Если None — через calc_metric_fn
        data_type: Тип данных
        exec_env: Среда выполнения ('default' или 'sberds')
        metric_agg: Название агрегации метрики
        calc_metric_fn: Функция подсчёта метрики при scorer=None.
            Сигнатура: (data: pd.DataFrame, metric_agg: str) -> Dict[str, float].
            Если None — используется calc_metric_from_data
    Return:
        Словарь {metric_name: value, ...} — все посчитанные метрики
    '''
    if scorer is not None:
        res = scorer.calc(sampler=sampler, data_type=data_type, metric_name=metric_agg)
    else:
        fn = calc_metric_fn if calc_metric_fn is not None else calc_metric_from_data
        y_data = get_data(sampler, data_type, 'y', exec_env)
        res = fn(y_data, metric_agg)
    return res


def calc_metric_from_data(data: pd.DataFrame,
                          metric_name: tp.Optional[str] = None,
                          metrics: tp.Dict[str, tp.Any] = METRICS) -> tp.Dict[str, float]:
    '''Подсчёт метрик по данным (как calc_metric из valtest_metric_agent).

    Parameters:
        data: DataFrame с данными для подсчёта метрик
        metric_name: Название агрегации метрики.
            Если None — считаются все метрики из metrics
        metrics: Словарь с метриками
    Return:
        Словарь {metric_name: value, ...} с посчитанными метриками
    '''
    if metric_name is not None:
        metrics = {metric_name: metrics[metric_name]}
    metric_res: tp.Dict[str, float] = {}
    for func_name, func_dict in metrics.items():
        if func_dict['is_singlecol']:
            for name_col in data:
                name = f'{func_name}_{name_col}'
                metric_res[name] = float(func_dict['call'](data[name_col]))
        else:
            metric_res[func_name] = float(func_dict['call'](data))
    return metric_res


def make_sampler_from_values(y_values: pd.Series, metric_column: str) -> Sampler:
    '''Создаёт Sampler с y-данными для подсчёта метрики через Scorer.

    Parameters:
        y_values: Значения метрики для одного инструмента
        metric_column: Название колонки с метрикой
    Return:
        Объект Sampler с test-атрибутом, содержащим X и y
    '''
    metric_dict = {
        'X': pd.DataFrame(),
        'y': pd.DataFrame({metric_column: y_values.values}),
    }
    return Sampler(test=metric_dict)


def compute_tool_metrics(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                         scorer: tp.Optional[Scorer],
                         data_type: str,
                         exec_env: str,
                         metric_column: str,
                         metric_agg: str,
                         overall_metrics: tp.Dict[str, float],
                         greater_is_better: bool = True,
                         calc_metric_fn: tp.Optional[tp.Callable] = None) -> pd.DataFrame:
    '''Вычисляет метрики качества для каждого инструмента.

    Фильтрует трейсы по aef_kind='tool', удаляет инструменты,
    присутствующие во всех сессиях.

    Parameters:
        sampler: Объект класса Sampler с трейсами
        scorer: Объект класса Scorer для подсчёта метрики.
            Может быть None на sberds
        data_type: Тип данных для извлечения из sampler
        exec_env: Среда выполнения ('default' или 'sberds')
        metric_column: Название колонки с целевой метрикой в y
        metric_agg: Название агрегации метрики (ключ из METRICS)
        overall_metrics: Словарь {metric_name: value} — общие метрики по всем данным
        greater_is_better: True, если чем больше метрика, тем лучше.
            Используется для корректного знака изменения:
            при True положительное изменение = улучшение,
            при False знак инвертируется
        calc_metric_fn: Функция подсчёта метрики при scorer=None.
            Сигнатура: (data: pd.DataFrame, metric_agg: str) -> Dict[str, float].
            Если None — используется calc_metric_from_data
    Return:
        DataFrame с колонками span_name, metric_name, metric_value, abs_change, rel_change
    '''
    traces = get_data(sampler, data_type, 'traces', exec_env).copy()

    if 'aef_kind' in traces.columns:
        traces = traces[traces['aef_kind'] == 'tool'].copy()
    else:
        warnings.warn(
            f'Колонка aef_kind отсутствует в трейсах.'
            f' Фильтрация не применяется — метрики считаются по всем нодам.',
            stacklevel=2,
        )

    total_sessions = traces['session_id'].nunique()
    span_session_counts = traces.groupby('span_name')['session_id'].nunique()
    tools_in_all_sessions = span_session_counts[span_session_counts == total_sessions].index
    if len(tools_in_all_sessions) > 0:
        traces = traces[~traces['span_name'].isin(tools_in_all_sessions)].copy()

    tool_metrics_list = collect_tool_metrics(
        sampler,
        scorer,
        data_type,
        exec_env,
        metric_column,
        metric_agg,
        traces,
        calc_metric_fn,
    )

    if not tool_metrics_list:
        result = pd.DataFrame(columns=['span_name', 'metric_name', 'metric_value', 'abs_change', 'rel_change'])
    else:
        rows: tp.List[tp.Dict[str, tp.Any]] = []
        sign = 1 if greater_is_better else -1
        for tool_row in tool_metrics_list:
            span_name = tool_row['span_name']
            tool_metrics_dict = tool_row['tool_metrics']
            for metric_key, metric_value in tool_metrics_dict.items():
                overall_val = overall_metrics.get(metric_key, 1.0)
                abs_change = (metric_value - overall_val) * sign
                rel_change = float(
                    np.where(
                        overall_val != 0,
                        (metric_value - overall_val) / abs(overall_val) * sign,
                        0.0,
                    )
                )
                rows.append(
                    {
                        'span_name': span_name,
                        'metric_name': metric_key,
                        'metric_value': metric_value,
                        'abs_change': abs_change,
                        'rel_change': rel_change,
                    }
                )
        result = pd.DataFrame(rows)
    return result


def collect_tool_metrics(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                         scorer: tp.Optional[Scorer],
                         data_type: str,
                         exec_env: str,
                         metric_column: str,
                         metric_agg: str,
                         traces: pd.DataFrame,
                         calc_metric_fn: tp.Optional[tp.Callable] = None) -> tp.List[tp.Dict[str, tp.Any]]:
    ''' Собирает словари метрик по каждому инструменту из трейсов.

    Фильтрация y-данных выполняется по session_id,
    извлечённому из трейсов для каждого инструмента.

    Parameters:
        sampler: Объект класса Sampler с данными
        scorer: Объект класса Scorer. Если None — через calc_metric_fn
        data_type: Тип данных
        exec_env: Среда выполнения ('default' или 'sberds')
        metric_column: Название колонки с метрикой
        metric_agg: Название агрегации метрики
        traces: Отфильтрованный DataFrame с трейсами
        calc_metric_fn: Функция подсчёта метрики при scorer=None.
            Сигнатура: (data: pd.DataFrame, metric_agg: str) -> Dict[str, float].
            Если None — используется calc_metric_from_data
    Return:
        Список словарей {'span_name': ..., 'tool_metrics': {metric: value, ...}}
    '''
    y_data = get_data(sampler, data_type, 'y', exec_env)

    tool_metrics_list = []

    for span_name in traces['span_name'].unique():
        span_session_ids = traces.loc[traces['span_name'] == span_name, 'session_id'].unique()

        span_y = y_data.loc[y_data.index.isin(span_session_ids), metric_column]

        if span_y.empty:
            continue

        if scorer is not None:
            span_sampler = make_sampler_from_values(span_y, metric_column)
            span_metrics = scorer.calc(
                sampler=span_sampler,
                data_type='test',
                metric_name=metric_agg,
            )
        else:
            fn = calc_metric_fn if calc_metric_fn is not None else calc_metric_from_data
            span_metrics = fn(span_y.to_frame(metric_column), metric_agg)

        tool_metrics_list.append(
            {
                'span_name': span_name,
                'tool_metrics': span_metrics,
            }
        )

    return tool_metrics_list


def format_result_table(tool_metrics: pd.DataFrame,
                        greater_is_better: bool = True) -> pd.DataFrame:
    '''Форматирует результат в таблицу.

    Parameters:
        tool_metrics: DataFrame с колонками span_name, metric_name, metric_value,
            abs_change, rel_change. Если уже отформатирован (есть колонка
            'Нода/инструмент'), возвращается как есть
        greater_is_better: True, если чем больше метрика, тем лучше.
            Определяет порядок сортировки: наверху таблицы — самые
            "плохие" инструменты (с наименьшим положительным изменением
            или с наибольшим отрицательным изменением метрики)
    Return:
        Отформатированный DataFrame, отсортированный так, что сверху
        оказываются инструменты с наихудшим влиянием на метрику
    '''
    if 'Нода/инструмент' in tool_metrics.columns:
        result_df = tool_metrics
    elif tool_metrics.empty:
        result_df = pd.DataFrame(
            {
                'Нода/инструмент': [],
                'Метрика': [],
                'Абсолютное (относительное) изменение': [],
            }
        )
    else:
        sorted_metrics = tool_metrics.sort_values(
            by='rel_change',
            ascending=greater_is_better,
        ).reset_index(drop=True)

        rel_change_pct = sorted_metrics['rel_change'] * 100
        abs_change_str = sorted_metrics['abs_change'].round(4).astype(str)
        rel_change_str = rel_change_pct.round(1).apply(lambda x: f'{x:+.1f}%')

        result_df = pd.DataFrame(
            {
                'Нода/инструмент': sorted_metrics['span_name'],
                'Метрика': sorted_metrics['metric_value'].round(4),
                'Абсолютное (относительное) изменение': abs_change_str + ' (' + rel_change_str + ')',
            }
        )
    return result_df


def get_data(sampler_obj: tp.Any, data_type: str, key: str, exec_env: str = 'default') -> tp.Any:
    '''Извлекает данные из sampler по data_type и key.

    Parameters:
        sampler_obj: Объект сэмплера, содержащий данные для извлечения
        data_type: Тип данных, на которых производится расчет метрики.
            Hint: название атрибута в sampler'е
        key: Ключ или имя атрибута для извлечения конкретных данных
            (например, 'X', 'y', 'traces')
        exec_env: Переменная исполняемого окружения ('default' или 'sberds')

    Return:
        Содержимое sampler_obj из data_type по ключу key
    '''
    if exec_env == 'default':
        res = getattr(sampler_obj, data_type)[key]
    elif exec_env == 'sberds':
        res = sampler_obj[data_type][key]
    else:
        res = None
    return res
