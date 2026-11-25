import pandas as pd
import typing as tp

from collections import deque


def prepare_traces(data: pd.DataFrame,
                   col_case: str='trace_id',
                   col_stage: str='span_name',
                   col_start_time: str='start_time_ns',
                   col_end_time: str='end_time_ns') -> pd.DataFrame:
    '''
    Подготовить события трейсов к расчёту графа и временных статистик.
    Parameters:
        data: Таблица событий. Исходная таблица не изменяется.
        col_case: Колонка идентификатора трейса.
        col_stage: Колонка названия ноды.
        col_start_time: Начало события: datetime, строка или Unix-время в наносекундах.
        col_end_time: Конец события в том же формате.
    Return:
        События с каноническими именами колонок и duration в секундах, отсортированные
        по трейсу, началу, концу и ноде. Строки без ID трейса или названия ноды исключаются.
    '''
    columns = [col_case, col_stage, col_start_time, col_end_time]
    traces = data[columns].copy()
    traces.columns = ['trace_id', 'span_name', 'start_time_ns', 'end_time_ns']
    missing_pattern = r'\s*(<?na[nt]?>?|n/a|none|null|<?unknown>?|\-+|—+|\.+|)'
    for column in traces.columns:
        missing = traces[column].astype(str).str.fullmatch(missing_pattern, case=False, na=False)
        traces[column] = traces[column].mask(missing)
    traces = traces.dropna(subset=['trace_id', 'span_name'])
    for column in ['start_time_ns', 'end_time_ns']:
        traces[column] = pd.to_datetime(traces[column], utc=True)
    traces = traces.sort_values(['start_time_ns', 'end_time_ns', 'trace_id', 'span_name'])
    traces = traces.reset_index(drop=True)
    traces['duration'] = (traces['end_time_ns'] - traces['start_time_ns']).dt.total_seconds()
    return traces


def find_connected_components(adj_matrix: tp.Sequence[tp.Sequence[int]]) -> tp.List[tp.List[int]]:
    '''
    Найти группы достижимых вершин обходом матрицы по строкам.
    Parameters:
        adj_matrix: Матрица смежности. Для направленного графа учитываются исходящие рёбра,
            как в прежней проверке связности трейсов.
    Return:
        Группы индексов вершин в порядке обхода в ширину.
    '''
    visited = set()
    components = []
    for node in range(len(adj_matrix)):
        if node in visited:
            continue
        component = []
        queue = deque([node])
        visited.add(node)
        while queue:
            current = queue.popleft()
            component.append(current)
            neighbors = [idx for idx, edge in enumerate(adj_matrix[current]) if edge == 1 and idx not in visited]
            visited.update(neighbors)
            queue.extend(neighbors)
        components.append(component)
    return components


def extract_graph_components(traces: pd.DataFrame) -> tp.Tuple[tp.List[str], tp.List[str], tp.List[tp.List[str]]]:
    '''
    Построить граф непосредственных переходов внутри каждого трейса.
    Parameters:
        traces: Результат prepare_traces с упорядоченными событиями.
    Return:
        Начальные ноды, конечные ноды и группы достижимых нод. Вспомогательные
        start/end-вершины в граф не добавляются. Порядок нод соответствует первому появлению.
    '''
    groups = traces.groupby('trace_id', observed=True)['span_name']
    start_nodes = groups.first().unique().tolist()
    end_nodes = groups.last().unique().tolist()
    nodes = traces['span_name'].unique().tolist()
    indices = {node: index for index, node in enumerate(nodes)}
    adjacency = [[0] * len(nodes) for _ in nodes]
    next_stages = groups.shift(-1)
    has_next = next_stages.notna()
    for source, target in zip(traces.loc[has_next, 'span_name'], next_stages[has_next], strict=True):
        adjacency[indices[source]][indices[target]] = 1
    components = [[nodes[index] for index in group] for group in find_connected_components(adjacency)]
    return start_nodes, end_nodes, components


def calc_time_statistic(traces: pd.DataFrame) -> pd.DataFrame:
    '''
    Посчитать среднее время ноды и среднее между квантилями 5% и 95%.
    Parameters:
        traces: Результат prepare_traces с duration в секундах.
    Return:
        Таблица временных статистик. Границы квантилей исключаются; если между
        ними нет наблюдений, среднее без выбросов равно NaN, как в прежнем расчёте.
    '''
    results = []
    for stage, group in traces.groupby('span_name', observed=True):
        durations = group['duration']
        normal_mask = (durations > durations.quantile(0.05)) & (durations < durations.quantile(0.95))
        results.append({'Нода': stage,
                        'Среднее время': round(durations.mean(), 4),
                        'Среднее время без выбросов': round(durations.loc[normal_mask].mean(), 4)})
    return pd.DataFrame(results, columns=['Нода', 'Среднее время', 'Среднее время без выбросов'])


def _start_loop_mask(traces: pd.DataFrame) -> pd.Series:
    '''
    Воспроизвести прежнюю разметку возвратов в начальную ноду.
    Parameters:
        traces: Упорядоченные события с общим RangeIndex после prepare_traces.
    Return:
        Маска событий после первого события до границы последнего возврата.
        Граница берётся из общего индекса, как в исходном алгоритме: это сохраняет
        результаты существующих отчётов для нескольких трейсов.
    '''
    groups = traces.groupby('trace_id', observed=True)['span_name']
    is_start = traces['span_name'].eq(groups.transform('first'))
    repeats = is_start.groupby(traces['trace_id'], observed=True).transform('sum')
    last_start = (traces.index.to_series().where(is_start, -1)
                  .groupby(traces['trace_id'], observed=True).transform('max'))
    positions = groups.cumcount()
    return (repeats > 1) & (positions > 0) & (positions < last_start)


def calc_loop_statistic(traces: pd.DataFrame) -> pd.DataFrame:
    '''
    Рассчитать пять прежних показателей зацикленности без внешнего анализатора.
    Parameters:
        traces: Результат prepare_traces с упорядоченными событиями.
    Return:
        Таблица нормализованных показателей по нодам. Сначала считаются доли
        размеченных событий ноды, затем каждый столбец нормализуется между нодами
        методом min-max. Константный столбец становится нулевым.
    '''
    stages = traces['span_name']
    groups = traces.groupby('trace_id', observed=True)['span_name']
    ping_pong = stages.eq(groups.shift(-3)).groupby(traces['trace_id'], observed=True).shift(1, fill_value=False)
    duplicated = traces.duplicated(['trace_id', 'span_name'], keep=False)
    repeated_before_last = traces.duplicated(['trace_id', 'span_name'], keep='last')
    masks = pd.DataFrame({'В начало': _start_loop_mask(traces),
                          'В себя': stages.eq(groups.shift(-1)),
                          '«Возврат»': stages.eq(groups.shift(-2)),
                          '«Пинг-Понг»': ping_pong,
                          'В произвольную операцию': repeated_before_last.eq(duplicated)})
    statistics = masks.groupby(stages, observed=True).mean().reindex(stages.unique())
    minimum = statistics.min()
    span = statistics.max() - minimum
    statistics = (statistics - minimum).div(span.where(span != 0, 1))
    return statistics.rename_axis('Нода').reset_index()
