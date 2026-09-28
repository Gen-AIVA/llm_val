import typing as tp
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import ttest_1samp

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import worst_semaphore, stat_significance_semaphore
from llm_val.val_test.model_quality.valtest_stat_significance import bootstrap_p_value


def report_valtest_metric_by_thema_agent(metric_value_by_scenario: tp.Dict[tp.Any, float],
                                         main_metric: str,
                                         p_value_by_scenario: tp.Dict[tp.Any, tp.Dict[str, tp.Dict[str, tp.Optional[float]]]],
                                         value_counts_by_scenario: tp.Optional[pd.Series]=None,
                                         significance_level: float=0.05,
                                         greater_is_better: bool=True,
                                         other_scenarios: tp.Tuple[str, ...]=('другое', 'other'),
                                         is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        metric_value_by_scenario: Значение ключевой метрики в разрезе сценариев
        main_metric: Название ключевой метркики качества. Hint: один из ключей словаря, который выдает scorer
        p_value_by_scenario: Значения p-value метрик в разрезе сценариев
        value_counts_by_scenario: Series с количеством элеметов в разрезе сценариев. Hint: df['scenario'].value_counts()
        significance_level: Уровень значимости для проверки бизнес-критерия
        greater_is_better: True, если чем больше метрика, тем лучше
        other_scenarios: Сценарии, не учитываемые в финальном светофоре
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    result_dataframes = []
    metric_color_by_scenario = get_metric_color_by_scenario(p_value_by_scenario=p_value_by_scenario,
                                                            significance_level=significance_level,
                                                            greater_is_better=greater_is_better)
    global_color_list = {scenario: worst_semaphore(list(color_by_metric.values()))
                         for scenario, color_by_metric in metric_color_by_scenario.items()}
    if value_counts_by_scenario is not None:
        info_df = value_counts_by_scenario.to_frame(
            name='Количество элементов в разрезе сценариев'
        )
        result_dataframes.append(info_df)
    report_df = create_metric_report_table(metric_value_by_scenario=metric_value_by_scenario,
                                           main_metric=main_metric,
                                           metric_color_by_scenario=metric_color_by_scenario)
    result_color_list = [color for scenario, color in global_color_list.items()
                         if scenario not in other_scenarios]
    color = 'gray' if is_info else worst_semaphore(result_color_list)
    report_df.loc[('Общий итог', ''), :] = ['', color]
    result_dataframes.insert(0, report_df)
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = result_dataframes
    return res_dict


def valtest_metric_by_thema_agent(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                  scorer: Scorer,
                                  main_metric: str,
                                  business_metric_threshold: float=0.7,
                                  data_type: str='test',
                                  metric_agg: str='single_mean',
                                  significance_level: float=0.05,
                                  main_column: tp.Optional[str]=None,
                                  greater_is_better: bool=True,
                                  other_scenarios: tp.Tuple[str, ...]=('другое', 'other'),
                                  exec_env: tp.Literal['default', 'sberds']='default',
                                  bootstrap: bool=False,
                                  bootstrap_iterations: int=1000,
                                  random_state: tp.Optional[int]=None,
                                  metric_fn: tp.Optional[tp.Callable]=None,
                                  is_info: bool=False,
                                  metric_value_by_scenario: tp.Optional[tp.Dict[tp.Any, float]]=None,
                                  p_value_by_scenario: tp.Optional[tp.Dict[tp.Any, tp.Dict[str, tp.Dict[str, tp.Optional[float]]]]]=None,
                                  **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест достижения качества по ключевой метрике в разрезе тематик/доменов
    Parameters:
        sampler: Объект класса Sampler или словарь
        scorer: Объект класса Scorer
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        business_metric_threshold: Бизнес-критерий для ключевой метрики качества
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        significance_level: Уровень значимости для проверки бизнес-критерия
        main_column: Название колонки в y с объектным значением ключевой метрики
        greater_is_better: True, если чем больше метрика, тем лучше
        other_scenarios: Сценарии, не учитываемые в финальном светофоре
        exec_env: Среда выполнения. Принимает значения 'default' или 'sberds'
        bootstrap: Флаг для использования bootstrap при расчете p-value
        bootstrap_iterations: Число bootstrap итераций
        random_state: seed генератора случайных чисел
        metric_fn: Функция агрегации объектных значений метрики для bootstrap
        is_info: True, если тест информативный
        metric_value_by_scenario: Опциональное значение метрики.
            Если не None, то подсчет метрики не производится, и на основе переданного аргумента выставляется светофор.
        p_value_by_scenario: Опциональные значения p-value метрик в разрезе сценариев.
            Если не None, то подсчет статистики не производится, и на основе переданного аргумента выставляется светофор.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    value_counts_by_scenario = None
    
    if metric_value_by_scenario is None or p_value_by_scenario is None:
        data = get_sampler_data(sampler=sampler,
                                data_type=data_type,
                                exec_env=exec_env)
        X, y = data['X'], data['y']
        if 'session_id' in X.columns:
            X = X.drop_duplicates(subset='session_id').set_index('session_id')
            
    if metric_value_by_scenario is None:
        value_counts_by_scenario = X['scenario'].value_counts(dropna=False)
        if bool(X['scenario'].isna().all()):
            is_info = True
        metric_value_by_scenario = calculate_metric_by_thema_agent(X=X,
                                                                   y=y,
                                                                   scorer=scorer,
                                                                   main_metric=main_metric,
                                                                   metric_agg=metric_agg,
                                                                   data_type=data_type)
    if p_value_by_scenario is None:
        p_value_by_scenario = calculate_p_value_by_thema_agent(X=X,
                                                               y=y,
                                                               business_metric_threshold=business_metric_threshold,
                                                               main_metric=main_metric,
                                                               metric_agg=metric_agg,
                                                               main_column=main_column,
                                                               bootstrap=bootstrap,
                                                               bootstrap_iterations=bootstrap_iterations,
                                                               random_state=random_state,
                                                               metric_fn=metric_fn)
    precomputed = {'metric_value_by_scenario': deepcopy(metric_value_by_scenario),
                   'p_value_by_scenario': deepcopy(p_value_by_scenario)}
    report = report_valtest_metric_by_thema_agent(metric_value_by_scenario=metric_value_by_scenario,
                                                  main_metric=main_metric,
                                                  p_value_by_scenario=p_value_by_scenario,
                                                  value_counts_by_scenario=value_counts_by_scenario,
                                                  significance_level=significance_level,
                                                  greater_is_better=greater_is_better,
                                                  other_scenarios=other_scenarios,
                                                  is_info=is_info)
    return {'report': report,
            'precomputed': precomputed}


def get_metric_color_by_scenario(p_value_by_scenario: tp.Dict[tp.Any, tp.Dict[str, tp.Dict[str, tp.Optional[float]]]],
                                 significance_level: float=0.05,
                                 greater_is_better: bool=True) -> tp.Dict[tp.Any, tp.Dict[str, str]]:
    '''
    Считает светофоры метрик по p-value в разрезе сценариев
    Parameters:
        p_value_by_scenario: Значения p-value метрик в разрезе сценариев
        significance_level: Уровень значимости для проверки бизнес-критерия
        greater_is_better: True, если чем больше метрика, тем лучше
    Return:
        Светофоры метрик в разрезе сценариев
    '''
    metric_color_by_scenario = {}
    for scenario, p_value_by_metric in p_value_by_scenario.items():
        metric_color_by_scenario[scenario] = {
            name: stat_significance_semaphore(p_value=p_value,
                                              significance_level=significance_level,
                                              greater_is_better=greater_is_better)
            for name, p_value in p_value_by_metric.items()
        }
    return metric_color_by_scenario


def create_metric_report_table(metric_value_by_scenario: tp.Dict[tp.Any, float],
                               main_metric: str,
                               metric_color_by_scenario: tp.Dict[tp.Any, tp.Dict[str, str]]) -> pd.DataFrame:
    '''
    Создает таблицу результатов в формате методики
    Parameters:
        metric_value_by_scenario: Значение ключевой метрики в разрезе сценариев
        main_metric: Название ключевой метрики качества
        metric_color_by_scenario: Светофоры метрик в разрезе сценариев
    Return:
        DataFrame с результатами теста
    '''
    rows = []
    for scenario, metric_value in metric_value_by_scenario.items():
        color_by_metric = get_value_by_scenario(metric_color_by_scenario, scenario)
        rows.append({'Тематика': scenario,
                     'Наименование критерия': main_metric,
                     'Ключевая метрика качества': metric_value,
                     'Результат теста': color_by_metric[main_metric]})
    report_df = pd.DataFrame(rows, dtype=object).set_index(['Тематика', 'Наименование критерия'])
    return report_df


def get_value_by_scenario(value_by_scenario: tp.Dict[tp.Any, tp.Any],
                          scenario: tp.Any) -> tp.Any:
    '''
    Получает значение по названию сценария с учетом пустых значений
    Parameters:
        value_by_scenario: Значения в разрезе сценариев
        scenario: Название сценария
    Return:
        Значение для сценария
    '''
    scenario_value = value_by_scenario.get(scenario)
    if scenario_value is None and pd.isna(scenario):
        scenario_value = get_nan_scenario_value(value_by_scenario)
    if scenario_value is None:
        raise ValueError(f'Сценарий {scenario} отсутствует в рассчитанных значениях')
    return scenario_value


def get_sampler_data(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                     data_type: str,
                     exec_env: tp.Literal['default', 'sberds']='default') -> tp.Dict[str, pd.DataFrame]:
    '''
    Получает данные из sampler'а
    Parameters:
        sampler: Объект класса Sampler или словарь
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        exec_env: Среда выполнения. Принимает значения 'default' или 'sberds'
    Return:
        Словарь с X и y
    '''
    data = (tp.cast(tp.Dict[str, tp.Any], sampler)[data_type]
            if exec_env == 'sberds' else getattr(sampler, data_type))
    return data


def calculate_p_value_by_thema_agent(X: pd.DataFrame,
                                     y: pd.DataFrame,
                                     business_metric_threshold: float,
                                     main_metric: str='',
                                     metric_agg: str='single_mean',
                                     main_column: tp.Optional[str]=None,
                                     bootstrap: bool=False,
                                     bootstrap_iterations: int=1000,
                                     random_state: tp.Optional[int]=None,
                                     metric_fn: tp.Optional[tp.Callable]=None
                                     ) -> tp.Dict[tp.Any, tp.Dict[str, tp.Dict[str, tp.Optional[float]]]]:
    '''
    Расчет p-value метрик в разрезе сценариев
    Parameters:
        X: DataFrame из sampler'а
        y: DataFrame из sampler'а
        business_metric_threshold: Бизнес-критерий для ключевой метрики качества
        main_metric: Название ключевой метрики качества
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        main_column: Название колонки в y с объектным значением ключевой метрики
        bootstrap: Флаг для использования bootstrap при расчете p-value
        bootstrap_iterations: Число bootstrap итераций
        random_state: seed генератора случайных чисел
        metric_fn: Функция агрегации объектных значений метрики для bootstrap
    Return:
        Словарь c p-value метрик в разрезе сценариев
    '''
    resolved_main_column = resolve_main_column(y=y,
                                               main_metric=main_metric,
                                               metric_agg=metric_agg,
                                               main_column=main_column)
    p_value_by_scenario = {}
    for scenario in X['scenario'].unique():
        if pd.isna(scenario):
            scenario_mask = X['scenario'].isnull()
        else:
            scenario_mask = X['scenario'] == scenario
        p_value_by_scenario[scenario] = calculate_p_value_for_scenario(y=y.loc[X.index[scenario_mask]],
                                                                        business_metric_threshold=business_metric_threshold,
                                                                        main_metric=main_metric,
                                                                        main_column=resolved_main_column,
                                                                        bootstrap=bootstrap,
                                                                        bootstrap_iterations=bootstrap_iterations,
                                                                        random_state=random_state,
                                                                        metric_fn=metric_fn)
    return p_value_by_scenario


def resolve_main_column(y: pd.DataFrame,
                        main_metric: str,
                        metric_agg: str,
                        main_column: tp.Optional[str]=None) -> str:
    '''
    Определяет колонку с объектными значениями ключевой метрики
    Parameters:
        y: DataFrame из sampler'а
        main_metric: Название ключевой метрики качества
        metric_agg: Название агрегации метрики
        main_column: Название колонки в y с объектным значением ключевой метрики
    Return:
        Название колонки с объектными значениями ключевой метрики
    '''
    metric_column_candidate = get_metric_column_candidate(main_metric=main_metric,
                                                          metric_agg=metric_agg)
    if main_column is not None and main_column in y.columns:
        resolved_main_column = main_column
    elif main_metric in y.columns:
        resolved_main_column = main_metric
    elif metric_column_candidate in y.columns:
        resolved_main_column = metric_column_candidate
    elif len(y.columns) == 1:
        resolved_main_column = tp.cast(str, y.columns[0])
    else:
        raise ValueError(
            f'Не удалось определить колонку с объектными значениями ключевой метрики. '
            f'Передайте main_column явно. main_metric={main_metric}, '
            f'metric_agg={metric_agg}, доступные колонки y={list(y.columns)}'
        )
    return resolved_main_column


def get_metric_column_candidate(main_metric: str,
                                metric_agg: str) -> str:
    '''
    Получает возможное имя объектной колонки по названию агрегированной метрики
    Parameters:
        main_metric: Название ключевой метрики качества
        metric_agg: Название агрегации метрики
    Return:
        Возможное имя колонки с объектными значениями ключевой метрики
    '''
    prefix = f'{metric_agg}_'
    candidate = main_metric[len(prefix):] if main_metric.startswith(prefix) else main_metric
    return candidate


def get_nan_scenario_value(value_by_scenario: tp.Dict[tp.Any, tp.Any]) -> tp.Any:
    '''
    Получает значение для пустого сценария
    Parameters:
        value_by_scenario: Значения в разрезе сценариев
    Return:
        Значение для пустого сценария
    '''
    scenario_value = None
    for scenario, value in value_by_scenario.items():
        if pd.isna(scenario):
            scenario_value = value
            break
    return scenario_value


def calculate_p_value_for_scenario(y: pd.DataFrame,
                                   business_metric_threshold: float,
                                   main_metric: str='',
                                   main_column: str='',
                                   bootstrap: bool=False,
                                   bootstrap_iterations: int=1000,
                                   random_state: tp.Optional[int]=None,
                                   metric_fn: tp.Optional[tp.Callable]=None) -> tp.Dict[str, tp.Dict[str, tp.Optional[float]]]:
    '''
    Расчет p-value метрик для одного сценария
    Parameters:
        y: DataFrame из sampler'а для одного сценария
        business_metric_threshold: Бизнес-критерий для ключевой метрики качества
        main_metric: Название ключевой метрики качества
        main_column: Название колонки в y с объектным значением ключевой метрики
        bootstrap: Флаг для использования bootstrap при расчете p-value
        bootstrap_iterations: Число bootstrap итераций
        random_state: seed генератора случайных чисел
        metric_fn: Функция агрегации объектных значений метрики для bootstrap
    Return:
        Словарь с p-value метрик для одного сценария
    '''
    metric_values = tp.cast(pd.Series, y[main_column])
    p_value = {
        main_metric: calculate_p_value_for_metric_values(metric_values=metric_values,
                                                         business_metric_threshold=business_metric_threshold,
                                                         bootstrap=bootstrap,
                                                         bootstrap_iterations=bootstrap_iterations,
                                                         random_state=random_state,
                                                         metric_fn=metric_fn)
    }
    return p_value


def calculate_p_value_for_metric_values(metric_values: pd.Series,
                                        business_metric_threshold: float,
                                        bootstrap: bool=False,
                                        bootstrap_iterations: int=1000,
                                        random_state: tp.Optional[int]=None,
                                        metric_fn: tp.Optional[tp.Callable]=None) -> tp.Dict[str, tp.Optional[float]]:
    '''
    Расчет p-value для объектных значений метрики
    Parameters:
        metric_values: Объектные значения метрики
        business_metric_threshold: Бизнес-критерий для ключевой метрики качества
        bootstrap: Флаг для использования bootstrap при расчете p-value
        bootstrap_iterations: Число bootstrap итераций
        random_state: seed генератора случайных чисел
        metric_fn: Функция агрегации объектных значений метрики для bootstrap
    Return:
        Словарь с p-value для направленных гипотез
    '''
    p_value: tp.Dict[str, tp.Optional[float]]
    metric_values = metric_values.dropna()
    if len(metric_values) < 2:
        p_value = {'greater': None,
                   'less': None}
    elif bootstrap:
        sample = np.asarray(metric_values)
        metric_fn = np.mean if metric_fn is None else metric_fn
        p_value = {
            'greater': bootstrap_p_value(sample,
                                         business_metric_threshold,
                                         bootstrap_iterations,
                                         'greater',
                                         random_state,
                                         metric_fn=metric_fn,
                                         strategy='rigorous'),
            'less': bootstrap_p_value(sample,
                                      business_metric_threshold,
                                      bootstrap_iterations,
                                      'less',
                                      random_state,
                                      metric_fn=metric_fn,
                                      strategy='rigorous')
        }
    else:
        greater_p_value = tp.cast(tp.Any, ttest_1samp(metric_values,
                                                      business_metric_threshold,
                                                      alternative='greater')).pvalue
        less_p_value = tp.cast(tp.Any, ttest_1samp(metric_values,
                                                   business_metric_threshold,
                                                   alternative='less')).pvalue
        p_value = {
            'greater': None if pd.isna(greater_p_value) else float(greater_p_value),
            'less': None if pd.isna(less_p_value) else float(less_p_value)
        }
    return p_value


def calculate_metric_by_thema_agent(X: pd.DataFrame,
                                    y: pd.DataFrame,
                                    scorer: Scorer,
                                    main_metric: str,
                                    metric_agg: str='single_mean',
                                    data_type: str='test') -> tp.Dict[tp.Any, float]:
    '''
    Расчет метрик в разрезе сценариев
    Parameters:
        X: DataFrame из sampler'а
        y: DataFrame из sampler'а
        scorer: Объект класса Scorer
        main_metric: Название ключевой метрики качества
        metric_agg: Название агрегации метрики
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
    Return:
        Словарь c значениями ключевой метрики
    '''
    metric_value = {}
    for scenario in X['scenario'].unique():
        if pd.isna(scenario):
            X_tmp = X[X['scenario'].isnull()].copy()
        else:
            X_tmp = X[X['scenario'] == scenario].copy()
        y_tmp = y.loc[X_tmp.index]
        sampler = Sampler(**{data_type: {'X': X_tmp, 'y': y_tmp}})
        scenario_metric_value = scorer.calc(sampler=sampler,
                                            data_type=data_type,
                                            metric_name=metric_agg)
        if main_metric not in scenario_metric_value:
            raise ValueError(f'Ключевая метрика {main_metric} отсутствует в результате scorer.calc')
        metric_value[scenario] = scenario_metric_value[main_metric]
    return metric_value
