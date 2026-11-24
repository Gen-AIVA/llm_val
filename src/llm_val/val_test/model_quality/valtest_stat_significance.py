import numpy as np
import typing as tp
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon, ttest_1samp, ttest_ind
from functools import partial

from llm_val.sampler.Sampler import Sampler
from llm_val.scorer.Scorer import Scorer
from llm_val.utils.report_helper import semaphore_by_threshold


def report_valtest_stat_significance(p_value: float,
                                     extra_samplers_keys: tp.List=[],
                                     data_type: str='test',
                                     semaphore_threshold: tp.Tuple[float, float]=(0.01, 0.05),
                                     figure: plt.Figure=None,
                                     is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        p_value: Значение p-value
        extra_samplers_keys: Названия сравниваемых моделей
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold: Границы светофора. Указываются в порядке возрастания
        figure: График с визуализацией доверительных интервалов
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    samplers_str = '' if not extra_samplers_keys else ' ' + ', '.join(extra_samplers_keys)
    color = 'gray' if is_info else semaphore_by_threshold(value=p_value,
                                                          threshold=semaphore_threshold,
                                                          greater_is_better=False,
                                                          left_border_is_bigger=True,
                                                          right_border_is_bigger=False)
    res_table = pd.DataFrame({f'Значение p-value на {data_type}{samplers_str}': [p_value],
                              'Результат теста': [color]})
    fig_list = [figure] if figure is not None else []
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = fig_list
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_stat_significance(sampler: tp.Union[Sampler, tp.Dict],
                              scorer: tp.Optional[Scorer],
                              metric_agg: str,
                              data_type: str='test',
                              main_column: tp.Optional[str]=None,
                              value_threshold: float=0.5,
                              extra_samplers: tp.Optional[tp.Dict[str, tp.Any]]=None,
                              same_questions: bool=False,
                              semaphore_threshold: tp.Tuple[float, float]=(0.01, 0.05),
                              bootstrap: bool=False,
                              bootstrap_iterations: int=1000,
                              bootstrap_ci_type: str='percentile',
                              alternative: str='greater',
                              exec_env: str='default',
                              random_state: tp.Optional[int]=None,
                              is_info: bool=False,
                              p_value: tp.Optional[float]=None,
                              calc_metric_fn: tp.Optional[tp.Callable]=None,
                              **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на статзначимость различия ключевой метрики качества
    Parameters:
        sampler: Объект класса Sampler
        scorer: Объект класса Scorer. Используется при exec_env='default'
        metric_agg: Название агрегации метрики.
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        main_column: Название колонки, содержащей ключевую метрику качества; если None, то усредняются все колонки в y
        value_threshold: Пороговое значение ключевой метрики качества при использовании 1-выбор-ого крит-ия Стьюдента
        extra_samplers: Дополнительные объекты Sampler, используемые, если в тесте участвует более одной модели
        same_questions: True, если разные модели отвечали на одинаковые вопросы; False, если на разные
        semaphore_threshold: Границы светофора
        bootstrap: Флаг для использования bootstrap
        bootstrap_iterations: Число bootstrap итераций
        bootstrap_ci_type: Тип доверительного интервала ('percentile', 'basic', 'normal')
        alternative: Тип альтернативной гипотезы ('greater', 'less', 'two-sided')
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
        random_state: seed генератора случайных чисел
        is_info: True, если тест информативный
        p_value: Опциональное значение p-value.
            Если не None, то подсчет статистики не производится, и на основе переданного аргумента выставляется светофор.
        calc_metric_fn: Пользовательская функция расчета метрики.
            Используется при metric_agg != 'single_mean' и scorer is None.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    extra_samplers_keys = [] if extra_samplers is None else list(extra_samplers.keys())
    original_metric_y = get_real_metric(sampler,
                                        data_type,
                                        exec_env,
                                        main_column)
    figure = None
    metric_fn = np.mean
    bootstrap_data = None
    precomputed = {}
    if bootstrap:
        if not main_column:
            raise ValueError('Для режима bootstrap необходимо явно указать параметр main_column.')

        y_data = get_data(sampler, data_type, 'y', exec_env)
        valid_mask = y_data[main_column].notna()
        bootstrap_data = y_data[valid_mask].reset_index(drop=True)
        
        if metric_agg == 'single_mean':
            strategy = 'rigorous'
            bootstrap_data = bootstrap_data[main_column].values
            metric_fn = np.mean
        else:
            strategy = 'empirical'
            if scorer is not None:
                metric_fn = partial(calc_metric_values,
                                    scorer=scorer,
                                    metric_agg=metric_agg,
                                    data_type=data_type)
            elif calc_metric_fn is not None:
                metric_fn = calc_metric_fn
            else:
                raise ValueError(
                    "Для режима bootstrap с metric_agg != 'single_mean' необходимо передать "
                    "либо scorer, либо пользовательскую функцию расчета метрики через параметр calc_metric_fn."
                )

        ci_low, ci_high = bootstrap_confidence_interval(bootstrap_data,
                                                        bootstrap_iterations,
                                                        ci_type=bootstrap_ci_type,
                                                        random_state=random_state,
                                                        metric_fn=metric_fn)
        ci_low_99, ci_high_99 = bootstrap_confidence_interval(bootstrap_data,
                                                              bootstrap_iterations,
                                                              ci=0.99,  
                                                              ci_type=bootstrap_ci_type,
                                                              random_state=random_state,
                                                              metric_fn=metric_fn)
        precomputed['ci_low'] = ci_low
        precomputed['ci_high'] = ci_high
        precomputed['ci_low_99'] = ci_low_99
        precomputed['ci_high_99'] = ci_high_99
        figure = bootstrap_plot(bootstrap_data,
                                bootstrap_iterations,
                                ci_low,
                                ci_high,
                                random_state,
                                metric_fn=metric_fn,
                                ci_low_99=ci_low_99,
                                ci_high_99=ci_high_99)
    if p_value is None:
        if bootstrap and extra_samplers is None:
            p_value = bootstrap_p_value(bootstrap_data,
                                        value_threshold,
                                        bootstrap_iterations,
                                        alternative,
                                        random_state,
                                        metric_fn=metric_fn,
                                        strategy=strategy)
        elif extra_samplers is None:
            p_value = ttest_1samp(original_metric_y, 
                                  value_threshold, 
                                  alternative=alternative).pvalue
        else:
            p_value = test_many_samples(original_metric_y,
                                        extra_samplers,
                                        same_questions,
                                        data_type,
                                        exec_env,
                                        main_column)
    precomputed['p_value'] = p_value  
    report = report_valtest_stat_significance(p_value=p_value,
                                              extra_samplers_keys=extra_samplers_keys,
                                              data_type=data_type,
                                              semaphore_threshold=semaphore_threshold,
                                              figure=figure,
                                              is_info=is_info)
    return {'report': report, 
            'precomputed': precomputed}
      
            
def test_many_samples(original_metric_y: pd.Series,
                      extra_samplers: tp.Dict[str, tp.Any],
                      same_questions: bool=False,
                      data_type: str='test',
                      exec_env: str='default',
                      main_column: tp.Optional[str]=None) -> float:
    '''
    Функция для подсчета критериев для более чем одной модели
    Parameters:
        original_metric_y: Массив с метрикой валидируемой модели в разрезе объектов
        extra_samplers: Sampler'ы сравниваемых моделей
        same_questions: True, если разные модели отвечали на одинаковые вопросы, False, если на разные
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
        main_column: Название колонки, содержащей ключевую метрику качества; если None, то усредняются все колонки в y
    Return:
        p_value критериев для более чем одной модели
    '''
    if same_questions:
        p_value = test_many_samples_same_questions(original_metric_y,
                                                   extra_samplers,
                                                   data_type,
                                                   exec_env,
                                                   main_column)
    else:
        comparable_answers = get_real_metric(list(extra_samplers.values())[0],
                                             data_type,
                                             exec_env,
                                             main_column)
        p_value = ttest_ind(original_metric_y,
                            comparable_answers,
                            equal_var=False).pvalue
    return p_value
        
        
def test_many_samples_same_questions(original_metric_y: pd.Series,
                                     extra_samplers: tp.Dict[str, tp.Any],
                                     data_type: str='test',
                                     exec_env: str='default',
                                     main_column: tp.Optional[str]=None) -> float:
    '''
    Функция для подсчета критериев для более чем одной модели для случая с ответами на одни вопросы
    Parameters:
        original_metric_y: Массив с метрикой валидируемой модели в разрезе объектов
        extra_samplers: Sampler'ы сравниваемых моделей
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
        main_column: Название колонки, содержащей ключевую метрику качества; если None, то усредняются все колонки в y
    Return:
        p_value критериев для более чем одной модели для случая с ответами на одни вопросы
    '''
    if len(extra_samplers) == 1:
        p_value = calculate_wilcoxon(original_metric_y,
                                     extra_samplers,
                                     data_type,
                                     exec_env,
                                     main_column)
    else:
        p_value = calculate_friedman(original_metric_y,
                                     extra_samplers,
                                     data_type,
                                     exec_env,
                                     main_column)
    return p_value


def calculate_wilcoxon(original_metric_y: pd.Series,
                       extra_samplers: tp.Dict[str, tp.Any],
                       data_type: str='test',
                       exec_env: str='default',
                       main_column: tp.Optional[str]=None) -> float:
    '''
    Функция для подсчета критерия Уилкоксона
    Parameters:
        original_metric_y: Массив с метрикой валидируемой модели в разрезе объектов
        extra_samplers: Sampler сравниваемой модели
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
        main_column: Название колонки, содержащей ключевую метрику качества; если None, то усредняются все колонки в y
    Return:
        p_value критерия Уилкоксона
    '''
    comparable_answers = get_real_metric(list(extra_samplers.values())[0],
                                         data_type,
                                         exec_env,
                                         main_column)
    diff_for_test = original_metric_y - comparable_answers  
    p_value = wilcoxon(diff_for_test).pvalue
    return p_value


def calculate_friedman(original_metric_y: pd.Series, 
                       extra_samplers: tp.Dict[str, tp.Any],
                       data_type: str='test',
                       exec_env: str='default',
                       main_column: tp.Optional[str]=None) -> float:
    '''
    Функция для подсчета критерия Фридмана
    Parameters:
        original_metric_y: Массив с метрикой валидируемой модели в разрезе объектов
        extra_samplers: Sampler'ы сравниваемых моделей
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
        main_column: Название колонки, содержащей ключевую метрику качества; если None, то усредняются все колонки в y
    Return:
        p_value критерия Фридмана
    '''
    is_corrects = [original_metric_y]
    for sampler in extra_samplers.values():
        comparable_answers = get_real_metric(sampler,
                                             data_type,
                                             exec_env,
                                             main_column)
        is_corrects.append(comparable_answers)
    p_value = friedmanchisquare(*is_corrects).pvalue
    return p_value


def get_real_metric(sampler: tp.Union[Sampler, tp.Dict],
                    data_type: str='test',
                    exec_env: str='default',
                    main_column: tp.Optional[str]=None) -> pd.Series:
    '''
    Parameters:
        sampler: Sampler с данными, в которых хранится колонка с ключевой метрикой качества
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
        main_column: Название колонки, содержащей ключевую метрику качества; если None, то усредняются все колонки в y
    Return: 
        Массив с метрикой валидируемой модели в разрезе объектов
    '''
    y_data = get_data(sampler,
                      data_type,
                      'y',
                      exec_env)
    if main_column:
        if main_column not in y_data.columns:
            raise ValueError(
                f"Колонка '{main_column}' не найдена в y_data. "
                f"Доступные колонки: {list(y_data.columns)}"
            )
        original_metric_y = y_data[main_column].dropna()
    else:
        y_data = y_data.dropna().reset_index(drop=True)
        original_metric_y = y_data.mean(axis=1)
        
    return original_metric_y


def calc_metric_values(y_df: pd.DataFrame,
                       scorer: Scorer,
                       metric_agg: str,
                       data_type: str='test') -> float:
    '''
    Рассчитывает метрику через scorer для переданного DataFrame.
    Parameters:
        y_df: DataFrame с данными для расчета метрики 
        scorer: Объект класса Scorer
        metric_agg: Название агрегации метрики (ключ из словаря scorer.metrics)
        data_type: Тип данных, на которых производится расчет метрики
    Return: 
        Скалярное значение рассчитанной метрики 
    '''
    class _MockSampler: pass
    mock_sampler = _MockSampler()
    setattr(mock_sampler,
            data_type,
            {'y': y_df})

    res = scorer.calc(sampler=mock_sampler, 
                      data_type=data_type, 
                      metric_name=metric_agg)
    return float(res[metric_agg])


def bootstrap_p_value(sample: tp.Union[pd.DataFrame, pd.Series, np.ndarray],
                      value_threshold: float,
                      iterations: int=1000,
                      alternative: str='greater',
                      random_state: tp.Optional[int]=None,
                      metric_fn: tp.Callable = None,
                      strategy: str='empirical') -> float:
    '''
    Parameters:
        sample: Массив наблюдений
        value_threshold: Проверяемый порог 
        iterations: Количество bootstrap итераций
        alternative: Тип альтернативной гипотезы ('greater', 'less', 'two-sided')
        random_state: seed генератора случайных чисел
        metric_fn: Функция расчета метрики
        strategy: Стратегия бутстрапа ('rigorous' для теста с центровкой под H0).
    Return: 
        p-value одновыборочного критерия Стьюдента для бутстрапированной выборки
    '''
    rng = np.random.default_rng(random_state)
    n = len(sample)
    observed = metric_fn(sample)
    if strategy == 'rigorous':
        data = np.asarray(sample)
        centered_data = data - observed + value_threshold
        boot_metrics = np.empty(iterations)
        for i in range(iterations):
            boot_metrics[i] = metric_fn(rng.choice(centered_data, size=n, replace=True))
            
        if alternative == 'greater':
            count = np.sum(boot_metrics >= observed)
        elif alternative == 'less':
            count = np.sum(boot_metrics <= observed)
        else:
            count = np.sum(np.abs(boot_metrics - value_threshold) >= np.abs(observed - value_threshold))
    else:
        idx = np.arange(n)
        boot_metrics = np.empty(iterations)
        for i in range(iterations):
            if isinstance(sample, (pd.DataFrame, pd.Series)):
                boot_sample = sample.iloc[rng.choice(idx, size=n, replace=True)]
            else:
                boot_sample = sample[rng.choice(idx, size=n, replace=True)]
            boot_metrics[i] = metric_fn(boot_sample)
    
        if alternative == 'greater':
            count = np.sum(boot_metrics <= value_threshold)
        elif alternative == 'less':
            count = np.sum(boot_metrics >= value_threshold)
        else:
            count = np.sum(np.abs(boot_metrics - value_threshold) >= np.abs(observed - value_threshold))
            
    return float((count + 1) / (iterations + 1))


def bootstrap_confidence_interval(sample: tp.Union[pd.DataFrame, pd.Series, np.ndarray],
                                  iterations: int=1000,
                                  ci: float=0.95,
                                  ci_type: str='percentile',
                                  random_state: tp.Optional[int]=None,
                                  metric_fn: tp.Callable = None) -> tp.Tuple[float, float]:
    '''
    Parameters:
        sample: Массив наблюдений
        iterations: Количество bootstrap итераций
        ci: Уровень доверительного интервала
        ci_type: Тип доверительного интервала ('percentile', 'basic', 'normal')
        random_state: seed генератора случайных чисел
        metric_fn: Функция расчета метрики
    Return: 
        Кортеж с границами доверительного интервала
    '''
    rng = np.random.default_rng(random_state)
    boot_metrics = []
    n = len(sample)
    idx = np.arange(n)
    
    if isinstance(sample, (pd.DataFrame, pd.Series)):
        for _ in range(iterations):
            boot_metrics.append(metric_fn(sample.iloc[rng.choice(idx, size=n, replace=True)]))
    else:
        for _ in range(iterations):
            boot_metrics.append(metric_fn(sample[rng.choice(idx, size=n, replace=True)]))
            
    boot_metrics = np.array(boot_metrics)
    alpha = (1 - ci) / 2
    theta_hat = metric_fn(sample)
    
    if ci_type == 'percentile':
        lower = np.quantile(boot_metrics, alpha)
        upper = np.quantile(boot_metrics, 1 - alpha)
    elif ci_type == 'basic':
        lower = 2 * theta_hat - np.quantile(boot_metrics, 1 - alpha)
        upper = 2 * theta_hat - np.quantile(boot_metrics, alpha)
    elif ci_type == 'normal':
        std = np.std(boot_metrics)
        z = norm.ppf(1 - alpha)
        lower = theta_hat - z * std
        upper = theta_hat + z * std
    return (float(lower), float(upper))


def bootstrap_plot(sample: tp.Union[pd.DataFrame, pd.Series, np.ndarray],
                   iterations: int=1000,
                   ci_low: float=None,
                   ci_high: float=None,
                   random_state: tp.Optional[int]=None,
                   metric_fn: tp.Callable = None,
                   ci_low_99: float=None,   
                   ci_high_99: float=None) -> plt.Figure:
    '''
    Parameters:
        sample: Массив наблюдений
        iterations: Количество bootstrap итераций
        ci_low: Нижняя граница доверительного интервала
        ci_high: Верхняя граница доверительного интервала
        random_state: Seed генератора случайных чисел для воспроизводимости
        metric_fn: Функция расчета метрики
        ci_low_99: Нижняя граница 99% доверительного интервала
        ci_high_99: Верхняя граница 99% доверительного интервала
    Return: 
        График с визуализацией доверительных интервалов
    '''   
    rng = np.random.default_rng(random_state)
    n = len(sample)
    idx = np.arange(n)
    boot_metrics = []
    if isinstance(sample, (pd.DataFrame, pd.Series)):
        for _ in range(iterations):
            boot_metrics.append(metric_fn(sample.iloc[rng.choice(idx, size=n, replace=True)]))
    else:
        for _ in range(iterations):
            boot_metrics.append(metric_fn(sample[rng.choice(idx, size=n, replace=True)]))         
    fig, ax = plt.subplots()
    counts, bins, _ = ax.hist(boot_metrics, bins=30, color='blue', alpha=0.7)
    mu = np.mean(boot_metrics)
    sigma = max(np.std(boot_metrics), 1e-8)
    x = np.linspace(bins[0], bins[-1], 200)
    y = norm.pdf(x, mu, sigma)
    y_scaled = y * len(boot_metrics) * (bins[1] - bins[0])
    ax.plot(x,
            y_scaled,
            color='gray',
            label='normal distribution')
    if ci_low is not None and ci_high is not None:
        ax.axvline(ci_low,
                   color='red',
                   linestyle='--',
                   label='95% CI')
        ax.axvline(ci_high,
                   color='red',
                   linestyle='--')
    if ci_low_99 is not None and ci_high_99 is not None:
        ax.axvline(ci_low_99,
                   color='orange',
                   linestyle=':',
                   label='99% CI')
        ax.axvline(ci_high_99,
                   color='orange',
                   linestyle=':')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Count bootstrap')
    ax.set_title('Bootstrap method')
    ax.legend()
    fig.tight_layout()
    return fig


def get_data(sampler_obj: tp.Union[Sampler, tp.Dict],
             data_type: str, 
             key: str, 
             exec_env: str='default') -> tp.Any:
    '''
    Parameters:
        sampler_obj: Объект сэмплера, содержащий данные для извлечения
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        key: Ключ или имя атрибута для извлечения конкретных данных (например, 'X', 'y')
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
    Return: 
        Содержимое sampler_obj из data_type по ключу key
    '''
    res = None
    if exec_env == 'default':
        res = getattr(sampler_obj, data_type)[key]
    elif exec_env == 'sberds':
        res = sampler_obj[data_type][key]
    return res
