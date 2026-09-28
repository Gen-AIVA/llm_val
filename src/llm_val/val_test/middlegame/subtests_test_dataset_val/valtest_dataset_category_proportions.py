import typing as tp
import numpy as np
import pandas as pd

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.psi_calculate import psi
from llm_val.utils.report_helper import semaphore_by_threshold_without_yellow


def report_valtest_dataset_category_proportions(result_df: pd.DataFrame,
                                                is_info: bool,
                                                psi_value: float,
                                                threshold: float,
                                                extra_categories: tp.List[str],
                                                missing_categories: tp.List[str],
                                                apriori_dict: tp.Dict[str, float],
                                                max_deviation: float = 0.0) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        result_df: DataFrame с детальным распределением категорий.
        is_info: True, если тест информативный.
        psi_value: Значение PSI.
        threshold: Порог PSI для светофора (один порог).
        extra_categories: Лишние категории в данных.
        missing_categories: Отсутствующие категории.
        apriori_dict: Ожидаемые пропорции категорий.
        max_deviation: Максимальное абсолютное отклонение доли.
    Return:
        report в соответствии с концепцией llm_val
    '''
    if is_info or not apriori_dict:
        color = 'gray'
    elif extra_categories or missing_categories:
        color = 'red'
    else:
        color = semaphore_by_threshold_without_yellow(psi_value,
                                                      threshold=threshold,
                                                      greater_is_better=False)
    if not apriori_dict:
        test_result = 'Не проводилась'
    elif color == 'red':
        test_result = 'Не пройдена'
    else:
        test_result = 'Пройдена'

    comment = build_comment(extra_categories,
                            missing_categories,
                            apriori_dict,
                            psi_value,
                            threshold,
                            max_deviation)

    summary_df = pd.DataFrame(
        {
            'Проверка': 'Проверка соответствия фактического распределения категорий в датасете априорному распределению\
                из отчёта о разработке [основная проверка]',
            'Результат': [test_result],
            'Метрика': [psi_value],
            'Комментарий': [comment],
        }
    )

    return {'semaphore': color, 
            'result_plots': [], 
            'result_dataframes': [summary_df, result_df]}


def valtest_dataset_category_proportions(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                         apriori_dict: tp.Dict[str, float],
                                         data_type: str = 'test',
                                         threshold: float = 0.1,
                                         is_info: bool = False,
                                         exec_env: tp.Literal['default', 'sberds'] = 'default',
                                         category_stats: tp.Optional[tp.Dict[str, tp.Any]] = None,
                                         **kwargs: tp.Any) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Валидационный тест для проверки соответствия фактического распределения категорий в датасете
    априорному распределению из отчёта о разработке.
    Parameters:
        sampler: Объект класса Sampler с заполненной колонкой 'thema'
        apriori_dict: Словарь с наименованием категории и её априорной долей
        data_type: Тип данных, на которых производится расчёт метрики
        threshold: Порог PSI для светофора (один порог)
        is_info: True, если тест информативный
        exec_env: Среда выполнения ('default' или 'sberds')
        category_stats: Предпосчитанные данные (из precomputed)
    Return:
        Словарь в соответствии с концепцией llm_val.
    '''
    category_col = 'scenario'
    if category_stats is not None:
        psi_value = category_stats.get('psi_value')
        actual_props = category_stats.get('actual_props')
        extra_categories = category_stats.get('extra_categories')
        missing_categories = category_stats.get('missing_categories')
    elif not apriori_dict:
        psi_value = None
        actual_props = {}
        extra_categories = []
        missing_categories = []
    else:
        data = getattr(sampler, data_type)['X'] if exec_env == 'default' else sampler[data_type]['X']
        if 'session_id' in data.columns:
            data = data.groupby('session_id', as_index=False).nth(0)
        if category_col not in data.columns:
            raise KeyError(f"Колонка '{category_col}' отсутствует в данных. Доступные колонки: {list(data.columns)}")
        actual_counts = data[category_col].value_counts(normalize=True)

        extra_categories = list(set(actual_counts.index) - set(apriori_dict.keys()))
        missing_categories = list(set(apriori_dict.keys()) - set(actual_counts.index))

        themes = list(apriori_dict.keys())
        prior_distribution = np.array([apriori_dict[theme] for theme in themes])
        real_distribution = np.array([actual_counts.get(theme, 0) for theme in themes])

        psi_value = psi(prior_distribution, real_distribution)

        actual_props = dict(actual_counts)

    if psi_value is None or psi_value == 0:
        max_deviation = 0
    else:
        max_deviation = (
            max(abs(apriori_dict[cat] - actual_props.get(cat, 0)) for cat in apriori_dict) if apriori_dict else 0
        )

    if apriori_dict:
        sorted_categories = sorted(apriori_dict.keys())
        result_df = pd.DataFrame(
            {
                'Тематика': sorted_categories,
                'Реальное распределение': [actual_props.get(cat, 0) for cat in sorted_categories],
                'Априорное распределение': [apriori_dict[cat] for cat in sorted_categories],
            }
        )
    else:
        result_df = pd.DataFrame({'Проверка': ['Категоризация не предусмотрена']})

    report = report_valtest_dataset_category_proportions(result_df,
                                                         is_info,
                                                         psi_value if psi_value is not None else 0,
                                                         threshold,
                                                         extra_categories if apriori_dict else [],
                                                         missing_categories if apriori_dict else [],
                                                         apriori_dict,
                                                         max_deviation)
    precomputed = {'psi_value': psi_value,
                   'actual_props': actual_props,
                   'extra_categories': extra_categories,
                   'missing_categories': missing_categories,
                   'max_deviation': max_deviation}
    return {'report': report, 
            'precomputed': precomputed}


def build_comment(extra_categories: tp.List[str],
                  missing_categories: tp.List[str],
                  apriori_dict: tp.Dict[str, float],
                  psi_value: float,
                  threshold: float,
                  max_deviation: float) -> str:
    '''Собрать текстовый комментарий.'''
    parts = []
    if extra_categories:
        parts.append(f'Лишние категории: {", ".join(extra_categories)}\n')
    if missing_categories:
        parts.append(f'Отсутствуют категории: {", ".join(missing_categories)}\n')
    if apriori_dict:
        parts.append(f'PSI = {psi_value:.4f} (порог: {threshold})\n')
        parts.append(f'Макс. отклонение доли = {max_deviation:.4f}')
    else:
        parts.append('Категоризация не предусмотрена отчётом о разработке')
    return ''.join(parts)
