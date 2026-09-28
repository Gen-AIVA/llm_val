import numpy as np
import typing as tp
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold


def report_valtest_criterion_informativeness(uninformative_share: float,
                                             uninformative_list: tp.List[str],
                                             measures: tp.Dict[str, float],
                                             missing: tp.List[str],
                                             data_type: str,
                                             uninformativeness_threshold: float,
                                             semaphore_threshold: tp.Tuple[float, float],
                                             is_info: bool) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а в соответствии с концепцией llm_val.
    Parameters:
        uninformative_share: доля неинформативных критериев (только из найденных в данных)
        uninformative_list: список названий неинформативных критериев
        measures: словарь вида {критерий: значение меры информативности}
        missing: критерии из параметра criterion_columns, не найденные в данных
        data_type: Тип данных, на которых производится расчет метрики.
            Hint: название атрибута в sampler'е
        uninformativeness_threshold: пороговое значение меры (ниже которого критерий считается неинформативным)
        semaphore_threshold: границы светофора для доли неинформативных критериев
        is_info: информационный ли тест
    Returns:
        report в формате llm_val
    '''
    if is_info:
        color = 'gray'
    else:
        color = semaphore_by_threshold(value=uninformative_share,
                                       threshold=semaphore_threshold,
                                       greater_is_better=False,
                                       left_border_is_bigger=False)
    rows = []
    for criterion, measure in measures.items():
        rows.append({
            'Критерий': criterion,
            f'Мера информативности на {data_type}': round(measure, 4) if not pd.isna(measure) else 'нет данных',
            f'Неинформативный (мера < {uninformativeness_threshold})': measure < uninformativeness_threshold if not pd.isna(measure) else True
        })
    table = pd.DataFrame(rows)

    summary_text = (
        f'Доля «неинформативных» критериев – {uninformative_share:.1%} '
        f'({len(uninformative_list)}/{len(measures)} критериев).'
    )
    if uninformative_list:
        summary_text += '\n\nСписок «неинформативных» критериев:\n'
        for crit in uninformative_list:
            val_str = f"{measures[crit]:.4f}" if not pd.isna(measures[crit]) else "нет данных"
            summary_text += f'• {crit} (мера {val_str} < {uninformativeness_threshold})\n'

    summary_df = pd.DataFrame({
        'Показатель': ['Доля неинформативных критериев', 'Порог меры', 'Результат теста', 'Сводка', 'Не найденные критерии'],
        'Значение': [f'{uninformative_share:.1%}', uninformativeness_threshold, color, summary_text, missing]
    })

    return {
        'semaphore': color,
        'result_plots': [],
        'result_dataframes': [table, summary_df]
    }


def valtest_criterion_informativeness(sampler: tp.Union[Sampler, tp.Dict[str, tp.Dict[str, pd.DataFrame]]],
                                      data_type: str = 'test',
                                      criterion_columns: tp.Optional[tp.List[str]] = None,
                                      uninformativeness_threshold: float = 0.05,
                                      measure_func: tp.Optional[tp.Callable[[tp.List[float]], float]] = None,
                                      semaphore_threshold: tp.Tuple[float, float] = (0.0, 0.5),
                                      is_info: bool = False,
                                      exec_env: str = 'default',
                                      measures: tp.Optional[tp.Dict[str, float]] = None,
                                      **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Для каждого критерия вычисляется мера информативности на основе нормализованных
    оценок асессоров. Критерий считается неинформативным, если мера ниже порога.
    Parameters:
        sampler: объект Sampler или словарь с данными в зависимости от exec_env
        data_type: Тип данных, на которых производится расчет метрики.
            Hint: название атрибута в sampler'е
        criterion_columns: список колонок с критериями.
            Если None, используются колонки, заканчивающиеся на '_metric'.
        uninformativeness_threshold: пороговое значение меры (ниже которого критерий неинформативен)
        measure_func: функция для вычисления меры информативности.
            Сигнатура: (values: List[float]) -> float.
            По умолчанию – дисперсия нормализованных значений.
        semaphore_threshold: границы светофора для доли неинформативных критериев
        is_info: если True, светофор всегда 'gray'
        exec_env: окружение выполнения ('default' или 'sberds')
        measures: предварительно вычисленные значения мер.
            Если не None, пересчёт не выполняется.
    Returns:
        Словарь в соответствии с концепцией llm_val
    '''
    if measures is not None:
        existing_measures = {k: v for k, v in measures.items() if not pd.isna(v) or v is not None}
        uninformative_list = [crit for crit, val in measures.items() if pd.isna(val) or val < uninformativeness_threshold]
        uninformative_share = len([c for c in uninformative_list if c in existing_measures]) / len(existing_measures) if existing_measures else 0.0
        missing = []
    else:
        if exec_env == 'sberds':
            df = sampler[data_type]['X']
        else:
            df = getattr(sampler, data_type)['X']

        if measure_func is None:
            measure_func = default_measure

        measures, uninformative_list, uninformative_share, missing = compute_informativeness(
            df, 
            criterion_columns, 
            measure_func, 
            uninformativeness_threshold)

    precomputed_measures_out = {k: v for k, v in measures.items() if not pd.isna(v)}

    report = report_valtest_criterion_informativeness(uninformative_share=uninformative_share,
                                                      uninformative_list=uninformative_list,
                                                      measures=measures,
                                                      missing = missing,
                                                      data_type=data_type,
                                                      uninformativeness_threshold=uninformativeness_threshold,
                                                      semaphore_threshold=semaphore_threshold,
                                                      is_info=is_info)
    precomputed = {'measures': precomputed_measures_out,
                   'criterion_columns': criterion_columns}
    return {'report': report, 
            'precomputed': precomputed}


def default_measure(values: tp.List[float]) -> float:
    '''
    Стандартная мера информативности – дисперсия после min-max нормализации.
    Parameters:
        values: список числовых значений (уже нормализованных)
    Returns:
        Дисперсия
    '''
    return float(np.var(values))


def compute_informativeness(df: pd.DataFrame,
                            criterion_columns: tp.Optional[tp.List[str]],
                            measure_func: tp.Callable[[tp.List[float]], float],
                            uninformativeness_threshold: float) -> tp.Tuple[tp.Dict[str, float], tp.List[str], float, tp.List[str]]:
    '''
    Вычисляет меру информативности для каждого критерия.
    Parameters:
        df: DataFrame (одна строка – оценка одного асессора)
        criterion_columns: список колонок с критериями (опционально), иначе ищем колонки в названия которых входит "_metric"
        measure_func: функция, вычисляющая меру информативности по нормализованным значениям
        uninformativeness_threshold: порог меры (ниже которого критерий считается неинформативным)
    Returns:
        Кортеж:
            - measures: словарь вида {критерий: значение меры}
            - uninformative_list: список неинформативных критериев
            - uninformative_share: доля неинформативных критериев (только из найденных в данных)
            - missing_columns: список колонок, которых нет в данных
    '''
    if criterion_columns is None:
        criterion_columns = [
            col for col in df.columns
            if col.endswith('_metric') and pd.api.types.is_numeric_dtype(df[col])
        ]
        if not criterion_columns:
            criterion_columns = []
            
    existing = [col for col in criterion_columns if col in df.columns]
    missing = [col for col in criterion_columns if col not in df.columns]

    measures = {}
    uninformative_list = []

    for col in existing:
        all_vals = df[col].dropna().values.reshape(-1, 1)
        if all_vals.size == 0:
            measures[col] = np.nan
            uninformative_list.append(col)
            continue
        scaler = MinMaxScaler()
        norm_vals = scaler.fit_transform(all_vals).flatten()
        measure = measure_func(norm_vals)
        measures[col] = measure
        if measure < uninformativeness_threshold:
            uninformative_list.append(col)

    for col in missing:
        measures[col] = np.nan
        uninformative_list.append(col)

    total_existing = len(existing)
    uninformative_share = len([c for c in uninformative_list if c in existing]) / total_existing if total_existing > 0 else 0.0

    return measures, uninformative_list, uninformative_share, missing