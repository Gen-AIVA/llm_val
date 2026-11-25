from typing import Optional

import numpy as np
import pandas as pd


SET_LABEL_TYPES = (list, tuple, set, frozenset, np.ndarray)


def as_label_set(value):
    """Представляет неупорядоченный набор меток как хешируемую категорию."""
    if not isinstance(value, SET_LABEL_TYPES):
        raise ValueError('Многометочная оценка должна быть списком или множеством меток.')
    if any(pd.api.types.is_scalar(label) and pd.isna(label) for label in value):
        raise ValueError('Внутри набора меток обнаружено пропущенное значение.')
    return frozenset(value)


def canonical_integer_labels(labels: pd.Series) -> Optional[pd.Series]:
    """Преобразует целочисленные строки, которые восстанавливаются без изменений."""
    if pd.api.types.infer_dtype(labels) != 'string':
        return None
    integers = pd.to_numeric(labels, errors='coerce')
    if pd.api.types.is_integer_dtype(integers) and integers.astype(str).eq(labels).all():
        return integers
    return None


def aggregate_dawid_skene(
    ratings: pd.DataFrame,
    n_iter: Optional[int] = None,
    *,
    numeric_strings: bool = False,
) -> pd.Series:
    """Агрегирует оценки по методу Дэвида–Скина и восстанавливает исходные категории."""
    rows = ratings.dropna(subset=['label']).sort_values(['task', 'worker'])
    labels = rows['label']
    if labels.empty:
        return pd.Series(index=pd.Index([], name='task'), dtype=labels.dtype)

    set_labels = labels.map(lambda value: isinstance(value, SET_LABEL_TYPES)).any()
    if set_labels:
        labels = labels.map(lambda value: as_label_set(value) if isinstance(value, SET_LABEL_TYPES) else value)
    integers = canonical_integer_labels(labels)
    labels_to_encode = labels if integers is None else integers
    # Keep the natural class order for scalar labels and a stable order for sets.
    try:
        codes, categories = pd.factorize(labels_to_encode, sort=not set_labels)
    except TypeError:
        codes, categories = pd.factorize(labels_to_encode)
    if integers is not None and not numeric_strings:
        categories = categories.astype(str)
    if len(categories) == 1:
        tasks = pd.Index(rows['task'].unique(), name='task')
        return pd.Series([categories[0]] * len(tasks), index=tasks)

    from crowdkit.aggregation import DawidSkene

    model = DawidSkene() if n_iter is None else DawidSkene(n_iter=n_iter)
    predicted = model.fit_predict(rows.assign(label=codes))
    return predicted.map(dict(enumerate(categories)))
