import numpy as np
import pandas as pd
from typing import List, Tuple

def iterative_binning(data: pd.Series, 
                      min_bin_size: float=0.05) -> list:
    '''
    Разбиение данных на бины с минимальным размером (в процентах).
    Одинаковые значения всегда попадают в один бин.

    Parameters:
        data: Данные для разбиения.
        min_bin_size: Минимальный размер бина (в процентах от общего количества данных).
    Return:
        Список бинов, где каждый бин — это список значений.
    '''
    min_bin_size_len = max(1, round(len(data) * min_bin_size))
    sorted_data = sorted(data)
    bins = []
    current_bin = []

    for cur_elem in range(len(sorted_data)):
        current_bin.append(sorted_data[cur_elem])
        if len(current_bin) >= min_bin_size_len:
            if cur_elem != (len(sorted_data)-1):
                if not np.isclose(sorted_data[cur_elem], sorted_data[cur_elem+1]):
                    bins.append(current_bin)
                    current_bin = []
    if current_bin:
        bins.append(current_bin)

    return bins


def assign_bins(values, edges):
    """
    Назначает каждому значению индекс бина по заданным границам.

    Логика назначения:
      - Бин задаётся парой (lo, hi).
      - Значение попадает в бин, если lo <= value <= hi.
      - Для точечного бина (lo == hi) значение попадает, если value == lo.
      - Бины предполагаются отсортированными и непересекающимися.
      - Возвращается первый подходящий бин.

    Parameters:
        values : Iterable[Number] | pd.Series
                Последовательность значений (например, длины в токенах), для которых нужно
                определить индексы бинов. Допустимы числа или pd.Series.
        edges : Sequence[Tuple[Number, Number]]
            Список бинов в виде пар (lo, hi). Должны быть упорядочены по возрастанию lo.
    Return:
        np.ndarray[int]
            Вектор индексов бинов той же длины, что и `values`. Каждый элемент — индекс
            бина из `edges`, в который попало соответствующее значение.

        Пример
        -------
        >>> assign_bins([2, 5, 9], [(2, 5), (9, 12)])
        array([0, 0, 1])
    """
    res = []
    for val in values:
        for i, (lo, hi) in enumerate(edges):
            if lo <= val <= hi or (lo == hi and val == lo):
                res.append(i)
                break
    return np.array(res)


def bin_by_spec_with_merge(series: pd.Series, min_bin_size: float = 0.05):
    """
    Разбиение числового признака на бины с квантилями внутри сегментов
    и последующим слиянием маленьких бинов с ближайшим соседом по центру.

    Алгоритм:

    1) Пики:
       - Считаем доли значений (`value_counts(normalize=True)`).
       - Все значения с долей ≥ `min_bin_size` считаем «пиками» и создаём для них точечные бины `(v, v)`.

    2) Остатки → сегменты:
       - Берём `rest = s[~s.isin(peaks)]`.
       - Если пики есть, делим `rest` на сегменты:
         * левый сегмент: значения < первого пика;
         * промежуточные сегменты: между соседними пиками;
         * правый сегмент: значения > последнего пика.
         Если пиков нет — весь `rest` один сегмент.

    3) Квантили внутри каждого сегмента:
       - Для сегмента с размером len(seg_values) целевое число интервалов:
            q_num = max(1, int((1 / min_bin_size) * (len(seg_values) / n)))
       - Режем seg_values через pd.qcut()
       - Для каждого интервала из qcut берём (left, right) как (lo, hi) и
         подрезаем по границам соседних пиков (если есть), чтобы не пересекались.
       - Добавляем такие интервальные бины в общий список.

    4) Сортировка бинов.

    5) Слияние маленьких бинов:
       - Вычисляем индексы bin_ids = assign_bins(s, bins) и их доли shares.
       - Находим «маленькие» бины (< min_bin_size). Если нет или остался один бин — выходим.
       - Берём первый маленький бин i, считаем его центр, ищем соседей (левый/правый),
         выбираем соседа с минимальной дистанцией центра (при равенстве — с большей долей).
       - Сливаем пару (i, neighbor_idx): берём все значения s из этих двух бинов,
         ставим новые границы как (min(values), max(values)).
       - Повторяем цикл, пока «маленьких» не останется.

    6) Финальные целочисленные границы:
       - Снова считаем bin_ids = assign_bins(s, bins).
       - Для каждого бина берём фактический min и max значений и приводим к int:
         это и есть итоговые (low, high).

    7) Разметка исходной series:
       - bins_val = assign_bins(series, bins) — индексы бинов для каждого элемента исходной серии.

    8) Подписи:
       - Для точек: просто "v".
       - Для промежуточных (кроме последнего): "[low–high+1)".
       - Для последнего: "[low–high]".

    Parameters:
        series : pd.Series
            Исходный числовой ряд (значения могут быть нечисловыми/NaN — они отбрасываются на шаге очистки).
        min_bin_size : float, по умолчанию 0.05
            Минимальная доля наблюдений для:
            • определения «пика»;
            • порога «малого» бина при слиянии;
            • расчёта целевого числа квантильных интервалов внутри сегмента.

    Return:
        bins : List[Tuple[int, int]]
            Итоговые целочисленные границы бинов (каждая пара — [low, high], обе границы включаются).
        bins_val : np.ndarray[int]
            Индекс бина для каждого элемента *исходной* `series` (по порядку элементов).
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)

    # Пики
    vc = s.value_counts(normalize=True)
    peaks = sorted(vc[vc >= min_bin_size].index.tolist())
    bins = [(float(v), float(v)) for v in peaks]

    # Остатки → сегменты
    rest = s[~s.isin(peaks)]
    if not rest.empty:
        segments = []
        if peaks:
            sorted_peaks = sorted(peaks)
            left_seg = rest[rest < sorted_peaks[0]]
            if not left_seg.empty:
                segments.append((left_seg, None, sorted_peaks[0]))
            for p1, p2 in zip(sorted_peaks, sorted_peaks[1:]):
                mid_seg = rest[(rest > p1) & (rest < p2)]
                if not mid_seg.empty:
                    segments.append((mid_seg, p1, p2))
            right_seg = rest[rest > sorted_peaks[-1]]
            if not right_seg.empty:
                segments.append((right_seg, sorted_peaks[-1], None))
        else:
            segments.append((rest, None, None))

        # Квантили в каждом сегменте
        for seg_values, left_bound, right_bound in segments:
            q_num = max(1, int((1 / min_bin_size) * (len(seg_values) / n)))
            cats = pd.qcut(seg_values, q=np.linspace(0, 1, q_num + 1), duplicates="drop")
            for iv in cats.cat.categories:
                lo, hi = float(iv.left), float(iv.right)
                if left_bound is not None:
                    lo = max(lo, left_bound)
                if right_bound is not None:
                    hi = min(hi, right_bound)
                bins.append((lo, hi))

    bins.sort(key=lambda b: (b[0], b[1]))

    # Слияние мелких бинов с пересчётом границ
    while True:
        bin_ids = assign_bins(s, bins)
        shares = (
            pd.Series(bin_ids)
            .value_counts(normalize=True)
            .reindex(range(len(bins)), fill_value=0)
            .to_numpy()
        )
        small_bins = [i for i, share in enumerate(shares) if share < min_bin_size]
        if not small_bins or len(bins) == 1:
            break

        i = small_bins[0]
        c_small = (bins[i][0] + bins[i][1]) / 2
        neighbors = []
        if i > 0:
            c_left = (bins[i - 1][0] + bins[i - 1][1]) / 2
            neighbors.append((i - 1, abs(c_small - c_left), shares[i - 1]))
        if i < len(bins) - 1:
            c_right = (bins[i + 1][0] + bins[i + 1][1]) / 2
            neighbors.append((i + 1, abs(c_small - c_right), shares[i + 1]))
        neighbor_idx = min(neighbors, key=lambda x: (x[1], -x[2]))[0]

        # объединяем списки значений для пересчёта границ
        ids_to_merge = {i, neighbor_idx}
        merged_values = s[np.isin(bin_ids, list(ids_to_merge))]
        lo_new, hi_new = float(merged_values.min()), float(merged_values.max())

        new_bins = [b for k, b in enumerate(bins) if k not in ids_to_merge]
        new_bins.insert(min(i, neighbor_idx), (lo_new, hi_new))
        bins = new_bins

    # Финальные границы
    bin_ids = assign_bins(s, bins)
    clean_bins = []
    for i in range(len(bins)):
        values_in_bin = s[bin_ids == i]
        clean_bins.append((int(values_in_bin.min()), int(values_in_bin.max())))
    bins = clean_bins

    # Присвоение исходным данным
    bins_val = assign_bins(series, bins)

    return bins, bins_val


def group_indices_by_bins(series, bins_val):
    """
    Группирует индексы элементов исходной `series` по индексам бинов `bins_val`.
    Parameters:
        series : pd.Series
            Исходная серия значений, размер N.
        bins_val : Sequence[int] | np.ndarray
            Индексы бинов для каждого элемента `series`, размер N.
            Предполагается, что это результат `bin_by_spec_with_merge(...)[1]`.
    Return:
        List[List[int]]
            Список списков. Каждый внутренний список — индексы элементов `series`,
            которые принадлежат соответствующему бину.
            Порядок внешнего списка — по возрастанию индекса бина.
            Порядок внутри внутренних списков — как встречались в `series`.

        Пример
        -------
        >>> s = pd.Series([5, 2, 2, 3, 4, 4, 9, 10, 10, 11, 12, 12, 15, 2])
        >>> idx_bins = [3, 0, 0, 0, 1, 1, 1, 3, 2, 2, 2, 2, 3, 0]
        >>> group_indices_by_bins(s, idx_bins)
        [[1, 2, 3, 13], [4, 5, 6], [8, 9, 10, 11], [0, 7, 12]]
    """
    grouped = {}
    for idx, bin_idx in enumerate(bins_val):
        grouped.setdefault(bin_idx, []).append(idx)
    return [grouped[i] for i in sorted(grouped.keys())]
