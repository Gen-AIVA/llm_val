import math
import numpy as np
from typing import Union, List


def standardize(data: List[Union[int, float]]):
    """
    Функция-организатор для
    масштабирования данных

    За основу взята статья:
    'Умная нормализация данных'
    habr.com/ru/articles/527334/

    Подробнее:
    M. Huberta and E. Vandervierenb "An adjusted boxplot for skewed
    distributions" Computational Statistics & Data Analysis,
    vol. 52, pp. 5186-5201, August 2008.

    Parameters:
        data: лист с данными для масштабирования
              в диапазон от 0 до 1
    """
    std_params = standardize_fit(data)
    delta = std_params[0]
    scale_interval_border_left = std_params[1]
    scale_interval_border_right = std_params[2]
    scale_interval = std_params[3]
    return standardize_transform(data,
                                 delta,
                                 scale_interval_border_left,
                                 scale_interval_border_right,
                                 scale_interval)


def standardize_fit(data: List[Union[int,
                                     float]]) -> List[Union[np.int64,
                                                            np.float64]]:
    """
    Вычисляет на переданных данных
    медиану и скорректированный интервал,
    которые будут применяться для масштабирования.
    Parameters:
        data: данные, на которых будут происходить вычисления
    Return:
        Кортеж с коэф. дельта и параметрами интервала масштабирования
    """
    np_data = np.array(data)
    median = np.percentile(data, 50)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    mc = medcouple(data, median)
    if mc < 0.0:
        adj_int_left = q1 - 1.5 * math.exp(-3 * mc) * iqr
        adj_int_right = q3 + 1.5 * math.exp(4 * mc) * iqr
        adjusted_interval = (adj_int_left, adj_int_right)
    else:
        adj_int_left = q1 - 1.5 * math.exp(-4 * mc) * iqr
        adj_int_right = q3 + 1.5 * math.exp(3 * mc) * iqr
        adjusted_interval = (adj_int_left, adj_int_right)
    existing_border_left = np_data[np_data >= adjusted_interval[0]].min()
    existing_border_right = np_data[np_data <= adjusted_interval[1]].max()
    adjusted_scale_value = existing_border_right - existing_border_left
    if adjusted_scale_value == 0.0:
        adjusted_scale_value = 1.0

    delta = existing_border_left
    scale_interval_border_left = existing_border_left
    scale_interval_border_right = existing_border_right
    scale_interval = adjusted_scale_value
    return [delta,
            scale_interval_border_left,
            scale_interval_border_right,
            scale_interval]


def standardize_transform(data: List[Union[int, float]],
                          delta: np.float64,
                          scale_interval_border_left: np.float64,
                          scale_interval_border_right: np.float64,
                          scale_interval: np.float64) -> np.ndarray:
    """
    Функция, выполняющая масштабирование.
    Parameters:
        data: Данные, подвергаемые стандартизации
        delta: коэф. сдвига
        scale_interval_border_left: левая граница интервала масштабирования
        scale_interval_border_right: правая граница интервала масштабирования
        scale_interval: интервал масштабирования
    Return:
        масштабированные данные
    """
    np_data = np.array(data)
    lower_mask = np_data < scale_interval_border_left
    upper_mask = np_data > scale_interval_border_right
    np_data[lower_mask] = scale_interval_border_left
    np_data[upper_mask] = scale_interval_border_right
    return (np_data - delta) / scale_interval


def medcouple(cur_data: List[float],
              median: np.float64) -> np.float64:
    """
    Вычисляет робастную меру перекоса medcouple.
    Parameters:
        cur_data: Данные для получения medcouple.
        median: подсчитанная ранее медиана.
    Return:
        medcouple
    """
    z = np.array(cur_data) - median
    lower = z[z <= 0.0]
    upper = z[z >= 0.0]
    upper = upper[:, None]
    standartization = upper - lower
    is_zero = np.logical_and(upper == 0.0, lower == 0.0)
    standartization[is_zero] = np.inf
    spread = upper + lower
    h = spread / standartization
    num_ties = np.sum(lower == 0.0)
    if num_ties:
        # Builds matrix with 0 on diagonal
        # -1 above diagonal and 1 below diagonal
        replacements = np.ones((num_ties, num_ties)) - np.eye(num_ties)
        replacements -= 2*np.triu(replacements)
        # Convert diagonal to anti-diagonal
        replacements = np.fliplr(replacements)
        # Always replace upper right block
        h[:num_ties, -num_ties:] = replacements
    return np.median(h)