import numpy as np
from scipy.spatial.distance import cosine, euclidean
from typing import List


def modified_cosine_similarity(vector_a: List[float], 
                               vector_b: List[float]) -> float:
    '''
    Функция, вычисляющая косинусную близость.
    Подсчитывает измененную меру косинусной близости согласно идеологии, где метрика лежит в интервале [0, 1].
    Здесь 0 - значение для разнонаправленных векторов, 1 - для сонаправленных.
    Parameters:
        vector_a: Одноразмерный вектор чисел с плавающей точкой
        vector_b: Одноразмерный вектор чисел с плавающей точкой
    Return:
        Значение модифицированной косинусной близости
    '''
    return 1. - cosine(vector_a, vector_b) / 2.


def modified_euclidean_similarity(vector_a: List[float], 
                                  vector_b: List[float]) -> float:
    '''
    Функция, вычисляющая косинусную близость.
    Подсчитывает измененную меру евклидовой близости между нормализованными единичными векторами, согласно идеологии,
    где метрика лежит в интервале [0, 1]. Здесь 0 - значение для разнонаправленных векторов, 1 - для сонаправленных.
    Parameters:
        vector_a: Одноразмерный вектор чисел с плавающей точкой
        vector_b: Одноразмерный вектор чисел с плавающей точкой
    Return:
        Значение модифицированной эвклидовой близости.
    '''
    len_v_a = np.linalg.norm(vector_a)
    len_v_b = np.linalg.norm(vector_b)
    return 1. - euclidean(vector_a / len_v_a, vector_b / len_v_b) / 2.
