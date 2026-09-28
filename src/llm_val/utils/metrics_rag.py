import numpy as np
import typing as tp

def cosine_distance(embed1: tp.List[float], 
                    embed2: tp.List[float]) -> float:
    '''
    Функция для расчета косинусного расстояния между двумя эмбеддингами
    Parameters:
        embed1: Первый эмбеддинг
        embed2: Второй эмбеддинг
    Return:
        Косинусное расстояние между поданными в функцию эмбеддингами
    '''
    numerator = sum(val1 * val2 for val1, val2 in zip(embed1, embed2))
    denom1 = sum(val**2 for val in embed1)
    denom2 = sum(val**2 for val in embed2)
    cos_dist = 1 - numerator / (np.sqrt(denom1)*np.sqrt(denom2))
    return cos_dist


def sigmoid(x: float) -> float:
    '''
    Функция для расчета сигмоиды
    Parameters:
        x: Аргумент в функцию сигмоиды
    Return:
        Сигмоидное преобразование поданного аргумента
    '''
    return 1 / (1 + np.exp(-x))


def mean_cosine_distance(cos_dist_list: tp.Optional[tp.Iterable[float]]=None,
                         embs1: tp.Optional[tp.Iterable[float]]=None,
                         embs2: tp.Optional[tp.Iterable[float]]=None,
                         norm_fact: tp.Optional[float]=None) -> float:
    '''
    Функция для вычисления среднего косинусного расстояния.
    Можно передать как список готовых косинусных расстояний, так и два списка эмбеддингов.
    Parameters:
        cos_dist_list: список косинусных расстояний
        embs1: список эмбеддингов
        embs2: список эмбеддингов
        norm_fact: параметр нормализации
    Return:
        Среднее косинусное расстояние
    '''
    if cos_dist_list is None:
        cos_dist_list = [cosine_distance(e1, e2) for e1, e2 in zip(embs1, embs2)]
    mean_cos_dist = np.mean(cos_dist_list)
    if norm_fact is not None:
        mean_cos_dist /= sigmoid(norm_fact)
    return mean_cos_dist


def accuracy_cosine_distance(cos_dist_list: tp.List[float], 
                             cos_dist_threshold: float=0.2) -> float:
    '''
    Функция для расчета доли косинусных расстояний, значения которых не превышают порог cos_dist_threshold
    Parameters:
        cos_dist_list: Список косинусных расстояний
        cos_dist_threshold: Порог косинусного расстояния, при превышении которого эмбеддинги считаются далекими, и их мера близости оценивается как 0.
            Если наоборот, то эмбеддинги считаются близкими, и их мера близости оценивается как 1
    Return:
        Доли поданных в функцию косинусных расстояний, значения которых с учетом нормировочного коэффициента не превышают заданный порог
    '''
    return np.mean([int(cos_dist <= cos_dist_threshold) for cos_dist in cos_dist_list])


eval_function = {'MeanCosDist': mean_cosine_distance,
                 'AccuracyCosDist': accuracy_cosine_distance}


def dcg_at_k(relevance_scores: tp.List[int],
             topk: int) -> float:
    '''
    Функция для расчета накопленного дисконтированного прироста (DCG).
    Parameters:
        relevance_scores: Список оценок релевантности документов.
        topk: Количество первых элементов, учитываемых при вычислении DCG.
    Returns:
        Значение DCG для первых topk элементов.
    '''
    return np.sum([(2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance_scores[:topk])])


def ndcg_at_k(relevance_scores: tp.List[int],
              topk: int) -> float:
    '''
    Функция для расчета нормализованного накопленного дисконтированного прироста (nDCG).
    Parameters:
        relevance_scores: Список оценок релевантности документов.
        topk: Количество первых элементов, учитываемых при вычислении nDCG.
    Returns:
        Значение nDCG для первых topk элементов.
    '''
    ideal_dcg = dcg_at_k(sorted(relevance_scores, reverse=True), topk)
    return 0 if ideal_dcg == 0 else dcg_at_k(relevance_scores, topk) / ideal_dcg


def average_precision(relevance_scores: tp.List[int],
                      topk: int) -> float:
    '''
    Функция для расчета средней точности (Average Precision) до позиции k.
    Parameters:
        relevance_scores: Список оценок релевантности документов.
        topk: Количество первых элементов, учитываемых при вычислении средней точности.
    Returns:
        Средняя точность для первых topk элементов.
    '''
    precisions = [
        np.sum(relevance_scores[:i+1]) / (i+1)
        for i in range(topk)
        if relevance_scores[i] == 1
    ]
    return np.mean(precisions) if precisions else 0


def mean_average_precision_at_k(relevance_scores_list: tp.Union[tp.List[int], tp.List[tp.List[int]]],
                                topk: int) -> float:
    '''
    Рассчитывает среднюю точность (Mean Average Precision) до позиции k.
    Parameters:
        relevance_scores_list: Список оценок релевантности (один запрос) или список списков оценок (несколько запросов).
        topk: Количество первых элементов, учитываемых при вычислении средней точности.
    Returns:
        Средняя точность.
    '''
    if isinstance(relevance_scores_list[0], list):
        average_precisions = [
            average_precision(scores, 
                              topk)
            for scores in relevance_scores_list
        ]
    else:
        average_precisions = [average_precision(relevance_scores_list,
                                                topk)]
    return np.mean(average_precisions) if average_precisions else 0.0


def mean_precision_at_k(relevance_scores_list: tp.Union[tp.List[int], tp.List[tp.List[int]]],
                        topk: int) -> float:
    '''
    Рассчитывает среднюю точность (Mean Precision) до позиции k.
    Parameters:
        relevance_scores_list: Список оценок релевантности (один запрос) или список списков оценок (несколько запросов).
        topk: Количество первых элементов, учитываемых при вычислении точности.
    Returns:
        Средняя точность.
    '''
    if isinstance(relevance_scores_list[0], list):
        precisions = [
            np.sum(scores[:topk]) / topk
            for scores in relevance_scores_list
        ]
    else:
        precisions = [np.sum(relevance_scores_list[:topk]) / topk]
    return np.mean(precisions) if precisions else 0.0


def recall_at_k(relevance_scores: tp.List[int],
                topk: int) -> float:
    '''
    Функция для расчета полноты (Recall) до позиции k.
    Parameters:
        relevance_scores: Список оценок релевантности документов.
        topk: Количество первых элементов, учитываемых при вычислении полноты.
    Returns:
        Полнота для первых topk элементов.
    '''
    return 0.0 if np.sum(relevance_scores) == 0 else np.sum(relevance_scores[:topk]) / np.sum(relevance_scores)


def auc_at_k(relevance_scores: tp.List[int],
             topk: int) -> float:
    '''
    Функция для расчета площади под кривой (AUC) до позиции k.
    Parameters:
        relevance_scores: Список оценок релевантности документов.
        topk: Количество первых элементов, учитываемых при вычислении AUC.
    Returns:
        Значение AUC для первых topk элементов.
    '''
    sorted_indices = np.argsort(relevance_scores)[::-1]
    sorted_relevance = np.array(relevance_scores)[sorted_indices]
    auc = np.sum([sorted_relevance[i] * (1 / (i + 1)) for i in range(topk)])
    return auc


def expected_reciprocal_rank(relevance_scores: tp.List[int]) -> float:
    '''
    Функция для расчета ожидаемого обратного ранга (Expected Reciprocal Rank).
    Parameters:
        relevance_scores: Список оценок релевантности документов.
    Returns:
        Ожидаемый обратный ранг для списка оценок релевантности.
    '''
    probabilities = [(2 ** rel - 1) / 2 ** i for i, rel in enumerate(relevance_scores, start=1)]
    return np.sum(probabilities)


def mean_reciprocal_rank(relevance_scores_list: tp.List[tp.List[int]]) -> float:
    '''
    Функция для расчета среднего обратного ранга (Mean Reciprocal Rank).
    Parameters:
        relevance_scores_list: Список списков оценок релевантности для нескольких запросов.
    Returns:
        Средний обратный ранг для всех запросов.
    '''
    ranks = [next((i + 1 for i, rel in enumerate(scores) if rel == 1), 0) for scores in relevance_scores_list]
    return np.mean([1 / rank if rank > 0 else 0 for rank in ranks])


def hit_rate_at_k(relevance_scores_list: tp.List[tp.List[int]],
                  topk: int) -> float:
    '''
    Функция для расчета частоты попадания в верный ответ (Hit Rate) до позиции k.
    Parameters:
        relevance_scores_list: Список списков оценок релевантности для нескольких запросов.
        topk: Количество первых элементов, учитываемых при вычислении частоты попаданий.
    Returns:
        Частота попаданий для всех запросов.
    '''
    hits = [1 if any(scores[:topk]) else 0 for scores in relevance_scores_list]
    return np.mean(hits)


eval_function_ranking = {'nDCG@k': lambda scores, topk: ndcg_at_k(scores, 
                                                                  topk),
                         'MAP@k': lambda scores, topk: mean_average_precision_at_k(scores, 
                                                                                   topk),
                         'MP@k': lambda scores, topk: mean_precision_at_k(scores, 
                                                                          topk),
                         'Recall@k': lambda scores, topk: recall_at_k(scores, 
                                                                      topk),
                         'AUC@k': lambda scores, topk: auc_at_k(scores, 
                                                                topk),
                         'ERR@k': lambda scores, _: expected_reciprocal_rank(scores),
                         'MRR@k': lambda scores, _: mean_reciprocal_rank([scores]),
                         'HitRate@k': lambda scores, topk: hit_rate_at_k([scores], 
                                                                         topk)}