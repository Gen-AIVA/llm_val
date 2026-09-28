import numpy as np
import pandas as pd
import typing as tp


def semaphore_by_threshold(value: float,
                           threshold: tp.Tuple[float, float],
                           greater_is_better: bool=True,
                           left_border_is_bigger: bool=True,
                           right_border_is_bigger: bool=True) -> str:
    '''
    Расчет цвета светофора в зависимости от границ.
    Parameters:
        value: Значение
        threshold: Границы светофора
        greater_is_better: True, если чем больше метрика, тем лучше.
        left_border_is_bigger: Если True, то при равенстве метрики левой границе выставляется 
            светофор, соответствующий светофору с правой стороны от левой границы.
        right_border_is_bigger: Если True, то при равенстве метрики правой границе выставляется 
            светофор, соответствующий светофору с правой стороны от правой границы. 
    Return:
        Цвет светофора
    '''
    semaphore = {0: 'red', 1: 'yellow', 2: 'green'}
    if (value < threshold[0]) or (not left_border_is_bigger and value == threshold[0]):
        interval = 0
    elif (value < threshold[1]) or (not right_border_is_bigger and value == threshold[1]):
        interval = 1
    else:
        interval = 2
    if not greater_is_better:
        interval = abs(interval - 2)
    return semaphore[interval]

    
def semaphore_by_threshold_without_yellow(value: float,
                                          threshold: float,
                                          greater_is_better: bool=True,
                                          border_is_bigger: bool=True) -> str:
    '''
    Расчет цвета светофора в зависимости от границ (без желтого светофора).
    Parameters:
        value: Значение
        threshold: Граница светофора
        greater_is_better: True, если чем больше метрика, тем лучше.
        border_is_bigger: Если True, то при равенстве метрики границе выставляется 
            светофор, соответствующий светофору с правой стороны от границы.
    Return:
        Цвет светофора
    '''
    semaphore = {0: 'red', 
                 1: 'green'}
    if (value < threshold) or (not border_is_bigger and value == threshold):
        interval = 0
    else:
        interval = 1
    if not greater_is_better:
        interval = abs(interval - 1)
    return semaphore[interval]


def semaphore_by_threshold_without_red(value: float,
                                       threshold: float,
                                       greater_is_better: bool=True,
                                       border_is_bigger: bool=True) -> str:
    '''
    Расчет цвета светофора в зависимости от границ (без красного светофора).
    Parameters:
        value: Значение
        threshold: Граница светофора
        greater_is_better: True, если чем больше метрика, тем лучше.
        border_is_bigger: Если True, то при равенстве метрики границе выставляется 
            светофор, соответствующий светофору с правой стороны от границы.
    Return:
        Цвет светофора
    '''
    semaphore = {0: 'yellow', 
                 1: 'green'}
    if (value < threshold) or (not border_is_bigger and value == threshold):
        interval = 0
    else:
        interval = 1
    if not greater_is_better:
        interval = abs(interval - 1)
    return semaphore[interval]


def worst_semaphore(semaphore_list: tp.List[str]) -> str:
    '''
    Расчет худшего светофора
    Parameters:
        semaphore_list: Список со светофорами
    Return:
        Цвет худшего светофора
    '''
    if (semaphore_list is None) or (len(semaphore_list) == 0):
        return 'gray'
    semaphore_to_value = {'red': 0, 
                          'yellow': 1, 
                          'green': 2, 
                          'gray': 3}
    value_to_semaphore = {value: key for key, value in semaphore_to_value.items()}
    worst_value = min([semaphore_to_value[i] for i in semaphore_list])
    return value_to_semaphore[worst_value]


def tricky_semaphore(value: float,
                     semaphore_list: tp.List[str],
                     greater_is_better: bool=True,
                     threshold_metric: tp.Tuple[float, float]=(0.4, 0.6),
                     red_freq_threshold: float=0.2,
                     yellow_freq_threshold: float=0.2):
    '''
    Хитрый светофор (см. таблицу 4.14) 
    # TODO поменять номер таблицы после утверждения методики
    Parameters:
        value: Ключевая метрика качества
        semaphore_list: Список светофоров за критерии асессоров
        greater_is_better: Чем больше агрегированная метрика, тем лучше?
        threshold_metric: Границы светофора для агрегированного значения
        red_freq_threshold: Граница по относительному числу "красных" светофоров
        yellow_freq_threshold: Граница по относительному числу "желтых" светофоров
    Return:
        Цвет светофора
    '''
    color = 'yellow'
    value_color = semaphore_by_threshold(value, 
                                         threshold_metric, 
                                         greater_is_better)
    freq_color = pd.Series(semaphore_list, dtype='str').value_counts(normalize=True).to_dict()
    if (value_color == 'red') or ((freq_color.get('red', 0) > red_freq_threshold)):
        color = 'red'
    elif (value_color == 'green') and (freq_color.get('yellow', 0) < yellow_freq_threshold) and (freq_color.get('red', 0) <= 0):
        color = 'green'
    return color


def tricky_semaphore_agent(value: float,
                           semaphore_list: tp.List[str],
                           greater_is_better: bool=True,
                           threshold_metric: tp.Tuple[float, float]=(0.4, 0.6),
                           red_freq_threshold: float=0.2) -> str:
    '''
    Светофор для теста 6.3.6
    Parameters:
        value: Значение ключевой метрики качества
        semaphore_list: Список светофоров за второстепенные критерии
        greater_is_better: Чем больше агрегированная метрика, тем лучше?
        threshold_metric: Границы светофора для ключевой метрики
        red_freq_threshold: Граница по относительной доле красных светофоров
    Return:
        Цвет светофора
    '''
    value_color = semaphore_by_threshold(value,
                                         threshold_metric,
                                         greater_is_better)
    freq_color = (pd.Series(semaphore_list, dtype='str')
                  .value_counts(normalize=True).to_dict()
                  if semaphore_list else {})
    n_red_frac = freq_color.get('red', 0)
    if value_color == 'red' or n_red_frac >= red_freq_threshold:
        result = 'red'
    elif value_color == 'green' and n_red_frac == 0:
        result = 'green'
    else:
        result = 'yellow'
    return result


def quantile_semaphore(value: float,
                       threshold: tp.Tuple[float, float],
                       inner_is_better: bool=True) -> str:
    '''
    Расчет цвета светофора в зависимости от квантиля при двусторонних границах.
    Parameters:
        value: Значение
        threshold: Границы светофора
        inner_is_better: True, если светофор лучше при попадании внутрь границ
    Return:
        Цвет сфетофора
    '''
    semaphore = {0: 'red', 1: 'yellow', 2: 'green'}
    if (1 - threshold[0]) / 2 < value < (1 + threshold[0]) / 2:
        interval = 2
    elif (1 - threshold[1]) / 2 <= value <= (1 + threshold[1]) / 2:
        interval = 1
    else:
        interval = 0
    if not inner_is_better:
        interval = abs(interval - 2)
    return semaphore[interval]


def shuffle_ci_semaphore(mean_metric: float,
                         std_metric: float,
                         interval: tp.Tuple[float, float],
                         std_threshold: float) -> str:
    '''
    Расчет цвета светофора для теста с перемешиванием вариантов.
    Parameters:
        mean_metric: Среднее значение целевой метрики
        std_metric: Значение стандартного отклонения целевой метрики
        interval: Границы доверительного интервала
        std_threshold: Порог для светофора по стандартному отклонению
    Return:
        Цвет сфетофора
    '''
    if (mean_metric < interval[0]) or (mean_metric > interval[1]):
        return 'red'
    if std_metric > std_threshold:
        return 'yellow'
    return 'green'


def local_frequency_semaphore(value: float,
                              ideal_responses: float,
                              threshold: tp.Tuple[float, float]) -> str:
    '''
    Расчет цвета светофора в зависимости от частоты одного элемента относительно средней частоты.
    Parameters:
        value: Элемент массива частот
        ideal_responses: Математическое ожидание частот элементов в теоретической равномерно распределенной выборке
        threshold: Границы светофора
    Return:
        Цвет светофора.
    '''
    color = 'red'
    yellow_bound, red_bound = threshold[0], threshold[1]
    if (value >= red_bound * ideal_responses) or (value <= ideal_responses / red_bound):
        color = 'red'
    elif (value >= yellow_bound * ideal_responses) or (value <= ideal_responses / yellow_bound):
        color = 'yellow'
    else:
        color = 'green'
    return color


def frequency_semaphore(values: tp.List[float],
                        threshold: tp.Tuple[float, float]) -> tp.List[str]:
    '''
    Расчет цвета светофора в зависимости от частоты элемента относительно среднего числа элементов в выборке.
    Parameters:
        values: Массив частот элементов
        all_letters: Список всех букв, представленных как варинты ответов в answer_options
        threshold: Границы светофора
    Return:
        Список цветов светофора для каждой частоты буквы в true_answer.
    '''
    semaphore_list, color = [], 'red'
    ideal_responses = np.mean(values)
    for value in values:
        color = local_frequency_semaphore(value=value,
                                          ideal_responses=ideal_responses,
                                          threshold=threshold)
        semaphore_list.append(color)
    return semaphore_list


def proportion_semaphore(semaphores: tp.Union[pd.Series, tp.List[str]],
                         freq_threshold: tp.Tuple[float, float]) -> str:
    '''
    Расчет цвета итогового светофора в зависимости от доли различных светофоров в списке
    Parameters:
        semaphores: Список светофоров для каждого элемента выборки
        freq_threshold: Доля допустимых желтых и красных светофоров
    Return:
        Цвет светофора
    '''
    semaphores_series = pd.Series(semaphores)
    yellows = (semaphores_series == 'yellow').sum() / len(semaphores_series)
    reds = (semaphores_series == 'red').sum() / len(semaphores_series)
    if reds > freq_threshold[1]:
        color = 'red'
    elif yellows + reds > freq_threshold[0]:
        color = 'yellow'
    else:
        color = 'green'
    return color


def quanity_semaphore(semaphore_list: tp.List[str], 
                      threshold: tp.Tuple[int, int]) -> str:
    '''
    Расчет цвета итогового светофора в зависимости от количества вхождений частных светофоров
    Parameters:
        values: Массив c цветами частных светофоров
        threshold: Границы светофора
    Return:
        Цвет светофора
    '''
    yellow_count = sum(1 for color in semaphore_list if color == 'yellow')
    red_count = sum(1 for color in semaphore_list if color == 'red')
    if red_count >= threshold[0]:
        overall_semaphore = 'red'
    elif yellow_count >= threshold[1]:
        overall_semaphore = 'yellow'
    else:
        overall_semaphore = 'green'
    return overall_semaphore


def quantile_semaphore_without_yellow(value: float,
                                      threshold: tp.Tuple[float, float],
                                      inner_is_better: bool = True) -> str:
    '''
    Расчет цвета светофора в зависимости от квантиля при двусторонних границах (без желтого светофора).
    Parameters:
        value: Значение метрики.
        threshold: Границы светофора (нижняя и верхняя границы).
        inner_is_better: True, если светофор лучше при попадании внутрь границ.
    Returns:
        Цвет светофора: 'red' или 'green'.
    '''
    if inner_is_better:
        if threshold[0] <= value <= threshold[1]:
            return 'green'
        else:
            return 'red'
    else:
        if value < threshold[0] or value > threshold[1]:
            return 'green'
        else:
            return 'red'


def dialog_len_semaphore(stats: tp.Dict[str, float],
                         significance_level: float=0.05,
                         mean_decrease_threshold: float=0.3,
                         greater_is_better: bool=True) -> str:
    '''
    Выставление светофора для одной проверки. Красный, если p-value значимо и средняя КМ на подвыборке отклоняется от средней КМ по датасету в сторону ухудшения на заданную относительную величину; иначе зеленый
    Parameters:
        stats: Словарь со статистиками проверки (p_value, subset_mean, full_mean)
        significance_level: Уровень значимости
        mean_decrease_threshold: Относительная граница падения средней КМ на подвыборке
        greater_is_better: True, если чем больше метрика, тем лучше
    Return:
        Цвет светофора ('red' или 'green')
    '''
    is_significant = stats['p_value'] < significance_level
    if greater_is_better:
        is_degraded = stats['subset_mean'] <= stats['full_mean'] * (1 - mean_decrease_threshold)
    else:
        is_degraded = stats['subset_mean'] >= stats['full_mean'] * (1 + mean_decrease_threshold)
    color = 'red' if is_significant and is_degraded else 'green'
    
    return color


def semaphore_reproducibility_agent(orig_diversity: float,
                                    relative_diff: float,
                                    is_creative: bool,
                                    threshold_pp: tp.Tuple[float, float]=(0.03, 0.05),
                                    rel_threshold: float=0.5) -> str:
    '''
    Расчет цвета светофора для теста стабильности воспроизведения ответов агента.
    Для творческой задачи оценивается смысловое разнообразие, для точной - языковое.
    Parameters:
        orig_diversity: Значение разнообразия на исходных данных
        relative_diff: Относительная разница разнообразия между исходными и аугментированными данными
        is_creative: True для творческой задачи (смысловое разнообразие), False для точной (языковое)
        threshold_pp: Нижний и верхний пороги разнообразия в долях (3 п.п. и 5 п.п.)
        rel_threshold: Порог относительной разницы
    Return:
        Цвет светофора
    '''
    low_threshold, high_threshold = threshold_pp
    if is_creative:
        is_green = (orig_diversity > high_threshold) or (relative_diff > rel_threshold)
        is_red = (orig_diversity < low_threshold) and (relative_diff < rel_threshold)
    else:
        is_green = (orig_diversity < low_threshold) or (relative_diff > rel_threshold)
        is_red = (orig_diversity > high_threshold) and (relative_diff < rel_threshold)
    color = 'green' if is_green else ('red' if is_red else 'yellow')

    return color


def semaphore_by_threshold_for_lentest(est_by_len_result: pd.DataFrame,
                                       est_by_cluster_result: pd.DataFrame,
                                       est_by_context_result: pd.DataFrame=None,
                                       business_threshold: float=None,
                                       data_type: str='test',
                                       min_count: int=50,
                                       semaphore_threshold_ratio: float=0.8,
                                       semaphore_threshold_p_val: float=0.05):
    '''
    Выставление светофора на основе результатов статистических тестов
    Parameters:
        est_by_len_result: Результат теста по длинам запросов
        est_by_cluster_result: Результат теста по кластерам
        est_by_context_result: Результат теста по длинам контекста
        business_threshold: Минимальный проходной порог по КМ (при наличии)
        data_type: Тип данных
        min_count: Минимальное число запросов в бине/кластере
        semaphore_threshold_ratio: Граница светофора относительного падения качества
        semaphore_threshold_p_val: Граница светофора уровня значимости падения качества
    Return:
        Цвет светофора
    '''
    color_id = 0
    last_bin_len = est_by_len_result.shape[0] - 1

    for row in est_by_len_result.iterrows():
        if row[1]['p-value'] is None or pd.isna(row[1]['p-value']):
            continue
        if row[1]['p-value'] < semaphore_threshold_p_val and row[1]['diff_ratio'] < semaphore_threshold_ratio:   
            if (business_threshold is not None
                and (row[0] == 0 or row[0] == last_bin_len)
                and row[1]['business_thr_p_val'] is not None
                and not pd.isna(row[1]['business_thr_p_val'])
                and row[1]['business_thr_p_val'] < semaphore_threshold_p_val
                and row[1]['bin_size'] >= min_count):
                color_id = max(color_id, 3)
            else:
                color_id = max(color_id, 2)
        else:
            color_id = max(color_id, 1)

    for row in est_by_cluster_result.iterrows():
        if row[1]['p-value'] is None or pd.isna(row[1]['p-value']):
            continue
        if row[1]['cluster_size'] >= min_count:
            if row[1]['p-value'] < semaphore_threshold_p_val and row[1]['diff_ratio'] < semaphore_threshold_ratio:
                color_id = max(color_id, 2)
            else:
                color_id = max(color_id, 1)

    if est_by_context_result is not None:
        last_bin_ctx = est_by_context_result.shape[0] - 1
        for row in est_by_context_result.iterrows():
            if row[1]['p-value'] is None or pd.isna(row[1]['p-value']):
                continue
            if row[1]['p-value'] < semaphore_threshold_p_val and row[1]['diff_ratio'] < semaphore_threshold_ratio:   
                if (business_threshold is not None
                    and (row[0] == 0 or row[0] == last_bin_ctx)
                    and row[1]['business_thr_p_val'] is not None
                    and not pd.isna(row[1]['business_thr_p_val'])
                    and row[1]['business_thr_p_val'] < semaphore_threshold_p_val
                    and row[1]['bin_size'] >= min_count):
                    color_id = max(color_id, 3)
                else:
                    color_id = max(color_id, 2)
            else:
                color_id = max(color_id, 1)

    semaphore_dict = {3: 'red', 2: 'yellow', 1: 'green', 0: 'gray'}
    color = semaphore_dict[color_id]

    return color
    
    
def stat_significance_semaphore(p_value: tp.Dict[str, tp.Optional[float]],
                                significance_level: float=0.05,
                                greater_is_better: bool=True) -> str:
    '''
    Расчет светофора по статистической значимости бизнес-критерия
    Parameters:
        p_value: Значения p-value для направленных гипотез
        significance_level: Уровень значимости для проверки бизнес-критерия
        greater_is_better: True, если чем больше метрика, тем лучше
    Return:
        Цвет светофора
    '''
    better_p_value = p_value['greater'] if greater_is_better else p_value['less']
    worse_p_value = p_value['less'] if greater_is_better else p_value['greater']
    if better_p_value is None or worse_p_value is None:
        color = 'gray'
    elif better_p_value < significance_level:
        color = 'green'
    elif worse_p_value < significance_level:
        color = 'red'
    else:
        color = 'yellow'
    return color
