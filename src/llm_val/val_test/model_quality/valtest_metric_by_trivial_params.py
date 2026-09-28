import typing as tp

import numpy as np
import pandas as pd
from physt import h1
from scipy import stats
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score
from scipy.stats import norm, ttest_1samp

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.utils.retries import call_with_retries
from llm_val.utils.openai_utils import get_openai_embeddings
from llm_val.utils.report_helper import semaphore_by_threshold_for_lentest


def report_valtest_metric_by_trivial_params(est_by_len_result_: pd.DataFrame,
                                            est_by_cluster_result_: pd.DataFrame,
                                            est_by_context_result_: tp.Optional[pd.DataFrame]=None,
                                            business_threshold: float=None,
                                            data_type: str='test',
                                            fig: tp.Any = None,
                                            min_count: int=50,
                                            semaphore_threshold_ratio: float=0.8,
                                            semaphore_threshold_p_val: float=0.05,
                                            is_info: bool=False):
    '''
    Создание report'а
    Parameters:
        est_by_len_result_: Датафрейм с результатами теста на завсисимость динамики метрики от длины запроса
        est_by_cluster_result_: Датафрейм с результатами теста на завсисимость динамики метрики от кластера запроса
        est_by_context_result_: Датафрейм с результатами теста на завсисимость динамики метрики от длины контекста
        business_threshold: Минимальный проходной порог по КМ (при наличии)
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        fig: Графики
        semaphore_threshold_ratio: Граница светофора относительного падения качества (1 - <значение указанное в тексе методики>)
        semaphore_threshold_p_val: Граница светофора уровня значимости падения качества
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    if is_info:
        color = 'gray'
    else:
        color = semaphore_by_threshold_for_lentest(est_by_len_result_,
                                                   est_by_cluster_result_,
                                                   est_by_context_result_,
                                                   business_threshold,
                                                   data_type,
                                                   min_count,
                                                   semaphore_threshold_ratio,
                                                   semaphore_threshold_p_val)

    est_by_len_result = format_df(est_by_len_result_.copy())
    est_by_cluster_result = format_df(est_by_cluster_result_.copy(), is_cluster = True)

    if est_by_context_result_ is not None:
        est_by_context_result = format_df(est_by_context_result_.copy())
    else:
        est_by_context_result = None
    
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = fig
    res_dict['result_dataframes'] = [est_by_len_result, est_by_cluster_result, est_by_context_result]
    return res_dict


def valtest_metric_by_trivial_params(sampler: tp.Union[Sampler, tp.Dict],
                                     scorer: Scorer,
                                     embedding_client: tp.Optional[tp.Any]=None,
                                     embedding_model_name: tp.Optional[str]='EmbeddingsGigaR',
                                     data_type: str='test',
                                     business_threshold: float=None,
                                     context_test: bool=True,
                                     token_counter_func: tp.Optional[tp.Callable]=None,
                                     clustering_algo: str='kmeans',
                                     bins: tp.Union[str, int]='fd',
                                     metric_agg: str='single_mean',
                                     stattest_method: str='bootstrap',
                                     n_bootstrap: int=1000,
                                     min_count_size: int=50,
                                     min_count_pct: float=0.1,
                                     semaphore_threshold_ratio: float=0.8,
                                     semaphore_threshold_p_val: float=0.05,
                                     random_state: int=73,
                                     greater_is_better: tp.Optional[bool]=True,
                                     is_info: bool=False,
                                     timeout: float=60.0,
                                     retry_delay: float=0.1,
                                     max_attempts: int=3,
                                     exec_env: str='default',
                                     len_questions: tp.List[int]=None,
                                     embeddings: np.ndarray=None,                                     
                                     len_context: tp.List[int]=None,
                                     est_by_len_result: tp.Optional[pd.DataFrame]=None,
                                     est_by_cluster_result: tp.Optional[pd.DataFrame]=None,
                                     est_by_context_result: tp.Optional[pd.DataFrame]=None,
                                     **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на анализ динамики ключевой метрики качества модели в зависимости от простых статистик
    Parameters:
        sampler: Объект класса Sampler
        scorer: Объект класса Scorer
        embedding_client: OpenAI-совместимый клиент с методом embeddings.create. Обязателен при is_long_answer=True
        embedding_model_name: Название эмбеддинг-модели, передаваемое в embeddings.create
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        business_threshold: Минимальный проходной порог по КМ (при наличии). Необходим для выставления красного светофора
        context_test: флаг проведения проверки для контекста
        token_counter_func: Функция подсчета токенов, List[str] -> np.ndarray[int]
            Принимает тексты запросов и возвращает их  длины в токанех
            Не рекомендуется передавать при наличии колонки llm_prompt_tokens в трейсах, так как токенизация с помощью данной функции будет иметь высший приоритет
        clustering_algo: Алгоритм кластеризации ('kmeans' или 'hdbscan'). Параметры выбираются по умолчанию, подробнее можно посмотреть в соотвествующих модулях
        bins: Название способа бинаризации как параметр функции np.histogram_bin_edges или количество бинов
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        stattest_method: Название статтеста
                         - 'welch' - t-тест Уэлча
                         - 'MW' - U-тест Манна-Уитни
                         - 'bootstrap' - bootstrap
            Метод будет автоматически изменен на бутстрап, если формат метрики не подходит под условия статтеста
        n_bootstrap: Количество итераций бутстрапирования
        min_count_size: Минимальное количество семплов в бине или кластере для корректирования светофора
        min_count_pct: Минимальное доля семплов в бине или кластере для корректирования светофора
            Количество семплов в бинах в хвостах распределения по длинам будет корректироваться на основании этих двух значений
        semaphore_threshold_ratio: Граница светофора для относительного падения качества (1 - <значение указанное в тексе методики>)
        semaphore_threshold_p_val: Граница светофора для уровня значимости падения качества
        random_state: Рандом стейт
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
        exec_env: Переменная исполняемого окружения ('default' или 'sberds')
        len_questions: опицональный результат подсчета длин запросов
            Если не None, то вычисление не производится, и на основе переданного аргумента строятся графики. И если est_by_len_result=None, то также считаются статистики.
        embeddings: опицональный результат вычисления эмдеддингов запросов
            Если не None и est_by_cluster_result=None, то вычисление не производится, и на основе переданного аргумента считаются статистики.
        len_context: опицональный результат подсчета длин контекста
            Если не None, то вычисление не производится, и на основе переданного аргумента строятся графики. И если est_by_context_result=None, то также считаются статистики.
        est_by_len_result: опицональный результат теста для зависимости корректности запроса от его длины
            Если не None, то подсчет статистики не производится, и на основе переданного аргумента выставляется светофор.
        est_by_cluster_result: опицональный результат теста для зависимости корректности запроса от его кластера
            Если не None, то подсчет статистики не производится, и на основе переданного аргумента выставляется светофор.
        est_by_context_result: опицональный результат теста для зависимости корректности запроса от длины контекста
            Если не None, то подсчет статистики не производится, и на основе переданного аргумента выставляется светофор.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if exec_env == 'default':
        data = getattr(sampler, data_type)
    elif exec_env == 'sberds':
        data = sampler[data_type]

    y = data['y']

    is_dialogue = 'dialogue' in data['X'].columns
    text_col = 'dialogue' if is_dialogue else 'input_query'
    texts_tmp = data['X'].groupby('session_id')[text_col].first()
    questions_text_formated = questions_text = questions = texts_tmp.reindex(y.index).to_list()
    if is_dialogue:
        questions_text_formated = [dialogue_to_text(dialogue) for dialogue in questions]
        questions_text = [dialogue_to_text_clean(dialogue) for dialogue in questions]

    if len_questions is None and token_counter_func is None:
        if 'traces' in data.keys():
            token_series = data['traces'].sort_values('start_time_ns').groupby('session_id')['llm_prompt_tokens'].first()
            len_questions = token_series.reindex(y.index).to_list()
        else:
            len_questions = [len(x) for x in questions_text]
        if is_dialogue:
            dialogue_len = [len(x) for x in questions]
            len_questions = [int(x / z) for x, z in zip(len_questions, dialogue_len)]

    min_count = max(min_count_size, int(len(y)*min_count_pct))

    if ((y > 1) | (y < 0)).any().any() or metric_agg != 'single_mean':
        stattest_method = 'bootstrap'
    
    est_by_len_result_precomputed = est_by_len_result is not None
    est_by_cluster_result_precomputed = est_by_cluster_result is not None
    est_by_context_result_precomputed = est_by_context_result is not None
    len_questions, fig_question = calc_len_and_draw_plot(questions=questions_text,
                                                         y=y,
                                                         min_count=min_count,
                                                         scorer=scorer,
                                                         title='разбиение по длине запросов',
                                                         bins=bins,
                                                         metric_agg=metric_agg,
                                                         n_bootstrap=n_bootstrap,
                                                         random_state=random_state,
                                                         len_questions=len_questions,
                                                         token_counter_func=token_counter_func,
                                                         is_dialogue=is_dialogue)
    if est_by_len_result is None:
        est_by_len_result = est_len_stat(len_questions,
                                         y,
                                         scorer,
                                         min_count,
                                         business_threshold,
                                         bins,
                                         metric_agg,
                                         stattest_method,
                                         n_bootstrap,
                                         random_state,
                                         greater_is_better=greater_is_better)

    if embeddings is None:
        embeddings = call_with_retries(get_openai_embeddings,
                                       timeout,
                                       retry_delay,
                                       max_attempts,
                                       (embedding_client, questions_text_formated, embedding_model_name))
    if est_by_cluster_result is None:
        est_by_cluster_result = cluster_stat(embeddings,
                                            y,
                                            questions_text_formated,
                                            scorer,
                                            bins,
                                            metric_agg,
                                            stattest_method,
                                            n_bootstrap,
                                            random_state,
                                            clustering_algo=clustering_algo,
                                            greater_is_better=greater_is_better)

    fig_context = None
    est_by_context_result_report = None
    if context_test and ('traces' in data.keys()):
        if len_context is None:
            traces = data['traces'][data['traces']['llm_model'].notna()]
            tokens_df = traces.groupby('session_id')['llm_total_tokens'].sum()
            len_context = tokens_df.reindex(y.index).to_list()
            if is_dialogue:
                len_context = [int(x / y) for x, y in zip(len_context, dialogue_len)]
        len_context, fig_context = calc_len_and_draw_plot(questions=questions,
                                                          y=y,
                                                          min_count=min_count,
                                                          scorer=scorer,
                                                          title='разбиение по длине контекста',
                                                          bins=bins,
                                                          metric_agg=metric_agg,
                                                          n_bootstrap=n_bootstrap,
                                                          random_state=random_state,
                                                          len_questions=len_context)
        if est_by_context_result is None:
            est_by_context_result = est_len_stat(len_context,
                                                 y,
                                                 scorer,
                                                 min_count,
                                                 business_threshold,
                                                 bins,
                                                 metric_agg,
                                                 stattest_method,
                                                 n_bootstrap,
                                                 random_state,
                                                 greater_is_better=greater_is_better)
        est_by_context_result_report = est_by_context_result.copy()

    if not greater_is_better:
        if not est_by_len_result_precomputed:
            est_by_len_result['business_thr_ratio'] = 1 / est_by_len_result['business_thr_ratio']
            est_by_len_result['diff_ratio'] = 1 / est_by_len_result['diff_ratio']
        if not est_by_cluster_result_precomputed:
            est_by_cluster_result['diff_ratio'] = 1 / est_by_cluster_result['diff_ratio']
        if not est_by_context_result_precomputed:
            if est_by_context_result is not None:
                est_by_context_result['business_thr_ratio'] = 1 / est_by_context_result['business_thr_ratio']
                est_by_context_result['diff_ratio'] = 1 / est_by_context_result['diff_ratio']

    fig = [fig_question, fig_context]
    report = report_valtest_metric_by_trivial_params(est_by_len_result,
                                                     est_by_cluster_result,
                                                     est_by_context_result,
                                                     business_threshold,
                                                     data_type,
                                                     fig,
                                                     min_count,
                                                     semaphore_threshold_ratio,
                                                     semaphore_threshold_p_val,
                                                     is_info)

    precomputed = {'est_by_len_result':
                   est_by_len_result.copy(),
                   'est_by_cluster_result':
                   est_by_cluster_result.copy(),
                   'est_by_context_result':
                   est_by_context_result_report,
                   'len_questions': 
                   deepcopy(len_questions),
                   'embeddings': 
                   deepcopy(embeddings),
                   'len_context': 
                   deepcopy(len_context)}

    return {'report': report,
            'precomputed': precomputed}


def est_len_stat(len_questions: tp.List[int], 
                 y: tp.Union[pd.DataFrame, pd.Series],
                 scorer: Scorer,
                 min_count: int,
                 business_threshold: float=None,
                 bins: tp.Union[str, int]='fd',
                 metric_agg: str='single_mean',
                 stattest_method: str='bootstrap',
                 n_bootstrap: int=100,
                 random_state: int=73,
                 greater_is_better: bool=True) -> pd.DataFrame:
    '''
    Подсчет статистик по длинам вопросов
    Parameters:
        len_questions: Длины каждого вопроса в токенах
        y: DataFrame или Series с метриками в разрезе вопросов
        scorer: Объект класса Scorer
        min_count: Минимальное число запросов в бине
        business_threshold: Минимальный проходной порог по КМ (при наличии)
        bins: Название способа бинаризации как параметр функции np.histogram_bin_edges или количество бинов
        metric_agg: Название метрики агрегации
        stattest_method: Название статтеста
                         - 'welch' - t-тест Уэлча
                         - 'MW' - U-тест Манна-Уитни
                         - 'bootstrap' - bootstrap
        n_bootstrap: Количество итераций бутстрапирования
        random_state: Рандом стейт
        greater_is_better: True, если чем больше метрика, тем лучше
    Return:
        Датафрейм со всей статистикой
    '''
    bins_val, bins_bound = calc_bins_stat(len_questions,
                                          min_count,
                                          bins)
    bin_list = np.unique(bins_val)
    bins_val = np.array(bins_val)
    result = []
    for i in bin_list:
        y_chunk = y.iloc[bins_val == i]
        y_other = y.iloc[bins_val != i]
        y_chunk_metric = calc_metric_values(y_chunk, scorer, metric_agg)
        y_other_metric = calc_metric_values(y_other, scorer, metric_agg)
        diff_ratio = (y_chunk_metric / y_other_metric) if y_other_metric > 0 else 0
        y_chunk_size = len(y_chunk)
        y_other_size = len(y_other)

        if business_threshold is None:
            business_thr_p_val = None
            business_thr_ratio = None
        else:
            if ((y > 1) | (y < 0)).any().any() or metric_agg != 'single_mean':
                business_thr_p_val = bca_pvalue(y_chunk,
                                                scorer,
                                                business_threshold,
                                                metric_agg,
                                                n_bootstrap,
                                                random_state,
                                                greater_is_better)
            else:
                alternative = 'less' if greater_is_better else 'greater'
                business_thr_p_val = ttest_1samp(y_chunk,
                                                 business_threshold,
                                                 alternative=alternative).pvalue
                
            business_thr_ratio = (y_chunk_metric / business_threshold) if business_threshold > 1e-6 else 0

        p_val = calc_one_sided_test(y_chunk,
                                    y_other,
                                    scorer,
                                    stattest_method,
                                    metric_agg,
                                    n_bootstrap,
                                    random_state,
                                    greater_is_better)
        result.append((y_chunk_metric, y_other_metric, y_chunk_size, y_other_size, p_val,
                       diff_ratio, business_thr_p_val, business_thr_ratio, bins_bound[i]))
    result_df = pd.DataFrame(result,
                             columns = ['bin_metric', 'other_metric', 'bin_size', 'other_size', 'p-value',
                                        'diff_ratio', 'business_thr_p_val', 'business_thr_ratio', 'bounds'])

    return result_df


def cluster_stat(embeddings: np.ndarray,
                 y: tp.Union[pd.DataFrame, pd.Series],
                 questions: tp.List[str],
                 scorer: Scorer,
                 bins: tp.Union[str, int]='fd',
                 metric_agg: str='single_mean',
                 stattest_method: str='bootstrap',
                 n_bootstrap: int=100,
                 random_state: int=73,
                 clustering_algo: str='kmeans',
                 hdbscan_min_cluster_size: int=15,
                 greater_is_better: bool=True) -> pd.DataFrame:
    '''
    Подсчет бинов и статистик по ним
    Parameters:
        embeddings: Эмбеддинги каждого запроса
        y: DataFrame или Series с метриками в разрезе вопросов
        questions: Текст каждого запроса
        scorer: Объект класса Scorer
        bins: Название способа бинаризации как параметр функции np.histogram_bin_edges или количество бинов
        metric_agg: Название метрики агрегации
        stattest_method: Название статтеста
                         - 'welch' - t-тест Уэлча
                         - 'MW' - U-тест Манна-Уитни
                         - 'bootstrap' - bootstrap
        n_bootstrap: Количество итераций бутстрапирования
        random_state: Рандом стейт
        clustering_algo: Алгоритм кластеризации ('kmeans' или 'hdbscan')
        hdbscan_min_cluster_size: Минимальный размер кластера для HDBSCAN
        greater_is_better: True, если чем больше метрика, тем лучше
    Return:
        Датафрейм со всей статистикой
    '''
    norms = np.linalg.norm(embeddings, axis=1)
    embeddings = embeddings / norms[:, np.newaxis]

    if clustering_algo == 'hdbscan':
        clusters = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size).fit(embeddings).labels_
        if (clusters == -1).any():
            noise_label = clusters.max() + 1
            clusters = np.where(clusters == -1, noise_label, clusters)
    elif clustering_algo == 'kmeans':
        max_k = min(12, len(embeddings) - 1)
        silhouette_scores = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++',
                           n_init=10, max_iter=100, random_state=random_state)
            kmeans.fit(embeddings)
            silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))
        n_clusters = np.argmax(silhouette_scores) + 2

        clusters = KMeans(n_clusters=n_clusters, init='k-means++',
                          n_init=25,
                          max_iter=500,
                          random_state=random_state,
                          algorithm='lloyd')\
                   .fit(embeddings)\
                   .labels_
    else:
        raise ValueError(f"Неизвестный clustering_algo: {clustering_algo}. Допустимо: 'kmeans', 'hdbscan'.")

    unique_clusters = sorted(set(clusters))
    n_clusters = len(unique_clusters)

    r_samp = get_representative_samples(embeddings, clusters, questions)

    res_list = []
    for cl in unique_clusters:
        y_chunk = y.iloc[clusters == cl]
        y_other = y.iloc[clusters != cl]
        y_chunk_metric = calc_metric_values(y_chunk, scorer, metric_agg)
        y_other_metric = calc_metric_values(y_other, scorer, metric_agg)

        p_val = calc_one_sided_test(y_chunk,
                                    y_other,
                                    scorer,
                                    stattest_method,
                                    metric_agg,
                                    n_bootstrap,
                                    random_state,
                                    greater_is_better)
        diff_ratio = (y_chunk_metric / y_other_metric) if y_other_metric > 0 else 0
        cluster_size = len(y_chunk)
        other_size = len(y_other)
        res_list.append((y_chunk_metric, y_other_metric, cluster_size, other_size, p_val, diff_ratio, r_samp[cl]))
    cols = ['cluster_metric', 'other_metric', 'cluster_size', 'other_size', 'p-value', 'diff_ratio',  'sample_questions']
    res_df = pd.DataFrame(res_list, columns = cols)
    return res_df


def calc_metric_values(data: pd.Series, 
                       scorer: Scorer,
                       metric_agg: str='single_mean',
                       raw_output = False) -> tp.Union[tp.Dict[str, float], float]:
    '''
    Подсчет метрики качества
    Parameters:
        data: Series с значениями метрики качества в разрезе вопросов
        scorer: Объект класса Scorer
        metric_agg: Название метрики агрегации
    Return:
        Dict с подсчитанными метриками
    '''
    sampler = Sampler(test={'X': None, 'y': pd.DataFrame(data)})
    if raw_output:
        result = scorer.calc(sampler,
                             'test',
                             metric_agg)
    else:
        result = list(scorer.calc(sampler,
                       'test',
                       metric_agg).values())[0]
        
    return result


def enforce_min_bin_count(len_questions: tp.List[int], 
                          min_count: int, 
                          bins='fd') -> tp.List[float]:
    '''
    Объединение бинов для более точного подсчета статистик краевых
    Parameters:
        len_questions: Series с длиной вопросов
        min_count: Минимальное число запросов в бине
        bins: Название способа бинаризации как параметр функции np.histogram_bin_edges или количество бинов
    Return:
        List с границами бинов
    '''
    edges = np.histogram_bin_edges(len_questions, bins=bins)
    counts, _ = np.histogram(len_questions, bins=edges)
    
    inds = list(range(len(edges)))
    inds_to_drop = set()
    left = 1
    while len(counts) > 2 and counts[0] < min_count:
        inds_to_drop.add(left)
        counts[1] += counts[0]
        counts = counts[1:]
        left += 1
    if len(inds_to_drop) > 0:
        edges = [edges[i] for i in inds if i not in inds_to_drop]
    
    inds = list(range(len(edges)))
    inds_to_drop = set()
    right = len(edges)-2
    while len(counts) > 2 and counts[-1] < min_count:
        inds_to_drop.add(right)
        counts[-2] += counts[-1]
        counts = counts[:-1]
        right -= 1
    if len(inds_to_drop) > 0:
        edges = [edges[i] for i in inds if i not in inds_to_drop]

    return edges


def calc_bins_stat(len_questions: tp.List[int],
                   min_count: int,
                   bins: tp.Union[str, int]='fd') -> tp.Tuple[tp.List, tp.List]:
    '''
    Подсчет бинов и статистик по ним
    Parameters:
        len_questions: Список длин каждого вопроса
        min_count: Минимальное число запросов в бине
        bins: Название способа бинаризации как параметр функции np.histogram_bin_edges или количество бинов
    Return:
        List с номерами бинов для каждого вопроса;
        List с границами бинов.
    '''
    edges = enforce_min_bin_count(len_questions, min_count, bins)
    hist = h1(len_questions, bins=np.array(edges))
    bins_val = [hist.find_bin(i) for i in len_questions]
    bins_bound = [f'{np.floor(i[0]):.0f}-{np.floor(i[1]):.0f}' for i in hist.bins]
    return bins_val, bins_bound


def calc_len_and_draw_plot(questions: pd.Series,
                           y: tp.Union[pd.DataFrame, pd.Series],
                           min_count: int,
                           scorer: Scorer,
                           title: tp.Optional[str]=' ',
                           bins: tp.Union[str, int]='fd',
                           metric_agg: str='single_mean',
                           n_bootstrap: int=1000,
                           random_state: int=73,
                           len_questions: tp.List[int]=None,
                           token_counter_func: tp.Optional[tp.Callable]=None,
                           is_dialogue: bool=False):
    '''
    Подсчет длин запросов и построение графика
    Parameters:
        questions: Series с текстами запросов, либо с диалогами (если is_dialogue=True)
        y: DataFrame или Series с метриками в разрезе вопросов
        min_count: Минимальное число запросов в бине
        scorer: Объект класса Scorer
        bins: Название способа бинаризации
        metric_agg: Название метрики агрегации
        n_bootstrap: Количество итераций бутстрапирования
        random_state: Рандом стейт
        len_questions: Опциональные предпосчитанные длины
        token_counter_func: Функция подсчета токенов
        is_dialogue: True, если questions содержит диалоги (список кортежей
            (query_id, question, answer)); тогда длина считается как средняя длина вопросов
    Return:
        Список длин и фигура
    '''
    if len_questions is None:
        if is_dialogue:
            len_questions = mean_dialogue_question_lengths(questions, token_counter_func)
        else:
            len_questions = token_counter_func(questions)
    bins_val, bin_bounds = calc_bins_stat(len_questions,
                                          min_count,
                                          bins)

    fig = draw_plot(pd.DataFrame(y),
                    bins_val,
                    bin_bounds,
                    scorer,
                    title,
                    metric_agg,
                    n_bootstrap,
                    quantile = 0.95,
                    random_state = random_state)

    return len_questions, fig


def calc_one_sided_test(data1: pd.Series,
                        data2: pd.Series,
                        scorer: Scorer,
                        stattest_method: str = 'bootstrap',
                        metric_agg: str = 'single_mean',
                        n_bootstrap: int = 200,
                        random_state: int = 73,
                        greater_is_better: bool=True) -> tp.Dict[str, tp.Dict[str, float]]:
    '''
    Выполняет односторонний статистический тест для двух наборов данных.
    Вычисляет вероятность того, что метрика является на первом наборе данных ниже, чем на втором.

    Parameters:
        data1: Series с данными первой группы
        data2: Series с данными второй группы
        scorer: Объект класса Scorer
        stattest_method: Название статтеста
                         - 'welch' - t-тест Уэлча
                         - 'MW' - U-тест Манна-Уитни
                         - 'bootstrap' - bootstrap
        metric_agg: Название метрики агрегации
        n_bootstrap: Количество итераций бутстрапирования
        random_state: Рандом стейт
        greater_is_better: True, если чем больше метрика, тем лучше
    Return:
        Вероятность
    '''
    if greater_is_better:
        alternative = 'less'
    else:
        alternative = 'greater'
    if stattest_method == 'welch':
        pr = stats.ttest_ind(data1, data2, alternative=alternative).pvalue
    elif stattest_method == 'MW':
        pr = stats.mannwhitneyu(data1, data2, alternative=alternative).pvalue
    else:
        pr = calc_pr_bootstrap(data1, data2, scorer, metric_agg,
                               n_bootstrap, random_state, greater_is_better)

    return pr


def calc_pr_bootstrap(data1: pd.Series,
                      data2: pd.Series,
                      scorer: Scorer,
                      metric_agg: str = 'single_mean',
                      n_bootstrap: int = 200,
                      random_state: int = 73,
                      greater_is_better: bool=True) -> tp.Dict[str, float]:
    '''
    Вычисляет p-value для одностороннего теста (метрика data1 < метрика data2)
    с помощью бутстрапа.

    Parameters:
        data1: Series с данными первой группы
        data2: Series с данными второй группы
        scorer: Объект класса Scorer
        metric_agg: Название метрики агрегации
        n_bootstrap: Количество итераций бутстрапирования
        random_state: Рандом стейт
        greater_is_better: True, если чем больше метрика, тем лучше
    Return:
        Вероятность
    '''
    rng = np.random.default_rng(random_state)
    differences = []
    for _ in range(n_bootstrap):
        sample1 = data1.sample(frac=1, random_state=rng, replace=True)
        sample2 = data2.sample(frac=1, random_state=rng, replace=True)

        metric1 = calc_metric_values(sample1, scorer, metric_agg)
        metric2 = calc_metric_values(sample2, scorer, metric_agg)

        if greater_is_better:
            diff = metric2 - metric1
        else:
            diff = metric1 - metric2
        differences.append(diff)

    pr = sum(1 for d in differences if d < 0) / n_bootstrap

    return pr


def bca_pvalue(y: tp.Union[pd.Series, pd.DataFrame],
               scorer: Scorer,
               business_threshold: float,
               metric_agg: str='single_mean',
               n_bootstrap: int=5000,
               random_state: tp.Optional[int]=None,
               greater_is_better: bool=True) -> float:
    '''
    Односторонний BCa bootstrap p-value для H₀: metric >= business_threshold.
    Parameters:
        y: DataFrame или Series с метриками в разрезе вопросов
        scorer: Объект класса Scorer
        business_threshold: Минимальный проходной порог по КМ
        metric_agg: Название метрики агрегации
        n_bootstrap: Количество итераций бутстрапирования
        random_state: Рандом стейт
        greater_is_better:
            True  → H₀: metric >= threshold  (H₁: metric < threshold)
            False → H₀: metric <= threshold  (H₁: metric > threshold)
    Return:
        p-value в [0, 1]. p < alpha => отвергаем H0 на уровне alpha,
        т.е. метрика статистически значимо ниже порога.
    '''
    rng = np.random.default_rng(random_state)
    n = len(y)
    eps = 1.0 / n_bootstrap

    theta_hat = float(calc_metric_values(y, scorer, metric_agg))
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = calc_metric_values(y.iloc[idx], scorer, metric_agg)

    prop_less = float(np.clip(np.mean(boot_stats < theta_hat), eps, 1.0 - eps))
    z0 = float(norm.ppf(prop_less))

    all_idx = np.arange(n)
    jack = np.array([calc_metric_values(y.iloc[all_idx != i], scorer, metric_agg)
                     for i in range(n)])
    diffs = jack.mean() - jack
    sum_sq = float(np.sum(diffs ** 2))
    sum_cube = float(np.sum(diffs ** 3))
    a = sum_cube / (6.0 * sum_sq ** 1.5) if sum_sq > 0 else 0.0

    if greater_is_better:
        cmp = boot_stats <= business_threshold
    else:
        cmp = boot_stats >= business_threshold
    p_emp = float(np.clip(np.mean(cmp), eps, 1.0 - eps))
    z_p = float(norm.ppf(p_emp))
    v = z_p - z0
    z_alpha = z0 - v / (1.0 + a * v)
    pvalue = float(norm.cdf(z_alpha))

    return pvalue


def get_representative_samples(embeddings: np.ndarray, 
                               clusters: np.ndarray, 
                               questions: tp.List[str],
                               n_samples: int=3) -> tp.List[tp.List[str]]:
    '''
    Семплинг наиболее близких к центроиду текстов для каждого из предопредлеленных кластеров
    Parameters:
        embeddings: Эмбеддинги каждого запроса
        clusters: Кластер каждого запроса
        questions: Текст каждого запроса
        n_samples: Количество примеров для каждого кластера
    Return:
        List для каждого кластера с List'ами наиболее репрезентативных примеров запросов
    '''
    n_clusters = len(np.unique(clusters))
    representatives = []
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        cluster_texts = [questions[i] for i in cluster_indices]
        
        centroid = np.mean(cluster_embeddings, axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        closest_indices = np.argsort(distances)[:n_samples]
        
        representatives.append([cluster_texts[i] for i in closest_indices])
    
    return representatives


def calc_conf_interval(data: pd.DataFrame,
                       bins: tp.List[int],
                       scorer: Scorer,
                       metric_agg: str='single_mean',
                       n_bootstrap: int=100,
                       quantile: float=0.95,
                       random_state: int=73) -> tp.Dict[str, tp.Any]:
    '''
    Подсчет доверительных интервалов для каждого бина
    Parameters:
        data: Данные
        bins: Бины для каждого вопроса
        scorer: Объект класса Scorer
        metric_agg: Название метрики агрегации
        n_bootstrap: Количество итераций бутстрапирования
        quantile: Квантиль доверительного интервала. Должен быть в диапазоне (0; 1)
        random_state: Рандом стейт ¯\_(ツ)_/¯
    Return:
        Два dict'а с значениями статистики и p-value для каждой метрики
    '''
    stat_bins = {}
    grouped = data.groupby(pd.Series(bins, index=data.index))
    for i, group in grouped:
        mean_value = calc_metric_values(group,
                                        scorer,
                                        metric_agg,
                                        raw_output=True)
        stat_bins[i] = {'mean': mean_value}
        if ((data > 1) | (data < 0)).any().any():
            stat_bins[i]['conf_int'] = calc_conf_interval_by_bootstrap(group,
                                                                       scorer,
                                                                       metric_agg,
                                                                       n_bootstrap,
                                                                       quantile,
                                                                       random_state)
        else:
            stat_bins[i]['conf_int'] = calc_conf_interval_estimation(group,
                                                                     mean_value,
                                                                     quantile)
    return stat_bins


def calc_conf_interval_by_bootstrap(data: pd.DataFrame,
                                    scorer: Scorer,
                                    metric_agg: str='single_mean',
                                    n_bootstrap: int=100,
                                    quantile: float=0.95,
                                    random_state: int=73) -> tp.Dict[str, tp.Tuple[float]]:
    '''
    Подсчет доверительного интервала бутстрапированием
    Parameters:
        data: DataFrame с значениями метрик качества в разрезе вопросов
        scorer: Объект класса Scorer
        metric_agg: Название метрики агрегации
        n_bootstrap: Количество итераций бутстрапирования
        quantile: Квантиль доверительного интервала. Должен быть в диапазоне (0; 1)
        random_state: Рандом стейт ¯\_(ツ)_/¯
    Return:
        Словарь с доверительными интервалами метрик
    '''
    metric_values = []
    rng = np.random.default_rng(random_state)
    for _ in range(n_bootstrap):
        sample = data.sample(frac=1, 
                             random_state=rng,
                             replace=True)
        sampler = Sampler(test={'X': None, 'y': pd.DataFrame(sample)})
        metric_values.append(scorer.calc(sampler,
                                         'test',
                                         metric_agg))
    metric_values = {j: sorted([values[j] for values in metric_values]) for j in metric_values[0].keys()}
    return {k: calculate_quantile(metric_values[k]) for k in metric_values.keys()}


def calc_conf_interval_estimation(data: pd.DataFrame, 
                                  mean_value: tp.Dict[str, float],
                                  quantile: float=0.95) -> tp.Dict[str, tp.Tuple[float, float]]:
    '''
    Подсчет доверительного интервала
    Parameters:
        data: DataFrame с значениями метрик качества в разрезе вопросов
        mean_value: Словарь со средними значениями метрик 
        quantile: Квантиль доверительного интервала. Должен быть в диапазоне (0; 1)
    Return:
        Словарь с доверительными интервалами метрик
    '''
    mean_value = pd.DataFrame(mean_value, index=[0])
    n_sample = len(data)
    if (quantile >= 1) or (quantile <= 0):
        raise ValueError('quantile должен быть в диапазоне (0; 1)')
    t = norm.ppf((1 - quantile) / 2 + quantile)
    first = n_sample / (t ** 2 + n_sample)
    second = mean_value + (t ** 2) / (2 * n_sample)
    third = t * np.sqrt(mean_value * (1 - mean_value) / n_sample + (t / (2 * n_sample)) ** 2)
    left = (first * (second - third)).to_dict('records')[0]
    right = (first * (second + third)).to_dict('records')[0]
    return {k: (left[k], right[k]) for k in left.keys()}


def calculate_quantile(metric_values: tp.List[float],
                       quantile: float=0.95) -> tp.Tuple[float, float]:
    '''
    Подсчет границ доверительного интервала
    Parameters:
        metric_values: Список значений метрик
        quantile: Квантиль доверительного интервала. Должен быть в диапазоне (0; 1)
    Return:
        Левую и правую границы доверительного интервала
    '''
    left = (1 - quantile) / 2
    right = quantile + left
    left_bound = int(left * len(metric_values))
    right_bound = int(right * len(metric_values))
    return metric_values[left_bound], metric_values[right_bound]


def draw_plot(data: pd.DataFrame,
              question_bins: tp.List[int],
              bins: tp.List[str],
              scorer: Scorer,
              title: str,
              metric_agg: tp.Optional[str]='single_mean',
              n_bootstrap: tp.Optional[int]=100,
              quantile: tp.Optional[float]=0.95,
              random_state: tp.Optional[int]=73) -> tp.Any:
    '''
    Подсчет доверительных интервалов для каждого бина
    Parameters:
        data: Данные
        question_bins: Бины для каждого вопроса
        bins: Уникальные бины
        scorer: Объект класса Scorer
        metric_agg: Название метрики агрегации
        n_bootstrap: Количество итераций бутстрапирования
        quantile: Квантиль доверительного интервала. Должен быть в диапазоне (0; 1)
        random_state: Рандом стейт
    Return:
        График для разбиения по длинам
    '''
    bins_stat = calc_conf_interval(data,
                                   question_bins,
                                   scorer,
                                   metric_agg,
                                   n_bootstrap,
                                   quantile,
                                   random_state)

    key = list(list(bins_stat.items())[0][1]['mean'].keys())[0]
    means = [x['mean'][key] for x in bins_stat.values()]
    ci_lower = [x['conf_int'][key][0] for x in bins_stat.values()]
    ci_upper = [x['conf_int'][key][1] for x in bins_stat.values()]

    n_fig = len(data.columns)
    fig, axes = plt.subplots(n_fig, figsize=(10, 5*n_fig), squeeze=False)

    errors_lower = [m - l for m, l in zip(means, ci_lower)]
    errors_upper = [u - m for m, u in zip(means, ci_upper)]

    for i, col_name in enumerate(data.columns):
        ax = axes[i, 0] 
        ax.errorbar(
            x=bins, y=means,
            yerr=[errors_lower, errors_upper],
            fmt='o',
            capsize=5,
            color='black',
            elinewidth=2,
            markersize=8)
        ax.set_ylabel('Значение метрики на бине')
        ax.set_title(f'Статистика для {col_name} ({title})')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    return fig


def format_df(df_: pd.DataFrame,
              is_cluster: tp.Optional[bool] = False) -> pd.DataFrame:
    df = df_.copy()
    df['other_metric'] = df['other_metric'].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) else x
        ).fillna('NA').apply(lambda x: f'{x:.3f}' if type(x) != str else 'NA')
    df['p-value'] = df['p-value'].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) else x
        ).fillna('NA').apply(lambda x: f'{x:.3f}' if type(x) != str else 'NA')
    df['diff_ratio'] = df['diff_ratio'].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) else x
        ).fillna('NA').apply(lambda x: f'{x:.2f}' if type(x) != str else 'NA')
    if not is_cluster:  
        df['bin_metric'] = df['bin_metric'].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) else x
        ).fillna('NA').apply(lambda x: f'{x:.3f}' if type(x) != str else 'NA')
        df['business_thr_p_val'] = df['business_thr_p_val'].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) else x
        ).fillna('NA').apply(lambda x: f'{x:.3f}' if type(x) != str else 'NA')
        df['business_thr_ratio'] = df['business_thr_ratio'].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) else x
        ).fillna('NA').apply(lambda x: f'{x:.2f}' if type(x) != str else 'NA')
    else:
        df['cluster_metric'] = df['cluster_metric'].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) else x
        ).fillna('NA').apply(lambda x: f'{x:.3f}' if type(x) != str else 'NA')

    return df


def dialogue_to_text(dialogue: tp.Iterable[tp.Tuple]) -> str:
    '''
    Собирает реплики диалога в единый текст в формате чата.
    Parameters:
        dialogue: список кортежей вида (query_id, question, answer)
    Return:
        Строка вида 'USER: <question>\nASSISTANT:<answer>\nUSER: <question>\n...'
    '''
    parts = []
    for _, question, answer in dialogue:
        parts.append(f'USER: {question}\n')
        parts.append(f'ASSISTANT:{answer}\n')
    return ''.join(parts)


def dialogue_to_text_clean(dialogue: tp.Iterable[tp.Tuple]) -> str:
    '''
    Выборка реплик пользователя из диалога
    Parameters:
        dialogue: список кортежей вида (query_id, question, answer)
    Return:
        Строка вида '<question>\n<question>\n...'
    '''
    user_requests = ''
    for _, question, answer in dialogue:
        user_requests += question if question is not None else ''
        user_requests += '\n'
    return user_requests


def mean_dialogue_question_lengths(dialogues: tp.Iterable,
                                    token_counter_func: tp.Optional[tp.Callable]=None) -> tp.List[float]:
    '''
    Считает среднюю длину вопросов (<question>) внутри каждого диалога.
    Parameters:
        dialogues: итерабельность списков кортежей вида (query_id, question, answer)
        token_counter_func: функция подсчета длины, List[str] -> np.ndarray[int].
    Return:
        Список средних длин вопросов, по одному значению на диалог
    '''
    return [float(np.mean(token_counter_func([question for _, question, _ in dialogue])))
            for dialogue in dialogues]
