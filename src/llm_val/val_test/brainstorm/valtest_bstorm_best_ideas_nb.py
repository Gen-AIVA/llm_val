import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm.auto import tqdm
from types import MappingProxyType
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from typing import Any, Optional, Union, List, Dict, Tuple

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.Embedder.Embedder import Embedder
from llm_val.utils.json_reader import read_json_config
from llm_val.utils.TextPreprocessor import TextPreprocessor
from llm_val.utils.brainstorm_metrics import modified_cosine_similarity, modified_euclidean_similarity


DISTANCE_REGISTRY = MappingProxyType({'cosine': modified_cosine_similarity, 'euclidean': modified_euclidean_similarity})


def report_bstorm_best_ideas_nb(metrics_dfs_per_run: List[pd.DataFrame], 
                                plots_per_run: List[Figure]) -> Dict[str, Any]:
    '''
    Функция для составления отчета по результатам работы теста.
    Parameters:
        metrics_dfs_per_run: Список датафреймов с метриками для каждого запуска.
        plots_per_run: Список отрисованных графиков для каждого соответствующего датафрейма из metrics_dfs_per_run
    Return:
        Отчет в соответствии с концепцией llm_val
    '''
    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_plots'] = plots_per_run
    res_dict['result_dataframes'] = metrics_dfs_per_run
    return res_dict


def valtest_bstorm_best_ideas_nb(sampler: Sampler,
                                 task_config_path: str,
                                 model_storage: Dict[str, Dict[str, Any]],
                                 cached_embeds: Optional[Dict[Tuple[str, str, str, str],
                                                              Dict[str, List[Union[List[float],
                                                                                   List[List[float]]]]]]]=None,
                                 dist_binning_thresholds: Optional[Dict[Tuple[str, str, str, str],
                                                                        Dict[str, List[float]]]]=None,
                                 max_ideas: int=6,
                                 nb_of_scores: int=3,
                                 n_bootstrap: int=1000,
                                 seed: int=42,
                                 **kwargs) -> Dict[str, Dict[str, Any]]:
    '''
    Функция для проведения информативного теста зависимости стабильности качества генерации идей от их числа.
    Parameters:
        sampler: Инициализированный объект sampler'a
        task_config_path: Путь к конфигурационному файлу, содержащему параметры запуска эмбеддеров, предобработки
            и мер близости
        model_storage: Словарь, содержащий в себе инициализированные объекты эмбеддеров
            и токенайзеров
        cached_embeds: Словарь с закешированными эмбеддингами для каждого уникального эмбеддера с параметрами
            предобработки. В качестве ключей выступают кортежи значений параметра запуска из конфига (имя эмбеддера,
            флаг очистки текста, флаг нормализации, флаг лемматизации).
        dist_binning_thresholds: Словарь, содержащий значения мер близости для биннинга в оценки для каждого
            уникального эмбеддера с параметрами предобработки. В качестве ключей выступают кортежи значений параметра
            запуска из конфига (имя эмбеддера, флаг очистки текста, флаг нормализации, флаг лемматизации).
        max_ideas: Максимальное число идей в датасете
        nb_of_scores: Число возможных уникальных значений дискретных значений оценок асессоров. Оценки - целые числа в
            отрезке [0, nb_of_scores-1]
        n_bootstrap: Кол-во итераций для бутстрапирования при определении порогов для биннинга
            значений мер близости в оценки
        seed: Сид для генератора случайных чисел. При несоответствии сида тому, что был задан в конфиге, закэшированные
            эмбеддинги и пороговые расстояния для биннинга не будут использованы и будут пересчитаны заново.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    task_config_data = read_json_config(task_config_path)

    task_runs = task_config_data['runs']
    gt_score_df = construct_gt_df(sampler, 
                                  max_ideas)

    if seed == task_config_data['seed']:
        cached_embeds = deepcopy(cached_embeds) if cached_embeds else {}
        dist_binning_thresholds = deepcopy(dist_binning_thresholds) if dist_binning_thresholds else {}
    else:
        cached_embeds = {}
        dist_binning_thresholds = {}

    precomputed = {'cached_embeds': {}, 'dist_binning_thresholds': {}}

    report_metrics_dfs = []
    report_plots = []
    for task_run in task_runs:
        if cached_embeds.get(tuple(task_run['embedder'].items())):
            task_run['cached_embeds'] = cached_embeds.get(tuple(task_run['embedder'].items()))
        if dist_binning_thresholds.get(tuple(task_run['embedder'].items())):
            task_run['dist_binning_thresholds'] = dist_binning_thresholds.get(tuple(task_run['embedder'].items()))
        run_results = process_single_run(model_storage,
                                         sampler,
                                         task_run,
                                         task_config_data['stopwords_path'],
                                         nb_of_scores=nb_of_scores,
                                         n_bootstrap=n_bootstrap,
                                         max_ideas=max_ideas,
                                         seed=seed)

        metrics_df = construct_metrics_df_for_single_run(task_run['embedder'],
                                                         run_results[0],
                                                         run_results[1],
                                                         max_ideas)
        metrics_df = pd.concat([metrics_df, gt_score_df], verify_integrity=True)
        report_metrics_dfs.append(metrics_df)
        report_plots.append(plot_metrics_df(metrics_df))

        cached_embeds[tuple(task_run['embedder'].items())] = run_results[2]
        dist_binning_thresholds[tuple(task_run['embedder'].items())] = run_results[3]

    report = report_bstorm_best_ideas_nb(report_metrics_dfs, 
                                         report_plots)

    precomputed['cached_embeds'] = cached_embeds
    precomputed['dist_binning_thresholds'] = dist_binning_thresholds
    precomputed['seed'] = seed

    return {'report': report, 
            'precomputed': precomputed}


def construct_gt_df(sampler: Sampler, 
                    max_ideas: int) -> pd.DataFrame:
    '''
    Создает датафрейм в общем стиле с дискретными оценками асессоров.
    Parameters:
        sampler: Инициализированный объект sampler'a
        max_ideas: Максимальное число идей в датасете
    Return:
        pd.DataFrame: Датафрейм с разметкой
    '''
    mean_gt_score_per_idea = calculate_gt_assessor_score(sampler.test['X'].to_dict(orient='list'), max_ideas)
    mean_gt_score_per_idea['run_setup'] = ['weighted manual mean gt']
    mean_gt_score_per_idea['distance'] = ['-']
    mean_gt_score_per_idea['metric'] = ['discrete score']
    return pd.DataFrame.from_dict(mean_gt_score_per_idea).set_index(['run_setup', 'distance', 'metric'])


def construct_report_index_col_name(emb_params: Dict[str, str]) -> str:
    '''
    Вспомогательная функция для построение информативной строки с параметрами запуска эмбеддера.
    Parameters:
        emb_params: Словарь с параметрами эмбеддера и предобработки
    Return:
        Строка с информативным отображением параметров запуска
    '''
    name = emb_params['name']
    clean = 'cleaned' if emb_params['use_text_cleaner'] == 'True' else ''
    normalize = 'normalized' if emb_params['normalize'] == 'True' else ''
    lemmatize = 'lemmatized' if emb_params['use_lemmatization'] == 'True' else ''
    stripped_entities = ' '.join([name, clean, normalize, lemmatize]).split()
    return ' '.join(stripped_entities)


def construct_metrics_df_for_single_run(emb_params: Dict[str, str],
                                        bs_metrics_per_idea_by_dist: Dict[str, Dict[str, float]],
                                        scores_per_idea_by_dist: Dict[str, Dict[str, float]],
                                        max_ideas: int) -> pd.DataFrame:
    '''
    Функция, оформляющая результаты с метриками и оценками единичного запуска в датафрейм для отчета.
    Parameters:
        emb_params: Словарь с параметрами эмбеддера и предобработки
        bs_metrics_per_idea_by_dist: Агрегированные (средним) метрики качества генерации идей для каждого поднабора идей
        scores_per_idea_by_dist: Агрегированные (средним) синтетические оценки качества генерации идей для каждого
            поднабора идей
        max_ideas: Максимальное число идей в датасете
    Return:
        Датафрейм с метриками и оценками, проиндексированные через параметры запуска, меру близости и тип метрики
            качества (дискретная оценка или значение метрики, основанной на выбранной мере близости)
    '''
    metrics_df_dict = defaultdict(list)
    for distance in bs_metrics_per_idea_by_dist:
        metrics_df_dict['run_setup'].extend([construct_report_index_col_name(emb_params)] * 2)
        metrics_df_dict['metric'].extend(['distance-based', 'discrete score'])
        metrics_df_dict['distance'].extend([distance] * 2)

        bs_metrics_per_idea = bs_metrics_per_idea_by_dist[distance]
        scores_per_idea = scores_per_idea_by_dist[distance]
        for i in range(1, max_ideas + 1):
            metrics_df_dict[f'{i} idea(s)'].extend([bs_metrics_per_idea[i], scores_per_idea[i]])

    metrics_df = pd.DataFrame.from_dict(metrics_df_dict)
    return metrics_df.set_index(['run_setup', 'distance', 'metric'])


def calculate_gt_assessor_score(test_data: Dict[str, List[Any]], 
                                max_ideas: int) -> Dict[str, float]:
    '''
    Функция вычисления агрегированной оценки ручной оценки асессоров по тестовому датасету.
    Parameters:
        test_data: Тестовый датасет, полученный из sampler'a
        max_ideas: Максимальное число идей в датасете
    Return:
        Словарь с агрегированной (средним) оценкой асессоров
    '''
    weighted_scores_per_n_ideas = defaultdict(list)
    for sample_scores in test_data['per_idea_score']:
        for i in range(1, max_ideas + 1):
            weighted_scores_per_n_ideas[f'{i} idea(s)'].append(
                (np.array(sample_scores[:i]) > 0).mean() * sum(sample_scores[:i]))
    return {n_ideas: np.mean(weighted_gt_scores) for n_ideas, weighted_gt_scores in weighted_scores_per_n_ideas.items()}


def plot_metrics_df(metrics_df: pd.DataFrame) -> List[Figure]:
    '''
    Функция, отрисовывающая графики для отчета.
    На выходе список из 2 графиков - один показывает зависимость величины асессорского скора от кол-ва идей
    для каждой меры близости и ручной оценкой. Другой - зависимость невзвешенного кумулятивного значения метрики,
    основанной на выбранной мере близости, от кол-ва идей для каждой из мер близости в конфиге
    Parameters:
        metrics_df: Датафрейм с оценками и метриками на основе расстояний для одного параметризированного запуска
    Return:
        Список с отрисованными графиками
    '''
    metrics_df = deepcopy(metrics_df).reset_index()
    dist_df = metrics_df[metrics_df['metric'] == 'distance-based'].drop(columns=['metric'])
    dist_df = dist_df.set_index(['run_setup', 'distance'])

    score_df = metrics_df[metrics_df['metric'] != 'distance-based'].drop(columns=['metric'])
    score_df = score_df.set_index(['run_setup', 'distance'])

    return [plot_single_df(score_df,
                           title='Synthetic and GT assessor scores by the number of generated ideas',
                           ylabel='Mean weighted assessor score',
                           xlabel='The number of ideas'),
            plot_single_df(dist_df,
                           title='Distance-based BS quality by the number of generated ideas',
                           ylabel='Mean BS distance-based quality score',
                           xlabel='The number of ideas')]


def plot_single_df(df: pd.DataFrame,
                   title: str,
                   xlabel: str,
                   ylabel: str,
                   figsize: Tuple[int, int]=(8, 5),
                   legend: bool=True,
                   grid: bool=True) -> Figure:
    '''
    Строит график на базе единичного датафрейма с метриками.
    Parameters:
        df: Датафрейм с оценками или метриками на основе расстояний для одного параметризированного запуска
        title: Заголовок для отрисованного графика
        xlabel: Подпись к оси Х
        ylabel: Подпись к осу Y
        figsize: Размер отрисованного графика
        legend: Флаг для отображения подписей к каждому графику
        grid: Флаг для отображение сетки на графике
    Return:
        Отрисованный график
    '''
    plt.figure()
    for (run_params, distance), row in df.iterrows():
        ax = row.plot(title=title,
                      figsize=figsize,
                      legend=legend,
                      grid=grid,
                      label=f'{run_params} / {distance}',
                      xlabel=xlabel,
                      ylabel=ylabel)
    fig = ax.get_figure()
    plt.close(fig)
    return fig


def process_single_run(model_storage: Dict[str, Dict[str, Any]],
                       sampler: Sampler,
                       run_params: Dict[str, Any],
                       stopwords_path: str,
                       nb_of_scores: int,
                       max_ideas: int,
                       n_bootstrap: int,
                       seed: int) -> Tuple[Dict[str, Dict[str, float]],
                                           Dict[str, Dict[str, float]],
                                           Dict[Tuple[str, str, str, str],
                                                Dict[str, List[Union[List[float], List[List[float]]]]]],
                                           Dict[Tuple[str, str, str, str], Dict[str, List[float]]]]:
    '''
    Функция, содержащая в себе верхнеуровневую имплементацию подсчета метрик/оценок для одного запуска.
    Parameters:
        model_storage: Словарь, содержащий в себе инициализированные объекты эмбеддеров и токенайзеров
        sampler: Инициализированный объект sampler'a
        run_params: Словарь с параметрами запуска (параметры эмбеддера + меры близости)
        stopwords_path: Путь к стоп-словам при преобработке текста
        nb_of_scores: Число возможных уникальных значений дискретных значений оценок асессоров. Оценки - целые числа
            в отрезке [0, nb_of_scores-1]
        max_ideas: Максимальное число идей в датасете
        n_bootstrap: Кол-во итераций для бутстрапирования при определении порогов для биннинга значений расстояний
            в оценки
        seed: Сид для генератора случайных чисел.
    Return:
        Кортеж из словарей с агрегированными (средним) метриками и оценками для каждого поднабора идей,
        а также закэшированные эмбеддинги и пороги биннинга в оценки для каждой из мер близости
    '''
    train_data = sampler.train['X'].to_dict(orient='list')
    test_data = sampler.test['X'].to_dict(orient='list')

    model = get_initialized_model(model_storage,
                                  run_params['embedder'],
                                  stopwords_path)

    thresholds_by_distance = run_params.get('dist_binning_thresholds')
    if not thresholds_by_distance:
        thresholds_by_distance = calibrate_binning_metric_thresholds(model,
                                                                     train_data,
                                                                     run_params['distances'],
                                                                     nb_of_scores=nb_of_scores,
                                                                     n_bootstrap=n_bootstrap,
                                                                     seed=seed)

    cached_embeds = run_params.get('cached_embeds', {'task': [], 'ideas': []})
    if len(cached_embeds['task']) != len(cached_embeds['ideas']) or \
       len(cached_embeds['task']) != len(test_data['question']):
        cached_embeds = {'task': [], 'ideas': []}

    bs_metrics_per_n_ideas_by_dist = defaultdict(list)
    synth_assessor_scores_per_n_ideas_by_dist = defaultdict(list)
    enumerate_data_generator = enumerate(zip(test_data['question'], test_data['splitted_ideas']))
    for idx, (task, ideas) in tqdm(enumerate_data_generator, total=len(test_data['question'])):
        if len(cached_embeds['task']) < len(test_data['question']):
            task_embed, ideas_embeds = model.get_embedding(content=[task, *ideas])
            cached_embeds['task'].append(task_embed)
            cached_embeds['ideas'].append(ideas_embeds)

        for distance in run_params['distances']:
            task_ideas_similarities, ideas_ideas_similarities = calc_bs_quality_metrics(cached_embeds['task'][idx],
                                                                                        cached_embeds['ideas'][idx],
                                                                                        DISTANCE_REGISTRY[distance])
            bs_metrics_per_n_ideas, synth_assessor_scores_per_n_ideas = calc_bs_metrics_and_scores_for_subgroups(
                task_ideas_similarities,
                ideas_ideas_similarities,
                thresholds_by_distance[distance],
                max_ideas)
            bs_metrics_per_n_ideas_by_dist[distance].append(bs_metrics_per_n_ideas)
            synth_assessor_scores_per_n_ideas_by_dist[distance].append(synth_assessor_scores_per_n_ideas)

    mean_bs_metrics_per_n_ideas_by_dist = reduce_metrics_by_distance(bs_metrics_per_n_ideas_by_dist, max_ideas)
    mean_synth_assessor_scores_per_n_ideas_by_dist = reduce_metrics_by_distance(
        synth_assessor_scores_per_n_ideas_by_dist,
        max_ideas)
    return (mean_bs_metrics_per_n_ideas_by_dist,
            mean_synth_assessor_scores_per_n_ideas_by_dist,
            cached_embeds,
            thresholds_by_distance)


def reduce_metrics_by_distance(metrics_per_dist_dict: Dict[str, List[Dict[int, float]]],
                               max_ideas: int) -> Dict[str, Dict[str, float]]:
    '''
    Функция для агрегации метрик или оценок для каждой заявленной меры близости.
    Подразумевается следующий формат данных:
    {мера близости: [{1 идея: оценка, 2 идея: оценка, ...}, ...], ...}
    Parameters:
        metrics_per_dist_dict: Словарь с метриками для каждого
        сэмпла и каждой мерой близости.
        max_ideas: Ожидаемое максимальное кол-во идей в датасете.
    Return:
        Словарь с агрегированными (средним) метриками/оценками для каждого поднабора идей.
    '''
    reduced_metrics_by_dist = defaultdict(dict)
    for distance, metrics_per_idea_list in metrics_per_dist_dict.items():
        for n_ideas in range(1, max_ideas + 1):
            reduced_metrics_by_dist[distance][n_ideas] = np.mean(
                [sample_metrics_dict[n_ideas] for sample_metrics_dict in metrics_per_idea_list])
    return reduced_metrics_by_dist


def calc_bs_metrics_and_scores_for_subgroups(sample_task_ideas_sims: np.ndarray,
                                             sample_ideas_ideas_sims: np.ndarray,
                                             dist_bin_thrs: np.ndarray,
                                             max_ideas: int) -> Tuple[Dict[int, float], Dict[int, float]]:
    '''
    Вычисляет метрики/оценки для подмножеств идей при заданной мере близости.
    Parameters:
        sample_task_ideas_sims: 1-D массив, содержащий расстояния между каждой идеей и задачей
        sample_ideas_ideas_sims: 2-D массив, содержащий значения мер близости идей друг между другом
        dist_bin_thrs: 1-D массив с порогами для биннинга для фиксированной меры близости
        max_ideas: Максимальное число идей в датасете
    Return:
        Кортеж из 2 словарей, где первый содержит метрики на базе мер близости для поднаборов идей (от 1 до max_ideas)
        в рамках одного сэмпла, а другой - синтетическую асессорскую оценку при тех же условиях
    '''
    bs_metrics_per_n_ideas = defaultdict(float)
    synth_assessor_scores_per_n_ideas = defaultdict(float)
    for n_ideas in range(1, max_ideas + 1):
        bs_metrics_per_idea = sample_task_ideas_sims[:n_ideas] * reduce_ideas_ideas_similarities(
            sample_ideas_ideas_sims,
        )[:n_ideas]
        bs_metrics_per_n_ideas[n_ideas] = bs_metrics_per_idea.sum()
        synth_assessor_score_per_idea = np.digitize(bs_metrics_per_idea, dist_bin_thrs)
        relevant_ideas_proportion = (synth_assessor_score_per_idea > 0).mean()
        synth_assessor_scores_per_n_ideas[n_ideas] = relevant_ideas_proportion * synth_assessor_score_per_idea.sum()
    return bs_metrics_per_n_ideas, synth_assessor_scores_per_n_ideas


def get_initialized_model(model_storage: Dict[str, Dict[str, Any]],
                          task_emb_params: Dict[str, Any],
                          stopwords_path: str) -> Embedder:
    '''
    Извлекает токенайзер и модель из storage'a и инициализирует параметрами последнюю.
    Parameters:
        model_storage: Словарь с эмбеддерами и соответствующими им токенайзерами
        task_emb_params: Параметры запуска для эмбеддера из конфига
        stopwords_path: Путь к стоп-словам для предобработки
    Return:
        Инициализированная модель-эмбеддер
    '''
    model = model_storage[task_emb_params['name']]['model']
    tokenizer = model_storage[task_emb_params['name']]['tokenizer']

    model.tokenizer = tokenizer
    model.use_text_cleaner = task_emb_params.get('use_text_cleaner') == 'True'
    model.normalize = task_emb_params.get('normalize') == 'True'
    if task_emb_params.get('use_text_cleaner') == 'True':
        model.text_preprocessor = TextPreprocessor(
            stopwords_path,
            use_lemmatization=task_emb_params.get('use_lemmatization') == 'True')
    return model


def reduce_ideas_ideas_similarities(ideas_ideas_similarities: np.ndarray) -> np.ndarray:
    '''
    Функция для "схлопывания" значений мер близости идей друг другу.
    Изначально значения мер близости идей между собой представляют собой двумерный массив, где каждая идея оценивается
    с остальными. Нам необходимо привести эти данные к одномерному виду, равному по длине кол-ву идей, с репрезентацией,
    похожей на оценку асессоров.
    Алгоритм следующий: Значение меры близости для первой идеи оценивается как 1 (т.е максимально уникальная),
    поскольку мы ее сравниваем не с самой собой, а с отсутствием идеи. Затем оценка каждой следующей идеи идет
    с предыдущими. Полезный инкремент получаемой информации для очередной идеи - это минимум близости между текущей
    идеей и всеми предыдущими.
    Parameters:
        ideas_ideas_similarities: 2-D массив, содержащий близости идей друг между другом
    Return:
        1-D агрегированные близости, согласно описанному алгоритму
    '''
    reduced_ideas_ideas_sims = [1.]
    tril_idxs_rows, tril_idxs_cols = np.tril_indices_from(ideas_ideas_similarities, k=-1)
    cur_row_idx = tril_idxs_rows[0]
    min_distance = 1e6
    for row_idx, col_idx in zip(tril_idxs_rows, tril_idxs_cols):
        if row_idx != cur_row_idx:
            reduced_ideas_ideas_sims.append(min_distance)
            cur_row_idx = row_idx
            min_distance = 1e6
        min_distance = min(min_distance, ideas_ideas_similarities[row_idx][col_idx])
    reduced_ideas_ideas_sims.append(min_distance)
    return np.array(reduced_ideas_ideas_sims)


def calc_bs_quality_metrics(task_embed: List[float],
                            ideas_embeds: List[List[float]],
                            sim_func: callable) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Метод для подсчета метрик оценки качества конкретного брейншторма.
    Предполагаем заранее, что измеряемая мера близости находится в диапазоне от [0, 1],
    где 0 подразумевает нескоррелированные входные данные, а 1 - полностью идентичные.
    При этом, поскольку для нас в рамках измерения меры близости между идеями важно, чтобы
    они были как можно дальше друг от друга, для выравнивания показателей качества мы
    для второй метрики возьмем разность между максимальным значением (1.) и значением меры близости.
    Parameters:
        task_embed: Эмбеддинг постановки задачи мозгового штурма
        ideas_embeds: Список эмбеддингов, полученных из ответов модели на поставленную задачу
        sim_func: Функция вычисления меры близости
    Return:
        Кортеж из двух 1-D массивов, содержащих значения меры близости, отражающие качество генерации для задачи
        брейншторма. Первый массив - это значения мер близости между каждой идеей и задачей, когда вторая - это
        агрегированная алгоритмически разница между единицей и значением меры близости между идеями для каждой идеи.
    '''
    task_ideas_similarities = np.zeros(len(ideas_embeds))
    ideas_ideas_similarities = np.zeros((len(ideas_embeds), len(ideas_embeds)))

    for i in range(len(ideas_embeds)):
        task_ideas_similarities[i] = sim_func(task_embed, ideas_embeds[i])
        for j in range(i + 1, len(ideas_embeds)):
            ideas_ideas_similarities[i][j] = 1. - sim_func(ideas_embeds[i], ideas_embeds[j])
            ideas_ideas_similarities[j][i] = ideas_ideas_similarities[i][j]
    return task_ideas_similarities, ideas_ideas_similarities


def calibrate_binning_metric_thresholds(model: Embedder,
                                        train_data: Dict[str, List[Any]],
                                        task_distances: List[str],
                                        nb_of_scores: int,
                                        n_bootstrap: int=1000,
                                        seed: int=42) -> Dict[str, List[float]]:
    '''
    Функция получения порогов для построения последующего отображения непрерывных расстояний в дискретные оценки.
    Parameters:
        model: Инициализированная модель-эмбеддер для инференса
        train_data: Данные из sampler'a для калибровки порогов
        task_distances: Список из мер близости, которые будут использоваться в тесте
        nb_of_scores: Число возможных уникальных значений дискретных значений оценок асессоров. Оценки - целые числа
            в отрезке [0, nb_of_scores-1]
        n_bootstrap: Кол-во итераций для бутстрапирования при определении порогов для биннинга расстояний в оценки
        seed: Сид для генератора случайных чисел
    Return:
        Словарь, содержащий список порогов для каждой из предоставленных мер близости
    '''
    metrics_by_dist = defaultdict(list)

    data_zip_generator = zip(train_data['question'], train_data['splitted_ideas'])
    for task, ideas in tqdm(data_zip_generator, total=len(train_data['question'])):
        task_embed, ideas_embeds = model.get_embedding(content=[task, *ideas])
        for distance in task_distances:
            sample_task_ideas_sims, sample_ideas_ideas_sims = calc_bs_quality_metrics(task_embed,
                                                                                      ideas_embeds,
                                                                                      DISTANCE_REGISTRY[distance])
            bs_metrics_per_idea = sample_task_ideas_sims * reduce_ideas_ideas_similarities(sample_ideas_ideas_sims)
            metrics_by_dist[distance].extend(bs_metrics_per_idea)
    thresholds_by_dist = {}
    for distance in task_distances:
        thresholds_by_dist[distance] = calc_bootstrap_thresholds(metrics_by_dist[distance],
                                                                 nb_of_scores,
                                                                 n_bootstrap,
                                                                 seed)
    return thresholds_by_dist


def calc_bootstrap_thresholds(metric_values: List[float],
                              nb_of_scores: int,
                              n_bootstrap: int,
                              seed: int) -> np.ndarray:
    '''
    Функция получения порогов путем бутстрапирования мер близости для последующего биннинга.
    На каждом шаге бутстрэпа мы сэмплим с возвращением из всего пула расстояний, считаем перцентили
    (их число равномерно распределено за счет параметра числа оценок), сохраняем. В конце усредняем
    полученные пороги по перцентилям и возвращаем для биннинга.
    Parameters:
        metric_values: Список со значениями конкретной меры близости
        nb_of_scores:  Число возможных уникальных значений дискретных значений оценок асессоров. Оценки - целые числа
            в отрезке [0, nb_of_scores-1]
        n_bootstrap: Кол-во итераций для бутстрапирования при определении порогов для биннинга
            расстояний в оценки
        seed: Сид для генератора случайных чисел
    Return:
        Пороги для биннинга
    '''
    percentiles = [int(round(score / nb_of_scores, 2) * 100) for score in range(1, nb_of_scores)]
    thresholds = defaultdict(list)
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        sample = rng.choice(metric_values, size=len(metric_values), replace=True)
        for percentile in percentiles:
            thresholds[percentile].append(np.percentile(sample, percentile))
    return np.array([np.mean(thresholds[percentile]) for percentile in percentiles])
