import re
import math
import numpy as np
import pandas as pd
from typing import List, Any, Dict, Tuple, Callable, Optional

from llm_val.utils.standardizer import standardize
from llm_val.utils.Embedder.Embedder import Embedder
from llm_val.utils.json_reader import read_json_config
from llm_val.utils.SimFunCollection import SimFunCollection as sfc
from llm_val.utils.TextPreprocessor import TextPreprocessor as tepr
from llm_val.utils.LongAnswerMetricDrawer import draw, draw_red_cross
from llm_val.sampler.BrainStormSampler import BrainStormSampler as bss


def report_brainstorm_quality(run_results: Tuple[List[List[Any]],
                                                 pd.DataFrame,
                                                 List[str]],
                              sort_table: bool,
                              get_top_k: bool,
                              top_k: int) -> Dict[str, Any]:
    """
    Создание отчета.
    Parameters:
        run_results: Обработанные результаты работы execute_tasks
        sort_table: Флаг, поднятие которого приведет к сортировке таблицы
                    и добавлению в нее столбца с индексом,
                    обозначающим номер запуска конфигурации
        get_top_k: Флаг, поднятие которого включит в итоговый отчет
                   только top_k лучших пайплайнов. Выполнится, только если
                   таблица была отсортирована.
        top_k: Количество пайплайнов, которые будут взяты среди лидеров
               из таблицы по всем запускам
    Return:
        Отчет в соответствии с концепцией llm_val
    """
    res_dict = dict()
    res_dict['semaphore'] = 'gray'
    res_dict['result_plots'] = run_results[0]
    if sort_table:
        df_res = run_results[1]
        df_res.columns = df_res.columns.map(' '.join).str.strip()
        df_res = df_res.sort_values(by=['abs_error_mean',
                                        'abs_error_median']).reset_index()
        res_dict['result_dataframes'] = df_res
    else:
        res_dict['result_dataframes'] = run_results[1]
    if get_top_k and sort_table:
        res_dict['result_dataframes'] = res_dict['result_dataframes'][:top_k]
        plot_idxs = res_dict['result_dataframes'].loc[:, 'index'].values
        res_dict['result_plots'] = [res_dict['result_plots'][i] for i in plot_idxs]
    return res_dict


def valtest_brainstorm_quality(sampler: bss,
                               global_config_path: str,
                               task_config_path: str,
                               model_storage: Dict[str,
                                                   Dict[str,
                                                        Any]],
                               sort_table: bool=True,
                               get_top_k: bool=True,
                               top_k: int=5,
                               **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Тест, выполняющий оценку качества генерации
    идей с помощью автоматической метрики
    Parameters:
        sampler: Объект класса BrainStormSampler c
                 переданными в него данными и конфигами
        global_config_path: Путь до глобального конфигурационного файла
        task_config_path: Путь до конфигурационного файла задач
        model_storage: ассоциативный массив ассоциативных массивов,
                       содержащий эмбеддер и соотв. ему токенайзер
        sort_table: Флаг, поднятие которого обеспечит
                    сортировку итоговой таблицы результатов
        get_top_k: Флаг, поднятие которого включит в итоговый отчет
                   только top_k лучших пайплайнов. Выполнится, только если
                   таблица была отсортирована.
        top_k: Количество пайплайнов, которые будут взяты среди лидеров
               из таблицы по всем запускам
    Return:
        Словарь в соответствии с концепцией llm_val
    """
    runs_results = execute_tasks(global_config_path,
                                 task_config_path,
                                 sampler,
                                 model_storage)
    parsed_results = handle_results(runs_results,
                                    global_config_path)
    report = report_brainstorm_quality(parsed_results,
                                       sort_table,
                                       get_top_k=get_top_k,
                                       top_k=top_k)
    precomputed = {'all_raw_data': runs_results,
                   'info_strings': parsed_results[-1],
                   'all_plots': parsed_results[0],
                   'full_table': parsed_results[1]}
    return {'report': report,
            'precomputed': precomputed}


def plot_configuration(plot_data: Optional[List[Dict[str, Any]]],
                       global_config_path: str,
                       x_label: str,
                       y_label: str,
                       contains_score: bool,
                       contains_theme: bool) -> List[Any]:
    """
    Построение графика распределения
    текстов с учетом компонент метрики
    и иных данных
    Parameters:
        plot_data: Данные для построения графиков
        global_config_path: Путь до глобального конфига
        x_label: Наименование меры близости между идеями и задачей
        y_label: Наименование меры близости между идеями
        contains_score: Флаг, показывающий наличие оценок асессоров в датасете
        contains_theme: Флаг, показывающий наличие тематик в датасете
    Return:
        Список полученных графиков в порядке
        запуска соотв. конвейера
    """
    plots = []
    silent = not (contains_score or contains_theme)
    theme_and_score = contains_score and contains_theme
    if plot_data is None:
        plots.append(draw_red_cross())
    else:
        if not theme_and_score:
            plots.append(draw(data=plot_data,
                              global_config_path=global_config_path,
                              draw_score=contains_score,
                              silent=silent,
                              x_ax_label=x_label,
                              y_ax_label=y_label))
        else:
            plots.append(draw(data=plot_data,
                              global_config_path=global_config_path,
                              silent=silent,
                              draw_score=True,
                              x_ax_label=x_label,
                              y_ax_label=y_label))
            plots.append(draw(data=plot_data,
                              global_config_path=global_config_path,
                              silent=silent,
                              draw_score=False,
                              x_ax_label=x_label,
                              y_ax_label=y_label))
    return plots


def handle_results(plot_data: List[List[List[Any]]],
                   global_config_path: str) -> Tuple[List[List[Any]],
                                                     pd.DataFrame,
                                                     List[str]]:
    """
    Функция для обработки результатов запуска
    Parameters:
        plot_data: Данные запусков по каждой конфигурации для обработки
        global_config_path: Путь до глобального конфига
    Return:
        Кортеж, состоящий из списка графиков,
        dataframe и списка информационных строк
    """
    plots = []
    pd_data = []
    info_strings = []
    for pipeline in plot_data:
        for configuration in pipeline:
            handled_data = handle_configuration(configuration)
            pd_data.append([*handled_data['pd_metric_row'],
                            *handled_data['info_lst'],
                            handled_data['status']])
            info_strings.append(' '.join(handled_data['info_lst']))
            plots.append(plot_configuration(handled_data['metric_data'],
                                            global_config_path,
                                            handled_data['x_label'],
                                            handled_data['y_label'],
                                            handled_data['contains_score'],
                                            handled_data['contains_theme']))
    return (plots,
            pd.DataFrame(pd_data,
                         columns=[handled_data['pd_headers']]),
            info_strings)


def handle_configuration(configuration: List[Any]) -> Dict[str, Any]:
    """
    Функция, подготавливающая данные итога
    работы конкретной конфигурации конвейера
    Parameters:
        configuration: Данные, полученные после запуска конфигурации конвейера
    Return:
        Обработанные данные итога работы конфигурации конвейера
    """
    pd_headers = ['abs_error_median', 'abs_error_mean',
                  'embedder_name', 'task_idea_sim',
                  'idea_idea_sim', 'is_text_cleaned',
                  'emb_norm_used', 'use_lemmatization',
                  'status']
    handled_data = {'pd_headers': pd_headers}
    handled_data['metric_data'] = configuration[0]
    handled_data['contains_score'] = configuration[1]
    handled_data['contains_theme'] = configuration[2]
    handled_data['pd_metric_row'] = configuration[3]
    handled_data['info_lst'] = configuration[4].split('|')
    handled_data['status'] = configuration[5]
    (handled_data['x_label'],
     handled_data['y_label']) = handled_data['info_lst'][1:3]
    return handled_data


def process_batch(embedder: Embedder,
                  task: str,
                  ideas: List[str],
                  sim_fun_1: Callable,
                  sim_fun_2: Callable) -> Tuple[float,
                                                Tuple[float,
                                                      float]]:
    """
    Метод для обработки брейншторма: производит
    подсчет эмбеддингов и затем вычисление
    метрики.
    Parameters:
        embedder: объект-эмбеддер, имеющий метод get_embedding
        task: Постановка задачи для мозгового штурма
        ideas: Список сгенерированных идей на поставленную задачу
        sim_fun_1: Мера близости для оценки соответствия
                   идей модели к поставленной задаче
        sim_fun_2: Мера близости для оценки разнообразия
                   предложенных моделью идей
    Return:
        Кортеж, содержащий значение метрики и
        кортеж со значениями компонент метрики
    """
    all_embeds = embedder.get_embedding(content=[task, *ideas])
    task_embed = all_embeds[0]
    ideas_embeds = all_embeds[1]
    return calc_metric(task_embed,
                       ideas_embeds,
                       sim_fun_1,
                       sim_fun_2)


def calc_metric(task_embed: List[float],
                ideas_embeds: List[List[float]],
                sim_fun_1: Callable,
                sim_fun_2: Callable) -> Tuple[float,
                                              Tuple[float, float]]:
    """
    Метод для подсчета метрики для брейншторма.
    Parameters:
        task_embed: Эмбеддинг постановки задачи мозгового штурма
        ideas_embeds: Список эмбеддингов, полученных из ответов
                      модели на поставленную задачу
        sim_fun_1: Мера близости для оценки соответствия
                   идей модели к поставленной задаче
        sim_fun_2: Мера близости для оценки схожести
                   предложенных моделью идей
    Return:
        Кортеж, содержащий значение метрики и
        кортеж со значениями компонент метрики
    """
    sim_sum_1_factor = 0
    sim_sum_2_factor = 0

    for i in range(len(ideas_embeds)):
        sim_sum_1_factor += sim_fun_1(task_embed, ideas_embeds[i])
        ideas_sim_sum = 0
        norm_coef = 1
        for j in range(i+1, len(ideas_embeds)):
            norm_coef = len(ideas_embeds) - 1 - i
            ideas_sim_sum += (1 - sim_fun_2(ideas_embeds[i],
                                            ideas_embeds[j]))
        sim_sum_2_factor += (ideas_sim_sum / norm_coef)
    sim_sum_1_factor /= len(ideas_embeds)
    sim_sum_2_factor /= (max(1, len(ideas_embeds) - 1))
    final_metric = sim_sum_1_factor * sim_sum_2_factor
    return final_metric, (sim_sum_1_factor, sim_sum_2_factor)


def measure_distance(global_config_data: Dict[str, Any],
                     data: List[Dict[str, Any]],
                     max_point: Tuple[float,
                                      float] = (1, 1)) -> List[Dict[str, Any]]:
    """
    Функция добавление для каждого брейншторма
    в ассоциативный массив расстояния
    от него до "идеального брейншторма"
    Paramters:
        global_config_data: ассоциативный массив
                            со всеми данными из глобального конфига
        data: набор из всех подсчитанных брейнштормов
        max_point: точка "идеального брейншторма";
                   (1, 1) по умолчанию
    Return:
        Набор данных, с добавленным полем расстояния
        до идеального брейншторма
    """
    point_cell_name = global_config_data['names'][0]['point_name']
    x_cord_name = global_config_data['names'][0]['task_idea_name']
    y_cord_name = global_config_data['names'][0]['idea_idea_name']
    distance_cell_name = global_config_data['names'][0]['dist_cell_name']
    for idx in range(len(data)):
        cur_point = data[idx][point_cell_name]
        cur_point = (cur_point[x_cord_name], cur_point[y_cord_name])
        data[idx][distance_cell_name] = math.hypot(max_point[0] - cur_point[0],
                                                   max_point[1] - cur_point[1])
    return data


def calc_score_diff(global_config_data: Dict[str, Any],
                    data: List[Dict[str, Any]]) -> Tuple[float,
                                                         float]:
    """
    Вычисление отклонений оценки инструмента
    от асессорской разметки
    Parameters:
        global_config_data: ассоциативный массив
                            со всеми данными из глобального конфига
        data: набор данных по всем подсчитанным брейнштормам
    Return:
        Кортеж из медианы и среднего арифметического
        абсолютного отклонения автооценок от оценок асессоров
    """
    dist_cell_name = global_config_data['names'][0]['dist_cell_name']
    score_name = global_config_data['names'][0]['score_name']
    data = measure_distance(global_config_data, data.copy())
    dist = list(map(lambda x: x[dist_cell_name], data))
    score = list(map(lambda x: x[score_name], data))
    norm_score = standardize(score)
    norm_dist = standardize(dist)
    diff_abs = []
    for idx in range(len(norm_score)):
        cur_diff = norm_score[idx] - (1 - norm_dist[idx])
        diff_abs.append(np.abs(cur_diff))
    return np.median(diff_abs), np.mean(diff_abs)


def split_ideas(global_names: Dict[str, Any],
                ideas: str) -> List[str]:
    """
    Функция разделения идей,
    предложенных на одну задачу
    Parameters:
        global_names: часть глобального конфига, описывающая
                      актуальные наименования полей и
                      правило разделения в датасете
        ideas: строка, состоящая из всех идей,
               относящихся к одной постановке задач
    Return:
        Список, состоящий из разделенных идей
    """
    changed_ideas = re.sub(global_names['re_splitter'],
                           '|<|',
                           ideas)
    return [x for x in changed_ideas.split('|<| ') if len(x) > 0]


def pipeline_runner_get_metric(embedder: Embedder,
                               data: pd.DataFrame,
                               global_names: Dict[str, str],
                               idx: int,
                               sim_fun_1: Callable,
                               sim_fun_2: Callable) -> Tuple[Dict[str, Any], Tuple[bool, bool]]:
    """
    Выполняет подсчет компонент метрики и необходимых
    данных для оценки конкретного брейншторма
    Parameters:
        embedder: эмбеддер, применяемый в данной конфигурации пайплайна
        data: датасет
        global_names: часть глобального конфига, описывающая
                      актуальные наименования полей и
                      правило разделения в датасете
        idx: Номер брейншторма, для которого необходимо подсчитать метрику
        sim_fun_1: Метод подсчета меры близости для постановки задачи
                   и идеями к ней. Если был получен None - используется
                   косинусная мера близости
        sim_fun_2: Метод подсчета меры разнообразия между предложенными
                   идеями. Если был получен None - используется
                   косинусная мера близости
    Return:
        Обработанные данные и флаги, показывающие наличие
        оценок асессора и тематики задачи
    """
    processed_data = []

    task = data.loc[idx, global_names['task_name']]
    ideas = data.loc[idx, global_names['ideas_name']]
    ideas = list(map(lambda x: x.strip(), split_ideas(global_names, ideas)))
    contains_score = contains_theme = False
    is_score_available = 'score_name' in global_names.keys()
    is_theme_available = 'theme_name' in global_names.keys()
    if is_score_available:
        score = data.loc[idx, global_names['score_name']]
        if score is not None:
            contains_score = True
        else:
            score = 'Нет данных'
    else:
        score = 'Нет данных'
    if is_theme_available:
        theme = data.loc[idx, global_names['theme_name']]
        if theme is not None:
            contains_theme = True
        else:
            theme = 'Нет данных'
    else:
        theme = 'Нет данных'
    metric, factors = process_batch(embedder,
                                    task,
                                    ideas,
                                    sim_fun_1=sim_fun_1,
                                    sim_fun_2=sim_fun_2)
    processed_data = {global_names['point_name']:
                      {global_names['task_idea_name']: factors[0],
                       global_names['idea_idea_name']: factors[1]},
                      global_names['score_name']: score,
                      global_names['theme_name']: theme}
    return processed_data, (contains_score, contains_theme)


def get_pipeline_results(global_config_data: Dict[str, Any],
                         embedder: Embedder,
                         data: pd.DataFrame,
                         sim_fun_1: Callable,
                         sim_fun_2: Callable) -> Tuple[List[Dict[str, Any]], bool, bool, Tuple[float, float]]:
    """
    Метод запускает конвейер с заданными параметрами
    Parameters:
        global_config_data: ассоциативный массив
                            со всеми данными из глобального конфига
        embedder: эмбеддер, применяемый в данной конфигурации пайплайна
        data: DataFrame с данными датасета
        sim_fun_1: Мера близости для оценки
                   соответствия идей поставленной
                   задаче
        sim_fun_2: Мера близости для оценки
                   разнообразия между предложенными
                   идеями
    Return:
        Возвращает кортеж, состоящий из полученных значений метрики,
        флага наличия оценок асессоров, флага наличия темы сессии,
        кортежа из медианы и среднего арифметического для
        абсолютных отклонений
    """
    global_names = dict()
    name_space = global_config_data['names'][0]
    nlp_space = global_config_data['nlp'][0]
    global_names['re_splitter'] = nlp_space['re_ideas_splitter']
    for cur_key in name_space.keys():
        global_names[cur_key] = name_space[cur_key]

    processed_data = []
    contains_score = contains_theme = False
    for idx in range(data.shape[0]):
        metric_components = pipeline_runner_get_metric(embedder,
                                                       data,
                                                       global_names,
                                                       idx,
                                                       sim_fun_1,
                                                       sim_fun_2)
        processed_data.append(metric_components[0])
        contains_score = contains_score or metric_components[1][0]
        contains_theme = contains_theme or metric_components[1][1]

    mean_errors = (None, None)
    if contains_score:
        mean_errors = calc_score_diff(global_config_data,
                                      processed_data)
    return (processed_data,
            contains_score,
            contains_theme,
            mean_errors)


def run_pipeline(global_config_data: Dict[str, Any],
                 data: pd.DataFrame,
                 embedder: Embedder,
                 functions_pairs: List[List[str]],
                 use_text_cleaner: bool,
                 normalize: bool,
                 use_lammatization: bool,
                 emb_name: str) -> List[Any]:
    """
    Метод, выполняющий запуск конвейера, полученного
    из конфигурационного файла задач
    Parameters:
        global_config_data: ассоциативный массив
                            со всеми данными из глобального конфига
        data: DataFrame с данными датасета
        embedder: эмбеддер, применяемый в данной конфигурации пайплайна
        function_pairs: Список из списков пар мер близостей,
                        применяемых соответственно для оценки
                        уместности предложенных идей к поставленной задаче и
                        оценки разнообразия среди предложенных идей
        use_text_cleaner: Флаг, поднятие которого обеспечит
                          предобработку текстов брейнштормов
        normalize: Флаг, поднятие которого обеспечит нормализацию эмбеддингов,
                   реализовано не для всех эмбеддеров
        use_lemmatization: Флаг, поднятие которого при очистке текста приведет
                           к лемматизации текста, а не стеммингу
        emb_name: Наименование эмбеддера
    Return:
        Результаты запуска конвейера со всеми заданными параметрами
    """
    sfc_ = sfc()
    collected_data = []
    for function_pair in functions_pairs:
        param_str = (f'{emb_name}|{function_pair[0]}'
                     f'|{function_pair[1]}|'
                     f'{use_text_cleaner}|'
                     f'{normalize}'
                     f'|{use_lammatization}')
        has_sim_fun_1 = hasattr(sfc_, function_pair[0])
        has_sim_fun_2 = hasattr(sfc_, function_pair[1])
        has_sim_fun_pair = has_sim_fun_1 and has_sim_fun_2
        if has_sim_fun_pair:
            sim_fun_1 = getattr(sfc_, function_pair[0])
            sim_fun_2 = getattr(sfc_, function_pair[1])
            results = get_pipeline_results(global_config_data,
                                           embedder,
                                           data,
                                           sim_fun_1=sim_fun_1,
                                           sim_fun_2=sim_fun_2)
            collected_data.append([*results, param_str, 'OK'])
        else:
            collected_data.append([None,
                                   False,
                                   False,
                                   (None, None),
                                   param_str,
                                   (f'Не вышло получить одну из следующих'
                                    f' мер близости:'
                                    f' {function_pair[0]},'
                                    f' {function_pair[1]}.'
                                    f' Убедитесь в правильности написания'
                                    f' названия мер и наличии их'
                                    f' реализации '
                                    f'в SimFunCollection.py. ')])
    return collected_data


def execute_tasks(global_config_path: str,
                  task_config_path: str,
                  sampler: bss,
                  model_storage: Dict[str,
                                      Dict[str,
                                           Any]]) -> List[List[Any]]:
    """
    Метод, выполняющий запуск конвейеров по указанным
    в конфигурационном файле и агрегирующий
    результаты их работы
    Parameters:
        global_config_path: путь до глобального конфигурационного файла
        task_config_path: путь до конфигурационного файла задач
        sampler: Объект класса BrainStormSampler c
                 переданными в него данными и конфигами
        model_storage: ассоциативный массив ассоциативных массивов,
                       содержащий эмбеддер и соотв. ему токенайзер
    Return:
        Список результатов запусков конвейеров во всех конфигурациях
    """
    global_config_data = read_json_config(global_config_path)
    task_config_data = read_json_config(task_config_path)
    data = sampler.test['X']
    runs = task_config_data['runs']
    runs_data = []
    for run in runs:
        run_keys = list(run.keys())
        emb_name = run['embedders_name']
        function_pairs = run['function_pairs']
        if 'use_text_cleaner' in run_keys:
            is_true = run['use_text_cleaner'] == 'True'
            use_text_cleaner = (False, True)[is_true]
        else:
            use_text_cleaner = False
        if 'use_lemmatization' in run_keys:
            is_true = run['use_lemmatization'] == 'True'
            use_lemmatization = (False, True)[is_true]
        else:
            use_lemmatization = False
        if 'normalize' in run_keys:
            is_true = run['normalize'] == 'True'
            normalize = (False, True)[is_true]
        else:
            normalize = False
        model = model_storage[emb_name]['model']
        tokenizer = model_storage[emb_name]['tokenizer']
        model = tune_model(model,
                           tokenizer,
                           global_config_data,
                           use_text_cleaner,
                           use_lemmatization,
                           normalize)
        pipeline_data = run_pipeline(global_config_data,
                                     data,
                                     model,
                                     function_pairs,
                                     use_text_cleaner,
                                     normalize,
                                     use_lemmatization,
                                     emb_name)
        runs_data.append(pipeline_data)
    return runs_data


def tune_model(model: Embedder,
               tokenizer: Any,
               global_config_data: Dict[str, Any],
               use_text_cleaner: bool,
               use_lemmatization: bool,
               normalize: bool) -> Embedder:
    """
    Функция, подготавливающая модель к работе
    Parameters:
        model: эмбеддер, имеющий метод get_embedding
        tokenizer: токенайзер, применяемый для используемого эмбеддера
        global_config_data: ассоциативный массив
                            со всеми данными из глобального конфига
        use_text_cleaner: Флаг, поднятие которого обеспечит
                          предобработку текста брейнштормов
        use_lemmatization: Флаг, поднятие которого при очистке текста приведет
                           к лемматизации текста, а не стеммингу
        normalize: Флаг, поднятие которого обеспечит нормализацию эмбеддингов
                   Реализовано не для всех эмбеддеров
    Return:
        Модель, получившая все необходимые настройки для работы
    """
    model.tokenizer = tokenizer
    model.use_text_cleaner = use_text_cleaner
    if use_text_cleaner:
        stopwords = global_config_data['nlp'][0]['stopwords_path']
        model.text_preprocessor = tepr(stopwords,
                                       use_lemmatization=use_lemmatization)
    model.normalize = normalize
    return model
