import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from llm_val.utils.json_reader import read_json_config


def get_global_config_names(config_path):
    """
    Функция для получения актуальных имен и палитры из глобального словаря
    """
    global_names = dict()
    config_data = read_json_config(config_path)
    for name_key in config_data['names'][0].keys():
        global_names[name_key] = config_data['names'][0][name_key]
    palette = config_data['palettes'][0]['drawer_palette']
    return global_names, palette


def update_points(archive: Dict[Any, List[Tuple[float, float]]],
                  key: Any,
                  point: Tuple[float, float]):
    """
    Функция реализует разделение точек, содержащих в качестве
    координат компоненты метрики, по выбранному признаку

    Parameters:
        archive: Хранилище точек
        key: Ключ, по которому происходит разделение точек.
             Тип зависит от устройства датасета
        point: Точка, получаемая в процессе подсчета метрик
             для одной сессии мозгового штурма
    """
    if key in archive.keys():
        archive[key].append(point)
    else:
        archive[key] = [point]


def process_data(data,
                 global_names,
                 class_points,
                 score_points):
    """
    Функция производит разделение точек-сессий
    по тематике и асессорской оценке
    """
    for i in range(len(data)):
        update_points(class_points,
                      data[i][global_names['theme_name']],
                      data[i][global_names['point_name']])
        update_points(score_points,
                      data[i][global_names['score_name']],
                      data[i][global_names['point_name']])
    return class_points, score_points


def draw(data,
         global_config_path: str,
         fig_width: int=6,
         fig_height: int=4,
         silent: bool=False,
         draw_score: bool=False,
         y_ax_label: str='Unknown',
         x_ax_label: str='Unknown',
         allow_plot: bool=False) -> np.ndarray:
    """
    Функция, отвечающая за отрисовку распределения сессий
    мозгового штурма на плоскости
    Чем ближе точка к (1, 1), тем лучше ее качество
    По оси Х описано соответствие ответов модели поставленной задаче
    По оси Y описано разнообразие предложенных моделью идей

    Parameters:
        draw_score: Флаг, поднятие которого выделит цветом точки,
                    в зависимости от асессорской оценки
                    иначе цветовое разделение будет основано на
                    тематике сессии мозгового штурма
                    Флаг применяется, если для объекта не был
                    поднят флаг silent
        y_ax_label: Наименование меры близости, применяемой для
                    оценки разнообразия предложенных идей
        x_ax_label: Наименование меры близости, применяемой для
                    оценки соответствия предложенных идей тематике сессии
        allow_plot: Флаг, поднятие которого разрешит немедленную
                    отрисовку изображение по окончанию его построения
    Returns:
        Построенный график
    """
    score_points = dict()
    class_points = dict()

    max_x = 1
    max_y = 1
    min_x = 0
    min_y = 0

    global_names, palette = get_global_config_names(global_config_path)
    class_points, score_points = process_data(data,
                                              global_names,
                                              class_points,
                                              score_points)

    fig, ax = plt.subplots(figsize=(fig_width,
                                    fig_height))
    if draw_score:
        data_source = score_points
    else:
        data_source = class_points
    for idx, group in enumerate(data_source.keys()):
        x = []
        y = []
        for sample in data_source[group]:
            x.append(sample[global_names['task_idea_name']])
            y.append(sample[global_names['idea_idea_name']])

        if max_x > max(x):
            max_x = max(x)
        if max_y > max(y):
            max_y = max(y)
        if min_x < min(x):
            min_x = min(x)
        if min_y < min(y):
            min_y = min(y)

        if silent:
            cur_color = 'tab:blue'
        else:
            cur_color = palette[idx]

        ax.scatter(x, y, c=cur_color,
                   label=group, alpha=1, edgecolors='none')

    if not silent:
        ax.legend()
    ax.set_title("Распределение образцов по двум компонентам метрики")
    ax.set_ylabel("Разнообразие предложенных идей\n"
                  f" (ideas_diff: {y_ax_label})")
    ax.set_xlabel("Близость предложенных идей к запросу\n"
                  f" (query_ideas_sim: {x_ax_label})")

    if allow_plot:
        plt.show()
    else:
        plt.close()

    return fig


def draw_red_cross(figsize: Tuple[int, int]=(6, 4),
                   silent: bool=True):
    """
    Функция для получения графика-заглушки для ситуаций,
    когда построение полноценного невозможно
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axline((0, 0), (1, 1),
              c='red', linewidth=3)
    ax.axline((0, 1), (1, 0),
              c='red', linewidth=3)
    ax.text(0.35, 0.5, 'Нет данных', fontsize='x-large',
            fontfamily='monospace', fontweight='bold')
    ax.axis('off')
    if silent:
        plt.close()
    else:
        plt.show()
    return fig
