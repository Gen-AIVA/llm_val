import re
import random
import numpy as np
import typing as tp
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.utils.custom_instructions import mmlu_instructions
from llm_val.utils.custom_checkers import check_if_correct_with_options
from llm_val.utils.report_helper import quantile_semaphore_without_yellow

def report_valtest_indifferent(original_metric: float,
                               confidence_interval: tp.Tuple[float, float],
                               num_observations: int,
                               random_metrics: tp.List[float],
                               metric_name: str='single_mean_metric_value',
                               confidence: float=0.95,
                               is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создает отчет о результатах теста на индифферентность.
    Parameters:
        original_metric: Значение метрики для оригинальных данных.
        confidence_interval: Доверительный интервал.
        num_observations: Количество наблюдений в датасете.
        random_metrics: Список метрик для случайных текстов.
        metric_name: Название метрики качества.
        is_info: True, если тест информативный.
    Returns:
        Отчет в соответствии с концепцией llm_val.
    '''
    color = 'gray' if is_info else quantile_semaphore_without_yellow(original_metric, 
                                                                     confidence_interval,
                                                                     inner_is_better=False)
    fig = plot_confidence_interval(original_metric, 
                                   confidence_interval, 
                                   random_metrics,
                                   confidence,
                                   metric_name=metric_name)

    res_table = pd.DataFrame({'Значение метрики для исходных данных': [original_metric],
                              'Доверительный интервал для сгенерированных данных': str(confidence_interval),
                              'Количество наблюдений в датасете': [num_observations],
                              'Результат': [color]})

    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = [fig]
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_indifferent(sampler: Sampler,
                        scorer: Scorer,
                        model: tp.Any=None,
                        metric_name: str='single_mean_metric_value',
                        metric_agg: str='single_mean',
                        data_type: str='test',
                        confidence: float=0.95,
                        n_repeat: int=2,
                        check_correct_answer: tp.Callable[[str, tp.List[str]], bool]=check_if_correct_with_options,
                        n_gram_param: int=2,
                        is_info: bool=False,
                        **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на индифферентность модели относительно входных данных.
    Parameters:
        sampler: Объект класса Sampler.
        scorer: Объект класса Scorer.
        model: Валидируемая модель; имеет метод predict(список вопросов).
        metric_name: Название метрики качества.
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        confidence: Уровень доверия для доверительного интервала.
        n_repeat: Количество случайных текстов, генерируемых для каждого предложения.
        check_correct_answer: Функция для проверки корректности ответа для задачи RuBQ.
        n_gram_param: Длина для n-грамм.
        is_info: True, если тест информативный.
    Returns:
        Словарь в соответствии с концепцией llm_val.
    '''
    dataset_X_orig = getattr(sampler, data_type)['X'].copy(deep=True)
    original_metric = scorer.calc(sampler=sampler, 
                                  data_type=data_type, 
                                  metric_name=metric_agg)[metric_name]
    random_metrics = []
    y_to_scorers = {}
    
    prefix_to_check_aug_preds = 'pred_' + 'aug_valtest_indifferent_'
    if getattr(sampler, data_type)['X'].columns.str.startswith(prefix_to_check_aug_preds).any():
        dataset_X_aug = getattr(sampler, data_type)['X'].filter(regex=f'^{prefix_to_check_aug_preds}')
        for aug_column in dataset_X_aug.columns:
            if 'answer_options' in dataset_X_orig.columns:
                y_to_scorers[aug_column] = \
                    pd.DataFrame({'metric_value': (dataset_X_orig['true_answer'] == dataset_X_orig[aug_column]).astype(int)})
            else:
                y_to_scorers[aug_column] = \
                    pd.DataFrame({'metric_value': dataset_X_orig.apply(lambda x: check_correct_answer(x[aug_column], 
                                                                                                      x['true_answer']), axis=1).astype(int)})      
    else:    
        aug_df = get_augmented_data(sampler,
                                    data_type,
                                    n_repeat,
                                    n_gram_param)
        getattr(sampler, data_type)['X'][aug_df.columns] = aug_df
        
        prefix_to_check_augs = 'aug_valtest_indifferent_'
        dataset_X_aug_wo_preds = getattr(sampler, data_type)['X'].filter(regex=f'^{prefix_to_check_augs}')
        for aug_column in dataset_X_aug_wo_preds.columns:
            dataset_X_orig['question'] = aug_column
            if 'answer_options' in dataset_X_orig.columns:
                dict_with_predicts = get_mmlu_predicts(dataset_X_orig, 
                                                       model)
            else:
                dict_with_predicts = get_rubq_predicts(dataset_X_orig, 
                                                       model,
                                                       check_correct_answer)
            getattr(sampler, data_type)['X']['pred_' + aug_column] = dict_with_predicts['X']['model_answer']
            y_to_scorers['pred_' + aug_column] = dict_with_predicts['y']
        dataset_X_aug = getattr(sampler, data_type)['X'].filter(regex=f'^{prefix_to_check_aug_preds}')
    
    for aug_column in dataset_X_aug.columns:
        dict_for_scorer = {'X': None,
                           'y': y_to_scorers[aug_column]}
        metric_value = scorer.calc(sampler=Sampler(test=dict_for_scorer),
                                   data_type='test',
                                   metric_name=metric_agg)[metric_name]
        random_metrics.append(metric_value)
            
    confidence_interval = calculate_confidence_interval(random_metrics, 
                                                        confidence)
    num_observations = len(dataset_X_orig)
    report = report_valtest_indifferent(original_metric,
                                        confidence_interval, 
                                        num_observations, 
                                        random_metrics, 
                                        metric_name, 
                                        confidence, 
                                        is_info)
    precomputed = {}
    return {'report': report, 
            'precomputed': precomputed}


def plot_confidence_interval(original_metric: float,
                             confidence_interval: tp.Tuple[float, float],
                             random_metrics: tp.List[float],
                             confidence: float=0.95,
                             metric_name: str='single_mean_metric_value') -> plt.Figure:
    '''
    Строит график доверительного интервала и отображает значение метрики для оригинальных данных.
    Parameters:
        original_metric: Значение метрики для оригинальных данных.
        confidence_interval: Доверительный интервал (нижняя граница, верхняя граница).
        random_metrics: Список метрик для случайных текстов.
        confidence: Уровень доверия для доверительного интервала.
        metric_name: Название метрики качества.
    Returns:
        Объект Figure с графиком.
    '''
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle(f'Доверительный интервал и значение метрики для оригинальных данных ({metric_name})', fontsize=14)
    ax.set_ylabel('% of evidence')
    min_ = min(original_metric, min(random_metrics))
    min_ = min_ * (0.9 if min_ > 0 else 1.1)
    max_ = max(original_metric, max(random_metrics))
    if 0 <= max_ <= 1:
        max_ = min(max_ * 1.1, 1.001)
    else:
        max_ = max_ * 1.1
    ax.set_xlim(min_, max_)
    bins = np.linspace(min(original_metric, min(random_metrics)),
                       max(original_metric, max(random_metrics)),
                       100)
    ax.axvline(original_metric, color='red', linestyle='--', label=f'Исходная метрика: {original_metric:.3f}')
    ax.hist(random_metrics, bins=bins, label='Случайные данные', alpha=0.7,
            weights=np.ones_like(random_metrics) / float(len(random_metrics)), color='blue')
    ax.plot([confidence_interval[0], confidence_interval[0], confidence_interval[1], confidence_interval[1]],
             [0, 0.01, 0.01, 0],
             label=f'{confidence * 100:.0f}% доверительный интервал', color='green', linestyle='dashed')
    ax.legend(loc='upper left')
    return fig


def generate_random_texts(dataset_X: pd.DataFrame,
                          text_column: str='question',
                          n_repeat: int=1,
                          n_gram_param: int=2) -> tp.List[str]:
    '''
    Генерирует случайные тексты на основе n-грамм из всех текстов в указанном столбце датасета.
    Parameters:
        dataset_X: DataFrame с данными.
        text_column: Столбец датасета, из которого извлекаются тексты.
        n_repeat: Количество повторений для каждого текста.
        n_gram_param: Длина для n-грамм.
    Returns:
        Список случайных текстов.
    '''
    all_ngrams = []
    total_words = 0
    for text in dataset_X[text_column]:
        words = text.split()
        total_words += len(words)
        ngrams_in_text = [tuple(words[i:i + n_gram_param]) for i in range(len(words) - n_gram_param + 1)]
        all_ngrams.extend(ngrams_in_text)
    avg_words_per_text = total_words / len(dataset_X)
    avg_ngrams_per_text = int(np.ceil(avg_words_per_text / n_gram_param))
    random_texts = []
    for _ in range(len(dataset_X) * n_repeat):
        random_ngrams = random.sample(all_ngrams, k=avg_ngrams_per_text)
        random_text = ' '.join([' '.join(gram) for gram in random_ngrams])
        random_texts.append(random_text)
    return random_texts


def calculate_confidence_interval(metrics: tp.List[float], 
                                  confidence: float=0.95) -> tp.Tuple[float, float]:
    '''
    Рассчитывает доверительный интервал для метрик.
    Parameters:
        metrics: Список значений метрик.
        confidence: Уровень доверия.
    Returns:
        Доверительный интервал.
    '''
    mean = np.mean(metrics)
    sem = stats.sem(metrics)
    h = sem * stats.t.ppf((1 + confidence) / 2, len(metrics) - 1)
    return round(mean - h, 3), round(mean + h, 3)


def get_rubq_predicts(data: pd.DataFrame,
                      model: tp.Any,
                      check_correct_answer: tp.Callable[[str, tp.List[str]], bool]=check_if_correct_with_options) -> tp.Dict[str, float]:
    '''
    Получение предсказаний модели для RuBQ-like датасета.
    Parameters:
        data: Исследуемый набор данных, содержит колонки question и true_answer.
        model: Валидируемая модель; имеет метод predict(список вопросов).
        check_correct_answer: Функция для проверки корректности ответа для задачи RuBQ.
    Returns:
        Словарь, который можно использовать для подсчета значения метрики качества.
    '''
    data['model_answer'] = model.predict(data['question'].to_list())
    data['is_correct'] = data.apply(lambda x: check_correct_answer(x['model_answer'], x['true_answer']), axis=1)
    dict_from_rubq = {'X': data, 
                      'y': pd.DataFrame({'metric_value': (data['is_correct']).astype(int)})}
    return dict_from_rubq


def get_mmlu_predicts(data: pd.DataFrame,
                      model: tp.Any) -> tp.Dict[str, float]:
    '''
    Получение предсказаний модели для MMLU-like датасета.
    Parameters:
        data: Исследуемый набор датасет.
        model: Валидируемая модель; имеет метод predict(список вопросов).
    Returns:
        Словарь, который можно использовать для подсчета значения метрики качества.
    '''
    mmlu_prompt = mmlu_instructions['instruction'] + mmlu_instructions['shot']
    options_prefix, postfix = mmlu_instructions['options_prefix'], mmlu_instructions['postfix']
    data['question'] = data.apply(lambda x: mmlu_prompt + x['question'] + options_prefix + str(x['answer_options']) + postfix, axis=1)
    data['model_answer'] = model.predict(data['question'].to_list())
    data['model_answer'] = data['model_answer'].apply(lambda x: re.match(r'[a-zA-Z]', x)[0])
    dict_from_mmlu = {'X': data, 
                      'y': pd.DataFrame({'metric_value': (data['true_answer'] == data['model_answer']).astype(int)})}
    return dict_from_mmlu


def get_augmented_data(sampler: Sampler,
                       data_type: str='test',
                       n_repeat: int=2,
                       n_gram_param: int=2) -> pd.DataFrame:
    '''
    Собирает датафрейм с созданными случайными текстами.
    Parameters:
        sampler: Объект класса Sampler.
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        n_repeat: Количество повторений для каждого текста.
        n_gram_param: Длина для n-грамм.
    Returns:
        Датафрейм случайных текстов.
    '''
    aug_df = pd.DataFrame()
    for cur_repeat in range(n_repeat):
        random_texts = generate_random_texts(dataset_X=getattr(sampler, data_type)['X'],
                                             n_gram_param=n_gram_param)
        aug_df['aug_valtest_indifferent_' + str(cur_repeat)] = random_texts
    return aug_df