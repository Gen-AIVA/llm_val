import re
import numpy as np
import typing as tp
import pandas as pd
from functools import partial
from collections import Counter

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.utils.stop_words import STOP_WORDS
from llm_val.utils.custom_instructions import mmlu_instructions
from llm_val.utils.custom_checkers import check_if_correct_with_options
from llm_val.utils.report_helper import semaphore_by_threshold, worst_semaphore


def report_valtest_metric_aug_stability(metric_baseline: float,
                                        metrics_augmented: tp.Dict[str, float],
                                        augmentations_list: tp.List[str]=['stop_words',
                                                                          'rare_words',
                                                                          'popular_words',
                                                                          'full_words'],
                                        data_type: str='test',
                                        semaphore_threshold: tp.Tuple[float]=(0.015, 0.03),
                                        greater_is_better: bool=True,
                                        is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        metric_baseline: Исходное значение метрики
        metrics_augmented: Словарь со значениями метрики после аугментации.
        augmentations_list: Список преобразований, которые будут использованы в тесте
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold: Кортеж с границами светофора. Границы в кортеже указываются в порядке возрастания  
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    metrics_augmented_values = [metrics_augmented[name] for name in augmentations_list]
    absolute_difference = metric_baseline - np.array(metrics_augmented_values)
    semaphore_colors = [semaphore_by_threshold(diff, semaphore_threshold, greater_is_better=False) for diff in absolute_difference]
    color = 'gray' if is_info else worst_semaphore(semaphore_colors)
    semaphore_colors.append(color)

    results = pd.DataFrame({'Тип преобразования': augmentations_list + ['Итог'],
                            f'Исходное значение на {data_type}': [metric_baseline] * len(augmentations_list) + ['-'],
                            'Значение метрики после аугментации': metrics_augmented_values + ['-'],
                            'Абсолютное изменение': absolute_difference.tolist() + ['-'],
                            'Результат теста': semaphore_colors})

    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [results]
    return res_dict


def valtest_metric_aug_stability(sampler: Sampler,
                                 scorer: Scorer,
                                 model: tp.Any,
                                 augmentations_list: tp.List[str]=['stop_words',
                                                                   'rare_words',
                                                                   'popular_words',
                                                                   'full_words'],
                                 metric_agg: str='single_mean',
                                 metric_name: str='single_mean_metric_value',
                                 data_type: str='test',
                                 percent: int=90,
                                 min_word: int=1,
                                 language: str='russian',
                                 semaphore_threshold: tp.Tuple[float]=(0.015, 0.03),
                                 greater_is_better: bool=True,
                                 is_info: bool=False,
                                 rubq_answer_checker: tp.Callable[[str, tp.List[str]], bool]=check_if_correct_with_options,
                                 metric_baseline: tp.Optional[float]=None,
                                 metrics_augmented: tp.Optional[tp.Dict[str, float]]=None,
                                 **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на определение влияния стоп-слов, редких и популярных слов на результат работы модели
    Parameters:
        sampler: Объект класса Sampler
        scorer: Объект класса Scorer
        model: Валидируемая модель, которая должна иметь функцию predict(список вопросов)
        augmentations_list: Список преобразований, которые будут использованы в тесте
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        metric_name: Название метрики, которое будет извлекаться из словаря, который отдает scorer
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        percent: Процентное значение для удаления популярных слов
        min_word: Минимальное количество слов для удаления редких слов
        language: Язык данных
        semaphore_threshold: Кортеж с границами светофора. Границы в кортеже указываются в порядке возрастания
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info:  True, если тест информативный
        rubq_answer_checker: Функция проверки ответа для RuBQ-like наборов
        metric_baseline: Опциональное исходное значение метрики
        metrics_augmented: Опциональный словарь со значениями метрики после аугментации. Если metric_baseline и metrics_augmented не None, то подсчет метрик не производится и на их основе выставляется светофор
    Return:
        report в соответствии с концепцией llm_val
    '''
    if metric_baseline is None or metrics_augmented is None:
        data = getattr(sampler, data_type)['X']
        word_counts = Counter(' '.join(data['question']).split())

        augmentations = {
            'stop_words': partial(remove_stop_words, language=language),
            'rare_words': partial(remove_rare_words, word_counts=word_counts, min_word=min_word),
            'popular_words': partial(remove_popular_words, word_counts=word_counts, percent=percent),
            'full_words': partial(remove_full_words, word_counts=word_counts, language=language, min_word=min_word, percent=percent)
        }

        if metric_baseline is None:
            metric_baseline = scorer.calc(sampler=sampler, data_type=data_type, metric_name=metric_agg)[metric_name]

        metrics_augmented = {}
        for augmentation in augmentations_list:
            augmented_data = data.copy()
            augmented_data['question'] = augmented_data['question'].apply(augmentations[augmentation])
            
            if 'answer_options' in augmented_data.columns:
                score_func = partial(get_mmlu_score, scorer=scorer, model=model, metric_name=metric_agg)
            else:
                score_func = partial(get_rubq_score, scorer=scorer, model=model, metric_name=metric_agg, check_correct_answer=rubq_answer_checker)
            
            metrics_augmented[augmentation] = score_func(augmented_data)[metric_name]

    report = report_valtest_metric_aug_stability(metric_baseline,
                                                 metrics_augmented,
                                                 augmentations_list,
                                                 data_type,
                                                 semaphore_threshold,
                                                 greater_is_better,
                                                 is_info)

    precomputed = {'metric_baseline': metric_baseline,
                   'metrics_augmented': metrics_augmented}

    return {'report': report,
            'precomputed': precomputed }


def remove_stop_words(text: str, 
                      language: str) -> str:
    '''
    Удаление стоп-слов из текста.
    Parameters:
        text: Текст
        language: Язык данных
    Return:
        Текст без стоп-слов
    '''
    return ' '.join([word for word in text.split() if word.lower() not in STOP_WORDS[language]])


def remove_rare_words(text: str, 
                      word_counts: Counter, 
                      min_word: int) -> str:
    '''
    Удаление редких слов.
    Parameters:
        text: Текст
        word_counts: Словарь с количеством слов
        min_word: Минимальное количество слов для удаления редких слов
    Return:
        Текст без редких слов
    '''
    return ' '.join([word for word in text.split() if word_counts[word] > min_word])


def remove_popular_words(text: str, 
                         word_counts: Counter, 
                         percent: float) -> str:
    '''
    Удаление популярных слов, включая стоп-слова, если они популярны.
    Parameters:
        text: Текст
        word_counts: Словарь с количеством слов
        percent: Процент популярных слов для удаления
    Return:
        Текст без популярных слов, включая стоп-слова, если они популярны
    '''
    threshold = np.percentile(list(word_counts.values()), percent)
    filtered_words = [word for word in text.split() if word_counts[word] < threshold]
    return ' '.join(filtered_words)


def remove_full_words(text: str, 
                      word_counts: Counter, 
                      language: str, 
                      min_word: int, 
                      percent: float) -> str:
    '''
    Удаление редких, популярных и стоп-слов
    Parameters:
        text: Текст
        word_counts: Словарь с количеством слов
        language: Язык данных
        min_word: Минимальное количество слов для удаления редких слов
        percent: Процентное значение для удаления популярных слов
    Return:
        Текст без редких, популярных и стоп-слов
    '''
    return remove_popular_words(remove_rare_words(remove_stop_words(text, 
                                                                    language), 
                                                  word_counts, 
                                                  min_word), 
                                word_counts, 
                                percent)


def get_rubq_score(data: pd.DataFrame,
                   scorer: Scorer,
                   model: tp.Any,
                   metric_name: str,
                   check_correct_answer: tp.Callable[[str, tp.List[str]], bool]=check_if_correct_with_options
                   ) -> tp.Dict[str, float]:
    '''
    Подсчет среднего качества работы модели на RuBQ-like датасете
    Parameters:
        data: Исследуемый набор данных, имеет колонки question, answer_options и true_answer
        scorer: Объект класса Scorer
        model: Валидируемая модель; имеет метод predict(список вопросов)
        metric_name: Название метрики качества
        check_correct_answer: Функция для проверки корректности ответа 
    Return:
        Значение метрики качества на данном наборе
    '''
    data['model_answer'] = model.predict(data['question'].to_list())
    data['is_correct'] = data.apply(lambda x: check_correct_answer(x['model_answer'], x['true_answer']), axis=1)
    dict_from_rubq = {'X': None, 'y': pd.DataFrame({'metric_value': (data['is_correct']).astype(int)})}
    return scorer.calc(sampler=Sampler(test=dict_from_rubq),
                       data_type='test',
                       metric_name=metric_name)


def get_mmlu_score(data: pd.DataFrame,
                   scorer: Scorer,
                   model: tp.Any,
                   metric_name: str) -> tp.Dict[str, float]:
    '''
    Подсчет среднего качества работы модели на MMLU-like датасете
    Parameters:
        data: Исследуемый набор данных, имеет колонки question, answer_options и true_answer
        scorer: Объект класса Scorer
        model: Валидируемая модель; имеет метод predict(список вопросов)
        metric_name: Название метрики качества
    Return:
        Значение метрики качества на данном наборе
    '''
    mmlu_prompt = mmlu_instructions['instruction'] + mmlu_instructions['shot']
    options_prefix, postfix = mmlu_instructions['options_prefix'], mmlu_instructions['postfix']
    data['question'] = data.apply(lambda x: mmlu_prompt + x['question'] + options_prefix + str(x['answer_options']) + postfix, axis=1)
    data['model_answer'] = model.predict(data['question'].to_list())
    data['model_answer'] = data['model_answer'].apply(lambda x: re.match(r'[a-zA-Z]', x)[0])
    dict_from_mmlu = {'X': None, 'y': pd.DataFrame({'metric_value': (data['true_answer'] == data['model_answer']).astype(int)})}
    return scorer.calc(sampler=Sampler(test=dict_from_mmlu),
                       data_type='test',
                       metric_name=metric_name)