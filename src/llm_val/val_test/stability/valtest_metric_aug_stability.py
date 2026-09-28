import re
import random
import numpy as np
import typing as tp
import pandas as pd
from functools import partial

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.utils.paraphraser import batch_parapraser
from llm_val.utils.word_augmentations import batch_swap_word
from llm_val.utils.custom_instructions import mmlu_instructions
from llm_val.utils.human_augmentations import batch_augmentations
from llm_val.utils.asr_augmentations import batch_asr_augmentations
from llm_val.utils.custom_checkers import check_if_correct_with_options
from llm_val.utils.report_helper import semaphore_by_threshold, worst_semaphore


def report_valtest_metric_aug_stability(metric_baseline: tp.Dict[str, float],
                                        metrics_augmented: tp.Dict[str, tp.Dict[str, float]],
                                        main_metric: str,
                                        augmentations_list: tp.List[str] = [
                                            'symbol_swap',
                                            'butter_finger',
                                            'human_level_augmentations',
                                            'word_swap',
                                            'paraphrase',
                                            'asr_augmentations'],
                                        data_type: str='test',
                                        semaphore_threshold: tp.Tuple[float]=(0.05, 0.10),
                                        greater_is_better: bool=False,
                                        is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        metric_baseline: Словарь со значением метрики на данных без изменений
        metrics_augmented: Словарь со словарями значений метрики на аугментированных данных разными методами
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        augmentations_list: Список выполняемых аугментаций
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: Если True, то тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    metric_baseline = np.array(metric_baseline.get(main_metric))
    metrics_augmented = metrics_augmented.get(main_metric)
    metrics_augmented = np.array([metrics_augmented[name] for name in augmentations_list])
    absolute_difference = metric_baseline - metrics_augmented
    relative_difference = absolute_difference / metric_baseline * 100
    semaphore_colors = [semaphore_by_threshold(diff, semaphore_threshold, greater_is_better=greater_is_better) for diff in absolute_difference]
    color = 'gray' if is_info else worst_semaphore(semaphore_colors)
    semaphore_colors.append(color)
    results = pd.DataFrame({f'Исходное значение на {data_type}': metric_baseline.tolist(),
        'Тип преобразования': augmentations_list + ['Итог'],
        f'Значение {main_metric}': metrics_augmented.tolist() + ['-'],
        f'Абсолютное изменение {main_metric}': absolute_difference.tolist() + ['-'],
        f'Относительное изменение, % {main_metric}': relative_difference.tolist() + ['-'],
        'Результат теста': semaphore_colors})
    res_dict = {}
    res_dict['semaphore'] = color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [results]
    return res_dict


def valtest_metric_aug_stability(sampler: Sampler,
                                 scorer: Scorer,
                                 model: tp.Any,
                                 main_metric: str,
                                 augmentations_list: tp.List[str]=[
                                     'symbol_swap',
                                     'butter_finger',
                                     'human_level_augmentations',
                                     'word_swap',
                                     'paraphrase',
                                     'asr_augmentations'],
                                 p_augmented: float=0.1,
                                 paraphrase_model: tp.Optional[tp.Any]=None,
                                 paraphrase_tokenizer: tp.Optional[tp.Any]=None,
                                 paraphrase_model_config: tp.Optional[tp.Dict[str, tp.Any]]=None,
                                 paraphrase_tokenizer_config: tp.Optional[tp.Dict[str, tp.Any]]=None,
                                 rubq_answer_checker: tp.Callable[[str, tp.List[str]], bool]=check_if_correct_with_options,
                                 metric_agg: str='single_mean',
                                 data_type: str='test',
                                 semaphore_threshold: tp.Tuple[float]=(0.05, 0.10),
                                 greater_is_better: bool=False,
                                 is_info: bool=False,
                                 seed: int=42,
                                 metric_baseline: tp.Optional[tp.Dict[str, float]]=None,
                                 metrics_augmented: tp.Optional[tp.Dict[str, tp.Dict[str, float]]]=None,
                                 **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на стабильность прогноза модели в зависимости от аугментаций запросов  
    Parameters:
        sampler: Объект класса Sampler
        scorer: Объект класса Scorer
        model: Валидируемая модель; имеет метод predict(список вопросов)
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        augmentations_list: Список выполняемых аугментаций
        p_augmented: Вероятность аугментации символа/слова/предложения каждой аугментацией
        paraphrase_model: Языковая модель для перефразирования, имеет методы to_device и generate 
        paraphrase_tokenizer: Соответствующий модели токенизатор, имеет методы encode и decode
        paraphrase_model_config: Параметры языковой модели для перефразирования
        paraphrase_tokenizer_config: Параметры токенизатора
        rubq_answer_checker: Функция для проверки корректности ответа для RuBQ-like наборов
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        semaphore_threshold: Красный и желтый порог светофора по абсолютной разнице
        greater_is_better: True, если чем больше метрика, тем лучше
        is_info: True, если тест информативный
        seed: Начальное значение датчика случайных чисел
        metric_baseline: Опциональный словарь со значением метрики на данных без изменений
            Если не None, то подсчет метрики на данных без изменений не производится,
                и светофор выставляется на основе данного значения
        metrics_augmented: Опциональный словарь со словарями значений метрики на аугментированных данных разными методами
            Если не None, то подсчет метрики на аугментированных данных не производится,
                и светофор выставляется на основе данного значения
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    X = getattr(sampler, data_type)['X']
    char_aug_dict = {'human_level_augmentations': ['use_deletion',
                                                   'use_swap',
                                                   'use_insert',
                                                   'use_bf',
                                                   'use_special_symbol'],
                     'symbol_swap': ['use_swap'],
                     'butter_finger': ['use_bf'],
                     'insert': ['use_insert'],
                     'deletion': ['use_deletion']}
    if 'answer_options' in X.columns:
        get_score_func = get_mmlu_score
        cols_subset = ['true_answer', 'question', 'answer_options']
    else:
        get_score_func = partial(get_rubq_score, check_correct_answer=rubq_answer_checker)
        cols_subset = ['true_answer', 'question']
    if metrics_augmented is None:
        rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        metrics_augmented = {}
        for aug_name in augmentations_list:
            data = X.loc[:, cols_subset].copy(deep=True)
            if aug_name == 'word_swap':
                data['question'] = batch_swap_word(batch=data['question'].to_list(),
                                                   rng=rng,
                                                   p=p_augmented)
            elif aug_name == 'paraphrase':
                data['question'] = batch_parapraser(batch=data['question'].to_list(),
                                                    rng=rng,
                                                    model=paraphrase_model,
                                                    tokenizer=paraphrase_tokenizer,
                                                    model_config=paraphrase_model_config,
                                                    tokenizer_config=paraphrase_tokenizer_config,
                                                    p=p_augmented)
            elif aug_name == 'asr_augmentations':
                data['question'] = batch_asr_augmentations(batch=data['question'].to_list(),
                                                           rng=rng,
                                                           p=p_augmented)
            else:
                aug_params = {par: True for par in char_aug_dict[aug_name]}
                data = X.loc[:, cols_subset]
                data['question'] = batch_augmentations(batch=data['question'].to_list(),
                                                       rng=rng,
                                                       p=p_augmented,
                                                       **aug_params)
            score = get_score_func(data, scorer, model, metric_agg)
            for key, value in score.items():
                sub_dict = metrics_augmented.get(key, {})
                sub_dict[aug_name] = value
                metrics_augmented[key] = sub_dict
    if metric_baseline is None:
        data = X.loc[:, cols_subset]
        metric_baseline = get_score_func(data, scorer, model, metric_agg)
    report = report_valtest_metric_aug_stability(metric_baseline,
                                                 metrics_augmented,
                                                 main_metric,
                                                 augmentations_list,
                                                 data_type,
                                                 semaphore_threshold,
                                                 greater_is_better,
                                                 is_info)
    precomputed = {'metric_baseline': metric_baseline,
                   'metrics_augmented': metrics_augmented,
                   'augmentations_list': augmentations_list}
    return {'report': report,
            'precomputed': precomputed}


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
