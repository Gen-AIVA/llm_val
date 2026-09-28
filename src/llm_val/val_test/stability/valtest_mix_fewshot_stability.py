import re
import random
import numpy as np
import typing as tp
import pandas as pd
from copy import deepcopy
from tqdm.auto import tqdm
from functools import partial
from itertools import permutations

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold
from llm_val.utils.custom_checkers import check_if_correct_with_options


def report_valtest_mix_fewshot_stability(base_metric_value: tp.Dict[str, float],
                                         mix_few_shot_metric_value: tp.Dict[str, float],
                                         main_metric: str,
                                         semaphore_threshold: tp.Tuple[float, float]=(0.01, 0.02),
                                         is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        base_metric_value: Значение метрики с исходным порядком подсказок
        mix_few_shot_metric_value: Значение метрики с перемешанными подсказками
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        semaphore_threshold: Желтый и красный пороги светофора по относительной разнице
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    relative_metric_change = np.abs((base_metric_value[main_metric] - mix_few_shot_metric_value[main_metric]) \
                                   / base_metric_value[main_metric])
    color = 'gray' if is_info else semaphore_by_threshold(relative_metric_change, 
                                                          threshold=semaphore_threshold,
                                                          greater_is_better=False,
                                                          right_border_is_bigger=False)
    res_table = pd.DataFrame({'Относительное изменение метрики': [relative_metric_change],
                              'Светофор': color})
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else color
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_mix_fewshot_stability(sampler: Sampler,
                                  scorer: Scorer,
                                  model: tp.Any,
                                  task_type: str,
                                  main_metric: str,
                                  data_type: str='test',
                                  metric_agg: str='single_mean',
                                  semaphore_threshold: tp.Tuple[float, float]=(0.01, 0.02),
                                  rubq_checker: tp.Optional[tp.Callable[[tp.List[str]], float]]=check_if_correct_with_options,
                                  longanswer_autometric: tp.Optional[tp.Callable[[tp.Tuple[str]], float]]=None,
                                  permute_answer_options: bool=False,
                                  query_parser: tp.Optional[tp.Callable[[str], tp.Dict[str, tp.Any]]]=None,
                                  shot_permut_count: int=1,
                                  answers_permut_per_shot: int=1,
                                  random_state: int=29,
                                  is_info: bool=False,
                                  verbose: bool=False,
                                  base_metric_value: tp.Optional[float]=None,
                                  mix_few_shot_metric_value: tp.Optional[float]=None,
                                  **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на динамику изменения целевой метрики при изменении порядка few-shot подсказок 
    Parameters:
        sampler: Объект класса Sampler
        scorer: Объект класса Scorer
        model: Инстанс модели
        task_type: Тип решаемой задачи (mmlu, rubq, longanswer)
        main_metric: Название ключевой метрики качества. Hint: один из ключей словаря, который выдает scorer
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        metric_agg: Название агрегации метрики. Hint: один из ключей в словаре metrics в scorer'е
        semaphore_threshold: Желтый и красный пороги светофора по относительной разнице
        rubq_checker: Функция для проверки корректности ответов для RuBQ-like
        longanswer_autometric: Автометрика для подсчета качества работы модели на LongAnswer
        permute_answer_options: Если True, для MMLU-like будут перемешиваться варианты ответа
        query_parser: Парсер для запроса в few-shot, нужен только для MMLU-like при перемешивании вариантов ответов; 
                      Должен возвращать Dict со следующими полями:
                      answer_options: Обязательно, список пар (List[Tuple]) буква-ответ
                      question: Обязательно, вопрос
                      context: Опционально, контекст
                      answer_options_prefix: Опционально, префикс перед перчислением вариантов ответа
                      question_prefix: Опционально, префикс перед вопросом
                      context_prefix: Опционально, префикс перед контекстом
                      postfix: Опционально, завершение запроса, "передача слова" модели
        shot_permut_count: Количество перестановок подсказок, которые будут взяты для оценки метрик
        answers_permut_per_shot: Количество перестановок вариантов ответа MMLU-формата,
                                 которые будут взяты для оценки метрик
        random_state: Рандом стейт ¯\_(ツ)_/¯
        is_info: True, если тест информативный
        verbose: Если True, будет выводиться прогресс перемешивания подсказок
        base_metric_value: Опциональное значение целевой метрики с исходным порядком подсказок
        mix_few_shot_metric_value: Опциональное значение целевой метрики при перемешивании подсказок
            Если base_metric_value и mix_few_shot_metric_value не None, то подсчет не производится
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    tqdm.pandas()
    data = deepcopy(getattr(sampler, data_type)['X'])
    data.loc[data['few_shot'].isna(), 'few_shot'] = tuple([])
    metric_calculators = {'mmlu': get_mmlu_score,
                          'rubq': partial(get_rubq_score, checker=rubq_checker),
                          'longanswer': partial(get_la_score, autometric=longanswer_autometric)}
    metric_calc_fn = metric_calculators[task_type.lower()]
    
    if not base_metric_value:
        base_metric_value = metric_calc_fn(data=data, 
                                           scorer=scorer,
                                           model=model,
                                           metric_agg=metric_agg,
                                           few_shot_column='few_shot')   
        
    if not mix_few_shot_metric_value:
        if task_type.lower() != 'mmlu':
            permute_answer_options = False
        if verbose:
            data['few_shot_permutations'] = data['few_shot'].progress_apply(permute_fewshot, 
                                                                            permute_answer_options=permute_answer_options,
                                                                            query_parser=query_parser,
                                                                            shot_permut_count=shot_permut_count,
                                                                            answers_permut_per_shot=answers_permut_per_shot,
                                                                            random_state=random_state)
        else:
            data['few_shot_permutations'] = data['few_shot'].apply(permute_fewshot, 
                                                                   permute_answer_options=permute_answer_options,
                                                                   query_parser=query_parser,
                                                                   shot_permut_count=shot_permut_count,
                                                                   answers_permut_per_shot=answers_permut_per_shot,
                                                                   random_state=random_state)
        
        flat_along_permutations = data.explode('few_shot_permutations').reset_index(drop=True)
        if permute_answer_options:
            flat_along_permutations = flat_along_permutations.explode('few_shot_permutations').reset_index(drop=True)
        else:
            flat_along_permutations['few_shot_permutations'] = flat_along_permutations['few_shot_permutations'].apply(lambda x: flatten_list(x))
        mix_few_shot_metric_value = metric_calc_fn(data=flat_along_permutations, 
                                                   scorer=scorer,
                                                   model=model,
                                                   metric_agg=metric_agg,
                                                   few_shot_column='few_shot_permutations')

    report = report_valtest_mix_fewshot_stability(base_metric_value=base_metric_value,
                                                  mix_few_shot_metric_value=mix_few_shot_metric_value,
                                                  main_metric=main_metric,
                                                  semaphore_threshold=semaphore_threshold,
                                                  is_info=is_info)
    precomputed = {'base_metric_value': base_metric_value,
                   'mix_few_shot_metric_value': mix_few_shot_metric_value}
    return {'report': report,
            'precomputed': precomputed}


def flatten_list(x: tp.List[tp.Any]):
    '''
    Снижение размерности списка
    Parameters:
        x: Список
    Return:
        Сплющенный список
    '''
    new = []
    for item in x:
        new.extend(item)
    return new


def permute_fewshot(few_shot: tp.Tuple[tp.Dict[str, str]],
                    permute_answer_options: bool=True,
                    shot_permut_count: int=1,
                    answers_permut_per_shot: int=1,
                    query_parser: tp.Optional[tp.Callable[[str], tp.Dict[str, tp.Any]]]=None,
                    random_state: int=17) -> tp.Tuple[tp.Dict[str, str]]:
    '''
    Создание и выбор перестановок пар вопрос-ответ в истории диалога с GC
    Parameters:
        few_shot: Few-shot формата GigaChat API
        permute_answer_options: Если True, для MMLU-формата будут перемешиваться варианты ответа
        shot_permut_count: Количество перестановок подсказок, которые будут взяты для оценки метрик
        answers_permut_per_shot: Количество перестановок вариантов ответов MMLU-формата,
                                 которые будут взяты для оценки метрик
        query_parser: Парсер для запроса в few-shot, нужен только для MMLU-like при перемешивании вариантов ответов. 
                      Должен возвращать Dict со следующими полями
                      answer_options: Обязательно, список пар (List[Tuple]) буква-ответ
                      question: Обязательно, вопрос
                      context: Опционально, контекст
                      answer_options_prefix: Опционально, префикс перед перчислением вариантов ответа
                      question_prefix: Опционально, префикс перед вопросом
                      context_prefix: Опционально, префикс перед контекстом
                      postfix: Опционально, завершение запроса, "передача слова" модели
        random_state: Рандом стейт ¯\_(ツ)_/¯
    Return:
        Tuple с перестановками 
    '''
    random.seed(random_state)
    
    qa_pairs = []
    for q in range(0, len(few_shot), 2):
        qa_pairs.append([few_shot[q], few_shot[q+1]])
        
    few_shot_permutations = []
    pair_permutations = [p for p in permutations(qa_pairs)]
    shot_permut_count = min((shot_permut_count, len(pair_permutations)))
    pairs_perms_to_use = random.choices(pair_permutations, k=shot_permut_count)
    for pairs_perm in pairs_perms_to_use:
        if permute_answer_options:
            pairs_with_permuted_answers = []
            for pair in pairs_perm:
                options_perms_to_save = permute_answer_options_for_pair(pair=pair,
                                                                        query_parser=query_parser,
                                                                        answers_perm_count=answers_permut_per_shot)
                pairs_with_permuted_answers.append(options_perms_to_save)
            pairs_perm = reshape_permutations(pairs_with_permuted_answers)
        few_shot_permutations.append(pairs_perm)
    return tuple(few_shot_permutations)
             

def permute_answer_options_for_pair(pair: tp.List[tp.Dict[str, str]],
                                    query_parser: tp.Callable[[str], tp.Dict[str, tp.Any]],
                                    answers_perm_count: int=1) -> tp.List[tp.List[tp.Dict[str, str]]]:
    '''
    Создание answers_perm_count перестановок вариантов ответов в MMLU-like запросе.
    Parameters: 
        pair: Пара вопрос-ответ в целевом формате GigaChat API
        query_parser: Парсер для запроса в few-shot, см. permute_fewshot
        answers_perm_count: Количество перестановок вариантов ответа
    Return: 
        List с парами вопрос-ответ с перемешанными вариантами ответов
    ''' 
    query = pair[0]['content']
    true_letter = pair[1]['content']
    parsed_query = query_parser(query)
    answer_options = dict(parsed_query['answer_options'])
    question = parsed_query['question']
    context = parsed_query.get('context', '')
    answer_options_prefix = parsed_query.get('answer_options_prefix', 'Варианты ответа:\n')
    question_prefix = parsed_query.get('question_prefix', 'Вопрос:\n')
    context_prefix = parsed_query.get('context_prefix', 'Контекст:\n')
    postfix = parsed_query.get('postfix', 'Твой ответ:\n')
    total_letters = list(answer_options.keys())
    true_answer = answer_options[true_letter]

    permuted_options = [[(total_letters[i], perm[i]) for i in range(len(perm))] for perm in permutations(answer_options.values())]
    answers_perm_count = min((answers_perm_count, len(permuted_options)))
    options_perms_to_use = random.choices(permuted_options, k=answers_perm_count)
    options_perms_to_save = []
    for options_perm in options_perms_to_use:
        new_query, new_answer = reconstruct_query(options_perm,
                                                  question=question,
                                                  true_answer=true_answer,
                                                  answer_options=answer_options,
                                                  context=context,
                                                  answer_options_prefix=answer_options_prefix,
                                                  question_prefix=question_prefix,
                                                  context_prefix=context_prefix,
                                                  postfix=postfix)
        options_perms_to_save.append([new_query, new_answer])
    
    return options_perms_to_save
        
    
def reconstruct_query(options_perm: tp.List[tp.Tuple[str]],
                      question: str,
                      true_answer: str,
                      answer_options: tp.Dict[str, str],
                      context: tp.Optional[str]='',
                      answer_options_prefix: tp.Optional[str]='Варианты ответа:\n',
                      question_prefix: tp.Optional[str]='Вопрос:\n',
                      context_prefix: tp.Optional[str]='Контекст:\n',
                      postfix: tp.Optional[str]='Твой ответ:\n') -> tp.Tuple[tp.Dict[str, str]]:
    '''
    Воссоздание запроса в модель c перемешанными вариантами ответа
    Parameters:
        options_perm: Список Tuple с перемешанными вариантами ответов в виде (буква, ответ)
        question: Вопрос
        true_answer: Строка с полным правильным ответом
        answer_options: Словарь буква-ответ
        context: Контекст
        answer_options_prefix: Префикс для вариантов ответа
        question_prefix: Префикс для вопроса
        context_prefix: Префикс контекста
        postfix: Постфикс запроса
    Return:
        Запрос в модель и ее ответ в формате GigaChat API
    '''
    options_perm = dict(options_perm)
    options_reversed = {v: k for k, v in options_perm.items()}
    new_question = context_prefix + ' ' + context + '.\n'
    new_question += question_prefix + ' ' + question
    new_question += answer_options_prefix + ' '
    new_options = '\n'.join([f'[{k}] {v}.' for k, v in options_perm.items()]) 
    new_question += new_options + '.\n'
    new_question += postfix
    new_true_letter = options_reversed[true_answer]
    new_query = {'role': 'user',
                 'content': new_question}
    new_answer = {'role': 'assistant',
                  'content': new_true_letter}
    return new_query, new_answer
    

def reshape_permutations(qa_pairs: tp.List[tp.List[tp.Any]]) -> tp.List[tp.List[tp.Any]]:
    '''
    Техническая функция для трансформации групп вопросов 
    после перемешивания вариантов ответов в истории диалогов 
    Parameters: 
        qa_pairs: Список списков следующего вида с запросами в модель. qN - вопрос N, vM - перестановка M.
                  q1v1-q1v2-q1v3
                  q2v1-q2v2-q2v3
                  q3v1-q3v2-q3v3
    Return:
        Список запросов в модель в виде историй диалогов
        q1v1-q2v1-q3v1
        q1v2-q2v2-q3v2
        q1v3-q2v3-q3v3        
    '''
    reshaped_pairs = []
    for option_idx in range(len(qa_pairs[0])):
        row = []
        for pair_idx in range(len(qa_pairs)):
            row.extend(qa_pairs[pair_idx][option_idx])
        reshaped_pairs.append(row)   
    return reshaped_pairs
            
    
    
def get_rubq_score(data: pd.DataFrame,
                   scorer: Scorer,
                   model: tp.Any,
                   metric_agg: str,
                   few_shot_column: str,
                   checker: tp.Callable[[str], tp.List[str]]=check_if_correct_with_options) -> tp.Dict[str, float]:
    '''
    Подсчет среднего качества работы модели на RuBQ-like датасете
    Parameters:
        data: Исследуемый набор данных, имеет колонки question, answer_options, true_answer и few_shot
        scorer: Объект класса Scorer
        model: Валидируемая модель; имеет метод predict(список вопросов)
        metric_agg: Название метрики качества
        few_shot_column: Колонка с few-shot,
        checker: Функция для проверки корректности ответа
    Return:
        Значение метрики качества на данном наборе
    '''
    data['query'] = data.apply(lambda x: 'Контекст:\n' + str(x['context']) + '\nВопрос:\n' + x['question'], axis=1)
    data['model_answer'] = model.predict(data['query'].to_list(), chat_history=data[few_shot_column].to_list())
    data['true_answer'] = data['true_answer'].apply(lambda x: x if (isinstance(x, list) or isinstance(x, tuple)) else [x])
    data['is_correct'] = data.apply(lambda x: checker(x['model_answer'], x['true_answer']), axis=1)
    dict_from_rubq = {'X': None, 'y': pd.DataFrame({'metric_value': (data['is_correct']).astype(int)})}
    return scorer.calc(sampler=Sampler(test=dict_from_rubq),
                       data_type='test',
                       metric_name=metric_agg)


def get_mmlu_score(data: pd.DataFrame,
                   scorer: Scorer,
                   model: tp.Any,
                   metric_agg: str,
                   few_shot_column: str) -> tp.Dict[str, float]:
    '''
    Подсчет среднего качества работы модели на MMLU-like датасете
    Parameters:
        data: Исследуемый набор данных, имеет колонки question, answer_options, true_answer и few_shot
        scorer: Объект класса Scorer
        model: Валидируемая модель; имеет метод predict(список вопросов)
        metric_agg: Название метрики качества
        few_shot_column: Колонка с few-shot
    Return:
        Значение метрики качества на данном наборе
    '''
    data['query'] = data.apply(lambda x: 'Контекст:\n' + str(x['context']) + '\nВопрос:\n' + x['question'] + 
                                         '\nВарианты ответа\n' + str(x['answer_options']) + '\nТвой ответ:', axis=1)
    data['model_answer'] = model.predict(data['query'].to_list(), chat_history=data[few_shot_column].to_list())
    data['model_answer'] = data['model_answer'].str.extract(r'([a-zA-Z])', expand=False).fillna('')
    dict_from_mmlu = {'X': None, 'y': pd.DataFrame({'metric_value': (data['true_answer'] == data['model_answer']).astype(int)})}
    return scorer.calc(sampler=Sampler(test=dict_from_mmlu),
                       data_type='test',
                       metric_name=metric_agg)


def get_la_score(data: pd.DataFrame,
                 scorer: Scorer,
                 model: tp.Any,
                 few_shot_column: str,
                 metric_agg: str,
                 autometric: tp.Callable[[tp.Tuple[str, str]], float]) -> tp.Dict[str, float]:
    '''
    Подсчет среднего качества работы модели на LongAnswer датасете
    Parameters:
        data: Исследуемый набор данных, имеет колонки question, answer_options, true_answer и few_shot
        scorer: Объект класса Scorer
        model: Валидируемая модель; имеет метод predict(список вопросов)
        metric_agg: Название метрики качества
        few_shot_column: Колонка с few-shot
        autometric: Автометрика для подсчета качества работы модели на LongAnswer
    Return:
        Значение метрики качества на данном наборе
    '''
    data['query'] = data.apply(lambda x: 'Контекст:\n' + str(x['context']) + '\nВопрос:\n' + x['question'], axis=1)
    data['model_answer'] = model.predict(data['query'].to_list(), chat_history=data[few_shot_column].to_list())
    dict_from_la = {'X': None, 'y': pd.DataFrame({'metric_value': data.apply(lambda x: autometric(x['true_answer'], x['model_answer']), axis=1)})}
    return scorer.calc(sampler=Sampler(test=dict_from_la),
                       data_type='test',
                       metric_name=metric_agg)