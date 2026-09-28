import os
import glob
import time
import numpy as np
import transformers
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from matplotlib.colors import LinearSegmentedColormap

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.extract_evaluators.chrf import CHRF
from llm_val.model.MultiThreadGigaModel import MultiThreadGigaModel
from llm_val.utils.extract_evaluators.exact_match import ExactMatch


def report_valtest_needle_test(pivot_table: pd.DataFrame, 
                               figs: Optional[List[Any]]=None) -> Dict[str, Any]:
    """
    Создание report'а
    Parameters:
        pivot_table: Усреднение метрик для всех иголок по параметрам глубины и длины.
        figs: Список из графика для метрик в разрезе глубина/длина.
    Return:
        report в соответствии с концепцией llm_val
    """
    res_dict = {}
    res_dict['result_dataframes'] = [pivot_table]
    res_dict['semaphore'] = 'gray'
    res_dict['result_plots'] = figs
    return res_dict


def valtest_needle_test(sampler: Sampler,
                        model: Union[MultiThreadGigaModel, Any],
                        evaluator: Union[str, Any]=CHRF(),
                        data_type: str='test',
                        haystack_dir: str=os.environ.get('haystack_for_detect'),
                        tokenizer: Optional[Union[str,
                                                  transformers.PreTrainedTokenizer,
                                                  transformers.PreTrainedTokenizerFast]]=None,
                        context_lengths_min: int=1000,
                        context_lengths_max: int=30000,
                        depth_greed: Union[List, int]=list(range(0, 110, 10)),
                        lens_greed: Union[List, int]=[1000, 4000, 8000, 16000, 24000, 30000],
                        system: Optional[str]=None,
                        print_ongoing_status: bool=True,
                        log_for_each_needle: Optional[Dict[int, List[Dict[str, Any]]]]=None,
                        **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Тест для определения того, насколько хорошо LLM ищет ответ на вопрос в большом контексте (ищет "иголку" в стоге "сена").
    В тесте рассчитывается метрика близости между реальным ответом и предсказанием модели в разрезе длин передаваемых контекстов и глубины, на которой расположена иголка относительно 100% контекста.
    В тесте используются данные типа LongAnswer. Необходимо отсутствие Nan-ов в колонках true_answer и question.
    Можно использовать отдельный системный промпт для этого теста. Его можно передать самому или использовать предложенный в ноутбуке 5.2.
    Parameters:
        sampler: Объект класса Sampler
        model: Валидируемая модель. Должна иметь метод predict, который, получив на вход промпт в формате списка из ролей и текста (List[Dict]), возвращает ответ модели (как в MultiThreadGigaModel).
        evaluator: Оценщик близости ответа и правильного ответа. Рекомендуем использовать метрику chrF. Можно перейти к ExactMatch, установив evaluator = 'EM', или прописать свой evaluator - класс с методом evaluate_response, принимающим response и reference на вход.
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е.
        haystack_dir: Относительный путь до директории с текстом для "стога". Внутри директории находятся тексты - файлы в формате txt. 
        tokenizer: Токенизатор LLM (можно передать как загруженный токенизатор, так и путь для него на hf/локальной машине). Должен иметь методы encode(), decode(), get_vocab(). 
            Если его нет, используется opensource токенизатор для GigaChat (https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct).
        context_lengths_min: Минимальное значение длины контекста в сетке. 
        context_lengths_max: Максимальное значение длины контекста в сетке.
        depth_greed: Сетка для глубины иголки в стоге сена. Может быть как списком из значений в рамках [0, 100], так и целым числом - на какое количество равноудаленных значеений поделить [0, 100].
        lens_greed: Сетка для глубины иголки в стоге сена. Может быть как списком из значений в рамках [context_lengths_min, context_lengths_max], так и целым числом - на какое количество равноудаленных значеений поделить [context_lengths_min, context_lengths_max].
        system: Текст системного промпта для LLM с инструкциями для extract спецификации. Приоритетнее системного промпта, установленного в валидируемой модели.
        print_ongoing_status: True, если хочется, чтобы выводились каждая иголка, глубина/длина, ответ LLM и скор - может быть полезно для анализа результатов теста. 
        log_for_each_needle: Cловарь с ключами - номер иголки (0, 1, 2... длина датасета-1) и значениями - списком из словарей, в каждом словаре - результат тестирования для каждого конкретного значения длины/глубины.
    Return:
        Словарь в соответствии с концепцией llm_val.
    """
    if isinstance(depth_greed, int):
        document_depth_percents = None
        n_depth = depth_greed
        document_depth_percent_intervals = depth_greed
    else:
        document_depth_percents = depth_greed
        n_depth = len(depth_greed)
        document_depth_percent_intervals = None

    if isinstance(lens_greed, int):
        context_lengths = None
        n_lens = lens_greed
        context_lengths_num_intervals = lens_greed
    else:
        context_lengths = lens_greed
        n_lens = len(lens_greed)
        context_lengths_num_intervals = None

    if not evaluator:
        evaluator = CHRF()
    elif evaluator == 'EM':
        evaluator = ExactMatch()

    df = getattr(sampler, data_type)['X'].copy()
    df = df.reset_index(drop=True)
    n_needles = df.shape[0]

    list_of_needles = check_scored(log_for_each_needle, 
                                   n_depth=n_depth, 
                                   n_lens=n_lens, 
                                   n_needles=n_needles)
    if list_of_needles:
        if not log_for_each_needle:
            log_for_each_needle = {}

        tokenizer = load_tokenizer(tokenizer)

        for i in tqdm(list_of_needles, desc='Test NiH', total=len(list_of_needles)):
            results_for_needle = run_test_needle(model_to_test=model,
                                                 evaluator=evaluator,
                                                 tokenizer=tokenizer,
                                                 haystack_dir=haystack_dir,
                                                 needle=df.loc[i, 'true_answer'],
                                                 retrieval_question=df.loc[i, 'question'],
                                                 document_depth_percent_intervals=document_depth_percent_intervals,
                                                 document_depth_percents=document_depth_percents,
                                                 context_lengths_num_intervals=context_lengths_num_intervals,
                                                 context_lengths=context_lengths,
                                                 needle_num=i,
                                                 context_lengths_min=context_lengths_min,
                                                 context_lengths_max=context_lengths_max,
                                                 print_ongoing_status=print_ongoing_status,
                                                 system=system)
            log_for_each_needle[i] = results_for_needle

    precomputed = {'log_for_each_needle': log_for_each_needle}
    pivot_table = agg_all_needles(precomputed['log_for_each_needle'])
    figs = plot_heatmap(pivot_table)
    report = report_valtest_needle_test(pivot_table, 
                                        figs)
    return {'report': report, 
            'precomputed': precomputed}


def plot_heatmap(pivot_table: pd.DataFrame) -> Any:
    """
    Строит heatmap по таблице усредненных по всем иголкам метрик для значений глубина / длина контекста.
    Parameters:
        pivot_table: Таблица усредненных по всем иголкам метрик для значений глубина / длина контекста
    Return:
        Heatmap усредненных по всем иголкам метрик для значений глубина / длина контекста
    """
    vmin = pivot_table.min().min()
    vmax = pivot_table.max().max()
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#21BA72'])
    fig = plt.figure(figsize=(17.5, 8)) 
    sns.heatmap(pivot_table,
                vmin=vmin,
                vmax=vmax,
                fmt='.1f',
                cmap=cmap,
                cbar_kws={'label': 'Score'},annot=True,)
    plt.xlabel('Token Limit') 
    plt.ylabel('Depth Percent')
    plt.xticks(rotation=45)  
    plt.yticks(rotation=0)  
    plt.tight_layout()
    return [fig]


def run_test_needle(model_to_test: Any,
                    tokenizer: Any,
                    evaluator: Any,
                    needle: str,
                    retrieval_question: str,
                    haystack_dir: str,
                    needle_num: int=1,
                    context_lengths_min: int=1000,
                    context_lengths_max: int=32000,
                    context_lengths_num_intervals: int=10,
                    context_lengths: Optional[List[int]]=None,
                    document_depth_percent_intervals: int=10,
                    document_depth_percents: Optional[List[int]]=None,
                    system: Optional[str]=None,
                    print_ongoing_status: bool=True) -> List[Dict[str, Union[int, float, str]]]:
    """
    Parameters:
        model_to_test: Валидируемая модель. Должна иметь метод predict, который, получив на вход промпт в формате списка из ролей и текста (List[Dict]), возвращает ответ модели (как в MultiThreadGigaModel).
        tokenizer: Токенизатор LLM (можно передать как загруженный токенизатор, так и путь для него на hf/локальной машине). Должен иметь методы encode(), decode(), get_vocab(). 
            Если его нет, используется opensource токенизатор для GigaChat (https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct).
        evaluator: Оценщик близости ответа и правильного ответа. Рекомендуем использовать метрику chrF. Можно перейти к ExactMatch, установив evaluator = 'EM' или прописать свой evaluator - класс с методом evaluate_response, принимающим response и reference на вход.
        needle: Правильный ответ на вопрос.
        retrieval_question: Сам вопрос из тестируемого сета.
        haystack_dir: Относительный путь до директории с текстом для "стога".
        needle_num: Номер тестируемой иголки (номер вопроса)
        context_lengths_min: Минимальное значение длины контекста в сетке. 
        context_lengths_max: Максимальное значение длины контекста в сетке. 
        context_lengths_num_intervals: Количество интервалов для сетки длин.
        context_lengths: Список значений длин контекстов.
        document_depth_percent_intervals: Количество интервалов для сетки глубин. 
        document_depth_percents: Список значений для процентов глубины. 
        system: Текст системного промпта для LLM с инструкциями для extract спецификации. Приоритетнее системного промпта, установленного в валидируемой модели.
        print_ongoing_status: Если хочется, чтобы печаталась каждая иголка, глубина/длина, ответ LLM и скор.
    Return:
        testing_results: Результат теста для каждого набора глубина/длина.
    """
     
    testing_results = []
    if context_lengths is None:
        context_lengths = np.round(np.linspace(context_lengths_min,
                                               context_lengths_max,
                                               num=context_lengths_num_intervals,
                                               endpoint=True)).astype(int)
    if document_depth_percents is None:
        document_depth_percents = np.round(np.linspace(0, 
                                                       100, 
                                                       num=document_depth_percent_intervals, 
                                                       endpoint=True)).astype(int)
    if not system:
        system = model_to_test.sys_promt
    context_max = read_context_files(haystack_dir, 
                                     context_lengths, 
                                     tokenizer)
    if print_ongoing_status:
        print('\n')
        print('Starting Needle In A Haystack Testing...')
        print(f'- Context Lengths: {len(context_lengths)}, Min: {min(context_lengths)}, Max: {max(context_lengths)}')
        print(f'- Document Depths: {len(document_depth_percents)}, Min: {min(document_depth_percents)}%, Max: {max(document_depth_percents)}%')
        print(f'- Needle: {needle.strip()}')
        print('\n\n')

    for context_length in context_lengths:
        for depth_percent in document_depth_percents: 
            context = generate_context(needle, 
                                       context_max, 
                                       depth_percent, 
                                       context_length, 
                                       tokenizer)
            chat_history = create_prompt(context, 
                                         system)

            test_start_time = time.time()
            response = model_to_test.predict([retrieval_question], [chat_history])[0]
            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time
            
            score = evaluator.evaluate_response(response, needle)
            results = {'context_length': int(context_length),
                       'depth_percent': float(depth_percent),
                       'version': needle_num,
                       'needle': needle,
                       'model_response': response,
                       'score': score,
                       'test_duration_seconds': test_elapsed_time,
                       'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'),}
            testing_results.append(results)

            if print_ongoing_status:
                print(f'-- Test Summary -- ')
                print(f'Duration: {test_elapsed_time:.1f} seconds')
                print(f'Context: {context_length} tokens')
                print(f'Depth: {depth_percent}%')
                print(f'Score: {score}')
                print(f'response: {response}\n')
    return testing_results


def agg_all_needles(log_for_each_needle: Dict[int, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Агрегируем метрики по значениям глубины/длины.
    Parameters:
       log_for_each_needle: Cловарь с ключами - номер иголки (0, 1, 2... длина датасета-1) и значениями - списком из словарей, в каждом словаре - результат тестирования для каждого конкретного значения длины/глубины.
    Return:
       Средние метрики для каждой иголки в разрезе глубина/контекст
    """
    df_all = pd.DataFrame()
    for needle in log_for_each_needle:
        data = []
        for iteration in log_for_each_needle[needle]:
            document_depth = iteration.get('depth_percent', None)
            context_length = iteration.get('context_length', None)
            score = iteration.get('score', None)
            data.append({'Document Depth': document_depth,
                         'Context Length': context_length,
                         'Score': score})
        df = pd.DataFrame(data)
        df_all = pd.concat([df_all, df], axis=0).reset_index(drop=True)
    pivot_table = pd.pivot_table(df_all,
                                 values='Score',
                                 index=['Document Depth', 'Context Length'],
                                 aggfunc='mean').reset_index()  
    pivot_table = pivot_table.pivot(index='Document Depth', 
                                    columns='Context Length',
                                    values='Score')  
    return pivot_table


def check_scored(log_for_each_needle: Optional[Dict[int, List[Dict[str, Any]]]],
                 n_depth: int=10,
                 n_lens: int=10,
                 n_needles: int=10) -> List[int]:
    """
    Функция проверяет логи для иголок: проверяет, сколько иголок уже было полностью проскорено и возвращает номера иголок, для которых это не так.
    Parameters:
        log_for_each_needle: Cловарь с ключами - номер иголки (0, 1, 2... длина датасета-1) и значениями - списком из словарей, в каждом словаре - результат тестирования для каждого конкретного значения длины/глубины.
        n_depth: Количество интервалов глубины
        n_lens: Количество интервалов в длину
        n_needles: Количество иголок
    Return:
        list_of_needles: Список номеров иголок, которые нужно проскорить.
    """
    list_of_needles = list(range(n_needles))
    if log_for_each_needle:
        for needle in log_for_each_needle:
            if len(log_for_each_needle[needle]) == n_depth * n_lens:
                list_of_needles.remove(needle)
    return list_of_needles


def load_tokenizer(tokenizer: Optional[Union[str,
                                             transformers.PreTrainedTokenizer,
                                             transformers.PreTrainedTokenizerFast]]=None) -> Any:
    """
    Вспомогательный метод для инициализации токенизатора.
    Parameters:
        tokenizer: либо путь до токенизатора на локальной машине / имя на hf строкой, либо уже загруженный токенизатор, который должен иметь методы encode(), decode(), get_vocab().
    Return:
        tokenizer: загруженный токенизатор.
    """
    if tokenizer:
        if isinstance(tokenizer, str):
            tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer,
                                                                   revision='main',
                                                                   trust_remote_code=True,
                                                                   use_fast=True)
        else:
            tokenizer = tokenizer
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(os.environ.get("gigachat_tokenizer", None),
                                                               trust_remote_code=True)
    return tokenizer


def create_prompt(context: str, 
                  system: Optional[str] = None) -> List[str]:
    """
    Используя контекст (иголка + сено) и системный промпт, создает промпт для модели.
    Parameters:
        context: Иголка + сено.
        system: Текст системного промпта модели.
    Return:
        chat_history: Запрос в формате диалога.
    """
    chat_history = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': context}]
    return chat_history


def read_context_files(haystack_dir: str, 
                       context_lengths: List[int], 
                       tokenizer: Any) -> str:
    """
    Читаем контекст из директории для сена.
    Parameters:
        haystack_dir: Относительный путь до директории с текстом для "стога". Внутри директории находятся тексты - файлы в формате txt.
        context_lengths: Список значений длин контекстов.
        tokenizer: Токенизатор LLM. Должен иметь метод encode().
    Return:
        context: Текст "стога", обрезанный до максимального значения длины.
    """
    context = ''
    max_context_length = max(context_lengths)

    while len(tokenizer.encode(context)) < max_context_length:
        for file in glob.glob(os.path.join(haystack_dir, '*.txt')):
            with open(file, 'r') as f:
                context += f.read()
    return context


def generate_context(needle: str, 
                     context: str, 
                     depth_percent: int, 
                     context_length: int, 
                     tokenizer: Any) -> str:
    """
    Генерируем контекст для заданной длины и глубины. Обрезаем сено по длине и вставляем иголку в заданную глубину.
    Parameters:
        needle: Правильный ответ на вопрос.
        context: Текст "стога".
        depth_percent: Заданный уровень глубины, на который вставим иголку в контекст.
        context_length: Заданный уровень длины, до которого обрезаем контекст.
        tokenizer: Токенизатор LLM. Должен иметь методы encode(), decode(), get_vocab().
    Return:
        context: Текст "стога" и вставленная иголка.
    """
    tokens = tokenizer.encode(context)
    if len(tokens) > context_length:
        context = tokenizer.decode(tokens[:context_length])
    
    tokens_needle = tokenizer.encode(needle)
    tokens_context = tokenizer.encode(context)
    context_length -= 200
    period_tokens = [v
                     for k, v in tokenizer.get_vocab().items()
                     if ('.' in k) or ('?' in k) or ('!' in k) or ('\n' in k)]

    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[: context_length - len(tokens_needle)]

    if depth_percent == 100:
        tokens_new_context = tokens_context + tokens_needle
    else:
        insertion_point = int(len(tokens_context) * (depth_percent / 100))
        tokens_new_context = tokens_context[:insertion_point]

        while (tokens_new_context and tokens_new_context[-1] not in period_tokens):
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]
        tokens_new_context += tokens_needle + tokens_context[insertion_point:]

    new_context = tokenizer.decode(tokens_new_context)
    return new_context