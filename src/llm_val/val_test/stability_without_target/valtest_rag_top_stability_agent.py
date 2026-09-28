import random
import numpy as np
import pandas as pd
import typing as tp

from llm_val.sampler.Sampler import Sampler
from llm_val.ragexecutor.RAGExecutor import RAGExecutor
from llm_val.utils.human_augmentations import batch_augmentations
from llm_val.utils.report_helper import semaphore_by_threshold, worst_semaphore
from llm_val.utils.api_augmentations import batch_paraphrase_api, batch_synonym_api, batch_translation_api


def report_valtest_rag_top_stability_agent(iou_symbol_results: tp.Dict[int, float],
                                           iou_word_results: tp.Dict[int, float],
                                           topK: int,
                                           semaphore_threshold: tp.Tuple[float, float] = (0.2, 0.7),
                                           is_info: bool = False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        iou_symbol_results: Словарь со значениями метрики IOU для каждого размера топа на уровне символов
        iou_word_results: Словарь со значениями метрики IOU для каждого размера топа на уровне слов
        topK: Размер топа, для которого определяется цвет светофора
        semaphore_threshold: Кортеж из двух значений (нижний порог, верхний порог) для определения цвета светофора по метрике IOU
        is_info: True, если тест информативный
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    res_table = pd.DataFrame({'Размер топа': list(iou_symbol_results.keys()),
                              'IOU на уровне символов': list(iou_symbol_results.values()),
                              'IOU на уровне слов': list(iou_word_results.values())})

    if is_info:
        color = 'gray'
    else:
        val_sym = iou_symbol_results.get(topK)
        val_wrd = iou_word_results.get(topK)

        semaphores = []
        if val_sym is not None and not (isinstance(val_sym, float) and np.isnan(val_sym)):
            sem_sym = semaphore_by_threshold(value=val_sym,
                                             threshold=(semaphore_threshold[0], semaphore_threshold[1]),
                                             greater_is_better=True,
                                             left_border_is_bigger=False,
                                             right_border_is_bigger=True)
            semaphores.append(sem_sym)

        if val_wrd is not None and not (isinstance(val_wrd, float) and np.isnan(val_wrd)):
            sem_wrd = semaphore_by_threshold(value=val_wrd,
                                             threshold=(semaphore_threshold[0], semaphore_threshold[1]),
                                             greater_is_better=True,
                                             left_border_is_bigger=False,
                                             right_border_is_bigger=True)
            semaphores.append(sem_wrd)

        color = 'gray' if not semaphores else worst_semaphore(semaphores)

    res_dict = {
        'semaphore': color,
        'result_plots': [],
        'result_dataframes': [res_table]
    }
    return res_dict


def valtest_rag_top_stability_agent(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                                    executor: RAGExecutor,
                                    augmentations_list: tp.Optional[tp.List[str]] = ['symbol_swap',
                                                                                     'butter_finger',
                                                                                     'human_level_augmentations',
                                                                                     'paraphrase',
                                                                                     'synonym',
                                                                                     'translation'],
                                    exec_env: tp.Literal['default', 'sberds'] = 'default',
                                    data_type: str = 'test',
                                    semaphore_threshold: tp.Tuple[float, float] = (0.2, 0.7),
                                    p_augmented: float = 0.1,
                                    translation_language: str = 'английский',
                                    word_replace_ratio: float = 0.1,
                                    seed: int = 42,
                                    max_attempts: int = 3,
                                    aug_client: tp.Optional[tp.Any] = None,
                                    aug_model_name: tp.Optional[str] = None,
                                    aug_model_prompts: tp.Optional[tp.Dict[str, str]] = None,
                                    is_info: bool = False,
                                    iou_symbol_results: tp.Optional[tp.Dict[int, float]] = None,
                                    iou_word_results: tp.Optional[tp.Dict[int, float]] = None,
                                    **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на стабильность выдачи RAG в зависимости от аугментаций текстов на уровне символов или слов

    Parameters:
        sampler: Объект класса Sampler
        executor: Объект класса RAGExecutor
        augmentations_list: Список выполняемых аугментаций.
            Поддерживаемые значения: 'symbol_swap', 'butter_finger', 'human_level_augmentations' (символьные),
            'paraphrase', 'synonym', 'translation' (словные).
        exec_env: Среда выполнения ('default' или 'sberds')
        data_type: Тип данных, на которых производится расчет метрики (название атрибута в sampler)
        semaphore_threshold: Красный и желтый порог светофора
        p_augmented: Вероятность аугментации символа каждой аугментацией
        translation_language: Язык для аугментации translation (указанный на русском языке)
        word_replace_ratio: Доля слов для замены при синонимической аугментации
        seed: Начальное значение датчика случайных чисел
        max_attempts: Максимальное количество попыток аугментации для текстов, которые не удалось изменить
        aug_client: OpenAI-совместимый клиент для генерации аугментаций
        aug_model_name: Название модели для генерации аугментаций
        aug_model_prompts: Опциональный словарь с системными промптами для аугментаций.
                           Допустимые ключи: 'paraphrase', 'synonym', 'translation_fwd', 'translation_bwd'.
                           Если ключ отсутствует или словарь None, используются промпты по умолчанию.
        is_info: True, если тест информативный
        iou_symbol_results: Опциональный словарь с предпосчитанными результатами теста на уровне символов.
            Если не None, то тест не запускается, и на основе переданного аргумента выставляется светофор.
        iou_word_results: Опциональный словарь с предпосчитанными результатами теста на уровне слов.
            Если не None, то тест не запускается, и на основе переданного аргумента выставляется светофор.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    aug_col_prefix = 'aug_valtest_rag_top_stability_agent_'
    symbol_augmentations = {'symbol_swap', 'butter_finger', 'human_level_augmentations'}
    word_augmentations = {'paraphrase', 'synonym', 'translation'}

    type_config = {
        'symbol': {
            'aug_list': [a for a in augmentations_list if a in symbol_augmentations],
            'precomputed': iou_symbol_results,
        },
        'word': {
            'aug_list': [a for a in augmentations_list if a in word_augmentations],
            'precomputed': iou_word_results,
        }
    }

    params = executor.get_parameters()
    topK = params['topK']

    need_compute = {}
    for aug_type, cfg in type_config.items():
        need_compute[aug_type] = (cfg['precomputed'] is None) and bool(cfg['aug_list'])

    if any(need_compute.values()):
        data = sampler[data_type] if exec_env == 'sberds' else getattr(sampler, data_type)
        queries = data['X']['input_query'].values.tolist()
        original_tops_max = executor.get_topK_passages(questions=queries, topK=topK)
        original_tops = {
            k: [res['documents'][:k] for res in original_tops_max.values()]
            for k in range(1, topK + 1)
        }

    results = {}
    for aug_type, cfg in type_config.items():
        if not need_compute[aug_type]:
            results[aug_type] = cfg['precomputed']
            continue

        augmented_data = get_augmented_data(sampler=sampler,
                                            data_type=data_type,
                                            type_augmentation=aug_type,
                                            augmentations_list=cfg['aug_list'],
                                            p_augmented=p_augmented,
                                            seed=seed,
                                            aug_client=aug_client,
                                            aug_model_name=aug_model_name,
                                            translation_language=translation_language,
                                            word_replace_ratio=word_replace_ratio,
                                            max_attempts=max_attempts,
                                            aug_model_prompts=aug_model_prompts,
                                            exec_env=exec_env)

        augmented_tops_dict = {}
        for aug_name in augmented_data.columns:
            augmented_queries = augmented_data[aug_name]
            max_topK_results = executor.get_topK_passages(questions=augmented_queries, topK=topK)
            augmented_tops_dict[aug_name] = max_topK_results

        iou_dict = {k: np.nan for k in range(1, topK + 1)}
        for k in range(1, topK + 1):
            tops_list = []
            for aug_name in cfg['aug_list']:
                max_topK_results = augmented_tops_dict[aug_col_prefix + aug_name]
                current_k_results = [res['documents'][:k] for res in max_topK_results.values()]
                tops_list.append(current_k_results)
            scores = [evaluate_tops(original_tops[k], aug_tops) for aug_tops in tops_list]
            iou_dict[k] = np.mean(scores)

        results[aug_type] = iou_dict

    iou_symbol_results = results.get('symbol') or {k: np.nan for k in range(1, topK + 1)}
    iou_word_results = results.get('word') or {k: np.nan for k in range(1, topK + 1)}

    precomputed = {
        'iou_symbol_results': iou_symbol_results,
        'iou_word_results': iou_word_results
    }

    report = report_valtest_rag_top_stability_agent(iou_symbol_results, iou_word_results,
                                                    topK, semaphore_threshold, is_info)
    return {'report': report, 'precomputed': precomputed}


def get_augmented_data(sampler: tp.Union[Sampler, tp.Dict[str, tp.Any]],
                       data_type: str='test',
                       type_augmentation: tp.Literal['symbol', 'word']='symbol',
                       augmentations_list: tp.Optional[tp.List[str]]=None,
                       p_augmented: float=0.1,
                       seed: int=42,
                       aug_client: tp.Optional[tp.Any]=None,
                       aug_model_name: tp.Optional[str]=None,
                       translation_language: str='английский',
                       word_replace_ratio: float=0.1,
                       max_attempts: int=3,
                       aug_model_prompts: tp.Optional[tp.Dict[str, str]]=None,
                       exec_env: tp.Literal['default', 'sberds']='default') -> pd.DataFrame:
    '''
    Проводит символьные аугментации запросов всеми методами из списка для проведения реинференса
    Parameters:
        sampler: Объект класса Sampler (exec_env='default') или словарь с данными (exec_env='sberds')
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        type_augmentation: Тип применямых аугментаций (на уровне символов или на уровне слов)
        augmentations_list: Список выполняемых аугментаций
        p_augmented: Вероятность аугментации символа каждой аугментацией
        seed: Начальное значение датчика случайных чисел
        aug_client: OpenAI-совместимый клиент для генерации аугментаций. Опционален, если aug-колонки
                    уже есть в данных.
        aug_model_name: Название модели для генерации аугментаций
        translation_language: Язык перевода для аугментации translation
        word_replace_ratio: Доля слов для замены при синонимической аугментации
        max_attempts: Максимальное количество попыток аугментации для текстов, которые не удалось изменить
        aug_model_prompts: Опциональный словарь с системными промптами для аугментаций.
        exec_env: Среда выполнения ('default' или 'sberds')
    Return:
        DataFrame с аугментированными запросами в колонках aug_valtest_metric_aug_stability_agent_<aug>
    '''
    aug_col_prefix = 'aug_valtest_rag_top_stability_agent_'
    data = sampler[data_type]['X'] if exec_env == 'sberds' else getattr(sampler, data_type)['X']
    augmentations_list_with_prefix = list(map(lambda aug: aug_col_prefix + aug, augmentations_list))
    aug_df = pd.DataFrame(index=data.index, columns=augmentations_list_with_prefix)
    rng = build_rng(seed)
    
    if type_augmentation == 'symbol':
        for aug_name in augmentations_list:
            col_name = f'{aug_col_prefix}{aug_name}'
            if col_name in data.columns:
                aug_df[col_name] = data[col_name]
                continue
                
            aug_df[col_name] = build_augmented_column(data, 
                                                      aug_name,
                                                      rng, 
                                                      p_augmented)

    else:
        paraphrase_prompt = aug_model_prompts.get('paraphrase') if aug_model_prompts else None
        synonym_prompt = aug_model_prompts.get('synonym') if aug_model_prompts else None
        translation_fwd_prompt = aug_model_prompts.get('translation_fwd') if aug_model_prompts else None
        translation_bwd_prompt = aug_model_prompts.get('translation_bwd') if aug_model_prompts else None
        for aug_name in augmentations_list:
            col_name = f'{aug_col_prefix}{aug_name}'
            if col_name in data.columns:
                aug_df[col_name] = data[col_name]
                continue
            if aug_client is None or aug_model_name is None:
                continue
            original_questions = data['input_query'].tolist()
            if aug_name == 'paraphrase':
                aug_df[col_name] = batch_paraphrase_api(
                    batch=original_questions, aug_client=aug_client, aug_model_name=aug_model_name,
                    system_prompt=paraphrase_prompt, max_attempts=max_attempts
                )
            elif aug_name == 'synonym':
                aug_df[col_name] = batch_synonym_api(
                    batch=original_questions, aug_client=aug_client, aug_model_name=aug_model_name,
                    rng=rng, word_replace_ratio=word_replace_ratio, system_prompt=synonym_prompt,
                    max_attempts=max_attempts
                )
            elif aug_name == 'translation':
                aug_df[col_name] = batch_translation_api(
                    batch=original_questions, aug_client=aug_client, aug_model_name=aug_model_name,
                    target_language=translation_language,
                    system_prompt_fwd=translation_fwd_prompt, system_prompt_bwd=translation_bwd_prompt,
                    max_attempts=max_attempts
                )
            else:
                raise ValueError(f"Неизвестный тип аугментации: {aug_name}")
    return aug_df

    
def build_rng(seed: int) -> tp.Any:
    '''
    Создает генератор случайных чисел и фиксирует датчики случайных чисел
    Parameters:
        seed: Начальное значение датчика случайных чисел
    Return:
        Генератор случайных чисел numpy
    '''
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def build_augmented_column(data: pd.DataFrame,
                           aug_name: str,
                           rng: tp.Any,
                           p_augmented: float) -> tp.List[str]:
    '''
    Формирует значения одной aug-колонки с аугментированными запросами
    Parameters:
        data: Исследуемый набор данных
        aug_name: Название аугментации
        rng: Генератор случайных чисел
        p_augmented: Вероятность аугментации символа
    Return:
        Список аугментированных запросов
    '''
    return apply_augmentation(data['input_query'].to_list(), aug_name, rng, p_augmented)


def apply_augmentation(texts: tp.List[str],
                       aug_name: tp.Optional[str],
                       rng: tp.Any,
                       p_augmented: float) -> tp.List[str]:
    '''
    Применяет к набору текстов выбранную символьную аугментацию
    Parameters:
        texts: Набор текстов (строк)
        aug_name: Название аугментации либо None для исходных текстов
        rng: Генератор случайных чисел
        p_augmented: Вероятность аугментации символа
    Return:
        Аугментированный набор текстов либо исходный набор при aug_name None
    '''
    if aug_name is None:
        augmented_texts = list(texts)
    else:
        char_aug_dict = {'human_level_augmentations': ['use_deletion', 'use_swap', 'use_insert',
                                                       'use_bf', 'use_special_symbol'],
                         'symbol_swap': ['use_swap'],
                         'butter_finger': ['use_bf']}
        aug_params = {param: True for param in char_aug_dict[aug_name]}
        augmented_texts = list(batch_augmentations(batch=list(texts), rng=rng, p=p_augmented, **aug_params))
    return augmented_texts


def iou_strategy(original_top: tp.List, 
                 modified_top: tp.List) -> float:
    '''
    Вычисляет Intersection over Union (IOU) для двух списков пассажей.
    Parameters:
        original_top: Cписок исходных пассажей в топе.
        modified_top: Список измененных пассажей в топе.
    Return:
        Значение IOU по вопросу
    '''
    set_orig = set(original_top)
    set_mod = set(modified_top)
    if not set_orig or not set_mod:
        raise ValueError('Original_top or modified_top are empty. IoU is undefined.')
    intersection = len(set_orig & set_mod)
    union = len(set_orig | set_mod)
    return intersection / union if union != 0 else 0.0


def evaluate_tops(original_tops: tp.List[tp.List], 
                  modified_tops: tp.List[tp.List]) -> float:
    '''
    Оценивает изменения топов.
    Parameters:
        original_tops: Список списков исходных пассажей в топе.
        modified_tops: Список списков измененных пассажей в топе.
    Return:       
        Среднее значение изменений по всем вопросам.
    '''
    total_score = 0
    for original_top, modified_top in zip(original_tops, modified_tops):
        total_score += iou_strategy(original_top, 
                                    modified_top)
    return total_score / len(original_tops)