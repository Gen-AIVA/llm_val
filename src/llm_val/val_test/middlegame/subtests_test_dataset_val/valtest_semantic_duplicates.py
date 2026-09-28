import numpy as np
import pandas as pd
import typing as tp
from tqdm.auto import tqdm as tqdm
from sklearn.metrics import pairwise
from sklearn.preprocessing import normalize

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.retries import call_with_retries
from llm_val.utils.openai_utils import get_openai_embeddings
from llm_val.utils.report_helper import semaphore_by_threshold_without_red

def report_valtest_semantic_duplicates(value: float,
                                       duplicate_pairs: tp.Set[tp.Tuple[int, int]],
                                       threshold: float=0.1,
                                       is_info: bool=False):
    '''
    Создание report'а
    Parameters:
        value: Доля дубликатов в выборках
        duplicate_pairs: Список семантических дубликатов
        threshold: Граница светофора - допустимая доля дубликатов в выборке.
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_dict = {}
    res_dict['result_plots'] = []
    res_dict['semaphore'] = 'gray' if is_info else semaphore_by_threshold_without_red(value, 
                                                                                      threshold=threshold, 
                                                                                      greater_is_better=False, 
                                                                                      border_is_bigger=False)
    test_result_map = {'green': 'Пройдена', 'yellow': 'Не пройдена', 'gray': 'Пройдена'}
    test_result = test_result_map.get(res_dict['semaphore'], 'Не пройдена')
    if duplicate_pairs:
        comment = f'Пары дубликатов: {", ".join(str(t) for t in duplicate_pairs)}'
    else:
        comment = ''

    result_df = pd.DataFrame({
        'Проверка': 'Отсутствие семантических дубликатов среди примеров [вспомогательная проверка]',
        'Доля семантических дубликатов': [value],
        'Результат': [test_result],
        'Комментарий': comment
    })
    res_dict['result_dataframes'] = [result_df]

    return res_dict


def valtest_semantic_duplicates(sampler: Sampler,
                                exact_duplicates: tp.List[tp.Tuple[int, int]],
                                embedding_client: tp.Optional[tp.Any]=None,
                                embedding_model_name: tp.Optional[str]='EmbeddingsGigaR',
                                data_type: str='test',
                                threshold: float=0.1,
                                threshold_duplicates: float=0.95,
                                is_info: bool=False,
                                timeout: float=60.0,
                                retry_delay: float=0.1,
                                max_attempts: int=3,
                                exec_env: str='default',
                                embeddings: tp.Optional[np.ndarray]=None,
                                similarity: tp.Optional[np.ndarray]=None,
                                duplicate_rate: tp.Optional[int]=None,
                                duplicate_pairs: tp.Optional[tp.Set[tp.Tuple[int, int]]]=None,
                                **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест доли семантических дубликатов вопросов в test выборке.
    Parameters:
        sampler: Объект класса Sampler
        exact_dupicates: список кортежей наблюдений-дубликатов
        embedding_client: OpenAI-совместимый клиент с методом embeddings.create. Обязателен при is_long_answer=True
        embedding_model_name: Название эмбеддинг-модели, передаваемое в embeddings.create
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        threshold: Граница светофора - максимально допустимая доля дубликатов в выборке.
        threshold_duplicates: Граница признания семантических дубликатов.
        is_info: True, если тест информативный
        timeout: Максимальное время ожидания одного обращения к модели в секундах
        retry_delay: Базовая задержка между повторными попытками обращения к модели в секундах
        max_attempts: Максимальное количество попыток обращения к модели
        exec_env: Переменная исполняемого окружения ('default' или 'sberds')
        embeddings: опицональный результат вычисления нормализованных эмдеддингов запросов
            Если не None, то вычисление не производится, и на основе переданного аргумента считаются статистики.
        similarity: опицональный результат вычисления попарного косинусного расстояния между запросами
            Если не None, то вычисление не производится, и на основе переданного аргумента считаются статистики.
        duplicate_rate: опицональный результат вычисления доли семантических дубликатов.
            Если не None, то вычисление не производится, и на основе переданного аргумента строится вывод теста.
        duplicate_pairs: опицональный результат вычисления списка пар семантических дубликатов.
            Если не None, то вычисление не производится, и на основе переданного аргумента строится вывод теста.
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    if duplicate_rate is None or duplicate_pairs is None:
        if similarity is None and embeddings is None:
            if exec_env == 'default':
                data = getattr(sampler, data_type)
            elif exec_env == 'sberds':
                data = sampler[data_type]

            is_dialogue = 'dialogue' in data['X'].columns
            text_col = 'dialogue' if is_dialogue else 'input_query'
            questions_text = questions = data['X'].groupby('session_id')[text_col].first().to_list()
            if is_dialogue:
                questions_text = [dialogue_to_text(dialogue) for dialogue in questions]
            
            embeddings = call_with_retries(get_openai_embeddings,
                                           timeout,
                                           retry_delay,
                                           max_attempts,
                                           (embedding_client, questions_text, embedding_model_name))
            embeddings = normalize(embeddings, norm='l2', axis=1)

        exact_indices = set()
        for tpl in exact_duplicates:
            exact_indices.update(sorted(tpl)[1:])
        
        similarity, duplicate_rate, duplicate_pairs, duplicate_indices = calc_duplicate_pairs(embeddings=embeddings,
                                                                                              exact_indices=exact_indices,
                                                                                              threshold_duplicates=threshold_duplicates,
                                                                                              similarity=similarity)
    
    precomputed = {}
    precomputed['embeddings'] = embeddings
    precomputed['similarity'] = similarity
    precomputed['duplicate_rate'] = duplicate_rate
    precomputed['semantic_duplicate_indices'] = duplicate_indices
    precomputed['semantic_duplicate_pairs'] = duplicate_pairs
    value = 0.0
    value = max(value, duplicate_rate)

    report = report_valtest_semantic_duplicates(value=value,
                                                duplicate_pairs=duplicate_pairs,
                                                threshold=threshold,
                                                is_info=is_info)
    return {'report': report,
            'precomputed': precomputed}


def calc_duplicate_pairs(embeddings: tp.Optional[np.ndarray]=None,
                         exact_indices: tp.Set[int]={},
                         threshold_duplicates: float=0.95,
                         similarity: tp.Optional[np.ndarray]=None) -> tp.Tuple[np.ndarray, float, tp.List[tp.Tuple[int, int]]]:
    '''
    Функция создания таблицы семантической схожести (косинусная близость векторов эмбеддингов) между тексами. 
    Parameters:
        embeddings: список эмбеддингов запросов
        exact_indices: множество эклементов эвляющихся точными дубликатами предыдущих
        threshold_duplicates: Граница признания семантических дубликатов
        similarity: опицональный результат вычисления попарного косинусного расстояния между запросами
            Если не None, то вычисление не производится, и на основе переданного аргумента считаются статистики
    Return:
        Таблица расстояний между всеми парами текстов.
        Доля семантических дубликатов.
        Список индексов семантических дубликатов.
    '''
    if similarity is None:
        similarity = pairwise.cosine_similarity(embeddings)

    n = similarity.shape[0]
    n_duplicate_indices = sum([max(similarity[i, 0:i]) > threshold_duplicates for i in range(1, n) if i not in exact_indices])

    duplicate_pairs = set()
    duplicate_indices = set()
    for i in range(1, n):
        if i in exact_indices:
            continue
        local_duplicate_indices = np.argwhere(similarity[i, 0:i] > threshold_duplicates).squeeze(axis=1)
        if local_duplicate_indices.shape[0] == 0:
            continue
        duplicate_indices.add(i)
        duplicate_indices.update(local_duplicate_indices)
        for item in local_duplicate_indices:
            duplicate_pairs.add((item, i))
    duplicate_rate =  n_duplicate_indices / (n - len(exact_indices))

    return similarity, duplicate_rate, duplicate_pairs, duplicate_indices


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