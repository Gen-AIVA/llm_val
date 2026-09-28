import numpy as np
import typing as tp
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.autometrics import metrics2func

def report_valtest_autometrics_correlation(autometrics_correlation_df: pd.DataFrame,
                                           data_type: str='test',
                                           corr_threshold: float=0.7) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        autometrics_correlation: pd.DataFrame с корреляциями между автометриками 
            и агрегированными оценками асессоров по разным критериям
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        corr_threshold: Порог по корреляции для метрик. Если значение корреляции с разметкой асессоров 
            выше данного порога, то метрика считается коррелирующей 
    Return:
        report в соответствии с концепцией llm_val
    '''
    result_dataframes = []
    for column_name, column in autometrics_correlation_df.items():
        res_table = pd.DataFrame([column, column.map(
            lambda x: "Коррелирующая" if x is not None and float(x) > corr_threshold else "Не коррелирующая")], 
            index=[f'Значение корреляции автометрики с критерием \'{column_name}\' на {data_type}',
                   'Результат теста']).T
        result_dataframes.append(res_table)
    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_plots'] = []
    res_dict['result_dataframes'] = result_dataframes
    return res_dict


def valtest_autometrics_correlation(sampler: Sampler,
                                    metrics_to_calc: tp.List[str]=['meteor', 'bleu', 'rouge', 'cosine_dist'],
                                    data_type: str='test',
                                    corr_threshold: float=0.7,
                                    p_value_threshold: float=0.05,
                                    embedder: tp.Optional[tp.Any]=None,
                                    judge_model: tp.Optional[tp.Any]=None,
                                    corr_results: tp.Optional[pd.DataFrame]=None,
                                    **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест подсчета корреляции между метриками и метками асессоров
    Parameters:
        sampler: Объект класса Sampler с данными, в которых содержится ответ модели, ground truth
            и агрегированные метки асессоров
        metrics_to_calc: Набор интересующих метрик.
            Поддерживаемые метрики:
            1. meteor
            2. bleu
            3. rouge
            4. cosine_dist - нужно передать свой embedder, по умолчанию используются TfidfVectorizer
            5. llm_as_judge - нужно передать judge_model с методом predict
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        corr_threshold: Порог по корреляции для метрик. Если значение корреляции с разметкой асессоров 
            выше данного порога, то метрика считается коррелирующей 
        p_value_threshold: Порог значимости корреляции
        embedder: Объект или функция для получения эмбеддингов (должен иметь метод fit_transform
            или __call__, возвращающий матрицу эмбеддингов). Если None, используется TfidfVectorizer
        judge_model: Модель (объект с методом predict) для оценки качества ответа. Требования описаны в ноутбуке 5.2 в описании теста
        corr_results: Опциональный pd.DataFrame со значениями корреляций.
            Если не None, то подсчет корреляций не производится, и на основе переданного аргумента выставляется светофор
    Return: 
        Словарь в соответствии с концепцией llm_val
    '''

    if corr_results is None:
        data = getattr(sampler, data_type)
        metrics = get_metrics(data['X'], 
                              metrics_to_calc, 
                              embedder=embedder, 
                              judge_model=judge_model)
        assessors_labels = data['y']
        corr_results = pd.DataFrame({criterion: [spearmanr(metrics[mt], assessors_labels[criterion], nan_policy='omit')
                                                 for mt in metrics.columns]
                                                 for criterion in assessors_labels.columns},
                                                 index = metrics.columns)
        corr_results = corr_results.applymap(lambda x: x[0] if x[1] < p_value_threshold else None)
    report = report_valtest_autometrics_correlation(corr_results.copy(),
                                                    data_type=data_type,
                                                    corr_threshold=corr_threshold)
    precomputed = {'corr_values': corr_results}
    return {'report': report,
            'precomputed': precomputed}

def cosine_distance_score(references: tp.Iterable[str],
                          predictions: str,
                          embedder: tp.Any,
                          **kwargs) -> float:
    '''
    Функция для подсчета косинусного расстояния при наличии нескольких референсных ответов.
    Parameters:
        references: Набор корректных ответов
        predictions: Набор ответов модели
        embedder: Объект или функция для получения эмбеддингов (должен иметь метод fit_transform
            или __call__, возвращающий матрицу эмбеддингов)
    Return:
        Максимальное значение косинусной близости (1 - расстояние)
            при попарном сравнении референсов и ответа модели
    '''
    tokenizer = word_tokenize    
    references = [' '.join(tokenizer(r)) for r in references]
    predictions = ' '.join(tokenizer(predictions))

    if hasattr(embedder, 'fit_transform'):
        vectors = embedder.fit_transform(references + [predictions])
        vectors = vectors.toarray() if hasattr(vectors, 'toarray') else vectors
    else:
        vectors = embedder(references + [predictions])

    pred_vec = vectors[-1].reshape(1, -1)
    ref_vecs = vectors[:-1]
    
    sims = cosine_similarity(pred_vec, ref_vecs).flatten()
    
    return float(np.max(sims))

def llm_as_judge_score(references: tp.Iterable[str],
                       prediction: str,
                       judge_model: tp.Any,
                       **kwargs) -> float:
    """
    Функция для оценки предсказания модели с помощью LLM.

    Parameters:
        references: Список эталонных ответов
        prediction: Предсказанный ответ
        judge_model: Модель (объект с методом predict) для оценки качества ответа. Требования описаны в ноутбуке 5.2 в описании теста
        **kwargs: дополнительные параметры, которые модель может использовать
    Return:
        float: оценка моделью
    """
    
    response = judge_model.predict(references=references, 
                                   prediction=prediction, 
                                   **kwargs)

    return response

def get_metrics(data: pd.DataFrame,
                metrics_to_calc: tp.List[str],
                embedder: tp.Optional[tp.Any]=None,
                judge_model: tp.Optional[tp.Any]=None) -> pd.DataFrame:
    '''
    Функция для посчета метрик качества. 
    Parameters:
        data: pd.DataFrame со столбцами model_answer (ответ модели) и ground truth (true_answer)
        metrics_to_calc: Набор интересующих метрик.
            Поддерживаемые метрики:
            1. meteor
            2. bleu
            3. rouge
            4. cosine_dist - нужно передать свой embedder, по умолчанию используются TfidfVectorizer
            5. llm_as_judge - нужно передать judge_model с методом predict 
        embedder: Объект или функция для получения эмбеддингов (должен иметь метод fit_transform
            или __call__, возвращающий матрицу эмбеддингов). Если None, используется TfidfVectorizer
        judge_model: Модель (объект с методом predict) для оценки качества ответа. Требования описаны в ноутбуке 5.2 в описании теста
    Return: 
        pd.DataFrame с подсчитанными метриками для каждого запроса из sampler'а
    '''
    tqdm.pandas()
    results = pd.DataFrame()
    for metric in metrics_to_calc:
        if metric in metrics2func:
            results[metric] = data.apply(lambda x: metrics2func[metric](x['true_answer'], 
                                                                                 x['model_answer']), 
                                                                                 axis=1)
        elif metric == 'cosine_dist':
            if embedder is None:
                embedder = TfidfVectorizer()
            results['cosine_dist'] = data.progress_apply(lambda x: cosine_distance_score(x['true_answer'], 
                                                                                         x['model_answer'], 
                                                                                         embedder=embedder), 
                                                                                         axis=1)

        elif metric == 'llm_as_judge':
            if judge_model != None:
                results['llm_as_judge'] = data.progress_apply(lambda x: llm_as_judge_score(x['true_answer'], 
                                                                                           x['model_answer'], 
                                                                                           judge_model=judge_model), 
                                                                                           axis=1)
            else:
                print('Не передан аргумент judge_model')
        
        else:
            print(f'Метрика {metric} недоступна!')
        
    return results