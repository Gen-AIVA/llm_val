import numpy as np
import pandas as pd
import typing as tp
from tqdm import tqdm
from copy import deepcopy

from llm_val.sampler.Sampler import Sampler
from llm_val.ragexecutor.RAGExecutor import RAGExecutor
from llm_val.utils.metrics_rag import cosine_distance, eval_function


def report_valtest_rag_quality_by_k_passages(stats_df: pd.DataFrame,
                                             figure: tp.Any) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        stats_df: Таблица результатов теста в зависимости от количества пассажей, передаваемых в промпт
        figure: График зависимости метрики качества от числа пассажей
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_plots'] = [figure]
    res_dict['result_dataframes'] = [stats_df]
    return res_dict


def valtest_rag_quality_by_k_passages(sampler: Sampler,
                                      executor: RAGExecutor,
                                      data_type: str='test',
                                      test_by: str='true_answer',
                                      use_llm: bool=True,
                                      eval_metric: str='MeanCosDist',
                                      cos_dist_threshold: float=0.25,
                                      **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест на анализ влияния количества пассажей, передаваемых в промпт, на работу RAG-системы
    Parameters:
        sampler: объект класса RAGSampler
        executor: объект класса RAGExecutor
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        test_by: Название сущности, по которой будет проводиться оценка близости top-K релевантных пассажей
            Hint: предполагается только значение 'true_answer' либо аналогичное
        use_llm: Использовать ли LLM в расчетах
        eval_metric: Название метрики, согласно которой будет оцениваться качество RAG-системы.
            Hint: На данный момент реализованы метрики MeanCosDist и AccuracyCosDist
        cos_dist_threshold: Порог значения косинусного расстояния, при превышении которого набор релевантных пассажей для query не будет считаться релевантным.
            Hint: Используется только при eval_metric = `AccuracyCosDist`
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    df = deepcopy(getattr(sampler, data_type)['X'])
    params = executor.get_parameters()
    queries = {}
    queries['systems'] = df['system'].values.tolist()
    queries['questions'] = df['question'].values.tolist()
    queries['true_answers'] = df[test_by].values.tolist()
    tqdm.pandas(desc='Calculating true answers embeddings')
    queries['true_answers_embeddings'] = df[test_by].progress_apply(lambda x: executor.embed_text(x)).values.tolist()
    metrics = {}
    for k in tqdm(params['topK_list'], desc='Iterating over top-K passages'):
        metrics[k] = get_stats_by_topk(executor=executor,
                                       queries=queries,
                                       use_llm=use_llm,
                                       topK=k,
                                       eval_metric=eval_metric,
                                       cos_dist_threshold=cos_dist_threshold)
    result_df = pd.DataFrame({'num_passages': list(metrics.keys()),
                              'metrics': list(metrics.values())})
    fig = plot_metrics_dynamics(plot_df=result_df,
                                metrics=eval_metric)
    report = report_valtest_rag_quality_by_k_passages(stats_df=result_df,
                                                      figure=fig)
    precomputed = {'baseline_score': metrics[params['topK']]}
    return {'report': report,
            'precomputed': precomputed}


def get_stats_by_topk(executor: tp.Any,
                      queries: tp.Dict[str, tp.List],
                      use_llm: bool=True,
                      topK: int=4,
                      eval_metric: str='MeanCosDist',
                      cos_dist_threshold: float=0.25) -> float:
    '''
    Функция для расчета baseline метрики качества
    Parameters:
        executor: Объект класса RAGExecutor
        queries: Словарь, включающий в себя списки questions, true_answers и true_answers_embeddings
        use_llm: Флаг, отвечающий за то, используется ли LLM в процессе проведения теста.
            Hint: Если False, то тест проводится с учетом косинусных расстояний между top-K релевантными для вопроса пассажами и
                  тем, что указано в параметре `test_by`. 
                  Если True, то тест проводится с учетом косинусных расстояний между ответом LLM на основе top-K релевантных пассажей и
                  тем, что указано в параметре `test_by`.
        topK: Число релевантных пассажей, на которых будет строиться RAG-система
        eval_metric: Название метрики, согласно которой будет оцениваться качество RAG-системы
        cos_dist_threshold: Порог значения косинусного расстояния, при превышении которого набор релевантных пассажей для query не будет считаться релевантным.
            Hint: Используется только при eval_metric = `AccuracyCosDist`
    Return:
        Значение baseline метрики качества
    '''
    if use_llm:
        questions_answers = executor.get_llm_answers(baseline=True,
                                                     questions=queries['questions'],
                                                     systems=queries['systems'],
                                                     topK=topK)
        questions_answers = get_true_answers_distances_to_llm_answers(queries=queries,
                                                                      questions_answers=questions_answers)
        topK_cosine_distances = [value['cosine_distance'] for key, value in questions_answers.items()]
        eval_metric_value = eval_function[eval_metric](topK_cosine_distances, cos_dist_threshold)
    elif topK != 0:
        passages = executor.get_passages_with_embeddings(baseline=True)
        topK_passages = executor.get_topK_passages(baseline=True,
                                                   questions=queries['questions'],
                                                   topK=topK)
        topK_passages = get_true_answers_distances_to_topK_passages(queries=queries,
                                                                    topK_passages=topK_passages)
        topK_cosine_distances = [value['mean_cosine_distance'] for key, value in topK_passages.items()]
        eval_metric_value = eval_function[eval_metric](topK_cosine_distances, cos_dist_threshold)
    else:
        eval_metric_value = None
    return eval_metric_value


def get_true_answers_distances_to_topK_passages(queries: tp.Dict[str, tp.List[tp.Any]],
                                                topK_passages: tp.Dict[str, tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Функция для получения косинусных расстояний и средних косинусных расстояний между true_answers и top-K релевантными пассажами
    Parameters:
        queries: Словарь, включающий в себя списки questions, true_answers и true_answers_embeddings
        topK_passages: Словарь top-K пассажей с их эмбеддингами
    Return:
        Словарь top-K пассажей с их эмбеддингами, косинусными расстояниями и средними косинусными расстояниями
    '''
    for question, true_answer_embedding in zip(queries['questions'], queries['true_answers_embeddings']):
        topK_passages[question]['cosine_distances'] = [cosine_distance(true_answer_embedding, passage_embedding) for 
                                                       passage_embedding in topK_passages[question]['embeddings']]
        topK_passages[question]['mean_cosine_distance'] = np.mean(topK_passages[question]['cosine_distances'])
    return topK_passages


def get_true_answers_distances_to_llm_answers(queries: tp.Dict[str, tp.List[tp.Any]],
                                              questions_answers: tp.Dict[str, tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Функция для получения косинусных расстояний между true_answers и llm_answers
    Parameters:
        queries: Словарь, включающий в себя списки questions, true_answers и true_answers_embeddings
        questions_answers: Словарь с ответами LLM на вопросы и их эмбеддингами
    Return:
        Словарь llm_answers с их эмбеддингами и косинусными расстояниями до true_answers
    '''
    for question, true_answer_embedding in zip(queries['questions'], queries['true_answers_embeddings']):
        questions_answers[question]['cosine_distance'] = cosine_distance(true_answer_embedding, 
                                                                         questions_answers[question]['llm_answer_embedding'])
    return questions_answers


def plot_metrics_dynamics(plot_df: pd.DataFrame,
                          metrics: str) -> tp.Any:
    '''
    Функция для построения графика-отчета
    Parameters:
        plot_df: Dataframe с числом пассажей и метриками соответственно
        metrics: название метрики
    Return:
        Объект графика
    '''
    return plot_df.plot.bar(x='num_passages',
                            y='metrics',
                            color='forestgreen',
                            title='Влияние количества пассажей на ответы модели',
                            xlabel='Количество пассажей',
                            ylabel=metrics,
                            legend=False)
