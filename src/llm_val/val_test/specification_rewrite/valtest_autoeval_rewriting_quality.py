import sklearn
import pandas as pd
import typing as tp
from copy import deepcopy
import matplotlib.pyplot as plt
from Levenshtein import distance

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.autometrics import metrics2func

def report_valtest_autoeval_rewriting_quality(semantic_sim_values: tp.List[float],
                                              lexical_diff_values: tp.List[float],
                                              rewrite_quality_values: tp.List[float],
                                              figure: tp.Any) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        semantic_sim_values: Значения близости текстов
        lexical_diff_values: Значения лексической разницы текстов
        rewrite_quality_values: Значения качества перефразирования
        figure: График распределения примеров перефразирования текстов по двум компонентам метрики RewriteQuality
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_table = pd.DataFrame({'Семантическая близость': semantic_sim_values,
                              'Лексическая разница': lexical_diff_values,
                              'Качество перефразирования': rewrite_quality_values})
    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_plots'] = [figure]
    res_dict['result_dataframes'] = [res_table]
    return res_dict


def valtest_autoeval_rewriting_quality(sampler: Sampler,
                                       embedding_model: tp.Optional[tp.Any]=None,
                                       semantic_sim_func: str='cosine_similarity',
                                       lexical_diff_func: str='levenshtein_distance',
                                       data_type: str='test',
                                       **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на расчет оценки качества перефразирования исходного текста (rewrite) с помощью автоматической метрики (информативный тест)
    Parameters:
        sampler: Объект класса Sampler
        embedding_model: Модель для получения эмбеддингов, имеет метод get_embedding(список вопросов)
        semantic_sim_func: Функция для расчета близости между текстами. Для расчета доступны метрики cosine_similarity, bleu, rouge и meteor
        lexical_diff_func: Функция для расчета лексической разницы между текстами. Для расчета доступны метрики levenshtein_distance, bleu, rouge и meteor
        data_type: Тип данных, на котором сравниваются метрики
    Return:
        Словарь в соответствии с концепцией llm_val
    '''
    data = getattr(sampler, data_type)['X']
    if semantic_sim_func == 'cosine_similarity':
        question_embeddings = embedding_model.get_embedding(data['question'].tolist())
        rewrited_embeddings = embedding_model.get_embedding(data['model_answer'].tolist())
        semantic_sim_values = [cosine_similarity(question_emb['embedding'], 
                                                 rewrited_emb['embedding']) 
                               for question_emb, rewrited_emb in zip(question_embeddings, rewrited_embeddings)]
    else:
        semantic_sim_values = data.apply(lambda x: metrics2func[semantic_sim_func]([x['question']],
                                                                                    x['model_answer']), axis=1)
    if lexical_diff_func == 'levenshtein_distance':
        lexical_diff_values = levenshtein_distance(data['question'].tolist(), data['model_answer'].tolist())
    else:
        lexical_diff_values = data.apply(lambda x: 1 - metrics2func[lexical_diff_func]([x['question']],
                                                                                        x['model_answer']), axis=1)
    rewrite_quality_values = [sim_value * lex_diff_value for sim_value, lex_diff_value in zip(semantic_sim_values, lexical_diff_values)]
    precomputed = {semantic_sim_func: deepcopy(semantic_sim_values),
                   lexical_diff_func: deepcopy(lexical_diff_values),
                   'rewrite_quality_values': rewrite_quality_values}
    figure = plot_semantic_similarity_lexical_diff(semantic_sim_values, 
                                                   lexical_diff_values,
                                                   rewrite_quality_values,
                                                   semantic_sim_func, 
                                                   lexical_diff_func)
    report = report_valtest_autoeval_rewriting_quality(semantic_sim_values,
                                                       lexical_diff_values,
                                                       rewrite_quality_values,
                                                       figure)
    return {'report': report,
            'precomputed': precomputed}


def cosine_similarity(first_emb: tp.List[float], 
                      second_emb: tp.List[float]) -> float:
    '''
    Расчёт косинусной близости между двумя эмбеддингами.
    Parameters:
        first_emb: первый эмбеддинг, представляет из себя список чисел формата float
        second_emb: второй эмбеддинг, представляет из себя список чисел формата float
    Return:
        Косинусная близость между векторами a и b
    '''
    cosine_similarity = sklearn.metrics.pairwise.cosine_similarity([first_emb], [second_emb])[0][0]
    return cosine_similarity


def levenshtein_distance(reference: tp.List[str], 
                         hypothesis: tp.List[str]) -> tp.List[float]:
    '''
    Вычисляет расстояние Левенштейна между списками текстов.
    Parameters:
        reference: Список текстов, исправленных моделью
        hypothesis: Список исходных текстов
    Return:
        Список с расстояниями Левенштейна для каждой пары текстов
    '''
    distances = [distance(ref, hyp) for ref, hyp in zip(reference, hypothesis)]
    distances = [d/max(len(ref), len(hyp)) if max(len(ref), len(hyp)) > 0 else 0 for d, ref, hyp in zip(distances, reference, hypothesis)]
    return distances


def plot_semantic_similarity_lexical_diff(semantic_sim_values: tp.List[float],
                                          lexical_diff_values: tp.List[float],
                                          rewrite_quality_values: tp.List[float],
                                          semantic_sim_func: str='cosine_similarity',
                                          lexical_diff_func: str='levenshtein_distance') -> tp.Any:
    '''
    Строит график распределения примеров перефразирования текстов по двум компонентам метрики RewriteQuality
    Parameters:
        semantic_sim_values: Значения близости текстов
        lexical_diff_values: Значения лексической разницы текстов
        lexical_diff_func: Функция для расчета лексической разницы между текстами. Для расчета доступны метрики levenshtein_distance, bleu, rouge и meteor
    Return:
        График распределения примеров перефразирования текстов по двум компонентам метрики RewriteQuality 
    '''
    fig = plt.figure(figsize=(9, 7))
    plt.scatter(semantic_sim_values, lexical_diff_values, s=[size * 50 for size in rewrite_quality_values])
    plt.xlabel(f'Семантическая схожесть текстов ({semantic_sim_func})')
    plt.ylabel(f'Лексическая разница между текстами ({lexical_diff_func})')
    plt.title('Распределение примеров перефразирования текстов по двум компонентам метрики RewriteQuality')
    return fig
