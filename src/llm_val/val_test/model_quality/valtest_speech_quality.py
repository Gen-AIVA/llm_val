import numpy as np
import typing as tp
import pandas as pd

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.speech_quality.utils import score_data_by_model
from llm_val.utils.speech_quality.spelling import spelling_score_api
from llm_val.utils.speech_quality.literacy import plot_literacy_confusion_matrix
from llm_val.utils.speech_quality.readability import corpus_readability_COLE_index, plot_readability_confusion_matrix


def report_valtest_speech_quality(metrics_results: pd.DataFrame,
                                  tests_bool_list: tp.List[bool],
                                  data_type: str='test',
                                  figs: tp.Optional[tp.List[tp.Any]]=None) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        metrics_results: Словарь с подсчитанными метриками
        tests_bool_list: Список булевских флагов, указывающих на то, какие тесты на качество речи проводились
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        figs: Список графиков
    Return:
        report в соответствии с концепцией llm_val
    '''
    filtered = [filter(lambda x: tests_bool_list[x[0]], enumerate(column))
                for column in [['Лингвистическая грамотность', 'Читаемость', 'Орфография'],
                               ['Linguistic Acceptability score', 'COLE', 'WER'],
                               ['literacy_score', 'readability_score', 'spelling_score']]]
    columns = [list(map(lambda x: x[1], f)) for f in filtered]
    result = pd.DataFrame({'Проверка': columns[0],
                           'Метрика': columns[1],
                           f'Значение метрики на {data_type}': [metrics_results[name] for name in columns[2]]})
    res_dict = {}
    res_dict['semaphore'] = 'gray'
    res_dict['result_plots'] = figs
    res_dict['result_dataframes'] = [result]
    return res_dict


def valtest_speech_quality(sampler: Sampler,
                           data_type: str='test',
                           test_literacy: bool=True,
                           test_readability: bool=True,
                           test_spelling: bool=False,
                           spelling_model: tp.Optional[tp.Any]=None,
                           spelling_model_cache: tp.Optional[tp.Dict[str, str]]={},
                           literacy_model: tp.Optional[tp.Any]=None,
                           literacy_tokenizer: tp.Optional[tp.Any]=None,
                           literacy_model_config: tp.Optional[tp.Dict[str, tp.Any]]=None,
                           literacy_tokenizer_config: tp.Optional[tp.Dict[str, tp.Any]]=None,
                           literacy_batch_size: tp.Optional[int]=64,
                           metrics_results: tp.Optional[pd.DataFrame]=None,
                           **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест подсчета качества речи финальных ответов
    Parameters:
        sampler: Объект класса Sampler с данными, в которых содержится ответ модели
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        test_literacy: Булевский флаг, если True, то тест на орфографию проводится
        test_readability: Булевский флаг, если True, то тест на читаемость проводится
        test_spelling: Булевский флаг, если True, то тест на лингвистическую приемлемость проводится.  
            N.B. Для проведения теста также нужен переданный параметр spelling_model.
        spelling_model: Класс-обертка над API моделью для исправления ошибок, имеет метод get_answer_sage. 
            Если передать None, то тест проводиться не будет.
        spelling_model_cache: Словарь для кэширования работы модели исправления орфографии. 
            Если параметр не передать, автоматически создастся пустой словарь; заполенный словарь впоследствии будет возвращен в precomputed.
        literacy_model: Языковая модель для оценки лингвистической приемлемости, имеет методы to_device и generate 
        literacy_tokenizer: Соответствующий модели токенизатор, имеет методы encode и decode
        literacy_model_config: Параметры языковой модели для оценки лингвистической приемлемости
        literacy_tokenizer_config: Параметры соответствующего токенизатора для модели для оценки лингвистической приемлемости
        literacy_batch_size: Размер батча при обработке моделью для оценки лингвистической приемлемости 
        metrics_results: Опциональный словарь со значениями метрик. 
            Если не None, то подсчет метрик не производится, и на основе переданного аргумента выставляется светофор.
    Return: 
        Словарь в соответствии с концепцией llm_val
    '''
    figs = []
    if metrics_results is None:
        data = getattr(sampler, data_type)['X']
        model_answer = data['model_answer'].to_list()
        metrics_results = {}
        if test_literacy:
            metrics_results['literacy_score'] = score_data_by_model(model_answer,
                                                                    literacy_model,
                                                                    literacy_tokenizer,
                                                                    literacy_model_config,
                                                                    literacy_tokenizer_config,
                                                                    literacy_batch_size)
            prompt_literacy = score_data_by_model(data['question'].to_list(),
                                                  literacy_model,
                                                  literacy_tokenizer,
                                                  literacy_model_config,
                                                  literacy_tokenizer_config,
                                                  literacy_batch_size)
            figs.append(plot_literacy_confusion_matrix(prompt_literacy,
                                                       metrics_results['literacy_score']))
            metrics_results['literacy_score'] = np.mean(metrics_results['literacy_score'])
        if test_readability:
            metrics_results['readability_score'] = corpus_readability_COLE_index(model_answer)
            prompt_readability = corpus_readability_COLE_index(data['question'].to_list())
            figs.append(plot_readability_confusion_matrix(prompt_readability, metrics_results['readability_score']))
            metrics_results['readability_score'] = np.mean(metrics_results['readability_score'])
        if test_spelling and spelling_model is not None:
            metrics_results['spelling_score'] = spelling_score_api(model_answer,
                                                                   spelling_model,
                                                                   normalization=True,
                                                                   cache_api=spelling_model_cache)[0]
    report = report_valtest_speech_quality(metrics_results.copy(),
                                           [test_literacy, test_readability, test_spelling],
                                           data_type,
                                           figs)
    precomputed = {'metrics_results': metrics_results,
                   'spelling_model_cache': spelling_model_cache}
    return {'report': report,
            'precomputed': precomputed}