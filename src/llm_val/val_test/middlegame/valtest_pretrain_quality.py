import time
import random
import numpy as np
import typing as tp
import pandas as pd

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.input_prepocessors import preprocess_pkl_input
from llm_val.utils.speech_quality.utils import score_data_by_model
from llm_val.utils.base_repeatness import get_sentences_repeatness_ratio
from llm_val.val_test.model_quality.valtest_speech_quality import valtest_speech_quality
from llm_val.utils.report_helper import semaphore_by_threshold_without_yellow, semaphore_by_threshold_without_red, worst_semaphore
from llm_val.val_test.data_quality.valtest_complex_data_quality import get_hashes, get_duplicates_and_contradictions, set_of_sets_size


def report_valtest_pretrain_quality(speech_quality_res: tp.Optional[tp.Dict[str, float]]=None,
                                    consistency_ratio: tp.Optional[float]=None,
                                    themes_goal_consistency_ratio: tp.Optional[float]=None,
                                    toxity_ratio: tp.Optional[bool]=None,
                                    duplicates_ratio: tp.Optional[float]=None,
                                    repeats_ratio: tp.Optional[float]=None,
                                    consistency_threshold: float=0.5,
                                    toxity_threshold: float=0.99,
                                    theme_goal_threshold: float=0.8,
                                    duplicates_threshold: float=0.4,
                                    repeats_threshold: float=0.3,
                                    speech_quality_threshold: tp.Tuple[float, float, float]=(50.0, 0.5, 50.0),
                                    figs: tp.Optional[tp.List[tp.Any]]=None,
                                    is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        speech_quality_res: Словарь с результатом тестов на качество речи
        consistency_ratio: Доля запросов с изменяющимися во времени данными
        themes_goal_consistency_ratio: Доля запросов соответствующих заявленной целе/теме дообучения 
        toxity_ratio: Доля токсичных запросов
        duplicates_ratio: Доля дубликатов в запросах
        repeats_ratio: Доля запросов с зацикливаниями
        consistency_threshold: Порог доли запросов для теста на изменчивость данных для pretrain во времени
        toxity_threshold: Порог доли запросов для теста на токсичность данных
        theme_goal_threshold: Порог доли запросов для теста на соответствие тематики документов цели дообучения
        duplicates_threshold: Порог доли запросов для теста на дубликаты
        repeats_threshold: Порог доли запросов для теста на зацикливания
        speech_quality_threshold: Кортеж порогов для теста на качество речи в следующем порядке:
            порог для теста на читаемость, для теста на лингвистическую грамотность, для теста на орфографию.
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        figs: Список графиков
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    test_res = ['Изменчивость во времени', 'Соответствие тематики цели дообучения', 'Токсичность', 'Наличие дубликатов', 'Доля зацикливаний']
    test_values = [consistency_ratio, themes_goal_consistency_ratio, toxity_ratio, duplicates_ratio, repeats_ratio]
    result = pd.DataFrame({name: val for name, val in zip(test_res, test_values) if val is not None}, index=['Значение']).T
    semaphore_funcs = [semaphore_by_threshold_without_yellow, semaphore_by_threshold_without_red]
    semaphore_params =  [(1, consistency_threshold, True),
                         (0, theme_goal_threshold, True),
                         (0, toxity_threshold, False),
                         (1, duplicates_threshold, False),
                         (1, repeats_threshold, False)]
    result['Результат теста'] = [semaphore_funcs[params[0]](val, params[1], params[2]) for val, params in zip(test_values, semaphore_params) if val is not None]
    speech_dict = {'readability_score': 'Читаемость',  'literacy_score': 'Лингвистическая грамотность', 'spelling_score': 'Орфография'}
    speech_threshold = {'readability_score': speech_quality_threshold[0],  'literacy_score': speech_quality_threshold[1], 'spelling_score': speech_quality_threshold[2]}
    if speech_quality_res:
        for key, value in speech_quality_res.items():
            result.loc[speech_dict[key]] = [value, semaphore_by_threshold_without_red(value, speech_threshold[key])]
    res_dict = {}
    res_dict['semaphore'] = 'gray' if is_info else worst_semaphore(result['Результат теста'].values)
    res_dict['result_plots'] = figs
    res_dict['result_dataframes'] = [result]
    return res_dict


def valtest_pretrain_quality(input_list: tp.List[str],
                             model: tp.Any,
                             preprocess_input: tp.Callable[[str], tp.List[str]]=preprocess_pkl_input,
                             chunk_size: int=1024,
                             speech_quality_args: tp.Optional[tp.Dict[str, tp.Any]]=None,
                             speech_quality_thresholds: tp.Tuple[float, float, float]=(50.0, 0.5, 50.0),
                             consistency_sample_frac: float=0.1,
                             consistency_check_prompt: tp.Optional[str]=None,
                             consistency_threshold: float=0.5,
                             goal_theme: tp.Optional[str]=None,
                             themes_goal_sample_frac: float=0.1,
                             theme_goal_threshold: float=0.8,
                             theme_goal_prompt: tp.Optional[str]=None,
                             toxity_args: tp.Optional[tp.Dict[str, tp.Any]]=None,
                             toxity_threshold: float=0.1,
                             toxity_frac_threshold: float=0.99,
                             repeats_threshold: float=0.1,
                             duplicates_threshold: float=0.1,
                             seed: int=42,
                             is_info: bool=False,
                             speech_quality_res: tp.Optional[tp.Dict[str, tp.Any]]=None,
                             consistency_ratio: tp.Optional[float]=None,
                             themes_goal_consistency_ratio: tp.Optional[float]=None,
                             toxity_ratio: tp.Optional[float]=None,
                             repeats_ratio: tp.Optional[float]=None,
                             duplicates_ratio: tp.Optional[float]=None,                             
                             **kwargs) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    '''
    Тест проверки качества PreTrain
    Parameters:
        input_list: Список путей к набору исходных данных
        model: Модель, используемая для проверки соответствия запросов целевой тематике и изменчивости во времени
        chunk_size: Длина строк, на которые будет разбиваться содержимое файлов с исходными данными
        speech_quality_args: Словарь аргументов для тестов на качество речи
        speech_quality_threshold: Кортеж порогов для теста на качество речи в следующем порядке:
            порог для теста на читаемость, для теста на лингвистическую грамотность, для теста на орфографию.
        consistency_sample_frac: Доля запросов для проверки на изменчивость во времени
        consistency_check_prompt: Промпт, используемый для запросов в модель для проверки запросов 
            на изменчивость данных во времени - f-строка с параметром question
        consistency_threshold: Порог доли запросов для теста на изменчивость данных во времени
        goal_theme: Целевая тематика для дообучения 
        themes_goal_sample_frac: Доля запросов для проверки на соответствие целевой тематике
        theme_goal_threshold: Порог доли запросов для теста на соответствие целевой тематике
        theme_goal_prompt: Промпт, используемый для запросов в модель для проверки соответствия 
            запросов целевой тематике - f-строка с параметром question
        toxity_args: Словарь аргументов для теста на наличие токсичности 
        toxity_threshold: Порог доли запросов для теста на токсичность данных
        toxity_frac_threshold: Пороговый уровень токсичности, выше которого запрос считается токсичным
        repeats_threshold: Порог доли запросов для теста на зацикливания
        duplicates_threshold: Порог доли запросов для теста на дубликаты
        seed: Начальное значение датчика случайных чисел
        is_info: True, если тест информативный
        speech_quality_res: Опциональный словарь с результатом тестов на качество речи
        consistency_ratio: Опциональная доля запросов с изменяющимися во времени данными
        themes_goal_consistency_ratio: Опциональная доля запросов, соответствующих целевой тематике
        toxity_ratio: Опциональное значение доли токсичных запросов
        repeats_ratio: Опциональный флаг с проверкой на зацикливания
        duplicates_ratio: Опциональный флаг с проверкой на дубликаты
        Для всех опциональных аргументов: если флаг не None, то проверка не производится,
            и на основе переданного аргумента выставляется светофор.
    Return: 
        Словарь в соответствии с концепцией llm_val
    '''
    text_list = preprocess_input(input_list, chunk_size)
    data = pd.DataFrame({'model_answer': text_list})
    data['question'] = ''
    n_chunks = len(text_list)
    random.seed(seed)
    np.random.seed(seed)
    speech_quality_plots = []
    if speech_quality_res is None and speech_quality_args:
        smplr = Sampler
        smplr.test = {'X': data, 'y': None}
        speech_quality_args['sampler'] = smplr
        speech_quality_dict = valtest_speech_quality(**speech_quality_args)
        speech_quality_res = speech_quality_dict['precomputed']['metrics_results']
        speech_quality_plots = speech_quality_dict['report']['result_plots']
    if consistency_ratio is None:
        if consistency_check_prompt is None:
            consistency_check_prompt = 'Представь, что ты эксперт в теме' + goal_theme + '. Может ли ответ на следующий вопрос измениться в ближайшие несколько месяцев? Вопрос: "{question}". Ответь одним словом: "Да" или "Нет".'
        questions = [consistency_check_prompt.format(question=question) for question in data['model_answer'].sample(frac=consistency_sample_frac, random_state=1)]
        answers = get_model_answers(model, questions)
        consistency_ratio = (np.array(['Да' in answer for answer in answers])*1).sum() / len(questions)
    if themes_goal_consistency_ratio is None:
        if theme_goal_prompt is None:
            theme_goal_prompt = 'Тема: ' + goal_theme + '. Соответствует ли этой теме текст : "{question}"? Ответь один словом: "Да" или "Нет".'
        questions = [theme_goal_prompt.format(question=question) for question in data['model_answer'].sample(frac=themes_goal_sample_frac, random_state=1)]
        answers = get_model_answers(model, questions)
        themes_goal_consistency_ratio = (np.array(['Да' in answer for answer in answers]) * 1).sum() / len(questions)
    if toxity_ratio is None:
        if toxity_args:
            toxity_res = score_data_by_model(text_list, **toxity_args)
            toxity_ratio = (np.array(toxity_res) > toxity_frac_threshold).sum() / n_chunks
    if duplicates_ratio is None:
        duplicates, _ = get_duplicates_and_contradictions(get_hashes(text_list))
        duplicates_ratio = set_of_sets_size(duplicates) / n_chunks
    if repeats_ratio is None:
        repeats_ratio = get_sentences_repeatness_ratio(text_list)
    report = report_valtest_pretrain_quality(speech_quality_res,
                                             consistency_ratio,
                                             themes_goal_consistency_ratio,
                                             toxity_ratio,
                                             duplicates_ratio,
                                             repeats_ratio,
                                             consistency_threshold,
                                             toxity_threshold,
                                             theme_goal_threshold,
                                             duplicates_threshold,
                                             repeats_threshold,
                                             speech_quality_thresholds,
                                             speech_quality_plots,
                                             is_info)
    precomputed = {'speech_quality_res': speech_quality_res,
                   'consistency_ratio': consistency_ratio,
                   'themes_goal_consistency_ratio': themes_goal_consistency_ratio,
                   'toxity_ratio': toxity_ratio,
                   'duplicates_ratio': duplicates_ratio,
                   'repeats_ratio': repeats_ratio}
    return {'report': report,
            'precomputed': precomputed}


def get_model_answers(model: tp.Any,
                      questions: tp.List[str],
                      batch_size: int=64) -> tp.List[str]:
    '''
    Функция для получения ответов языковой модели по батчам
    Parameters:
        model: Языковая модель, имеет метод predict(список вопросов)
        questions: Список вопросов
        batch_size: Размер батча
    Return:
        Список ответов языковой модели
    '''
    questions_list = questions
    answers = []
    for start in range(0, len(questions_list), batch_size):
        answers.extend(model.predict(questions_list[start: (start + batch_size)]))
        time.sleep(5) ## гигачат часто отваливается если к нему непрерывно стучаться, кажется это добавляет стабильности
    return answers
