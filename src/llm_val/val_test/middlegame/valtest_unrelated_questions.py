import sys
import typing as tp
import pandas as pd
from tqdm.auto import tqdm as tqdm

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold

def report_valtest_unrelated_questions(value: float,
                                       model_answers: tp.List[str],
                                       threshold: tp.Tuple[float, float]=(0.0, 0.05),
                                       is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        value: Число, отвечающее цвету светофора
        model_answers: Ответы модели
        threshold: Границы светофора
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_dict = {}
    res_dict['result_plots'] = []
    res_dict['semaphore'] = 'gray' if is_info else semaphore_by_threshold(value,
                                                                          threshold=threshold,
                                                                          greater_is_better=False, 
                                                                          left_border_is_bigger=False, 
                                                                          right_border_is_bigger=True)
    res_dict['result_dataframes'] = [pd.DataFrame(pd.DataFrame(model_answers, 
                                                               columns=['Model answers']).value_counts())]
    return res_dict


def valtest_unrelated_questions(sampler: Sampler,
                                data_type: str='test',
                                themes: tp.List[tp.List[str]]=None,
                                model: tp.Optional[tp.Any]=None,
                                threshold: tp.Tuple[float, float]=(0.0, 0.05),
                                is_info: bool=False,
                                **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест наличия вопросов, не относящихся к бизнес-процессу (не по цели бизнес-процесса).
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        themes: Список списков тематик. 
                themes[i] - список тематик, которые относятся к вопросу question[i].
                Валидатор обязан сам определить этот список на основе наличия тематик в вопросах или
                исходя из бизнес-смысла конкретного применения.
        model: Модель, определяющая релевантность вопроса question[i] к списку тематик, определенных в themes[i]
               Модель обязана иметь метод q_relevance, который на вход принимает два обязательных параметра: 
                   1. question - вопрос
                   2. topics - список возможных тематик для этого вопроса
               Данный метод должен возвращать строку "да" или "нет"
               либо строку, в которой эта информация ("да" или "нет") присутствует.
        threshold: Границы светофора
        is_info: True, если тест информативный
    Return:
        Словарь в соответствии с концепцией llm_val   
    '''
    answers = []
    value = 0
    X = getattr(sampler, data_type)['X']
    lenght = len(X)
    with tqdm(total=lenght, desc='Calculate...', file=sys.stdout, mininterval=0.05) as pbar:
        for question, topics in zip(list(X['question']), themes):
            answers.append(model.q_relevance(question=question, 
                                             topics=topics))
            pbar.update(1)
            sys.stdout.flush()

    value = 1.0 - len([ans for ans in answers if 'да' in ans.lower()]) / len(answers)
    report = report_valtest_unrelated_questions(value=value,
                                                model_answers=answers,
                                                threshold=threshold,
                                                is_info=is_info)
    precomputed = {'answers': answers,
                   'value': value}
    return {'report': report, 
            'precomputed': precomputed}
