import pandas as pd
import typing as tp

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold_without_red

def report_valtest_definition_themes(value: float,
                                     themes: tp.Dict[str, int],
                                     threshold: float=0.5,
                                     is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Parameters:
        value: Число, отвечающее цвету светофора. 
               Если в рамках одной тематики больше одного системного промпта, то value равен 1.
        themes: Словарь, ключами которого являются тематики, а значениями - список 
                системных промптов для конкретной тематики.
        threshold: Граница светофора
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_dict = {}
    res_dict['result_plots'] = []
    res_dict['semaphore'] = 'gray' if is_info else semaphore_by_threshold_without_red(value,
                                                                                      threshold=threshold,
                                                                                      greater_is_better=False, 
                                                                                      border_is_bigger=True)
    res_dict['result_dataframes'] = [pd.DataFrame.from_dict(themes,
                                                            orient='index', 
                                                            columns = ['Число вопросов'])]
    return res_dict


def valtest_definition_themes(sampler: Sampler,
                              data_type: str='test',
                              threshold: float=0.5,
                              is_info: bool=False,
                              **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на присутсвие тематик у вопросов.
    Parameters:
        sampler: Объект класса Sampler
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        threshold: Граница светофора
        is_info: True, если тест информативный
    Return:
        Словарь в соответствии с концепцией llm_val   
    '''
    value = 0
    X = getattr(sampler, data_type)['X']
    X['thema'] = X['thema'].fillna('unkwown_thema')
    themes = X['thema'].value_counts().to_dict()
    value = 0 if 'unkwown_thema' not in themes else 1
    report = report_valtest_definition_themes(value=value,
                                              themes=themes,
                                              threshold=threshold,
                                              is_info=is_info)
    precomputed = {'themes': themes}
    return {'report': report, 
            'precomputed': precomputed}