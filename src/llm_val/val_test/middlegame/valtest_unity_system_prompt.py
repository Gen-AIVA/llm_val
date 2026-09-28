import pandas as pd
import typing as tp

from llm_val.sampler.Sampler import Sampler
from llm_val.utils.report_helper import semaphore_by_threshold

def report_valtest_unity_system_prompt(value: float,
                                       theme_to_prompts: tp.Dict[str, tp.List[str]],
                                       threshold: tp.Tuple[float, float]=(0.05, 0.5),
                                       is_info: bool=False) -> tp.Dict[str, tp.Any]:
    '''
    Создание report'а
    Parameters:
        value: Число, отвечающее цвету светофора. 
               Если в ML пайплайне отсутствует тематический классификатор, и в тестовом сете больше одного
               системного промпта, то value равен 1, иначе 0.
               Если тематический классификатор есть, и в рамках одной тематики больше одного системного 
               промпта, то value равен 1. Если указан только один системный промпт для каждой тематики, то 
               value равен 0.1, иначе 0
        theme_to_prompts: Словарь, ключами которого являются тематики, а значениями - список 
                          системных промптов для конкретной тематики.
        threshold: Границы светофора
        is_info: True, если тест информативный
    Return:
        report в соответствии с концепцией llm_val
    '''
    res_dict = {}
    res_dict['result_plots'] = []
    res_dict['semaphore'] = 'gray' if is_info else semaphore_by_threshold(value,
                                                                          threshold=threshold,
                                                                          greater_is_better=False)
    res_dict['result_dataframes'] = [pd.DataFrame.from_dict(theme_to_prompts, 
                                                            orient='index', 
                                                            columns=['list_of_prompts'])]
    return res_dict


def valtest_unity_system_prompt(sampler: Sampler,
                                thematic_classifier: bool,
                                data_type: str='test',
                                is_info: bool=False,
                                **kwargs) -> tp.Dict[str, tp.Any]:
    '''
    Тест на единство системного промпта.
    Parameters:
        sampler: Объект класса Sampler
        thematic_classifier: Флаг, используется ли тематический классификатор.
                             Валидатору необходимо уточнить у разработчиков - определяются ли тематики у 
                             вопросов автоматически. Если тематики определяются автоматически, то в рамках 
                             каждой тематики обязан быть один системный промпт. Иначе системный промпт
                             обязан быть единым для всех вопросов в тестовой корзине.
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        is_info: True, если тест информативный
    Return:
        Словарь в соответствии с концепцией llm_val   
    '''
    value = 0
    X = getattr(sampler, data_type)['X']
    X['thema'] = X['thema'].fillna('None')
    unique_system_prompts = list(X['system'].unique())
    theme_to_prompts = X.groupby('thema')['system'].unique().apply(list).to_dict()
    if thematic_classifier is False:
        value = 0 if len(unique_system_prompts) == 1 else 1
    else:
        if len(theme_to_prompts) > 1:
            value = 0.1
        for theme, prompts in theme_to_prompts.items():
            if len(prompts) != 1:
                value = 1
                break
    report = report_valtest_unity_system_prompt(value=value,
                                                theme_to_prompts=theme_to_prompts,
                                                is_info=is_info)
    precomputed = {'unique_system_prompts': unique_system_prompts,
                   'theme_to_prompts': theme_to_prompts}
    return {'report': report, 
            'precomputed': precomputed}
