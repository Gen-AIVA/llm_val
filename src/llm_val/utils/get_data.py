import typing as tp

from llm_val.sampler.Sampler import Sampler


def get_data(sampler_obj: tp.Union[Sampler, tp.Dict],
             data_type: str, 
             key: str, 
             exec_env: str='default') -> tp.Any:
    '''
    Parameters:
        sampler_obj: Объект сэмплера, содержащий данные для извлечения
        data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
        key: Ключ или имя атрибута для извлечения конкретных данных (например, 'X', 'y')
        exec_env: Переменная исполняемого окружения ('default' или 'sberds') 
    Return: 
        Содержимое sampler_obj из data_type по ключу key
    '''
    res = None
    if exec_env == 'default':
        res = getattr(sampler_obj, data_type)[key]
    elif exec_env == 'sberds':
        res = sampler_obj[data_type][key]
    return res
