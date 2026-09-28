import typing as tp
from abc import ABC, abstractmethod
from llm_val.utils.api import GigaChatAPI

class GigaModel(ABC):
    '''
    Абстрактный класс, который является оберткой над GigaChatAPI, для удобной работы с API.
    Все классы-обертки API'шки должны наследоваться от данного класса.
    '''
    def __init__(self,
                 contour: tp.Optional[str]=None,
                 token: tp.Optional[str]=None,
                 auth_data: tp.Optional[str]=None,
                 api_url: tp.Optional[str]=None,
                 oauth_url: tp.Optional[str]=None,
                 rquid: tp.Optional[str]=None,
                 model: str='GigaChat:latest',
                 sys_promt: tp.Optional[str]=None):
        '''
        contour: Название контура. Принимает одно из трех значениий ['sigma', 'mlspace', 'datalab']
        token: Токен для авторизации. Передается, если уже получен
        auth_data: Авторизационные данные. Client id и Client Secret, закодированные в Base64
        api_url: URL до API GigaChat 
        oauth_url: URL до сервиса авторизации GigaChat
        rquid: Уникальный идентификатор запроса. Соответствует формату uuid4.
                Параметр для журналирования входящих вызовов и разбора инцидентов.
        model: Название модели
        sys_promt: Системный промт
        '''
        self.api = GigaChatAPI(contour=contour,
                               token=token,
                               auth_data=auth_data,
                               api_url=api_url,
                               oauth_url=oauth_url,
                               rquid=rquid)
        self.model = model
        self.sys_promt = {'role': 'system', 
                          'content': sys_promt} if sys_promt is not None else None


    @abstractmethod
    def fit(self, messages):
        '''fit method'''
        raise NotImplementedError(f'Определите fit в {self.__class__.__name__}')


    @abstractmethod
    def predict(self, messages):
        '''predict method'''
        raise NotImplementedError(f'Определите predict в {self.__class__.__name__}')
