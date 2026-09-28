import typing as tp
from abc import ABC, abstractmethod

from llm_val.sampler.Sampler import Sampler


class Scorer(ABC):
    '''
    Абстрактный класс скорера.         
    Все скореры должны наследоваться от данного класса.
    '''
    def __init__(self,
                 metrics: tp.Dict[str, tp.Any]):
        self.metrics = metrics


    @abstractmethod
    def calc(self,
             sampler: Sampler,
             data_type: str,
             metric_name: str) -> tp.Dict[str, float]:
        '''Метод подсчета метрик'''
        raise NotImplementedError(f'Определите calc в {self.__class__.__name__}')
