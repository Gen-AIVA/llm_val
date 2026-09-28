import typing as tp

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler


class UniversalScorer(Scorer):
    def __init__(self,
                 metrics: tp.Dict[str, tp.Any]):
        '''
        metrics: словарь, где ключ - название метрики, 
                              значение - словарь с двумя обязательными ключами: `call`(`Callable`) и `is_singlecol`(`bool`)
        '''
        super().__init__(metrics)


    def calc(self,
             sampler: Sampler,
             data_type: str='test',
             metric_name: tp.Optional[str]=None) -> tp.Dict[str, float]:
        '''
        Метод для подсчета метрик
        Parameters:
            sampler: Объект класса Sampler
            data_type: Тип данных, на которых производится расчет метрики. Hint: название атрибута в sampler'е
            metric_name: Название возвращаемой метрики
        Return:
            Словарь с посчитанными метриками
        '''
        metrics = self.metrics
        data = getattr(sampler, data_type)['y']
        if metric_name is not None:
            metrics = {metric_name: self.metrics[metric_name]}
        metric_res = {}
        for func_name, func_dict in metrics.items():
            if func_dict['is_singlecol']:
                for name_col in data:
                    name = f'{func_name}_{name_col}'
                    metric_res[name] = func_dict['call'](data[name_col])
            else:
                metric_res[func_name] = func_dict['call'](data)
        return metric_res
