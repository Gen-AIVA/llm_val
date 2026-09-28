import typing as tp
from copy import deepcopy


class Sampler:
    '''
    Простейший("базовый") семплер, который просто хранит переданные в него данные
    '''
    def __init__(self,
                 train: tp.Optional[tp.Any]=None,
                 val: tp.Optional[tp.Any]=None,
                 test: tp.Optional[tp.Any]=None,
                 oot: tp.Optional[tp.Any]=None):
        '''
        train: Обучающий набор данных
        val: Валидационный набор данных
        test: Тестовый набор данных
        oot: Набор данных out-of-time
        '''
        self.train = train
        self.val = val
        self.test = test
        self.oot = oot

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, var):
        self._check_var(var, 'train')
        self._train = deepcopy(var)

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, var):
        self._check_var(var, 'val')
        self._val = deepcopy(var)

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, var):
        self._check_var(var, 'test')
        self._test = deepcopy(var)
        
    @property
    def oot(self):
        return self._oot

    @oot.setter
    def oot(self, var):
        self._check_var(var, 'oot')
        self._oot = deepcopy(var)


    def _check_var(self, var, var_name):
        if var is not None:
            if not isinstance(var, dict):
                raise AttributeError(f'{var_name} должен быть словарем')
            
            keys = set(var.keys())
            required = {'X', 'y'}
            allowed = {'X', 'y', 'traces'}
            
            if not required.issubset(keys):
                missing = required - keys
                raise AttributeError(f'{var_name} должен обязательно содержать ключи: {", ".join(missing)}')
            
            if not keys.issubset(allowed):
                extra = keys - allowed
                raise AttributeError(f'{var_name} содержит недопустимые ключи: {", ".join(extra)}. '
                                   f'Разрешены только X, y и traces.')
