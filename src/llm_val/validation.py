import os
import shutil
import joblib
import inspect
import warnings
import datetime
import traceback
import typing as tp
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from importlib import import_module

from llm_val.scorer.Scorer import Scorer
from llm_val.sampler.Sampler import Sampler
from llm_val.utils.parser import parse_json, parse_pkl
from llm_val.utils.XlsxAgregator import XlsxAgregator


class Validation():
    '''
    Аггрегатор для проведения валидации
    '''
    def __init__(self, 
                 sampler: Sampler,
                 scorer: Scorer,
                 param_config: tp.Dict[str, tp.Any],
                 desc_config: tp.Union[os.PathLike, tp.Dict],
                 agg_config: tp.Union[os.PathLike, tp.Dict],
                 save_dir: os.PathLike='./test_results',
                 is_semaphore_desc_info: bool=False,
                 precomputed: tp.Optional[tp.Union[os.PathLike, tp.Dict]]=None):
        '''
        Конструктор
        Parameters:
            sampler: Объект класса Sampler
            scorer: Объект класса Scorer
            param_config: Словарь с конфигурацией каждого теста.
                Ключ - название теста, значение - словарь с параметрами теста 
            desc_config: Описание тестов
            agg_config: Информация о агрегации тестов
            save_dir: Путь до директории сохранения результатов
            is_semaphore_desc_info: Если False, то информация о критериях светофоров не выводится
            precomputed: Словарь с предподсчитанными значениями каждого теста.
                Ключ - название теста, значение - словарь с предподсчитанными значениями
        '''
        self.sampler = sampler
        self.scorer = scorer
        self.desc_config = desc_config if isinstance(desc_config, dict) else parse_json(desc_config)
        self.agg_config = agg_config if isinstance(agg_config, dict) else parse_json(agg_config)
        self.param_config = self._get_param_config(param_config)
        self.save_dir = save_dir
        self.is_semaphore_desc_info = is_semaphore_desc_info
        if precomputed is not None:
            precomputed = precomputed if isinstance(precomputed, dict) else parse_pkl(precomputed)
        self.precomputed = dict() if precomputed is None else precomputed
        

    def validate(self,
                 save_res_pkl: bool=True,
                 save_excel: bool=True,
                 save_word: bool=False) -> tp.Tuple[tp.Dict, tp.Dict]:
        '''
        Метод для проведения валидации
        Parameters:
            save_res_pkl: Если True, то сохраняет результаты тестов в .pkl
            save_excel: Если True, то сохраняет результаты тестов в .xlsx
            save_word: Если True, то сохраняет результаты тестов в .doc (TODO)
        Return:
            Кортеж со словарем с результатами тестов и со словарем с агрегацией по блокам
        '''
        tests_result = {}
        semaphore_blocks = defaultdict(list)
        self.store_dir = self._create_store_dir(self.save_dir)
        for test_name, test_params in self.param_config.items():
            try:
                print(f'Run test {test_name}...')
                func_test = self._get_callable_from_test_name(test_name)
                prec_params = self.precomputed.get(test_name, {})
                test_res = self._run_test(func_test,
                                          test_params, 
                                          prec_params)
                tests_result[test_name] = test_res[0]
                self.precomputed[test_name] = test_res[1]
                test_block = self.desc_config[test_name]['block_key']
                semaphore_blocks[test_block].append(test_res[0]['semaphore'])
            except Exception as e:
                err_msg = traceback.format_exc()
                print(f'Test: {test_name} raised an exception:')
                print(err_msg)
                tests_result[test_name] = self._create_error_res_test(err_msg)
            self._save_information(self.precomputed,
                                   self.store_dir,
                                   self.save_dir,
                                   'precomputed.pkl')
        agg_table = self._agg_semaphore(semaphore_blocks)
        
        if save_res_pkl:
            self._save_information(tests_result,
                                   self.store_dir,
                                   self.save_dir,
                                   'test_results.pkl')
        if save_excel:
            self._save_excel(tests_result,
                             agg_table,
                             self.desc_config,
                             self.store_dir,
                             self.save_dir,
                             'test_results.xlsx',
                             self.is_semaphore_desc_info)
        if save_word:
            self._save_word()
        return tests_result, agg_table


    def _run_test(self,
                  func_test: tp.Callable,
                  test_params: tp.Dict[str, tp.Any],
                  prec_params: tp.Dict[str, tp.Any]) -> tp.Tuple[tp.Dict, tp.Dict]:
        '''
        Метод для запуска теста
        Parameters:
            func_test: tp.Callable объект функции теста
            test_params: Параметры теста
            prec_params: Предподсчитанные значения теста
        Return:
            Кортеж с результатом теста и precomputed
        '''
        func_signature = inspect.signature(func_test)
        func_params = func_signature.parameters
        if 'scorer' in func_params:
            test_res = func_test(sampler=self.sampler, 
                                scorer=self.scorer,
                                **prec_params,
                                **test_params)
        else:
            test_res = func_test(sampler=self.sampler, 
                                **prec_params,
                                **test_params)
        return test_res['report'], test_res['precomputed']


    def _save_information(self,
                          data: tp.Any,
                          local_dir: os.PathLike,
                          global_dir: os.PathLike,
                          filename: str):
        '''
        Метод для сохранения пиклов
        Parameters:
            data: Данные, которые необходимо сохранить
            local_dir: Локальная директория, куда сохраняется пикл
            global_dir: Глобальная директория, куда сохраняется пикл
            filename: Название файла
        '''
        local_path = os.path.join(local_dir, filename)
        global_path = os.path.join(global_dir, filename)
        joblib.dump(data, local_path)
        joblib.dump(data, global_path)


    def _get_param_config(self, 
                          param_config: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
        '''
        Метод для верификации параметров тестов
        Parameters:
            param_config: Параметры тестов
        Return:
            Словарь с параметрами тестов, которые будут проведены
        '''
        wrong_test = []
        new_param_config = {}
        for test_name, test_params in param_config.items():
            if test_name in self.desc_config:
                new_param_config[test_name] = deepcopy(test_params)
            else:
                wrong_test.append(test_name)
        if len(wrong_test) > 0:
            wrong_test_msg = ', '.join(wrong_test)
            warn_msg = f'Тесты {wrong_test_msg} не будут проведены, так как их нет в конфиге допустимых тестов'
            warnings.warn(warn_msg, category=UserWarning)
        return new_param_config


    def _get_callable_from_test_name(self,
                                     test_name: str) -> tp.Callable:
        '''
        Метод получения tp.Callable объекта по названию теста
        Parameters:
            test_name: Название теста
        Return:
            tp.Callable объект функции теста
        '''
        path = self.desc_config.get(test_name)['target']
        func = self._get_callable(path)
        return func


    def _get_callable(self,
                      func_path: str) -> tp.Callable:
        '''
        Метод получения tp.Callable объекта по пути
        Parameters:
            func_path: Путь до теста
        Return:
            tp.Callable объект
        '''
        return getattr(import_module(".".join(func_path.split(".")[:-1])), func_path.split(".")[-1])


    def _create_store_dir(self,
                          save_dir: os.PathLike) -> os.PathLike:
        '''
        Метод создания директории с результатами тестов
        Parameters:
            save_dir: Путь до директории для сохранения результатов
        Return:
            Путь до директории
        '''
        store_dir = os.path.join(save_dir,
                                 datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        os.makedirs(store_dir, exist_ok=True)
        return store_dir


    def _create_error_res_test(self, 
                               err_msg: str) -> tp.Dict[str, tp.Any]:
        '''
        Метод создания словаря теста с ошибкой
        Parameters:
            err_msg: Сообщение ошибки
        Return:
            Словарь с ошибкой
        '''
        return {'semaphore': 'gray',
                'result_dataframes': [pd.DataFrame([{'error': err_msg}])],
                'result_plots': []}


    def _agg_semaphore(self, 
                       semaphore_blocks: tp.Dict[str, tp.List[str]]) -> tp.Dict[str, tp.Any]:
        '''
        Метод агрегации светофоров в разрезе блоков
        Parameters:
            semaphore_blocks: Словарь со списками цветов светофоров
        Return:
            Словарь с информацией о агрегации тестов в разрезе блоков
        '''
        block_color = {}
        for block_name, block_list in semaphore_blocks.items():
            agg_block = self.agg_config.get(block_name)
            if agg_block is None:
                continue
            func = self._get_callable(agg_block['func'])
            if func is None:
                continue
            block_color[agg_block['name']] = {'Светофор по блоку': func(block_list),
                                              'Количество проведенных тестов': len(block_list)}
        return block_color
            

    def _save_excel(self, 
                    data: tp.Dict[str, tp.Dict],
                    agg_table: tp.Dict[str, tp.Dict],
                    desc_config: tp.Dict,
                    local_dir: os.PathLike,
                    global_dir: os.PathLike,
                    filename: str, 
                    is_semaphore_desc_info: bool=False):
        '''
        Метод агрегации светофоров в разрезе блоков
        Parameters:
            data: Словарь с результатами проведения тестов. 
                Ключ - название теста, значение - report по тестам
            agg_table: Словарь с агрегированными светофорами по блокам
                Ключ - название блока, значение - информация о агрегации
            desc_config: Словрь с описанием тестов.
                Ключ - название теста, значение - информация о тесте
            local_dir: Локальная директория, куда сохраняется .xlsx
            global_dir: Глобальная директория, куда сохраняется .xlsx
            filename: Название файла
            is_semaphore_desc_info: Если False, то информация о критериях светофоров не выводится
        '''
        path = os.path.join(local_dir, filename)
        excel_aggregator = XlsxAgregator(path, is_semaphore_desc_info)
        excel_aggregator.save_xlsx(data, agg_table, desc_config)
        shutil.copyfile(path, str(os.path.join(global_dir, filename)))


    def _save_word(self):
        pass