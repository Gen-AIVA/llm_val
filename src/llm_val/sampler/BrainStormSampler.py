import re
import json
import copy
import pandas as pd
from typing import Dict, List, Any, Optional

from llm_val.sampler.Sampler import Sampler


class BrainStormSampler(Sampler):
    '''
    Класс, созданный для подготовки и проверки целостности данных
    Parameters:
        global_config_path: Путь до глобального конфигурационного файла
        json_path: Путь до json-файла, содержащего данные для обработки
        test: Данные для обработки. указывается в случае,
              когда нет отдельного json-файла с данными
    '''
    def __init__(self,
                 global_config_path: str,
                 json_path: Optional[str]=None,
                 test: Optional[Dict[str, Dict[str, Any]]]=None):

        self.global_config_path = global_config_path
        self.json_path = json_path

        if self.json_path is None and test is None:
            raise AttributeError('В sampler не были переданы данные.')
        self.__read_important_names()
        test_ = self.get_data(test)
        super().__init__(test=test_)

    def __read_important_names(self):
        '''
        Метод, считывающий наименования важных
        полей датасета из глобального конфига
        '''
        with open(self.global_config_path, 'r', encoding='utf-8') as config_file:
            config_data = json.loads(config_file.read())
            self.task_name = config_data['names'][1]['raw_task_name']
            self.ideas_name = config_data['names'][1]['raw_ideas_name']
            self.thema_name = config_data['names'][1]['raw_theme_name']
            self.score_name = config_data['names'][1]['raw_score_name']
            self.test_task_name = config_data['names'][0]['task_name']
            self.test_theme_name = config_data['names'][0]['theme_name']
            self.test_score_name = config_data['names'][0]['score_name']
            self.test_ideas_name = config_data['names'][0]['ideas_name']

    def get_data(self,
                 test: Optional[Dict[str, Any]]) -> Dict[str,
                                                         Optional[pd.DataFrame]]:
        '''
        Формирование DataFrame с приведением имен
        к требуемому виду
        Return:
            DataFrame с подготовленными данными
        '''
        data_dict = {'X': None, 'y': None}
        if self.json_path is not None:
            with open(self.json_path, 'r', encoding='utf-8') as data_file:
                data = json.loads(data_file.read())
                standardized_data = []
                for sample in data:
                    cur_dict = dict()
                    cur_dict['system'] = None
                    cur_dict['context'] = None
                    cur_dict['true_answer'] = None
                    cur_dict[self.test_task_name] = sample[self.task_name]
                    cur_dict[self.test_ideas_name] = sample[self.ideas_name]
                    if self.thema_name in sample.keys():
                        cur_dict[self.test_theme_name] = sample[self.thema_name]
                    else:
                        cur_dict[self.test_theme_name] = None
                    if self.score_name in sample.keys():
                        cur_dict[self.test_score_name] = sample[self.score_name]
                    else:
                        cur_dict[self.test_score_name] = None
                    standardized_data.append(cur_dict)
                data_dict['X'] = pd.DataFrame.from_dict(standardized_data)
        else:
            x_dict = copy.deepcopy(test['X'])
            if 'system' not in x_dict.keys():
                x_dict['system'] = None
            if 'context' not in x_dict.keys():
                x_dict['context'] = None
            if 'true_answer' not in x_dict.keys():
                x_dict['true_answer'] = None

            if self.test_task_name not in x_dict.keys():
                x_dict[self.test_task_name] = x_dict[self.task_name]
                if self.test_task_name != self.task_name:
                    del x_dict[self.task_name]

            if self.test_ideas_name not in x_dict.keys():
                x_dict[self.test_ideas_name] = x_dict[self.ideas_name]
                if self.test_ideas_name != self.ideas_name:
                    del x_dict[self.ideas_name]

            if self.thema_name in x_dict.keys():
                if self.test_theme_name not in x_dict.keys():
                    x_dict[self.test_theme_name] = x_dict[self.thema_name]
                    if self.test_theme_name != self.ideas_name:
                        del x_dict[self.thema_name]
            else:
                x_dict[self.test_theme_name] = None
            if self.score_name in x_dict.keys():
                if self.test_score_name not in x_dict.keys():
                    x_dict[self.test_score_name] = x_dict[self.score_name]
                    if self.test_score_name != self.score_name:
                        del x_dict[self.score_name]
            else:
                x_dict[self.test_score_name] = None
            x_table = pd.DataFrame.from_dict(x_dict)
            y_table = pd.DataFrame.from_dict(test['y'])
            data_dict['X'] = pd.concat([x_table, y_table], axis=1)
        return data_dict


class BrainStormSamplerUpgraded(Sampler):
    '''
    Класс для подготовки данных к группе тестов по оценке качества генерации идей.
    На момент написания используется только в тесте по оценке зависимости стабильности качества генерации идей
    от их числа.
    '''

    def __init__(self, 
                 train: pd.DataFrame, 
                 test: pd.DataFrame) -> None:
        '''
        Инициализация конструктора.
        Parameters:
            train: Датафрейм, содержащий набор данных для калибровки порогов биннинга.
            test: Датафрейм, содержащий тестовый набор данных.
        '''
        train = self._preproc(copy.deepcopy(train))
        test = self._preproc(copy.deepcopy(test))
        super().__init__(train=train, test=test)

    def _preproc(self, 
                 var: pd.DataFrame) -> Dict[str, 
                                            Optional[pd.DataFrame]]:
        col_map = {'system_prompt': 'system', 'task': 'question', 'ideas': 'model_answer', 'theme': 'thema'}
        var = var.rename(columns=col_map)
        var[['context', 'true_answer']] = float('nan')
        var['splitted_ideas'] = var['model_answer'].apply(self._get_splitted_ideas)
        var['n_ideas'] = var['splitted_ideas'].apply(self._get_n_ideas)
        var['total_score'] = var['per_idea_score'].apply(self._get_total_score)
        return {'X': var, 'y': None}

    def _get_splitted_ideas(self, 
                            model_answer: str) -> List[str]:
        '''
        Функция для разбиения идей через регулярные выражения.
        Parameters:
            model_answer: Сгенерированные моделью идеи
        Return:
            Список разделенных идей
        '''
        re_ideas_split_pattern = r'(\\n\\n)?№\d[\.:]'
        ideas = re.sub(re_ideas_split_pattern, '|<|', model_answer)
        return [idea.strip() for idea in ideas.split('|<| ') if len(idea)]

    def _get_n_ideas(self, 
                     sep_ideas: List[str]) -> int:
        '''
        Функция получения полного числа идей в рамках одного запроса.
        Parameters:
            sep_ideas: Список идей
        Return:
            Число идей
        '''
        return len(sep_ideas)

    def _get_total_score(self, 
                         per_idea_score: List[int]) -> float:
        '''
        Функция подсчета совокупной дискретной асессорской оценки в рамках одного запроса.
        Parameters:
            per_idea_score: Список с дискретными оценками для каждой идеи в рамках запроса.
        Return:
            Совокупная оценка
        '''
        n_relevant_ideas = sum([1 for score in per_idea_score if score > 0])
        weight = n_relevant_ideas / len(per_idea_score)
        return round(weight * sum(per_idea_score), 4)
