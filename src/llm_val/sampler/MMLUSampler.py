import re
import pandas as pd
import typing as tp
from llm_val.sampler.Sampler import Sampler


class MMLUSampler(Sampler):
    '''
    Пример реализации sampler'а для MMLU датасетов.
    '''
    def __init__(self, 
                 test: pd.DataFrame):
        '''
        test: Тестовый набор данных
        '''
        test = self._preproc(test)
        super().__init__(test=test)


    def _preproc(self, var: pd.DataFrame) -> tp.Dict[str, pd.DataFrame]:
        X = pd.DataFrame()
        y = pd.DataFrame()
        X['system'] = None
        X['question'] = var['query'].apply(self._parse_question)
        X['context'] = var['query'].apply(self._parse_context)
        X['thema'] = None
        X['answer_options'] = var['query'].apply(self._parse_answer)
        X['true_answer'] = var['true_answer'].apply(self._parse_true_answer)
        X['raw_prompt'] = var.apply(lambda row: self._parse_all_prompt(row), axis=1)
        X['model_answer'] = var['answer'].apply(self._parse_model_answer)
        X['few_shot'] = var.get('few_shot', [None for _ in range(X.shape[0])])
        y['metric_value'] = (X['true_answer'] == X['model_answer']).astype(int)
        return {'X': X, 'y': y}

    
    def _parse_all_prompt(self, row: pd.Series):
        '''
        Парсит весь запрос, который целиком будет передаваться в модель.
        Parameters:
            row: Строка датасета
        Return:
            Запрос в модель
        '''
        prompt = row['query']
        return prompt


    def _parse_question(self, text: str) -> str:
        '''
        Парсит вопрос из запроса
        Parameters:
            text: Запрос, в котором содержится вопрос
        Return:
            Текст вопроса
        '''
        pattern = r'Вопрос:\s*(.+?)(?=\nВарианты ответов:)'
        question_match = re.search(pattern, text, re.DOTALL)
        question = question_match.group(1) if question_match else None
        return question


    def _parse_true_answer(self, true_answ: str) -> str:
        '''
        Парсит правильные ответы
        Parameters:
            true_answ: Строка с правильными ответами
        Return:
            Буква правильного ответа
        '''
        replace_dict = {"['A']": 'A',
                        "['B']": 'B',
                        "['C']": 'C',
                        "['D']": 'D',
                        "['E']": 'E'}
        return replace_dict[true_answ]


    def _parse_model_answer(self, text: str) -> str:
        '''
        Парсит ответ модели
        Parameters:
            text: Строка с ответом модели
        Return:
            Буква правильного ответа
        '''
        pattern = r'^\[?([A-Z])\]?'
        model_match = re.search(pattern, text, re.MULTILINE)
        model_answ = model_match.group(1) if model_match else None
        return model_answ


    def _parse_context(self, text: str) -> tp.List[str]:
        '''
        Парсит контекст из запроса
        Parameters:
            text: Запрос, в котором содержится контекст
        Return:
            Список с текстами контекста
        '''
        pattern = r'Контекст:?\n(.*?)Вопрос:'
        context_match = re.search(pattern, text, re.DOTALL)
        context = context_match.group(1) if context_match else None
        context_list = [i for i in context.split('\n') if len(i.strip()) != 0]
        return context_list


    def _parse_answer(self, text: str) -> tp.Dict[str, str]:
        '''
        Парсит варианты ответов из запроса
        Parameters:
            text: Запрос, в котором содержится варианты ответов
        Return:
            Словарь, где ключ - буква, а значение - ответ
        '''
        pattern = r'Варианты ответов:\s*(.+?)(?=\nТвой ответ:)'
        answer_match = re.search(pattern, text, re.DOTALL)
        answer = answer_match.group(1) if answer_match else None
        answer_list = [i for i in answer.split('\n') if len(i.strip()) != 0]
        answer_dict = {self._parse_model_answer(i): self._parse_answer_text(i) for i in answer_list}
        return answer_dict


    def _parse_answer_text(self, text: str):
        '''
        Парсит текст ответа
        Parameters:
            text: Вариант ответа
        Return:
            Текст вопроса
        '''
        pattern = r'^\[?[A-Z][\)\]]?\s*(.*)'
        answer_match = re.search(pattern, text)
        answer = answer_match.group(1) if answer_match else None
        return answer
            