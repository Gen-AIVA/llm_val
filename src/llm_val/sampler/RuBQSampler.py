import re
import typing as tp
import pandas as pd
from llm_val.sampler.Sampler import Sampler


class RuBQSampler(Sampler):
    '''
    Пример реализации sampler'а для RuBQ датасетов.
    '''
    def __init__(self, test: pd.DataFrame):
        '''
        test: Тестовый набор данных
        '''
        test = self._preproc(test)
        super().__init__(test=test)

    def _preproc(self, var: pd.DataFrame) -> tp.Dict[str, pd.DataFrame]:
        X = pd.DataFrame()
        y = pd.DataFrame()
        X['system'] = var['system'].apply(str.strip)
        X['question'] = var['query'].apply(self._parse_question)
        X['context'] = var['query'].apply(self._parse_context)
        X['thema'] = None
        X['raw_prompt'] = var.apply(lambda row: self._parse_all_prompt(row), axis=1)
        X['model_answer'] = var['answer'].apply(str.strip)
        X['true_answer'] = var['true_answer'].apply(self._split_true_answer)
        X['few_shot'] = var.get('few_shot', [None for _ in range(X.shape[0])])
        y['metric_value'] = X.apply(lambda row: self._create_is_correct(row['model_answer'], row['true_answer']), axis=1)
        return {'X': X, 'y': y}


    def _parse_all_prompt(self, row: pd.Series):
        '''
        Парсит весь запрос, который целиком будет передаваться в модель.
        Parameters:
            row: Строка датасета
        Return:
            Запрос в модель
        '''
        prompt = row['system'] + ' ' + row['query']
        return prompt


    def _split_true_answer(self, answer: str):
        return [i.strip().lower() for i in answer.split(';')]


    def _create_is_correct(self, answer: str, true_answer: tp.List[str]):
        is_correct = False
        for i in true_answer:
            is_correct = i in answer.lower()
            if is_correct:
                break
        return int(is_correct)


    def _parse_question(self, text: str) -> str:
        '''
        Парсит вопрос из запроса
        Parameters:
            text: Запрос, в котором содержится вопрос
        Return:
            Текст вопроса
        '''
        pattern = r'\nQUESTION:(.*?)========='
        question_match = re.search(pattern, text, re.DOTALL)
        question = question_match.group(1) if question_match else None
        return question.strip('\n').strip()


    def _parse_context(self, text: str) -> tp.List[str]:
        '''
        Парсит контекст из запроса
        Parameters:
            text: Запрос, в котором содержится контекст
        Return:
            Список с текстами контекста
        '''
        pattern = r'Content:(.*?)(?:\*\*\*КОНЕЦ СТАТЬИ\*\*\*)'
        matches = re.findall(pattern, text, re.DOTALL)
        results = [match.strip('\n').strip() for match in matches]
        return results
