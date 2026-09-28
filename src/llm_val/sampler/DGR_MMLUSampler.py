import re
import pandas as pd
import typing as tp
from llm_val.sampler.Sampler import Sampler


class DGR_MMLUSampler(Sampler):
    '''
    Реализации sampler'а ЦМРБИТК для MMLU датасетов (контрагент - ДГР).
    Датасет должен иметь 3 колонки [system_prompt, query, true_answer].
    Запрос query имеет вид:
    "Вопрос... [A] Ответ1 [B] Ответ2 [C] Ответ3 [D] Ответ4 [E] Ответ5"
    '''
    def __init__(self,
                 test: pd.DataFrame):
        '''
        test: Тестовый набор данных
        '''
        test = self._preproc(test)
        super().__init__(test=test)


    def _preproc(self, var: pd.DataFrame) -> tp.Dict[str, pd.DataFrame]:
        X, y = pd.DataFrame(), pd.DataFrame()
        X['system'] = var['system_prompt'].apply(self._parse_system_prompt)
        X['question'] = var['query'].apply(self._parse_question)
        X['context'] = None
        X['thema'] = None
        X['answer_options'] = var['query'].apply(self._parse_answer)
        X['true_answer'] = var['true_answer'].apply(self._parse_true_answer)
        X['raw_prompt'] = var.apply(lambda row: self._parse_all_prompt(row), axis=1)
        X['model_answer'] = None
        y['metric_value'] = None
        return {'X': X, 'y': y}


    def _parse_system_prompt(self, prompt: str) -> str:
        '''
        Парсит системный промпт
        Parameters:
            prompt: Системный промпт
        Return:
            Текст системного промпта без лишних пробелов
        '''
        return str(prompt).strip()
    

    def _parse_all_prompt(self, row: pd.Series):
        '''
        Парсит весь запрос, который целиком будет передаваться в модель.
        Parameters:
            row: Строка датасета
        Return:
            Запрос в модель
        '''
        prompt = row['system_prompt'] + ' ' + row['query']
        return prompt


    def _parse_question(self, text: str) -> str:
        '''
        Парсит вопрос из запроса
        Parameters:
            text: Запрос, в котором содержится вопрос
        Return:
            Текст вопроса
        '''
        question = text.split("[A]", 1)[0].strip()
        return question


    def _parse_true_answer(self, true_answ: str) -> str:
        '''
        Парсит правильные ответы
        Parameters:
            true_answ: Строка с правильными ответами
        Return:
            Буква правильного ответа
        '''
        dict_to_latin = str.maketrans({
            'а': 'A', 'в': 'B', 'с': 'C', 'е': 'E'
        })
        true_answ = str(true_answ).lower().translate(dict_to_latin).upper()
        replace_dict = {"[A]": 'A',
                        "[B]": 'B',
                        "[C]": 'C',
                        "[D]": 'D',
                        "[E]": 'E'}
        if true_answ in replace_dict:
            return replace_dict[true_answ]
        else:
            return true_answ


    def _parse_determined_answer(self, text: str, answer_label: str) -> str:
        '''
        Из запроса парсит ответ под определенной буквой
        Parameters:
            text: Запрос, в котором содержится вопрос и варианты ответов
            answer_label: Буква, определяющая ответ из пула предложенных в запросе
        Return:
            Ответ, находящийся под буквой answer_label
        '''
        pattern = rf'\[{re.escape(answer_label)}\]\s*(.*?)(?=\s*\[[A-Z]\]|\Z)'
        answer = re.search(pattern, text, re.DOTALL).group(1).strip()
        return answer


    def _parse_answer(self, text: str) -> tp.Dict[str, str]:
        '''
        Парсит варианты ответов из запроса
        Parameters:
            text: Запрос, в котором содержится варианты ответов
        Return:
            Словарь, где ключ - буква, а значение - ответ
        '''
        letters = ['A', 'B', 'C', 'D', 'E']
        answer_dict = {}
        for v in letters:
            try: 
                answer_dict[v] = self._parse_determined_answer(text, v)
            except: 
                continue
        return answer_dict
    