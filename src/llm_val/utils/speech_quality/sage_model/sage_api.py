import requests
import typing as tp
from langchain.text_splitter import RecursiveCharacterTextSplitter

from llm_val.utils.lib_vars import SAGE_API_HEADERS, SAGE_API_URL

class SageAPI():
    '''
    Класс, реализующий доступ к модели Sage, исправляющей орфографические ошибки в тексте.
    Работает в sigma, mlspace.
    '''
    def __init__(self, 
                 contour: str):
        '''
        contour: Название контура. Принимает одно из двух значениий ['sigma', 'mlspace']
        '''
        self.contour = contour
        self.headers = SAGE_API_HEADERS
        self.url = SAGE_API_URL
        self.chunk_size = 999
        self.chunk_overlap = 0
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, 
                                                            chunk_overlap=self.chunk_overlap)

    def get_answer_sage(self, 
                        text: str) -> str:
        '''
        Метод исправления орфографических ошибок в тексте
        Parameters:
            text: Входной текст
        Return:
            Исправленный текст с точки зрения орфографии
        '''
        corrected_chunks = []
        for chunk in self.text_splitter.split_text(text):
            pyload = {"instances": [{"text": chunk}]}
            response = requests.post(self.url[self.contour], json=pyload, headers=self.headers)
            response.raise_for_status()
            data_dict = response.json()
            corrected_chunks.append(data_dict['predictions'])
        return " ".join(corrected_chunks)
        