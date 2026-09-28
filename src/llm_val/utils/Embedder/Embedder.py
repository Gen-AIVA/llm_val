import typing as tp
from abc import ABC, abstractmethod


class Embedder(ABC):
    """
    Абстрактный класс для классов-образователей эмбеддеров.
    """
    def __init__(self):
        self.text_preprocessor = None
        self.tokenizer = None

    def set_text_preprocessor(self, text_preprocessor):
        self.text_preprocessor = text_preprocessor

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def get_embedding(self, content: tp.List[str]) -> tp.Tuple[
                                                        tp.List[float],
                                                        tp.List[
                                                            tp.List[float]
                                                        ]]:
        """
        Абстрактный метод.
        Ожидается, что класс-наследник реализует
        подсчет эмбеддингов для каждого текста
        из списка
        """
        raise NotImplementedError


    @abstractmethod
    def embed_documents(self, content: tp.List[tp.List[str]]) -> tp.List[tp.List[float]]:
        '''
        Абстрактный метод.
        Ожидается, что класс-наследник реализует подсчет эмбеддингов для каждого текста из списка
        '''
        raise NotImplementedError


    @abstractmethod
    def embed_query(self, content: tp.List[str]) -> tp.List[float]:
        '''
        Абстрактный метод.
        Ожидается, что класс-наследник реализует подсчет эмбеддингов для каждого query
        '''
        raise NotImplementedError
