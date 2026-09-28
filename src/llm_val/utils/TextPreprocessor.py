import string
import pymorphy3
import typing as tp
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TreebankWordTokenizer


class TextPreprocessor:
    """
    Класс, реализующий предобработку текста
    """
    def __init__(self,
                 stopwords_path: str,
                 language: str='russian',
                 use_lemmatization: bool=False):
        self.stopwords_path = stopwords_path
        self.language = language

        self.morph = pymorphy3.MorphAnalyzer()
        self.use_lemmatization = use_lemmatization
        self.stemmer = SnowballStemmer(self.language)
        self.text_tokenizer = TreebankWordTokenizer()
        self.cur_stopwords = self.__read_stopwords()

    def __read_stopwords(self,
                         add_punkt: bool=True) -> tp.List[str]:
        """
        Получение стоп-слов из спец. файла

        Parameters:
            add_punkt: Флаг, поднимаемый при необходимости
                       фильтрации символов пунктуации

        Return:
            Список стоп-слов
        """
        stopwords = []
        with open(self.stopwords_path, 'r', encoding='utf-8') as sw_file:
            while True:
                cur_line = sw_file.readline()
                if not cur_line:
                    break
                stopwords.append(cur_line.strip())
        if add_punkt:
            stopwords.extend(string.punctuation)
        return stopwords

    def __is_stop_word(self,
                       word: str) -> bool:
        """
        Проверка слова на принадлежность к стоп-словам

        Parameters:
            word: Проверяемое слово

        Return:
            Принадлежность слова к списку стоп-слов
        """
        if word.lower() in self.cur_stopwords:
            return False
        return True

    def __stem_token(self,
                     token: str) -> str:
        """
        Метод для стемминга токена

        Parameters:
            token: Токен, от которого нужно получить стемму

        Return:
            Стемма переданного токена
        """
        return self.stemmer.stem(token)

    def __tokenize_text(self,
                        text: str) -> tp.List[str]:
        """
        Метод для токенизации переданного текста

        Parameters:
            text: Текст, который необходимо токенизировать

        Return:
            Список токенов
        """
        return self.text_tokenizer.tokenize(text)

    def __get_lemma(self,
                    word: str) -> str:
        """
        Метод для лемматизации переданного слова

        Parameters:
            word: слово, для которого нужно получить лемму

        Return:
            Лемма для переданного слова
        """
        return self.morph.parse(word)[0].normal_form

    def clean_text(self,
                   text: str) -> tp.List[str]:
        """
        Метод очистки текста от стоп-слов и его стемминг

        Parameters:
            text: Очищаемый текст

        Return:
            Список токенов, прошедших очищающий фильтр
        """
        tokens = self.__tokenize_text(text)
        filtered_tokens = list(map(self.__is_stop_word, tokens))
        processed_tokens = []
        for idx in range(len(filtered_tokens)):
            if filtered_tokens[idx]:
                if self.use_lemmatization:
                    processed_tokens.append(self.__get_lemma(tokens[idx]))
                else:
                    processed_tokens.append(self.__stem_token(tokens[idx]))
        return processed_tokens
