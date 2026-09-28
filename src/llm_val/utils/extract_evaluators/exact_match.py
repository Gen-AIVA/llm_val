import os
import spacy
import string
import pymorphy3


class ExactMatch:
    def __init__(self,
                 lowercase: bool=True,
                 lemmatize: bool=True):
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.nlp = spacy.load(os.environ.get("ru_spacy_sm", None))

    def evaluate_response(self, 
                          response: str, 
                          reference: str)-> int:
        if self.lowercase:
            response = response.lower()
            reference = reference.lower()
        if self.lemmatize:
            response = ' '.join(list(lemmatize(response)))
            reference = ' '.join(list(lemmatize(reference)))
        return int(response == reference)


def list_words(sentences):
    words = []
    for s in sentences:
        for w in s:
            w = str(w)
            if w not in string.punctuation:
                words.append(w)
    return words


def normalize_words(word_list):
    normalized_words = []
    for word in word_list:
        parsed_word = pymorphy3.MorphAnalyzer().parse(word)[0]
        normal_form = parsed_word.lexeme[0][0].replace('ё', 'е')
        normalized_words.append(normal_form)
    return normalized_words


def lemmatize(text):
    STOP_WORDS = spacy.lang.ru.stop_words.STOP_WORDS
    document = self.nlp(text)
    sentences = list(document.doc.sents)
    words = list_words(sentences)
    words = normalize_words(words)
    return set(words) - set(STOP_WORDS)
