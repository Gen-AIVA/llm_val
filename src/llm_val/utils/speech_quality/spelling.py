import torch
import jiwer
import typing as tp
from tqdm import tqdm
from jiwer import wer
from Levenshtein import distance

def spelling_score_api(text_list: tp.List[str], 
                       model: tp.Any, 
                       normalization: bool=True, 
                       cache_api: tp.Dict[str, str]={}) -> tp.Tuple[float, float]:
    '''
    Вычисляет среднее качество орфографии для списка строк
        (word error rate между поданным списком и референсом, полученным с помощью модели для исправления ошибок) 
    Parameters:
        text_list: Список строк
        model: Класс-обертка над API моделью для исправления ошибок, имеет метод get_answer_sage
               True, если используем нормировку в расстоянии Левенштейна, False  в противном случае
        cache_api: Словарь для кэширования работы модели исправления орфографии
    Return:
        Значение среднего качества орфографии - WER, Levenshein distance
    '''
    reference = []
    hypothesis = []
    transformation = jiwer.Compose([jiwer.ToLowerCase(),
                                    jiwer.RemoveMultipleSpaces(),
                                    jiwer.RemovePunctuation(),
                                    jiwer.ReduceToSingleSentence(), 
                                    jiwer.Strip()])
    for sentence in tqdm(text_list):
        sentence_preproc = transformation(sentence)
        if sentence not in cache_api:
            res = model.get_answer_sage(sentence_preproc)
            cache_api.update({sentence: res})
        else:
            res = cache_api[sentence]
        reference.append(res)
        hypothesis.append(sentence_preproc)
    ans_transformation = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=' ')])
    return wer(reference, hypothesis), levenshtein_distance(reference, hypothesis, normalization=normalization)


def levenshtein_distance(reference: tp.List[str], 
                         hypothesis: tp.List[str],
                         normalization: bool=True) -> float:
    '''
    Вычисляет расстояние Левенштейна между списками текстов.
    Parameters:
        reference: Список текстов, исправленных моделью
        hypothesis: Список исходных текстов
        normalization: True, если используем нормировку, False  в противном случае
    Return:
        Расстояние Левенштейна
    '''
    distances = [distance(corrected, original) for corrected, original in zip(reference, hypothesis)]
    if normalization:
        distances = [d/len(original) if len(original) > 0 else 0 for d, original in zip(distances, hypothesis)]
    avg_distances = float(sum(distances))
    if normalization:
        avg_distances /= len(distances)
    return avg_distances