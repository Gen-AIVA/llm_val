import math
import nltk
import typing as tp
import pandas as pd


def preprocess_pkl_input(input_list: tp.List[str],
                         chunk_size: int=2048) -> tp.List[str]:
    '''
    Предобрабатывает список файлов, хранящие словари в формате .pkl
    Parameters:
        input_list: Список путей к файлам
        chunk_size: Максимальная длина строки после обработки
    Return:
        Список строк готовых для проведения теста
    '''
    final_input_list = []
    for file_name in input_list:
        file = pd.read_pickle(file_name)
        chunk = ''
        for item in file.items():
            long_string = item[1]
            sent_text = nltk.sent_tokenize(long_string, language='russian')
            for sent in sent_text:
                if len(chunk) + len(sent) <= chunk_size:
                    chunk += sent
                else:
                    if len(sent) > chunk_size:
                        chunk += sent
                        chunks = math.ceil(len(chunk) // chunk_size)
                        sublist = [chunk[i: i + chunk_size] for i in range(0, chunks * chunk_size, chunk_size)]
                        final_input_list.extend(sublist[:-1])
                        chunk = sublist[-1]
                    else:
                        final_input_list.append(chunk)
                        chunk = sent
    if chunk != '':
        final_input_list.append(chunk)
    return final_input_list
