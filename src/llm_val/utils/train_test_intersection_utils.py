import typing as tp
from tqdm.auto import tqdm

def read_book(file_path: str) -> str:
    '''
    Читалка для train-файлов по умолчанию
    Parameters:
        file_path: Путь до файла для чтения
    Return: 
        Текст из файла
    '''
    with open(file_path, 'r') as file:
        book_text = file.read()
    return book_text


def parse_train_samples_from_path(paths: tp.Union[str, tp.List[str]],
                                  chunk_size: int=-1,
                                  verbose: bool=False) -> tp.Tuple[tp.List[str], tp.List[str]]:
    
    '''
    Преобразователь train-файлов в список текстов по умолчанию
    Parameters:
        paths: Путь/пути до файла/файлов, которые необходимо прочитать
        chunk_size: Размер чанка в словах (из расчета 1000 символов ~= 150 слов) при разбиении текста на чанки;
                    -1, если разбивать не нужно
        verbose: Показывать ли прогресс чтения файлов?
    Return:
        Tuple с текстами (чанками) и уникальными индексами для них
    '''
    if isinstance(paths, str):
        paths = [paths]
    texts, indices = [], []
    if verbose:
        print('Reading train_files...')
        iterator = tqdm(paths)
    else:
        iterator = paths
    for path in iterator:
        book_text = read_book(path)
        if chunk_size == -1:
            texts.append(book_text)
            indices.append(f'{path}')
        else:
            book_text = book_text.split()
            for chunk_idx, i in enumerate(range(0, len(book_text), chunk_size)):
                text_chunk = ' '.join(book_text[i:i+chunk_size])
                indices.append(f'{path}_chunk_{chunk_idx}')
                texts.append(text_chunk)
                
    return texts, indices