import string
import numpy as np
import typing as tp

from llm_val.utils.letters_mappings import bf_mapping, ss_mapping, translit_mapping


def symbol_deletion(text_list: tp.List[tp.Any],
                    index: int) -> tp.List[tp.Any]:
    '''
    Функция для удаления элемента из списка
    Parameters:
        text_list: Список (символов) 
        index: Индекс удаляемого элемента
    Return:
        Список без указанного элемента
    '''
    text_list.pop(index)
    return text_list


def symbol_swap(text_list: tp.List[tp.Any],
                index: int) -> tp.List[tp.Any]:
    '''
    Функция для перестановки соседних элементов из списка
    Parameters:
        text_list: Список (символов) 
        index: Индекс первого переставляемого элемента
    Return:
        Список с переставленными элементами
    '''
    if index + 1 < len(text_list):
        text_list[index], text_list[index + 1] = text_list[index + 1], text_list[index]
    return text_list


def symbol_insert(text_list: tp.List[str],
                  index: int) -> tp.List[str]:
    '''
    Функция для вставки близкого по клавиатуре символа рядом с выбранным  
    Parameters:
        text_list: Список (символов) 
        index: Индекс позиции для вставки
    Return:
        Список со вставленным элементом
    '''
    symbol = text_list[index].lower()
    if symbol in bf_mapping:
        candidates = list(bf_mapping[symbol])
        new_symbol = np.random.choice(candidates)
        if text_list[index].isupper():
            new_symbol = new_symbol.upper()
        text_list.insert(index, new_symbol)
    return text_list


def symbol_butter_finger(text_list: tp.List[str],
                         index: int) -> tp.List[str]:
    '''
    Функция для замены символа на близкий (на клавиатуре)
    Parameters:
        text_list: Список (символов) 
        index: Индекс заменяемого символа
    Return:
        Список с измененным символом
    '''
    symbol = text_list[index].lower()
    if symbol in bf_mapping:
        candidates = list(bf_mapping[symbol])
        new_symbol = np.random.choice(candidates)
        if text_list[index].isupper():
            new_symbol = new_symbol.upper()
        text_list[index] = new_symbol
    return text_list


def symbol_special(text_list: tp.List[str],
                   index: int) -> tp.List[str]:
    '''
    Функция для замены символа на соответствующую ему комбинацию спец. символов
    Parameters:
        text_list: Список (символов) 
        index: Индекс изменяемого символа
    Return:
        Список с измененным символом
    '''
    symbol = text_list[index].upper()
    if symbol in ss_mapping:
        candidates = list(ss_mapping[symbol])
        new_symbol = np.random.choice(candidates)
        text_list[index] = new_symbol
    return text_list


def symbol_translit(text_list: tp.List[str],
                    index: int) -> tp.List[str]:
    '''
    Функция для замены символа на соответствующую ему транслитерацию
    Parameters:
        text_list: Список (символов) 
        index: Индекс изменяемого символа
    Return:
        Список с измененным символом
    '''
    symbol = text_list[index].upper()
    if symbol in translit_mapping:
        candidates = list(translit_mapping[symbol])
        new_symbol = np.random.choice(candidates)
        text_list[index] = new_symbol
    return text_list


def human_augmentations(text: str,
                        rng: tp.Any,
                        p: float=0.02,
                        use_deletion: bool=False,
                        use_swap: bool=False,
                        use_insert: bool=False,
                        use_bf: bool=False,
                        use_special_symbol: bool=False,
                        use_translit_symbol: bool=False) -> str:
    '''
    Функция для аугментации строки
    Parameters:
        text: Строка
        rng: Датчик случайных чисел
        p: Вероятность аугментации для каждого символа
        use_deletion: Если True, то удаляются случайные символы
        use_swap: Если True, то переставляются случайные символы
        use_insert: Если True, то вставляются случайные символы
        use_bf: Если True, то случайные символы заменяются на близкие (на клавиатуре)
        use_special_symbol: Если True, то случайные символы заменяются на соответствующие им комбинации спец. символов
        use_translit_symbol: Если True, то случайные символы заменяются на соответствующую им транслитерацию
    Return:
        Аугментированная строка
    '''
    aug_functions = [symbol_deletion, symbol_swap, symbol_insert, symbol_butter_finger, symbol_special, symbol_translit]
    flag_list =  [use_deletion, use_swap, use_insert, use_bf, use_special_symbol, use_translit_symbol]
    aug_functions = [func for idx, func in enumerate(aug_functions) if flag_list[idx]]
    text_list = list(text)
    aug_idx = []
    pass_fine = 0
    for i, symbol in enumerate(text_list):
        if rng.random() + pass_fine < p:
            if symbol == ' ' or symbol in string.punctuation:
                pass_fine -= 1
            else:
                aug_idx.append(i)
                pass_fine = 0
    delta = 0
    for i in aug_idx:
        augmentation_function = np.random.choice(aug_functions)
        prev_len = len(text_list)
        text_list = augmentation_function(text_list, i + delta)
        delta += len(text_list) - prev_len
    return ''.join(text_list)


def batch_augmentations(batch: tp.List[str],
                        rng: tp.Any,
                        p: float=0.02,
                        use_deletion: bool=True,
                        use_swap: bool=True,
                        use_insert: bool=True,
                        use_bf: bool=True,
                        use_special_symbol: bool=False,
                        use_translit_symbol: bool=False) -> tp.Iterable[str]:
    '''
    Функция для аугментации набора строк
    Parameters:
        batch: Набор строк
        rng: Датчик случайных чисел
        p: Вероятность аугментации для каждого символа
        use_deletion: Если True, то удаляются случайные символы
        use_swap: Если True, то переставляются случайные символы
        use_insert: Если True, то вставляются случайные символы
        use_bf: Если True, то случайные символы заменяются на близкие (на клавиатуре)
        use_special_symbol: Если True, то случайные символы заменяются 
            на соответствующие им комбинации спец. символов
        use_translit_symbol: Если True, то случайные символы заменяются
            на соответствующую им транслитерацию
    Return:
        Аугментированный набор строк
    '''
    rngs = rng.spawn(len(batch))
    return [human_augmentations(text=text,
                                rng=rngs[i],
                                p=p,
                                use_deletion=use_deletion,
                                use_swap=use_swap,
                                use_insert=use_insert,
                                use_bf=use_bf,
                                use_special_symbol=use_special_symbol,
                                use_translit_symbol=use_translit_symbol) for i, text in enumerate(batch)]
