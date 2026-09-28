import os
import json
import joblib
import typing as tp


def parse_json(path: tp.Union[str, os.PathLike]) -> tp.Dict[str, tp.Any]:
    if not isinstance(path, (str, os.PathLike)):
        raise AttributeError(f'Должен быть тип str или os.PathLike, но был получен {type(path)}')
    if not str(path).endswith('.json'):
        raise ValueError(f'Должен быть путь до .json, но был получен путь до {path}')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def parse_pkl(path: tp.Union[str, os.PathLike]) -> tp.Any:
    if not isinstance(path, (str, os.PathLike)):
        raise AttributeError(f'Должен быть тип str или os.PathLike, но был получен {type(path)}')
    if not str(path).endswith('.pkl'):
        raise ValueError(f'Должен быть путь до .pkl, но был получен путь до {path}')
    data = joblib.load(path)
    return data
