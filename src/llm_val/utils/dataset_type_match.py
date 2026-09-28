import re
import uuid
import typing as tp

import numpy as np


class Primitive:
    def __init__(self, name: str, predicate: tp.Callable[[tp.Any], bool]):
        self.name = name
        self._predicate = predicate

    def check(self, value: tp.Any) -> bool:
        return self._predicate(value)


def _check_string(v: tp.Any) -> bool:
    return isinstance(v, str)


def _check_int(v: tp.Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        result = False
    elif isinstance(v, (int, np.integer)):
        result = True
    elif isinstance(v, (float, np.floating)):
        result = bool(np.isfinite(v)) and float(v).is_integer()
    else:
        result = False
    return result


def _check_float(v: tp.Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        result = False
    else:
        result = isinstance(v, (int, float, np.integer, np.floating))
    return result


def _check_uuid(v: tp.Any) -> bool:
    if isinstance(v, uuid.UUID):
        result = True
    elif isinstance(v, str):
        try:
            uuid.UUID(v)
            result = True
        except ValueError:
            result = False
    else:
        result = False
    return result


STRING = Primitive('String', _check_string)
INT = Primitive('Int', _check_int)
FLOAT = Primitive('Float', _check_float)
UUID = Primitive('UUID', _check_uuid)


class Union:
    def __init__(self, *specs: tp.Any):
        self.specs = specs

    def check(self, value: tp.Any) -> bool:
        return any(spec.check(value) for spec in self.specs)


class ListOf:
    def __init__(self, item: tp.Any = None, 
                 length: tp.Optional[int] = None):
        self.item = item
        self.length = length

    def check(self, value: tp.Any) -> bool:
        if not isinstance(value, (list, tuple, np.ndarray)):
            result = False
        elif self.length is not None and len(value) != self.length:
            result = False
        elif self.item is not None:
            result = all(self.item.check(e) for e in value)
        else:
            result = True
        return result


def _pattern_to_regex(pattern: str) -> tp.Pattern:
    parts = re.split(r'\[[^\]]+\]', pattern)
    return re.compile('(.+)'.join(re.escape(p) for p in parts) + '$')


def _compile_schema(type_dict: tp.Dict[str, tp.Any]) -> tp.Tuple[dict, list]:
    is_pattern = lambda x: '[' in x and ']' in x
    exact = {k: v for k, v in type_dict.items() if not is_pattern(k)}
    patterns = [(_pattern_to_regex(k), v) for k, v in type_dict.items() if is_pattern(k)]
    return exact, patterns


def _resolve_spec(column: str,
                  exact: dict,
                  patterns: list) -> tp.Tuple[tp.Any, bool]:
    '''
    Сопоставление имени колонки с её типом по предкомпилированной схеме.
    '''
    result = (None, False)
    if column in exact:
        result = (exact[column], False)
    else:
        for regex, spec in patterns:
            if regex.match(column):
                result = (spec, True)
                break
    return result

X_TYPES = {
    "solution_version": STRING,
    "scenario": STRING,
    "session_id": Union(UUID, INT),
    "query_id": Union(UUID, INT, STRING),
    "input_query_count": INT,
    "input_query": STRING,
    "output_answer": STRING,
    "reference_answer": STRING,
    "assessor_id": Union(UUID, INT, STRING),

    "[metric_name]_metric": FLOAT,
    "[module_name]_input_query": STRING,
    "[module_name]_output_answer": STRING,
    "[module_name]_reference_answer": STRING,
    "[module_name]_[metric_name]_metric": FLOAT,
}

X_DIALOGUE_TYPES = {
    "solution_version": STRING,
    "scenario": STRING,
    "session_id": Union(UUID, INT),
    "dialogue": ListOf(ListOf(length=3)),
    "assessor_id": Union(UUID, INT, STRING),
    
    "[metric_name]_metric": FLOAT,
}