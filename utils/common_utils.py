from typing import Any, Dict, List


def dict_filter(dictionary: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    if len(keys) == 1 and keys[0] == 'all':
        return dictionary

    return {key: value for key, value in dictionary.items() if key in keys}


def multi_map(iterable, function, *other):
    if other:
        return multi_map(map(function, iterable), *other)

    return map(function, iterable)


def identity(inputs):
    return inputs
