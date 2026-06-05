from typing import Any, Dict, List


def as_text(value: Any) -> str:
    """Flatten a record value to a plain string (lists are space-joined)."""
    if isinstance(value, (list, tuple)):
        return ' '.join(str(item) for item in value)

    return '' if value is None else str(value)


def dict_filter(dictionary: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    if not keys:
        return dictionary

    return {key: value for key, value in dictionary.items() if key in keys}


def multi_map(iterable, function, *other):
    if other:
        return multi_map(map(function, iterable), *other)

    return map(function, iterable)


def identity(inputs):
    return inputs
