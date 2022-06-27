from typing import Any, Dict, List

from datasets import DownloadMode, load_dataset


def load_masader_dataset():
    return list(
        load_dataset(
            'arbml/masader',
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
            ignore_verifications=True,
        )['train']
    )


def dict_filter(dictionary: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    if len(keys) == 1 and keys[0] == 'all':
        return dictionary

    return {key: value for key, value in dictionary.items() if key in keys}
