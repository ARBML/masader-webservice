from typing import Dict, List, Tuple, Union

from datasets import Dataset, DownloadMode, load_dataset

from constants import SUBSETS_FEATURES


def load_masader_dataset() -> Tuple[List[Dict[str, Union[str, int]]], Dict[str, List[Union[str, int]]]]:
    masader = load_dataset(
        'arbml/masader',
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
        ignore_verifications=True,
    )['train']

    return list(masader), get_features_tags(masader)


def get_features_tags(masader: Dataset) -> Dict[str, List[Union[str, int]]]:
    tags = dict()

    for feature in masader.features:
        if feature == 'Subsets':
            for subsets_feature in SUBSETS_FEATURES:
                tags[f'Subsets:{subsets_feature}'] = set()

            for subsets in masader['Subsets']:
                for subset in subsets:
                    for subsets_feature in SUBSETS_FEATURES:
                        try:
                            tags[f'Subsets:{subsets_feature}'].add(subset[subsets_feature])
                        except KeyError:
                            pass

            for subsets_feature in SUBSETS_FEATURES:
                tags[f'Subsets:{subsets_feature}'] = sorted(tags[f'Subsets:{subsets_feature}'])
        else:
            tags[feature] = sorted(set(masader[feature]))

    return tags
