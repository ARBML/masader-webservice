from typing import Dict, List, Tuple, Union

from datasets import Dataset, DownloadMode, load_dataset

from constants import SUBSETS_FEATURES
from utils.common_utils import identity, multi_map


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
                            if subsets_feature == 'Dialect':
                                tags[f'Subsets:{subsets_feature}'].update(
                                    list(
                                        map(
                                            extract_country_from_dialect_feature,
                                            subset[subsets_feature].split(','),
                                        ),
                                    ),
                                )
                            else:
                                tags[f'Subsets:{subsets_feature}'].add(subset[subsets_feature])
                        except KeyError:
                            pass

            for subsets_feature in SUBSETS_FEATURES:
                tags[f'Subsets:{subsets_feature}'] = sorted(tags[f'Subsets:{subsets_feature}'])
        elif feature == 'Dialect':
            tags['Dialect'] = set()

            for dialects in masader['Dialect']:
                tags['Dialect'].update(
                    list(
                        multi_map(
                            dialects.split(','),
                            extract_country_from_dialect_feature,
                            str.strip,
                        ),
                    ),
                )

            tags['Dialect'] = sorted(tags['Dialect'])
        elif feature == 'Tasks':
            tags['Tasks'] = set()

            for tasks in masader['Tasks']:
                tags['Tasks'].update(
                    list(
                        filter(
                            identity,
                            map(str.strip, tasks.split(',')),
                        ),
                    ),
                )

            tags['Tasks'] = sorted(tags['Tasks'])
        else:
            tags[feature] = sorted(set(masader[feature]))

    tags['Dialect'] = sorted(set(tags['Dialect'] + tags['Subsets:Dialect']))

    for feature in tags:
        try:
            tags[feature].remove('nan')
        except ValueError:
            pass

    return tags


def extract_country_from_dialect_feature(dialect_feature: str) -> str:
    country = dialect_feature.split('(')[-1].split(')')[0]

    if country == 'Modern Standard Arabic':
        return 'MSA'

    return country
