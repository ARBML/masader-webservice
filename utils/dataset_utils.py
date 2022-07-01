from typing import Dict, List, Set, Tuple, Union

from redis import Redis
from datasets import Dataset, DownloadMode, load_dataset

from constants import SUBSETS_FEATURES

from utils.common_utils import identity, multi_map
from utils.embeddings_utils import get_masader_embeddings
from utils.clusters_utils import get_masader_clusters


def load_masader_dataset(db: Redis) -> Tuple[List[Dict[str, Union[str, int]]], Dict[str, List[Union[str, int]]]]:
    masader = load_dataset(
        'arbml/masader',
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
        ignore_verifications=True,
    )['train']

    tags = get_features_tags(masader)
    masader = list(masader)

    embeddings = get_masader_embeddings(masader, db)
    clusters, reduced_embeddings = get_masader_clusters(embeddings)

    for dataset, dataset_cluster, dataset_reduced_embeddings in zip(masader, clusters, reduced_embeddings):
        dataset['Cluster'] = dataset_cluster
        dataset['Embeddings'] = dataset_reduced_embeddings

    return masader, tags


def get_features_tags(masader: Dataset) -> Dict[str, List[Union[str, int]]]:
    tags: Dict[str, Union[Set[str], List[Union[str, int]]]] = dict()

    for feature in masader.features:
        if feature == 'Subsets':
            tags = process_subsets_feature(tags, masader['Subsets'])
        elif feature == 'Dialect':
            tags = process_dialect_feature(tags, masader['Dialect'])
        elif feature == 'Tasks':
            tags = process_tasks_feature(tags, masader['Tasks'])
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


def process_subsets_feature(
    tags: Dict[str, Union[Set[str], List[Union[str, int]]]],
    subsets_feature: Dict[str, str],
) -> Dict[str, Union[Set[str], List[Union[str, int]]]]:
    for element in SUBSETS_FEATURES:
        tags[f'Subsets:{element}'] = set()

    for subsets in subsets_feature:
        for subset in subsets:
            for element in SUBSETS_FEATURES:
                try:
                    if element == 'Dialect':
                        tags[f'Subsets:{element}'].update(
                            list(
                                map(
                                    extract_country_from_dialect_feature,
                                    subset[element].split(','),
                                ),
                            ),
                        )
                    else:
                        tags[f'Subsets:{element}'].add(subset[element])
                except KeyError:
                    pass

    for element in SUBSETS_FEATURES:
        tags[f'Subsets:{element}'] = sorted(tags[f'Subsets:{element}'])

    return tags


def process_dialect_feature(
    tags: Dict[str, Union[Set[str], List[Union[str, int]]]],
    dialect_feature: List[str],
) -> Dict[str, Union[Set[str], List[Union[str, int]]]]:
    tags['Dialect'] = set()

    for dialects in dialect_feature:
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

    return tags


def process_tasks_feature(
    tags: Dict[str, Union[Set[str], List[Union[str, int]]]],
    tasks_feature: List[str],
) -> Dict[str, Union[Set[str], List[Union[str, int]]]]:
    tags['Tasks'] = set()

    for tasks in tasks_feature:
        tags['Tasks'].update(
            list(
                filter(
                    identity,
                    map(str.strip, tasks.split(',')),
                ),
            ),
        )

    tags['Tasks'] = sorted(tags['Tasks'])

    return tags
