import json

from typing import Dict, List, Set, Union

from redis import Redis
from datasets import Dataset, DownloadMode, load_dataset

from constants import FILE_FEATURE, MASADER_ID_MAP_KEY, MASADER_KEY, MASADER_TAGS_KEY, SUBSETS_FEATURES

from utils.common_utils import identity, multi_map
from utils.embeddings_utils import get_masader_embeddings
from utils.clusters_utils import get_masader_clusters
from datasets import VerificationMode


def refresh_masader_and_tags(db: Redis) -> None:
    masader = load_dataset(
        'utils/masader',
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
        trust_remote_code=True,
        verification_mode=VerificationMode.NO_CHECKS,
    )['train']
    print(masader)
    tags = get_features_tags(masader)
    masader = list(masader)

    embeddings = get_masader_embeddings(masader, db)
    clusters, reduced_embeddings = get_masader_clusters(embeddings)

    _assign_persistent_ids(db, masader)

    for dataset, dataset_cluster, dataset_reduced_embeddings in zip(
        masader,
        clusters,
        reduced_embeddings,
    ):
        dataset['Cluster'] = dataset_cluster
        dataset['Embeddings'] = dataset_reduced_embeddings

    masader.sort(key=lambda d: d['Id'])

    db.set(MASADER_KEY, json.dumps(masader))
    db.set(MASADER_TAGS_KEY, json.dumps(tags))


def _assign_persistent_ids(db: Redis, masader: List[Dict]) -> None:
    """Assign stable integer IDs keyed by the dataset's source filename.

    The mapping ({filename: int_id}) is persisted in Redis so a dataset keeps
    its ID across refreshes even as new datasets are added or others removed.
    Removed datasets leave gaps (their IDs are retired, never reused).
    """
    id_map: Dict[str, int] = json.loads(db.get(MASADER_ID_MAP_KEY) or '{}')
    next_id = max(id_map.values(), default=0) + 1

    masader.sort(key=lambda d: d.get(FILE_FEATURE, ''))

    for dataset in masader:
        key = dataset.get(FILE_FEATURE)
        if not key:
            continue

        if key in id_map:
            dataset['Id'] = id_map[key]
        else:
            dataset['Id'] = next_id
            id_map[key] = next_id
            next_id += 1

    db.set(MASADER_ID_MAP_KEY, json.dumps(id_map))


def get_features_tags(masader: Dataset) -> Dict[str, List[Union[str, int]]]:
    tags: Dict[str, Union[Set[str], List[Union[str, int]]]] = dict()

    for feature in masader.features:
        if feature == FILE_FEATURE:
            continue
        if feature == 'Dialect Subsets':
            tags = process_subsets_feature(tags, masader['Dialect Subsets'])
        elif feature == 'Dialect':
            tags = process_dialect_feature(tags, masader['Dialect'])
        elif feature == 'Tasks':
            tags = process_tasks_feature(tags, masader['Tasks'])
        else:
            tags[feature] = sorted(v for v in set(masader[feature]) if v is not None)

    tags['Dialect'] = sorted(
        v for v in set(tags['Dialect'] + tags['Subsets:Dialect']) if v is not None
    )

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
        if not isinstance(dialects, str):
            continue
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
        if not isinstance(tasks, str):
            continue
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
