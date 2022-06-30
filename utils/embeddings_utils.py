import json

from typing import Dict, List, Tuple, Union

from redis import Redis
from flask import current_app as app

from constants import HF_EMBEDDINGS_MODEL, HF_REQUEST_BATCH_SIZE, HF_FEATURE_EXTRACTION_TASK

from utils.hf_utils import request_hf_model


def get_masader_embeddings(masader: List[Dict[str, Union[str, int]]], db: Redis) -> List[List[float]]:
    cached_embedding, new_prompts = get_cached_embeddings_and_new_prompts(masader, db)
    new_prompts_embeddings = compute_embeddings(list(new_prompts.values()), HF_EMBEDDINGS_MODEL)

    for (index, new_prompt), new_prompt_embeddings in zip(new_prompts.items(), new_prompts_embeddings):
        db.set(new_prompt, json.dumps(new_prompt_embeddings))
        cached_embedding[index] = new_prompt_embeddings

    return list(zip(*sorted(cached_embedding.items(), key=lambda element: element[0])))[1]


def get_cached_embeddings_and_new_prompts(
    masader: List[Dict[str, Union[str, int]]],
    db: Redis,
) -> Tuple[Dict[int, List[float]], Dict[int, str]]:
    cached_embeddings = dict()
    new_prompts = dict()

    for index, dataset in enumerate(masader):
        dataset_prompt = build_dataset_prompt(dataset)

        dataset_cached_embeddings = db.get(dataset_prompt)

        if dataset_cached_embeddings:
            cached_embeddings[index] = json.loads(dataset_cached_embeddings)
        else:
            new_prompts[index] = dataset_prompt

    return cached_embeddings, new_prompts


def compute_embeddings(texts: List[str], model_name: str) -> List[List[float]]:
    embeddings = list()

    for i in range(0, len(texts), HF_REQUEST_BATCH_SIZE):
        response = request_hf_model(
            model_name,
            HF_FEATURE_EXTRACTION_TASK,
            texts[i : i + HF_REQUEST_BATCH_SIZE],
            app.config['HF_SECRET_KEY'],
        )

        embeddings.extend(response.json())

    return embeddings


def build_dataset_prompt(dataset: Dict[str, Union[str, int]]) -> str:
    return f"{dataset['Description']} {dataset['Tasks']} {dataset['Abstract']}"
