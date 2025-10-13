from typing import List

import requests
from constants import HF_API_URL


def request_hf_model(model_name: str, task: str, data: List[str], hf_secret_key: str) -> requests.Response:
    return requests.post(
        f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/{task}",
        headers={
            'Authorization': f'Bearer {hf_secret_key}',
            'Content-Type': 'application/json',
        },
        json={
            'inputs': data,
            'options': {
                'wait_for_model': True,
            },
        },
    )
