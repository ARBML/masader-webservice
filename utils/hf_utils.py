from typing import List

import requests

from constants import HF_API_URL


def request_hf_model(model_name: str, modelURL: str, data: List[str], hf_secret_key: str) -> requests.Response:
    return requests.post(
        f'{HF_API_URL}{model_name}{modelURL}',
        headers={'Authorization': f'Bearer {hf_secret_key}'},
        json={
            'inputs': data,
            'options': {
                'wait_for_model': True,
            },
        },
    )
