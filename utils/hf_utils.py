from typing import List

import requests


API_URL = 'https://api-inference.huggingface.co/pipeline/feature-extraction/'


def request_hf_model(model_name: str, data: List[str], hf_secret_key: str) -> requests.Response:
    return requests.post(
        f'{API_URL}{model_name}',
        headers={'Authorization': f'Bearer {hf_secret_key}'},
        json={
            'inputs': data,
            'options': {
                'wait_for_model': True,
            },
        },
    )
