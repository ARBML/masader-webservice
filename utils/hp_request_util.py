import requests
import os

API_TOKEN = os.environ['HF_SECRET_KEY']
API_URL = 'https://api-inference.huggingface.co/pipeline/feature-extraction/'
headers = {'Authorization': f'Bearer {API_TOKEN}'}


def hf_request(model, data):
    return requests.post(API_URL + model,
                             headers=headers,
                             json=
                             {
                              'inputs': data,
                              'options': {'wait_for_model': True}}
                             )
