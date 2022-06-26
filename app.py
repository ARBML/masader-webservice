from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os
import requests
import pandas as pd
from flask import Flask, jsonify
from datasets import DownloadMode, load_dataset

app = Flask(_name_)

API_TOKEN = os.environ['HF_SECRET_KEY']
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def load_masader_dataset_as_dict():
    return load_dataset(
        'arbml/masader',
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
    )['train']


@app.route('/datasets', defaults={'index': None})
@app.route('/datasets/<index>')
def datasets(index: str):
    global masader

    if index:
        index = int(index)

        if 1 <= index <= len(masader):
            return jsonify(masader[index - 1])
        else:
            return jsonify(f'Dataset index is out of range, the index should be between 1 and {len(masader)}.'), 404
    else:
        return jsonify(masader)


@app.route('/datasets/refresh')
def refresh_datasets():
    global masader
    global embeddings
    global cluster

    print('Refreshing the dataset...')
    data = list(load_masader_dataset_as_dict())
    masader = data
    embeddings = get_embeddings_data(data)
    cluster = get_cluster_data()

    return jsonify(f'The datasets updated successfully! The current number of available datasets is {len(masader)}.')


# -----------Clusters-------------

@app.route('/cluster')
def cluster_req():
    return cluster


def get_cluster_data():
    global tsne_data

    clustering = KMeans(n_clusters=15).fit(tsne_data)
    return pd.DataFrame(clustering.labels_.reshape(len(tsne_data), 1)).to_json()


# -----------Embeddings-------------

@app.route('/embeddings')
def embeddings_req():
    return embeddings


def get_embeddings_data(dataset):
    desc = [desc if desc != 'nan' else "" for desc in dataset['Description']]
    tasks = [tasks if tasks != 'nan' else "" for tasks in dataset['Tasks']]
    abstracts = [abs if abs != 'nan' else "" for abs in dataset['Abstract']]

    full = [" ".join(tpl) for tpl in list(zip(*[desc, tasks, abstracts]))]

    return embeddings_data(sentence_transformer_request(full))


def sentence_transformer_request(d_abstracts):
    requested_data = None
    first = 0
    last = 0

    print(len(d_abstracts))

    while last < len(d_abstracts):

        if last + 200 < len(d_abstracts):
            last += 200
        else:
            last += len(d_abstracts) - last

        response = requests.post(API_URL + "sentence-transformers/all-MiniLM-L6-v2", headers=headers,
                                 json={"inputs": d_abstracts[first:last], "options": {"wait_for_model": True}})

        first = last

        if requested_data is None:
            requested_data = response.json()
        else:
            requested_data += response.json()

    return pd.DataFrame(requested_data)


def embeddings_data(abstract_embeddings):
    global tsne_data

    model = TSNE(n_components=2, random_state=0)
    tsne_data = model.fit_transform(abstract_embeddings)

    return pd.DataFrame(tsne_data - tsne_data.min()).to_json()


# ------------------------

print('Downloading the dataset, embeddings, and clusters...')
tsne_data = None
data = load_masader_dataset_as_dict()
masader = list(data)
embeddings = get_embeddings_data(data)
cluster = get_cluster_data()