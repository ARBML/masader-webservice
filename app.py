from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os
import json
import requests
import pandas as pd
from flask import Flask, jsonify
from datasets import DownloadMode, load_dataset

app = Flask(__name__)

API_TOKEN = os.environ['HF_SECRET_KEY']
API_URL = "https://api-inference.huggingface.co/models/"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def load_masader_dataset_as_dict():
    return list(load_dataset(
        'arbml/masader',
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
    )['train'])


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
    masader = get_masader_dataset_as_dict()
    embeddings = get_embeddings_data()
    cluster = get_cluster_data()

    return jsonify(f'The datasets updated successfully! The current number of available datasets is {len(masader)}.')


# -----------Clusters-------------

@app.route('/cluster')
def cluster_req():
    return cluster


def get_cluster_data():
    clustering = KMeans(n_clusters=15).fit(tsne_data)
    return jsonify(pd.DataFrame(clustering.labels_.reshape(len(tsne_data), 1)).to_json('clusters.json', orient='split'))


# -----------Embeddings-------------

@app.route('/embeddings')
def embeddings_req():
    return embeddings


def get_embeddings_data():
    df = pd.DataFrame.from_dict(masader)
    df.columns.values[0] = "No."
    df.columns.values[1] = 'Name'
    df_main_set = df[~df['No.'].isnull()]
    df_main_set.loc[df_main_set['Abstract'].isnull(), 'Abstract'] = ""

    d_tasks = df_main_set[~df_main_set['Tasks'].isnull()]['Tasks']
    d_tasks = [t for i, t in enumerate(d_tasks.values.tolist())]

    df_main_set.loc[df_main_set['Abstract'].isnull(), 'Abstract'] = df_main_set.loc[
        df_main_set['Abstract'].isnull(), 'Description']
    d_abstracts = df_main_set['Abstract']
    d_abstracts = [d_tasks[i] + t.strip() for i, t in enumerate(d_abstracts.values.tolist())]

    return embeddings_data(sentence_transformer_request(d_abstracts))


def sentence_transformer_request(d_abstracts):
    data = json.dumps({
        'sentences': d_abstracts,
        'source_sentence': "",
        'options': {'wait_for_model': True}
    })

    response = requests.request("POST", API_URL + "sentence-transformers/all-MiniLM-L6-v2", headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def embeddings_data(abstract_embeddings):
    model = TSNE(n_components=2, random_state=0)
    tsne_data = model.fit_transform(abstract_embeddings)
    return jsonify(pd.DataFrame(tsne_data - tsne_data.min()))


# ------------------------


print('Downloading the dataset, embeddings, and clusters...')
masader = load_masader_dataset_as_dict()
embeddings = get_embeddings_data()
cluster = get_cluster_data()
