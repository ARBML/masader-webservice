#from sklearn.manifold import TSNE
import json
import requests
import pandas as pd
from flask import Flask, jsonify
from datasets import DownloadMode, load_dataset

app = Flask(__name__)

API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def load_masader_dataset_as_dict():
    return list(load_dataset(
            'arbml/masader',
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
        )['train'])


def load_masader_dataset_as_list():
    return list(load_masader_dataset_as_dict())

print('Downloading the dataset...')
masader = load_masader_dataset_as_dict()
masaderDict = load_masader_dataset_as_dict()


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

    print('Refreshing the dataset...')
    masader = load_masader_dataset_as_dict()

    return jsonify(f'The datasets updated successfully! The current number of available datasets is {len(masader)}.')


def embeddings_data():

    df = pd.DataFrame.from_dict(masaderDict)
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

    return embeddings_request(sentence_transformer_request(d_abstracts))


def sentence_transformer_request(d_abstracts):
    data = json.dumps({
        'sentences': d_abstracts,
        'source_sentence': "",
        'options': {'wait_for_model': True}
    })

    response = requests.request("POST", API_URL + "sentence-transformers/all-MiniLM-L6-v2", headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

def embeddings_request(abstract_embeddings):
    # model = TSNE(n_components=2, random_state=0)
    # tsne_data = model.fit_transform(abstract_embeddings)
    # a = pd.DataFrame(tsne_data - tsne_data.min())
    # print(a)
    pass

#embeddings_data()