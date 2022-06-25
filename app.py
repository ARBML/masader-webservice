import random

from flask import Flask, jsonify
from datasets import DownloadMode, load_dataset

app = Flask(__name__)


def load_masader_dataset_as_dict():
    return list(
        load_dataset(
            'arbml/masader',
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
        )['train']
    )


print('Downloading the dataset...')
masader = load_masader_dataset_as_dict()


@app.route('/datasets', defaults={'index': None})
@app.route('/datasets/<index>')
def datasets(index: str):
    global masader

    if random.random() <= 0.1:
        print('Re-downloading the dataset...')
        masader = load_masader_dataset_as_dict()

    if index:
        index = int(index)

        if 1 <= index <= len(masader):
            return jsonify(masader[index - 1])
        else:
            return jsonify(f'Dataset index is out of range, the index should be between 1 and {len(masader)}.'), 404
    else:
        return jsonify(masader)
