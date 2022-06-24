import random

from flask import Flask, jsonify
from datasets import DownloadMode, load_dataset

app = Flask(__name__)

print('Downloading the dataset...')
masader = list(
    load_dataset(
        'arbml/masader',
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
    )['train']
)


@app.route('/datasets')
def datasets():
    global masader

    if random.random() <= 0.1:
        print('Re-downloading the dataset...')
        masader = list(
            load_dataset(
                'arbml/masader',
                download_mode=DownloadMode.FORCE_REDOWNLOAD,
            )['train']
        )

    return jsonify(masader)
