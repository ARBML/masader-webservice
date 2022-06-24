import random

from flask import Flask, jsonify
from datasets import DownloadMode, load_dataset

app = Flask(__name__)

print('Downloading the dataset...')
masader = load_dataset('arbml/masader')


@app.route('/datasets')
def datasets():
    global masader

    if random.random() <= 0.1:
        print('Re-downloading the dataset...')
        masader = load_dataset('arbml/masader', download_mode=DownloadMode.FORCE_REDOWNLOAD)

    return jsonify(list(masader['train']))
