import json

from multiprocessing import Process
from typing import Dict, List, Optional, Union

import redis

from flask import Flask, jsonify, request
from flask_cors import CORS

from utils.common_utils import dict_filter
from utils.dataset_utils import refresh_masader_and_tags
from utils.gh_utils import report_issue


app = Flask(__name__)
app.config.from_object('config.Config')
CORS(app)


db = redis.from_url(app.config['REDIS_URL'])
masader: Optional[List[Dict[str, Union[str, int]]]] = None
tags: Dict[str, List[Union[str, int]]] = None


@app.route('/datasets/schema')
def datasets_schema():
    global masader

    return jsonify(list(masader[0].keys()))


@app.route('/datasets')
def get_datasets():
    global masader

    page = request.args.get('page', default=1, type=int)
    size = request.args.get('size', default=len(masader), type=int)
    features = request.args.get('features', default='all', type=str).split(',')

    masader_page = masader[(page - 1) * size : page * size]

    if not masader_page:
        return jsonify('Page not found.'), 404

    return jsonify(list(map(lambda element: dict_filter(element, features), masader_page)))


@app.route('/datasets/<int:index>')
def get_dataset(index: int):
    global masader

    features = request.args.get('features', default='all', type=str).split(',')

    if not (1 <= index <= len(masader)):
        return jsonify(f'Dataset index is out of range, the index should be between 1 and {len(masader)}.'), 404

    return jsonify(dict_filter(masader[index - 1], features))


@app.route('/datasets/tags')
def get_tags():
    global tags

    features = request.args.get('features', default='all', type=str).split(',')

    return jsonify(dict_filter(tags, features))


@app.route('/refresh')
def refresh():
    global db, masader, tags

    print('Refreshing globals...')

    Process(name='refresh_globals', target=refresh_masader_and_tags, args=(db,)).start()

    masader = json.loads(db.get('masader'))
    tags = json.loads(db.get('tags'))

    return jsonify(f'The datasets updated successfully! The current number of available datasets is {len(masader)}.')


@app.route('/datasets/<int:index>/issues', methods=['POST'])
def report_card_issue(index: int):
    if not (1 <= index <= len(masader)):
        return jsonify(f'Dataset index is out of range, the index should be between 1 and {len(masader)}.'), 404

    title = request.get_json().get('title', '')
    message = request.get_json().get('message', '')

    return jsonify(report_issue(title, message))


with app.app_context():
    refresh()
