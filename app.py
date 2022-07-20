import json

from multiprocessing import Process

import pandas as pd
import redis

from flask import Flask, jsonify, request
from flask_cors import CORS
from utils.common_utils import dict_filter
from utils.dataset_utils import refresh_masader_and_tags


from utils.gh_utils import create_issue



app = Flask(__name__)
app.config.from_object('config.Config')
CORS(app)


db = redis.from_url(app.config['REDIS_URL'])


@app.route('/datasets/schema')
def datasets_schema():
    masader = json.loads(db.get('masader'))

    return jsonify(list(masader[0].keys()))


@app.route('/datasets')
def get_datasets():
    masader = json.loads(db.get('masader'))

    page = request.args.get('page', default=1, type=int)
    size = request.args.get('size', default=len(masader), type=int)
    features = list(filter(None, request.args.get('features', default='', type=str).split(',')))
    query = request.args.get('query', default='', type=str)

    masader_page = masader[(page - 1) * size : page * size]

    if not masader_page:
        return jsonify('Page not found.'), 404

    masader_page = pd.DataFrame(masader_page)

    if query:
        masader_page = masader_page.query(query)

    if features:
        masader_page = masader_page[features]

    return jsonify(masader_page.to_dict('records'))


@app.route('/datasets/<int:index>')
def get_dataset(index: int):
    masader = json.loads(db.get('masader'))

    features = list(filter(None, request.args.get('features', default='', type=str).split(',')))

    if not (1 <= index <= len(masader)):
        return jsonify(f'Dataset index is out of range, the index should be between 1 and {len(masader)}.'), 404

    return jsonify(dict_filter(masader[index - 1], features))


@app.route('/datasets/tags')
def get_tags():
    tags = json.loads(db.get('tags'))

    features = list(filter(None, request.args.get('features', default='', type=str).split(',')))

    return jsonify(dict_filter(tags, features))




@app.route('/datasets/<int:index>/issues', methods=['POST'])
def create_dataset_issue(index: int):
    masader = json.loads(db.get('masader'))

    if not (1 <= index <= len(masader)):
        return jsonify(f'Dataset index is out of range, the index should be between 1 and {len(masader)}.'), 404

    title = request.get_json().get('title', '')
    body = request.get_json().get('body', '')

    return jsonify({'issue_url': create_issue(f"{masader[index]['Name']}: {title}", body)})


@app.route('/refresh/<string:password>')
def refresh(password: str):
    print('Refreshing globals...')

    if password != app.config['REFRESH_PASSWORD']:
        return jsonify(f'Password is incorrect.'), 403

    Process(name='refresh_globals', target=refresh_masader_and_tags, args=(db,)).start()

    return jsonify('The datasets updated successfully!')


with app.app_context():
    refresh(app.config['REFRESH_PASSWORD'])
