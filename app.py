import json
import logging
import os
import queue

from threading import Thread

import pandas as pd
import redis

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from constants import CHAT_TOP_K
from utils.chat_utils import (
    build_retrieval_query,
    datasets_cited_in_history,
    get_masader_records,
    merge_datasets,
    stream_chat,
)
from utils.common_utils import dict_filter
from utils.dataset_utils import refresh_masader_and_tags
from utils.gh_utils import create_issue
from utils.logging_utils import (
    configure_logging,
    new_turn_id,
    set_session_id,
    start_log_capture,
    stop_log_capture,
)
from utils.token_usage import commit_turn_usage, reset_turn_usage, usage_payload
from utils.retrieval import get_retriever
from utils.router import get_router


configure_logging()
logger = logging.getLogger(__name__)


def _sse(payload):
    return f'data: {json.dumps(payload)}\n\n'


app = Flask(__name__)
app.config.from_object('config.Config')
CORS(app)


db = redis.from_url(app.config['REDIS_URL'])

# Selected via the RETRIEVAL_ALGORITHM config; register new algorithms in
# utils.retrieval.RETRIEVERS to make them selectable.
retriever = get_retriever(
    app.config['RETRIEVAL_ALGORITHM'],
    api_key=app.config['API_KEY'],
    api_url=app.config['API_URL'],
    model_name=app.config['MODEL_NAME'],
    schema_url=app.config['MASADER_SCHEMA_URL'],
)

# Decides whether each turn needs a fresh retrieval (RETRIEVAL_ROUTER config).
router = get_router(
    app.config['RETRIEVAL_ROUTER'],
    api_key=app.config['API_KEY'],
    api_url=app.config['API_URL'],
    model_name=app.config['MODEL_NAME'],
)


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
        # TODO: Use fuzzy search here instead of exact match for the "Name" field
        # difflib implements a simple algorithm for this
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


@app.route('/chat', methods=['POST'])
def chat():
    api_key = app.config['API_KEY']
    api_url = app.config['API_URL']
    model_name = app.config['MODEL_NAME']

    if not api_key:
        return jsonify('Chat is not configured: missing API_KEY.'), 503

    body = request.get_json(silent=True) or {}
    messages = body.get('messages', [])
    session_id = body.get('session_id', '')

    if not isinstance(messages, list) or not messages:
        return jsonify("Request body must include a non-empty 'messages' list."), 400

    # The whole pipeline runs inside the generator so the captured logs can be
    # streamed to the UI in real time (interleaved with the answer tokens).
    def event_stream():
        new_turn_id()
        set_session_id(session_id)
        reset_turn_usage()
        log_queue = start_log_capture()

        def drain_logs():
            while True:
                try:
                    entry = log_queue.get_nowait()
                except queue.Empty:
                    break
                yield _sse({'type': 'log', 'level': entry['level'], 'message': entry['message']})

        try:
            logger.info('chat turn started: %d messages', len(messages))
            yield from drain_logs()

            query = build_retrieval_query(messages)
            records = get_masader_records(db)
            yield from drain_logs()

            if not records:
                logger.warning('catalogue unavailable (Redis down and fallback failed)')
                yield from drain_logs()
                yield _sse({'type': 'error', 'message': 'Catalogue is not available (Redis is down and the fallback source could not be loaded).'})
                return

            # Keep datasets already discussed in earlier turns in context
            # (pinned first) so follow-up questions don't lose them when fresh
            # retrieval for the new query surfaces a different set.
            pinned = datasets_cited_in_history(records, messages)
            yield from drain_logs()

            # On a follow-up about datasets already under discussion, reuse them
            # and skip a fresh retrieval. Only consult the router when there is
            # prior context to reuse; otherwise we must retrieve.
            is_followup = bool(pinned) and not router.should_retrieve(messages)
            if is_followup:
                logger.info('reusing %d pinned datasets; retrieval skipped', len(pinned))
                datasets = merge_datasets(pinned, limit=2 * CHAT_TOP_K)
            else:
                retrieved = retriever.retrieve(records, query, k=CHAT_TOP_K)
                datasets = merge_datasets(pinned, retrieved, limit=2 * CHAT_TOP_K)
                logger.info(
                    'retrieval ran: %d retrieved, %d pinned, %d in context',
                    len(retrieved), len(pinned), len(datasets),
                )
            yield from drain_logs()

            logger.info(
                'starting answer stream over %d datasets (followup=%s)',
                len(datasets),
                is_followup,
            )
            yield from drain_logs()

            for chunk in stream_chat(
                messages=messages,
                datasets=datasets,
                api_key=api_key,
                api_url=api_url,
                model_name=model_name,
                cite_sources=not is_followup,
            ):
                yield from drain_logs()
                yield chunk

            yield from drain_logs()
            session_usage = commit_turn_usage()
            yield _sse({'type': 'usage', **usage_payload(session_usage)})
            yield from drain_logs()
        finally:
            stop_log_capture()

    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/highlights')
def get_highlights():
    return jsonify({'highlights': os.environ.get('HIGHLIGHTS', '')})


@app.route('/refresh/<string:password>')
def refresh(password: str):
    print('Refreshing globals...')

    if password != app.config['REFRESH_PASSWORD']:
        return jsonify('Password is incorrect.'), 403

    def refresh_with_context():
        with app.app_context():
            refresh_masader_and_tags(db)

    Thread(name='refresh_globals', target=refresh_with_context).start()

    return jsonify('Datasets refresh process initiated successfully!')


with app.app_context():
    refresh(app.config['REFRESH_PASSWORD'])
