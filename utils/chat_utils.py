import json
import logging
import re

from typing import Any, Dict, Iterator, List, Optional

import redis
import requests

from constants import (
    CATALOGUE_QUERY_RESULT_NAME,
    CHAT_FOLLOWUP_SYSTEM_PROMPT,
    CHAT_SITE_NAME,
    CHAT_SITE_URL,
    CHAT_SOURCES_MARKER,
    CHAT_SYSTEM_PROMPT,
)
from utils.common_utils import as_text
from utils.token_usage import TokenUsage, record_usage


logger = logging.getLogger(__name__)

# In-memory fallback catalogue used when Redis is unavailable (e.g. local dev).
_FALLBACK_RECORDS: Optional[List[Dict[str, Any]]] = None


def _load_fallback_records() -> List[Dict[str, Any]]:
    """Load the catalogue straight from the HF source when Redis is down.

    Embeddings/clusters are skipped (BM25 only needs the metadata), and each
    record gets the same 1-based `Id` used everywhere else.
    """
    global _FALLBACK_RECORDS

    if _FALLBACK_RECORDS is None:
        logger.info('loading fallback catalogue from HF source')
        from datasets import VerificationMode, load_dataset

        masader = load_dataset(
            'utils/masader',
            trust_remote_code=True,
            verification_mode=VerificationMode.NO_CHECKS,
        )['train']

        records = list(masader)
        for index, record in enumerate(records):
            record['Id'] = index + 1

        _FALLBACK_RECORDS = records

    return _FALLBACK_RECORDS


def get_masader_records(db: 'redis.Redis') -> Optional[List[Dict[str, Any]]]:
    """Return the catalogue from Redis, falling back to the HF source."""
    try:
        raw_masader = db.get('masader')
    except redis.exceptions.RedisError:
        logger.warning('Redis unavailable; trying fallback catalogue')
        raw_masader = None

    if raw_masader:
        records = json.loads(raw_masader)
        logger.info('loaded %d datasets from Redis', len(records))
        return records

    try:
        return _load_fallback_records()
    except Exception:
        logger.exception('failed to load fallback catalogue')
        return None


_REFINEMENT_RE = re.compile(
    r'\b(?:such|those|these|them|they|their|it|its|the ones?|above|mentioned|'
    r'earlier|same|similar|that|the same)\b',
    re.IGNORECASE,
)

_LIST_INTENT_RE = re.compile(
    r'\b(?:list|which|show|name|names|examples?|give me|some of)\b',
    re.IGNORECASE,
)

_COUNT_QUESTION_RE = re.compile(
    r'^how many\s+datasets?\s+(.+)$',
    re.IGNORECASE,
)

_COUNT_FRAMING_RE = re.compile(
    r'\b(?:how many|count(?: the)?|number of)\s+',
    re.IGNORECASE,
)


def _criteria_from_user_turn(turn: str) -> str:
    """Keep filter criteria from a prior turn; drop count/list framing."""
    text = turn.strip()
    match = _COUNT_QUESTION_RE.match(text)
    if match:
        return f'datasets {match.group(1)}'

    stripped = _COUNT_FRAMING_RE.sub('', text).strip()
    return stripped or text


def build_retrieval_query(messages: List[Dict[str, str]], max_turns: int = 4) -> str:
    """Build a retrieval query from recent user turns.

    Use the latest turn by default so a new question (e.g. "which Egyptian
    datasets…") is not polluted by an earlier one ("how many…"). When the latest
    turn is an underspecified follow-up ("list 3 of such datasets", "how was it
    collected?"), walk backward and prepend prior user turns until the head of
    the query is self-contained.
    """
    user_turns = [
        str(message.get('content', '')).strip()
        for message in messages
        if message.get('role') == 'user' and message.get('content')
    ]

    if not user_turns:
        return ''

    parts = [user_turns[-1]]
    for turn in reversed(user_turns[:-1]):
        if len(parts) >= max_turns:
            break
        if _REFINEMENT_RE.search(parts[0]):
            parts.insert(0, _criteria_from_user_turn(turn))
        else:
            break

    query = ' '.join(parts)
    if _LIST_INTENT_RE.search(parts[-1]):
        query = _COUNT_FRAMING_RE.sub('', query).strip() or query

    logger.info('built retrieval query from %d user turn(s)', len(parts))

    return query


def dataset_to_card(dataset: Dict[str, Any]) -> Dict[str, Any]:
    tasks = dataset.get('Tasks', '')
    if isinstance(tasks, str):
        tasks = [task.strip() for task in tasks.split(',') if task.strip()]

    description = as_text(dataset.get('Description', ''))

    return {
        # `Id` matches the identifier used by GET /datasets/<id> and the card page.
        'id': dataset.get('Id'),
        'name': dataset.get('Name'),
        'year': dataset.get('Year'),
        'dialect': dataset.get('Dialect'),
        'tasks': tasks,
        'link': dataset.get('Link'),
        'description': description[:240],
    }


def _format_subsets(subsets: Any) -> str:
    """Render the per-subset breakdown (name, dialect, volume) as one line."""
    if not isinstance(subsets, list) or not subsets:
        return ''

    parts = []
    for subset in subsets:
        if not isinstance(subset, dict):
            continue

        name = as_text(subset.get('Name', '')).strip()
        dialect = as_text(subset.get('Dialect', '')).strip()
        volume = as_text(subset.get('Volume', '')).strip()
        unit = as_text(subset.get('Unit', '')).strip()

        label = name or dialect or 'subset'
        meta = []
        if dialect and dialect != name:
            meta.append(dialect)
        if volume:
            meta.append(f"{volume} {unit}".strip())

        parts.append(f"{label} ({', '.join(meta)})" if meta else label)

    return '; '.join(parts)


# Internal / verbose fields excluded from the LLM context.
_CONTEXT_EXCLUDE = {'Embeddings', 'Cluster', 'Abstract'}


def _context_block(datasets: List[Dict[str, Any]]) -> str:
    blocks = []
    for dataset in datasets:
        attributes = []

        for key, value in dataset.items():
            if key in _CONTEXT_EXCLUDE:
                continue

            if key == 'Dialect Subsets':
                formatted = _format_subsets(value)
                if formatted:
                    attributes.append(('subsets', formatted))
                continue

            text = as_text(value).strip()
            if text:
                attributes.append((key.lower().replace(' ', '_'), text))

        if not attributes:
            continue

        first_key, first_value = attributes[0]
        lines = [f"- {first_key}: {first_value}"]
        lines.extend(f"  {key}: {value}" for key, value in attributes[1:])
        blocks.append('\n'.join(lines))

    return '\n'.join(blocks)


def build_llm_messages(
    messages: List[Dict[str, str]],
    datasets: List[Dict[str, Any]],
    *,
    cite_sources: bool = True,
) -> List[Dict[str, str]]:
    # Some models reject more than one system message, so combine the
    # instructions and the dataset context into a single leading system prompt.
    system_prompt = CHAT_SYSTEM_PROMPT if cite_sources else CHAT_FOLLOWUP_SYSTEM_PROMPT
    system_content = system_prompt + '\n\nRelevant datasets:\n' + _context_block(datasets)

    # Keep only conversational turns from the client; never trust an injected
    # system role mid-conversation.
    conversation = [
        message
        for message in messages
        if message.get('role') in ('user', 'assistant') and message.get('content')
    ]

    return [{'role': 'system', 'content': system_content}, *conversation]


def _sse(payload: Dict[str, Any]) -> str:
    return f'data: {json.dumps(payload)}\n\n'


def _format_openrouter_error(response: 'requests.Response') -> str:
    """Build a human-readable message from an OpenRouter error response."""
    detail = ''
    try:
        body = response.json()
        detail = (body.get('error') or {}).get('message') or body.get('message') or ''
    except (ValueError, AttributeError):
        detail = (response.text or '').strip()

    message = f'LLM API error {response.status_code}'
    if detail:
        message += f': {detail}'

    if response.status_code == 429:
        retry_after = response.headers.get('Retry-After')
        hint = (
            ' (rate limit / quota reached. Add credits or set MODEL_NAME '
            'to a model you have access to.'
        )
        if retry_after:
            hint += f' Retry after {retry_after}s.'
        message += hint + ')'

    return message


def _is_catalogue_meta(dataset: Dict[str, Any]) -> bool:
    """True for synthetic SQL aggregate rows (not real catalogue datasets)."""
    return (
        dataset.get('Id') == 0
        or str(dataset.get('Name', '')).strip() == CATALOGUE_QUERY_RESULT_NAME
    )


def _cited_datasets(datasets: List[Dict[str, Any]], answer: str) -> List[Dict[str, Any]]:
    """Keep only datasets whose name is actually mentioned in the answer.

    The model is instructed to refer to datasets by their exact name, so we can
    surface cards only for the ones it cited (rather than every BM25 candidate).
    Matching is case-insensitive and bounded by non-alphanumeric characters to
    avoid false hits on short names.
    """
    answer_lower = answer.lower()
    cited = []

    for dataset in datasets:
        if _is_catalogue_meta(dataset):
            continue

        name = str(dataset.get('Name', '')).strip()
        if not name:
            continue

        pattern = r'(?<![a-z0-9])' + re.escape(name.lower()) + r'(?![a-z0-9])'
        if re.search(pattern, answer_lower):
            cited.append(dataset)

    return cited


def merge_datasets(*groups: List[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Concatenate dataset groups, de-duplicating by name and preserving order.

    Earlier groups take precedence, so callers should pass higher-priority
    datasets (e.g. ones already under discussion) first; later duplicates are
    dropped and the result is truncated to `limit` if given.
    """
    seen = set()
    merged: List[Dict[str, Any]] = []

    for group in groups:
        for dataset in group:
            key = str(dataset.get('Name', '')).strip().lower()
            if not key or key in seen:
                continue

            seen.add(key)
            merged.append(dataset)

    return merged if limit is None else merged[:limit]


def datasets_cited_in_history(records: List[Dict[str, Any]], messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Return catalogue datasets that prior assistant turns referenced by name.

    The chat endpoint is stateless and re-runs retrieval every turn, so a
    follow-up like "what is common in such datasets?" can retrieve a different
    BM25 set than the datasets already under discussion. Pinning the
    previously-named datasets back into context stops the model from claiming
    they "do not exist in the provided context".
    """
    assistant_text = '\n'.join(
        str(message.get('content', ''))
        for message in messages
        if message.get('role') == 'assistant' and message.get('content')
    )

    if not assistant_text:
        return []

    cited = _cited_datasets(records, assistant_text)
    logger.info('found %d dataset(s) cited in prior turns', len(cited))

    return cited


# Tolerant matcher for the sources marker: variable percent counts, optional
# surrounding markdown/whitespace, optional trailing colon. This prevents the
# marker from leaking into the visible answer if the model formats it loosely.
_SOURCES_RE = re.compile(r'\s*[*_#>\-]*\s*%{2,}\s*SOURCES\s*%*\s*:?', re.IGNORECASE)

# A trailing, incomplete sources marker (e.g. truncated output ending in
# '%%%SOUR' or '%%%'). Stripped so a half-written marker never leaks. Requires
# two or more '%' so a legitimate trailing single '%' (e.g. "90%") is preserved
# in the final flush.
_SOURCES_PARTIAL_RE = re.compile(
    r'\s*[*_#>\-]*\s*%{2,}\s*(?:S(?:O(?:U(?:R(?:C(?:E(?:S)?)?)?)?)?)?)?\s*$',
    re.IGNORECASE,
)

# Streaming-only variant that also holds back a trailing single '%'. While
# streaming, a lone '%' at the stable-prefix boundary may be the first
# character of a forming '%%%SOURCES%%%' marker (the rest still in the held-back
# tail), so it must not be emitted yet. A genuine prose '%' is only delayed for
# a token or two and is flushed once more non-'%' text arrives.
_SOURCES_STREAM_PARTIAL_RE = re.compile(
    r'\s*[*_#>\-]*\s*%+\s*(?:S(?:O(?:U(?:R(?:C(?:E(?:S)?)?)?)?)?)?)?\s*$',
    re.IGNORECASE,
)

# Characters to trim from each parsed source name (markdown, punctuation, %).
_SOURCE_STRIP = ' \t\r\n.,;:*_#%`"\'()[]'

# Reasoning / "thinking" blocks emitted by some models (e.g. Qwen). We disable
# thinking in the request, but strip any leftover blocks as a safety net.
_THINK_BLOCK_RE = re.compile(r'<think>.*?</think>\s*', re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r'<think>', re.IGNORECASE)
# A trailing, still-forming think tag, e.g. '<', '<thi', '</thin' (no closing '>'
# yet). Held back so a partial tag is never emitted mid-stream.
_THINK_PARTIAL_RE = re.compile(r'</?(?:t(?:h(?:i(?:n(?:k)?)?)?)?)?$', re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from the model output."""
    text = _THINK_BLOCK_RE.sub('', text)

    # Drop any still-open (unclosed) reasoning block.
    open_match = _THINK_OPEN_RE.search(text)
    if open_match:
        text = text[:open_match.start()]

    # Hold back a partially-streamed think tag at the very end.
    return _THINK_PARTIAL_RE.sub('', text)


def _datasets_from_sources(datasets: List[Dict[str, Any]], sources_text: str) -> List[Dict[str, Any]]:
    """Map the model's explicit SOURCES list to retrieved dataset records.

    Only names that exactly match a retrieved dataset (case-insensitive) are
    kept, which filters out incidental mentions (sources/derivations) and any
    hallucinated names.
    """
    names = {
        part.strip(_SOURCE_STRIP).strip().lower()
        for part in re.split(r'[,\n]', sources_text)
        if part.strip(_SOURCE_STRIP).strip()
    }

    by_name = {str(d.get('Name', '')).strip().lower(): d for d in datasets}

    seen = set()
    result = []
    for name in names:
        dataset = by_name.get(name)
        if (
            dataset is not None
            and name not in seen
            and not _is_catalogue_meta(dataset)
        ):
            seen.add(name)
            result.append(dataset)

    return result


def stream_chat(
    messages: List[Dict[str, str]],
    datasets: List[Dict[str, Any]],
    api_key: str,
    api_url: str,
    model_name: str,
    *,
    cite_sources: bool = True,
) -> Iterator[str]:
    """Yield Server-Sent-Events: token events, then a single datasets event.

    The frontend re-renders the answer bubble on each token, so dataset cards
    are emitted only after the answer is complete. Only datasets the model
    actually referenced by name are surfaced as cards (skipped on follow-ups
    when ``cite_sources`` is False).
    """
    logger.info(
        'streaming chat completion via %s (cite_sources=%s)',
        model_name,
        cite_sources,
    )
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': CHAT_SITE_URL,
        'X-Title': CHAT_SITE_NAME,
    }
    payload = {
        'model': model_name,
        'messages': build_llm_messages(messages, datasets, cite_sources=cite_sources),
        'stream': True,
        # Ask OpenAI-compatible servers (vLLM, OpenRouter, ...) to emit usage
        # in the final stream chunk(s).
        'stream_options': {'include_usage': True},
        # Disable "thinking" / reasoning output (Qwen3 & compatible servers).
        'chat_template_kwargs': {'enable_thinking': False},
    }

    try:
        with requests.post(
            api_url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=120,
        ) as response:
            if response.status_code >= 400:
                logger.warning('chat completion API error %s', response.status_code)
                yield _sse({'type': 'error', 'message': _format_openrouter_error(response)})
                return

            full = ''
            emitted = 0
            latest_usage: Optional[TokenUsage] = None
            # Hold back enough trailing chars to cover a partially-streamed
            # marker or reasoning tag (e.g. "</think>") before it is detected.
            hold = (len(CHAT_SOURCES_MARKER) + 16) if cite_sources else 16

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith('data:'):
                    continue

                data = line[len('data:'):].strip()
                if data == '[DONE]':
                    break

                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    continue

                usage = TokenUsage.from_api(parsed)
                if usage is not None:
                    latest_usage = usage

                choices = parsed.get('choices') or []
                if not choices:
                    continue

                delta = choices[0].get('delta') if isinstance(choices[0], dict) else None
                if not isinstance(delta, dict):
                    continue

                token = delta.get('content')
                if not token:
                    continue

                full += token

                # Work on a "stable" prefix (hold back the tail) so partial
                # reasoning tags / markers forming across token boundaries are
                # not emitted. Strip reasoning, then cut at the sources marker.
                stable = full[: max(0, len(full) - hold)]
                visible = _strip_reasoning(stable)

                if cite_sources:
                    match = _SOURCES_RE.search(visible)
                    if match:
                        limit = match.start()
                    else:
                        # Hold back a trailing partial marker, including a lone '%'
                        # that may be the marker's first character (e.g. '%%%SOUR'
                        # or just '%').
                        limit = len(_SOURCES_STREAM_PARTIAL_RE.sub('', visible))
                else:
                    limit = len(visible)

                if limit > emitted:
                    yield _sse({'type': 'token', 'text': visible[emitted:limit]})
                    emitted = limit

            if latest_usage is not None:
                record_usage(latest_usage, 'chat')
            else:
                logger.warning('chat stream finished without usage data from the API')
    except requests.RequestException as error:
        logger.warning('chat completion request failed: %s', error)
        yield _sse({'type': 'error', 'message': str(error)})
        return

    visible = _strip_reasoning(full)

    if cite_sources:
        # Drop a trailing incomplete marker (truncated generation) before matching.
        visible = _SOURCES_PARTIAL_RE.sub('', visible)
        match = _SOURCES_RE.search(visible)

        if match is None:
            # Model didn't emit the marker: flush the remainder and fall back to
            # scanning the prose for dataset names.
            if len(visible) > emitted:
                yield _sse({'type': 'token', 'text': visible[emitted:]})
            cited = _cited_datasets(datasets, visible)
        else:
            if match.start() > emitted:
                yield _sse({'type': 'token', 'text': visible[emitted:match.start()]})
            cited = _datasets_from_sources(datasets, visible[match.end():])
    else:
        visible = _SOURCES_PARTIAL_RE.sub('', visible)
        match = _SOURCES_RE.search(visible)
        if match is not None:
            visible = visible[:match.start()]
        if len(visible) > emitted:
            yield _sse({'type': 'token', 'text': visible[emitted:]})
        cited = []

    logger.info('answer complete: %d dataset card(s) cited', len(cited))
    yield _sse({'type': 'datasets', 'datasets': [dataset_to_card(d) for d in cited]})
    yield 'data: [DONE]\n\n'
