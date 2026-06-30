"""Pluggable dataset retrieval strategies for the chat endpoint.

A `Retriever` turns the full catalogue plus a free-text query into the top-`k`
most relevant dataset records. The chat flow depends only on this interface, so
new retrieval methods (dense/embedding, hybrid, ...) can be added by
subclassing `Retriever` and implementing `retrieve` — no changes to the chat
pipeline required.
"""

import json
import logging
import re
import sqlite3
import threading

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Type

import requests
from rank_bm25 import BM25Okapi

from constants import (
    CATALOGUE_QUERY_RESULT_NAME,
    CHAT_TOP_K,
    MASADER_SCHEMA_URL,
    SYSTEM_SQL_PROMPT,
)
from utils.common_utils import as_text
from utils.token_usage import TokenUsage, record_usage


logger = logging.getLogger(__name__)

Record = Dict[str, Any]


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower(), flags=re.UNICODE)


class Retriever(ABC):
    """Strategy interface for ranking catalogue datasets against a query."""

    @abstractmethod
    def retrieve(self, records: List[Record], query: str, k: int = CHAT_TOP_K) -> List[Record]:
        """Return up to `k` records most relevant to `query`, best first."""
        raise NotImplementedError


class BM25Retriever(Retriever):
    """Lexical retrieval over dataset metadata using Okapi BM25.

    The index is built lazily and cached on the instance, keyed by a cheap
    signature of the catalogue, so it is only rebuilt when the catalogue
    actually changes (e.g. after a /refresh).
    """

    # Metadata fields concatenated into each dataset's searchable document.
    DEFAULT_FIELDS: Sequence[str] = (
        'Name',
        'Description',
        'Abstract',
        'Tasks',
        'Domain',
        'Dialect',
        'License',
        'Year',
        'Form',
        'Provider',
    )

    def __init__(self, fields: Optional[Sequence[str]] = None) -> None:
        self._fields = tuple(fields) if fields is not None else tuple(self.DEFAULT_FIELDS)
        self._signature: Optional[str] = None
        self._index: Optional[BM25Okapi] = None
        self._records: List[Record] = []

    def _document(self, dataset: Record) -> str:
        return ' '.join(as_text(dataset.get(field, '')) for field in self._fields)

    @staticmethod
    def _signature_of(records: List[Record]) -> str:
        if not records:
            return '0'

        return f"{len(records)}:{records[0].get('Name')}:{records[-1].get('Name')}"

    def _ensure_index(self, records: List[Record]) -> None:
        signature = self._signature_of(records)
        if signature != self._signature:
            logger.info('building BM25 index over %d datasets', len(records))
            self._index = BM25Okapi([_tokenize(self._document(record)) for record in records])
            self._records = records
            self._signature = signature

    def retrieve(self, records: List[Record], query: str, k: int = CHAT_TOP_K) -> List[Record]:
        logger.info('BM25 retrieve: k=%d, query=%r', k, query[:120])
        self._ensure_index(records)
        if self._index is None:
            return []

        scores = self._index.get_scores(_tokenize(query))
        ranked = sorted(zip(scores, self._records), key=lambda pair: pair[0], reverse=True)
        results = [record for score, record in ranked[:k] if score > 0]
        logger.info('BM25 retrieve: %d datasets above threshold', len(results))

        return results


# Internal catalogue fields that are not dataset metadata.
_SQL_EXCLUDE_FIELDS = frozenset({'Embeddings', 'Cluster'})


def _sql_column_name(key: str) -> str:
    """Map catalogue field names to SQLite-safe identifiers (spaces -> underscores)."""
    return key.replace(' ', '_')


def _record_field(record: Record, sql_column: str) -> Any:
    if sql_column in record:
        return record[sql_column]

    spaced = sql_column.replace('_', ' ')
    return record.get(spaced)


def _serialize_cell(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], dict):
            return json.dumps(value, ensure_ascii=False)
        return ', '.join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _format_schema_for_prompt(
    schema: Dict[str, Any],
    sql_columns: Optional[Sequence[str]] = None,
) -> str:
    lines = ['- id (int): dataset identifier']
    columns = sorted(sql_columns) if sql_columns is not None else sorted(schema)

    for column in columns:
        meta = schema.get(column)
        if not isinstance(meta, dict):
            meta = schema.get(column.replace('_', ' '))
        if not isinstance(meta, dict):
            lines.append(f'- {column} (str)')
            continue

        answer_type = meta.get('answer_type', 'str')
        description = meta.get('description', '')
        line = f'- {column} ({answer_type})'
        if description:
            line += f': {description}'

        options = meta.get('options')
        if options:
            line += f" Options: {', '.join(str(option) for option in options)}"

        lines.append(line)

    return '\n'.join(lines)


# LLMs often treat list columns as JSON; rewrite to LIKE on comma-separated text.
_JSON_EACH_IN_RE = re.compile(
    r"'([^']+)'\s+IN\s+\(\s*SELECT\s+(?:value|json_each\.value)\s+FROM\s+"
    r'json_each\(\s*"?([A-Za-z_][A-Za-z0-9_ ]*)"?\s*\)\s*\)',
    re.IGNORECASE,
)


def _sanitize_sql(sql: str) -> str:
    """Rewrite common invalid patterns (e.g. json_each on text columns) to LIKE."""
    def _to_like(match: re.Match) -> str:
        column = _sql_column_name(match.group(2).strip())
        value = match.group(1)
        return f'{column} LIKE \'%{value}%\''

    return _JSON_EACH_IN_RE.sub(_to_like, sql)


_SELECT_FROM_DATASETS_RE = re.compile(
    r'^(\s*SELECT\s+)(.+?)(\s+FROM\s+["`]?DATASETS["`]?)',
    re.IGNORECASE | re.DOTALL,
)

_AGGREGATE_SQL_RE = re.compile(
    r'\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(|\bGROUP\s+BY\b',
    re.IGNORECASE,
)

_LIST_INTENT_RE = re.compile(
    r'\b(?:list|which|show|name|names|examples?|give me|some of)\b',
    re.IGNORECASE,
)


def _normalize_retrieval_select(sql: str) -> str:
    """Retrieval only needs ids; drop extra SELECT columns the model may add."""
    if _AGGREGATE_SQL_RE.search(sql):
        return sql

    match = _SELECT_FROM_DATASETS_RE.match(sql)
    if match is None:
        return sql

    return f'{match.group(1)}id, Name{match.group(3)}{sql[match.end():]}'


def _normalize_column_names(sql: str, sql_columns: Sequence[str]) -> str:
    """Rewrite space-separated catalogue names to underscore SQLite columns."""
    normalized = sql
    for column in sql_columns:
        spaced = column.replace('_', ' ')
        if spaced == column:
            continue

        normalized = normalized.replace(f'"{spaced}"', column)
        normalized = normalized.replace(f'`{spaced}`', column)
        normalized = re.sub(rf'\b{re.escape(spaced)}\b', column, normalized)

    return normalized


def _prepare_sql(sql: str, sql_columns: Sequence[str]) -> str:
    sql = _normalize_retrieval_select(sql)
    sql = _normalize_column_names(sql, sql_columns)
    return _sanitize_sql(sql)


def _extract_sql(text: str) -> str:
    text = (text or '').strip()
    match = re.search(r'```(?:sql)?\s*(.*?)```', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip().strip('`').rstrip(';').strip()


def _aggregate_context_record(
    sql: str,
    columns: List[str],
    rows: List[tuple],
    catalogue_size: int,
) -> Record:
    """Wrap aggregate SQL output as a single context block for the answer LLM."""
    formatted_rows = []
    for row in rows:
        parts = [f'{column}={value}' for column, value in zip(columns, row)]
        formatted_rows.append(', '.join(parts))

    results_text = '; '.join(formatted_rows) if formatted_rows else 'no rows'
    description = (
        f'This is not a dataset entry; it is the result of a statistical SQL query '
        f'over the Masader catalogue. Query: {sql}. Result: {results_text}. '
        f'The DATASETS table currently holds {catalogue_size} dataset records.'
    )

    return {
        'Id': 0,
        'Name': CATALOGUE_QUERY_RESULT_NAME,
        'Description': description,
    }


def _is_safe_select(sql: str) -> bool:
    normalized = sql.strip().rstrip(';').strip()
    if not normalized or ';' in normalized:
        return False

    upper = normalized.upper()
    if not upper.startswith('SELECT'):
        return False

    forbidden = (
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE',
        'ATTACH', 'DETACH', 'PRAGMA', 'REPLACE', 'TRUNCATE',
    )
    return not any(word in upper for word in forbidden)


@dataclass
class _ThreadDbState:
    """Per-thread SQLite catalogue; connections are not shared across workers."""
    signature: Optional[str] = None
    conn: Optional[sqlite3.Connection] = None
    records_by_id: Dict[int, Record] = field(default_factory=dict)
    sql_columns: List[str] = field(default_factory=list)


def _fetch_masader_schema(schema_url: str) -> Dict[str, Any]:
    response = requests.post(schema_url, data={'name': 'ar'}, timeout=30)
    response.raise_for_status()
    schema = response.json()
    if isinstance(schema, dict) and schema.keys() == {'detail'}:
        raise RuntimeError(schema['detail'])

    return schema


class SQLRetriever(Retriever):
    """LLM-generated SQL retrieval over an in-memory SQLite catalogue.

    The model receives the Masader schema (from mextract) and the user query,
    returns a SELECT over the DATASETS table, and matching rows are mapped back
    to full dataset records.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        schema_url: str = MASADER_SCHEMA_URL,
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url
        self._model_name = model_name
        self._schema_url = schema_url
        self._timeout = timeout
        self._schema: Optional[Dict[str, Any]] = None
        self._thread_db = threading.local()
        self._bm25_fallback = BM25Retriever()

    def _db_state(self) -> _ThreadDbState:
        state = getattr(self._thread_db, 'state', None)
        if state is None:
            state = _ThreadDbState()
            self._thread_db.state = state

        return state

    def _load_schema(self) -> Dict[str, Any]:
        if self._schema is None:
            logger.info('loading Masader schema from %s', self._schema_url)
            self._schema = _fetch_masader_schema(self._schema_url)

        return self._schema

    def _ensure_db(self, records: List[Record]) -> None:
        state = self._db_state()
        signature = BM25Retriever._signature_of(records)
        if signature == state.signature and state.conn is not None:
            return

        logger.info('building SQLite catalogue over %d datasets', len(records))
        if state.conn is not None:
            state.conn.close()

        state.conn = sqlite3.connect(':memory:')
        state.records_by_id = {}
        state.signature = signature
        state.sql_columns = []

        if not records:
            return

        sql_keys = set()
        for record in records:
            for key in record:
                if key in _SQL_EXCLUDE_FIELDS or key == 'Id':
                    continue
                sql_keys.add(_sql_column_name(key))

        columns = ['id', *sorted(sql_keys)]
        state.sql_columns = columns
        column_defs = ', '.join(f'"{column}" TEXT' for column in columns)
        state.conn.execute(f'CREATE TABLE DATASETS ({column_defs})')

        quoted_columns = ', '.join(f'"{column}"' for column in columns)
        placeholders = ', '.join('?' for _ in columns)
        insert_sql = f'INSERT INTO DATASETS ({quoted_columns}) VALUES ({placeholders})'

        for record in records:
            record_id = record.get('Id')
            if record_id is None:
                continue

            row = [record_id]
            row.extend(
                _serialize_cell(_record_field(record, column))
                for column in sorted(sql_keys)
            )
            state.conn.execute(insert_sql, row)
            state.records_by_id[int(record_id)] = record

        state.conn.commit()

    def _generate_sql(self, query: str, schema: Dict[str, Any]) -> Optional[str]:
        if not self._api_key or not self._api_url or not self._model_name:
            logger.warning('SQL retriever missing API credentials')
            return None

        system_content = SYSTEM_SQL_PROMPT.format(
            schema=_format_schema_for_prompt(schema, self._db_state().sql_columns),
        )
        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': self._model_name,
            'messages': [
                {'role': 'system', 'content': system_content},
                {'role': 'user', 'content': query},
            ],
            'stream': False,
            'temperature': 0,
        }

        logger.info('generating SQL via %s', self._model_name)
        try:
            response = requests.post(
                self._api_url,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
            if response.status_code >= 400:
                logger.warning('SQL generation API error %s', response.status_code)
                return None

            body = response.json()
            record_usage(TokenUsage.from_api(body), 'sql')
            content = body['choices'][0]['message']['content']
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
            logger.warning('SQL generation failed (%s)', error)
            return None

        sql = _extract_sql(content)
        logger.info('generated SQL: %r', sql[:200])
        return sql

    def _run_sql(self, sql: str, k: int) -> Optional[List[Record]]:
        """Execute SQL. Returns None on error, otherwise matched records (may be empty)."""
        state = self._db_state()
        if state.conn is None or not _is_safe_select(sql):
            logger.warning('rejecting unsafe or empty SQL')
            return None

        try:
            cursor = state.conn.execute(sql)
            columns = [description[0] for description in cursor.description or []]
            rows = cursor.fetchmany(k)
        except sqlite3.Error as error:
            logger.warning('SQL execution failed: %s', error)
            return None

        id_column = None
        for index, column in enumerate(columns):
            if column.lower() == 'id':
                id_column = index
                break

        if id_column is None:
            if rows:
                logger.info('SQL aggregate result: %d row(s)', len(rows))
                return [
                    _aggregate_context_record(
                        sql, columns, rows, len(state.records_by_id),
                    )
                ]

            logger.warning('SQL result has no id column')
            return []

        results: List[Record] = []
        seen = set()
        for row in rows:
            try:
                record_id = int(row[id_column])
            except (TypeError, ValueError):
                continue

            if record_id in seen:
                continue

            record = state.records_by_id.get(record_id)
            if record is not None:
                seen.add(record_id)
                results.append(record)

        return results

    def _execute_sql(self, sql: str, k: int) -> Optional[List[Record]]:
        prepared = _prepare_sql(sql, self._db_state().sql_columns)
        if prepared != sql:
            logger.info('normalized SQL: %r', prepared[:200])

        results = self._run_sql(prepared, k)
        if results is not None:
            return results

        if prepared != sql:
            return self._run_sql(sql, k)

        return None

    def _list_sql_hint(self) -> str:
        return (
            '\nReturn SELECT id, Name FROM DATASETS with a WHERE clause; '
            'do not use COUNT or other aggregates.'
        )

    def _is_aggregate_result(self, results: List[Record]) -> bool:
        return (
            len(results) == 1
            and (
                results[0].get('Id') == 0
                or results[0].get('Name') == CATALOGUE_QUERY_RESULT_NAME
            )
        )

    def _generate_list_sql(self, query: str, schema: Dict[str, Any]) -> Optional[str]:
        sql = self._generate_sql(query + self._list_sql_hint(), schema)
        if sql and _AGGREGATE_SQL_RE.search(sql):
            return None
        return sql

    def retrieve(self, records: List[Record], query: str, k: int = CHAT_TOP_K) -> List[Record]:
        logger.info('SQL retrieve: k=%d, query=%r', k, query[:120])
        self._ensure_db(records)

        try:
            schema = self._load_schema()
        except (requests.RequestException, RuntimeError) as error:
            logger.warning('failed to load schema (%s)', error)
            return []

        wants_list = bool(_LIST_INTENT_RE.search(query))
        sql = self._generate_sql(query, schema)
        if sql and wants_list and _AGGREGATE_SQL_RE.search(sql):
            logger.info('SQL was aggregate for a list query; retrying with list hint')
            sql = self._generate_list_sql(query, schema)

        if not sql:
            logger.info('SQL retrieve: generation failed, falling back to BM25')
            return self._bm25_fallback.retrieve(records, query, k)

        results = self._execute_sql(sql, k)
        if results is None:
            logger.info('SQL retrieve: execution failed, falling back to BM25')
            return self._bm25_fallback.retrieve(records, query, k)

        if wants_list and self._is_aggregate_result(results):
            logger.info('aggregate result for list query; retrying with list SQL')
            list_sql = self._generate_list_sql(query, schema)
            if list_sql:
                list_results = self._execute_sql(list_sql, k)
                if list_results and not self._is_aggregate_result(list_results):
                    results = list_results

        if not results:
            logger.info('SQL retrieve: no matches, falling back to BM25')
            return self._bm25_fallback.retrieve(records, query, k)

        logger.info('SQL retrieve: %d datasets matched', len(results))
        return results


# Registry of available retrieval algorithms, keyed by the name used in config
# (RETRIEVAL_ALGORITHM). Register new retrievers here to make them selectable.
RETRIEVERS: Dict[str, Type[Retriever]] = {
    'bm25': BM25Retriever,
    'sql': SQLRetriever,
}


def get_retriever(
    name: str = 'bm25',
    *,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model_name: Optional[str] = None,
    schema_url: Optional[str] = None,
) -> Retriever:
    """Instantiate the retriever registered under `name` (case-insensitive)."""
    key = (name or '').strip().lower()

    try:
        retriever_cls = RETRIEVERS[key]
    except KeyError:
        available = ', '.join(sorted(RETRIEVERS))
        raise ValueError(f"Unknown retrieval algorithm '{name}'. Available: {available}.")

    if key == 'sql':
        return SQLRetriever(
            api_key=api_key,
            api_url=api_url,
            model_name=model_name,
            schema_url=schema_url or MASADER_SCHEMA_URL,
        )

    return retriever_cls()
