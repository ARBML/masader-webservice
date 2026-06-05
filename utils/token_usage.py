"""Track LLM token usage per chat turn and per client session.

Each `/chat` request may trigger several LLM calls (router, SQL generation,
answer streaming). Usage from the API `usage` object is accumulated for the
turn, then merged into the session total keyed by `session_id` from the client.
"""

import logging
import threading

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

_session_id: ContextVar[str] = ContextVar('session_id', default='-')
_turn_usage: ContextVar['TokenUsage'] = ContextVar('turn_usage')

_sessions: Dict[str, 'TokenUsage'] = {}
_sessions_lock = threading.Lock()


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: 'TokenUsage') -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    @classmethod
    def from_api(cls, payload: Dict[str, Any]) -> Optional['TokenUsage']:
        usage = payload.get('usage')
        if not isinstance(usage, dict):
            return None

        prompt = int(usage.get('prompt_tokens') or 0)
        completion = int(usage.get('completion_tokens') or 0)
        total = int(usage.get('total_tokens') or (prompt + completion))
        if total <= 0:
            return None

        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )


def get_session_id() -> str:
    return _session_id.get()


def set_session_id(session_id: str) -> None:
    _session_id.set((session_id or '').strip() or '-')


def reset_turn_usage() -> TokenUsage:
    usage = TokenUsage()
    _turn_usage.set(usage)
    return usage


def record_usage(usage: Optional[TokenUsage], source: str) -> None:
    if usage is None or usage.total_tokens <= 0:
        return

    try:
        turn = _turn_usage.get()
    except LookupError:
        turn = reset_turn_usage()

    turn.add(usage)
    logger.info(
        'tokens %s: prompt=%d completion=%d total=%d',
        source,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
    )


def get_session_usage(session_id: Optional[str] = None) -> TokenUsage:
    """Return accumulated token usage for a client chat session."""
    session_key = (session_id or get_session_id()).strip() or '-'
    if session_key == '-':
        return TokenUsage()

    with _sessions_lock:
        return _sessions.get(session_key, TokenUsage())


def usage_payload(usage: TokenUsage) -> Dict[str, int]:
    return {
        'prompt_tokens': usage.prompt_tokens,
        'completion_tokens': usage.completion_tokens,
        'total_tokens': usage.total_tokens,
    }


def commit_turn_usage() -> TokenUsage:
    """Merge this turn into the session total and return session usage."""
    try:
        turn = _turn_usage.get()
    except LookupError:
        return get_session_usage()

    session_id = _session_id.get()
    if turn.total_tokens > 0 and session_id != '-':
        with _sessions_lock:
            session_total = _sessions.setdefault(session_id, TokenUsage())
            session_total.add(turn)

        logger.info(
            'token usage turn total: prompt=%d completion=%d total=%d | '
            'session %s cumulative: prompt=%d completion=%d total=%d',
            turn.prompt_tokens,
            turn.completion_tokens,
            turn.total_tokens,
            session_id,
            session_total.prompt_tokens,
            session_total.completion_tokens,
            session_total.total_tokens,
        )
        return session_total

    if turn.total_tokens > 0:
        logger.info(
            'token usage turn total: prompt=%d completion=%d total=%d',
            turn.prompt_tokens,
            turn.completion_tokens,
            turn.total_tokens,
        )

    return get_session_usage()
