"""Per-turn logging helpers for tracing the chat pipeline.

Each `/chat` request gets a short turn id (via a `ContextVar`) that is injected
into every log record, so the steps taken during one turn can be followed
end-to-end even under concurrent requests.

Logs can also be captured into a per-turn queue (`start_log_capture`) so the
chat endpoint can stream them to the UI in real time. Capture is context-local,
so concurrent requests never see each other's logs.
"""

import logging
import queue
import uuid

from contextvars import ContextVar
from typing import Optional

from utils.token_usage import get_session_id, set_session_id as _set_session_id


_turn_id: ContextVar[str] = ContextVar('turn_id', default='-')
_log_queue: ContextVar[Optional['queue.Queue']] = ContextVar('log_queue', default=None)

LOG_FORMAT = '%(asctime)s %(levelname)s [turn %(turn_id)s] [session %(session_id)s] %(message)s'


class _TurnIdFilter(logging.Filter):
    """Attach the current turn id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.turn_id = _turn_id.get()
        record.session_id = get_session_id()
        return True


class _QueueCaptureHandler(logging.Handler):
    """Mirror records into the current turn's queue, if one is active."""

    def emit(self, record: logging.LogRecord) -> None:
        log_queue = _log_queue.get()
        if log_queue is None:
            return

        # Skip framework access logs; the panel should show pipeline steps only.
        if record.name.startswith('werkzeug'):
            return

        try:
            log_queue.put_nowait({'level': record.levelname, 'message': record.getMessage()})
        except Exception:
            # Live logging is best-effort; never let it break the request.
            pass


def configure_logging(level: int = logging.INFO) -> None:
    """Install handlers: one to the stream, one to the per-turn capture queue."""
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    stream_handler.addFilter(_TurnIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(stream_handler)
    root.addHandler(_QueueCaptureHandler())
    root.setLevel(level)


def set_session_id(session_id: str) -> None:
    """Bind the current request to a client chat session."""
    _set_session_id(session_id)


def new_turn_id() -> str:
    """Start a new turn: generate, set, and return a short correlation id."""
    turn_id = uuid.uuid4().hex[:8]
    _turn_id.set(turn_id)

    return turn_id


def start_log_capture() -> 'queue.Queue':
    """Begin capturing this context's logs into a fresh queue and return it."""
    log_queue: 'queue.Queue' = queue.Queue()
    _log_queue.set(log_queue)

    return log_queue


def stop_log_capture() -> None:
    """Stop capturing logs for this context."""
    _log_queue.set(None)
