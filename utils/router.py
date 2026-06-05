"""Decide whether a chat turn needs a fresh catalogue retrieval.

The chat endpoint is stateless and previously re-ran retrieval on every turn.
A `RetrievalRouter` lets a follow-up question ("how was it collected?") reuse
the datasets already under discussion instead of triggering a new search.

Routers only influence *whether* retrieval runs; the actual reuse of prior
datasets is handled by the caller (via `datasets_cited_in_history`). All
implementations fail safe: if they cannot decide, they request retrieval so
context is never silently lost.
"""

import logging

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

import requests

from constants import CHAT_SITE_NAME, CHAT_SITE_URL, ROUTER_SYSTEM_PROMPT
from utils.token_usage import TokenUsage, record_usage


logger = logging.getLogger(__name__)

Message = Dict[str, str]


class RetrievalRouter(ABC):
    """Strategy interface for deciding if a turn needs fresh retrieval."""

    @abstractmethod
    def should_retrieve(self, messages: List[Message]) -> bool:
        """Return True if the latest user turn needs a new catalogue search."""
        raise NotImplementedError


class AlwaysRetrieveRouter(RetrievalRouter):
    """Baseline router that retrieves on every turn (legacy behaviour)."""

    def should_retrieve(self, messages: List[Message]) -> bool:
        logger.info('always-retrieve router: retrieve')
        return True


class LLMRetrievalRouter(RetrievalRouter):
    """Use the chat model to classify the latest turn as NEW vs FOLLOWUP.

    A short, non-streaming classification call is made before the main answer.
    Any failure (network, API error, unexpected output) defaults to retrieving,
    so the router can only ever *save* a retrieval, never drop needed context.
    """

    def __init__(
        self,
        api_key: str,
        api_url: str,
        model_name: str,
        max_turns: int = 6,
        timeout: int = 15,
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url
        self._model_name = model_name
        self._max_turns = max_turns
        self._timeout = timeout

    def _recent_turns(self, messages: List[Message]) -> List[Message]:
        # Cap each turn so a long pasted answer can't blow up the prompt.
        conversation = [
            {'role': message['role'], 'content': str(message.get('content', ''))[:1000]}
            for message in messages
            if message.get('role') in ('user', 'assistant') and message.get('content')
        ]

        return conversation[-self._max_turns:]

    def should_retrieve(self, messages: List[Message]) -> bool:
        has_assistant_turn = any(
            message.get('role') == 'assistant' and message.get('content')
            for message in messages
        )

        # Nothing has been discussed yet: there is no prior context to reuse.
        if not has_assistant_turn:
            logger.info('no prior assistant turn: retrieve')
            return True

        if not self._api_key:
            logger.warning('router has no API key: defaulting to retrieve')
            return True

        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': CHAT_SITE_URL,
            'X-Title': CHAT_SITE_NAME,
        }
        payload = {
            'model': self._model_name,
            'messages': [
                {'role': 'system', 'content': ROUTER_SYSTEM_PROMPT},
                *self._recent_turns(messages),
            ],
            'stream': False,
            'temperature': 0,
            'max_tokens': 8,
            # Disable "thinking" so the reply is just the classification word.
            'chat_template_kwargs': {'enable_thinking': False},
        }

        logger.info('classifying turn (NEW vs FOLLOWUP) via %s', self._model_name)
        try:
            response = requests.post(
                self._api_url,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
            if response.status_code >= 400:
                logger.warning('router API error %s: defaulting to retrieve', response.status_code)
                return True

            body = response.json()
            record_usage(TokenUsage.from_api(body), 'router')
            content = body['choices'][0]['message']['content']
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
            logger.warning('router call failed (%s): defaulting to retrieve', error)
            return True

        # Reuse prior datasets only when the model clearly says FOLLOWUP;
        # anything else (including NEW or garbled output) triggers retrieval.
        retrieve = 'FOLLOWUP' not in (content or '').upper()
        logger.info(
            'router decision: %s (model said %r)',
            'RETRIEVE' if retrieve else 'REUSE', (content or '').strip(),
        )

        return retrieve


# Registry of available routers, keyed by the name used in config
# (RETRIEVAL_ROUTER). Register new routers here to make them selectable.
ROUTERS: Dict[str, Type[RetrievalRouter]] = {
    'always': AlwaysRetrieveRouter,
    'llm': LLMRetrievalRouter,
}


def get_router(
    name: str = 'llm',
    *,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model_name: Optional[str] = None,
) -> RetrievalRouter:
    """Instantiate the router registered under `name` (case-insensitive)."""
    key = (name or '').strip().lower()

    if key == 'always':
        return AlwaysRetrieveRouter()

    if key == 'llm':
        return LLMRetrievalRouter(api_key=api_key, api_url=api_url, model_name=model_name)

    available = ', '.join(sorted(ROUTERS))
    raise ValueError(f"Unknown retrieval router '{name}'. Available: {available}.")
