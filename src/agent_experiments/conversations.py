# SPDX-License-Identifier: MIT

"""Conversation history held in the pod, bound to its workspace and bounded (agent-experiments#35).

One pod answers turns from every workspace, so a conversation id alone is not an owner: a turn from
workspace B naming workspace A's conversation id must never see A's messages, nor write into them.
History is therefore keyed by `ConversationKey`, the customer, the workspace path Lake handed the
turn (`lake_url`, already checked to be on Lake), the workspace and user ids when the turn carries
them, and the conversation id. A turn with another key gets its own, empty history.

The store is also bounded, so no caller can grow the pod's memory without limit:

* at most `MAX_CONVERSATIONS` conversations; the least recently used is dropped first;
* a conversation unused for `IDLE_SECONDS` expires;
* a stored history keeps its system prompt and at most `MAX_MESSAGES` messages and `MAX_CHARS`
  characters after it, dropping the oldest whole exchanges (from one user message to the next), so
  a tool reply is never left without the call that asked for it.

A turn works on a copy (`checkout`) and saves it back whole (`store`), both under the lock, so two
turns on one conversation never interleave their messages; the later one is kept.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MAX_CONVERSATIONS = 256
IDLE_SECONDS = 60 * 60
MAX_MESSAGES = 60
MAX_CHARS = 200_000


@dataclass(frozen=True)
class ConversationKey:
    customer_id: str
    workspace: str
    workspace_id: str
    user_id: str
    conversation_id: str


def conversation_key(body: dict[str, Any], lake_url: str, conversation_id: str) -> ConversationKey:
    """The owner of a turn's conversation, from the turn and its checked workspace path."""

    def text(*names: str) -> str:
        for name in names:
            value = body.get(name)
            if value:
                return str(value)
        return ""

    return ConversationKey(
        customer_id=text("customer_id"),
        workspace=lake_url,
        workspace_id=text("workspace_id", "workspace"),
        user_id=text("user_id", "requested_by", "subject"),
        conversation_id=conversation_id,
    )


def _size(message: dict[str, Any]) -> int:
    return len(json.dumps(message, default=str))


def bounded(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`history` cut to its system prompt plus the newest whole exchanges within the bounds."""
    head = history[:1] if history and history[0].get("role") == "system" else []
    rest = history[len(head):]
    while rest and (len(rest) > MAX_MESSAGES or sum(_size(item) for item in rest) > MAX_CHARS):
        # Drop the oldest exchange: everything up to the next user message.
        cut = next(
            (index for index in range(1, len(rest)) if rest[index].get("role") == "user"),
            len(rest),
        )
        rest = rest[cut:]
    return head + rest


class ConversationStore:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.lock = threading.Lock()
        self.entries: OrderedDict[ConversationKey, tuple[float, list[dict[str, Any]]]] = (
            OrderedDict()
        )

    def _expire(self, now: float) -> None:
        while self.entries:
            key, (used, _) = next(iter(self.entries.items()))
            if now - used <= IDLE_SECONDS:
                break
            del self.entries[key]

    def checkout(self, key: ConversationKey) -> list[dict[str, Any]]:
        """A copy of the conversation's history; empty when it is new or expired."""
        with self.lock:
            now = self.clock()
            self._expire(now)
            entry = self.entries.get(key)
            if entry is None:
                return []
            self.entries.move_to_end(key)
            self.entries[key] = (now, entry[1])
            return list(entry[1])

    def store(self, key: ConversationKey, history: list[dict[str, Any]]) -> None:
        """Save the turn's history, bounded, and drop what is over the conversation limit."""
        kept = bounded(list(history))
        with self.lock:
            now = self.clock()
            self.entries[key] = (now, kept)
            self.entries.move_to_end(key)
            self._expire(now)
            while len(self.entries) > MAX_CONVERSATIONS:
                self.entries.popitem(last=False)

    def __len__(self) -> int:
        with self.lock:
            return len(self.entries)
