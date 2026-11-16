# SPDX-License-Identifier: MIT

"""Conversation history is bound to its workspace and bounded (agent-experiments#35)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent_experiments import conversations
from agent_experiments.conversations import ConversationKey, ConversationStore, bounded
from agent_experiments.service import State, run_turn

LAKE = "https://lake.test"


def key(conversation: str = "X", customer: str = "c-a") -> ConversationKey:
    return ConversationKey(customer, f"{LAKE}/w/ws/workspaces/{customer}/zetta", "", "", conversation)


def recording_agent(monkeypatch) -> list[list[dict]]:
    """An answer_turn that records the history it was given and appends the turn's message."""
    seen: list[list[dict]] = []

    def answer_turn(turn, history, client, model, lake=None):
        seen.append(list(history))
        if not history:
            history.append({"role": "system", "content": "prompt"})
        history.append({"role": "user", "content": turn["message"]})
        history.append({"role": "assistant", "content": "observation: " + turn["message"]})
        return {"reply": "ok", "result": {"kind": "status"}}

    module = SimpleNamespace(Turn=lambda **values: values, answer_turn=answer_turn)
    monkeypatch.setitem(__import__("sys").modules, "agent_experiments.agent", module)
    return seen


def turn(customer: str, message: str, conversation: str = "X") -> dict:
    return {
        "kind": "experiment_status",
        "message": message,
        "lake_token": f"token-{customer}",
        "customer_id": customer,
        "conversation_id": conversation,
        "lake_url": f"{LAKE}/w/ws-{customer}/workspaces/{customer}/zetta",
    }


def test_another_workspace_with_the_same_conversation_id_sees_nothing(monkeypatch) -> None:
    seen = recording_agent(monkeypatch)
    state = State()
    state.lake_url = LAKE

    assert run_turn(turn("c-a", "A's secret dataset rows"), state)["ok"]
    assert run_turn(turn("c-b", "what did we say?"), state)["ok"]

    assert seen[1] == [], "B's turn must not receive A's history"
    assert run_turn(turn("c-a", "and next?"), state)["ok"]
    sent_to_a = json.dumps(seen[2])
    assert "A's secret dataset rows" in sent_to_a
    assert "what did we say?" not in sent_to_a, "B cannot write into A's conversation"


def test_a_turn_works_on_a_copy_saved_back_whole(monkeypatch) -> None:
    recording_agent(monkeypatch)
    state = State()
    state.lake_url = LAKE
    run_turn(turn("c-a", "first"), state)
    history = state.conversations.checkout(
        conversations.conversation_key(turn("c-a", "first"),
                                       turn("c-a", "first")["lake_url"], "X"))
    history.append({"role": "user", "content": "not saved"})
    again = state.conversations.checkout(
        conversations.conversation_key(turn("c-a", "first"),
                                       turn("c-a", "first")["lake_url"], "X"))
    assert {"role": "user", "content": "not saved"} not in again


def test_10000_conversations_stay_within_the_bound() -> None:
    store = ConversationStore()
    for index in range(10_000):
        store.store(key(f"conversation-{index}"), [{"role": "user", "content": "x" * 100}])
    assert len(store) == conversations.MAX_CONVERSATIONS
    assert store.checkout(key("conversation-0")) == [], "the oldest is dropped first"
    assert store.checkout(key("conversation-9999")) != []


def test_10000_new_conversation_ids_through_the_service_stay_bounded(monkeypatch) -> None:
    recording_agent(monkeypatch)
    state = State()
    state.lake_url = LAKE
    for index in range(10_000):
        run_turn(turn("c-a", "hello", conversation=f"id-{index}"), state)
    assert len(state.conversations) <= conversations.MAX_CONVERSATIONS


def test_an_idle_conversation_expires() -> None:
    now = [0.0]
    store = ConversationStore(clock=lambda: now[0])
    store.store(key(), [{"role": "user", "content": "hello"}])
    now[0] = conversations.IDLE_SECONDS - 1
    assert store.checkout(key()) != []
    now[0] += conversations.IDLE_SECONDS + 1
    assert store.checkout(key()) == []
    assert len(store) == 0


def test_a_long_history_keeps_its_prompt_and_newest_whole_exchanges() -> None:
    history = [{"role": "system", "content": "prompt"}]
    for index in range(200):
        history += [
            {"role": "user", "content": f"question {index}"},
            {"role": "assistant", "tool_calls": [{"id": f"call-{index}"}]},
            {"role": "tool", "tool_call_id": f"call-{index}", "content": "y" * 3000},
            {"role": "assistant", "content": f"answer {index}"},
        ]

    kept = bounded(history)

    assert kept[0] == {"role": "system", "content": "prompt"}
    assert kept[1]["role"] == "user", "a tool reply never loses the call that asked for it"
    assert len(kept) - 1 <= conversations.MAX_MESSAGES
    assert sum(len(json.dumps(item)) for item in kept[1:]) <= conversations.MAX_CHARS
    assert kept[-1] == {"role": "assistant", "content": "answer 199"}


def test_a_stored_history_is_bounded() -> None:
    store = ConversationStore()
    store.store(key(), [{"role": "system", "content": "prompt"}]
                + [{"role": "user", "content": "z" * 10_000} for _ in range(100)])
    kept = store.checkout(key())
    assert kept[0]["role"] == "system"
    assert sum(len(json.dumps(item)) for item in kept[1:]) <= conversations.MAX_CHARS
