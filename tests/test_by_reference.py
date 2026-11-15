"""Task credentials by reference (lake#871, lake docs/api/agent-bus.md).

A claim asks Lake for `credentials: by_reference`: the body arrives with no credential, and every
call made for the task carries this agent's bus credential and `X-Lake-Agent-Task: <task id>`. A
Lake from before lake#871 refuses the field; the claim is asked again and the body's own credential
is used, as before.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.parse
from types import SimpleNamespace
from typing import Any

import httpx

from agent_experiments import lake_bus, model, service
from agent_experiments.agent import Lake, Turn
from agent_experiments.service import State, run_turn

TASK = {"task_id": "t1", "customer_id": "c", "workspace_id": "w", "kind": "experiment"}


class BusLake:
    """lake's bus in memory. `supports_by_reference` False is a Lake from before lake#871."""

    def __init__(self, supports_by_reference: bool) -> None:
        self.supports_by_reference = supports_by_reference
        self.claims: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.stop = threading.Event()
        self.received = False

    def send(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, str]:
        self.headers.append(headers)
        capability = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
        if capability == "receive_agent_tasks":
            tasks = [] if self.received else [{"sequence": 1, **TASK}]
            self.received = True
            if not tasks:
                self.stop.set()
            return 200, json.dumps(
                {"ok": True, "value": {"tasks": tasks, "next_sequence": 2}}
            )
        arguments = json.loads(body or b"{}")
        if capability == "claim_agent_task":
            self.claims.append(arguments)
            if "credentials" in arguments and not self.supports_by_reference:
                return 400, json.dumps(
                    {
                        "ok": False,
                        "error": {
                            "code": "invalid_input",
                            "message": "unexpected field credentials",
                            "path": "/credentials",
                        },
                    }
                )
            by_reference = arguments.get("credentials") == "by_reference"
            credential = "" if by_reference else "lat.t1.secret"
            value = {**TASK, "body": {"message": "hi", "lake_token": credential}}
            if by_reference:
                value["credentials"] = "by_reference"
            return 200, json.dumps({"ok": True, "value": value})
        return 200, json.dumps({"ok": True, "value": {}})


def handed_body(lake: BusLake) -> dict[str, Any]:
    handed: list[dict[str, Any]] = []
    bus = lake_bus.LakeBus("http://lake", "bus-token", "agent-experiments", send=lake.send)
    lake_bus.work(bus, lambda body: handed.append(body) or {"ok": True}, lake.stop, workers=1)
    assert len(handed) == 1, handed
    return handed[0]


def test_a_claim_asks_by_reference_and_the_turn_holds_no_task_credential() -> None:
    lake = BusLake(supports_by_reference=True)
    body = handed_body(lake)
    assert lake.claims == [
        {"task_id": "t1", "agent": "agent-experiments", "credentials": "by_reference"}
    ]
    assert body["lake_token"] == "bus-token", "the turn calls Lake with the bus credential"
    assert body["lake_headers"] == {"X-Lake-Agent-Task": "t1"}
    assert lake_bus.task_headers(body) == {"X-Lake-Agent-Task": "t1"}
    # Bus calls themselves never name a task: Lake would answer them as the task.
    assert all(lake_bus.TASK_HEADER not in headers for headers in lake.headers)


def test_a_lake_from_before_by_reference_is_claimed_again_with_the_credential_in_the_body() -> None:
    lake = BusLake(supports_by_reference=False)
    body = handed_body(lake)
    assert lake.claims == [
        {"task_id": "t1", "agent": "agent-experiments", "credentials": "by_reference"},
        {"task_id": "t1", "agent": "agent-experiments"},
    ]
    assert body["lake_token"] == "lat.t1.secret", "the body's own credential is used, as before"
    assert "lake_headers" not in body
    assert lake_bus.task_headers(body) == {}


def test_a_claim_refused_for_another_reason_is_not_asked_again() -> None:
    assert not lake_bus.refuses_by_reference(
        lake_bus.BusError("invalid_input", "task_id is required")
    )
    assert lake_bus.refuses_by_reference(
        lake_bus.BusError("invalid_input", "unexpected field credentials")
    )


def test_task_headers_take_only_the_task_header() -> None:
    assert lake_bus.task_headers(
        {"lake_headers": {"X-Lake-Agent-Task": "t1", "Authorization": "Bearer other"}}
    ) == {"X-Lake-Agent-Task": "t1"}
    assert lake_bus.task_headers({"lake_headers": "t1"}) == {}
    assert lake_bus.task_headers({}) == {}


def by_reference_body() -> dict[str, Any]:
    claimed = {
        **TASK,
        "credentials": "by_reference",
        "body": {
            "kind": "experiment",
            "message": "design a study",
            "lake_token": "",
            "lake_url": "http://lake:8080/w/ws-1/workspaces/c-1/zetta",
        },
    }
    return lake_bus.task_credentials("bus-token", claimed)


def test_the_model_calls_of_a_by_reference_turn_name_its_task(monkeypatch) -> None:
    seen: list[httpx.Request] = []
    turns: list[dict[str, Any]] = []

    def gateway(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "gpt-5.5",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "A study."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})

    def answer_turn(turn, history, client, model_name, lake):
        turns.append(turn)
        completion = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": turn["message"]}])
        return {"reply": completion.choices[0].message.content, "result": {}}

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "agent_experiments.agent",
                        SimpleNamespace(Turn=lambda **values: values, answer_turn=answer_turn))
    monkeypatch.setattr(service, "gateway_client", lambda url, token, **kw: model.gateway_client(
        url, token, http_client=httpx.Client(transport=httpx.MockTransport(gateway)), **kw))
    state = State()
    state.lake_url = "http://lake:8080"
    answer = run_turn(by_reference_body(), state)
    assert answer["ok"], answer
    assert turns[0]["lake_token"] == "bus-token"
    assert turns[0]["lake_headers"] == {"X-Lake-Agent-Task": "t1"}
    assert seen[0].headers["authorization"] == "Bearer bus-token"
    assert seen[0].headers["x-lake-agent-task"] == "t1"


def test_every_lake_call_of_a_by_reference_turn_names_its_task() -> None:
    requests: list[httpx.Request] = []

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "value": {}})

    turn = Turn(kind="experiment", message="design", lake_url="https://lake.test",
                lake_token="bus-token", lake_headers={"X-Lake-Agent-Task": "t1"})
    lake = Lake(turn, transport=httpx.MockTransport(transport))
    try:
        lake.read("get_study", {"study": "study-1"})
        lake.write("create_study", {"name": "s"})
    finally:
        lake.close()
    assert len(requests) == 2
    for request in requests:
        assert request.headers["authorization"] == "Bearer bus-token"
        assert request.headers["x-lake-agent-task"] == "t1"


def test_a_turn_with_its_credential_in_the_body_sends_no_task_header() -> None:
    requests: list[httpx.Request] = []

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "value": {}})

    turn = Turn(kind="experiment", message="design", lake_url="https://lake.test",
                lake_token="lat.t1.secret")
    lake = Lake(turn, transport=httpx.MockTransport(transport))
    try:
        lake.read("get_study", {"study": "study-1"})
    finally:
        lake.close()
    assert requests[0].headers["authorization"] == "Bearer lat.t1.secret"
    assert "x-lake-agent-task" not in requests[0].headers
