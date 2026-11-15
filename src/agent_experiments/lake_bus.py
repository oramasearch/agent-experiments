# SPDX-License-Identifier: MIT

"""Receive ExperimentsAgent turns from lake's agent bus."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

WAIT_MS = 10_000
Send = Callable[[str, str, dict[str, str], bytes | None], tuple[int, str]]


def _urllib_send(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
) -> tuple[int, str]:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=WAIT_MS / 1000 + 20) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


class BusError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class LakeBus:
    """This agent's authenticated access to the bus through lake."""

    def __init__(
        self,
        lake_url: str,
        token: str,
        agent: str,
        send: Send | None = None,
    ) -> None:
        self.lake_url = lake_url.rstrip("/")
        self.token = token
        self.agent = agent
        self.send = send or _urllib_send

    def call(
        self, capability: str, arguments: dict[str, Any], read: bool
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        url = f"{self.lake_url}/v1/{capability}"
        body = None
        if read:
            encoded = {
                key: value if isinstance(value, str) else json.dumps(value)
                for key, value in arguments.items()
            }
            url += "?" + urllib.parse.urlencode(encoded)
        else:
            headers["Content-Type"] = "application/json"
            body = json.dumps(arguments).encode()
        status, text = self.send("GET" if read else "POST", url, headers, body)
        try:
            answer = json.loads(text)
        except json.JSONDecodeError:
            raise BusError(
                "store_error", f"lake answered HTTP {status} without JSON"
            ) from None
        if not isinstance(answer, dict) or not answer.get("ok"):
            error = answer.get("error") if isinstance(answer, dict) else None
            details = error if isinstance(error, dict) else {}
            code = str(details.get("code") or "store_error")
            message = str(details.get("message") or f"HTTP {status}")
            raise BusError(code, message)
        value = answer.get("value") or {}
        return value if isinstance(value, dict) else {}

    def receive(self, from_sequence: int) -> tuple[list[dict[str, Any]], int]:
        value = self.call(
            "receive_agent_tasks",
            {
                "agent": self.agent,
                "from_sequence": from_sequence,
                "wait_ms": WAIT_MS,
            },
            read=True,
        )
        tasks = value.get("tasks") or []
        next_sequence = int(value.get("next_sequence") or from_sequence)
        return list(tasks), next_sequence

    def claim(self, task_id: str) -> dict[str, Any] | None:
        try:
            return self.call(
                "claim_agent_task",
                {"task_id": task_id, "agent": self.agent},
                read=False,
            )
        except BusError as error:
            if error.code in {"conflict", "not_found"}:
                return None
            raise

    def publish(
        self,
        task_id: str,
        kind: str,
        message: str = "",
        result: dict[str, Any] | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "task_id": task_id,
            "agent": self.agent,
            "type": kind,
            "message": message,
        }
        if result is not None:
            event["result"] = result
        self.call("publish_agent_event", event, read=False)


def _log(**fields: Any) -> None:
    print(json.dumps({"bus": True, **fields}, default=str), flush=True)


def run_task(
    bus: LakeBus,
    handle: Callable[[dict[str, Any]], dict[str, Any]],
    claimed: dict[str, Any],
) -> dict[str, Any]:
    task_id = str(claimed["task_id"])
    try:
        bus.publish(task_id, "progress", "Handling the experiment turn")
    except BusError as error:
        _log(task=task_id, progress_not_published=str(error))
    try:
        answer = handle(claimed.get("body") or {})
    except Exception as error:  # noqa: BLE001
        answer = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    if answer.get("ok"):
        bus.publish(task_id, "result", result=answer)
    else:
        bus.publish(task_id, "failed", str(answer.get("error") or "the agent failed"))
    _log(task=task_id, ok=bool(answer.get("ok")))
    return answer


def work(
    bus: LakeBus,
    handle: Callable[[dict[str, Any]], dict[str, Any]],
    stop: threading.Event,
    workers: int = 4,
) -> None:
    sequence = 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while not stop.is_set():
            try:
                tasks, sequence = bus.receive(sequence)
            except (BusError, OSError) as error:
                _log(agent=bus.agent, receive_failed=str(error))
                stop.wait(2)
                continue
            for task in tasks:
                try:
                    claimed = bus.claim(str(task["task_id"]))
                except (BusError, OSError) as error:
                    _log(task=task.get("task_id"), claim_failed=str(error))
                    continue
                if claimed is not None:
                    pool.submit(run_task, bus, handle, claimed)


def start_from_env(
    handle: Callable[[dict[str, Any]], dict[str, Any]],
    default_agent: str,
) -> threading.Thread | None:
    lake_url = os.environ.get("LAKE_URL", "")
    token = os.environ.get("AGENT_BUS_TOKEN", "")
    agent = os.environ.get("AGENT_BUS_AGENT", default_agent)
    if not (lake_url and token and agent):
        _log(agent=agent, status="not configured")
        return None
    stop = threading.Event()
    thread = threading.Thread(
        target=work,
        args=(LakeBus(lake_url, token, agent), handle, stop),
        name="lake-bus",
        daemon=True,
    )
    thread.start()
    _log(agent=agent, status="taking tasks", lake=lake_url)
    return thread
