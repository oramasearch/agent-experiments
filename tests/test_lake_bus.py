from __future__ import annotations

import json

from agent_experiments.lake_bus import LakeBus, run_task


def test_bus_uses_agent_credential_and_task_credential_stays_in_body() -> None:
    requests: list[tuple[str, str, dict[str, str], bytes | None]] = []

    def send(method, url, headers, body):
        requests.append((method, url, headers, body))
        if "receive_agent_tasks" in url:
            value = {"tasks": [], "next_sequence": 2}
        else:
            value = {"task_id": "task-1", "body": {"lake_token": "person-token"}}
        return 200, json.dumps({"ok": True, "value": value})

    bus = LakeBus("https://lake.test", "bus-token", "agent-experiments", send)
    tasks, sequence = bus.receive(1)
    claimed = bus.claim("task-1")

    assert tasks == []
    assert sequence == 2
    assert claimed is not None
    assert claimed["body"]["lake_token"] == "person-token"
    assert all(
        request[2]["Authorization"] == "Bearer bus-token" for request in requests
    )


class RecordingBus:
    token = "bus-token"

    def __init__(self) -> None:
        self.events: list[tuple[str, str, str, dict | None]] = []

    def publish(self, task_id, kind, message="", result=None):
        self.events.append((task_id, kind, message, result))


def test_claimed_task_publishes_progress_and_result() -> None:
    bus = RecordingBus()
    answer = run_task(
        bus,
        lambda body: {"ok": True, "echo": body["message"]},
        {"task_id": "task-1", "body": {"message": "design it"}},
    )

    assert answer == {"ok": True, "echo": "design it"}
    assert [event[1] for event in bus.events] == ["progress", "result"]
