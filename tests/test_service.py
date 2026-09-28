from __future__ import annotations

from types import SimpleNamespace

from agent_experiments.service import State, run_turn


def test_turn_requires_supported_kind_person_credential_and_lake_url(tmp_path) -> None:
    state = State()
    state.trace_dir = tmp_path

    missing_kind = run_turn({"message": "design a study"}, state)
    missing_credential = run_turn(
        {"kind": "experiment", "message": "design a study"},
        state,
    )

    assert not missing_kind["ok"]
    assert "kind" in missing_kind["error"]
    assert not missing_credential["ok"]
    assert "lake_token" in missing_credential["error"]


def test_turn_preserves_result_contract_and_writes_trace(tmp_path, monkeypatch) -> None:
    state = State()
    state.trace_dir = tmp_path
    expected = {
        "reply": "Ready to review.",
        "result": {
            "kind": "proposal",
            "study": {},
            "confirm": True,
        },
    }

    def answer_turn(*_args, **_kwargs):
        return expected

    module = SimpleNamespace(Turn=lambda **values: values, answer_turn=answer_turn)
    monkeypatch.setitem(__import__("sys").modules, "agent_experiments.agent", module)

    answer = run_turn(
        {
            "kind": "experiment",
            "message": "design a study",
            "lake_token": "person-token",
            "lake_url": "https://lake.test",
        },
        state,
        client=SimpleNamespace(),
    )

    assert answer["ok"]
    assert answer["result"] == expected["result"]
    traces = list(tmp_path.glob("*.json"))
    assert len(traces) == 1
    assert '"kind": "experiment"' in traces[0].read_text()
