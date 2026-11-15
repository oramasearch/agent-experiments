from __future__ import annotations

from types import SimpleNamespace

from agent_experiments.service import State, run_turn


def test_turn_requires_supported_kind_person_credential_and_lake_url(tmp_path) -> None:
    state = State()

    missing_kind = run_turn({"message": "design a study"}, state)
    missing_credential = run_turn(
        {"kind": "experiment", "message": "design a study"},
        state,
    )

    assert not missing_kind["ok"]
    assert "kind" in missing_kind["error"]
    assert not missing_credential["ok"]
    assert "lake_token" in missing_credential["error"]


def test_turn_preserves_result_contract_and_keeps_nothing_on_disk(tmp_path, monkeypatch) -> None:
    """agent-experiments#11: the pod serves every tenant, and the message and the answer are
    workspace content, so a turn writes no trace file, even when the operator still sets
    TRACE_DIR. Lake keeps the answer in the turn's workspace Zetta."""
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    state = State()
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
    assert answer["trace_id"], "the turn is still named, for lake to store with the answer"
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []
    assert not hasattr(state, "trace_dir")


def test_turn_calls_its_model_through_lakes_gateway_with_the_turns_key(tmp_path, monkeypatch) -> None:
    """lake#248: the model is lake's `/openai/v1`, with the turn's own key, never a provider key."""
    import httpx

    from agent_experiments import model, service

    seen: list[httpx.Request] = []

    def gateway(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "gpt-5.5",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "A study."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})

    def answer_turn(turn, history, client, model_name, lake):
        completion = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": turn["message"]}])
        return {"reply": completion.choices[0].message.content, "result": {}}

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "agent_experiments.agent",
                        SimpleNamespace(Turn=lambda **values: values, answer_turn=answer_turn))
    monkeypatch.setattr(service, "gateway_client", lambda url, token, **kw: model.gateway_client(
        url, token, http_client=httpx.Client(transport=httpx.MockTransport(gateway)), **kw))
    state = State()
    state.lake_url = "http://lake:8080"
    # lake hands ExperimentsAgent the workspace's Zetta path as lake_url; the gateway is at root.
    answer = run_turn({"kind": "experiment", "message": "design a study",
                       "lake_token": "lat.task.secret",
                       "lake_url": "http://lake:8080/w/ws-1/workspaces/c-1/zetta"}, state)
    assert answer["ok"] and answer["reply"] == "A study."
    assert [str(request.url) for request in seen] == ["http://lake:8080/openai/v1/chat/completions"]
    assert seen[0].headers["authorization"] == "Bearer lat.task.secret"


def test_health_reports_the_installed_version_and_prompt_version() -> None:
    from importlib.metadata import version

    from agent_experiments.service import PROMPT_VERSION, health

    body = health(State())

    assert body["ok"] and body["agent"] == "agent-experiments"
    assert body["version"] == version("agent-experiments")
    assert body["prompt_version"] == PROMPT_VERSION


def test_a_turn_without_a_lake_client_lets_the_agent_open_its_own(monkeypatch) -> None:
    """The Lake URL from LAKE_URL is never handed to the agent as its Lake client: with none
    given, answer_turn opens its own (a string there failed every Experiments turn, T12)."""
    state = State()
    seen: dict[str, object] = {}

    def answer_turn(turn, history, client, model, lake=None):
        seen["lake"] = lake
        return {"reply": "watching", "result": {"kind": "status"}}

    module = SimpleNamespace(Turn=lambda **values: values, answer_turn=answer_turn)
    monkeypatch.setitem(__import__("sys").modules, "agent_experiments.agent", module)

    answer = run_turn(
        {"kind": "experiment_status", "message": "how is it going", "lake_token": "person-token",
         "lake_url": state.lake_url or "https://lake.test"},
        state,
    )

    assert answer["ok"], answer
    assert seen["lake"] is None
