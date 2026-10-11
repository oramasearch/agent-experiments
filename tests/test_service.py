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
    monkeypatch.setattr(service, "gateway_client", lambda url, token: model.gateway_client(
        url, token, http_client=httpx.Client(transport=httpx.MockTransport(gateway))))
    state = State()
    state.trace_dir = tmp_path
    state.lake_url = "http://lake:8080"
    # lake hands ExperimentsAgent the workspace's Zetta path as lake_url; the gateway is at root.
    answer = run_turn({"kind": "experiment", "message": "design a study",
                       "lake_token": "lat.task.secret",
                       "lake_url": "http://lake:8080/w/ws-1/workspaces/c-1/zetta"}, state)
    assert answer["ok"] and answer["reply"] == "A study."
    assert [str(request.url) for request in seen] == ["http://lake:8080/openai/v1/chat/completions"]
    assert seen[0].headers["authorization"] == "Bearer lat.task.secret"
