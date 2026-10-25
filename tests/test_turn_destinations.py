"""A turn's credential goes only to Lake at LAKE_URL, never to an address the request names (#18)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from agent_experiments import service
from agent_experiments.service import State, run_turn

LAKE = "http://lake-prod.lake.svc.cluster.local:80"
ROUTE = LAKE + "/w/ws-1/workspaces/cust-1/zetta"


@pytest.fixture
def calls(monkeypatch, tmp_path):
    """Where the turn would send its credential: the Lake address of the turn and the model URL."""
    seen: dict[str, list[str]] = {"lake": [], "model": []}

    def answer_turn(turn, history, client, model_name, lake):
        seen["lake"].append(turn["lake_url"])
        return {"reply": "ok", "result": {}}

    def gateway_client(url, token):
        seen["model"].append(url)
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "agent_experiments.agent",
                        SimpleNamespace(Turn=lambda **values: values, answer_turn=answer_turn))
    monkeypatch.setattr(service, "gateway_client", gateway_client)
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    return seen


def turn(**extra):
    return {"kind": "experiment", "message": "design a study", "lake_token": "lat.task.secret",
            **extra}


@pytest.mark.parametrize(
    "lake_url",
    [
        "http://attacker.example/w/ws-1/workspaces/cust-1/zetta",
        "https://lake-prod.lake.svc.cluster.local/w/ws-1/workspaces/cust-1/zetta",
        "http://lake-prod.lake.svc.cluster.local:8080/w/ws-1/workspaces/cust-1/zetta",
        "http://lake-prod.lake.svc.cluster.local.attacker.example/w/ws-1/zetta",
        "http://user@lake-prod.lake.svc.cluster.local/w/ws-1/zetta",
        "http://lake-prod.lake.svc.cluster.local:80@attacker.example/zetta",
        "http://lake-prod.lake.svc.cluster.local/w/ws-1/zetta?next=http://attacker.example",
        "http://lake-prod.lake.svc.cluster.local/w/ws-1/zetta#frag",
        "http://lake-prod.lake.svc.cluster.local/w/../../admin",
        "file:///etc/passwd",
        "lake-prod.lake.svc.cluster.local/w/ws-1/zetta",
        "http://lake-prod.lake.svc.cluster.local:notaport/zetta",
    ],
)
def test_a_lake_url_off_lakes_origin_is_refused_before_any_request(monkeypatch, calls, lake_url):
    monkeypatch.setenv("LAKE_URL", LAKE)
    answer = run_turn(turn(lake_url=lake_url), State())
    assert not answer["ok"]
    assert "lake_url is not on Lake at LAKE_URL" in answer["error"]
    assert calls == {"lake": [], "model": []}


def test_lakes_workspace_route_is_used_and_the_model_is_at_lake_url(monkeypatch, calls):
    monkeypatch.setenv("LAKE_URL", LAKE)
    answer = run_turn(turn(lake_url=ROUTE + "/"), State())
    assert answer["ok"], answer
    assert calls == {"lake": [ROUTE], "model": [LAKE]}


def test_the_default_port_and_host_case_are_the_same_origin(monkeypatch, calls):
    monkeypatch.setenv("LAKE_URL", LAKE)
    route = "http://LAKE-PROD.lake.svc.cluster.local/w/ws-1/workspaces/cust-1/zetta"
    assert run_turn(turn(lake_url=route), State())["ok"]
    assert calls["lake"] == [route]


def test_without_a_turn_lake_url_lake_url_itself_is_used(monkeypatch, calls):
    monkeypatch.setenv("LAKE_URL", LAKE + "/")
    assert run_turn(turn(), State())["ok"]
    assert calls == {"lake": [LAKE], "model": [LAKE]}


def test_without_lake_url_no_turn_sends_its_credential(monkeypatch, calls):
    monkeypatch.delenv("LAKE_URL", raising=False)
    answer = run_turn(turn(lake_url=ROUTE), State())
    assert not answer["ok"] and "LAKE_URL is not set" in answer["error"]
    assert calls == {"lake": [], "model": []}


def test_a_lake_url_must_be_under_lake_urls_path():
    lake = "http://lake:80/base"
    assert service.workspace_url("http://lake/base/w/1/zetta", lake) == "http://lake/base/w/1/zetta"
    assert service.workspace_url("http://lake/basement/w/1/zetta", lake) is None
    assert service.workspace_url("http://lake/other/w/1/zetta", lake) is None
    assert service.workspace_url("http://lake/base/../admin", lake) is None
