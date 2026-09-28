from __future__ import annotations

import json
import math
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx

from agent_experiments.agent import (
    Lake,
    Turn,
    answer_turn,
    divergence_reason,
    proposal_defects,
)


def parameter(
    name: str = "model_family",
    kind: str = "categorical",
    minimum: float = 0,
    maximum: float = 0,
    step: float = 0,
    log: bool = False,
    choices: list | None = None,
    condition_parameter: str = "",
    condition_values: list | None = None,
) -> dict:
    return {
        "name": name,
        "kind": kind,
        "minimum": minimum,
        "maximum": maximum,
        "step": step,
        "log": log,
        "choices": ["gbdt", "linear"] if choices is None else choices,
        "condition_parameter": condition_parameter,
        "condition_values": condition_values or [],
    }


def proposal() -> dict:
    return {
        "kind": "proposal",
        "reply": "Review this bounded study before it starts.",
        "study": {
            "name": "churn-tuning",
            "project": "retention",
            "pipeline": "production",
            "dataset": "accounts-snapshot",
            "methodology": "M-1",
            "objective": {
                "metric": "validation.auc",
                "direction": "maximize",
                "validation_split": "validation",
            },
            "search_space": [
                parameter(),
                parameter(
                    name="learning_rate",
                    kind="float",
                    minimum=0.001,
                    maximum=0.2,
                    step=0,
                    log=True,
                    choices=[],
                    condition_parameter="model_family",
                    condition_values=["gbdt"],
                ),
            ],
            "sampler": "tpe",
            "pruner": "asha",
            "budgets": {
                "max_trials": 40,
                "max_wall_time_ms": 3_600_000,
                "max_parallel_trials": 4,
                "cost_cap": 25.0,
            },
            "seed": 1138,
            "differential_privacy": False,
            "released_metrics": [],
        },
        "explanations": {
            "objective": "AUC measures ranking quality.",
            "validation_split": "Validation preserves test data for final evaluation.",
            "search_space": "The range covers conservative tree and linear settings.",
            "sampler": "TPE uses prior trials to focus the search.",
            "pruner": "ASHA stops weak trials after comparable steps.",
            "budgets": "Finite limits cap trials, elapsed time, concurrency, and cost.",
        },
    }


class Message:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = []

    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content}


class Scripted:
    def __init__(self, *answers: dict) -> None:
        self.answers = list(answers)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **_arguments):
        answer = self.answers.pop(0)
        message = Message(json.dumps(answer))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeLake:
    def __init__(self) -> None:
        self.reads: list[tuple[str, dict]] = []
        self.writes: list[tuple[str, dict]] = []
        self.events: dict[str, list[dict]] = {}

    def read(self, capability, arguments):
        self.reads.append((capability, arguments))
        if capability == "get_study":
            return {
                "id": "study-1",
                "status": "running",
                "spent_cost": 2.5,
                "spec": proposal()["study"],
            }
        if capability == "list_trials":
            return {
                "study": "study-1",
                "trials": [
                    {"run": "run-1", "state": "running", "number": 1},
                ],
            }
        if capability == "receive_events":
            topic = arguments["topic"]
            events = self.events.get(topic, [])
            return {
                "events": events,
                "next_sequence": arguments["from_sequence"] + len(events),
            }
        if capability == "get_study_analysis":
            return {
                "best_trial": {
                    "run": "run-7",
                    "number": 7,
                    "value": 0.91,
                    "params": {"learning_rate": 0.05},
                },
                "parameter_importances": [
                    {"parameter": "learning_rate", "importance": 0.8},
                ],
            }
        return {}

    def write(self, capability, arguments):
        self.writes.append((capability, arguments))
        if capability == "create_study":
            return {"id": "study-1", "status": "running", "spec": arguments}
        if capability == "stop_trial":
            return {**arguments, "state": "failed"}
        if capability == "promote_trial":
            return {**arguments, "model": "model-1", "project": "retention"}
        return {}


def turn(kind="experiment", message="design a churn study", context=None) -> Turn:
    return Turn(
        kind=kind,
        message=message,
        lake_url="https://lake.test",
        lake_token="person-token",
        customer_id="customer-1",
        context=context or {},
    )


def test_initial_turn_returns_proposal_without_actions() -> None:
    lake = FakeLake()

    answer = answer_turn(turn(), [], Scripted(proposal()), "test-model", lake)

    assert answer["result"]["kind"] == "proposal"
    assert answer["result"]["confirm"] is True
    assert lake.writes == []


def test_plain_language_confirmation_is_not_authority() -> None:
    lake = FakeLake()

    answer = answer_turn(
        turn(message="yes, start it"),
        [],
        Scripted({"kind": "question", "reply": "Send the reviewed proposal."}),
        "test-model",
        lake,
    )

    assert answer["result"]["kind"] == "question"
    assert lake.writes == []


def test_confirmed_proposal_starts_exact_reviewed_study() -> None:
    lake = FakeLake()
    reviewed = proposal()
    context = {"confirmed_proposal": reviewed}

    answer = answer_turn(turn(context=context), [], None, "test-model", lake)

    assert answer["result"]["kind"] == "running"
    assert lake.writes == [("create_study", reviewed["study"])]


def test_proposal_rejects_test_split_bad_budget_and_forward_condition() -> None:
    candidate = proposal()
    candidate["study"]["objective"]["validation_split"] = "test"
    candidate["study"]["budgets"]["cost_cap"] = math.inf
    candidate["study"]["search_space"][0]["condition_parameter"] = "later"
    candidate["study"]["search_space"][0]["condition_values"] = ["x"]

    defects = proposal_defects(candidate)

    assert any("test split" in defect for defect in defects)
    assert any("cost_cap" in defect for defect in defects)
    assert any("earlier parameter" in defect for defect in defects)


def test_invalid_model_answer_is_repaired() -> None:
    invalid = proposal()
    invalid["study"]["objective"]["validation_split"] = "test"

    answer = answer_turn(
        turn(),
        [],
        Scripted(invalid, proposal()),
        "test-model",
        FakeLake(),
    )

    assert answer["result"]["kind"] == "proposal"


def test_status_tails_study_and_run_topics_and_stops_nan_trial() -> None:
    lake = FakeLake()
    lake.events["study-events.study-1"] = [{"type": "trial_started"}]
    lake.events["run-metrics.run-1"] = [
        {"run": "run-1", "step": 3, "values": {"train.loss": float("nan")}},
    ]

    answer = answer_turn(
        turn("experiment_status", "status", {"study": "study-1"}),
        [],
        None,
        "test-model",
        lake,
    )

    assert answer["result"]["kind"] == "status"
    assert set(answer["result"]["events"]) == {
        "study-events.study-1",
        "run-metrics.run-1",
    }
    assert lake.writes == [
        ("stop_trial", {"study": "study-1", "run": "run-1"}),
    ]
    assert "not finite" in answer["result"]["stopped_trials"][0]["reason"]


def test_divergence_detector_covers_gradient_and_overfitting() -> None:
    gradient = [
        {"values": {"train.grad_norm": 12.0}},
        {"values": {"train.grad_norm": 1200.0}},
    ]
    overfitting = [
        {"values": {"train.loss": 0.40, "validation.loss": 0.70}},
        {"values": {"train.loss": 0.35, "validation.loss": 0.72}},
        {"values": {"train.loss": 0.30, "validation.loss": 0.75}},
    ]

    assert "gradient norm" in divergence_reason(gradient)
    assert "overfits" in divergence_reason(overfitting)


def test_explain_proposes_promotion_without_applying_it() -> None:
    lake = FakeLake()

    answer = answer_turn(
        turn("experiment_explain", "explain it", {"study": "study-1"}),
        [],
        None,
        "test-model",
        lake,
    )

    assert answer["result"]["kind"] == "explanation"
    assert answer["result"]["promotion"] == {"study": "study-1", "run": "run-7"}
    assert answer["result"]["confirm"] is True
    assert "variance" in answer["reply"]
    assert "overuse" in answer["reply"]
    assert lake.writes == []


def test_promotion_requires_reviewed_object() -> None:
    lake = FakeLake()
    context = {
        "study": "study-1",
        "confirmed_promotion": {
            "promotion": {"study": "study-1", "run": "run-7"},
        },
    }

    answer = answer_turn(
        turn("experiment_explain", "confirmed", context),
        [],
        None,
        "test-model",
        lake,
    )

    assert answer["result"]["kind"] == "promoted"
    assert lake.writes == [
        ("promote_trial", {"study": "study-1", "run": "run-7"}),
    ]


def test_promotion_refuses_a_trial_that_is_not_the_current_best() -> None:
    lake = FakeLake()
    context = {
        "study": "study-1",
        "confirmed_promotion": {
            "promotion": {"study": "study-1", "run": "run-6"},
        },
    }

    answer = answer_turn(
        turn("experiment_explain", "confirmed", context),
        [],
        None,
        "test-model",
        lake,
    )

    assert answer["result"]["kind"] == "question"
    assert "no longer" in answer["reply"]
    assert lake.writes == []


def test_lake_uses_exact_capability_routes_and_delegated_credential() -> None:
    requests = []

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "value": {}})

    lake = Lake(turn(), transport=httpx.MockTransport(transport))
    try:
        lake.read("get_study", {"study": "study-1"})
        lake.read("analyze_dataset", {"dataset": "accounts"})
        lake.write("create_study", proposal()["study"])
    finally:
        lake.close()

    assert [request.url.path for request in requests] == [
        "/v1/get_study",
        "/v1/analyze_dataset",
        "/v1/create_study",
    ]
    assert requests[0].method == "GET"
    assert parse_qs(requests[0].url.query.decode())["study"] == ["study-1"]
    assert requests[1].method == "POST"
    assert requests[2].method == "POST"
    assert all(
        request.headers["Authorization"] == "Bearer person-token"
        for request in requests
    )
