# SPDX-License-Identifier: MIT

from __future__ import annotations

from agent_experiments.evals import (
    CASE_IDS,
    KNOWN_GOOD_SPACE,
    EvalLake,
    grade_known_good,
    run_divergence_stop,
    run_promotion_confirmation,
)


def test_eval_matrix_covers_required_cases() -> None:
    assert CASE_IDS == (
        "known-good-search-space",
        "divergence-stop",
        "promotion-confirmation",
    )


def test_eval_workspace_exposes_known_good_facts() -> None:
    lake = EvalLake()

    analysis = lake.read("analyze_dataset", {"dataset": "churn-accounts"})
    features = lake.read("plan_features", {"project": "retention"})
    projects = lake.read("list_projects", {})
    project = lake.read("get_project", {"name": "retention"})
    methodologies = lake.read("list_methodologies", {})

    assert analysis["target"] == "churned"
    assert analysis["columns"][0]["name"] == "event_at"
    assert features["leakage_checks"] == "passed"
    assert projects == {"projects": ["retention"]}
    assert project["graphs"] == []
    assert (
        methodologies["methodologies"][0]["known_good_search_space"] == KNOWN_GOOD_SPACE
    )


def test_known_good_grader_rejects_a_question() -> None:
    reasons = grade_known_good(
        {
            "reply": "Which dataset?",
            "result": {
                "kind": "question",
                "study": {},
                "explanations": {},
            },
        },
        EvalLake(),
    )

    assert any("objective" in reason for reason in reasons)
    assert any("known-good search space" in reason for reason in reasons)


def test_deterministic_runtime_evals_pass() -> None:
    divergence = run_divergence_stop()
    promotion = run_promotion_confirmation()

    assert divergence["passed"], divergence["reasons"]
    assert promotion["passed"], promotion["reasons"]
