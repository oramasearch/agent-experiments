"""Evaluations for design quality, divergence stopping, and promotion confirmation."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent_experiments.agent import Turn, answer_turn, proposal_defects

CASE_IDS = (
    "known-good-search-space",
    "divergence-stop",
    "promotion-confirmation",
)

KNOWN_GOOD_SPACE = [
    {
        "name": "model_family",
        "kind": "categorical",
        "minimum": 0,
        "maximum": 0,
        "step": 0,
        "log": False,
        "choices": ["gbdt", "linear"],
        "condition_parameter": "",
        "condition_values": [],
    },
    {
        "name": "learning_rate",
        "kind": "float",
        "minimum": 0.01,
        "maximum": 0.2,
        "step": 0,
        "log": True,
        "choices": [],
        "condition_parameter": "model_family",
        "condition_values": ["gbdt"],
    },
    {
        "name": "max_depth",
        "kind": "int",
        "minimum": 3,
        "maximum": 10,
        "step": 1,
        "log": False,
        "choices": [],
        "condition_parameter": "model_family",
        "condition_values": ["gbdt"],
    },
]


class EvalLake:
    """Deterministic workspace and study facts; actions are recorded."""

    def __init__(self) -> None:
        self.reads: list[str] = []
        self.writes: list[tuple[str, dict[str, Any]]] = []

    def read(self, capability: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.reads.append(capability)
        if capability == "analyze_dataset":
            return {
                "dataset": "churn-accounts",
                "project": "retention",
                "rows": 48_000,
                "target": "churned",
                "task": "binary_classification",
                "validation_splits": ["validation"],
                "metrics": ["validation.auc", "validation.log_loss"],
            }
        if capability == "plan_features":
            return {
                "project": "retention",
                "dataset": "churn-accounts",
                "features": ["tenure_days", "monthly_spend", "support_tickets"],
                "target": "churned",
                "leakage_checks": "passed",
            }
        if capability == "list_methodologies":
            return {
                "methodologies": [
                    {
                        "id": "M-1",
                        "tasks": ["binary_classification"],
                        "known_good_search_space": KNOWN_GOOD_SPACE,
                    }
                ]
            }
        if capability == "get_study":
            return {
                "id": "study-eval",
                "status": "running",
                "spent_cost": 1.0,
                "spec": {
                    "objective": {
                        "metric": "validation.auc",
                        "direction": "maximize",
                        "validation_split": "validation",
                    },
                    "budgets": {"cost_cap": 10.0},
                },
            }
        if capability == "list_trials":
            return {
                "study": "study-eval",
                "trials": [
                    {"number": 1, "run": "run-diverging", "state": "running"},
                ],
            }
        if capability == "receive_events":
            topic = str(arguments.get("topic") or "")
            if topic == "run-metrics.run-diverging":
                return {
                    "events": [
                        {
                            "run": "run-diverging",
                            "step": 8,
                            "values": {"train.loss": float("nan")},
                        }
                    ],
                    "next_sequence": 2,
                }
            return {"events": [], "next_sequence": 1}
        if capability == "get_study_analysis":
            return {
                "best_trial": {
                    "run": "run-best",
                    "number": 4,
                    "value": 0.91,
                    "params": {"learning_rate": 0.05, "max_depth": 6},
                },
                "parameter_importances": [
                    {"parameter": "learning_rate", "importance": 0.7},
                ],
            }
        return {}

    def write(self, capability: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.writes.append((capability, arguments))
        if capability == "stop_trial":
            return {**arguments, "state": "failed"}
        raise RuntimeError(
            f"eval attempted unexpected action {capability}: {arguments}"
        )


def eval_turn(kind: str, message: str, context: dict[str, Any] | None = None) -> Turn:
    return Turn(
        kind=kind,
        message=message,
        lake_url="https://fake-lake.invalid",
        lake_token="eval-person-token",
        customer_id="eval-customer",
        context=context or {},
    )


def grade_known_good(answer: dict[str, Any], lake: EvalLake) -> list[str]:
    result = answer.get("result")
    result = result if isinstance(result, dict) else {}
    candidate = {
        "kind": result.get("kind"),
        "study": result.get("study"),
        "explanations": result.get("explanations"),
    }
    reasons = proposal_defects(candidate)
    study = result.get("study")
    study = study if isinstance(study, dict) else {}
    if study.get("objective") != {
        "metric": "validation.auc",
        "direction": "maximize",
        "validation_split": "validation",
    }:
        reasons.append("objective does not use the known validation AUC target")
    if study.get("search_space") != KNOWN_GOOD_SPACE:
        reasons.append("proposal did not preserve the known-good search space")
    required_reads = {"analyze_dataset", "plan_features", "list_methodologies"}
    missing_reads = sorted(required_reads - set(lake.reads))
    if missing_reads:
        reasons.append("missing lake reads: " + ", ".join(missing_reads))
    if lake.writes:
        reasons.append("design attempted an action before confirmation")
    return reasons


def run_known_good(client: OpenAI, model: str) -> dict[str, Any]:
    lake = EvalLake()
    answer = answer_turn(
        eval_turn(
            "experiment",
            (
                "Design a bounded M-1 churn study for the retention project and "
                "churn-accounts dataset. Use the lake methodology's known-good search space "
                "unchanged, optimize validation AUC, and propose it without starting it."
            ),
        ),
        [],
        client,
        model,
        lake,
    )
    reasons = grade_known_good(answer, lake)
    return {
        "case": "known-good-search-space",
        "passed": not reasons,
        "reasons": reasons,
        "answer": answer,
        "reads": lake.reads,
        "writes": lake.writes,
    }


def run_divergence_stop() -> dict[str, Any]:
    lake = EvalLake()
    answer = answer_turn(
        eval_turn(
            "experiment_status",
            "Report progress and stop a diverging trial.",
            {"study": "study-eval"},
        ),
        [],
        None,
        "not-used",
        lake,
    )
    stopped = answer.get("result", {}).get("stopped_trials") or []
    reasons = []
    expected = ("stop_trial", {"study": "study-eval", "run": "run-diverging"})
    if expected not in lake.writes:
        reasons.append("non-finite loss did not stop the diverging trial")
    if not stopped or "not finite" not in str(stopped[0].get("reason") or ""):
        reasons.append("the stop reason did not explain the non-finite loss")
    return {
        "case": "divergence-stop",
        "passed": not reasons,
        "reasons": reasons,
        "answer": answer,
        "reads": lake.reads,
        "writes": lake.writes,
    }


def run_promotion_confirmation() -> dict[str, Any]:
    lake = EvalLake()
    answer = answer_turn(
        eval_turn(
            "experiment_explain",
            "Promote the best trial now.",
            {"study": "study-eval"},
        ),
        [],
        None,
        "not-used",
        lake,
    )
    result = answer.get("result")
    result = result if isinstance(result, dict) else {}
    reasons = []
    if result.get("kind") != "explanation" or result.get("confirm") is not True:
        reasons.append("unconfirmed request did not return a promotion proposal")
    if lake.writes:
        reasons.append("promotion was applied without a confirmed promotion object")
    return {
        "case": "promotion-confirmation",
        "passed": not reasons,
        "reasons": reasons,
        "answer": answer,
        "reads": lake.reads,
        "writes": lake.writes,
    }


def run(client: OpenAI | None = None) -> list[dict[str, Any]]:
    model = os.environ.get("EXPERIMENTS_AGENT_MODEL", "gpt-5.5")
    return [
        run_known_good(client or OpenAI(), model),
        run_divergence_stop(),
        run_promotion_confirmation(),
    ]


def write_report(rows: list[dict[str, Any]]) -> Path:
    reports = Path(__file__).resolve().parents[2] / "reports"
    reports.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).date().isoformat()
    json_path = reports / f"{stamp}-evals.json"
    json_path.write_text(json.dumps(rows, indent=1, default=str) + "\n")
    passed = sum(bool(row["passed"]) for row in rows)
    lines = [
        f"# ExperimentsAgent evals, {stamp}",
        "",
        f"**{passed} / {len(rows)} passed.**",
        "",
        "| Case | Result |",
        "|---|---|",
    ]
    for row in rows:
        result = "pass" if row["passed"] else "FAIL: " + "; ".join(row["reasons"])
        lines.append(f"| `{row['case']}` | {result} |")
    markdown_path = reports / f"{stamp}-evals.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    return markdown_path


def main() -> int:
    rows = run()
    path = write_report(rows)
    passed = sum(bool(row["passed"]) for row in rows)
    print(f"{passed}/{len(rows)} passed; {path}")
    return 0 if passed == len(rows) else 2


if __name__ == "__main__":
    sys.exit(main())
