"""Design, run, watch, explain, and promote AutoML studies."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

PROMPT_VERSION = "v1"
PROMPTS = Path(__file__).parent / "prompts"
MAX_MODEL_ROUNDS = 10
MAX_REPAIRS = 2
MAX_EVENTS = 100
EVENT_WAIT_MS = 20_000
EXPLODING_GRAD_NORM = 1_000.0
OVERFITTING_GAP = 0.25
OVERFITTING_POINTS = 3

READ_CAPABILITIES = {
    "analyze_dataset",
    "get_project",
    "get_run_metrics",
    "get_solution",
    "get_study",
    "get_study_analysis",
    "list_methodologies",
    "list_projects",
    "list_studies",
    "list_trials",
    "plan_features",
    "receive_events",
    "validate_study",
}
# Lake proxies every workspace read as GET and answers POST with 405 method_not_allowed.
POST_READS: set[str] = set()
ACTION_CAPABILITIES = {
    "create_study",
    "promote_trial",
    "stop_study",
    "stop_trial",
}
EXPLANATION_FIELDS = {
    "objective",
    "validation_split",
    "search_space",
    "sampler",
    "pruner",
    "budgets",
}
STUDY_FIELDS = {
    "name",
    "project",
    "pipeline",
    "dataset",
    "methodology",
    "objective",
    "search_space",
    "from_defaults",
    "sampler",
    "pruner",
    "budgets",
    "seed",
    "differential_privacy",
    "released_metrics",
}
PARAMETER_FIELDS = {
    "name",
    "kind",
    "minimum",
    "maximum",
    "step",
    "log",
    "choices",
    "condition_parameter",
    "condition_values",
}
OBJECTIVE_FIELDS = {"metric", "direction", "validation_split"}
BUDGET_FIELDS = {
    "max_trials",
    "max_wall_time_ms",
    "max_parallel_trials",
    "cost_cap",
}


@dataclass
class Turn:
    kind: str
    message: str
    lake_url: str
    lake_token: str
    customer_id: str = ""
    conversation_id: str = ""
    context: dict[str, Any] = field(default_factory=dict)


class LakeError(Exception):
    def __init__(self, capability: str, code: str, message: str) -> None:
        super().__init__(f"{capability}: {code}: {message}")
        self.capability = capability
        self.code = code
        self.message = message


class Lake:
    """Lake capabilities called as the signed-in person for one turn."""

    def __init__(
        self,
        turn: Turn,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.client = httpx.Client(
            base_url=turn.lake_url.rstrip("/"),
            timeout=180.0,
            headers={"Authorization": f"Bearer {turn.lake_token}"},
            transport=transport,
        )

    def close(self) -> None:
        self.client.close()

    def call(
        self,
        capability: str,
        arguments: dict[str, Any],
        read: bool,
        use_post: bool = False,
    ) -> dict[str, Any]:
        try:
            if read and not use_post:
                params = {
                    key: value if isinstance(value, str) else json.dumps(value)
                    for key, value in arguments.items()
                    if value is not None
                }
                response = self.client.get(f"/v1/{capability}", params=params)
            else:
                response = self.client.post(f"/v1/{capability}", json=arguments)
            body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise LakeError(capability, "transport_error", str(error)) from error
        if not isinstance(body, dict) or not body.get("ok"):
            details = body.get("error") if isinstance(body, dict) else None
            error = details if isinstance(details, dict) else {}
            raise LakeError(
                capability,
                str(error.get("code") or "lake_error"),
                str(error.get("message") or f"HTTP {response.status_code}"),
            )
        value = body.get("value") or {}
        return value if isinstance(value, dict) else {"value": value}

    def read(self, capability: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if capability not in READ_CAPABILITIES:
            raise LakeError(capability, "not_allowed", "not an ExperimentsAgent read")
        return self.call(
            capability,
            arguments,
            read=True,
            use_post=capability in POST_READS,
        )

    def write(self, capability: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if capability not in ACTION_CAPABILITIES:
            raise LakeError(capability, "not_allowed", "not an ExperimentsAgent action")
        return self.call(capability, arguments, read=False)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lake_read",
            "description": (
                "Read lake facts needed to design an AutoML study. Allowed capabilities: "
                + ", ".join(sorted(READ_CAPABILITIES))
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "capability": {"type": "string", "enum": sorted(READ_CAPABILITIES)},
                    "input": {"type": "object"},
                },
                "required": ["capability", "input"],
            },
        },
    }
]


def parse_json(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def proposal_defects(answer: dict[str, Any]) -> list[str]:
    kind = answer.get("kind")
    if kind == "question":
        question = answer.get("reply") or answer.get("question")
        return (
            []
            if isinstance(question, str) and question.strip()
            else ["question is empty"]
        )
    if kind != "proposal":
        return ["kind must be question or proposal"]
    study = answer.get("study")
    explanations = answer.get("explanations")
    if not isinstance(study, dict):
        return ["a proposal needs a study object"]
    problems = study_defects(study)
    if not isinstance(explanations, dict):
        problems.append("a proposal needs an explanations object")
    else:
        for field in sorted(EXPLANATION_FIELDS):
            value = explanations.get(field)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"explanations.{field} must explain the choice")
    return problems


def study_defects(study: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    unknown = sorted(set(study) - STUDY_FIELDS)
    if unknown:
        problems.append("unknown study fields: " + ", ".join(unknown))
    for name in ("name", "methodology"):
        if not isinstance(study.get(name), str) or not str(study[name]).strip():
            problems.append(f"study.{name} must be a non-empty string")
    for name in ("project", "pipeline", "dataset"):
        if not isinstance(study.get(name), str):
            problems.append(f"study.{name} must be a string")
    project = study.get("project")
    pipeline = study.get("pipeline")
    dataset = study.get("dataset")
    if isinstance(project, str) and isinstance(pipeline, str):
        if project and not pipeline:
            problems.append(
                "study.project must be empty for a dataset study or paired with a pipeline"
            )
        if pipeline and not project:
            problems.append("study.pipeline requires study.project")
    if isinstance(dataset, str) and isinstance(pipeline, str):
        if not dataset and not pipeline:
            problems.append("study must name a dataset or a pipeline")
    if "from_defaults" in study and not isinstance(study["from_defaults"], bool):
        problems.append("study.from_defaults must be boolean")
    problems.extend(objective_defects(study.get("objective")))
    problems.extend(search_space_defects(study.get("search_space")))
    if study.get("sampler") not in {"grid", "random", "tpe"}:
        problems.append("study.sampler must be grid, random, or tpe")
    if study.get("pruner") not in {"none", "median", "asha"}:
        problems.append("study.pruner must be none, median, or asha")
    problems.extend(budget_defects(study.get("budgets")))
    seed = study.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        problems.append("study.seed must be a non-negative integer")
    if "differential_privacy" in study and not isinstance(
        study["differential_privacy"], bool
    ):
        problems.append("study.differential_privacy must be boolean")
    released = study.get("released_metrics", [])
    if not isinstance(released, list) or not all(
        isinstance(metric, str) and metric for metric in released
    ):
        problems.append("study.released_metrics must contain metric names")
    elif len(set(released)) != len(released):
        problems.append("study.released_metrics must be unique")
    elif study.get("differential_privacy"):
        objective = study.get("objective")
        metric = objective.get("metric") if isinstance(objective, dict) else None
        if metric not in released:
            problems.append(
                "a differentially private study may optimize only a released metric"
            )
    return problems


def objective_defects(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["study.objective must be an object"]
    problems: list[str] = []
    missing = sorted(OBJECTIVE_FIELDS - set(value))
    unknown = sorted(set(value) - OBJECTIVE_FIELDS)
    if missing:
        problems.append("study.objective is missing fields: " + ", ".join(missing))
    if unknown:
        problems.append("study.objective has unknown fields: " + ", ".join(unknown))
    if not isinstance(value.get("metric"), str) or not value["metric"].strip():
        problems.append("study.objective.metric must be a non-empty string")
    if value.get("direction") not in {"minimize", "maximize"}:
        problems.append("study.objective.direction must be minimize or maximize")
    split = value.get("validation_split")
    if not isinstance(split, str) or not split.strip():
        problems.append("study.objective.validation_split must be a non-empty string")
    else:
        normalized = split.strip().lower()
        if (
            normalized == "test"
            or normalized.startswith("test_")
            or normalized.endswith("_test")
        ):
            problems.append("the test split cannot be used for model selection")
    return problems


def search_space_defects(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["study.search_space must contain at least one parameter"]
    problems: list[str] = []
    names: list[str] = []
    parents: dict[str, dict[str, Any]] = {}
    for index, parameter in enumerate(value):
        path = f"study.search_space[{index}]"
        if not isinstance(parameter, dict):
            problems.append(f"{path} must be an object")
            continue
        missing = sorted(PARAMETER_FIELDS - set(parameter))
        unknown = sorted(set(parameter) - PARAMETER_FIELDS)
        if missing:
            problems.append(f"{path} is missing fields: " + ", ".join(missing))
        if unknown:
            problems.append(f"{path} has unknown fields: " + ", ".join(unknown))
        name = parameter.get("name")
        if not isinstance(name, str) or not name:
            problems.append(f"{path}.name must be a non-empty string")
            continue
        if name in names:
            problems.append(f"{path}.name must be unique")
        names.append(name)
        kind = parameter.get("kind")
        if kind not in {"int", "float", "categorical"}:
            problems.append(f"{path}.kind must be int, float, or categorical")
        elif kind == "categorical":
            problems.extend(categorical_defects(parameter, path))
        else:
            problems.extend(numeric_defects(parameter, path, kind))
        parent = parameter.get("condition_parameter", "")
        condition_values = parameter.get("condition_values", [])
        if not isinstance(parent, str):
            problems.append(f"{path}.condition_parameter must be a string")
        elif parent and parent not in parents:
            problems.append(
                f"{path}.condition_parameter must name an earlier parameter"
            )
        if not isinstance(condition_values, list):
            problems.append(f"{path}.condition_values must be an array")
        elif bool(parent) != bool(condition_values):
            problems.append(f"{path} must set both condition fields or neither")
        elif parent and parent in parents:
            choices = parents[parent].get("choices", [])
            if choices and any(item not in choices for item in condition_values):
                problems.append(
                    f"{path}.condition_values must be choices from {parent}"
                )
        parents[name] = parameter
    return problems


def categorical_defects(parameter: dict[str, Any], path: str) -> list[str]:
    problems: list[str] = []
    choices = parameter.get("choices")
    if not isinstance(choices, list) or len(choices) < 2:
        problems.append(f"{path}.choices must contain at least two values")
    else:
        if any(
            not isinstance(choice, (str, int, float, bool)) or not finite_json(choice)
            for choice in choices
        ):
            problems.append(f"{path}.choices must contain finite scalar values")
        encoded = [json.dumps(choice, sort_keys=True) for choice in choices]
        if len(set(encoded)) != len(encoded):
            problems.append(f"{path}.choices must be unique")
    for name in ("minimum", "maximum", "step"):
        if not finite_number(parameter.get(name)):
            problems.append(f"{path}.{name} must be numeric")
    if not isinstance(parameter.get("log"), bool):
        problems.append(f"{path}.log must be boolean")
    return problems


def numeric_defects(
    parameter: dict[str, Any],
    path: str,
    kind: str,
) -> list[str]:
    minimum = parameter.get("minimum")
    maximum = parameter.get("maximum")
    step = parameter.get("step")
    problems: list[str] = []
    if not finite_number(minimum) or not finite_number(maximum):
        problems.append(f"{path} needs finite minimum and maximum")
    elif float(minimum) >= float(maximum):
        problems.append(f"{path}.minimum must be less than maximum")
    if not finite_number(step) or float(step) < 0:
        problems.append(f"{path}.step must be a non-negative number")
    if kind == "int" and any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (minimum, maximum, step)
    ):
        problems.append(f"{path} integer bounds and step must be integers")
    if not isinstance(parameter.get("log"), bool):
        problems.append(f"{path}.log must be boolean")
    elif parameter.get("log") and finite_number(minimum) and float(minimum) <= 0:
        problems.append(f"{path}.minimum must be positive for logarithmic sampling")
    return problems


def budget_defects(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["study.budgets must be an object"]
    problems: list[str] = []
    missing = sorted(BUDGET_FIELDS - set(value))
    unknown = sorted(set(value) - BUDGET_FIELDS)
    if missing:
        problems.append("study.budgets is missing fields: " + ", ".join(missing))
    if unknown:
        problems.append("study.budgets has unknown fields: " + ", ".join(unknown))
    for name in ("max_trials", "max_wall_time_ms", "max_parallel_trials"):
        number = value.get(name)
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            problems.append(f"study.budgets.{name} must be a positive integer")
    cost_cap = value.get("cost_cap")
    if not finite_number(cost_cap) or float(cost_cap) <= 0:
        problems.append("study.budgets.cost_cap must be a positive finite number")
    trials = value.get("max_trials")
    parallel = value.get("max_parallel_trials")
    if isinstance(trials, int) and isinstance(parallel, int) and parallel > trials:
        problems.append("study.budgets.max_parallel_trials cannot exceed max_trials")
    return problems


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def finite_json(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    return value is not None


def answer_turn(
    turn: Turn,
    history: list[dict[str, Any]],
    client: OpenAI | None,
    model: str,
    lake: Lake | None = None,
) -> dict[str, Any]:
    owned = lake is None
    lake = lake or Lake(turn)
    try:
        if turn.kind == "experiment":
            confirmed = turn.context.get("confirmed_proposal")
            if isinstance(confirmed, dict):
                return run_confirmed(turn, confirmed, lake)
            if client is None:
                raise RuntimeError("a model client is required to design a study")
            return design(turn, history, client, model, lake)
        if turn.kind == "experiment_status":
            return watch(turn, lake)
        if turn.kind == "experiment_explain":
            confirmed = turn.context.get("confirmed_promotion")
            if isinstance(confirmed, dict):
                return promote_confirmed(turn, confirmed, lake)
            return explain(turn, lake)
        raise ValueError(f"unsupported turn kind {turn.kind}")
    finally:
        if owned:
            lake.close()


def design(
    turn: Turn,
    history: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    lake: Lake,
) -> dict[str, Any]:
    if not history:
        history.append(
            {
                "role": "system",
                "content": (PROMPTS / f"{PROMPT_VERSION}.md").read_text(),
            }
        )
    history.append({"role": "user", "content": turn.message})
    answer: dict[str, Any] = {}
    problems: list[str] = []
    for repair in range(MAX_REPAIRS + 1):
        answer = model_answer(history, client, model, lake)
        problems = proposal_defects(answer)
        study = answer.get("study")
        if (
            not problems
            and answer.get("kind") == "proposal"
            and isinstance(study, dict)
        ):
            problems.extend(reference_defects(study, lake))
        if not problems:
            break
        if repair < MAX_REPAIRS:
            history.append(
                {
                    "role": "user",
                    "content": (
                        "Fix every validation failure and answer again with the complete JSON "
                        "object:\n" + "\n".join(f"- {problem}" for problem in problems)
                    ),
                }
            )
    if problems:
        return question_result(
            "I could not produce a valid study proposal: " + "; ".join(problems)
        )
    if answer.get("kind") == "question":
        return question_result(str(answer.get("reply") or answer.get("question")))
    return {
        "reply": str(
            answer.get("reply") or "Review this study proposal before it starts."
        ),
        "result": {
            "kind": "proposal",
            "study": answer["study"],
            "explanations": answer["explanations"],
            "confirm": True,
        },
    }


def model_answer(
    history: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    lake: Lake,
) -> dict[str, Any]:
    for _round in range(MAX_MODEL_ROUNDS):
        completion = client.chat.completions.create(
            model=model,
            messages=history,
            tools=TOOLS,
            response_format={"type": "json_object"},
        )
        message = completion.choices[0].message
        history.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return parse_json(message.content)
        for call in message.tool_calls:
            arguments = parse_json(call.function.arguments)
            capability = str(arguments.get("capability") or "")
            inputs = arguments.get("input")
            inputs = inputs if isinstance(inputs, dict) else {}
            try:
                observation = lake.read(capability, inputs)
                content = json.dumps({"ok": True, "value": observation}, default=str)
            except LakeError as error:
                content = json.dumps(
                    {
                        "ok": False,
                        "error": {"code": error.code, "message": error.message},
                    }
                )
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": content[:30_000],
                }
            )
    return {}


def question_result(reply: str) -> dict[str, Any]:
    return {
        "reply": reply,
        "result": {
            "kind": "question",
            "study": {},
            "explanations": {},
            "confirm": False,
        },
    }


def reference_defects(study: dict[str, Any], lake: Lake) -> list[str]:
    """Check every external study reference against current lake facts."""
    problems: list[str] = []
    dataset = str(study.get("dataset") or "")
    methodology = str(study.get("methodology") or "")
    project = str(study.get("project") or "")
    pipeline = str(study.get("pipeline") or "")

    if dataset:
        try:
            analysis = lake.read("analyze_dataset", {"dataset": dataset})
        except LakeError as error:
            problems.append(f"study.dataset does not resolve: {error.message}")
        else:
            if not any(
                analysis.get(field) for field in ("table_id", "table", "dataset")
            ):
                problems.append("study.dataset did not resolve to a dataset")

    try:
        listed = lake.read("list_methodologies", {})
    except LakeError as error:
        problems.append(f"study.methodology could not be checked: {error.message}")
    else:
        methodologies = listed.get("methodologies")
        methodologies = methodologies if isinstance(methodologies, list) else []
        matches = [
            item
            for item in methodologies
            if isinstance(item, dict)
            and methodology in {str(item.get("id") or ""), str(item.get("name") or "")}
        ]
        if not matches:
            problems.append(
                f"study.methodology does not resolve to a listed methodology: {methodology}"
            )
        elif matches[0].get("runnable") is False:
            problems.append(f"study.methodology is not runnable: {methodology}")
        elif matches[0].get("studies") is False:
            reason = str(matches[0].get("study_reason") or "no study driver")
            problems.append(
                f"study.methodology does not support studies: {methodology} ({reason})"
            )

    if not problems:
        problems.extend(validation_defects(study, lake))

    if project and pipeline:
        try:
            found = lake.read("get_project", {"name": project})
        except LakeError as error:
            problems.append(f"study.project does not resolve: {error.message}")
        else:
            graphs = found.get("graphs")
            graphs = graphs if isinstance(graphs, list) else []
            graph = next(
                (
                    item
                    for item in graphs
                    if isinstance(item, dict)
                    and pipeline
                    in {str(item.get("id") or ""), str(item.get("name") or "")}
                ),
                None,
            )
            if graph is None:
                problems.append(
                    f"study.pipeline does not exist in project {project}: {pipeline}"
                )
            elif graph.get("runnable") is False:
                problems.append(f"study.pipeline is not runnable: {pipeline}")
    return problems


def run_confirmed(
    turn: Turn,
    confirmed: dict[str, Any],
    lake: Lake,
) -> dict[str, Any]:
    proposal = (
        confirmed.get("result")
        if isinstance(confirmed.get("result"), dict)
        else confirmed
    )
    study = proposal.get("study") if isinstance(proposal, dict) else None
    explanations = proposal.get("explanations") if isinstance(proposal, dict) else None
    candidate = {
        "kind": "proposal",
        "study": study,
        "explanations": explanations,
    }
    problems = proposal_defects(candidate)
    if not problems and isinstance(study, dict):
        problems.extend(reference_defects(study, lake))
    if problems:
        return question_result(
            "The confirmed study proposal is invalid and was not started: "
            + "; ".join(problems)
        )
    assert isinstance(study, dict)
    created = lake.write("create_study", study)
    study_id = str(created.get("id") or created.get("study") or "")
    if not study_id:
        raise LakeError("create_study", "invalid_response", "study id is missing")
    return {
        "reply": f"The confirmed study is running as {study_id}.".strip(),
        "result": {
            "kind": "running",
            "study": created,
            "confirm": False,
        },
    }


def validation_defects(study: dict[str, Any], lake: Lake) -> list[str]:
    """Run zetta's create_study checks without creating anything (validate_study)."""
    try:
        verdict = lake.read("validate_study", study)
    except LakeError as error:
        if error.code in {"unknown_capability", "not_found", "not_allowed"}:
            # An engine without the preflight: create_study still validates at confirm.
            return []
        return [f"study could not be validated: {error.message}"]
    if verdict.get("valid", True):
        return []
    path = str(verdict.get("path") or "study")
    reason = str(verdict.get("reason") or "the engine rejected the study")
    return [f"{path}: {reason}"]


def study_id_from_context(context: dict[str, Any]) -> str:
    value = context.get("study")
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("id") or value.get("study") or "")
    return str(context.get("study_id") or "")


def watch(turn: Turn, lake: Lake) -> dict[str, Any]:
    study_id = study_id_from_context(turn.context)
    if not study_id:
        return question_result("Which study should I watch?")
    study = lake.read("get_study", {"study": study_id})
    listed = lake.read("list_trials", {"study": study_id})
    trials = listed.get("trials") or study.get("trials") or []
    trials = [trial for trial in trials if isinstance(trial, dict)]
    cursors = turn.context.get("cursors")
    cursors = dict(cursors) if isinstance(cursors, dict) else {}
    events: dict[str, list[dict[str, Any]]] = {}

    study_topic = f"study-events.{study_id}"
    study_batch, study_cursor = receive_topic(lake, study_topic, cursors)
    events[study_topic] = study_batch
    cursors[study_topic] = study_cursor

    stopped: list[dict[str, str]] = []
    for trial in trials:
        if str(trial.get("state") or "").lower() != "running":
            continue
        run = str(trial.get("run") or "")
        if not run:
            continue
        topic = f"run-metrics.{run}"
        batches, cursor = receive_topic(lake, topic, cursors)
        events[topic] = batches
        cursors[topic] = cursor
        reason = divergence_reason(batches)
        if reason:
            lake.write("stop_trial", {"study": study_id, "run": run})
            stopped.append({"run": run, "reason": reason})

    reply = progress_reply(study, trials, stopped)
    return {
        "reply": reply,
        "result": {
            "kind": "status",
            "study": study,
            "trials": trials,
            "events": events,
            "stopped_trials": stopped,
            "cursors": cursors,
            "confirm": False,
        },
    }


def receive_topic(
    lake: Lake,
    topic: str,
    cursors: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    from_sequence = cursors.get(topic, 1)
    if not isinstance(from_sequence, int) or isinstance(from_sequence, bool):
        from_sequence = 1
    value = lake.read(
        "receive_events",
        {
            "topic": topic,
            "from_sequence": from_sequence,
            "max": MAX_EVENTS,
            "wait_ms": EVENT_WAIT_MS,
        },
    )
    raw_events = value.get("events") or []
    events = [event for event in raw_events if isinstance(event, dict)]
    next_sequence = value.get("next_sequence")
    if not isinstance(next_sequence, int) or isinstance(next_sequence, bool):
        next_sequence = from_sequence + len(events)
    return events, next_sequence


def divergence_reason(events: list[dict[str, Any]]) -> str:
    batches = [metric_values(event) for event in events]
    batches = [batch for batch in batches if batch]
    for batch in batches:
        for name, value in batch.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                return f"stopped because {name} is not finite"
    grad_norms = [
        abs(float(value))
        for batch in batches
        for name, value in batch.items()
        if name.endswith("grad_norm") and finite_number(value)
    ]
    if any(value >= EXPLODING_GRAD_NORM for value in grad_norms):
        return "stopped because the gradient norm exploded"
    if len(grad_norms) >= 2:
        baseline = max(grad_norms[0], 1.0)
        if grad_norms[-1] >= 100.0 and grad_norms[-1] >= baseline * 10.0:
            return "stopped because the gradient norm grew more than tenfold"
    gaps = []
    validation_losses = []
    for batch in batches:
        train_loss = named_metric(batch, "train.loss")
        validation_loss = named_metric(batch, "validation.loss", "val.loss")
        if finite_number(train_loss) and finite_number(validation_loss):
            gaps.append(float(validation_loss) - float(train_loss))
            validation_losses.append(float(validation_loss))
    if len(gaps) >= OVERFITTING_POINTS:
        recent_gaps = gaps[-OVERFITTING_POINTS:]
        recent_validation = validation_losses[-OVERFITTING_POINTS:]
        if all(gap >= OVERFITTING_GAP for gap in recent_gaps) and nondecreasing(
            recent_validation
        ):
            return "stopped because validation loss is worsening while the fit overfits"
    return ""


def metric_values(event: dict[str, Any]) -> dict[str, Any]:
    candidates = [event, event.get("data"), event.get("value"), event.get("event")]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        values = candidate.get("values")
        if isinstance(values, dict):
            return values
    return {}


def named_metric(values: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in values:
            return values[name]
    return None


def nondecreasing(values: list[float]) -> bool:
    return all(current >= previous for previous, current in pairwise(values))


def progress_reply(
    study: dict[str, Any],
    trials: list[dict[str, Any]],
    stopped: list[dict[str, str]],
) -> str:
    states: dict[str, int] = {}
    for trial in trials:
        state = str(trial.get("state") or "unknown")
        states[state] = states.get(state, 0) + 1
    summary = ", ".join(f"{count} {state}" for state, count in sorted(states.items()))
    status = str(study.get("status") or "unknown")
    reply = f"Study status is {status}. Trials: {summary or 'none yet'}."
    if stopped:
        reasons = "; ".join(f"{item['run']}: {item['reason']}" for item in stopped)
        reply += f" I stopped {len(stopped)} diverging trial(s): {reasons}."
    budgets = (
        study.get("spec", {}).get("budgets")
        if isinstance(study.get("spec"), dict)
        else {}
    )
    spent = study.get("spent_cost")
    if (
        finite_number(spent)
        and isinstance(budgets, dict)
        and finite_number(budgets.get("cost_cap"))
    ):
        reply += f" Cost is {float(spent):g} of {float(budgets['cost_cap']):g}."
    return reply


def explain(turn: Turn, lake: Lake) -> dict[str, Any]:
    study_id = study_id_from_context(turn.context)
    if not study_id:
        return question_result("Which completed study should I explain?")
    study = lake.read("get_study", {"study": study_id})
    analysis = lake.read("get_study_analysis", {"study": study_id})
    best = analysis.get("best_trial")
    best = best if isinstance(best, dict) else {}
    run = str(best.get("run") or "")
    promotion = {"study": study_id, "run": run} if run else {}
    return {
        "reply": analysis_reply(study, analysis),
        "result": {
            "kind": "explanation",
            "study": study,
            "analysis": analysis,
            "promotion": promotion,
            "confirm": bool(promotion),
        },
    }


def analysis_reply(study: dict[str, Any], analysis: dict[str, Any]) -> str:
    best = analysis.get("best_trial")
    if not isinstance(best, dict) or not best:
        return "The study has no complete trial to explain or promote yet."
    run = str(best.get("run") or "unknown run")
    value = best.get("value")
    params = best.get("params") if isinstance(best.get("params"), dict) else {}
    importances = analysis.get("parameter_importances")
    importances = importances if isinstance(importances, list) else []
    ranked = [item for item in importances if isinstance(item, dict)]
    ranked.sort(key=lambda item: float(item.get("importance") or 0), reverse=True)
    top = str(ranked[0].get("parameter") or "") if ranked else ""
    objective = (
        study.get("spec", {}).get("objective")
        if isinstance(study.get("spec"), dict)
        else {}
    )
    metric = objective.get("metric") if isinstance(objective, dict) else "objective"
    reply = f"Best trial {run} reached {metric}={value} with {params}."
    if top:
        reply += f" {top} mattered most; the next study should narrow its range near the winner."
    else:
        reply += " The next study should add trials before narrowing the search space."
    reply += (
        " Treat the ranking cautiously: trial variance can change the winner, and repeated choices "
        "against one validation split can overuse that split. Confirm the proposed promotion only "
        "after reviewing those caveats."
    )
    return reply


def promote_confirmed(
    turn: Turn,
    confirmed: dict[str, Any],
    lake: Lake,
) -> dict[str, Any]:
    proposal = confirmed.get("promotion")
    if not isinstance(proposal, dict):
        result = confirmed.get("result")
        if isinstance(result, dict):
            proposal = result.get("promotion")
    if not isinstance(proposal, dict):
        proposal = confirmed
    study_id = str(proposal.get("study") or "")
    run = str(proposal.get("run") or "")
    expected = study_id_from_context(turn.context)
    if not study_id or not run or (expected and expected != study_id):
        return question_result(
            "The confirmed promotion is invalid and was not applied."
        )
    analysis = lake.read("get_study_analysis", {"study": study_id})
    best = analysis.get("best_trial")
    best_run = str(best.get("run") or "") if isinstance(best, dict) else ""
    if run != best_run:
        return question_result(
            "The confirmed trial is no longer the study's best trial and was not promoted."
        )
    promoted = lake.write("promote_trial", {"study": study_id, "run": run})
    return {
        "reply": f"Trial {run} is now the promoted model for study {study_id}.",
        "result": {
            "kind": "promoted",
            "study": study_id,
            "promotion": promoted,
            "confirm": False,
        },
    }
