"""HTTP and bus surfaces for ExperimentsAgent."""

from __future__ import annotations

import hmac
import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol

from openai import OpenAI

from agent_experiments import lake_bus
from agent_experiments.model import gateway_client

PROMPT_VERSION = "v1"
TURN_KINDS = {"experiment", "experiment_status", "experiment_explain"}


class TurnRunner(Protocol):
    def __call__(
        self,
        body: dict[str, Any],
        state: State,
        client: OpenAI | None = None,
        lake: Any = None,
    ) -> dict[str, Any]: ...


class State:
    def __init__(self) -> None:
        self.model = os.environ.get("EXPERIMENTS_AGENT_MODEL", "gpt-5.5")
        self.token = os.environ.get("AGENT_SERVICE_TOKEN", "")
        self.lake_url = os.environ.get("LAKE_URL", "")
        self.lock = threading.Lock()
        self.conversations: dict[str, list[dict[str, Any]]] = {}

    def history(self, conversation_id: str) -> list[dict[str, Any]]:
        with self.lock:
            return self.conversations.setdefault(conversation_id, [])


def package_version() -> str:
    try:
        return version("agent-experiments")
    except PackageNotFoundError:
        return "unknown"


def health(state: State) -> dict[str, Any]:
    return {
        "ok": True,
        "agent": "agent-experiments",
        "version": package_version(),
        "model": state.model,
        "prompt_version": PROMPT_VERSION,
    }


def run_turn(
    body: dict[str, Any],
    state: State,
    client: OpenAI | None = None,
    lake: Any = None,
) -> dict[str, Any]:
    kind = str(body.get("kind") or "")
    message = str(body.get("message") or "")
    token = str(body.get("lake_token") or "")
    lake_url = str(body.get("lake_url") or state.lake_url)
    if kind not in TURN_KINDS:
        return {
            "ok": False,
            "error": "kind must be experiment, experiment_status, or experiment_explain",
        }
    if not message or not token or not lake_url:
        return {
            "ok": False,
            "error": "a turn needs a message, the person's lake_token and lake URL",
        }
    from agent_experiments.agent import Turn, answer_turn

    conversation_id = str(body.get("conversation_id") or uuid.uuid4())
    context = body.get("context") if isinstance(body.get("context"), dict) else {}
    turn = Turn(
        kind=kind,
        message=message,
        lake_url=lake_url,
        lake_token=token,
        customer_id=str(body.get("customer_id") or ""),
        conversation_id=conversation_id,
        context=context,
    )
    runtime_client = client
    if kind == "experiment" and not context.get("confirmed_proposal"):
        # The model is reached through lake with the turn's own key, which counts its tokens in
        # the turn's workspace (lake#248); this agent holds no provider key. The turn's lake_url
        # is the workspace's Zetta path, so the gateway is at lake's own root (LAKE_URL).
        runtime_client = runtime_client or gateway_client(state.lake_url or lake_url, token)
    answer = answer_turn(
        turn,
        state.history(conversation_id),
        runtime_client,
        state.model,
        lake,
    )
    # The trace id names the turn; lake stores it with the answer in the turn's workspace Zetta.
    # Nothing is written to the pod's disk: the pod serves every tenant, and the message and the
    # answer are workspace content (agent-experiments#11).
    trace_id = str(uuid.uuid4())
    return {
        "ok": True,
        "conversation_id": conversation_id,
        "trace_id": trace_id,
        **answer,
    }


def turn_refusal(authorization: str | None, token: str) -> tuple[int, str] | None:
    """Why `POST /v1/turn` is refused, as `(status, error)`, or None when the caller may proceed.

    The route fails closed: with `AGENT_SERVICE_TOKEN` unset or empty every turn is refused, so a
    service started without it is never open. The bearer is compared in constant time
    (`hmac.compare_digest` over bytes), so the comparison's timing says nothing about the token.
    """
    if not token:
        return 503, "AGENT_SERVICE_TOKEN is not set; this service refuses every turn"
    expected = f"Bearer {token}".encode()
    supplied = (authorization or "").encode("utf-8", "surrogateescape")
    if not hmac.compare_digest(supplied, expected):
        return 401, "unauthorized"
    return None


def serve() -> int:
    state = State()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send(200, health(state))
                return
            self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/turn":
                self._send(404, {"ok": False, "error": "not found"})
                return
            refusal = turn_refusal(self.headers.get("Authorization"), state.token)
            if refusal is not None:
                status, error = refusal
                self._send(status, {"ok": False, "error": error})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, ValueError):
                self._send(400, {"ok": False, "error": "the body is not JSON"})
                return
            try:
                answer = run_turn(body if isinstance(body, dict) else {}, state)
            except Exception as error:  # noqa: BLE001
                answer = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            self._send(200 if answer.get("ok") else 422, answer)

        def log_message(self, format: str, *args: Any) -> None:
            print(
                json.dumps({"agent": "agent-experiments", "request": format % args}),
                flush=True,
            )

    lake_bus.start_from_env(
        lambda body: run_turn(body, state),
        "agent-experiments",
    )
    port = int(os.environ.get("PORT", "8080"))
    print(json.dumps({"agent": "agent-experiments", "listening": port}), flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
    return 0
