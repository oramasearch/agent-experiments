# agent-experiments

The ExperimentsAgent: designs AutoML studies for a dataset, runs them in Zetta after the person confirms, watches trials and their training metrics live, stops diverging trials, explains the result, and promotes the best trial after confirmation. Built on agent-runtime. Epic: oramasearch/lake#305.

Current release: [v0.2.2](docs/releases/v0.2.2.md).

Private. Its image is published **private** as `oramasearch/agent-experiments` on Docker Hub.

## Service

The Python service exposes `POST /v1/turn` and `GET /healthz` on port 8080; `/healthz` reports
the package `version` and `prompt_version`. It can also receive
`agent_turn` tasks of kinds `experiment`, `experiment_status`, and `experiment_explain` from
lake's agent bus with `AGENT_BUS_TOKEN` and `AGENT_BUS_AGENT`. A person's delegated lake
credential is separate from the service and bus credentials.

Development uses Python 3.12 and uv:

```console
uv sync
uv run pytest
```

See the [documentation index](docs/README.md) for the service contract, study workflow and
[release notes](docs/releases/README.md).

**The model is reached through lake** (lake#248, lake `docs/api/model-gateway.md`): every call goes
to `<LAKE_URL>/openai/v1` with the turn's own `lake_token` as the API key (`model.py`), so lake
counts its tokens in the turn's workspace and enforces the account's model-token limit. The service
holds no `OPENAI_API_KEY`; only the offline evals, which run for no workspace, call OpenAI directly.

**A turn's credential goes only to Lake at this service's own `LAKE_URL`** (#18). The model address
is always `LAKE_URL`. A turn's `lake_url`, the workspace path Lake hands the agent
(`<LAKE_URL>/w/<workspace>/workspaces/<customer>/zetta`), is accepted only on `LAKE_URL`'s origin
(scheme, host and port) under its path, with no user info, query or fragment; any other address
refuses the turn before a request is made. Without a `lake_url`, the turn uses `LAKE_URL` itself.
With `LAKE_URL` unset, every turn is refused.

**Conversation history is bound to the workspace and bounded** (#35). The pod keeps a design
conversation's messages in memory keyed by the turn's customer, workspace and conversation id, so a
turn from another workspace with the same id starts empty. At most 256 conversations are kept, an
idle one expires after an hour, and each keeps at most 60 messages and 200,000 characters after its
system prompt ([service operation](docs/operations/service.md#conversation-history-agent-experiments35)).

## Study and confirmation contract

An initial `experiment` turn reads dataset, project, pipeline, feature, methodology, and optional
solution facts through lake and returns `result.kind: proposal` with `confirm: true`. It defaults a
single relevant project and an out-of-time validation split when dataset time evidence supports
them. A project without a suitable pipeline produces a dataset study instead of a placeholder
pipeline. Every external reference is checked before review and checked again after confirmation.
The search space comes from the methodology's published `study_search_space`, and the complete
study is preflighted with zetta's `validate_study` (the same checks as `create_study`, storing
nothing); a rejection is fed back to the model with its path and reason before a person sees it.
The initial turn performs no action. A follow-up turn must carry the reviewed object in
`context.confirmed_proposal`; only then does the agent call `create_study`.

`experiment_status` tails `study-events.<study>` and `run-metrics.<run>`, reports budget and trial
progress, and stops trials with non-finite loss, exploding gradient norm, or a sustained
overfitting gap. `experiment_explain` reports the best trial, parameter importance, next search,
and caveats. Promotion requires the reviewed object in `context.confirmed_promotion`; plain text
never authorizes `promote_trial`.

## Verification and evals

`uv run pytest` covers the service, bus, proposal schema, confirmation gates, exact lake
capabilities, event cursors, divergence rules, analysis, and promotion. Run
`uv run experiments-agent-evals` for the known-good search-space, non-finite-loss stop, and
promotion-without-confirmation cases against deterministic fake lake facts.
