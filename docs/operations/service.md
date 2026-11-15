# Service and bus operation

ExperimentsAgent serves `POST /v1/turn` and `GET /healthz` on `PORT`, which defaults to `8080`.
`/healthz` answers `{ok, agent, version, model, prompt_version}`; `version` is the installed
package version from `pyproject.toml`.
`AGENT_SERVICE_TOKEN` is required as the bearer on direct turn requests, compared in constant time
(`401` otherwise). The route fails closed: with it unset or empty every turn is refused with `503`.
Each turn carries one of `experiment`, `experiment_status`, or `experiment_explain` in `kind`, plus
the signed-in person's `lake_token`, `lake_url`, and optional `customer_id`. The agent uses that delegated
credential only for the duration of the turn.

The lake agent bus is enabled when `LAKE_URL`, `AGENT_BUS_TOKEN`, and `AGENT_BUS_AGENT` are set.
The worker long-polls `receive_agent_tasks`, claims each task once, and publishes progress followed
by a result or failure. Those three bus calls carry the bus credential alone.

**No task credential reaches this process** (lake#871, lake `docs/api/agent-bus.md`). Each claim
asks for `credentials: by_reference`: Lake keeps the task's credential and answers the body without
it (`lake_bus.task_credentials`). Every call the turn makes to Lake (its capability reads and writes,
and its model calls through the gateway) then carries this agent's bus credential and the header
`X-Lake-Agent-Task: <task id>`, which Lake honours only while that task is claimed and `Running` on
this agent. The bus calls themselves never send the header. A Lake from before lake#871 refuses the
`credentials` field before claiming anything; the claim is then asked again without it, and the
body's own `lake_token` is used as before, with no task header. Lake removes the in-body mode once
every library agent claims by reference.

`EXPERIMENTS_AGENT_MODEL` selects the model.

A turn keeps nothing on the pod's disk (agent-experiments#11). The pod serves every workspace, and
a turn's message and answer are that workspace's content, which belongs in its workspace Zetta. No
trace file is written: the answer goes back on the bus (or in the `/v1/turn` response) with its
`trace_id`, and lake keeps it in the task's record in the turn's workspace Zetta (lake
`docs/api/agent-bus.md`). `TRACE_DIR` is no longer read.

The image runs as UID 10001, exposes port 8080, and is private as
`oramasearch/agent-experiments`. The repository and image must remain private.
