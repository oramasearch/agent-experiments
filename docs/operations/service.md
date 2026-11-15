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
by a result or failure. The bus credential is used only for those three bus capabilities. The
person's delegated credential remains inside the claimed task body.

`EXPERIMENTS_AGENT_MODEL` selects the model.

A turn keeps nothing on the pod's disk. The pod serves every workspace, and
a turn's message and answer are that workspace's content, which belongs in its workspace Zetta. No
trace file is written: the answer goes back on the bus (or in the `/v1/turn` response) with its
`trace_id`, and lake keeps it in the task's record in the turn's workspace Zetta (lake
`docs/api/agent-bus.md`). `TRACE_DIR` is no longer read.

## Conversation history (agent-experiments#35)

A design conversation's messages are kept in the pod's memory between turns, and the pod serves
every workspace, so the history is bound to its owner and bounded (`conversations.py`):

- **Keyed by workspace and conversation.** The key is the turn's `customer_id`, its checked
  workspace path `lake_url`, its `workspace_id` and `user_id` when present, and its
  `conversation_id`. A turn from another workspace that names the same conversation id gets an
  empty history of its own: it neither reads nor writes the first workspace's messages.
- **Bounded.** At most 256 conversations (the least recently used is dropped first); a
  conversation unused for an hour expires; a stored history keeps its system prompt and at most
  60 messages and 200,000 characters after it, dropping the oldest whole exchanges so a tool reply
  never loses its call.
- **No interleaving.** A turn works on a copy taken under the lock and saves it back whole under
  the lock, so two turns on one conversation never mix their messages; the later one is kept.

History is a convenience, not a record: it is lost on restart, on expiry and on eviction, and the
conversation itself is stored by lake in the workspace Zetta. Lake also checks the conversation id
on its side (fixed separately in lake).

The image runs as UID 10001, exposes port 8080, and is private as
`oramasearch/agent-experiments`. The repository and image must remain private.
The image runs as UID 10001, exposes port 8080, and is published as
`oramasearch/agent-experiments`.
