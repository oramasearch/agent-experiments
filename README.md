# agent-experiments

The ExperimentsAgent: designs AutoML studies for a dataset, runs them in Zetta after the person confirms, watches trials and their training metrics live, stops diverging trials, explains the result, and promotes the best trial after confirmation. Built on agent-runtime. Epic: oramasearch/lake#305.

Private. Its image is published **private** as `oramasearch/agent-experiments` on Docker Hub.

## Service

The Python service exposes `POST /v1/turn` and `GET /healthz` on port 8080. It can also receive
`agent_turn` tasks of kinds `experiment`, `experiment_status`, and `experiment_explain` from
lake's agent bus with `AGENT_BUS_TOKEN` and `AGENT_BUS_AGENT`. A person's delegated lake
credential is separate from the service and bus credentials.

Development uses Python 3.12 and uv:

```console
uv sync
uv run pytest
```

See the [documentation index](docs/README.md) for the service contract and study workflow.
