# Releases

One page per tagged release, newest first. A page states what the release includes, what it does
not include, known issues, verification output, and where the image runs. The first release also
lists every turn kind, Lake capability, route and environment variable of the agent.

A release has one `vX.Y.Z` name shared by the Git tag, the GitHub release and the Docker Hub
image tag `oramasearch/agent-experiments:vX.Y.Z`. The image tag names the digest already built and
verified; it is never produced by a second build.

| Release | Date | Summary |
|---|---|---|
| [v0.2.2](v0.2.2.md) | 2026-10-26 | A patch: the Lake URL is never passed to the agent as its Lake client, so Experiments turns run again |
| [v0.2.1](v0.2.1.md) | 2026-10-26 | A patch: the Lake URL is never passed to the agent as its Lake client, so Experiments turns run again |
| [v0.2.0](v0.2.0.md) | 2026-10-25 | Security fixes: no turn trace on the pod's disk, a constant-time service token that is required, and a turn's credential only to Lake at `LAKE_URL` |
| [v0.1.0](v0.1.0.md) | 2026-10-17 | The first release: the `experiment`, `experiment_status` and `experiment_explain` turns with confirmation gates, the Lake bus worker, the model through Lake's gateway, and the image |

`pyproject.toml`'s `[project].version` is the one place the version is set; `GET /healthz` reports
it, and `tests/test_release.py` fails when that version has no page here.
