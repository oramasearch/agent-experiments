# AGENTS.md: working in agent-experiments

The ExperimentsAgent: designs, runs, watches and explains AutoML studies through Lake capabilities only (the same ones on REST and MCP). Built on agent-runtime. Read `README.md` and `docs/README.md` first.

## Build and test
- Python with uv, never pip: `uv sync`, then `uv run --group dev pytest`.

## Non-negotiables
- **No AI attribution** anywhere: no AI Co-Authored-By trailer, no "Generated with", no mention of AI in commits, PRs, tags or release notes.
- **No time or size estimates.** Do the work; report concrete results.
- **Docs in the same change:** update README.md (and any docs page) with every code change.
- **Feature branch + PR**, Conventional Commits. Never push to the default branch. PRs reference their issue with `Refs #N`, never an auto-close keyword.
- Never run a formatter over files you did not otherwise change.
- Never write the word "seam": name the mechanism (interface, extension point, boundary).
