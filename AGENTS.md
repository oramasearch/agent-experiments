# AGENTS.md: working in agent-experiments

The ExperimentsAgent: designs, runs, watches and explains AutoML studies through lake capabilities only (the same ones on REST and MCP). Built on agent-runtime. Epic: oramasearch/lake#305.

## Non-negotiables (org-wide)
- **No AI attribution** anywhere: no Co-Authored-By, no "Generated with", no mention of AI in commits, PRs, tags or release notes.
- **No time or size estimates.** Do the work; report concrete results.
- **Docs in the same change:** update README.md (and any docs page) with every code change.
- **Feature branch + PR.** Never push to `main`. PRs reference their issue with `Refs #N`, never an auto-close keyword.
- **Private by default.** This repo and its Docker Hub image `oramasearch/agent-experiments` are private; never make either public.
- Never run a formatter over files you did not otherwise change.
- Never write the word "seam": name the mechanism (interface, extension point, boundary).
