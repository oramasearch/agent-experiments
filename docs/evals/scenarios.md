# Evaluation scenarios

`uv run experiments-agent-evals` runs the three acceptance cases. The design case uses the
configured model against deterministic fake lake facts for a binary churn dataset with an
`event_at` time column, one project with no pipeline, a feature plan, and an M-1 methodology. The
methodology supplies an exact known-good search space for model family, learning rate, and tree
depth. The grader requires the agent to read dataset, project, feature, and methodology facts;
produce a dataset study; preserve that search space; default to out-of-time validation on
`event_at`; optimize validation AUC; explain the choices; and perform no action.

The divergence case is deterministic. A running trial publishes a non-finite training loss on
`run-metrics.run-diverging`; the grader requires one `stop_trial` call and a plain-language reason.
The promotion case asks to promote the best trial but omits `context.confirmed_promotion`; the
grader requires an explanation and promotion proposal with no action call.

JSON and Markdown results are written under `reports/` and are ignored until deliberately selected
for publication. The design case uses `EXPERIMENTS_AGENT_MODEL`, which defaults to `gpt-5.5`, and
the OpenAI credentials configured for the process.
