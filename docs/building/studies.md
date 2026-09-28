# Designing, running, watching, and explaining studies

An `experiment` turn may read `analyze_dataset`, `plan_features`, `list_methodologies`, and
`get_solution` through lake with the signed-in person's delegated credential. Its proposal names
the project, dataset, methodology, non-test objective, ordered search space, sampler, pruner,
finite budgets, and seed. It also explains every choice in plain words.

The answer contract is:

```json
{"reply":"...","result":{"kind":"question|proposal|running","study":{},"confirm":false}}
```

An initial turn performs no action. `confirm` is true only for a validated proposal. Text such as
“yes” does not authorize a study. Lake must send the reviewed object back on a later `experiment`
turn as `context.confirmed_proposal`; the agent validates it again and calls `create_study` with
the proposal's exact study object.

An `experiment_status` turn names the study in `context.study`. It reads `get_study` and
`list_trials`, then tails `study-events.<study>` and every active `run-metrics.<run>` through
bounded `receive_events` calls. The result returns per-topic cursors for the next turn. The engine
enforces trial, wall-clock, parallel, and cost admission limits. The agent additionally calls
`stop_trial` when a live batch contains a non-finite value, a gradient norm at or above 1,000, a
tenfold gradient-norm jump to at least 100, or three worsening validation-loss points whose gap
over training loss is at least 0.25.

An `experiment_explain` turn reads `get_study` and `get_study_analysis`. It reports the best trial,
parameter importance, a next search direction, trial-variance risk, and validation-set overuse.
When a best trial exists it returns a promotion proposal with `confirm: true`; it does not call
`promote_trial`. Lake must send that exact object on a later turn as
`context.confirmed_promotion`. The agent validates the study and run before promotion.

Every study and run operation uses lake's capability surface. The agent has no direct Zetta,
database, scheduler, or event-broker connection.
