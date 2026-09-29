# Designing, running, watching, and explaining studies

An `experiment` turn may read `analyze_dataset`, `plan_features`, `list_projects`, `get_project`,
`list_methodologies`, and `get_solution` through lake with the signed-in person's delegated
credential. It uses a named project or the single relevant project without asking. A project and
pipeline appear in the study only when `get_project` returns that runnable pipeline. Otherwise the
proposal is dataset-based, with empty project and pipeline fields.

When dataset analysis identifies a datetime column and no split was requested, the proposal uses
an out-of-time validation split: earlier rows train the trials, the last period before test is the
validation period, and the final period remains untouched. The explanation names the time column
and ordering so the person can edit the default. The agent asks only when lake facts provide no
sensible project, validation, objective, or budget default.

Before displaying a proposal, the agent resolves its dataset, runnable methodology, and any
project/pipeline pair through lake. It repeats those checks on the reviewed proposal before
`create_study`, so a deleted or edited reference cannot reach the action call. The proposal also
names the non-test objective, ordered search space, sampler, pruner, finite budgets, and seed, and
explains every choice in plain words.

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
