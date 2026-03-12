# Routellect

`Routellect` is an open-source LLM model selector: anonymous task features in, model decision out.

It is intended to be usable on its own:

- as a library that receives a model universe and returns routing decisions
- as a building block for downstream harnesses and hosted services

## Scope

`Routellect` owns:

- model selection contracts (`set_model_universe`, `select_model`, `record_outcome`)
- routing decision types and outcome recording
- local execution telemetry and cost helpers
- QA review helpers around generated artifacts

`Routellect` does **not** own:

- model discovery (that belongs in the harness or calling application)
- exploration policy (the harness decides when to explore vs exploit)
- workflow control-plane concerns like retries, promotion, or branching across projects
- business-specific orchestration

Those belong in a harness or downstream application.

## Canonical Interface

```python
from routellect import ModelCapability, RoutingDecision

# 1. Tell routellect what models are available
set_model_universe(models: list[ModelCapability])

# 2. Ask for a routing decision
decision: RoutingDecision = select_model(task_fingerprint, constraints)

# 3. Report what happened
record_outcome(decision, outcome_metrics)
```

The harness discovers which models each CLI backend can serve, converts them into `ModelCapability` records, and hands that universe to routellect before any task execution. Routellect treats that as the live universe for the current session — it never infers models itself.

## Harness Integration

`Routellect` ships an optional harness plugin module at `routellect.harness_plugins`.

It provides:

- a project adapter for preparing a disposable per-run Routellect git worktree
- a cognition adapter for project heartbeat reviews

When used with `accruvia-harness`, load it with:

```bash
export ACCRUVIA_PROJECT_ADAPTER_MODULES=routellect.harness_plugins
export ACCRUVIA_COGNITION_MODULES=routellect.harness_plugins
export ROUTELLECT_REPO_ROOT=/path/to/routellect
```

That plugin surface is optional. `Routellect` still works as a standalone package without the harness.

### Why The Harness Uses Disposable Worktrees

When Routellect is driven by a harness, each run gets its own disposable git worktree instead of writing directly into the
main checkout.

That change is intentional and important:

- blocked or failed runs must not dirty the source repo
- parallel tasks need isolated filesystem state
- run artifacts should be inspectable without polluting the branch a developer is using

The harness may discard a failed worktree later, but the main Routellect checkout should stay clean unless a successful
result is promoted deliberately.

## Migration Debt

`runner.py` and the issue registry (`issues.py`) contain legacy issue-runner execution logic. That code is migration debt — it works but does not reflect the intended product boundary. The durable product value is the routing/selection behavior and the telemetry generated from real usage.

## Temporary Server Support

This repo includes a lightweight hosted-service client surface to keep developer velocity high while the surrounding system is being split apart.

That client/server shape is a band-aid, not the long-term center of the product. The durable product value is the routing/runtime behavior and the telemetry/data generated from real usage. The default scaffold client targets `http://localhost:8000`; downstream deployments should configure their own hosted endpoint explicitly.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
PYTHONPATH=src python3 -m pytest tests -q
```
