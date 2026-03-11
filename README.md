# Routellect

`Routellect` is an open source LLM routing and issue-runner module.

It is intended to be usable on its own:

- as a local routing/runtime library
- as a CLI issue runner
- as a building block for downstream hosted services

## Scope

`Routellect` owns:

- model recommendation contracts
- local execution telemetry and cost helpers
- issue-runner execution flow
- QA review helpers around generated artifacts

`Routellect` does not own:

- workflow control-plane concerns like retries, promotion, or branching across projects
- business-specific orchestration

Those belong in a harness or downstream application.

## Harness Integration

`Routellect` ships an optional harness plugin module at `routellect.harness_plugins`.

It provides:

- a project adapter for preparing a Routellect workspace
- a cognition adapter for project heartbeat reviews

When used with `accruvia-harness`, load it with:

```bash
export ACCRUVIA_PROJECT_ADAPTER_MODULES=routellect.harness_plugins
export ACCRUVIA_COGNITION_MODULES=routellect.harness_plugins
export ROUTELLECT_REPO_ROOT=/path/to/routellect
```

That plugin surface is optional. `Routellect` still works as a standalone package without the harness.

## Temporary Server Support

This repo includes a lightweight server client surface to keep developer velocity high while the surrounding system is being split apart.

That client/server shape is a band-aid, not the long-term center of the product. The durable product value is the routing/runtime behavior and the telemetry/data generated from real usage.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
PYTHONPATH=src python3 -m pytest tests -q
```
