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
