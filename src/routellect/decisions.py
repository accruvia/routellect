"""A/B test decision log — record winners, retire losers, document learnings.

Decisions are stored in data/ab_decisions.json. Each entry records:
- Which variant won and which was retired
- The data that drove the decision (cost, success rate, iterations)
- Why (free-text rationale)

Usage:
  from accruvia_client.decisions import retire_variant, get_decisions, is_retired

  retire_variant("schema", rationale="Fails on complex issues", data={...})
  assert is_retired("schema")
"""

import json
from datetime import UTC, datetime
from pathlib import Path

_DECISIONS_FILE = "data/ab_decisions.json"


def _decisions_path(base_dir: Path | None = None) -> Path:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
    return base_dir / _DECISIONS_FILE


def get_decisions(base_dir: Path | None = None) -> list[dict]:
    path = _decisions_path(base_dir)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def is_retired(variant: str, base_dir: Path | None = None) -> bool:
    return variant in get_retired_variants(base_dir)


def get_retired_variants(base_dir: Path | None = None) -> set[str]:
    return {d["retired"] for d in get_decisions(base_dir) if "retired" in d}


def retire_variant(
    retired: str,
    *,
    winner: str | None = None,
    rationale: str,
    data: dict | None = None,
    base_dir: Path | None = None,
) -> dict:
    """Record a decision to retire a variant.

    Args:
        retired: The variant being retired (e.g. "schema")
        winner: The variant that won (e.g. "direct"), if applicable
        rationale: Why this decision was made
        data: Supporting data (cost comparison, success rates, etc.)
        base_dir: Project root (auto-detected if None)

    Returns:
        The decision record that was written
    """
    path = _decisions_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    decisions = get_decisions(base_dir)

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "retired": retired,
        "rationale": rationale,
    }
    if winner:
        record["winner"] = winner
    if data:
        record["data"] = data

    decisions.append(record)
    path.write_text(json.dumps(decisions, indent=2) + "\n")
    return record


def print_decisions(base_dir: Path | None = None):
    """Print all decisions in human-readable format."""
    decisions = get_decisions(base_dir)
    if not decisions:
        print("No decisions recorded yet.")
        return

    for i, d in enumerate(decisions, 1):
        print(f"\n--- Decision #{i} ({d['timestamp'][:10]}) ---")
        if "winner" in d:
            print(f"  Winner:  {d['winner']}")
        print(f"  Retired: {d['retired']}")
        print(f"  Reason:  {d['rationale']}")
        if "data" in d:
            for k, v in d["data"].items():
                print(f"  {k}: {v}")
