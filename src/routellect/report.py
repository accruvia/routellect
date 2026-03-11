"""A/B test report — aggregate results across runs.

Reads all results from data/ab_tests/{issue_id}/ and prints comparison.

Usage:
  python scripts/ab_report.py 437
  python scripts/ab_report.py  # all issues
"""

import json
import sys
from pathlib import Path

from routellect.telemetry.cost_model import estimate_cost


def load_results(issue_id: str, base_dir: Path) -> list[dict]:
    results_dir = base_dir / "data" / "ab_tests" / issue_id
    if not results_dir.exists():
        return []
    results = []
    for run_dir in sorted(results_dir.iterdir()):
        result_file = run_dir / "result.json"
        if result_file.exists():
            results.append(json.loads(result_file.read_text()))
    return results


def report(issue_id: str, base_dir: Path):
    results = load_results(issue_id, base_dir)
    if not results:
        print(f"No results for issue #{issue_id}")
        return

    by_mode: dict[str, list[dict]] = {}
    for r in results:
        mode = r["mode"]
        by_mode.setdefault(mode, []).append(r)

    modes = sorted(by_mode.keys())
    print(f"\nIssue #{issue_id} — {len(results)} total runs")
    print(f"{'=' * 60}")

    # Header
    header = f"{'Metric':<22}"
    for mode in modes:
        header += f" {mode:>15}"
    print(header)
    print("-" * len(header))

    # Runs
    row = f"{'Runs':<22}"
    for mode in modes:
        row += f" {len(by_mode[mode]):>15}"
    print(row)

    # Success rate
    row = f"{'Success rate':<22}"
    for mode in modes:
        runs = by_mode[mode]
        rate = sum(1 for r in runs if r["success"]) / len(runs) * 100
        row += f" {rate:>14.0f}%"
    print(row)

    # Avg iterations
    row = f"{'Avg iterations':<22}"
    for mode in modes:
        runs = by_mode[mode]
        avg = sum(r["iterations"] for r in runs) / len(runs)
        row += f" {avg:>15.1f}"
    print(row)

    # Avg time
    row = f"{'Avg time (s)':<22}"
    for mode in modes:
        runs = by_mode[mode]
        avg = sum(r["total_time_s"] for r in runs) / len(runs)
        row += f" {avg:>15.1f}"
    print(row)

    # Token metrics
    for metric, key in [
        ("Avg total tokens", "total"),
        ("Avg net input", "net"),
        ("Avg output tokens", "output"),
        ("Avg cached tokens", "cached"),
    ]:
        row = f"{metric:<22}"
        for mode in modes:
            runs = by_mode[mode]
            avg = sum(r["total_tokens"][key] for r in runs) / len(runs)
            row += f" {avg:>15,.0f}"
        print(row)

    # API cost estimates (per-provider, using actual token breakdown)
    provider = "anthropic"  # all runs currently use Claude
    print(f"\n--- API Cost Estimate ({provider}) ---")
    cost_metrics = [
        ("Avg input cost", "input_cost"),
        ("Avg cached cost", "cached_cost"),
        ("Avg output cost", "output_cost"),
        ("Avg total cost", "total_cost"),
        ("Avg cache savings", "savings_from_caching"),
    ]
    for label, key in cost_metrics:
        row = f"{label:<22}"
        for mode in modes:
            runs = by_mode[mode]
            costs = [
                estimate_cost(
                    input_tokens=r["total_tokens"]["input"],
                    cached_tokens=r["total_tokens"]["cached"],
                    output_tokens=r["total_tokens"]["output"],
                    provider=provider,
                )
                for r in runs
            ]
            avg = sum(c[key] for c in costs) / len(costs)
            row += f" ${avg:>14.6f}"
        print(row)

    # Cheapest mode highlight
    mode_totals = {}
    for mode in modes:
        runs = by_mode[mode]
        costs = [
            estimate_cost(
                input_tokens=r["total_tokens"]["input"],
                cached_tokens=r["total_tokens"]["cached"],
                output_tokens=r["total_tokens"]["output"],
                provider=provider,
            )
            for r in runs
        ]
        mode_totals[mode] = sum(c["total_cost"] for c in costs) / len(costs)
    cheapest = min(mode_totals, key=mode_totals.get)
    print(f"\n  Cheapest mode: {cheapest} (${mode_totals[cheapest]:.6f} avg)")

    # Std dev of total tokens (if multiple runs)
    for mode in modes:
        runs = by_mode[mode]
        if len(runs) > 1:
            totals = [r["total_tokens"]["total"] for r in runs]
            mean = sum(totals) / len(totals)
            variance = sum((t - mean) ** 2 for t in totals) / len(totals)
            std = variance**0.5
            print(f"  {mode} token std dev: {std:,.0f}")

    # Per-run detail
    print("\n--- Per-Run Detail ---")
    print(f"{'Mode':<8} {'Run ID':<20} {'OK?':<5} {'Iter':>5} {'Time':>7} {'Total Tok':>12} {'Net Tok':>10}")
    print(f"{'-' * 8} {'-' * 20} {'-' * 5} {'-' * 5} {'-' * 7} {'-' * 12} {'-' * 10}")
    for r in sorted(results, key=lambda x: (x["mode"], x["run_id"])):
        ok = "PASS" if r["success"] else "FAIL"
        print(
            f"{r['mode']:<8} {r['run_id']:<20} {ok:<5} "
            f"{r['iterations']:>5} {r['total_time_s']:>6.0f}s "
            f"{r['total_tokens']['total']:>12,} {r['total_tokens']['net']:>10,}"
        )


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    if len(sys.argv) > 1:
        report(sys.argv[1], base_dir)
    else:
        ab_dir = base_dir / "data" / "ab_tests"
        if ab_dir.exists():
            for issue_dir in sorted(ab_dir.iterdir()):
                if issue_dir.is_dir():
                    report(issue_dir.name, base_dir)
        else:
            print("No ab_tests data found")


if __name__ == "__main__":
    main()
