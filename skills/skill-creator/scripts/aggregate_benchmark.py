#!/usr/bin/env python3
"""
Aggregate individual run results into benchmark summary statistics.

Reads grading.json files from run directories and produces:
- run_summary with mean, stddev, min, max for each metric
- delta between with_skill and without_skill configurations

Usage:
    python aggregate_benchmark.py <benchmark_dir>

Example:
    python aggregate_benchmark.py benchmarks/2026-01-15T10-30-00/

The script supports two directory layouts:

    Workspace layout (from skill-creator iterations):
    <benchmark_dir>/
    └── eval-N/
        ├── with_skill/
        │   ├── run-1/grading.json
        │   └── run-2/grading.json
        └── without_skill/
            ├── run-1/grading.json
            └── run-2/grading.json

    Legacy layout (with runs/ subdirectory):
    <benchmark_dir>/
    └── runs/
        └── eval-N/
            ├── with_skill/
            │   └── run-1/grading.json
            └── without_skill/
                └── run-1/grading.json
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from .benchmark_evidence import inspect_run, make_metadata
except ImportError:
    from benchmark_evidence import inspect_run, make_metadata


def calculate_stats(values: list[float]) -> dict:
    """Calculate mean, stddev, min, max for a list of values."""
    if not values:
        return {"mean": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0}

    n = len(values)
    mean = sum(values) / n

    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        stddev = math.sqrt(variance)
    else:
        stddev = 0.0

    return {
        "mean": round(mean, 4),
        "stddev": round(stddev, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4)
    }


def load_run_results(benchmark_dir: Path, excluded: list | None = None) -> dict:
    """
    Load all run results from a benchmark directory.

    Returns dict keyed by config name (e.g. "with_skill"/"without_skill",
    or "new_skill"/"old_skill"), each containing a list of run results.
    """
    # Support both layouts: eval dirs directly under benchmark_dir, or under runs/
    runs_dir = benchmark_dir / "runs"
    if runs_dir.exists():
        search_dir = runs_dir
    elif list(benchmark_dir.glob("eval-*")):
        search_dir = benchmark_dir
    else:
        print(f"No eval directories found in {benchmark_dir} or {benchmark_dir / 'runs'}")
        return {}

    results: dict[str, list] = {}
    excluded = excluded if excluded is not None else []

    for eval_idx, eval_dir in enumerate(sorted(search_dir.glob("eval-*"))):
        metadata_path = eval_dir / "eval_metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path) as mf:
                    metadata = json.load(mf)
                    eval_id = metadata.get("eval_id", eval_idx)
            except (json.JSONDecodeError, OSError):
                eval_id = eval_idx
        else:
            try:
                eval_id = int(eval_dir.name.split("-")[1])
            except ValueError:
                eval_id = eval_idx

        # Discover config directories dynamically rather than hardcoding names
        for config_dir in sorted(eval_dir.iterdir(), key=lambda p: (p.name not in ("with_skill", "new_skill"), p.name)):
            if not config_dir.is_dir():
                continue
            # Skip non-config directories (inputs, outputs, etc.)
            if not list(config_dir.glob("run-*")):
                continue
            config = config_dir.name
            if config not in results:
                results[config] = []

            for run_dir in sorted(config_dir.glob("run-*")):
                run_number = int(run_dir.name.split("-")[1])
                grading_file = run_dir / "grading.json"

                if not grading_file.exists():
                    excluded.append({"path": str(run_dir.relative_to(benchmark_dir)), "reason": "Missing grading.json"})
                    continue

                try:
                    with open(grading_file) as f:
                        grading = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    excluded.append({"path": str(run_dir.relative_to(benchmark_dir)), "reason": str(e)})
                    continue

                try:
                    evidence = inspect_run(run_dir, grading)
                    case_metadata = {**metadata, **evidence["metadata"]}
                    if case_metadata.get("eval_id", eval_id) != eval_id:
                        raise ValueError("Eval ID contradicts directory metadata")
                except (ValueError, OSError, TypeError) as e:
                    excluded.append({"path": str(run_dir.relative_to(benchmark_dir)), "reason": str(e)})
                    continue
                result = {
                    "eval_id": eval_id,
                    "eval_name": case_metadata.get("eval_name", eval_dir.name),
                    "suite": case_metadata.get("suite", "unspecified"),
                    "run_number": run_number,
                    **evidence["summary"],
                    "executor_models": evidence["executor_models"],
                    "thinking_level_recorded": evidence["thinking_level_recorded"],
                    "execution_verified": evidence["execution_verified"],
                    "grading_path": str(grading_file.relative_to(benchmark_dir)),
                }

                # Extract timing — check grading.json first, then sibling timing.json
                timing = grading.get("timing", {})
                result["time_seconds"] = timing.get("total_duration_seconds", 0.0)
                result["tokens"] = timing.get("total_tokens")
                timing_file = run_dir / "timing.json"
                if timing_file.exists():
                    try:
                        with open(timing_file) as tf:
                            timing_data = json.load(tf)
                        if result["time_seconds"] == 0.0:
                            result["time_seconds"] = timing_data.get("total_duration_seconds", 0.0)
                        if result["tokens"] is None:
                            result["tokens"] = timing_data.get("total_tokens")
                    except json.JSONDecodeError:
                        pass

                # Extract metrics if available. Characters are not token usage.
                metrics = grading.get("execution_metrics", {})
                result["tool_calls"] = metrics.get("total_tool_calls", 0)
                result["errors"] = metrics.get("errors_encountered", 0)

                # inspect_run checked score totals against these assertions.
                result["expectations"] = grading["expectations"]

                # Extract notes from user_notes_summary
                notes_summary = grading.get("user_notes_summary", {})
                notes = []
                notes.extend(notes_summary.get("uncertainties", []))
                notes.extend(notes_summary.get("needs_review", []))
                notes.extend(notes_summary.get("workarounds", []))
                result["notes"] = notes

                results[config].append(result)

    return results


def aggregate_results(results: dict) -> dict:
    """
    Aggregate run results into summary statistics.

    Returns run_summary with stats for each configuration and delta.
    """
    run_summary = {}
    configs = list(results.keys())

    for config in configs:
        runs = results.get(config, [])

        if not runs:
            run_summary[config] = {
                "pass_rate": {"mean": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0},
                "time_seconds": {"mean": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0},
                "tokens": None
            }
            continue

        pass_rates = [r["pass_rate"] for r in runs]
        times = [r["time_seconds"] for r in runs]
        tokens = [r.get("tokens") for r in runs]

        run_summary[config] = {
            "pass_rate": calculate_stats(pass_rates),
            "time_seconds": calculate_stats(times),
            # Partial measurements must not look like a complete-run average.
            "tokens": calculate_stats(tokens) if all(t is not None for t in tokens) else None
        }

    # Calculate delta between the first two configs (if two exist)
    if len(configs) >= 2:
        primary = run_summary.get(configs[0], {})
        baseline = run_summary.get(configs[1], {})
    else:
        primary = run_summary.get(configs[0], {}) if configs else {}
        baseline = {}

    delta_pass_rate = primary.get("pass_rate", {}).get("mean", 0) - baseline.get("pass_rate", {}).get("mean", 0)
    delta_time = primary.get("time_seconds", {}).get("mean", 0) - baseline.get("time_seconds", {}).get("mean", 0)
    primary_tokens = primary.get("tokens")
    baseline_tokens = baseline.get("tokens")
    delta_tokens = (primary_tokens["mean"] - baseline_tokens["mean"]
                    if primary_tokens is not None and baseline_tokens is not None else None)

    run_summary["delta"] = {
        "pass_rate": f"{delta_pass_rate:+.2f}",
        "time_seconds": f"{delta_time:+.1f}",
        "tokens": f"{delta_tokens:+.0f}" if delta_tokens is not None else None
    }

    return run_summary


def generate_benchmark(benchmark_dir: Path, skill_name: str = "", skill_path: str = "") -> dict:
    """
    Generate complete benchmark.json from run results.
    """
    excluded = []
    results = load_run_results(benchmark_dir, excluded)
    derived_metadata = make_metadata(results, benchmark_dir)
    results = {config: runs for config, runs in results.items() if runs}
    run_summary = aggregate_results(results)

    # Build runs array for benchmark.json
    runs = []
    for config in results:
        for result in results[config]:
            runs.append({
                "eval_id": result["eval_id"],
                "eval_name": result["eval_name"],
                "suite": result["suite"],
                "executor_models": result["executor_models"],
                "grading_path": result["grading_path"],
                "configuration": config,
                "run_number": result["run_number"],
                "result": {
                    "pass_rate": result["pass_rate"],
                    "passed": result["passed"],
                    "failed": result["failed"],
                    "total": result["total"],
                    "time_seconds": result["time_seconds"],
                    "tokens": result.get("tokens"),
                    "tool_calls": result.get("tool_calls", 0),
                    "errors": result.get("errors", 0)
                },
                "expectations": result["expectations"],
                "notes": result["notes"]
            })

    # Determine eval IDs from results
    eval_ids = sorted(set(
        r["eval_id"]
        for config in results.values()
        for r in config
    ))

    benchmark = {
        "metadata": {
            "skill_name": skill_name or benchmark_dir.name,
            "skill_path": skill_path or None,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **derived_metadata
        },
        "runs": runs,
        "run_summary": run_summary,
        "excluded_runs": excluded,
        "suite_summaries": {},
        "notes": []
    }
    for suite in sorted({run["suite"] for run in runs}):
        subset = {config: [r for r in records if r["suite"] == suite] for config, records in results.items()}
        benchmark["suite_summaries"][suite] = aggregate_results({c: r for c, r in subset.items() if r})
    if excluded:
        derived_metadata["comparison_verified"] = False
        benchmark["metadata"]["comparison_verified"] = False
        benchmark["notes"].append(f"Excluded {len(excluded)} runs with missing, invalid, or inconsistent evidence. See excluded_runs.")
    if not derived_metadata["comparison_verified"]:
        benchmark["notes"].append("Matched execution, model, and thinking-level evidence is incomplete or differs. Deltas are not a verified comparison.")
    if len(results) >= 2 and runs and all(r["result"]["pass_rate"] == 1 for r in runs):
        benchmark["notes"].append("Both conditions pass all assertions. Added value is not established by these cases.")
    benchmark["notes"].append(f"Human review: {derived_metadata['human_review_status']}. Submitted feedback does not imply approval.")
    return benchmark


def generate_markdown(benchmark: dict) -> str:
    """Generate human-readable benchmark.md from benchmark data."""
    metadata = benchmark["metadata"]
    run_summary = benchmark["run_summary"]

    # Determine config names (excluding "delta")
    configs = [k for k in run_summary if k != "delta"]
    config_a = configs[0] if len(configs) >= 1 else "config_a"
    config_b = configs[1] if len(configs) >= 2 else "config_b"
    label_a = config_a.replace("_", " ").title()
    label_b = config_b.replace("_", " ").title()

    lines = [
        f"# Skill Benchmark: {metadata['skill_name']}",
        "",
        f"**Model**: {metadata['executor_model'] or 'Not uniformly recorded; see executor_models'}",
        f"**Date**: {metadata['timestamp']}",
        f"**Evals**: {', '.join(map(str, metadata['evals_run']))}",
        f"**Runs per eval and configuration**: {metadata['runs_per_configuration'] if metadata['runs_per_configuration'] is not None else 'Uneven or missing; see run_counts'}",
        f"**Human review**: {metadata.get('human_review_status', 'pending')}",
        f"**Actual run counts**: `{json.dumps(metadata.get('run_counts', {}), sort_keys=True)}`",
        "",
        "## Summary",
        "",
        f"| Metric | {label_a} | {label_b} | Delta |",
        "|--------|------------|---------------|-------|",
    ]

    a_summary = run_summary.get(config_a, {})
    b_summary = run_summary.get(config_b, {})
    delta = run_summary.get("delta", {})

    # Format pass rate
    a_pr = a_summary.get("pass_rate", {})
    b_pr = b_summary.get("pass_rate", {})
    lines.append(f"| Pass Rate | {a_pr.get('mean', 0)*100:.0f}% ± {a_pr.get('stddev', 0)*100:.0f}% | {b_pr.get('mean', 0)*100:.0f}% ± {b_pr.get('stddev', 0)*100:.0f}% | {delta.get('pass_rate', '—')} |")

    # Format time
    a_time = a_summary.get("time_seconds", {})
    b_time = b_summary.get("time_seconds", {})
    lines.append(f"| Time | {a_time.get('mean', 0):.1f}s ± {a_time.get('stddev', 0):.1f}s | {b_time.get('mean', 0):.1f}s ± {b_time.get('stddev', 0):.1f}s | {delta.get('time_seconds', '—')}s |")

    # Format tokens
    def format_tokens(stats):
        if stats is None:
            return "Not measured"
        return f"{stats['mean']:.0f} ± {stats['stddev']:.0f}"

    lines.append(f"| Tokens | {format_tokens(a_summary.get('tokens'))} | {format_tokens(b_summary.get('tokens'))} | {delta.get('tokens') or 'Not measured'} |")

    for suite, summary in benchmark.get("suite_summaries", {}).items():
        lines.extend(["", f"## Suite: {suite}", "", "| Configuration | Assertion pass rate |", "|---|---:|"])
        for config, stats in summary.items():
            if config != "delta":
                lines.append(f"| {config} | {stats['pass_rate']['mean']:.1%} |")

    # Notes section
    if benchmark.get("notes"):
        lines.extend([
            "",
            "## Notes",
            ""
        ])
        for note in benchmark["notes"]:
            lines.append(f"- {note}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark run results into summary statistics"
    )
    parser.add_argument(
        "benchmark_dir",
        type=Path,
        help="Path to the benchmark directory"
    )
    parser.add_argument(
        "--skill-name",
        default="",
        help="Name of the skill being benchmarked"
    )
    parser.add_argument(
        "--skill-path",
        default="",
        help="Path to the skill being benchmarked"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for benchmark.json (default: <benchmark_dir>/benchmark.json)"
    )

    args = parser.parse_args()

    if not args.benchmark_dir.exists():
        print(f"Directory not found: {args.benchmark_dir}")
        sys.exit(1)

    # Generate benchmark
    benchmark = generate_benchmark(args.benchmark_dir, args.skill_name, args.skill_path)

    if not benchmark["runs"]:
        print("No valid graded runs. No success report was generated.")
        for excluded in benchmark["excluded_runs"]:
            print(f"  {excluded['path']}: {excluded['reason']}")
        sys.exit(1)

    # Determine output paths
    output_json = args.output or (args.benchmark_dir / "benchmark.json")
    output_md = output_json.with_suffix(".md")

    # Write benchmark.json
    with open(output_json, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"Generated: {output_json}")

    # Write benchmark.md
    markdown = generate_markdown(benchmark)
    with open(output_md, "w") as f:
        f.write(markdown)
    print(f"Generated: {output_md}")

    # Print summary
    run_summary = benchmark["run_summary"]
    configs = [k for k in run_summary if k != "delta"]
    delta = run_summary.get("delta", {})

    print(f"\nSummary:")
    for config in configs:
        pr = run_summary[config]["pass_rate"]["mean"]
        label = config.replace("_", " ").title()
        print(f"  {label}: {pr*100:.1f}% pass rate")
    print(f"  Delta:         {delta.get('pass_rate', '—')}")


if __name__ == "__main__":
    main()
