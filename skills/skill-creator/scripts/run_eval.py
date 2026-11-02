#!/usr/bin/env python3
"""Run trigger evaluation for a skill description.

Tests whether a skill's description causes Pi to trigger (read the skill)
for a set of queries. Outputs results as JSON.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scripts.utils import parse_skill_md


def find_project_root() -> Path:
    """Use the caller's working directory for independent Pi runs."""
    return Path.cwd()


def run_single_query(
    query: str,
    skill_name: str,
    skill_description: str,
    timeout: int,
    project_root: str,
    model: str | None = None,
) -> bool:
    """Advertise only the candidate skill and detect a successful Pi read.

    Wait for completion: a timeout or failed model run is not a negative.
    The query stays unchanged so loading the skill remains the model's choice.
    """
    with tempfile.TemporaryDirectory(prefix="skill-trigger-") as temporary:
        skill_file = Path(temporary) / skill_name / "SKILL.md"
        skill_file.parent.mkdir()
        indented_desc = "\n  ".join(skill_description.split("\n"))
        skill_file.write_text(
            f"---\nname: {skill_name}\ndescription: |\n  {indented_desc}\n---\n\n"
            f"# {skill_name}\n\nThis skill handles: {skill_description}\n"
        )
        cmd = [
            "pi", "--mode", "json", "--no-session", "--no-skills",
            "--no-context-files", "--no-mcp",
            "--skill", str(skill_file),
        ]
        if model:
            cmd.extend(["--model", model])
        cmd.extend(["--", query])
        result = subprocess.run(
            cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            cwd=project_root, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Pi exited {result.returncode}: {result.stderr}")

        # Split only on LF: Unicode line separators can occur in JSON strings.
        events = [json.loads(line) for line in result.stdout.split("\n") if line.strip()]
        settled = [e for e in events if e.get("type") == "agent_settled"]
        messages = [e["message"] for e in events
                    if e.get("type") == "message_end"
                    and e.get("message", {}).get("role") == "assistant"]
        if (not settled or settled[-1].get("aborted") is not False
                or not messages or messages[-1].get("stopReason") != "stop"):
            raise RuntimeError("Pi did not complete normally")

        reads = set()
        completed = set()
        for event in events:
            if event.get("type") == "tool_execution_start" and event.get("toolName") == "read":
                path = event.get("args", {}).get("path")
                if path:
                    target = Path(path).expanduser()
                    if not target.is_absolute():
                        target = Path(project_root) / target
                    if target.resolve() == skill_file.resolve():
                        reads.add(event["toolCallId"])
            elif event.get("type") == "tool_execution_end":
                if event.get("isError") is not False:
                    raise RuntimeError("Pi tool execution failed")
                completed.add(event["toolCallId"])
        if not reads.issubset(completed):
            raise RuntimeError("Pi skill read did not complete")
        return bool(reads & completed)


def run_eval(
    eval_set: list[dict],
    skill_name: str,
    description: str,
    num_workers: int,
    timeout: int,
    project_root: Path,
    runs_per_query: int = 1,
    trigger_threshold: float = 0.5,
    model: str | None = None,
) -> dict:
    """Run the full eval set and return results."""
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_info = {}
        for item in eval_set:
            for run_idx in range(runs_per_query):
                future = executor.submit(
                    run_single_query,
                    item["query"],
                    skill_name,
                    description,
                    timeout,
                    str(project_root),
                    model,
                )
                future_to_info[future] = (item, run_idx)

        query_triggers: dict[str, list[bool]] = {}
        query_items: dict[str, dict] = {}
        for future in as_completed(future_to_info):
            item, _ = future_to_info[future]
            query = item["query"]
            query_items[query] = item
            if query not in query_triggers:
                query_triggers[query] = []
            # Propagate invalid runs; counting them as False rewards negative cases.
            query_triggers[query].append(future.result())

    for query, triggers in query_triggers.items():
        item = query_items[query]
        trigger_rate = sum(triggers) / len(triggers)
        should_trigger = item["should_trigger"]
        if should_trigger:
            did_pass = trigger_rate >= trigger_threshold
        else:
            did_pass = trigger_rate < trigger_threshold
        results.append({
            "query": query,
            "should_trigger": should_trigger,
            "trigger_rate": trigger_rate,
            "triggers": sum(triggers),
            "runs": len(triggers),
            "pass": did_pass,
        })

    passed = sum(1 for r in results if r["pass"])
    total = len(results)

    return {
        "skill_name": skill_name,
        "description": description,
        "results": results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run trigger evaluation for a skill description")
    parser.add_argument("--eval-set", required=True, help="Path to eval set JSON file")
    parser.add_argument("--skill-path", required=True, help="Path to skill directory")
    parser.add_argument("--description", default=None, help="Override description to test")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout per query in seconds")
    parser.add_argument("--runs-per-query", type=int, default=3, help="Number of runs per query")
    parser.add_argument("--trigger-threshold", type=float, default=0.5, help="Trigger rate threshold")
    parser.add_argument("--model", default=None, help="Pi model, optionally provider/id:thinking (default: user's configured model)")
    parser.add_argument("--verbose", action="store_true", help="Print progress to stderr")
    args = parser.parse_args()

    eval_set = json.loads(Path(args.eval_set).read_text())
    skill_path = Path(args.skill_path)

    if not (skill_path / "SKILL.md").exists():
        print(f"Error: No SKILL.md found at {skill_path}", file=sys.stderr)
        sys.exit(1)

    name, original_description, content = parse_skill_md(skill_path)
    description = args.description or original_description
    project_root = find_project_root()

    if args.verbose:
        print(f"Evaluating: {description}", file=sys.stderr)

    output = run_eval(
        eval_set=eval_set,
        skill_name=name,
        description=description,
        num_workers=args.num_workers,
        timeout=args.timeout,
        project_root=project_root,
        runs_per_query=args.runs_per_query,
        trigger_threshold=args.trigger_threshold,
        model=args.model,
    )

    if args.verbose:
        summary = output["summary"]
        print(f"Results: {summary['passed']}/{summary['total']} passed", file=sys.stderr)
        for r in output["results"]:
            status = "PASS" if r["pass"] else "FAIL"
            rate_str = f"{r['triggers']}/{r['runs']}"
            print(f"  [{status}] rate={rate_str} expected={r['should_trigger']}: {r['query'][:70]}", file=sys.stderr)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
