"""Validate run evidence before aggregating; never call a model."""
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def inspect_run(run_dir, grading):
    """Reject invalid executions and scores. Preserve unknown legacy metadata."""
    expectations = grading.get("expectations")
    if not isinstance(expectations, list) or not expectations:
        raise ValueError("No per-assertion grading evidence")
    for item in expectations:
        if (not isinstance(item, dict) or not isinstance(item.get("text"), str)
                or not item["text"].strip() or type(item.get("passed")) is not bool
                or not isinstance(item.get("evidence"), str) or not item["evidence"].strip()):
            raise ValueError("Each assertion needs text, a Boolean result, and evidence")
    passed = sum(item["passed"] for item in expectations)
    expected = {"passed": passed, "failed": len(expectations) - passed,
                "total": len(expectations), "pass_rate": passed / len(expectations)}
    summary = grading.get("summary", {})
    for field in ("passed", "failed", "total"):
        if type(summary.get(field)) is not int or summary[field] != expected[field]:
            raise ValueError(f"Score summary contradicts assertion results: {field}")
    rate = summary.get("pass_rate")
    if (type(rate) not in (int, float) or not math.isfinite(rate)
            or not math.isclose(rate, expected["pass_rate"], abs_tol=1e-8)):
        raise ValueError("Score summary contradicts assertion results: pass_rate")

    verified = False
    status_path = run_dir / "run-status.json"
    if status_path.exists():
        status = read_json(status_path)
        if (status.get("status") != "completed" or status.get("exit_code", 0) != 0
                or status.get("tool_errors", 0) or status.get("parse_errors")):
            raise ValueError("Run status is not a successful execution")
        verified = True

    models = set()
    thinking_recorded = True
    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        last_assistant = None
        settled = None
        # JSONL uses LF, not Python's broader splitlines definition.
        for line in events_path.read_text().split("\n"):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") == "agent_settled":
                settled = event
            if event.get("type") == "tool_execution_end" and event.get("isError"):
                raise ValueError("Execution contains a tool error")
            if event.get("type") == "message_end" and event.get("message", {}).get("role") == "assistant":
                last_assistant = event["message"]
                provider, model = last_assistant.get("provider"), last_assistant.get("model")
                thinking = last_assistant.get("thinkingLevel")
                if provider and model:
                    models.add(f"{provider}/{model}" + (f":{thinking}" if thinking else ""))
                thinking_recorded = thinking_recorded and thinking is not None
        if (not last_assistant or last_assistant.get("stopReason") != "stop"
                or not settled or settled.get("aborted") is not False):
            raise ValueError("Execution did not settle successfully")
        verified = True

    execution_path = run_dir / "execution.json"
    if execution_path.exists():
        execution = read_json(execution_path)
        if execution.get("status") != "completed":
            raise ValueError("Execution metadata is not successful")
        recorded = execution.get("executor_model")
        if recorded:
            if models and models != {recorded}:
                raise ValueError("Recorded model contradicts execution events")
            models.add(recorded)
            thinking_recorded = thinking_recorded and execution.get("thinking_level_recorded", False)
        verified = True

    metadata_path = run_dir / "eval_metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    if "assertions" in metadata and metadata["assertions"] != [item["text"] for item in expectations]:
        raise ValueError("Graded assertions differ from the fixed evaluation assertions")
    return {"summary": expected, "executor_models": sorted(models),
            "thinking_level_recorded": bool(models) and thinking_recorded,
            "execution_verified": verified, "metadata": metadata}


def make_metadata(results, benchmark_dir):
    """Derive counts and model identity from accepted runs, not defaults."""
    flat = [run for runs in results.values() for run in runs]
    ids = sorted({run["eval_id"] for run in flat})
    counts = {config: {str(i): sum(r["eval_id"] == i for r in runs) for i in ids}
              for config, runs in results.items()}
    cell_counts = [n for cells in counts.values() for n in cells.values()]
    uniform = cell_counts[0] if cell_counts and min(cell_counts) > 0 and len(set(cell_counts)) == 1 else None
    models = sorted({model for run in flat for model in run.get("executor_models", [])})
    all_known = bool(flat) and all(run.get("executor_models") for run in flat)
    matched_pairs = (len(results) == 2 and
                     len({frozenset((r["eval_id"], r["run_number"]) for r in runs)
                          for runs in results.values()}) == 1)
    comparable = (all_known and len(models) == 1 and matched_pairs
                  and all(r.get("execution_verified") and r.get("thinking_level_recorded") for r in flat))
    feedback_path = benchmark_dir / "feedback.json"
    review = "pending"
    if feedback_path.exists():
        try:
            feedback = read_json(feedback_path)
            if feedback.get("status") == "complete" and isinstance(feedback.get("reviews"), list):
                review = "submitted"
            elif feedback.get("status") == "in_progress":
                review = "in_progress"
            else:
                review = "invalid"
        except (ValueError, OSError):
            review = "invalid"
    return {"executor_model": models[0] if all_known and len(models) == 1 else None,
            "executor_models": models, "runs_per_configuration": uniform,
            "run_counts": counts, "evals_run": ids,
            "comparison_verified": comparable, "human_review_status": review}
