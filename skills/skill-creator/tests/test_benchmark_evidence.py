"""Report evidence regression tests. No model calls."""
import json
import tempfile
import unittest
from pathlib import Path
from scripts.aggregate_benchmark import generate_benchmark, generate_markdown


class BenchmarkEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def add(self, config="with_skill", number=1, case=1, model="fixture/model",
            thinking="medium", status="completed", suite="basic", events=True):
        run = self.root / f"eval-{case}-fixture" / config / f"run-{number}"
        run.mkdir(parents=True)
        grading = {"expectations": [{"text": "Expected answer", "passed": True, "evidence": "Output is correct"}],
                   "summary": {"passed": 1, "failed": 0, "total": 1, "pass_rate": 1}}
        (run / "grading.json").write_text(json.dumps(grading))
        (run / "run-status.json").write_text(json.dumps({"status": status, "exit_code": 0}))
        (run / "eval_metadata.json").write_text(json.dumps({"eval_id": case, "eval_name": "fixture", "suite": suite}))
        if events:
            provider, model_id = model.split("/", 1)
            entries = [{"type": "message_end", "message": {"role": "assistant", "provider": provider,
                        "model": model_id, "thinkingLevel": thinking, "stopReason": "stop"}},
                       {"type": "agent_settled", "aborted": False}]
            (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in entries))
        return run

    def test_actual_model_count_and_pending_review(self):
        self.add()
        self.add("without_skill")
        report = generate_benchmark(self.root)
        m = report["metadata"]
        self.assertEqual(m["executor_model"], "fixture/model:medium")
        self.assertEqual(m["runs_per_configuration"], 1)
        self.assertTrue(m["comparison_verified"])
        self.assertEqual(m["human_review_status"], "pending")
        self.assertNotIn("<model-name>", generate_markdown(report))

    def test_unknown_model_is_not_invented(self):
        self.add(events=False)
        m = generate_benchmark(self.root)["metadata"]
        self.assertIsNone(m["executor_model"])
        self.assertFalse(m["comparison_verified"])

    def test_mixed_models_are_not_hidden(self):
        self.add()
        self.add("without_skill", model="fixture/other")
        m = generate_benchmark(self.root)["metadata"]
        self.assertIsNone(m["executor_model"])
        self.assertEqual(len(m["executor_models"]), 2)
        self.assertFalse(m["comparison_verified"])

    def test_thinking_level_difference_disables_verified_comparison(self):
        self.add()
        self.add("without_skill", thinking="high")
        self.assertFalse(generate_benchmark(self.root)["metadata"]["comparison_verified"])

    def test_uneven_run_counts(self):
        self.add()
        self.add(number=2)
        self.add("without_skill")
        report = generate_benchmark(self.root)
        self.assertIsNone(report["metadata"]["runs_per_configuration"])
        self.assertEqual(report["metadata"]["run_counts"]["with_skill"]["1"], 2)
        self.assertIn("Uneven or missing", generate_markdown(report))

    def test_invalid_execution_excluded_even_if_grading_passes(self):
        self.add()
        self.add("without_skill", status="invalid")
        report = generate_benchmark(self.root)
        self.assertEqual(len(report["runs"]), 1)
        self.assertEqual(len(report["excluded_runs"]), 1)
        self.assertEqual(report["metadata"]["run_counts"]["without_skill"]["1"], 0)
        self.assertIsNone(report["metadata"]["runs_per_configuration"])

    def test_aborted_event_overrides_completed_status(self):
        run = self.add()
        self.add("without_skill")
        text = (run / "events.jsonl").read_text().replace('"aborted": false', '"aborted": true')
        (run / "events.jsonl").write_text(text)
        self.assertEqual(len(generate_benchmark(self.root)["excluded_runs"]), 1)

    def test_scores_must_match_assertions(self):
        run = self.add()
        self.add("without_skill")
        grading = json.loads((run / "grading.json").read_text())
        grading["summary"]["passed"] = 2
        (run / "grading.json").write_text(json.dumps(grading))
        report = generate_benchmark(self.root)
        self.assertIn("contradicts", report["excluded_runs"][0]["reason"])

    def test_grading_cannot_omit_fixed_assertions(self):
        run = self.add()
        metadata = json.loads((run / "eval_metadata.json").read_text())
        metadata["assertions"] = ["Expected answer", "Additional fixed check"]
        (run / "eval_metadata.json").write_text(json.dumps(metadata))
        report = generate_benchmark(self.root)
        self.assertFalse(report["runs"])
        self.assertIn("fixed evaluation assertions", report["excluded_runs"][0]["reason"])

    def test_excluded_pair_keeps_comparison_unverified(self):
        for config in ("with_skill", "without_skill"):
            self.add(config, case=1)
            self.add(config, case=2, status="invalid")
        report = generate_benchmark(self.root)
        self.assertFalse(report["metadata"]["comparison_verified"])
        self.assertEqual(len(report["excluded_runs"]), 2)

    def test_missing_grading_is_not_success(self):
        run = self.add()
        self.add("without_skill")
        (run / "grading.json").unlink()
        report = generate_benchmark(self.root)
        self.assertEqual(len(report["runs"]), 1)
        self.assertEqual(report["excluded_runs"][0]["reason"], "Missing grading.json")

    def test_submitted_feedback_is_not_approval(self):
        self.add()
        (self.root / "feedback.json").write_text(json.dumps({"status": "complete", "reviews": []}))
        report = generate_benchmark(self.root)
        self.assertEqual(report["metadata"]["human_review_status"], "submitted")
        self.assertTrue(any("does not imply approval" in note for note in report["notes"]))

    def test_suites_are_separate(self):
        self.add(suite="basic")
        self.add(case=2, suite="comparative")
        report = generate_benchmark(self.root)
        self.assertEqual(set(report["suite_summaries"]), {"basic", "comparative"})

    def test_old_skill_is_baseline_not_primary(self):
        self.add("with_skill")
        self.add("old_skill")
        report = generate_benchmark(self.root)
        self.assertEqual(report["runs"][0]["configuration"], "with_skill")

    def test_metadata_cannot_override_observed_model(self):
        run = self.add()
        (run / "execution.json").write_text(json.dumps({"status": "completed", "executor_model": "fixture/wrong:medium"}))
        report = generate_benchmark(self.root)
        self.assertFalse(report["runs"])
        self.assertIn("contradicts", report["excluded_runs"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
