"""Token source and missing-value regression tests; no model calls."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_benchmark import generate_benchmark, generate_markdown


class BenchmarkTokenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def add_run(self, config, number=1, timing=None, grading_timing=None):
        run = self.root / "eval-0" / config / f"run-{number}"
        run.mkdir(parents=True)
        grading = {
            "summary": {"pass_rate": 1, "passed": 1, "failed": 0, "total": 1},
            "execution_metrics": {"output_chars": 1234},
        }
        if grading_timing is not None:
            grading["timing"] = grading_timing
        (run / "grading.json").write_text(json.dumps(grading))
        if timing is not None:
            (run / "timing.json").write_text(json.dumps(timing))

    def test_missing_tokens_do_not_become_chars_or_zero(self):
        for config, timing in (("with_skill", {"total_duration_seconds": 2}), ("without_skill", None)):
            self.add_run(config, timing=timing)
        report = generate_benchmark(self.root)
        self.assertTrue(all(r["result"]["tokens"] is None for r in report["runs"]))
        for config in ("with_skill", "without_skill"):
            self.assertIsNone(report["run_summary"][config]["tokens"])
        self.assertIsNone(report["run_summary"]["delta"]["tokens"])
        self.assertIn("| Tokens | Not measured | Not measured | Not measured |", generate_markdown(report))

    def test_measured_zero_is_preserved(self):
        self.add_run("with_skill", timing={"total_tokens": 0})
        self.add_run("without_skill", timing={"total_tokens": 10})
        report = generate_benchmark(self.root)
        self.assertEqual(report["runs"][0]["result"]["tokens"], 0)
        self.assertEqual(report["run_summary"]["with_skill"]["tokens"]["mean"], 0)
        self.assertEqual(report["run_summary"]["delta"]["tokens"], "-10")

    def test_tokens_read_even_when_grading_has_duration(self):
        self.add_run("with_skill", timing={"total_tokens": 42, "total_duration_seconds": 2},
                     grading_timing={"total_duration_seconds": 3})
        report = generate_benchmark(self.root)
        self.assertEqual(report["runs"][0]["result"]["tokens"], 42)
        self.assertEqual(report["runs"][0]["result"]["time_seconds"], 3)
        self.assertIsNone(report["run_summary"]["delta"]["tokens"])

    def test_grading_tokens_are_accepted(self):
        self.add_run("with_skill", grading_timing={"total_tokens": 42, "total_duration_seconds": 3})
        report = generate_benchmark(self.root)
        self.assertEqual(report["runs"][0]["result"]["tokens"], 42)

    def test_partial_measurements_do_not_produce_mean_or_delta(self):
        self.add_run("with_skill", 1, timing={"total_tokens": 100})
        self.add_run("with_skill", 2, timing={"total_duration_seconds": 2})
        self.add_run("without_skill", timing={"total_tokens": 50})
        report = generate_benchmark(self.root)
        self.assertIsNone(report["run_summary"]["with_skill"]["tokens"])
        self.assertEqual(report["run_summary"]["without_skill"]["tokens"]["mean"], 50)
        self.assertIsNone(report["run_summary"]["delta"]["tokens"])
        self.assertIn("| Tokens | Not measured | 50 ± 0 | Not measured |", generate_markdown(report))

    def test_complete_measurements_keep_original_statistics(self):
        self.add_run("with_skill", 1, timing={"total_tokens": 100})
        self.add_run("with_skill", 2, timing={"total_tokens": 200})
        self.add_run("without_skill", 1, timing={"total_tokens": 50})
        self.add_run("without_skill", 2, timing={"total_tokens": 50})
        report = generate_benchmark(self.root)
        stats = report["run_summary"]["with_skill"]["tokens"]
        self.assertEqual(stats, {"mean": 150, "stddev": 70.7107, "min": 100, "max": 200})
        self.assertEqual(report["run_summary"]["delta"]["tokens"], "+100")


if __name__ == "__main__":
    unittest.main()
