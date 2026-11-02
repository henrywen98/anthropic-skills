"""Regression tests for the Pi CLI adaptation; no model calls."""
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_eval import run_eval, run_single_query
from scripts.improve_description import _call_pi, improve_description


class PiCompatibilityTests(unittest.TestCase):
    def run_query(self, events=None, *, code=0, timeout=False):
        def invoke(cmd, **kwargs):
            self.assertEqual(cmd[0:3], ["pi", "--mode", "json"])
            for flag in ("--no-session", "--no-skills", "--no-context-files", "--no-mcp"):
                self.assertIn(flag, cmd)
            self.assertNotIn("--tools", cmd)  # Keep normal Pi file tools available.
            self.assertNotIn("--no-tools", cmd)
            self.assertEqual(cmd[-2:], ["--", "raw query"])
            skill = Path(cmd[cmd.index("--skill") + 1])
            self.assertIn("name: sample", skill.read_text())
            self.assertEqual(cmd[cmd.index("--model") + 1], "provider/model:medium")
            self.skill_file = skill
            if timeout:
                raise subprocess.TimeoutExpired(cmd, 1)
            stream = events(skill) if callable(events) else events
            return subprocess.CompletedProcess(cmd, code, "\n".join(json.dumps(e) for e in stream), "diagnostic")
        with patch("scripts.run_eval.subprocess.run", side_effect=invoke):
            return run_single_query("raw query", "sample", "Example description", 1, str(Path.cwd()), "provider/model:medium")

    @staticmethod
    def normal():
        return [{"type": "message_end", "message": {"role": "assistant", "stopReason": "stop"}},
                {"type": "agent_settled", "aborted": False}]

    def test_successful_target_read(self):
        def events(skill):
            return [{"type": "tool_execution_start", "toolName": "read", "toolCallId": "a", "args": {"path": str(skill)}},
                    {"type": "tool_execution_end", "toolName": "read", "toolCallId": "a", "isError": False}] + self.normal()
        self.assertTrue(self.run_query(events))
        self.assertFalse(self.skill_file.exists())

    def test_no_read_or_name_mention_is_negative(self):
        events = self.normal()
        events[0]["message"]["content"] = [{"type": "text", "text": "I used sample"}]
        self.assertFalse(self.run_query(events))

    def test_other_path_is_not_a_trigger(self):
        events = [{"type": "tool_execution_start", "toolName": "read", "toolCallId": "a", "args": {"path": "/other/SKILL.md"}},
                  {"type": "tool_execution_end", "toolCallId": "a", "isError": False}] + self.normal()
        self.assertFalse(self.run_query(events))

    def test_invalid_completion_is_not_negative(self):
        for reason in ("error", "aborted", "length", "toolUse", "pending"):
            events = self.normal()
            events[0]["message"]["stopReason"] = reason
            with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                self.run_query(events)
        for events in ([], self.normal()[:1], self.normal()[1:],
                       [self.normal()[0], {"type": "agent_settled", "aborted": True}]):
            with self.assertRaises(RuntimeError):
                self.run_query(events)

    def test_failed_or_unfinished_read_is_invalid(self):
        for failed in (True, False):
            def events(skill):
                result = [{"type": "tool_execution_start", "toolName": "read", "toolCallId": "a", "args": {"path": str(skill)}}]
                if failed:
                    result.append({"type": "tool_execution_end", "toolCallId": "a", "isError": True})
                return result + self.normal()
            with self.subTest(failed=failed), self.assertRaises(RuntimeError):
                self.run_query(events)

    def test_nonzero_and_timeout_propagate_and_cleanup(self):
        for timeout in (False, True):
            with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)):
                self.run_query(self.normal(), code=1, timeout=timeout)
            self.assertFalse(self.skill_file.exists())

    def test_batch_does_not_convert_error_to_negative(self):
        class FailedFuture:
            def result(self):
                raise RuntimeError("invalid model run")
        class Pool:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def submit(self, *args): return FailedFuture()
        with patch("scripts.run_eval.ProcessPoolExecutor", return_value=Pool()), \
             patch("scripts.run_eval.as_completed", side_effect=lambda futures: iter(futures)), \
             self.assertRaisesRegex(RuntimeError, "invalid model run"):
            run_eval([{"query": "negative", "should_trigger": False}], "sample", "desc", 1, 1, Path.cwd())

    def test_text_call_preserves_stdin_and_model(self):
        with patch("scripts.improve_description.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "description", "")) as run:
            self.assertEqual(_call_pi("prompt", "provider/model:medium"), "description")
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[:2], ["pi", "-p"])
            self.assertIn("--no-tools", cmd)
            self.assertIn("--no-skills", cmd)
            self.assertEqual(cmd[-2:], ["--model", "provider/model:medium"])
            self.assertEqual(run.call_args.kwargs["input"], "prompt")

    def test_text_call_error_propagates(self):
        with patch("scripts.improve_description.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "partial", "error")), self.assertRaises(RuntimeError):
            _call_pi("prompt", "model")

    def test_original_description_parser_and_shortening_remain(self):
        results = {"results": [], "summary": {"passed": 0, "total": 1}}
        with patch("scripts.improve_description._call_pi", side_effect=["<new_description>" + "x" * 1025 + "</new_description>", "<new_description>Short description</new_description>"]) as call:
            actual = improve_description("sample", "body", "old", results, [], "model")
            self.assertEqual(actual, "Short description")
            self.assertEqual(call.call_count, 2)


if __name__ == "__main__":
    unittest.main()
