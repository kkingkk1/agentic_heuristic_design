import json
import sys
import unittest
from unittest.mock import patch

from scripts.p0_controller import StopRun
from scripts.p0_runtime import Runtime
from scripts.p1_classified_runtime import ClassifiedModelMixin, classify_failure
from scripts import p1_classified_runner as runner
from tests.test_p0_controller import CONFIG, ROOT, MemoryLog, small_config


PLUGIN_WARNING = ("WARN codex_core_plugins::manager: failed to refresh cached remote plugin catalog "
                  "error=error sending request for url /backend-api/ps/plugins/list")
POLICY_MESSAGE = "Invalid prompt: your prompt was flagged as potentially violating our usage policy."


def failure(message, stderr=PLUGIN_WARNING):
    events = [{"type": "thread.started"}, {"type": "error", "message": message},
              {"type": "turn.failed", "error": {"message": message}}]
    return {"exit_code": 1, "stdout": "\n".join(json.dumps(x) for x in events), "stderr": stderr}


def success(stderr=""):
    events = [{"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}},
              {"type": "turn.completed", "usage": {"input_tokens": 1}}]
    return {"exit_code": 0, "stdout": "\n".join(json.dumps(x) for x in events), "stderr": stderr}


class ClassifiedFixtureRuntime(ClassifiedModelMixin, Runtime):
    pass


class FailureClassificationTests(unittest.TestCase):
    def test_structured_policy_error_wins_over_plugin_network_warning(self):
        classified = classify_failure(failure(POLICY_MESSAGE))
        self.assertEqual((classified["category"], classified["source"]), ("policy", "turn.failed"))
        self.assertEqual(classified["plugin_warnings"], [PLUGIN_WARNING])

    def test_policy_rejection_never_uses_network_retry_budget(self):
        log = MemoryLog()
        backend = ClassifiedFixtureRuntime(ROOT, CONFIG, log, cli=["fixture-cli"])
        with patch("scripts.p0_model_probe.run_command", side_effect=[failure(POLICY_MESSAGE), success()]) as runner:
            with self.assertRaisesRegex(StopRun, "安全审核拒绝"):
                backend.model("fixture", "prompt")
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(backend.budget.model_retries, 0)
        self.assertTrue(any(x["event"] == "background_warning" and x["component"] == "plugin_catalog"
                            for x in log.items))
        self.assertTrue(any(x["event"] == "model_failure_classified" and x["category"] == "policy"
                            for x in log.items))
        self.assertFalse(any(x["event"].startswith("infrastructure_retry") for x in log.items))

    def test_structured_network_error_remains_network(self):
        result = failure("Connection failed: error sending request")
        self.assertEqual(classify_failure(result)["category"], "network")
        log = MemoryLog()
        backend = ClassifiedFixtureRuntime(ROOT, small_config(), log, cli=["fixture-cli"])
        with patch("scripts.p0_model_probe.run_command", return_value=result) as runner:
            with self.assertRaisesRegex(StopRun, "模型连接失败"):
                backend.model("fixture", "prompt")
        self.assertEqual(runner.call_count, 1)
        self.assertTrue(any(x["event"] == "infrastructure_retry_unavailable" for x in log.items))

    def test_real_network_failure_uses_existing_retry_allowance(self):
        log = MemoryLog()
        backend = ClassifiedFixtureRuntime(ROOT, CONFIG, log, cli=["fixture-cli"])
        replies = [{"exit_code": 1, "stdout": "", "stderr": "connection refused"}, success()]
        with patch("scripts.p0_model_probe.run_command", side_effect=replies) as runner:
            self.assertEqual(backend.model("fixture", "prompt"), "answer")
        self.assertEqual(runner.call_count, 2)
        self.assertEqual((backend.budget.model_normal, backend.budget.model_retries), (1, 1))

    def test_plugin_only_failure_is_not_a_model_network_or_quota_error(self):
        result = {"exit_code": 1, "stdout": "", "stderr": PLUGIN_WARNING + " 429"}
        self.assertEqual(classify_failure(result)["category"], "unknown")
        log = MemoryLog()
        backend = ClassifiedFixtureRuntime(ROOT, CONFIG, log, cli=["fixture-cli"])
        with patch("scripts.p0_model_probe.run_command", return_value=result) as runner:
            with self.assertRaisesRegex(StopRun, "模型调用失败或事件不可信"):
                backend.model("fixture", "prompt")
        self.assertEqual(runner.call_count, 1)
        self.assertFalse(any(x["event"].startswith("infrastructure_retry") for x in log.items))

    def test_structured_quota_error_remains_quota(self):
        result = failure("429 rate limit")
        self.assertEqual(classify_failure(result)["category"], "quota")
        backend = ClassifiedFixtureRuntime(ROOT, CONFIG, MemoryLog(), cli=["fixture-cli"])
        with patch("scripts.p0_model_probe.run_command", return_value=result) as runner:
            with self.assertRaisesRegex(StopRun, "额度或限流"):
                backend.model("fixture", "prompt")
        self.assertEqual(runner.call_count, 1)

    def test_plugin_warning_does_not_stop_a_successful_model_call(self):
        log = MemoryLog()
        backend = ClassifiedFixtureRuntime(ROOT, small_config(), log, cli=["fixture-cli"])
        with patch("scripts.p0_model_probe.run_command", return_value=success(PLUGIN_WARNING)):
            self.assertEqual(backend.model("fixture", "prompt"), "answer")
        self.assertTrue(any(x["event"] == "background_warning" for x in log.items))
        self.assertFalse(any(x["event"] == "model_failure_classified" for x in log.items))

    def test_successful_request_matches_frozen_runtime_command_and_budget(self):
        original = Runtime(ROOT, small_config(), MemoryLog(), cli=["fixture-cli"])
        classified = ClassifiedFixtureRuntime(ROOT, small_config(), MemoryLog(), cli=["fixture-cli"])
        with patch("scripts.p0_model_probe.run_command", return_value=success()) as run_original:
            self.assertEqual(original.model("fixture", "prompt"), "answer")
        with patch("scripts.p0_model_probe.run_command", return_value=success()) as run_classified:
            self.assertEqual(classified.model("fixture", "prompt"), "answer")
        self.assertEqual(run_original.call_args, run_classified.call_args)
        self.assertEqual((original.budget.model_normal, original.budget.model_retries),
                         (classified.budget.model_normal, classified.budget.model_retries))

    def test_unknown_terminal_error_is_not_overridden_by_stderr(self):
        result = failure("model unavailable", "connection refused")
        self.assertEqual(classify_failure(result)["category"], "unknown")

    def test_runner_dispatches_both_methods_without_leaking_overrides(self):
        cases = ((runner.p1_run_generation, "proposed", "TaskRuntime", runner.ClassifiedTaskRuntime),
                 (runner.p1_run_roco, "roco", "RoCoRuntime", runner.ClassifiedRoCoRuntime))
        for module, method, field, expected in cases:
            with self.subTest(method=method):
                previous = getattr(module, field)
                def check_dispatch():
                    self.assertIs(getattr(module, field), expected)
                    self.assertEqual(sys.argv[1:], ["--config", "fixture"])
                    return 0
                with patch.object(module, "main", side_effect=check_dispatch):
                    with patch.object(sys, "argv", ["runner", "--method", method, "--config", "fixture"]):
                        self.assertEqual(runner.main(), 0)
                        self.assertEqual(sys.argv[1:], ["--method", method, "--config", "fixture"])
                self.assertIs(getattr(module, field), previous)


if __name__ == "__main__":
    unittest.main()
