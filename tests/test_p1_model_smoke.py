import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import p1_model_smoke as smoke
from scripts.p0_controller import StopRun
from scripts.p1_task_contract import load_task_contract
from tests.test_p0_controller import MemoryLog


class FakeRuntime:
    def __init__(self, method):
        self.method, self.contract, self.log = method, load_task_contract("tsp_aco"), MemoryLog()
        self.budget = SimpleNamespace(model_normal=0, eval_starts=0)

    def prepare_proposal(self, candidate):
        pass

    def release_proposal(self, candidate):
        pass

    def evaluate(self, candidate):
        self.budget.eval_starts += 1
        return {"status": "passed", "average": 9.0, "fitness": -9.0}


def fake_model(runtime, stage, prompt):
    runtime.budget.model_normal += 1
    signature = runtime.contract.signature
    code = f"import numpy as np\n{signature}\n    return np.ones_like(distance_matrix)"
    if stage.startswith("post_thought_"):
        return "Prefer short edges."
    if stage == "harmony_parameters":
        parameterized = signature.replace(") ->", ", alpha: float = 0.5) ->")
        return (f"```python\nimport numpy as np\n{parameterized}\n"
                "    return np.ones_like(distance_matrix) * alpha\n```\n"
                "```python\nparameter_ranges = {'alpha': (0.1, 1.0)}\n```")
    if runtime.method in ("eoh", "mcts"):
        return "{Prefer short edges.}\n" + code
    if runtime.method in ("reevo", "hsevo"):
        return "```python\n" + code + "\n```"
    return "<thought>Prefer short edges.</thought><code>" + code + "</code>"


class TestP1Smoke(unittest.TestCase):
    def test_plan_and_simulated_run_bound_all_calls_and_evaluations(self):
        self.assertEqual(smoke.plan()["max_model_cli_calls"], 7)
        self.assertEqual(smoke.plan()["max_evaluation_starts"], 6)
        self.assertEqual(len(smoke.plan()["operations"]), 6)
        self.assertTrue(all(item["max_evaluations"] == 1 for item in smoke.plan()["operations"]))
        self.assertFalse(smoke.plan()["uses_formal_test_data"])
        counters = {"calls": 0, "lock": threading.Lock(), "stop": threading.Event()}
        quota = {"ordinary_usage_allowed": True, "remaining_model_calls": None}
        results = []
        with patch.object(smoke.Runtime, "model", fake_model), \
             patch.object(smoke.p0_budget_quota, "read_limits", return_value=quota):
            for name, _method_id, _runtime, calls, evaluations in smoke.METHODS:
                runtime = FakeRuntime(name)
                result = smoke.run_one(name, runtime, runtime.log, counters)
                self.assertTrue(result["valid"], result)
                self.assertEqual(runtime.budget.model_normal, calls)
                self.assertEqual(runtime.budget.eval_starts, evaluations)
                results.append(result)
        self.assertEqual(counters["calls"], 7)
        self.assertEqual(sum(item["evaluation_starts"] for item in results), 6)

    def test_configured_call_cap_stops_before_an_extra_request(self):
        counters = {"calls": 0, "lock": threading.Lock(), "stop": threading.Event()}
        runtime = FakeRuntime("roco")
        backend = smoke.SmokeBackend(runtime, counters, max_calls=2)
        quota = {"ordinary_usage_allowed": True, "remaining_model_calls": None}
        with patch.object(smoke.Runtime, "model", fake_model), \
             patch.object(smoke.p0_budget_quota, "read_limits", return_value=quota):
            backend.model("first", "prompt")
            backend.model("second", "prompt")
            with self.assertRaisesRegex(StopRun, "2 次调用上限"):
                backend.model("third", "prompt")
        self.assertEqual((counters["calls"], runtime.budget.model_normal), (2, 2))

    def test_single_method_plan_and_call_cap(self):
        plan = smoke.plan("proposed")
        self.assertEqual((plan["max_model_cli_calls"], plan["max_evaluation_starts"]), (1, 1))
        self.assertEqual([operation["method"] for operation in plan["operations"]], ["proposed"])
        with self.assertRaises(ValueError):
            smoke.plan("unknown")
        counters = {"calls": 0, "lock": threading.Lock(), "stop": threading.Event()}
        runtime = FakeRuntime("proposed")
        quota = {"ordinary_usage_allowed": True, "remaining_model_calls": None}
        with patch.object(smoke.Runtime, "model", fake_model), \
                patch.object(smoke.p0_budget_quota, "read_limits", return_value=quota):
            result = smoke.run_one("proposed", runtime, runtime.log, counters, max_calls=1)
            self.assertTrue(result["valid"])
            with self.assertRaisesRegex(StopRun, "1 次调用上限"):
                smoke.run_one("proposed", runtime, runtime.log, counters, max_calls=1)
        self.assertEqual((counters["calls"], runtime.budget.model_normal, runtime.budget.eval_starts), (1, 1, 1))

    def test_invalid_proposal_is_not_repaired_or_evaluated(self):
        counters = {"calls": 0, "lock": threading.Lock(), "stop": threading.Event()}
        runtime = FakeRuntime("proposed")
        quota = {"ordinary_usage_allowed": True, "remaining_model_calls": None}

        def invalid_model(runtime, stage, prompt):
            runtime.budget.model_normal += 1
            return "<thought>fixture</thought>\n```python\npass\n```"

        with patch.object(smoke.Runtime, "model", invalid_model), \
                patch.object(smoke.p0_budget_quota, "read_limits", return_value=quota):
            result = smoke.run_one("proposed", runtime, runtime.log, counters)
        self.assertFalse(result["valid"])
        self.assertIn("invalid_proposal", result["error"])
        self.assertEqual((counters["calls"], runtime.budget.model_normal, runtime.budget.eval_starts), (1, 1, 0))

    def test_unquantified_quota_keeps_the_explicit_run_choice(self):
        counters = {"calls": 0, "lock": threading.Lock(), "stop": threading.Event()}
        runtime = FakeRuntime("roco")
        backend = smoke.SmokeBackend(runtime, counters, max_calls=2, allow_unquantified_quota=False)
        quotas = [{"ordinary_usage_allowed": True, "remaining_model_calls": 2},
                  {"ordinary_usage_allowed": True, "remaining_model_calls": None}]
        with patch.object(smoke.Runtime, "model", fake_model), \
             patch.object(smoke.p0_budget_quota, "read_limits", side_effect=quotas):
            backend.model("first", "prompt")
            with self.assertRaisesRegex(StopRun, "剩余调用次数无法证明"):
                backend.model("second", "prompt")
        self.assertEqual((counters["calls"], runtime.budget.model_normal), (1, 1))


if __name__ == "__main__":
    unittest.main()
