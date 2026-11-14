import unittest
from unittest.mock import patch

from baselines.eoh import paper_controller as eoh
from baselines.eoh.runtime import EoHRuntime
from scripts.p0_controller import Candidate, StopRun
from scripts import p1_run_eoh
from scripts.p1_task_contract import TASKS, load_task_contract
from tests.test_p0_controller import FixtureBackend, MemoryLog
from tests import test_p1_runtime as runtime_tests


class EoHBackend(FixtureBackend):
    def __init__(self, contract, **kwargs):
        super().__init__(**kwargs)
        self.contract = contract

    def model(self, stage, prompt):
        response = super().model(stage, prompt)
        thought = response.split("<thought>", 1)[1].split("</thought>", 1)[0]
        code = response.split("<code>", 1)[1].split("</code>", 1)[0]
        code = code.replace("def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:", self.contract.signature)
        code = code.replace("np.ones_like(distance_matrix)", "np.array([1.0])")
        return "{" + thought + "}\n```python\n" + code + "```"

    def evaluate(self, candidate):
        with self.lock:
            self.evaluations.append(candidate.stage)
        objective = float(candidate.sequence)
        fitness = objective if self.contract.direction == "max" else -objective
        return {"status": "passed", "average": objective, "fitness": fitness}


class TestPaperEoH(unittest.TestCase):
    def test_all_task_conditions_accept_original_style_mock_output(self):
        for name in sorted(TASKS):
            for black_box in ((False,) if name == "tsp_gls" else (False, True)):
                with self.subTest(name=name, black_box=black_box):
                    contract = load_task_contract(name, black_box=black_box)
                    backend, log = EoHBackend(contract), MemoryLog()
                    selected = eoh.EoHController(backend, log, contract).run()
                    self.assertEqual((len(backend.calls), len(backend.evaluations), len(selected)), (60, 60, 10))
                    self.assertIn(contract.signature, dict(backend.calls)["init_01"])

    def test_five_operators_use_same_parent_population_and_preserve_elites(self):
        contract = load_task_contract("tsp_aco")
        backend, log = EoHBackend(contract), MemoryLog()
        controller = eoh.EoHController(backend, log, contract)
        selected = controller.run()
        self.assertEqual((len(backend.calls), len(backend.evaluations), len(selected)), (60, 60, 10))
        initial = next(event for event in log.items if event["event"] == "initial_population")
        initial_ids = set(initial["candidate_ids"])
        registered = [event for event in log.items if event["event"] == "candidate_registered" and event["generation"] == 1]
        self.assertEqual(len(registered), 50)
        for operator in eoh.OPERATORS:
            self.assertEqual(sum(event["operator"] == operator for event in registered), 10)
        for event in registered:
            self.assertTrue(set(event["parents"]) <= initial_ids)
            self.assertEqual(len(event["parents"]), 5 if event["operator"] in ("E1", "E2") else 1)
        selection = next(event for event in log.items if event["event"] == "selection")
        self.assertEqual(len(selection["pool"]), 60)
        self.assertEqual({item.candidate_id for item in selected}, initial_ids)
        self.assertEqual(backend.generation_checks, 1)
        calls = dict(backend.calls)
        self.assertNotIn("return np.", calls["init_01"])
        self.assertIn("redundant parts", calls["eoh_M3_01"])

    def test_two_generations_and_maximization(self):
        contract = load_task_contract("mkp_aco", black_box=True)
        backend, log = EoHBackend(contract), MemoryLog()
        controller = eoh.EoHController(backend, log, contract, generations=2)
        selected = controller.run()
        self.assertEqual((len(backend.calls), len(backend.evaluations)), (110, 110))
        self.assertEqual([item["generation"] for item in log.items if item["event"] == "selection"], [1, 2])
        self.assertEqual([item.objective for item in selected], sorted((item.objective for item in selected), reverse=True))

    def test_invalid_initial_output_stops_without_extra_proposal(self):
        contract = load_task_contract("tsp_aco")
        backend, log = EoHBackend(contract, invalid={"init_01"}), MemoryLog()
        controller = eoh.EoHController(backend, log, contract)
        with self.assertRaises(StopRun):
            controller.run()
        self.assertEqual(len(backend.calls), 10)
        self.assertEqual(len(backend.evaluations), 9)
        self.assertEqual(log.items[-1]["status"], "stopped")

    def test_source_style_rounded_objective_deduplicates(self):
        contract = load_task_contract("tsp_aco")
        controller = eoh.EoHController(EoHBackend(contract), MemoryLog(), contract, population_size=2)
        candidates = [Candidate("a", 1, "init_01", (), fitness=-1.000001),
                      Candidate("b", 2, "init_02", (), fitness=-1.000002),
                      Candidate("c", 3, "init_03", (), fitness=-2.0)]
        self.assertEqual([item.candidate_id for item in controller.select_eoh(candidates)], ["a", "c"])


class TestEoHRunGate(unittest.TestCase):
    @staticmethod
    def config():
        config = runtime_tests.TestTaskRuntime.config("tsp_aco", False)
        config["status"] = "frozen"
        config["method"] = {"id": "eoh_paper_adapted"}
        config["task"].update({"generations": 2, "selection_seed": 20261103,
                               "initial_proposals": 10, "population_size": 10,
                               "reference_seed_in_prompt_or_population": False})
        budget = config["budget"]
        budget["normal_model_calls"] = budget["normal_code_generation_slots"] = 110
        budget["max_initialization_and_generation_model_cli_invocations"] = 110 + budget["additional_infrastructure_retry_calls"]
        budget["max_initialization_and_generation_evaluation_starts"] = 110 + budget["additional_infrastructure_evaluation_starts"]
        return config

    def test_formal_budget_and_adaptation_label_are_required(self):
        self.assertEqual(p1_run_eoh.validate_config(self.config()), (2, 20261103))
        for field, value in (("status", "engineering"), ("id", "eoh"), ("normal_model_calls", 111)):
            config = self.config()
            target = config if field == "status" else config["method"] if field == "id" else config["budget"]
            target[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                p1_run_eoh.validate_config(config)

    def test_generation_budget_checks_fifty_slots_without_subscription_probe(self):
        config = self.config()
        runtime = runtime_tests.TestTaskRuntime.create(config, runtime_class=EoHRuntime)
        runtime.budget.model_normal = 10
        runtime.budget.eval_normal = runtime.budget.eval_starts = 10
        runtime.budget.eval_cpu_samples = [1.0]
        with patch("scripts.p0_budget_quota.read_limits", side_effect=AssertionError("unexpected quota probe")):
            runtime.before_generation()
        runtime.cfg["budget"]["normal_model_calls"] = 59
        with self.assertRaisesRegex(StopRun, "不足以覆盖一代"):
            runtime.before_generation()


if __name__ == "__main__":
    unittest.main()
