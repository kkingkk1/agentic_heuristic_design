import unittest

from baselines.roco.paper_controller import RoCoController
from baselines.roco.runtime import RoCoRuntime
from scripts import p1_run_generation, p1_run_roco
from scripts.p1_task_contract import TASKS, load_task_contract
from tests.test_p0_controller import FixtureBackend, MemoryLog
from tests import test_p1_runtime as runtime_tests


class RoCoBackend(FixtureBackend):
    def __init__(self, contract):
        super().__init__()
        self.contract = contract

    def model(self, stage, prompt):
        response = super().model(stage, prompt)
        return response.replace("def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:",
                                self.contract.signature).replace("np.ones_like(distance_matrix)", "np.array([1.0])")

    def evaluate(self, candidate):
        result = super().evaluate(candidate)
        if self.contract.direction == "max":
            result["fitness"] = result["average"]
        return result


class TestRoCo(unittest.TestCase):
    def test_all_conditions_use_original_role_goals_and_record_mapping(self):
        conditions = [(name, black_box) for name in sorted(TASKS)
                      for black_box in ((False,) if name == "tsp_gls" else (False, True))]
        for name, black_box in conditions:
            with self.subTest(name=name, black_box=black_box):
                contract = load_task_contract(name, black_box=black_box)
                backend, log = RoCoBackend(contract), MemoryLog()
                selected = RoCoController(backend, log, task_contract=contract).run()
                self.assertEqual((len(backend.calls), len(backend.evaluations), len(selected)), (95, 88, 10))
                calls = dict(backend.calls)
                self.assertIn("global diversity and long-term potential", calls["start_Q"])
                self.assertIn("short-term gains and local optimizations", calls["start_E"])
                self.assertNotIn("reducing computational cost", calls["start_E"])
                self.assertIn("Explorer", calls["critic_1"])
                self.assertIn("Exploiter", calls["fusion_1"])
                self.assertIn("Best Explorer", calls["fusion_final"])
                self.assertIn("Role: Exploiter", calls["reflection_E"])
                self.assertIn("Explorer:", calls["reflection_I"])
                self.assertEqual(log.items[0]["internal"], {"Q": "Explorer", "E": "Exploiter", "I": "Integrator"})
                roles = {event.get("source_role") for event in log.items if event["event"] == "candidate_registered"}
                self.assertTrue({"Explorer", "Exploiter", "Integrator"} <= roles)
                self.assertNotIn("Q", roles)
                self.assertNotIn("E", roles)

    def test_two_generations_retain_role_specific_histories(self):
        contract = load_task_contract("tsp_aco")
        backend, log = RoCoBackend(contract), MemoryLog()
        controller = RoCoController(backend, log, task_contract=contract, generations=2)
        controller.run()
        self.assertEqual((len(backend.calls), len(backend.evaluations)), (160, 146))
        self.assertEqual([len(controller.history[key]) for key in ("Q", "E", "I")], [2, 2, 2])
        self.assertEqual([event["role"] for event in log.items if event["event"] == "history_appended"],
                         ["Explorer", "Exploiter", "Integrator"] * 2)


class TestRoCoRunGate(unittest.TestCase):
    def test_roco_and_proposed_method_configs_cannot_be_interchanged(self):
        config = runtime_tests.TestFormalRunGate.config()
        config["method"] = {"id": "roco_paper_adapted"}
        self.assertEqual(p1_run_roco.validate_config(config), (6, 20261103))
        with self.assertRaises(ValueError):
            p1_run_generation.validate_config(config)
        runtime = runtime_tests.TestTaskRuntime.create(config, runtime_class=RoCoRuntime)
        self.assertTrue(any("baselines\\roco\\paper_controller.py" in str(path) for path in runtime.protected))
        config["method"]["id"] = "proposed_qe"
        with self.assertRaises(ValueError):
            p1_run_roco.validate_config(config)


if __name__ == "__main__":
    unittest.main()
