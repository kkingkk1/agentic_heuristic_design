import unittest
from unittest.mock import patch

from baselines.mcts_ahd import controller as mcts
from baselines.mcts_ahd.runtime import MCTSAHDRuntime
from scripts.p0_controller import StopRun
from scripts import p1_run_mcts_ahd
from scripts.p1_task_contract import TASKS, load_task_contract
from tests.test_p0_controller import FixtureBackend, MemoryLog
from tests import test_p1_runtime as runtime_tests


class MCTSBackend(FixtureBackend):
    def __init__(self, contract, *, invalid_stage=None):
        super().__init__()
        self.contract, self.invalid_stage = contract, invalid_stage

    def model(self, stage, prompt):
        with self.lock:
            self.calls.append((stage, prompt))
        if stage.startswith("post_thought_"):
            return f"refined design {stage}"
        if stage == self.invalid_stage:
            return "invalid proposal"
        return (f"{{idea for {stage}}}\nimport numpy as np\n{self.contract.signature}\n"
                f"    return np.array([1.0])  # {stage}\n")

    def evaluate(self, candidate):
        with self.lock:
            self.evaluations.append(candidate.stage)
        objective = float(1000 - candidate.sequence) if self.contract.direction == "min" else float(candidate.sequence)
        return {"status": "passed", "average": objective,
                "fitness": -objective if self.contract.direction == "min" else objective}


class TestMCTSAHD(unittest.TestCase):
    def test_e2_and_s1_can_list_ideas_before_the_braced_design(self):
        contract = load_task_contract("tsp_aco")
        response = ("Useful ideas: short edges and diversity.\n{Combine both ideas.}\n```python\n"
                    "import numpy as np\n" + contract.signature + "\n    return np.ones_like(distance_matrix)\n```")
        parsed = mcts.parse_proposal(contract, response)
        self.assertEqual(parsed["thought"], "Combine both ideas.")
        self.assertIn(contract.signature, parsed["code"])
        self.assertIn("First list useful common ideas", mcts.prompt_for(contract, "e2", ()))
        self.assertIn("First list useful common ideas", mcts.prompt_for(contract, "s1", ()))

    def test_all_task_conditions_preserve_tree_and_two_model_calls_per_candidate(self):
        conditions = [(name, black_box) for name in sorted(TASKS)
                      for black_box in ((False,) if name == "tsp_gls" else (False, True))]
        for name, black_box in conditions:
            with self.subTest(name=name, black_box=black_box):
                contract = load_task_contract(name, black_box=black_box)
                backend, log = MCTSBackend(contract), MemoryLog()
                controller = mcts.MCTSAHDController(backend, log, contract, max_evaluations=25)
                best = controller.run()
                self.assertTrue(best.valid)
                self.assertEqual((controller.initial_count, len(backend.calls), len(backend.evaluations)),
                                 (10 if black_box else 4, 50, 25))
                self.assertEqual(sum(stage.startswith("post_thought_") for stage, _ in backend.calls), 25)
                self.assertEqual(sum(event["event"] == "tree_node_added" for event in log.items), 25)
                self.assertTrue(any(event["event"] == "tree_descent" for event in log.items))
                self.assertTrue(any(event.get("depth", 0) >= 2 for event in log.items))
                self.assertEqual(log.items[-1]["status"], "completed")
                for candidate in controller.candidates:
                    self.assertTrue(candidate.thought.startswith("refined design"))

    def test_path_reasoning_and_objective_direction(self):
        contract = load_task_contract("mkp_aco")
        backend, log = MCTSBackend(contract), MemoryLog()
        controller = mcts.MCTSAHDController(backend, log, contract, max_evaluations=60)
        best = controller.run()
        self.assertEqual(best.objective, 60.0)
        path_prompts = [prompt for stage, prompt in backend.calls if stage.startswith("s1_")]
        self.assertTrue(path_prompts)
        self.assertIn("search path", path_prompts[0])
        self.assertIn("Objective value: -", path_prompts[0])
        self.assertEqual(log.items[-1]["attempts"], 60)

    def test_invalid_candidate_stops_without_refinement_retry_or_evaluation(self):
        contract = load_task_contract("tsp_aco")
        backend, log = MCTSBackend(contract, invalid_stage="i1_0001"), MemoryLog()
        with self.assertRaises(StopRun):
            mcts.MCTSAHDController(backend, log, contract).run()
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.evaluations, [])
        self.assertEqual(log.items[-1]["status"], "stopped")


class TestMCTSAHDRunGate(unittest.TestCase):
    @staticmethod
    def config(black_box=False):
        config = runtime_tests.TestTaskRuntime.config("tsp_aco", black_box)
        config["status"] = "frozen"
        config["method"] = {"id": "mcts_ahd_ee9c4f4_adapted"}
        config["task"].update({"selection_seed": 20261103, "initial_proposals": 10 if black_box else 4,
                               "population_size": 10, "max_evaluations": 20,
                               "reference_seed_in_prompt_or_population": False})
        budget = config["budget"]
        budget["normal_model_calls"], budget["normal_code_generation_slots"] = 40, 20
        budget["max_initialization_and_generation_model_cli_invocations"] = 40 + budget["additional_infrastructure_retry_calls"]
        budget["max_initialization_and_generation_evaluation_starts"] = 20 + budget["additional_infrastructure_evaluation_starts"]
        return config

    def test_formal_budget_and_condition_specific_initial_size(self):
        self.assertEqual(p1_run_mcts_ahd.validate_config(self.config()), (4, 20, 20261103))
        self.assertEqual(p1_run_mcts_ahd.validate_config(self.config(True)), (10, 20, 20261103))
        for section, field, value in ((None, "status", "engineering"), ("task", "initial_proposals", 4),
                                      ("task", "reference_seed_in_prompt_or_population", True),
                                      ("budget", "normal_code_generation_slots", 19),
                                      ("budget", "normal_model_calls", 39)):
            config = self.config(True)
            (config if section is None else config[section])[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                p1_run_mcts_ahd.validate_config(config)

    def test_search_step_checks_two_calls_and_one_evaluation(self):
        runtime = runtime_tests.TestTaskRuntime.create(self.config(), runtime_class=MCTSAHDRuntime)
        quota = {"ordinary_usage_allowed": True, "remaining_model_calls": 1000}
        with patch("baselines.mcts_ahd.runtime.p0_budget_quota.read_limits", return_value=quota):
            runtime.before_search_step()
        runtime.cfg["budget"]["normal_model_calls"] = 1
        with self.assertRaisesRegex(StopRun, "思想重述"):
            runtime.before_search_step()


if __name__ == "__main__":
    unittest.main()
