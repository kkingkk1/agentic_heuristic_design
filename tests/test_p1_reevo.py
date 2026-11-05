import hashlib
import unittest
from unittest.mock import patch

from baselines.reevo import controller as reevo
from baselines.reevo.runtime import ReEvoRuntime
from scripts.p0_controller import Candidate, StopRun
from scripts import p1_run_reevo
from scripts.p1_task_contract import TASKS, load_task_contract
from tests.test_p0_controller import FixtureBackend, MemoryLog
from tests import test_p1_runtime as runtime_tests


class ReEvoBackend(FixtureBackend):
    def __init__(self, contract, invalid=()):
        super().__init__()
        self.contract = contract
        self.invalid = set(invalid)

    def model(self, stage, prompt):
        with self.lock:
            self.calls.append((stage, prompt))
        if stage.startswith("short_reflection_"):
            return f"short insight {stage}"
        if stage == "long_reflection":
            return "long insight"
        if stage in self.invalid:
            return "not executable"
        return ("```python\nimport numpy as np\n" + self.contract.signature
                + f"\n    return np.array([1.0])  # {stage}\n```")

    def evaluate(self, candidate):
        with self.lock:
            self.evaluations.append(candidate.stage)
        if self.contract.direction == "min":
            objective = 1000.0 if candidate.stage == "seed" else float(1000 - candidate.sequence)
            fitness = -objective
        else:
            objective = 0.0 if candidate.stage == "seed" else float(candidate.sequence)
            fitness = objective
        return {"status": "passed", "average": objective, "fitness": fitness}


class TestReEvo(unittest.TestCase):
    def test_copied_prompts_match_fixed_upstream_git_blobs(self):
        blobs = {
            'common/system_generator.txt': '6009306f50cea957f4c0824c3b12ccec580c1bb5',
            'common/user_generator.txt': '1b2afe0d29923a6440d5ee1c1c7c8a08809e7f16',
            'common/seed.txt': '7a1b227cff6486cd787caf4c89f5bd863f5e9739',
            'common/system_reflector.txt': '8c1ceead05f0d5a4ef4acf811349eee66fe62435',
            'common/user_reflector_st.txt': '883b578aaa5f31dd6e81bb986c28e3009c785a32',
            'common/user_reflector_st_black_box.txt': 'fafbc7fe00375d28a02e59535b1c74917cb97087',
            'common/user_reflector_lt.txt': '850277e4e6ab76bb029344f0577e611e8f45e307',
            'common/crossover.txt': 'bd4f2909a71dfa75f4c99a523401bfa5d7150188',
            'common/mutation.txt': 'b0bc6d0819d21399e5d4ab8b1dcc0f6fe62f3203',
            'tsp_aco/seed_func.txt': '98ac834fd7880c7730ad34e3ad825ecfc2824753',
            'tsp_aco_black_box/seed_func.txt': 'a52854fa1d9235348baeabad4c8abd7e831b26d4',
            'op_aco/seed_func.txt': 'ad1dcd8759268b3eba1cdb279c3f9fa958f5fb18',
            'op_aco_black_box/seed_func.txt': '96b4485a73cc66362136ca7f226e30f62f573b11',
            'cvrp_aco/seed_func.txt': 'bb9df8e69694017ca74538bd365a028fc5a82015',
            'cvrp_aco_black_box/seed_func.txt': '39fefc2e4da5bc1171131e7eca5edf0929e75a08',
            'mkp_aco/seed_func.txt': '26a5373c82d4deb7ce329ed3dd4b4e244eea2e1a',
            'mkp_aco_black_box/seed_func.txt': '12350a0ac4a83e40b1001281ec75afd758bfa436',
            'bpp_offline_aco/seed_func.txt': '5894db80cbc9e46fd1a261d646a96876021e6a4b',
            'bpp_offline_aco_black_box/seed_func.txt': 'a97f137d643a537f88b2ec55e85bbd280754f24b',
            'tsp_gls/seed_func.txt': '4b42109cabae3a488210c2569a3d3b6a26923847',
            'tsp_aco/external_knowledge.txt': '5ab82fcba6ebf6d75015103f14898a5073b45de6',
            'op_aco/external_knowledge.txt': '5ab82fcba6ebf6d75015103f14898a5073b45de6',
            'cvrp_aco/external_knowledge.txt': '5ab82fcba6ebf6d75015103f14898a5073b45de6',
            'mkp_aco/external_knowledge.txt': '1d954a5231b58af493019696936a3babdd9004bc',
            'bpp_offline_aco/external_knowledge.txt': '5ab82fcba6ebf6d75015103f14898a5073b45de6',
        }
        for name, expected in blobs.items():
            with self.subTest(path=name):
                content = (reevo.UPSTREAM / name).read_bytes()
                actual = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
                self.assertEqual(actual, expected)

    def test_source_seed_matches_every_task_contract(self):
        for name in sorted(TASKS):
            for black_box in ((False,) if name == "tsp_gls" else (False, True)):
                with self.subTest(name=name, black_box=black_box):
                    contract = load_task_contract(name, black_box=black_box)
                    files = reevo.source_files(contract)
                    code = reevo.seed_code(contract, reevo.text(files["seed_func"]))
                    self.assertIn(contract.signature, code)
                    if name == "op_aco" and black_box:
                        self.assertNotIn("edge_constraint", code)

    def test_source_mechanisms_and_cost_counts_for_white_and_black(self):
        conditions = [(name, black_box) for name in sorted(TASKS)
                      for black_box in ((False,) if name == "tsp_gls" else (False, True))]
        for name, black_box in conditions:
            with self.subTest(name=name, black_box=black_box):
                contract = load_task_contract(name, black_box=black_box)
                backend, log = ReEvoBackend(contract), MemoryLog()
                controller = reevo.ReEvoController(backend, log, contract)
                elite = controller.run()
                self.assertTrue(elite.valid)
                self.assertEqual((len(backend.calls), len(backend.evaluations)), (56, 46))
                self.assertEqual(backend.evaluations.count("seed"), 1)
                self.assertEqual(sum(stage.startswith("short_reflection_") for stage, _ in backend.calls), 10)
                self.assertEqual(sum(stage.startswith("crossover_") for stage, _ in backend.calls), 10)
                self.assertEqual(sum(stage.startswith("mutation_") for stage, _ in backend.calls), 5)
                self.assertEqual(sum(stage == "long_reflection" for stage, _ in backend.calls), 1)
                calls = dict(backend.calls)
                self.assertIn("heuristics_v1", calls["init_01"])
                self.assertIn("[Worse code]", calls["short_reflection_01"])
                self.assertIn("[Better code]", calls["crossover_01"])
                self.assertIn("[Reflection]", calls["crossover_01"])
                self.assertIn("long insight", calls["mutation_01"])
                if black_box:
                    self.assertNotIn("external_knowledge", controller.prompts)
                    cross = [event for event in log.items if event["event"] == "candidate_registered"
                             and event["operator"] == "crossover"]
                    self.assertTrue(all(controller.seed.candidate_id not in event["parents"] for event in cross))

    def test_two_generations_have_correct_counts(self):
        contract = load_task_contract("op_aco")
        backend, log = ReEvoBackend(contract), MemoryLog()
        elite = reevo.ReEvoController(backend, log, contract, generations=2).run()
        self.assertTrue(elite.valid)
        self.assertEqual((len(backend.calls), len(backend.evaluations)), (82, 61))
        self.assertEqual([event["generation"] for event in log.items if event["event"] == "selection"], [1, 2])

    def test_invalid_initial_candidates_are_not_retried(self):
        contract = load_task_contract("tsp_aco")
        invalid = {f"init_{i:02d}" for i in range(1, 31)}
        backend, log = ReEvoBackend(contract, invalid=invalid), MemoryLog()
        with self.assertRaises(StopRun):
            reevo.ReEvoController(backend, log, contract).run()
        self.assertEqual((len(backend.calls), backend.evaluations), (30, ["seed"]))
        self.assertEqual(log.items[-1]["status"], "stopped")

    def test_black_box_rejects_candidates_no_better_than_seed(self):
        contract = load_task_contract("mkp_aco", black_box=True)
        backend, log = ReEvoBackend(contract), MemoryLog()
        controller = reevo.ReEvoController(backend, log, contract)
        controller.seed = controller.elite = Candidate("seed", 1, "seed", (), objective=5.0, fitness=5.0)
        population = [Candidate("a", 2, "init_01", (), objective=4.0, fitness=4.0),
                      Candidate("b", 3, "init_02", (), objective=5.0, fitness=5.0)]
        with self.assertRaisesRegex(StopRun, "优于种子"):
            controller.parent_pairs(population)


class TestReEvoRunGate(unittest.TestCase):
    @staticmethod
    def config():
        config = runtime_tests.TestTaskRuntime.config("op_aco", True)
        config["status"] = "frozen"
        config["method"] = {"id": "reevo_2f70042_adapted"}
        config["task"].update({"generations": 2, "selection_seed": 20261103, "initial_proposals": 30,
                               "population_size": 10, "mutation_count": 5,
                               "reference_seed_in_prompt_or_population": True})
        budget = config["budget"]
        budget["normal_model_calls"], budget["normal_code_generation_slots"] = 82, 61
        budget["max_initialization_and_generation_model_cli_invocations"] = 82 + budget["additional_infrastructure_retry_calls"]
        budget["max_initialization_and_generation_evaluation_starts"] = 61 + budget["additional_infrastructure_evaluation_starts"]
        return config

    def test_formal_budget_counts_seed_and_reflections(self):
        self.assertEqual(p1_run_reevo.validate_config(self.config()), (2, 20261103))
        for section, field, value in ((None, "status", "engineering"), ("method", "id", "reevo"),
                                      ("task", "reference_seed_in_prompt_or_population", False),
                                      ("budget", "normal_code_generation_slots", 60),
                                      ("budget", "normal_model_calls", 81)):
            config = self.config()
            (config if section is None else config[section])[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                p1_run_reevo.validate_config(config)

    def test_generation_budget_checks_reflections_evaluations_and_quota(self):
        runtime = runtime_tests.TestTaskRuntime.create(self.config(), runtime_class=ReEvoRuntime)
        runtime.budget.model_normal = 30
        runtime.budget.eval_normal = runtime.budget.eval_starts = 31
        runtime.budget.eval_cpu_samples = [1.0]
        quota = {"ordinary_usage_allowed": True, "remaining_model_calls": 1000}
        with patch("baselines.reevo.runtime.p0_budget_quota.read_limits", return_value=quota):
            runtime.before_generation()
        runtime.cfg["budget"]["normal_model_calls"] = 55
        with self.assertRaisesRegex(StopRun, "不足以覆盖"):
            runtime.before_generation()


if __name__ == "__main__":
    unittest.main()
