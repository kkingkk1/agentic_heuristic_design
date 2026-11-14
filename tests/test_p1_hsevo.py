import ast
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from baselines.hsevo import controller as hsevo
from baselines.hsevo.runtime import HSEvoRuntime
from scripts.p0_controller import StopRun
from scripts import p0_model_probe as probe, p1_run_hsevo
from scripts.p1_task_contract import TASKS, load_task_contract
from tests.test_p0_controller import MemoryLog
from tests.test_p1_reevo import ReEvoBackend
from tests import test_p1_runtime as runtime_tests


class HSEvoBackend(ReEvoBackend):
    def __init__(self, contract, *, invalid_harmony=False):
        super().__init__(contract)
        self.invalid_harmony = invalid_harmony

    def model(self, stage, prompt):
        if stage == "flash_reflection":
            with self.lock:
                self.calls.append((stage, prompt))
            return "**Analysis:** compare ranked heuristics\n**Experience:** favor better edges"
        if stage == "comprehensive_reflection":
            with self.lock:
                self.calls.append((stage, prompt))
            return "combine the useful ideas"
        if stage == "harmony_parameters":
            with self.lock:
                self.calls.append((stage, prompt))
            if self.invalid_harmony:
                return "unparseable"
            signature = self.contract.signature.replace(") ->", ", alpha: float = 0.5) ->")
            return ("```python\nimport numpy as np\n" + signature
                    + "\n    return np.array([alpha])\n```\n"
                    + "```python\nparameter_ranges = {'alpha': (0.0, 1.0)}\n```")
        return super().model(stage, prompt)


class TestHSEvo(unittest.TestCase):
    def test_official_common_prompt_git_blobs(self):
        blobs = {
            "system_generator": "e78d10ed2cc85f07e27ed4cebe18b2cf64ad6096",
            "user_generator": "c539e5cd6dde14f0dacaea4387784a9737dd56db",
            "seed": "ce88921830d6004e718c337605168fb0612151ad",
            "user_flash_reflection": "d24f9a1bf2228093dda7b1e3939896fed3b4e127",
            "user_comprehensive_reflection": "32940f41567d4e5371e59f5d6697daf172fa39d6",
            "system_reflector": "84a2fc9cd1469f61ce4bf0f9b14fe7c14c185de3",
            "crossover": "6dad421410da9185de560072693def4f74a86b41",
            "mutation": "824097b4d840325da17b8a26b48499c7320117f6",
            "system_harmony_search": "87ec03660e9634dc5e5565f97a7f8c339d4355dd",
            "harmony_search": "bcfae0e17cad352e117ab5524f3fe83a5523f772",
        }
        for name, expected in blobs.items():
            with self.subTest(name=name):
                content = (hsevo.COMMON / (name + ".txt")).read_bytes()
                self.assertEqual(hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest(), expected)

    def test_all_conditions_run_core_mechanisms_and_harmony_evaluations(self):
        conditions = [(name, black_box) for name in sorted(TASKS)
                      for black_box in ((False,) if name == "tsp_gls" else (False, True))]
        for name, black_box in conditions:
            with self.subTest(name=name, black_box=black_box):
                contract = load_task_contract(name, black_box=black_box)
                backend, log = HSEvoBackend(contract), MemoryLog()
                controller = hsevo.HSEvoController(backend, log, contract)
                elite = controller.run()
                self.assertTrue(elite.valid)
                self.assertEqual((len(backend.calls), len(backend.evaluations)), (48, 56))
                self.assertEqual(sum(stage == "flash_reflection" for stage, _ in backend.calls), 1)
                self.assertEqual(sum(stage == "comprehensive_reflection" for stage, _ in backend.calls), 1)
                self.assertEqual(sum(stage == "harmony_parameters" for stage, _ in backend.calls), 1)
                self.assertEqual(sum(stage.startswith("harmony_") for stage in backend.evaluations), 10)
                self.assertEqual(len([event for event in log.items if event["event"] == "harmony_candidate"]), 10)
                harmonies = [item for item in controller.candidates if item.stage.startswith("harmony_")]
                self.assertEqual(len({item.candidate_id for item in harmonies}), 10)
                for item in harmonies:
                    contract.parse_proposal(f"<thought>x</thought><code>{item.code}</code>")
                self.assertEqual(log.items[-1]["status"], "completed")

    def test_parameter_values_become_local_constants_without_changing_interface(self):
        contract = load_task_contract("tsp_aco")
        backend = HSEvoBackend(contract)
        template, ranges = hsevo.harmony_response(contract, backend.model("harmony_parameters", ""))
        code = hsevo.instantiate(contract, template, ranges, [0.75])
        tree = ast.parse(code)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        self.assertEqual([item.arg for item in function.args.args], ["distance_matrix"])
        self.assertEqual(function.args.defaults, [])
        namespace = {"np": np}
        exec(compile(tree, "<test>", "exec"), namespace)
        self.assertEqual(namespace["heuristics_v2"](np.ones((2, 2))).tolist(), [0.75])

    def test_two_generations_count_harmony_and_keep_reflection_history(self):
        contract = load_task_contract("tsp_aco")
        backend, log = HSEvoBackend(contract), MemoryLog()
        controller = hsevo.HSEvoController(backend, log, contract, generations=2)
        self.assertTrue(controller.run().valid)
        self.assertEqual((len(backend.calls), len(backend.evaluations)), (66, 81))
        self.assertEqual(len(controller.good) + len(controller.bad), 2)
        self.assertEqual([event["generation"] for event in log.items if event["event"] == "selection"], [1, 2])

    def test_parameter_dict_is_never_executed_and_invalid_output_stops_without_retry(self):
        contract = load_task_contract("tsp_aco")
        bad = ("```python\n" + contract.signature + "\n    return np.ones(1)\n```\n"
               "```python\nparameter_ranges = __import__('os').system('false')\n```")
        with self.assertRaises(probe.ProbeError):
            hsevo.harmony_response(contract, bad)
        backend, log = HSEvoBackend(contract, invalid_harmony=True), MemoryLog()
        with self.assertRaises(StopRun):
            hsevo.HSEvoController(backend, log, contract).run()
        self.assertEqual(sum(stage == "harmony_parameters" for stage, _ in backend.calls), 1)
        self.assertFalse(any(stage.startswith("harmony_") for stage in backend.evaluations))
        self.assertEqual(log.items[-1]["status"], "stopped")


class TestHSEvoRunGate(unittest.TestCase):
    @staticmethod
    def config():
        config = runtime_tests.TestTaskRuntime.config("mkp_aco", False)
        config["status"] = "frozen"
        config["method"] = {"id": "hsevo_fb569d6_adapted"}
        config["task"].update({"generations": 2, "selection_seed": 20261103, "initial_proposals": 30,
            "population_size": 10, "mutation_count": 5, "reference_seed_in_prompt_or_population": True,
            "reference_seed_source": "reevo_2f70042_task_adapted", "hm_size": 5, "hs_iterations": 5,
            "hmcr": 0.7, "par": 0.5, "bandwidth": 0.2})
        budget = config["budget"]
        budget["normal_model_calls"], budget["normal_code_generation_slots"] = 66, 81
        budget["max_initialization_and_generation_model_cli_invocations"] = 66 + budget["additional_infrastructure_retry_calls"]
        budget["max_initialization_and_generation_evaluation_starts"] = 81 + budget["additional_infrastructure_evaluation_starts"]
        return config

    def test_formal_budget_counts_all_harmony_evaluations(self):
        self.assertEqual(p1_run_hsevo.validate_config(self.config()), (2, 20261103))
        for section, field, value in ((None, "status", "engineering"), ("task", "hm_size", 4),
                                      ("task", "reference_seed_source", "official_hsevo"),
                                      ("budget", "normal_code_generation_slots", 80),
                                      ("budget", "normal_model_calls", 65)):
            config = self.config()
            (config if section is None else config[section])[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                p1_run_hsevo.validate_config(config)

    def test_generation_budget_checks_harmony_without_subscription_probe(self):
        runtime = runtime_tests.TestTaskRuntime.create(self.config(), runtime_class=HSEvoRuntime)
        runtime.budget.model_normal = 30
        runtime.budget.eval_normal = runtime.budget.eval_starts = 31
        runtime.budget.eval_cpu_samples = [1.0]
        with patch("scripts.p0_budget_quota.read_limits", side_effect=AssertionError("unexpected quota probe")):
            runtime.before_generation()
        runtime.cfg["budget"]["normal_code_generation_slots"] = 55
        with self.assertRaisesRegex(StopRun, "评价启动次数不足"):
            runtime.before_generation()


if __name__ == "__main__":
    unittest.main()
