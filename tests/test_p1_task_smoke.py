import unittest

from scripts import p1_task_smoke as smoke
from scripts.p1_task_contract import load_task_contract
from tests import test_p1_runtime as runtime_tests
from tests.test_p0_controller import FixtureBackend, MemoryLog


class TestTaskSmoke(unittest.TestCase):
    def test_all_eleven_configs_bind_real_task_and_one_evaluation(self):
        self.assertEqual(len(smoke.CONDITIONS), 11)
        self.assertEqual(len(set(smoke.CONDITIONS)), 11)
        self.assertEqual(smoke.plan()["max_model_cli_calls"], 11)
        self.assertFalse(smoke.plan()["uses_formal_test_data"])
        for name, black_box in smoke.CONDITIONS:
            with self.subTest(name=name, black_box=black_box):
                config = smoke.make_config(name, black_box)
                runtime = runtime_tests.TestTaskRuntime.create(config)
                self.assertEqual(runtime.contract.black_box, black_box)
                self.assertEqual(runtime.problem_name, name)
                self.assertEqual(config["budget"]["normal_model_calls"], 1)
                self.assertEqual(config["budget"]["normal_code_generation_slots"], 1)
                self.assertEqual(config["budget"]["additional_infrastructure_retry_calls"], 0)
                self.assertEqual(config["budget"]["additional_infrastructure_evaluation_starts"], 0)
                self.assertFalse(config["task"]["reference_seed_in_prompt_or_population"])

    def test_each_condition_generates_one_seedless_proposal(self):
        for name, black_box in smoke.CONDITIONS:
            contract = load_task_contract(name, black_box=black_box)

            class Backend(FixtureBackend):
                def __init__(self):
                    super().__init__()
                    self.contract = contract

                def model(self, stage, prompt):
                    self.calls.append((stage, prompt))
                    return ("<thought>fixture</thought><code>import numpy as np\n"
                            f"{contract.signature}\n    return np.ones(1)\n</code>")

                def evaluate(self, candidate):
                    self.evaluations.append(candidate.stage)
                    return {"status": "passed", "average": 1.0,
                            "fitness": -1.0 if contract.direction == "min" else 1.0}

            with self.subTest(name=name, black_box=black_box):
                backend, log = Backend(), MemoryLog()
                result = smoke.run_one(backend, log)
                self.assertTrue(result["valid"], result)
                self.assertEqual((len(backend.calls), len(backend.evaluations)), (1, 1))
                prompt = backend.calls[0][1]
                self.assertIn(contract.signature, prompt)
                self.assertIn("without an existing heuristic", prompt)
                self.assertNotIn("return np.", prompt)


if __name__ == "__main__":
    unittest.main()
