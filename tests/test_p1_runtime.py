import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import p1_run_generation, p1_runtime
from scripts.p0_controller import StopRun
from tests.test_p0_controller import CONFIG, MemoryLog


class TestTaskRuntime(unittest.TestCase):
    @staticmethod
    def config(name, black_box):
        config = copy.deepcopy(CONFIG)
        contract = p1_runtime.load_task_contract(name, black_box=black_box)
        evaluator, dataset, dataset_hash = p1_runtime.TRAINING[name]
        count, ants, iterations, generator_seed = p1_runtime.TRAINING_PARAMETERS[name]
        config["status"] = "engineering"
        config["task"].update({
            "name": name,
            "prompt_condition": "black_box" if black_box else "white_box",
            "problem_size": contract.train_size,
            "objective_direction": contract.direction,
            "function": contract.signature.removeprefix("def ").removesuffix(":"),
            "instance_count": count,
            "ants": ants,
            "iterations_per_instance": iterations,
            "dataset_generator_seed": generator_seed,
            "evaluation_seeds": [20261103 + i for i in range(count)],
            "dataset": dataset.relative_to(p1_runtime.ROOT).as_posix(),
            "dataset_sha256": dataset_hash,
            "whole_training_wall_timeout_seconds": evaluator.TIMEOUT_SECONDS,
        })
        if name == "tsp_gls":
            config["task"]["perturbation_moves"] = 30
        return config

    @staticmethod
    def create(config, runtime_class=p1_runtime.TaskRuntime):
        run_dir = p1_runtime.ROOT / "outputs" / "p1_runtime_fixture"
        config_path = (run_dir / "config.json").resolve()
        original_read = Path.read_text
        original_digest = p1_runtime.digest

        def read(path, *args, **kwargs):
            return json.dumps(config) if path == config_path else original_read(path, *args, **kwargs)

        def digest(path):
            return "config-fixture" if path == config_path else original_digest(path)

        with patch.object(Path, "read_text", read), patch.object(p1_runtime, "digest", digest):
            return runtime_class(run_dir, config_path, MemoryLog())

    def test_all_conditions_bind_the_real_task_entry_and_reader(self):
        for name in p1_runtime.TRAINING:
            for black_box in ((False,) if name == "tsp_gls" else (False, True)):
                with self.subTest(name=name, black_box=black_box):
                    runtime = self.create(self.config(name, black_box))
                    evaluator, dataset, _digest = p1_runtime.TRAINING[name]
                    self.assertEqual(runtime.problem_name, name)
                    self.assertEqual(runtime.eval_script.name, "eval_black_box.py" if black_box else "eval.py")
                    self.assertIs(runtime.read_complete, evaluator.read_complete)
                    self.assertIn(dataset, runtime.protected)
                    self.assertEqual(runtime.contract.black_box, black_box)
                    with self.assertRaisesRegex(StopRun, "工程配置禁止请求模型"):
                        runtime.model("test", "must not be sent")

    def test_task_or_budget_mismatch_is_rejected(self):
        variants = []
        for field, value in (("problem_size", 99), ("dataset_sha256", "0" * 64),
                             ("objective_direction", "max"), ("whole_training_wall_timeout_seconds", 1),
                             ("dataset", "problems/other/data.npy"), ("instance_count", 10),
                             ("ants", 20), ("iterations_per_instance", 100),
                             ("dataset_generator_seed", 5), ("evaluation_seeds", [1, 2, 3]),
                             ("function", "wrong")):
            config = self.config("tsp_aco", False)
            config["task"][field] = value
            variants.append((field, config))
        for field, config in variants:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.create(config)

    def test_speed_mode_binds_tsp_reader_and_requires_stagnation_window(self):
        config = self.config("tsp_aco", False)
        config["task"].update(evaluation_mode="quality_per_time", stagnation_rounds=2)
        runtime = self.create(config)
        self.assertIs(runtime.read_complete.func, p1_runtime.TRAINING["tsp_aco"][0].read_complete)
        self.assertEqual(runtime.read_complete.keywords, {"stagnation_rounds": 2})
        with self.assertRaisesRegex(StopRun, "工程配置禁止请求模型"):
            runtime.model("test", "must not be sent")
        for invalid in (None, 0, True):
            config["task"]["stagnation_rounds"] = invalid
            with self.subTest(stagnation_rounds=invalid), self.assertRaises(ValueError):
                self.create(config)
        unsupported = self.config("op_aco", False)
        unsupported["task"].update(evaluation_mode="quality_per_time", stagnation_rounds=2)
        with self.assertRaisesRegex(ValueError, "只支持 TSP-ACO"):
            self.create(unsupported)


class TestFormalRunGate(unittest.TestCase):
    @staticmethod
    def config():
        config = TestTaskRuntime.config("mkp_aco", False)
        config["status"] = "frozen"
        config["method"] = {"id": "proposed_solution_quality_and_speed_agents"}
        config["task"].update({"generations": 6, "selection_seed": 20261103,
                               "initial_proposals": 30, "population_size": 10,
                               "collaboration_rounds": 3,
                               "reference_seed_in_prompt_or_population": False})
        budget = config["budget"]
        budget["normal_model_calls"] = 420
        budget["normal_code_generation_slots"] = 378
        budget["max_initialization_and_generation_model_cli_invocations"] = 420 + budget["additional_infrastructure_retry_calls"]
        budget["max_initialization_and_generation_evaluation_starts"] = 378 + budget["additional_infrastructure_evaluation_starts"]
        return config

    def test_complete_frozen_config_has_expected_counts(self):
        self.assertEqual(p1_run_generation.validate_config(self.config()), (6, 20261103))
        config = self.config()
        config["method"]["id"] = "roco_paper_adapted"
        with self.assertRaisesRegex(ValueError, "方法标识"):
            p1_run_generation.validate_config(config)

    def test_unfrozen_seeded_or_inconsistent_budget_is_rejected(self):
        changes = (
            ("status", "engineering"),
            ("generations", 0),
            ("reference_seed_in_prompt_or_population", True),
            ("normal_model_calls", 419),
            ("normal_code_generation_slots", 379),
            ("max_initialization_and_generation_model_cli_invocations", 1),
            ("future_initialization_and_generation_eval_process_tree_cpu_seconds", 0),
        )
        for key, value in changes:
            config = self.config()
            if key == "status":
                config[key] = value
            elif key in config["task"]:
                config["task"][key] = value
            else:
                config["budget"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                p1_run_generation.validate_config(config)


if __name__ == "__main__":
    unittest.main()
