import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts import p1_formal_matrix as matrix
from scripts import p1_deepaco_test_evaluate
from scripts import p1_reference_test_evaluate, p1_test_evaluate
from scripts.p0_runtime import digest
from scripts.p1_freeze_candidate import METHODS as RUN_VALIDATORS


class MemoryLog:
    def __init__(self):
        self.events = []

    def event(self, name, **fields):
        self.events.append({"event": name, **fields})


class TestP1MatrixStatus(unittest.TestCase):
    def test_pending_matrix_cannot_be_used_as_formal_contract(self):
        config, _ = matrix.load_matrix(allow_pending=True)
        if config["status"] != "pending":
            self.skipTest("当前合同已冻结；待定状态不适用。")
        with self.assertRaisesRegex(ValueError, "未冻结"):
            matrix.load_matrix()


class TestP1FormalMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config, _ = matrix.load_matrix(allow_pending=True)
        if config["status"] == "pending":
            raise unittest.SkipTest("矩阵正在缩减预算；冻结后再验证正式合同。")
        cls.matrix, cls.data = matrix.load_matrix()

    def test_matrix_covers_conditions_and_pairs_replicate_seeds(self):
        cells = matrix.run_cells(self.matrix)
        self.assertEqual(len(cells), 4)
        self.assertEqual(len(set(cells)), 4)
        self.assertEqual(matrix.conditions(), [("tsp_aco", "white_box")])
        for task, condition in matrix.conditions():
            seeds = {}
            for method in matrix.METHODS:
                values = [matrix.training_config(self.matrix, task, condition, method, index)
                          ["task"]["selection_seed"] for index in (1, 2)]
                seeds[method] = values
            self.assertTrue(all(value == self.matrix["selection_seeds"] for value in seeds.values()))
        self.assertEqual(matrix.test_cells(self.data), [("tsp_aco", "white_box", 50)])
        self.assertEqual(matrix.reference_cells(self.data), [("expert", "tsp_aco", 50),
                                                             ("deepaco", "tsp_aco", 50)])
        self.assertEqual(matrix.summary(self.matrix, self.data), {
            "conditions": 1, "methods": 2, "research_runs": 4, "test_condition_scales": 1,
            "candidate_test_runs": 4, "candidate_test_instances": 256,
            "normal_model_calls": 156, "max_model_calls_including_retries": 156,
            "normal_evaluation_slots": 128, "max_evaluation_starts_including_retries": 128,
            "max_training_process_tree_cpu_seconds": 72000,
            "candidate_test_max_wall_seconds_sum": 3600,
            "candidate_test_max_cpu_seconds_sum": 100800,
            "reference_test_runs": 2, "reference_test_instances": 128,
            "reference_test_max_wall_seconds_sum": 1800,
            "reference_test_max_cpu_seconds_sum": 50400,
            "primary_comparisons": 1, "model_token_total_limit": None,
        })

    def test_all_training_and_test_configs_match_existing_gates(self):
        for task, condition, method, repeat in matrix.run_cells(self.matrix):
            with self.subTest(task=task, condition=condition, method=method, repeat=repeat):
                config = matrix.training_config(self.matrix, task, condition, method, repeat)
                RUN_VALIDATORS[method][1](config)
                self.assertEqual((config["task"]["initial_proposals"], config["task"]["population_size"],
                                  config["task"]["collaboration_rounds"]), (6, 2, 3))
                self.assertEqual(config["budget"]["max_initialization_and_generation_evaluation_starts"], 32)
                self.assertEqual(config["budget"]["max_initialization_and_generation_model_cli_invocations"], 39)
                self.assertEqual(config["budget"]["additional_infrastructure_retry_calls"], 0)
                self.assertEqual(config["budget"]["additional_infrastructure_evaluation_starts"], 0)
        for task, condition, size in matrix.test_cells(self.data):
            config = matrix.test_config(self.matrix, task, condition, size)
            candidate = {"status": "frozen_after_training", "task": task, "prompt_condition": condition}
            entry = p1_test_evaluate.load_test_contract(config, candidate, self.data)
            self.assertEqual((entry["size"], entry["instances"]), (size, 64))

    def test_tampering_with_seed_budget_source_or_replicate_is_rejected(self):
        config = matrix.training_config(self.matrix, "tsp_aco", "white_box", "proposed_solution_quality_and_speed_agents", 1)
        matrix.verify_training_config(config)
        for section, field, value in (("task", "selection_seed", 1),
                                      ("task", "population_size", 10),
                                      ("budget", "normal_model_calls", 96),
                                      ("model", "name", "other"),
                                      (None, "formal_replicate", 2),
                                      (None, "matrix_contract_sha256", "0" * 64)):
            edited = copy.deepcopy(config)
            (edited if section is None else edited[section])[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                matrix.verify_training_config(edited)
        with patch.object(matrix, "implementation_digest", return_value=("0" * 64, 1)):
            with self.assertRaisesRegex(ValueError, "实现或提示"):
                matrix.load_matrix()
        with self.assertRaises(ValueError):
            matrix.training_config(self.matrix, "tsp_gls", "black_box", "proposed_solution_quality_and_speed_agents", 1)
        with self.assertRaises(ValueError):
            matrix.training_config(self.matrix, "tsp_aco", "white_box", "eoh", 1)
        with self.assertRaises(ValueError):
            matrix.training_config(self.matrix, "tsp_aco", "white_box", "proposed_solution_quality_and_speed_agents", 3)
        with self.assertRaises(ValueError):
            matrix.test_config(self.matrix, "tsp_aco", "white_box", 100)
        test = matrix.test_config(self.matrix, "tsp_aco", "white_box", 50)
        matrix.verify_test_config(test)
        with self.assertRaises(ValueError):
            matrix.verify_test_config(dict(test, solver_repeats_per_instance=2))

    def test_all_reference_configs_match_method_gates(self):
        for kind, task, size in matrix.reference_cells(self.data):
            with self.subTest(kind=kind, task=task, size=size):
                config = matrix.reference_config(self.matrix, self.data, kind, task, size)
                if kind == "expert":
                    p1_reference_test_evaluate.load_reference_contract(
                        config, self.data, digest(matrix.DATA_CONTRACT), digest(matrix.RESOURCE_PATH))
                elif kind == "deepaco":
                    p1_deepaco_test_evaluate.load_reference_contract(config, self.data, digest(matrix.DATA_CONTRACT))
        config = matrix.reference_config(self.matrix, self.data, "deepaco", "tsp_aco", 50)
        self.assertEqual((config["checkpoint"], config["checkpoint_scale_mode"]), ("tsp100", "cross_scale"))
        with self.assertRaises(ValueError):
            matrix.reference_config(self.matrix, self.data, "gnngls", "tsp_gls", 200)
        expert = matrix.reference_config(self.matrix, self.data, "expert", "tsp_aco", 50)
        matrix.verify_reference_config(expert, "expert")
        with self.assertRaises(ValueError):
            matrix.verify_reference_config(dict(expert, whole_dataset_wall_timeout_seconds=1), "expert")

    def test_duplicate_run_detection_ignores_preflight_but_rejects_started_run(self):
        config = matrix.training_config(self.matrix, "tsp_aco", "white_box", "proposed_solution_quality_and_speed_agents", 1)
        previous = matrix.ROOT / "outputs/p1_control/previous/config.json"
        current = matrix.ROOT / "outputs/p1_control/current"
        original_read = Path.read_text

        def read(path, *args, **kwargs):
            if path == previous:
                return json.dumps(config)
            return original_read(path, *args, **kwargs)

        for event, expected in (("finish", None), ("formal_run_started", previous.parent)):
            with self.subTest(event=event), patch.object(Path, "glob", return_value=iter([previous])), \
                    patch.object(Path, "read_text", read), \
                    patch.object(Path, "open", return_value=io.StringIO(json.dumps({"event": event}) + "\n")):
                self.assertEqual(matrix.previously_started(config, current), expected)

    @unittest.skipUnless(os.name == "nt", "P1 互斥锁使用 Windows 命名互斥体")
    def test_guard_records_formal_identity_before_work(self):
        config = matrix.training_config(self.matrix, "tsp_aco", "white_box", "proposed_solution_quality_and_speed_agents", 1)
        log = MemoryLog()
        with patch.object(matrix, "verify_training_config"), patch.object(matrix, "previously_started", return_value=None):
            with matrix.formal_run_guard(config, Path("unused"), log):
                self.assertEqual(log.events[0]["event"], "formal_run_started")
                self.assertEqual(log.events[0]["formal_replicate"], 1)
                self.assertEqual(log.events[0]["selection_seed"], self.matrix["selection_seeds"][0])

    @unittest.skipUnless(os.name == "nt", "P1 互斥锁使用 Windows 命名互斥体")
    def test_guard_rejects_concurrent_research_process(self):
        code = (
            "import ctypes,sys; k=ctypes.WinDLL('kernel32',use_last_error=True);"
            "k.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_wchar_p];"
            "k.CreateMutexW.restype=ctypes.c_void_p;"
            "k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];"
            "k.WaitForSingleObject.restype=ctypes.c_uint32;"
            "k.ReleaseMutex.argtypes=[ctypes.c_void_p]; k.CloseHandle.argtypes=[ctypes.c_void_p];"
            "h=k.CreateMutexW(None,False,r'Local\\agentic_heuristic_p1_formal');"
            "r=k.WaitForSingleObject(h,0);print(r,flush=True);"
            "sys.stdin.readline();"
            "k.ReleaseMutex(h) if r in (0,128) else None;k.CloseHandle(h)"
        )
        process = subprocess.Popen([sys.executable, "-B", "-c", code], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertIn(process.stdout.readline().strip(), ("0", "128"))
            with self.assertRaisesRegex(ValueError, "已有 P1"):
                with matrix.formal_run_guard({}, Path("unused"), MemoryLog()):
                    self.fail("并发研发不应进入运行。")
        finally:
            process.communicate(input="\n", timeout=10)


if __name__ == "__main__":
    unittest.main()
