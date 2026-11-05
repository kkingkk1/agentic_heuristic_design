import copy
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from scripts import p1_deepaco_test_evaluate as supervisor
from scripts import p1_deepaco_worker
from scripts.p0_runtime import digest
from scripts.p1_runtime import DATA_CONTRACT
from tests.test_p1_bpp_aco import task as bpp_task
from tests.test_p1_cvrp_aco import task as cvrp_task
from tests.test_p1_mkp_aco import task as mkp_task
from tests.test_p1_op_aco import task as op_task


tsp_task = p1_deepaco_worker.load_task("tsp")


class TestReferenceContract(unittest.TestCase):
    @staticmethod
    def config(task="op_aco"):
        data = json.loads(DATA_CONTRACT.read_text(encoding="utf-8"))
        name, method, size, checkpoint, k_sparse = {
            "tsp_aco": ("tsp", "deepaco_tsp64_project_aco_adapted", 50, "tsp100", 10),
            "op_aco": ("op", "deepaco_op64_project_eval", 50, "op100", 10),
            "bpp_offline_aco": ("bpp", "deepaco_bpp64_project_eval", 500, "bpp120", None),
            "cvrp_aco": ("cvrp", "deepaco_cvrp64_project_eval", 50, "cvrp20", None),
            "mkp_aco": ("mkp", "deepaco_mkp64_project_eval", 100, "mkp300", None),
        }[task]
        versions = {key: "test" for key in ("python", "numpy", "torch", "torch-geometric", "pyparsing", "xxhash")}
        if name == "bpp":
            versions.update({"numba": "test", "llvmlite": "test"})
        if name in ("tsp", "cvrp"):
            versions["scipy"] = "test"
        return data, {"status": "frozen", "task": task, "method": method, "size": size,
                      "checkpoint": checkpoint, "checkpoint_scale_mode": "cross_scale", "k_sparse": k_sparse,
                      "source_commit": supervisor.SOURCE_COMMIT, "implementation_sha256": {"fixture": "abc"},
                      "software_versions": versions,
                      "checkpoint_git_blob": supervisor.CHECKPOINT_BLOBS[f"pretrained/{name}/{checkpoint}.pt"],
                      "data_contract_sha256": digest(DATA_CONTRACT),
                      "resource_config_sha256": digest(supervisor.RESOURCE_CONFIG), "solver_repeats_per_instance": 1,
                      "whole_dataset_wall_timeout_seconds": 600, "process_tree_cpu_seconds": 10000}

    def test_all_tasks_require_the_actual_frozen_data_contract(self):
        for task in supervisor.TASKS:
            data, config = self.config(task)
            entry = supervisor.load_reference_contract(config, data, digest(DATA_CONTRACT))
            self.assertEqual(entry["instances"], 64)
            self.assertEqual(entry["size"], config["size"])

    def test_unfrozen_wrong_source_scale_or_budget_are_rejected(self):
        changes = (("status", "engineering"), ("checkpoint_git_blob", "0" * 40),
                   ("checkpoint_scale_mode", "matched"), ("k_sparse", 50),
                   ("solver_repeats_per_instance", 2), ("process_tree_cpu_seconds", 0),
                   ("data_contract_sha256", "0" * 64), ("resource_config_sha256", "0" * 64),
                   ("source_commit", "0" * 40),
                   ("implementation_sha256", {}), ("software_versions", {}))
        for key, value in changes:
            data, config = self.config()
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                supervisor.load_reference_contract(config, data, digest(DATA_CONTRACT))

    def test_changed_implementation_is_rejected_before_start(self):
        _data, config = self.config()
        supervisor.check_reference_implementation(config, {"fixture": "abc"})
        with self.assertRaisesRegex(ValueError, "与冻结配置不一致"):
            supervisor.check_reference_implementation(config, {"fixture": "changed"})

    def test_cvrp_requires_scipy_version_for_distance_calculation(self):
        data, config = self.config("cvrp_aco")
        config["software_versions"].pop("scipy")
        with self.assertRaisesRegex(ValueError, "软件版本"):
            supervisor.load_reference_contract(config, data, digest(DATA_CONTRACT))

    def test_tsp_implementation_snapshot_includes_adapter_and_original_evaluator(self):
        _source, implementation = supervisor.reference_implementation_snapshot("tsp_aco", "tsp100")
        self.assertIn(str((supervisor.ROOT / "scripts/p1_deepaco_tsp.py").resolve()), implementation)
        self.assertIn(str((supervisor.ROOT / "problems/tsp_aco/eval.py").resolve()), implementation)

    def test_cvrp_and_mkp_loaders_reject_wrong_precision(self):
        for name in ("tsp", "cvrp", "mkp"):
            for dtype in (np.float64, np.float32):
                buffer = BytesIO()
                if name == "tsp":
                    np.save(buffer, np.zeros((64, 2, 2), dtype=dtype))
                elif name == "cvrp":
                    np.save(buffer, np.zeros((64, 3, 3), dtype=dtype))
                else:
                    np.savez(buffer, prizes=np.zeros((64, 2), dtype=dtype),
                             weights=np.zeros((64, 2, 5), dtype=dtype))
                buffer.seek(0)
                with self.subTest(task=name, dtype=dtype):
                    if dtype == np.float64:
                        supervisor.load_reference_instances(name, buffer, 2)
                    else:
                        with self.assertRaises(ValueError):
                            supervisor.load_reference_instances(name, buffer, 2)

    def test_task_loader_restores_existing_aco_module(self):
        original = ModuleType("aco")
        with patch.dict(sys.modules, {"aco": original}):
            evaluator = p1_deepaco_worker.load_task("op")
            self.assertIs(sys.modules["aco"], original)
            self.assertTrue(callable(evaluator.check_solution))

    def test_parallel_task_loaders_keep_their_own_aco(self):
        names = ["tsp", "op", "bpp", "cvrp", "mkp"] * 3
        with ThreadPoolExecutor(max_workers=4) as executor:
            loaded = list(executor.map(p1_deepaco_worker.load_task, names))
        for name, evaluator in zip(names, loaded):
            with self.subTest(task=name):
                expected = {"tsp": "tsp_aco", "op": "op_aco", "bpp": "bpp_offline_aco",
                            "cvrp": "cvrp_aco", "mkp": "mkp_aco"}[name]
                self.assertEqual(Path(evaluator.ACO.__init__.__code__.co_filename).parent.name, expected)


class TestReferenceCompletion(unittest.TestCase):
    @staticmethod
    def events(name):
        snapshot = {"model": "abc"}
        versions = {"fixture": "version"}
        entry = {"evaluation_seed_base": 125, "iterations": 2}
        dataset = Path("unused_fixture.npz").resolve()
        if name == "tsp":
            config = {"task": "tsp_aco", "size": 3, "checkpoint": "tsp100", "k_sparse": 1,
                      "software_versions": versions}
            instances = np.repeat(np.array([[[0.0, 0.0], [0.2, 0.0], [0.4, 0.2]]]), 64, axis=0)
            solution = [0, 1, 2]
            distance = tsp_task.distance_matrix(instances[0], instances[0])
            np.fill_diagonal(distance, 1)
            objective = float(distance[solution, np.roll(solution, 1)].sum())
            start = {"event": "start", "dataset": str(dataset), "seed_base": 125, "n_iterations": 2,
                     "n_ants": 30, "n_nodes": 3,
                     "reference": {"id": "deepaco_tsp64_project_aco_adapted", "checkpoint": "tsp100",
                                   "k_sparse": 1, "heuristic_scale": 10.0, "source_sha256": snapshot,
                                   "compute_dtype": "float64", "software_versions": versions}}
            extras = {}
            evaluator, ants, fitness = tsp_task, 30, -objective
        elif name == "op":
            config = {"task": "op_aco", "size": 3, "checkpoint": "op100", "k_sparse": 1,
                      "software_versions": versions}
            entry["max_path_length"] = 3.0
            instances = np.repeat(np.array([[[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]]]), 64, axis=0)
            prizes, _ = op_task.instance(instances[0])
            solution = [0, 1, 3]
            objective, distance = op_task.check_solution(solution, instances[0], prizes.numpy(), 3.0)
            start = {"event": "start", "dataset": str(dataset), "seed_base": 125, "n_iterations": 2,
                     "n_ants": 30, "n_nodes": 3, "max_len": 3.0,
                     "reference": {"id": "deepaco_op64_project_eval", "checkpoint": "op100",
                                   "k_sparse": 1, "source_sha256": snapshot, "compute_dtype": "float64",
                                   "software_versions": versions}}
            extras = {"distance": distance}
            evaluator, ants, fitness = op_task, 30, objective
        elif name == "bpp":
            config = {"task": "bpp_offline_aco", "size": 2, "checkpoint": "bpp120",
                      "software_versions": versions}
            instances = np.tile(np.array([50, 25], dtype=np.int32), (64, 1))
            solution, objective = [0, 0], 1
            start = {"event": "start", "dataset": str(dataset), "seed_base": 125, "n_iterations": 2,
                     "n_ants": 20, "n_items": 2,
                     "reference": {"id": "deepaco_bpp64_project_eval", "checkpoint": "bpp120",
                                   "source_sha256": snapshot, "compute_dtype": "float64",
                                   "software_versions": versions}}
            extras = {"source_fitness": (75 / 150) ** 2}
            evaluator, ants, fitness = bpp_task, 20, -objective
        elif name == "cvrp":
            config = {"task": "cvrp_aco", "size": 2, "checkpoint": "cvrp20",
                      "software_versions": versions}
            instance = np.array([[0, 0.5, 0.5], [1, 0.1, 0.1], [2, 0.9, 0.9]])
            instances = np.repeat(instance[None, :, :], 64, axis=0)
            solution = [0, 1, 2, 0]
            objective = cvrp_task.check_path(solution, instance[:, 1:], instance[:, 0])
            start = {"event": "start", "dataset": str(dataset), "seed_base": 125, "n_iterations": 2,
                     "n_ants": 20, "n_customers": 2, "capacity": 50,
                     "reference": {"id": "deepaco_cvrp64_project_eval", "checkpoint": "cvrp20",
                                   "source_sha256": snapshot, "compute_dtype": "float64",
                                   "software_versions": versions}}
            extras = {}
            evaluator, ants, fitness = cvrp_task, 20, -objective
        else:
            config = {"task": "mkp_aco", "size": 2, "checkpoint": "mkp300",
                      "software_versions": versions}
            prizes = np.tile(np.array([1.0, 2.0]), (64, 1))
            weights = np.tile(np.array([[0.25] * 5, [0.5] * 5]), (64, 1, 1))
            instances = (prizes, weights)
            solution = [0, 1, 2]
            objective = mkp_task.check_solution(solution, prizes[0], weights[0])
            start = {"event": "start", "dataset": str(dataset), "seed_base": 125, "n_iterations": 2,
                     "n_ants": 10, "n_items": 2, "n_dims": 5,
                     "reference": {"id": "deepaco_mkp64_project_eval", "checkpoint": "mkp300",
                                   "source_sha256": snapshot, "compute_dtype": "float64",
                                   "software_versions": versions}}
            extras = {}
            evaluator, ants, fitness = mkp_task, 10, objective
        solution_key = "path" if name == "cvrp" else "tour" if name == "tsp" else "solution"
        rows = [start]
        rows += [{"event": "instance", "index": i, "seed": 125 + i, "iterations": 2,
                  "objective": objective, solution_key: solution.copy(), **extras} for i in range(64)]
        rows.append({"event": "complete", "instance_count": 64, "average": float(objective), "fitness": fitness})
        return config, entry, dataset, instances, evaluator, snapshot, ants, rows

    def read(self, name, rows=None):
        config, entry, dataset, instances, evaluator, snapshot, ants, base = self.events(name)
        rows = base if rows is None else rows
        content = "\n".join(supervisor.PREFIX + json.dumps(row) for row in rows)
        with patch.object(Path, "read_text", return_value=content):
            return supervisor.read_reference_complete(Path("unused"), config, entry, dataset,
                                                      instances, evaluator, snapshot, ants)

    def test_64_instance_results_are_recomputed_for_all_tasks(self):
        for name in ("tsp", "op", "bpp", "cvrp", "mkp"):
            with self.subTest(name=name):
                result = self.read(name)
                self.assertEqual(len(result["objectives"]), 64)
                self.assertTrue(math.isfinite(result["average"]))
                if name == "bpp":
                    self.assertEqual((result["average"], result["fitness"]), (1.0, -1.0))

    def test_wrong_reference_objective_or_solution_never_completes(self):
        for name in ("tsp", "op", "bpp", "cvrp", "mkp"):
            _config, _entry, _dataset, _instances, _evaluator, _snapshot, _ants, rows = self.events(name)
            variants = []
            bad_reference = copy.deepcopy(rows)
            bad_reference[0]["reference"]["source_sha256"] = {"model": "wrong"}
            variants.append(bad_reference)
            missing_reference = copy.deepcopy(rows)
            missing_reference[0]["reference"] = None
            variants.append(missing_reference)
            bad_objective = copy.deepcopy(rows)
            bad_objective[1]["objective"] += 1
            variants.append(bad_objective)
            bad_solution = copy.deepcopy(rows)
            bad_solution[1]["path" if name == "cvrp" else "tour" if name == "tsp" else "solution"] = {
                "tsp": [0, 1, 1], "op": [0, 1, 1], "bpp": [0, 0, 0],
                "cvrp": [0, 1, 1, 0], "mkp": [0, 0]
            }[name]
            variants.append(bad_solution)
            variants.append(rows[:-1])
            if name == "tsp":
                bad_scale = copy.deepcopy(rows)
                bad_scale[0]["reference"]["heuristic_scale"] = 11
                variants.append(bad_scale)
            if name == "bpp":
                bad_internal = copy.deepcopy(rows)
                bad_internal[1]["source_fitness"] = 0.9
                variants.append(bad_internal)
                bad_bin = copy.deepcopy(rows)
                bad_bin[1]["solution"] = [0, 10**100]
                variants.append(bad_bin)
                bad_fitness = copy.deepcopy(rows)
                bad_fitness[-1]["fitness"] = True
                variants.append(bad_fitness)
            for variant in variants:
                with self.subTest(name=name, variant=variant[1]), self.assertRaises((ValueError, TypeError)):
                    self.read(name, variant)


if __name__ == "__main__":
    unittest.main()
