import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import p1_reference_test_evaluate as reference


class TestP1ExpertReferences(unittest.TestCase):
    def test_each_reference_matches_its_verified_reevo_seed(self):
        for task in reference.SEED_BLOBS:
            with self.subTest(task=task):
                local, source = reference.verify_source(task)
                self.assertTrue(local.is_file())
                self.assertTrue(source.is_file())

    def test_reference_records_runtime_paths_and_contract_files(self):
        packages = [("numpy", "2.3.5"), ("scipy", "1.0"), ("torch", "2.0"),
                    ("numba", "0.68.1"), ("llvmlite", "0.50.0"), ("numba", "0.1")]
        distributions = [SimpleNamespace(metadata={"Name": name}, version=version) for name, version in packages]
        resources = {"pythonpath": ["project-dependencies"]}
        with patch.object(reference.metadata, "distributions", return_value=distributions) as discover:
            versions = reference.software_versions("tsp_gls", resources)
        discover.assert_called_once_with(path=[r"D:\Tools\agentic-heuristic-p1-numba", "project-dependencies"])
        self.assertEqual(versions["numba"], "0.68.1")
        snapshot = reference.implementation_snapshot("tsp_gls")
        for path in ("cfg/problem/tsp_gls.yaml", "prompts/tsp_gls/func_desc.txt", "prompts/tsp_gls/func_signature.txt"):
            self.assertIn(str(reference.ROOT / path), snapshot)

    def test_frozen_contract_rejects_wrong_identity_and_budget(self):
        data = {"status": "frozen", "contract_id": "p1_data_20261104_v1",
                "tasks": {"tsp_gls": {"test": [{"size": 20, "instances": 64}]}}}
        config = {"status": "frozen", "task": "tsp_gls", "method": reference.method_for("tsp_gls"),
                  "source_commit": reference.SOURCE_COMMIT, "compute_dtype": "float64",
                  "solver_repeats_per_instance": 1, "data_contract_sha256": "data",
                  "resource_config_sha256": "resources", "size": 20,
                  "whole_dataset_wall_timeout_seconds": 30, "process_tree_cpu_seconds": 30,
                  "implementation_sha256": {"file": "hash"}, "software_versions": {}}
        self.assertEqual(reference.load_reference_contract(config, data, "data", "resources")["size"], 20)
        for key, value in (("status", "engineering"), ("method", "aco_reevo_seed_float64_adapted"),
                           ("compute_dtype", "float32"), ("solver_repeats_per_instance", 2),
                           ("process_tree_cpu_seconds", 0), ("size", 21)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                reference.load_reference_contract(dict(config, **{key: value}), data, "data", "resources")

    def test_tsp_routes_are_recomputed_independently(self):
        points = np.random.RandomState(508).random_sample((64, 20, 2))
        tour = list(range(20))
        events = []
        for index, coordinates in enumerate(points):
            length = math.fsum(math.dist(coordinates[a], coordinates[b]) for a, b in zip(tour, tour[1:] + tour[:1]))
            events.append({"event": "instance", "index": index, "tour": tour[:], "objective": length})
        payload = "\n".join(reference.PREFIX + json.dumps(event) for event in events)
        with patch.object(np, "load", return_value=points), patch.object(Path, "read_text", return_value=payload):
            reference.verify_tsp_routes(Path("stdout"), Path("dataset"), 20)
        events[10]["tour"][1] = events[10]["tour"][0]
        payload = "\n".join(reference.PREFIX + json.dumps(event) for event in events)
        with patch.object(np, "load", return_value=points), patch.object(Path, "read_text", return_value=payload):
            with self.assertRaises(ValueError):
                reference.verify_tsp_routes(Path("stdout"), Path("dataset"), 20)


if __name__ == "__main__":
    unittest.main()
