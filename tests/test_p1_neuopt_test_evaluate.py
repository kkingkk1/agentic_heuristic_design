import io
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.neuopt import synthetic_check
from scripts import p1_neuopt_test_evaluate as reference


class TestNeuOptTestReference(unittest.TestCase):
    def setUp(self):
        self.points = np.random.RandomState(501).random_sample((3, 20, 2))
        self.entry = {"size": 20, "instances": 64, "sha256": "dataset", "evaluation_seed_base": 800}
        self.config = {"status": "frozen", "task": "tsp_gls", "method": reference.METHOD,
                       "source_commit": reference.SOURCE_COMMIT, "compute_dtype": "float64",
                       "data_contract_sha256": "data", "resource_config_sha256": "resources",
                       "lkh_reference_sha256": "lkh", "solver_repeats_per_instance": 1,
                       "size": 20, "checkpoint": "tsp20.pt", "search_steps": 5, "device": "cpu",
                       "whole_dataset_wall_timeout_seconds": 30, "process_tree_cpu_seconds": 30,
                       "implementation_sha256": {"file": "hash"}, "software_versions": {}}
        self.data = {"status": "frozen", "contract_id": "p1_data_20261104_v1",
                     "tasks": {"tsp_gls": {"test": [self.entry]}}}
        self.lkh = {"sha256": "lkh", "objectives": [100.0, 100.0, 100.0], "mean": 100.0}

    def test_contract_requires_frozen_identity_budget_and_individual_seeds(self):
        self.assertEqual(reference.load_reference_contract(self.config, self.data, "data", "resources"), self.entry)
        for key, value in (("status", "pending"), ("compute_dtype", "float32"),
                           ("checkpoint", "tsp50.pt"), ("search_steps", 0),
                           ("solver_repeats_per_instance", 2), ("process_tree_cpu_seconds", float("inf"))):
            config = dict(self.config, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                reference.load_reference_contract(config, self.data, "data", "resources")

    def test_single_instance_uses_one_result_from_two_identical_copies(self):
        result = {"instances": [{"route": list(range(20)), "float64_recomputed_length": 2.0,
                                 "solver_reported_length": 2.0},
                                {"route": list(reversed(range(20))), "float64_recomputed_length": 1.0,
                                 "solver_reported_length": 1.0}],
                  "search_elapsed_seconds": 0.5, "peak_gpu_allocated_bytes": None}
        with patch.object(synthetic_check, "solve_batch", return_value=result) as solve:
            actual = synthetic_check.solve_one(self.points[0], 5, 800, "cpu")
        batch = solve.call_args.args[0]
        np.testing.assert_array_equal(batch[0], self.points[0])
        np.testing.assert_array_equal(batch[1], self.points[0])
        self.assertEqual(solve.call_args.kwargs["compute_dtype"], "float64")
        self.assertEqual(actual["objective"], 2.0)
        with self.assertRaises(ValueError):
            synthetic_check.solve_one(self.points[0].astype(np.float32), 5, 800, "cpu")

    def test_formal_data_requires_64_float64_instances(self):
        valid = np.zeros((64, 20, 2), dtype=np.float64)
        for value, accepted in ((valid, True), (valid[:63], False), (valid.astype(np.float32), False)):
            with self.subTest(shape=value.shape, dtype=value.dtype), patch.object(np, "load", return_value=value):
                if accepted:
                    self.assertEqual(reference.load_instances(Path("fixture"), 20).shape, (64, 20, 2))
                else:
                    with self.assertRaises(ValueError):
                        reference.load_instances(Path("fixture"), 20)

    def test_torch_default_precision_restored_when_solver_fails(self):
        before = torch.get_default_dtype()
        with patch.object(synthetic_check, "_solve_batch", side_effect=RuntimeError("fixture")):
            with self.assertRaises(RuntimeError):
                synthetic_check.solve_batch(self.points[:2], 5, 800, "cpu", "float64")
        self.assertEqual(torch.get_default_dtype(), before)

    def test_worker_rejects_different_software_before_running(self):
        resolved = (None, self.config, self.entry, Path("dataset"), self.lkh)
        with patch.object(sys, "argv", ["reference", "--worker", "fixture"]), \
                patch.object(sys, "stderr", new_callable=io.StringIO), \
                patch.object(reference, "resolve_input", return_value=resolved), \
                patch.object(reference, "software_versions", return_value={"torch": "different"}), \
                patch.object(reference, "run_instances") as run:
            self.assertEqual(reference.main(), 1)
        run.assert_not_called()

    def test_events_keep_seeds_and_independently_check_routes_and_gap(self):
        def stub(points, steps, seed, device):
            route = list(range(20))
            distance = math.fsum(math.dist(points[a], points[b]) for a, b in zip(route, route[1:] + route[:1]))
            return {"route": route, "objective": distance, "solver_reported_length": distance,
                    "search_elapsed_seconds": 0.01, "peak_gpu_allocated_bytes": None}

        events = []
        with patch.object(reference, "solve_one", side_effect=stub) as solver:
            reference.run_instances(self.points, self.entry, self.config, self.lkh, events.append)
        self.assertEqual([call.args[2] for call in solver.call_args_list], [800, 801, 802])
        self.assertEqual([event["seed"] for event in events[1:-1]], [800, 801, 802])
        payload = "\n".join(reference.PREFIX + json.dumps(event) for event in events)
        with patch.object(Path, "read_text", return_value=payload):
            audited = reference.read_complete(Path("fixture"), self.points, self.entry, self.config, self.lkh)
        self.assertLess(audited["lkh_gap_percent"], 0)
        events[1]["route"][1] = events[1]["route"][0]
        payload = "\n".join(reference.PREFIX + json.dumps(event) for event in events)
        with patch.object(Path, "read_text", return_value=payload), self.assertRaises(ValueError):
            reference.read_complete(Path("fixture"), self.points, self.entry, self.config, self.lkh)


if __name__ == "__main__":
    unittest.main()
