import io
import json
import math
import os
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from baselines.gnngls import reference as engine
from scripts import p1_gnngls_test_evaluate as runner


class SourceScalerUnpickler(engine.RestrictedScalerUnpickler):
    def find_class(self, module, name):
        if (module, name) == ("sklearn.preprocessing._data", "MinMaxScaler"):
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler
        return super().find_class(module, name)


class TestGNNGLSReference(unittest.TestCase):
    def setUp(self):
        self.points = np.random.RandomState(503).random_sample((3, 20, 2))
        self.entry = {"size": 20, "instances": 64, "sha256": "dataset", "evaluation_seed_base": 900}
        self.config = {"status": "frozen", "task": "tsp_gls", "method": runner.METHOD,
                       "source_commit": engine.SOURCE_COMMIT, "compute_dtype": "float64", "device": "cpu",
                       "data_contract_sha256": "data", "resource_config_sha256": "resources",
                       "lkh_reference_sha256": "lkh", "solver_repeats_per_instance": 1,
                       "size": 20, "checkpoint_size": 20, "scaler_size": 20,
                       "checkpoint_scale_mode": "matched", "scaler_scale_mode": "matched",
                       "search_time_limit_seconds": 3, "perturbation_moves": 3,
                       "whole_dataset_wall_timeout_seconds": 30, "process_tree_cpu_seconds": 30,
                       "implementation_sha256": {"file": "hash"}, "software_versions": {}}
        self.data = {"status": "frozen", "contract_id": "p1_data_20261104_v1",
                     "tasks": {"tsp_gls": {"test": [self.entry]}}}
        self.lkh = {"sha256": "lkh", "objectives": [100.0, 100.0, 100.0], "mean": 100.0}

    def test_official_source_and_assets_match_fixed_hashes(self):
        engine.verify_sources()

    def test_scaler_matches_source_transform_and_inverse(self):
        for size in (20, 50, 100):
            path = engine.ROOT / f"data/tsp{size}/scalers.pkl"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scalers = SourceScalerUnpickler(io.BytesIO(path.read_bytes())).load()["edges"]
            scale, offset, regret_scale, regret_offset = engine.load_scalers(size)[0]
            inputs = np.random.RandomState(size).random_sample((8, 15))
            actual = scalers["features"].transform(inputs)[:, 0]
            np.testing.assert_allclose(actual, inputs[:, 0] * scale + offset, rtol=0, atol=0)
            regret = np.random.RandomState(size + 1).random_sample((8, 1))
            actual_regret = scalers["regret"].inverse_transform(regret)[:, 0]
            np.testing.assert_allclose(actual_regret, (regret[:, 0] - regret_offset) / regret_scale, rtol=0, atol=0)

    def test_scaler_unpickler_rejects_unrelated_globals(self):
        with self.assertRaises(pickle.UnpicklingError):
            engine.RestrictedScalerUnpickler(io.BytesIO(pickle.dumps(os.system))).load()

    def test_contract_requires_frozen_identity_and_explicit_cross_scale(self):
        self.assertEqual(runner.load_reference_contract(self.config, self.data, "data", "resources"), self.entry)
        for key, value in (("status", "pending"), ("compute_dtype", "float32"),
                           ("checkpoint_size", 50), ("scaler_size", 100),
                           ("solver_repeats_per_instance", 2), ("search_time_limit_seconds", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.load_reference_contract(dict(self.config, **{key: value}), self.data, "data", "resources")

    def test_formal_data_requires_64_float64_instances(self):
        valid = np.zeros((64, 20, 2), dtype=np.float64)
        for value, accepted in ((valid, True), (valid[:63], False), (valid.astype(np.float32), False)):
            with self.subTest(shape=value.shape, dtype=value.dtype), patch.object(np, "load", return_value=value):
                if accepted:
                    self.assertEqual(runner.load_instances(Path("fixture"), 20).shape, (64, 20, 2))
                else:
                    with self.assertRaises(ValueError):
                        runner.load_instances(Path("fixture"), 20)

    def test_events_recompute_each_route_and_check_reference_stop(self):
        calls = []
        def stub(points, *, checkpoint_size, scaler_size, time_limit, perturbation_moves, reference):
            calls.append(1)
            route = list(range(20))
            distance = math.fsum(math.dist(points[a], points[b]) for a, b in zip(route, route[1:] + route[:1]))
            return {"route": route, "objective": distance, "source_cost": distance, "size": 20,
                    "checkpoint_size": checkpoint_size, "scaler_size": scaler_size, "compute_dtype": "float64",
                    "device": "cpu", "time_limit": time_limit, "perturbation_moves": perturbation_moves,
                    "stop_reason": "reference_reached", "iterations": 0, "prediction_seconds": 0.01,
                    "elapsed_seconds": 0.02, "graph_setup_seconds": 0.0,
                    "graph_cache_hit": len(calls) > 1, "scaler_source_sklearn": "0.24.2"}

        events = []
        with patch.object(runner, "solve", side_effect=stub):
            runner.run_instances(self.points, self.entry, self.config, self.lkh, events.append)
        self.assertEqual([event["seed"] for event in events[1:-1]], [900, 901, 902])
        payload = "\n".join(runner.PREFIX + json.dumps(event) for event in events)
        with patch.object(Path, "read_text", return_value=payload):
            result = runner.read_complete(Path("fixture"), self.points, self.entry, self.config, self.lkh)
        self.assertLess(result["lkh_gap_percent"], 0)
        self.assertEqual(result["stopping_reasons"], ["reference_reached"] * 3)
        events[1]["stop_reason"] = "time_limit"
        payload = "\n".join(runner.PREFIX + json.dumps(event) for event in events)
        with patch.object(Path, "read_text", return_value=payload), self.assertRaises(ValueError):
            runner.read_complete(Path("fixture"), self.points, self.entry, self.config, self.lkh)


if __name__ == "__main__":
    unittest.main()
