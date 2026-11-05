import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from baselines.neuopt.synthetic_check import route_from_successor, solve_batch, verify_sources
from scripts.p1_neuopt_check import audit


class TestNeuOptReference(unittest.TestCase):
    def test_official_sources_and_four_tsp_weights_match_fixed_commit(self):
        verify_sources()

    def test_successor_representation_must_be_one_complete_cycle(self):
        self.assertEqual(route_from_successor([1, 2, 0]), [0, 1, 2])
        for value in ([], [1, 0, 2], [1, 2, 3], [True, 0], [1, -1]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                route_from_successor(value)

    def test_single_instance_and_unknown_scale_are_rejected_before_inference(self):
        for points in (np.zeros((1, 20, 2)), np.zeros((2, 21, 2)), np.full((2, 20, 2), np.nan)):
            with self.subTest(shape=points.shape), self.assertRaises(ValueError):
                solve_batch(points, 5, 10, "cpu")

    def test_engineering_audit_recomputes_both_routes(self):
        size, seed = 20, 20790020
        points = np.random.RandomState(seed).random_sample((2, size, 2))
        route = list(range(size))
        instances = []
        for coordinates in points:
            length = math.fsum(math.hypot(*(coordinates[a] - coordinates[b]))
                               for a, b in zip(route, route[1:] + route[:1]))
            instances.append({"route": route[:], "float64_recomputed_length": length, "source_float32_length": length})
        event = {"status": "passed", "size": size, "steps": 5, "seed": seed, "device": "cuda",
                 "batch_size": 2, "instances": instances, "formal_test_data_used": False,
                 "search_elapsed_seconds": 1.0, "peak_gpu_allocated_bytes": 1024}
        with patch.object(Path, "read_text", return_value="P1_NEUOPT_JSON " + json.dumps(event)):
            self.assertEqual(audit(Path("fixture"), size, 5, seed, "cuda")["status"], "passed")
        event["instances"][0]["route"][1] = 0
        with patch.object(Path, "read_text", return_value="P1_NEUOPT_JSON " + json.dumps(event)):
            with self.assertRaises(ValueError):
                audit(Path("fixture"), size, 5, seed, "cuda")


if __name__ == "__main__":
    unittest.main()
