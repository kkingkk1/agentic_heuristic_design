import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts import p1_tsp_gls_lkh_test as reference


class TestLKHTestReference(unittest.TestCase):
    def test_float64_closed_tour_and_invalid_routes(self):
        points = np.array([[0.0, 0.0], [1.00000004, 0.0],
                           [1.00000004, 1.0], [0.0, 1.0]], dtype=np.float64)
        tour = [0, 1, 2, 3, 0]
        value = reference.reference_objective(points, tour)
        self.assertAlmostEqual(value, 4.00000008, places=13)
        self.assertNotEqual(value, float(np.float32(4.00000008)))
        for invalid in ([0, 1, 2, 3], [0, 1, 1, 3, 0], [0, 1, 2, 3, 1]):
            with self.subTest(tour=invalid), self.assertRaises(ValueError):
                reference.reference_objective(points, invalid)
        with self.assertRaises(ValueError):
            reference.reference_objective(points.astype(np.float32), tour)

    def test_frozen_datasets_have_256_float64_instances(self):
        datasets = reference.load_datasets()
        self.assertEqual(sorted(datasets), [20, 50, 100, 200])
        for size, data in datasets.items():
            with self.subTest(size=size):
                self.assertEqual(data["points"].shape, (64, size, 2))
                self.assertEqual(data["points"].dtype, np.float64)

    def test_completion_accepts_out_of_order_and_rejects_duplicate(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        datasets = {4: {"path": reference.ROOT / "fixture.npy", "sha256": "hash",
                        "points": np.stack([points, points])}}
        start = {"event": "start", "sizes": [4], "instances_per_size": 2,
                 "scale": reference.SCALE, "runs": reference.RUNS, "elkai_version": "2.0.1",
                 "data_contract_sha256": "hash", "optimality_proven": False}
        event = {"event": "instance", "size": 4, "tour": [0, 1, 2, 3, 0],
                 "objective": 4.0, "elapsed_seconds": 0.1}
        complete = {"event": "complete", "instances": 2, "optimality_proven": False}
        events = [start, {**event, "index": 1}, {**event, "index": 0}, complete]

        def check(rows):
            text = "\n".join(reference.PREFIX + json.dumps(row) for row in rows)
            with patch.object(reference, "SIZES", (4,)), patch.object(reference, "COUNT", 2), \
                    patch.object(reference, "digest", return_value="hash"), \
                    patch.object(Path, "read_text", return_value=text):
                return reference.read_complete(Path("unused"), datasets)

        _, result = check(events)
        self.assertEqual([item["index"] for item in result["4"]["instances"]], [0, 1])
        self.assertEqual(result["4"]["mean_objective"], 4.0)
        with self.assertRaisesRegex(ValueError, "重复"):
            check([start, {**event, "index": 0}, {**event, "index": 0}, complete])
        with self.assertRaisesRegex(ValueError, "不完整"):
            check(events[:-1])


if __name__ == "__main__":
    unittest.main()
