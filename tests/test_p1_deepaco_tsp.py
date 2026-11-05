import importlib.util
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.deepaco.reference import DeepACOReference, load_source
from scripts import p1_deepaco_tsp as reference
from scripts.p1_deepaco_worker import load_task


HAS_PYG = importlib.util.find_spec("torch_geometric") is not None


@unittest.skipUnless(HAS_PYG, "真实 DeepACO 网络需要已授权的 PyG 目录")
class TestRealTSP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DeepACOReference("tsp", "tsp100")
        cls.task = load_task("tsp")
        cls.positions = np.random.RandomState(465).rand(50, 2)

    def test_project_aco_matches_source_with_same_seed_for_full_training_iterations(self):
        scores = self.model.predict(coordinates=self.positions, k_sparse=20)
        distances = self.task.distance_matrix(self.positions, self.positions)
        np.fill_diagonal(distances, 1)
        adjusted = scores * reference.SCALE - 1e-9
        processed = self.task.checked_heuristic(lambda _distance: adjusted, distances)
        np.testing.assert_allclose(processed, reference.SCALE * scores, rtol=1e-15, atol=1e-24)
        source = load_source("tsp", "aco")
        torch.manual_seed(1310)
        original = source.ACO(torch.from_numpy(distances), n_ants=self.task.N_ANTS,
                              heuristic=torch.from_numpy(scores.copy()), device="cpu")
        source_objective = float(original.run(200))
        torch.manual_seed(1310)
        objective, tour, iterations = reference.solve(
            self.task, self.positions, self.model, k_sparse=20, n_iterations=200)
        self.assertEqual(tour, original.shortest_path.tolist())
        self.assertEqual(objective, source_objective)
        self.assertEqual(iterations, 200)

    def test_rejects_float32_input(self):
        with self.assertRaises(ValueError):
            reference.solve(self.task, self.positions.astype(np.float32), self.model,
                            k_sparse=20, n_iterations=1)

    def test_real_model_emits_project_events_on_synthetic_instance(self):
        data = self.positions[None, :, :]
        buffer = BytesIO()
        np.save(buffer, data)
        payload, path = buffer.getvalue(), Path("in_memory_deepaco_tsp.npy")
        expected_hash = sha256(payload).hexdigest()
        native_read = Path.read_bytes

        def read(candidate):
            return payload if candidate == path else native_read(candidate)

        events = []
        with patch.object(self.task, "test_dataset", return_value=(path, data)), \
             patch.object(self.task, "TEST_INSTANCES", 1), patch.object(self.task, "TEST_ITERATIONS", 1), \
             patch.object(self.task, "TEST_DATASET_SHA256", {50: expected_hash}), \
             patch.object(Path, "read_bytes", read), \
             patch.object(self.task, "emit", side_effect=lambda event, _prefix: events.append(event)):
            reference.run(self.task, self.model, size=50, phase="test", k_sparse=20)
        self.assertEqual([event["event"] for event in events], ["start", "instance", "complete"])
        self.assertEqual(events[0]["reference"]["id"], reference.REFERENCE_ID)
        tour = events[1]["tour"]
        self.assertEqual(sorted(tour), list(range(50)))
        distances = self.task.distance_matrix(self.positions, self.positions)
        expected = float(distances[tour, np.roll(tour, 1)].sum())
        self.assertAlmostEqual(events[1]["objective"], expected, places=12)
        self.assertEqual(events[-1]["fitness"], -events[-1]["average"])


if __name__ == "__main__":
    unittest.main()
