import importlib.util
from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import distance_matrix
import torch

from baselines.deepaco.reference import DeepACOReference, load_source
from tests.test_p1_cvrp_aco import task as cvrp_task
from tests.test_p1_mkp_aco import task as mkp_task


HAS_PYG = importlib.util.find_spec("torch_geometric") is not None


def run_synthetic_training_main(task, model, *arrays):
    buffer = BytesIO()
    if model.task == "cvrp":
        np.save(buffer, arrays[0])
        size, filename = "50", "in_memory_deepaco_cvrp.npy"
    else:
        np.savez(buffer, prizes=arrays[0], weights=arrays[1])
        size, filename = "100", "in_memory_deepaco_mkp.npz"
    payload, fixture_path = buffer.getvalue(), Path(filename)
    native_read, native_load = Path.read_bytes, np.load

    def read(path):
        return payload if path == fixture_path else native_read(path)

    def load(path, *args, **kwargs):
        return native_load(BytesIO(payload) if path == fixture_path else path, *args, **kwargs)

    events = []
    with patch.object(task, "DATASET", fixture_path), patch.object(task, "DATASET_SHA256", sha256(payload).hexdigest()), \
         patch.object(task, "N_INSTANCES", 1), patch.object(task, "N_ITERATIONS", 1), \
         patch.object(Path, "read_bytes", read), patch.object(np, "load", side_effect=load), \
         patch.object(task, "emit", side_effect=events.append), \
         patch.object(sys, "argv", ["eval.py", size, "unused", "train"]):
        task.main(deepaco_model=model)
    return events


@unittest.skipUnless(HAS_PYG, "真实 DeepACO 网络需要已授权的 PyG 目录")
class TestRealCVRP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DeepACOReference("cvrp", "cvrp20")
        rng = np.random.RandomState(126)
        cls.coordinates = rng.rand(11, 2)
        cls.coordinates[0] = (0.5, 0.5)
        cls.demands = np.r_[0, rng.randint(1, 10, size=10)].astype(np.float64)

    def test_source_search_and_project_route_length_match(self):
        scores = self.model.predict(coordinates=self.coordinates, demands=self.demands)
        distances = distance_matrix(self.coordinates, self.coordinates)
        np.fill_diagonal(distances, 0)
        source = load_source("cvrp", "aco")
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            torch.manual_seed(159)
            original = source.ACO(torch.from_numpy(distances), torch.from_numpy(self.demands.copy()),
                                  n_ants=cvrp_task.N_ANTS, heuristic=torch.from_numpy(scores.copy()),
                                  capacity=cvrp_task.CAPACITY, device="cpu")
            source_cost = float(original.run(2))
            torch.set_default_dtype(torch.float32)
            torch.manual_seed(159)
            objective, path, iterations = cvrp_task.solve_deepaco(
                self.coordinates, self.demands, self.model, n_iterations=2)
        finally:
            torch.set_default_dtype(original_dtype)
        self.assertEqual(path, original.shortest_path.tolist())
        self.assertEqual(objective, source_cost)
        self.assertEqual(iterations, 2)
        self.assertTrue(math.isclose(objective, cvrp_task.check_path(path, self.coordinates, self.demands),
                                     rel_tol=1e-12, abs_tol=1e-12))

    def test_network_matrix_uses_source_flattened_edge_order(self):
        points = torch.from_numpy(self.coordinates.copy())
        demands = torch.from_numpy(self.demands.copy())
        graph = self.model.utils.gen_pyg_data(demands, self.model.utils.gen_distance_matrix(points), "cpu")
        with torch.no_grad():
            expected = self.model.model(graph).reshape((len(points), len(points))) + 1e-10
        actual = self.model.predict(coordinates=self.coordinates, demands=self.demands)
        np.testing.assert_array_equal(actual, expected.numpy())

    def test_rejects_float32_coordinates_or_demands(self):
        for coordinates, demands in ((self.coordinates.astype(np.float32), self.demands),
                                     (self.coordinates, self.demands.astype(np.float32))):
            with self.subTest(dtype=(coordinates.dtype, demands.dtype)), self.assertRaises(ValueError):
                cvrp_task.solve_deepaco(coordinates, demands, self.model, n_iterations=1)

    def test_real_model_emits_project_events_on_synthetic_instance(self):
        rng = np.random.RandomState(128)
        instance = np.c_[np.r_[0, rng.randint(1, 10, size=50)], rng.rand(51, 2)]
        instance[0, 1:] = (0.5, 0.5)
        events = run_synthetic_training_main(cvrp_task, self.model, instance[None, :, :])
        self.assertEqual([event["event"] for event in events], ["start", "instance", "complete"])
        self.assertEqual(events[0]["reference"]["id"], "deepaco_cvrp64_project_eval")
        expected = cvrp_task.check_path(events[1]["path"], instance[:, 1:], instance[:, 0])
        self.assertAlmostEqual(events[1]["objective"], expected, places=12)
        self.assertEqual(events[-1]["fitness"], -events[-1]["average"])


@unittest.skipUnless(HAS_PYG, "真实 DeepACO 网络需要已授权的 PyG 目录")
class TestRealMKP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DeepACOReference("mkp", "mkp300")
        rng = np.random.RandomState(127)
        cls.prize = rng.rand(12)
        cls.weight = rng.rand(12, 5) / 6

    def test_source_search_and_project_capacity_match(self):
        scores = self.model.predict(prize=self.prize, weight=self.weight)
        source = load_source("mkp", "aco")
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            torch.manual_seed(160)
            original = source.ACO(torch.from_numpy(self.prize.copy()),
                                  torch.from_numpy(self.weight * (len(self.prize) // 2)),
                                  n_ants=mkp_task.N_ANTS,
                                  pheromone=torch.ones((len(self.prize) + 1, len(self.prize) + 1)),
                                  heuristic=torch.from_numpy(scores.copy()), device="cpu")
            source_objective, source_solution = original.run(2)
            torch.set_default_dtype(torch.float32)
            torch.manual_seed(160)
            objective, solution, iterations = mkp_task.solve_deepaco(
                self.prize, self.weight, self.model, n_iterations=2)
        finally:
            torch.set_default_dtype(original_dtype)
        self.assertEqual(solution, source_solution.tolist())
        self.assertEqual(objective, float(source_objective))
        self.assertEqual(iterations, 2)
        self.assertTrue(math.isclose(objective, mkp_task.check_solution(solution, self.prize, self.weight),
                                     rel_tol=1e-12, abs_tol=1e-12))

    def test_network_matrix_uses_source_order_and_normalization(self):
        prize = torch.from_numpy(self.prize.copy())
        weight = torch.from_numpy(self.weight * (len(self.prize) // 2))
        graph = self.model.utils.gen_pyg_data(prize, weight)
        with torch.no_grad():
            raw = self.model.model(graph).reshape((len(prize), len(prize)))
            expected = raw / (raw.min() + 1e-10) + 1e-10
        actual = self.model.predict(prize=self.prize, weight=self.weight)
        np.testing.assert_array_equal(actual, expected.numpy())

    def test_capacity_accumulator_remains_float64(self):
        source = load_source("mkp", "aco")
        update = source.ACO.update_knapsack
        observed = []

        def check(solver, mask, knapsack, item):
            observed.append((knapsack.dtype, solver.weight.dtype))
            return update(solver, mask, knapsack, item)

        def source_for_test(task, filename):
            return source if (task, filename) == ("mkp", "aco") else load_source(task, filename)

        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            torch.manual_seed(161)
            with patch.object(source.ACO, "update_knapsack", check), \
                 patch("baselines.deepaco.reference.load_source", side_effect=source_for_test):
                mkp_task.solve_deepaco(self.prize, self.weight, self.model, n_iterations=1)
        finally:
            torch.set_default_dtype(original_dtype)
        self.assertTrue(observed)
        self.assertTrue(all(pair == (torch.float64, torch.float64) for pair in observed))

    def test_rejects_float32_prizes_or_weights(self):
        for prize, weight in ((self.prize.astype(np.float32), self.weight),
                              (self.prize, self.weight.astype(np.float32))):
            with self.subTest(dtype=(prize.dtype, weight.dtype)), self.assertRaises(ValueError):
                mkp_task.solve_deepaco(prize, weight, self.model, n_iterations=1)

    def test_real_model_emits_project_events_on_synthetic_instance(self):
        rng = np.random.RandomState(129)
        prizes = rng.rand(1, 100)
        weights = rng.rand(1, 100, 5) / 50
        events = run_synthetic_training_main(mkp_task, self.model, prizes, weights)
        self.assertEqual([event["event"] for event in events], ["start", "instance", "complete"])
        self.assertEqual(events[0]["reference"]["id"], "deepaco_mkp64_project_eval")
        expected = mkp_task.check_solution(events[1]["solution"], prizes[0], weights[0])
        self.assertAlmostEqual(events[1]["objective"], expected, places=12)
        self.assertEqual(events[-1]["fitness"], events[-1]["average"])


if __name__ == "__main__":
    unittest.main()
