import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import distance_matrix
import torch

from scripts import p1_cvrp_aco_evaluate as supervisor


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "problems" / "cvrp_aco"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source_aco = load_module("cvrp_source_aco", TASK_DIR / "aco.py")
with patch.dict(sys.modules, {"aco": source_aco}):
    task = load_module("p1_cvrp_eval", TASK_DIR / "eval.py")
generator = load_module("p1_cvrp_data", TASK_DIR / "gen_p1_train.py")


class TestCVRP(unittest.TestCase):
    def test_training_data_matches_fixed_upstream_blob(self):
        self.assertEqual(generator.build_dataset(), task.DATASET.read_bytes())
        instances = np.load(task.DATASET, allow_pickle=False)
        self.assertEqual(instances.shape, (10, 51, 3))
        np.testing.assert_array_equal(instances[:, 0, 0], np.zeros(10))
        np.testing.assert_array_equal(instances[:, 0, 1:], np.full((10, 2), 0.5))

    def test_padding_has_zero_physical_cost(self):
        coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        demand = np.array([0.0, 10.0, 10.0])
        distances = distance_matrix(coordinates, coordinates)
        np.fill_diagonal(distances, 1.0)
        padded = torch.tensor([[0], [1], [2], [0], [0]], dtype=torch.int64)
        source = task.ACO(distances, demand, np.ones((3, 3)), 50, n_ants=1)
        source_cost = source.gen_path_costs(padded).item()
        physical = task.check_path([0, 1, 2, 0, 0], coordinates, demand)
        self.assertAlmostEqual(source_cost, physical + 1.0)
        self.assertAlmostEqual(source.gen_path_costs(padded[:-1]).item(), physical)
        np.fill_diagonal(distances, 0.0)
        fixed = task.ACO(distances, demand, np.ones((3, 3)), 50, n_ants=1)
        self.assertAlmostEqual(fixed.gen_path_costs(padded).item(), physical)

    def test_path_rejects_missing_duplicate_and_capacity(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        demands = np.array([0.0, 30.0, 30.0])
        self.assertGreater(task.check_path([0, 1, 0, 2, 0, 0], coords, demands), 0)
        for path in ([0, 1, 0], [0, 1, 1, 2, 0], [0, 1, 2, 0], [1, 0, 2, 0]):
            with self.subTest(path=path), self.assertRaises(ValueError):
                task.check_path(path, coords, demands)

    def test_black_box_receives_normalized_demands_and_copies(self):
        distances = np.array([[1.0, 2.0], [2.0, 1.0]])
        coords = np.array([[0.5, 0.5], [0.1, 0.1]])
        demands = np.array([0.0, 5.0])

        def heuristic(edge_attr, node_attr):
            np.testing.assert_allclose(node_attr, [0.0, 0.1])
            edge_attr[0, 0] = 99
            node_attr[1] = 99
            return np.ones((2, 2))

        task.checked_heuristic(heuristic, distances, coords, demands, black_box=True)
        self.assertEqual(distances[0, 0], 1.0)
        self.assertEqual(demands[1], 5.0)
        with self.assertRaises(ValueError):
            task.checked_heuristic(lambda *_args: np.full((2, 2), np.nan), distances, coords, demands, black_box=True)

    def test_100_iterations_match_original_solver_with_corrected_cost(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        demands = np.array([0.0, 10.0, 10.0, 10.0])
        heuristic = lambda d, _c, _d, _cap: 1.0 / d
        torch.manual_seed(9262)
        actual, path, iterations = task.solve(coords, demands, heuristic)
        distances = distance_matrix(coords, coords)
        np.fill_diagonal(distances, 1.0)
        scores = heuristic(distances, coords, demands, 50) + 1e-9
        np.fill_diagonal(distances, 0.0)
        torch.manual_seed(9262)
        original = task.ACO(distances, demands, scores, 50, n_ants=20).run(100).item()
        self.assertEqual(actual, original)
        self.assertAlmostEqual(actual, task.check_path(path, coords, demands))
        self.assertEqual(iterations, 100)


class TestCompletion(unittest.TestCase):
    @staticmethod
    def events():
        instances = np.load(task.DATASET, allow_pickle=False)
        start = {"event": "start", "seed_base": 20261103, "n_ants": 20,
                 "n_iterations": 100, "n_customers": 50, "capacity": 50}
        paths = []
        for index, instance in enumerate(instances):
            path = [0]
            for customer in range(1, 51):
                path.extend([customer, 0])
            coords = instance[:, 1:]
            objective = math.fsum(math.hypot(*(coords[a] - coords[b])) for a, b in zip(path, path[1:]))
            paths.append({"event": "instance", "index": index, "seed": 20261103 + index,
                          "iterations": 100, "objective": objective, "path": path})
        average = sum(x["objective"] for x in paths) / 10
        return [start, *paths, {"event": "complete", "instance_count": 10,
                               "average": average, "fitness": -average}]

    @staticmethod
    def read(events):
        text = "\n".join(supervisor.EVENT_PREFIX + json.dumps(event) for event in events)
        with patch.object(Path, "read_text", return_value=text):
            return supervisor.read_complete(Path("unused"))

    def test_complete_result_has_negative_fitness(self):
        result = self.read(self.events())
        self.assertEqual(result["fitness"], -result["average"])

    def test_partial_duplicate_capacity_and_wrong_cost_fail(self):
        partial = self.events()[:-1]
        duplicate = self.events()
        duplicate[1]["path"][1] = 2
        overloaded = self.events()
        overloaded[1]["path"] = [0, *range(1, 51), 0]
        wrong = self.events()
        wrong[1]["objective"] += 1
        for events in (partial, duplicate, overloaded, wrong):
            with self.subTest(events=events), self.assertRaises(ValueError):
                self.read(events)


if __name__ == "__main__":
    unittest.main()
