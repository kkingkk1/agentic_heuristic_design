import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import p1_op_aco_evaluate as supervisor


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "problems" / "op_aco"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source_aco = load_module("op_source_aco", TASK_DIR / "aco.py")
with patch.dict(sys.modules, {"aco": source_aco}):
    task = load_module("p1_op_eval", TASK_DIR / "eval.py")
generator = load_module("p1_op_data", TASK_DIR / "gen_p1_train.py")


class TestOP(unittest.TestCase):
    def test_training_data_matches_fixed_upstream_blob(self):
        self.assertEqual(generator.build_dataset(), task.DATASET.read_bytes())
        with np.load(task.DATASET, allow_pickle=False) as data:
            self.assertEqual(data["coordinates"].shape, (5, 50, 2))

    def test_prize_and_closed_path_match_source_protocol(self):
        positions = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        prizes, distances = task.instance(positions)
        np.testing.assert_allclose(prizes.numpy(), [0.01, 0.5, 1.0])
        self.assertEqual(distances[0, 0].item(), 1e9)
        reward, length = task.check_solution([0, 1, 2, 3, 3], positions, prizes.numpy())
        self.assertAlmostEqual(reward, 1.51)
        self.assertAlmostEqual(length, 2.0)

    def test_invalid_routes_rejected(self):
        positions = np.array([[0.0, 0.0], [0.5, 0.0], [3.0, 0.0]])
        prizes = np.array([0.01, 0.5, 1.0])
        for solution in ([1, 0], [0, 1, 1], [0, 3, 1], [0, 2], [0, 4]):
            with self.subTest(solution=solution), self.assertRaises(ValueError):
                task.check_solution(solution, positions, prizes)

    def test_heuristic_checks_and_copies(self):
        prizes = np.array([0.1, 0.2])
        distances = np.ones((2, 2))

        def heuristic(p, d, limit):
            self.assertEqual(limit, 3.0)
            p[0] = 99
            d[0, 0] = 99
            return np.array([[0.0, -1.0], [2.0, 1.0]])

        actual = task.checked_heuristic(heuristic, prizes, distances)
        np.testing.assert_allclose(actual, [[1e-9, 1e-9], [2.0 + 1e-9, 1.0 + 1e-9]])
        self.assertEqual(prizes[0], 0.1)
        self.assertEqual(distances[0, 0], 1.0)
        with self.assertRaises(ValueError):
            task.checked_heuristic(lambda *_args: np.full((2, 2), np.nan), prizes, distances)

    def test_100_iterations_match_fixed_solver(self):
        positions = np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [0.2, 0.2], [0.4, 0.2]])
        heuristic = lambda p, d, _m: p[np.newaxis, :] / d
        torch.manual_seed(8120)
        actual, solution, _distance, iterations = task.solve(positions, heuristic)
        prizes, distances = task.instance(positions)
        scores = heuristic(prizes.numpy(), distances.numpy(), 3.0) + 1e-9
        torch.manual_seed(8120)
        original, _ = task.ACO(prizes, distances, 3.0, torch.from_numpy(scores), 30).run(100)
        self.assertEqual(actual, float(original.item()))
        self.assertAlmostEqual(actual, task.check_solution(solution, positions, prizes.numpy())[0])
        self.assertEqual(iterations, 100)

    def test_float64_distance_accumulation_rejects_boundary_overrun(self):
        positions = np.array([[0.0, 0.0], [0.499999982, 0.0], [0.5, 0.0001]])
        full_length = sum(math.hypot(*(positions[a] - positions[b])) for a, b in ((0, 1), (1, 2), (2, 0)))
        limit = full_length - 5e-9
        prizes, distances = task.instance(positions)
        scores = torch.ones_like(distances)
        solver = task.CheckedACO(prizes, distances, scores, positions, max_len=limit)

        def choose(_solver, mask, _current):
            return torch.tensor([1 if mask[0, 1] else 2 if mask[0, 2] else 3] * _solver.n_ants)

        with patch.object(source_aco.ACO, "pick_node", choose):
            solutions = solver.gen_sol().T.tolist()
        self.assertTrue(all(2 not in solution for solution in solutions))
        self.assertTrue(all(task.check_solution(solution, positions, prizes.numpy(), limit)[1] <= limit
                            for solution in solutions))

    def test_float64_solver_preserves_real_distances_and_zeroes_dummy(self):
        positions = np.array([[0.0, 0.0], [1e-10, 0.0], [0.4, 0.1]])
        prizes, distances = task.instance(positions)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            solver = task.CheckedACO(prizes, distances, torch.ones_like(distances), positions)
        finally:
            torch.set_default_dtype(previous_dtype)
        self.assertEqual(solver.distances.dtype, torch.float64)
        self.assertTrue(torch.equal(solver.distances[:-1, :-1], distances))
        self.assertEqual(torch.count_nonzero(solver.distances[:, -1]).item(), 0)


class TestCompletion(unittest.TestCase):
    @staticmethod
    def events():
        start = {"event": "start", "seed_base": 20261103, "n_ants": 30,
                 "n_iterations": 100, "n_nodes": 50, "max_len": 3.0}
        instances = [{"event": "instance", "index": i, "seed": 20261103 + i,
                      "iterations": 100, "objective": 0.01, "solution": [0], "distance": 0.0}
                     for i in range(5)]
        return [start, *instances, {"event": "complete", "instance_count": 5,
                                   "average": 0.01, "fitness": 0.01}]

    @staticmethod
    def read(events):
        text = "\n".join(supervisor.EVENT_PREFIX + json.dumps(event) for event in events)
        with patch.object(Path, "read_text", return_value=text):
            return supervisor.read_complete(Path("unused"))

    def test_complete_result_has_positive_fitness(self):
        result = self.read(self.events())
        self.assertEqual(result["fitness"], result["average"])

    def test_partial_duplicate_invalid_length_and_wrong_reward_fail(self):
        partial = self.events()[:-1]
        duplicate = self.events()
        duplicate[1]["solution"] = [0, 1, 1]
        invalid_length = self.events()
        invalid_length[1]["distance"] = 100.0
        wrong = self.events()
        wrong[1]["objective"] += 1
        for events in (partial, duplicate, invalid_length, wrong):
            with self.subTest(events=events), self.assertRaises(ValueError):
                self.read(events)


if __name__ == "__main__":
    unittest.main()
