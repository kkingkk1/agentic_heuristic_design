import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import p1_mkp_aco_evaluate as supervisor


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "problems" / "mkp_aco"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source_aco = load_module("mkp_source_aco", TASK_DIR / "aco.py")
with patch.dict(sys.modules, {"aco": source_aco}):
    task = load_module("p1_mkp_eval", TASK_DIR / "eval.py")
generator = load_module("p1_mkp_data", TASK_DIR / "gen_p1_train.py")


class TestMKP(unittest.TestCase):
    def test_train_data_matches_fixed_upstream_blob(self):
        self.assertEqual(generator.build_dataset(), task.DATASET.read_bytes())
        with np.load(task.DATASET, allow_pickle=False) as data:
            self.assertEqual(data["prizes"].shape, (5, 100))
            self.assertEqual(data["weights"].shape, (5, 100, 5))
            self.assertTrue((data["weights"] >= 0).all())

    def test_constraint_and_duplicate_checks(self):
        prize = np.array([2.0, 3.0, 4.0])
        weight = np.array([[0.6, 0.1], [0.6, 0.1], [0.1, 0.2]])
        self.assertEqual(task.check_solution([0, 2, 3, 3], prize, weight), 6.0)
        for solution in ([0, 0], [0, 1], [3, 1], [4], [-1]):
            with self.subTest(solution=solution), self.assertRaises(ValueError):
                task.check_solution(solution, prize, weight)

    def test_heuristic_validation_and_input_copies(self):
        prize = np.array([1.0, 2.0, 3.0])
        weight = np.ones((3, 2)) / 3

        def heuristic(p, w):
            p[0] = 99
            w[0, 0] = 99
            return np.array([-1.0, 0.0, 2.0])

        actual = task.checked_heuristic(heuristic, prize, weight)
        np.testing.assert_allclose(actual, [1e-9, 1e-9, 2.0 + 1e-9])
        self.assertEqual(prize[0], 1.0)
        self.assertAlmostEqual(weight[0, 0], 1 / 3)
        for value in ([1, 1, 1], np.ones((3, 1)), np.array([1.0, np.nan, 1.0])):
            with self.subTest(value=value), self.assertRaises(ValueError):
                task.checked_heuristic(lambda _p, _w: value, prize, weight)

    def test_50_iterations_match_fixed_solver(self):
        prize = np.array([0.7, 0.5, 0.4, 0.2, 0.6, 0.3])
        weight = np.array([[0.3, 0.4], [0.4, 0.3], [0.2, 0.4],
                           [0.3, 0.2], [0.5, 0.4], [0.2, 0.3]])
        heuristic = lambda p, w: p / w.sum(axis=1)
        torch.manual_seed(8710)
        actual, solution, iterations = task.solve(prize, weight, heuristic)
        scores = heuristic(prize, weight) + 1e-9
        torch.manual_seed(8710)
        original, _ = task.ACO(torch.from_numpy(prize), torch.from_numpy(weight),
                               torch.from_numpy(scores), 10).run(50)
        self.assertEqual(actual, float(original.item()))
        self.assertAlmostEqual(actual, task.check_solution(solution, prize, weight))
        self.assertEqual(iterations, 50)

    def test_float64_capacity_accumulation_rejects_boundary_overrun(self):
        prize = np.array([1.0, 1.0])
        weight = np.array([[0.499999982], [0.500000023]])
        solver = task.CheckedACO(torch.from_numpy(prize), torch.from_numpy(weight), torch.ones(2, dtype=torch.float64), 1)

        def choose(_solver, mask, _dummy_mask):
            return torch.tensor([0 if mask[0, 0] else 1 if mask[0, 1] else 2])

        with patch.object(source_aco.ACO, "pick_item", choose):
            solution = solver.gen_sol().T.tolist()[0]
        self.assertEqual(solution, [0])
        self.assertEqual(task.check_solution(solution, prize, weight), 1.0)

    def test_pheromone_and_dummy_heuristic_follow_input_float64(self):
        prize = torch.tensor([1.0, 2.0], dtype=torch.float64)
        weight = torch.tensor([[0.3], [0.4]], dtype=torch.float64)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            solver = task.CheckedACO(prize, weight, torch.ones(2, dtype=torch.float64), 1)
        finally:
            torch.set_default_dtype(previous_dtype)
        self.assertEqual(solver.pheromone.dtype, torch.float64)
        self.assertEqual(solver.heuristic[-1].item(), 1e-8)


class TestCompletion(unittest.TestCase):
    @staticmethod
    def events():
        with np.load(task.DATASET, allow_pickle=False) as data:
            prizes = data["prizes"].copy()
        start = {"event": "start", "seed_base": 20261103, "n_ants": 10,
                 "n_iterations": 50, "n_items": 100, "n_dims": 5}
        instances = [{"event": "instance", "index": i, "seed": 20261103 + i,
                      "iterations": 50, "objective": float(prizes[i, i]), "solution": [i, 100]}
                     for i in range(5)]
        average = sum(x["objective"] for x in instances) / 5
        return [start, *instances, {"event": "complete", "instance_count": 5,
                                   "average": average, "fitness": average}]

    @staticmethod
    def read(events):
        text = "\n".join(supervisor.EVENT_PREFIX + json.dumps(event) for event in events)
        with patch.object(Path, "read_text", return_value=text):
            return supervisor.read_complete(Path("unused"))

    def test_valid_result_has_positive_fitness(self):
        result = self.read(self.events())
        self.assertGreater(result["average"], 0)
        self.assertEqual(result["fitness"], result["average"])

    def test_partial_duplicate_infeasible_and_wrong_objective_fail(self):
        partial = self.events()[:-1]
        duplicate = self.events()
        duplicate[1]["solution"] = [0, 0]
        infeasible = self.events()
        infeasible[1]["solution"] = list(range(100))
        wrong = self.events()
        wrong[1]["objective"] += 1
        for events in (partial, duplicate, infeasible, wrong):
            with self.subTest(events=events), self.assertRaises(ValueError):
                self.read(events)


if __name__ == "__main__":
    unittest.main()
