import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from scripts import p1_bpp_aco_evaluate as supervisor


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "problems" / "bpp_offline_aco"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source_aco = load_module("bpp_source_aco", TASK_DIR / "aco.py")
with patch.dict(sys.modules, {"aco": source_aco}):
    task = load_module("p1_bpp_eval", TASK_DIR / "eval.py")
generator = load_module("p1_bpp_data", TASK_DIR / "gen_p1_train.py")


class TestBPP(unittest.TestCase):
    def test_training_data_matches_fixed_upstream_blob(self):
        self.assertEqual(generator.build_dataset(), task.DATASET.read_bytes())
        with np.load(task.DATASET, allow_pickle=False) as data:
            self.assertEqual(data["demands"].shape, (5, 500))
            self.assertTrue(((data["demands"] >= 20) & (data["demands"] <= 100)).all())

    def test_capacity_and_shape_checks(self):
        demand = np.array([50, 80, 90, 40])
        self.assertEqual(task.check_solution(np.array([0, 0, 1, 1]), demand), 2)
        for path in (np.array([0, 0, 0, 1]), np.array([0, 1, -1, 2]), np.array([0, 1, 2]), [0, 1, 2, 3]):
            with self.subTest(path=path), self.assertRaises(ValueError):
                task.check_solution(path, demand)

    def test_heuristic_checks_and_input_copy(self):
        demand = np.array([20, 30])

        def heuristic(values, capacity):
            self.assertEqual(capacity, 150)
            values[0] = 99
            return np.array([[1.0, -1.0], [2.0, 3.0]])

        result = task.checked_heuristic(heuristic, demand)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(demand[0], 20)
        for value in (np.ones(2), np.full((2, 2), np.nan), -np.ones((2, 2))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                task.checked_heuristic(lambda _d, _c: value, demand)

    def test_100_iterations_match_original_source_under_numpy_alias(self):
        demand = np.array([20, 50, 70, 80, 100, 30])
        heuristic = lambda d, _capacity: np.tile(d / d.max(), (len(d), 1))
        source = (TASK_DIR / "aco.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("FloatArray = npt.NDArray[np.float64]"), 1)
        original = types.ModuleType("bpp_unmodified_source")
        with patch.object(np, "float_", np.float64, create=True):
            exec(compile(source.replace("FloatArray = npt.NDArray[np.float64]",
                                       "FloatArray = npt.NDArray[np.float_]"), "upstream_aco.py", "exec"), original.__dict__)
        np.random.seed(7643)
        original.uniform_generator = original.uniform_number_generator()
        old_obj, old_solution = original.ACO(demand.copy(), heuristic(demand, 150), 150, n_ants=20).run(100)
        np.random.seed(7643)
        task.source_aco.uniform_generator = task.source_aco.uniform_number_generator()
        actual, solution, iterations = task.solve(demand, heuristic)
        self.assertEqual(actual, old_obj)
        np.testing.assert_array_equal(solution, old_solution)
        self.assertEqual(iterations, 100)


class TestCompletion(unittest.TestCase):
    @staticmethod
    def events():
        start = {"event": "start", "seed_base": 20261103, "n_ants": 20,
                 "n_iterations": 100, "n_items": 500, "capacity": 150}
        instances = [{"event": "instance", "index": i, "seed": 20261103 + i,
                      "iterations": 100, "objective": 500, "solution": list(range(500))}
                     for i in range(5)]
        return [start, *instances, {"event": "complete", "instance_count": 5,
                                   "average": 500.0, "fitness": -500.0}]

    @staticmethod
    def read(events):
        text = "\n".join(supervisor.EVENT_PREFIX + json.dumps(event) for event in events)
        with patch.object(Path, "read_text", return_value=text):
            return supervisor.read_complete(Path("unused"))

    def test_complete_result_has_negative_fitness(self):
        result = self.read(self.events())
        self.assertEqual(result["fitness"], -result["average"])

    def test_partial_infeasible_and_wrong_box_count_fail(self):
        partial = self.events()[:-1]
        infeasible = self.events()
        infeasible[1]["solution"] = [0] * 500
        wrong = self.events()
        wrong[1]["objective"] = 499
        for events in (partial, infeasible, wrong):
            with self.subTest(events=events), self.assertRaises(ValueError):
                self.read(events)


if __name__ == "__main__":
    unittest.main()
