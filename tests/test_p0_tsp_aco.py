import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial import distance_matrix
import torch

from scripts import p0_tsp_aco_evaluate as supervisor


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "problems" / "tsp_aco"
sys.path.insert(0, str(TASK_DIR))
spec = importlib.util.spec_from_file_location("p0_tsp_eval", TASK_DIR / "eval.py")
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


class HeuristicValidationTests(unittest.TestCase):
    def setUp(self):
        self.distances = np.array([[1.0, 2.0], [2.0, 1.0]])

    def test_rejects_wrong_shape_type_and_nonfinite_values(self):
        for value in (
            [[1.0, 1.0], [1.0, 1.0]],
            np.ones((2, 1)),
            np.array([[1.0, np.nan], [1.0, 1.0]]),
            np.array([[1.0, np.inf], [1.0, 1.0]]),
            np.ones((2, 2), dtype=complex),
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                task.checked_heuristic(lambda _dist: value, self.distances)

    def test_finite_negative_scores_follow_upstream_clipping_and_input_is_copied(self):
        def heuristic(distances):
            distances[0, 0] = 99
            return np.array([[-2.0, 0.0], [3.0, 1.0]])

        actual = task.checked_heuristic(heuristic, self.distances)
        np.testing.assert_allclose(actual, [[1e-9, 1e-9], [3.0 + 1e-9, 1.0 + 1e-9]])
        self.assertEqual(self.distances[0, 0], 1.0)

    def test_path_check_rejects_duplicate_city_for_any_ant(self):
        aco = task.CheckedACO(np.ones((3, 3)), np.ones((3, 3)), n_ants=2)
        illegal = torch.tensor([[0, 0], [1, 1], [2, 1]])
        with patch.object(task.ACO, "gen_path", return_value=illegal), self.assertRaisesRegex(ValueError, "城市排列"):
            aco.gen_path()
        self.assertEqual(aco.checked_iterations, 0)
        valid = torch.tensor([[0, 2], [1, 0], [2, 1]])
        with patch.object(task.ACO, "gen_path", return_value=valid):
            self.assertTrue(torch.equal(aco.gen_path(), valid))
        self.assertEqual(aco.checked_iterations, 1)

    def test_rejects_nonfinite_path_cost(self):
        aco = task.CheckedACO(np.ones((3, 3)), np.ones((3, 3)), n_ants=2)
        with patch.object(task.ACO, "gen_path_costs", return_value=torch.tensor([1.0, float("nan")])):
            with self.assertRaisesRegex(ValueError, "代价无效"):
                aco.gen_path_costs(torch.zeros((3, 2), dtype=torch.int64))

    def test_reference_preserves_original_aco_result_for_full_budget(self):
        dataset = np.load(task.DATASET, allow_pickle=False)
        np.testing.assert_array_equal(dataset, np.random.RandomState(1234).rand(5, 50, 2))
        node_pos = dataset[0]
        distances = distance_matrix(node_pos, node_pos)
        np.fill_diagonal(distances, 1)
        heuristic = lambda matrix: 1 / matrix
        torch.manual_seed(task.SEED_BASE)
        actual, tour, iterations = task.solve(node_pos, heuristic)
        torch.manual_seed(task.SEED_BASE)
        original = task.ACO(distances, heuristic(distances) + 1e-9, n_ants=30).run(200).item()
        self.assertEqual(iterations, 200)
        self.assertEqual(actual, original)
        self.assertEqual(sorted(tour), list(range(50)))


class CompletionValidationTests(unittest.TestCase):
    @staticmethod
    def events():
        start = {"event": "start", "seed_base": 20261103, "n_ants": 30, "n_iterations": 200}
        instances = [
            {"event": "instance", "index": index, "seed": 20261103 + index,
             "iterations": 200, "objective": float(index + 1), "tour": list(range(50))}
            for index in range(5)
        ]
        return [start, *instances, {"event": "complete", "instance_count": 5, "average": 3.0}]

    @staticmethod
    def read(events):
        text = "\n".join(supervisor.EVENT_PREFIX + json.dumps(item) for item in events)
        with patch.object(Path, "read_text", return_value=text):
            return supervisor.read_complete(Path("unused"))

    def test_complete_minimization_has_negative_fitness(self):
        self.assertEqual(self.read(self.events())["fitness"], -3.0)

    def test_incomplete_or_inconsistent_results_never_receive_fitness(self):
        partial = self.events()[:-1]
        wrong_average = self.events()
        wrong_average[-1]["average"] = 4.0
        wrong_tour = self.events()
        wrong_tour[1]["tour"][0] = 1
        wrong_iterations = self.events()
        wrong_iterations[1]["iterations"] = 199
        for events in (partial, wrong_average, wrong_tour, wrong_iterations):
            with self.subTest(events=events), self.assertRaises(ValueError):
                self.read(events)

    @unittest.skipUnless(sys.platform == "win32", "Windows taskkill 路径")
    def test_missing_taskkill_stops_main_process_and_reports_tree_failure(self):
        process = Mock(pid=12345)
        process.poll.return_value = None
        with patch.object(supervisor.subprocess, "run", side_effect=FileNotFoundError("taskkill")):
            result = supervisor.terminate_tree(process)
        process.kill.assert_called_once()
        process.wait.assert_called_once()
        self.assertIn("进程树终止失败", result["error"])


if __name__ == "__main__":
    unittest.main()
