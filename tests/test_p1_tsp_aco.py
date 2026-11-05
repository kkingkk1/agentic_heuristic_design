import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial import distance_matrix


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "problems" / "tsp_aco"
sys.path.insert(0, str(TASK_DIR))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


task = load_module("p1_tsp_eval", TASK_DIR / "eval.py")
generator = load_module("p1_tsp_data", TASK_DIR / "gen_p1_test.py")
source_generator = load_module("p0_tsp_data", TASK_DIR / "gen_inst.py")


class TestDatasets(unittest.TestCase):
    def test_match_source_sequence_and_keep_training_disjoint(self):
        train, generated = generator.datasets()
        source = {}

        def capture(path, values):
            source[Path(path).name] = values.copy()

        random_state = np.random.get_state()
        try:
            with patch.object(source_generator.os, "makedirs"), \
                 patch.object(source_generator.np, "save", side_effect=capture):
                source_generator.generate_datasets()
        finally:
            np.random.set_state(random_state)
        np.testing.assert_array_equal(train, source["train50_dataset.npy"])
        np.testing.assert_array_equal(train, np.load(task.DATASET, allow_pickle=False))
        for size in task.TEST_SIZES:
            with self.subTest(size=size):
                _, disk = task.test_dataset(size)
                np.testing.assert_array_equal(disk, generated[size])
                if size in (50, 100):
                    np.testing.assert_array_equal(disk, source[f"test{size}_dataset.npy"])
                self.assertEqual(disk.shape, (64, size, 2))
                if size == 50:
                    for instance in train:
                        self.assertFalse(any(np.array_equal(instance, sample) for sample in disk))

    def test_hash_mismatch_is_rejected(self):
        with patch.dict(task.TEST_DATASET_SHA256, {50: "0" * 64}):
            with self.assertRaisesRegex(ValueError, "哈希不匹配"):
                task.test_dataset(50)

    def test_generator_refuses_to_overwrite_existing_datasets(self):
        paths = [task.test_dataset(size)[0] for size in task.TEST_SIZES]
        before = [path.read_bytes() for path in paths]
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(TASK_DIR / "gen_p1_test.py")],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("不覆盖", result.stderr)
        self.assertEqual(before, [path.read_bytes() for path in paths])


class TestEvaluation(unittest.TestCase):
    def test_black_box_edge_mapping_and_source_clipping(self):
        distances = np.array([[1.0, 2.0], [3.0, 4.0]])
        raw = np.array([0.25, -1.0, 0.0, 2.0], dtype=np.float32)

        def heuristic(features):
            np.testing.assert_array_equal(features, distances.reshape(-1, 1))
            features[0, 0] = 99
            return raw.copy()

        actual = task.checked_heuristic(heuristic, distances, black_box=True)
        expected = np.where(raw < 1e-9, 1e-9, raw).reshape(2, 2)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(distances[0, 0], 1.0)
        np.testing.assert_array_equal(
            task.checked_heuristic(lambda _x: raw.reshape(-1, 1), distances, black_box=True), expected
        )

    def test_black_box_rejects_invalid_shape_and_nonfinite_values(self):
        distances = np.ones((2, 2))
        for value in (np.ones((2, 2)), np.ones(3), np.array([1.0, 1.0, np.nan, 1.0]), [1, 1, 1, 1]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                task.checked_heuristic(lambda _x: value, distances, black_box=True)

    def test_black_box_matches_source_aco_on_independent_square(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        distances = distance_matrix(coords, coords)
        np.fill_diagonal(distances, 1.0)
        heuristic = lambda edges: 1.0 / edges[:, 0]
        torch.manual_seed(6842)
        objective, tour, iterations = task.solve(coords, heuristic, 200, black_box=True)
        source = heuristic(distances.reshape(-1, 1).copy())
        source = np.where(source < 1e-9, 1e-9, source).reshape(4, 4)
        torch.manual_seed(6842)
        original = task.ACO(distances, source, n_ants=30).run(200).item()
        self.assertEqual(objective, original)
        self.assertEqual(iterations, 200)
        self.assertEqual(sorted(tour), [0, 1, 2, 3])

    def test_500_iterations_agree_with_original_solver_on_independent_square(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        distances = distance_matrix(coords, coords)
        np.fill_diagonal(distances, 1.0)
        heuristic = lambda matrix: 1.0 / matrix
        torch.manual_seed(1927)
        objective, tour, iterations = task.solve(coords, heuristic, 500)
        torch.manual_seed(1927)
        original = task.ACO(distances, heuristic(distances) + 1e-9, n_ants=30).run(500).item()
        self.assertEqual(objective, original)
        self.assertAlmostEqual(objective, 4.0)
        self.assertEqual(sorted(tour), [0, 1, 2, 3])
        self.assertEqual(iterations, 500)

    def test_test_mode_records_all_instances_and_its_own_seed_range(self):
        synthetic = np.zeros((64, 50, 2))
        dataset = task.DATASET.parent / "test50_p1_dataset.npy"
        stream = io.StringIO()
        with patch.object(sys, "argv", ["eval.py", "50", "candidate", "test"]), \
             patch.object(task, "test_dataset", return_value=(dataset, synthetic)), \
             patch.object(task, "candidate_hash", return_value="a" * 64), \
             patch.object(task, "load_heuristic", return_value=object()), \
             patch.object(task, "solve", return_value=(5.0, list(range(50)), 500)) as solve, \
             contextlib.redirect_stdout(stream):
            task.main()
        events = [json.loads(line.removeprefix("P1_JSON "))
                  for line in stream.getvalue().splitlines() if line.startswith("P1_JSON ")]
        self.assertEqual([event["event"] for event in events], ["start"] + ["instance"] * 64 + ["complete"])
        self.assertEqual(events[0]["seed_base"], task.TEST_SEED_BASE + 5000)
        self.assertEqual(events[0]["n_iterations"], 500)
        self.assertEqual(events[0]["candidate_sha256"], "a" * 64)
        self.assertEqual(events[-1]["instance_count"], 64)
        self.assertEqual(events[-1]["average"], 5.0)
        self.assertEqual(solve.call_count, 64)

    def test_changed_candidate_never_emits_completion(self):
        dataset = task.DATASET.parent / "test50_p1_dataset.npy"
        stream = io.StringIO()
        with patch.object(sys, "argv", ["eval.py", "50", "candidate", "test"]), \
             patch.object(task, "test_dataset", return_value=(dataset, np.zeros((64, 50, 2)))), \
             patch.object(task, "candidate_hash", side_effect=["a" * 64, "b" * 64]), \
             patch.object(task, "load_heuristic", return_value=object()), \
             patch.object(task, "solve", return_value=(5.0, list(range(50)), 500)), \
             contextlib.redirect_stdout(stream):
            with self.assertRaisesRegex(ValueError, "候选代码发生变化"):
                task.main()
        self.assertNotIn('"event": "complete"', stream.getvalue())


if __name__ == "__main__":
    unittest.main()
