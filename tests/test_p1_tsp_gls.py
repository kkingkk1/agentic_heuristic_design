import contextlib
from hashlib import sha1
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from baselines.gls_reference import tsp_gls as gls_reference
from scripts import p1_tsp_gls_evaluate as supervisor, p1_tsp_gls_lkh_train as lkh
from scripts import p1_test_evaluate as test_supervisor
from scripts.p1_task_contract import load_task_contract


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "problems" / "tsp_gls"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(TASK_DIR))
source_gls = importlib.import_module("gls")
source_lkh = importlib.import_module("lkh_reference")
with patch.dict(sys.modules, {"gls": source_gls}):
    task = load_module("p1_gls_eval", TASK_DIR / "eval.py")
generator = load_module("p1_gls_data", TASK_DIR / "gen_p1_data.py")


class TestGLS(unittest.TestCase):
    def test_source_diff_is_float64_adaptation_and_final_newline(self):
        code = (TASK_DIR / "gls.py").read_bytes()
        marker = "# P1 测试：".encode("utf-8")
        self.assertEqual(code.count(marker), 1)
        code = code.split(marker, 1)[0].rstrip(b"\n") + b"\n"
        replacement = b"FloatArray = npt.NDArray[np.float64]"
        original = b"FloatArray = npt.NDArray[np.float_]"
        self.assertEqual(code.count(replacement), 1)
        self.assertEqual(code.count(b".astype(np.float64)"), 4)
        self.assertNotIn(b"nb.float32", code)
        code = code.replace(b"nb.float64", b"nb.float32")
        code = code.replace(b".astype(np.float64)", b".astype(np.float32)")
        code = code.replace(replacement, original)
        self.assertTrue(code.endswith(b"\n"))
        code = code[:-1]  # 固定来源文件没有末尾换行。
        blob = sha1(b"blob " + str(len(code)).encode("ascii") + b"\0" + code).hexdigest()
        self.assertEqual(blob, "01b0dfc34b64754dbc56d37cb742730f724eddd3")

    def test_data_match_source_and_refuse_overwrite(self):
        data = generator.build_datasets()
        for name, content in data.items():
            with self.subTest(name=name):
                self.assertEqual(generator.git_blob(content), generator.SOURCE_BLOBS[name])
                self.assertEqual((generator.DATASET_DIR / name).read_bytes(), content)
        with self.assertRaisesRegex(FileExistsError, "不覆盖"):
            generator.main()

    def test_heuristic_validation_and_input_copy(self):
        distance = np.array([[1e-5, 2.0], [2.0, 1e-5]])

        def heuristic(values):
            values[0, 0] = 99
            return values

        scores = task.checked_heuristic(heuristic, distance)
        self.assertEqual(scores.dtype, np.float64)
        self.assertEqual(distance[0, 0], 1e-5)
        self.assertEqual(task.checked_heuristic(lambda _d: np.full((2, 2), 1e300), distance)[0, 0], 1e300)
        for invalid in (np.ones(2), np.full((2, 2), np.nan), np.full((2, 2), np.inf)):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, FloatingPointError)):
                task.checked_heuristic(lambda _d: invalid, distance)

    def test_small_compiled_solver_returns_valid_tour(self):
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        code = Path(gls_reference.__file__).read_text(encoding="utf-8")
        load_task_contract("tsp_gls").parse_proposal(f"<thought>reference</thought><code>{code}</code>")
        objective, tour, iterations = task.solve(positions, gls_reference.heuristics_v2, 1, 0)
        self.assertEqual(sorted(tour), [0, 1, 2, 3])
        self.assertEqual(iterations, 1)
        self.assertAlmostEqual(objective, 4.0)

    def test_float64_cost_preserves_small_distance_differences(self):
        distances = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.00000004],
                              [1.0, 1.00000004, 0.0]], dtype=np.float64)
        tour = np.array([0, 1, 2], dtype=np.uint16)
        self.assertAlmostEqual(source_gls._calculate_cost(distances, tour), 3.00000004, places=13)

    def test_rejects_float32_coordinates_before_solving(self):
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "float64 原始坐标"):
            task.solve(positions, gls_reference.heuristics_v2, 1, 0)

    def test_reference_tolerances_and_better_route(self):
        for reference in (0.1, 4.0):
            with self.subTest(reference=reference):
                threshold = source_lkh.stop_threshold(reference)
                self.assertTrue(source_lkh.reached_reference(reference * 0.9, reference))
                self.assertTrue(source_lkh.reached_reference(threshold, reference))
                self.assertFalse(source_lkh.reached_reference(math.nextafter(threshold, math.inf), reference))
        for invalid in (0, -1, math.inf, math.nan, True):
            with self.subTest(reference=invalid), self.assertRaises(ValueError):
                source_lkh.stop_threshold(invalid)

    def test_early_stop_after_initial_search_or_iteration_limit(self):
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        objective, tour, iterations = task.solve(positions, gls_reference.heuristics_v2, 3, 0, 4.0)
        self.assertEqual((objective, iterations), (4.0, 0))
        self.assertEqual(sorted(tour), [0, 1, 2, 3])
        objective, _, iterations = task.solve(positions, gls_reference.heuristics_v2, 3, 0, 3.0)
        self.assertEqual((objective, iterations), (4.0, 3))

    def test_early_stop_after_perturbation(self):
        positions = np.random.RandomState(3).random_sample((20, 2))
        threshold = (3.5415204193622825 + 3.4536722711007277) / 2
        objective, tour, iterations = task.solve(positions, gls_reference.heuristics_v2, 10, 2, threshold)
        self.assertEqual(iterations, 1)
        self.assertLess(objective, threshold)
        self.assertEqual(sorted(tour), list(range(20)))

    def test_unreached_reference_matches_original_solver(self):
        from hashlib import sha256
        from scipy.spatial import distance_matrix

        expected = {20: "156d57651b75eaeef5c240e18830d9f5bbc531a82b025364618972c669f1e4dd",
                    50: "413312ae6cb1a358380d9f93b61eda4170f85beb6a4b071d30aa64f8bc671abb"}
        for size, expected_hash in expected.items():
            with self.subTest(size=size):
                points = np.random.RandomState(77 + size).random_sample((size, 2))
                distances = distance_matrix(points, points) + np.eye(size) * 1e-5
                original = source_gls.guided_local_search(distances, distances, 2, 3)
                adapted, iterations = source_gls.guided_local_search_until_reference(distances, distances, 2, 3, 0.0)
                self.assertTrue(np.array_equal(original, adapted))
                self.assertEqual(sha256(original.tobytes()).hexdigest(), expected_hash)
                self.assertEqual(iterations, 3)

    def test_reference_identity_and_dataset_mismatch(self):
        reference = source_lkh.load_reference(20, task.TEST_DATASET_SHA256[20])
        self.assertEqual(len(reference["objectives"]), 64)
        self.assertEqual(reference["mean"], 3.8362853943492015)
        with self.assertRaisesRegex(ValueError, "测试数据"):
            source_lkh.load_reference(20, "0" * 64)
        with patch.object(source_lkh, "REFERENCE_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "哈希"):
            source_lkh.load_reference(20, task.TEST_DATASET_SHA256[20])

    def test_candidate_test_parameters_without_running_candidates(self):
        for size, (moves, iterations) in task.TEST_PARAMETERS.items():
            events = []

            def fake_solve(points, heuristic, n_iterations, perturbation_moves, reference):
                self.assertEqual((len(points), n_iterations, perturbation_moves), (size, iterations, moves))
                self.assertGreater(reference, 0)
                return 1.0, list(range(size)), n_iterations

            with self.subTest(size=size), contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", ["eval.py", str(size), "unused", "test"]))
                stack.enter_context(patch.object(task, "candidate_hash", return_value="a" * 64))
                stack.enter_context(patch.object(task, "load_heuristic", return_value=object()))
                stack.enter_context(patch.object(task, "solve", side_effect=fake_solve))
                stack.enter_context(patch.object(task, "emit", side_effect=events.append))
                stack.enter_context(patch.object(task.random, "seed"))
                stack.enter_context(patch.object(np.random, "seed"))
                task.main()
            self.assertEqual(len(events), 66)
            self.assertEqual(events[0]["candidate_sha256"], "a" * 64)
            self.assertEqual(events[0]["n_iterations"], iterations)
            self.assertEqual(events[0]["perturbation_moves"], moves)
            self.assertEqual(events[1]["stop_reason"], "reference_reached")
            self.assertIn("lkh_gap_percent", events[-1])
            self.assertEqual(events[-1]["instance_count"], 64)

    def test_synthetic_test_entry_and_supervisor_end_to_end(self):
        angles = np.arange(20, dtype=np.float64) * (2.0 * np.pi / 20)
        coordinates = np.column_stack((0.5 + 0.25 * np.cos(angles), 0.5 + 0.25 * np.sin(angles)))
        instances = np.repeat(coordinates[np.newaxis, :, :], 64, axis=0)
        perimeter = 20 * 0.5 * math.sin(math.pi / 20)
        reference = {"objectives": [perimeter] * 64, "mean": perimeter, "sha256": "b" * 64}
        events = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["eval.py", "20", "unused", "test"]))
            stack.enter_context(patch.object(task, "candidate_hash", return_value="a" * 64))
            stack.enter_context(patch.object(task, "load_heuristic", return_value=gls_reference.heuristics_v2))
            stack.enter_context(patch.object(task, "load_reference", return_value=reference))
            stack.enter_context(patch.object(task, "reference_digest", return_value=reference["sha256"]))
            stack.enter_context(patch.object(task.np, "load", return_value=instances))
            stack.enter_context(patch.object(task, "emit", side_effect=events.append))
            task.main()
        self.assertEqual(len(events), 66)
        self.assertTrue(all(row["iterations"] == 0 for row in events[1:-1]))
        entry = {"size": 20, "evaluation_seed_base": task.TEST_SEED_BASE + 2000,
                 "iterations": task.TEST_PARAMETERS[20][1], "perturbation_moves": task.TEST_PARAMETERS[20][0]}
        manifest = {"task": "tsp_gls", "candidate_sha256": "a" * 64}
        text = "\n".join(test_supervisor.PREFIX + json.dumps(event) for event in events)
        with patch.object(Path, "read_text", return_value=text):
            result = test_supervisor.read_complete(Path("fixture"), entry, manifest, "min",
                                                   task.DATASET.parent / "test20_p1_dataset.npy", None, reference)
        self.assertEqual(result["stopping_reasons"].count("reference_reached"), 64)
        self.assertAlmostEqual(result["lkh_gap_percent"], 0.0, places=10)


class TestCompletion(unittest.TestCase):
    @staticmethod
    def events():
        points = np.load(task.DATASET, allow_pickle=False)
        tour = list(range(200))
        objectives = [math.fsum(math.hypot(*(instance[a] - instance[b]))
                                for a, b in zip(tour, tour[1:] + tour[:1])) for instance in points]
        start = {"event": "start", "seed_base": 20261103, "n_nodes": 200,
                 "n_iterations": 1200, "perturbation_moves": 30}
        instances = [{"event": "instance", "index": i, "seed": 20261103 + i, "iterations": 1200,
                      "objective": value, "tour": tour.copy()} for i, value in enumerate(objectives)]
        average = math.fsum(objectives) / 10
        return [start, *instances, {"event": "complete", "instance_count": 10,
                                   "average": average, "fitness": -average}]

    @staticmethod
    def read(events):
        text = "\n".join(supervisor.EVENT_PREFIX + json.dumps(event) for event in events)
        with patch.object(Path, "read_text", return_value=text):
            return supervisor.read_complete(Path("unused"))

    def test_complete_result_and_invalid_tours(self):
        events = self.events()
        result = self.read(events)
        self.assertEqual(result["fitness"], -result["average"])
        bad = self.events()
        bad[1]["tour"][0] = bad[1]["tour"][1]
        for value in (events[:-1], bad):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.read(value)


class TestLKHReference(unittest.TestCase):
    def test_small_instance_and_source_scale(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        result = lkh.solve_instance((0, points))
        self.assertEqual(result["index"], 0)
        self.assertEqual(result["tour"][0], result["tour"][-1])
        self.assertEqual(sorted(result["tour"][:-1]), [0, 1, 2, 3])
        self.assertAlmostEqual(result["objective"], 4.0)
        self.assertEqual((lkh.SCALE, lkh.RUNS), (1_000_000, 10))

    def test_completion_never_claims_optimality(self):
        points = np.load(task.DATASET, allow_pickle=False)
        tour = list(range(200)) + [0]
        values = [math.fsum(math.hypot(*(instance[a] - instance[b])) for a, b in zip(tour, tour[1:]))
                  for instance in points]
        events = [{"event": "start", "instances": 10, "nodes": 200, "scale": lkh.SCALE,
                   "runs": lkh.RUNS, "elkai_version": "2.0.1", "optimality_proven": False}]
        events += [{"event": "instance", "index": i, "tour": tour, "objective": value}
                   for i, value in enumerate(values)]
        events.append({"event": "complete", "instances": 10, "average": math.fsum(values) / 10,
                       "optimality_proven": False})
        text = "\n".join(lkh.PREFIX + json.dumps(event) for event in events)
        with patch.object(Path, "read_text", return_value=text):
            self.assertIs(lkh.read_complete(Path("unused"))["optimality_proven"], False)
        events[-1]["optimality_proven"] = True
        text = "\n".join(lkh.PREFIX + json.dumps(event) for event in events)
        with patch.object(Path, "read_text", return_value=text), self.assertRaises(ValueError):
            lkh.read_complete(Path("unused"))


if __name__ == "__main__":
    unittest.main()
