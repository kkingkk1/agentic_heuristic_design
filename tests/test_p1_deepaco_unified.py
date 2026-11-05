import importlib.util
from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.deepaco.reference import DeepACOReference, load_source
from scripts import p1_deepaco_test_evaluate as supervisor
from tests.test_p1_bpp_aco import task as bpp_task
from tests.test_p1_op_aco import task as op_task


HAS_PYG = importlib.util.find_spec("torch_geometric") is not None
HAS_NUMBA = importlib.util.find_spec("numba") is not None


def run_synthetic_training_main(task, model, data, name, *, k_sparse=None):
    buffer = BytesIO()
    np.savez(buffer, **{name: data})
    payload = buffer.getvalue()
    fixture_path = Path("in_memory_deepaco_fixture.npz")
    native_read, native_load = Path.read_bytes, np.load

    def read(path):
        return payload if path == fixture_path else native_read(path)

    def load(path, *args, **kwargs):
        return native_load(BytesIO(payload) if path == fixture_path else path, *args, **kwargs)

    events = []
    size = "50" if name == "coordinates" else "500"
    with patch.object(task, "DATASET", fixture_path), patch.object(task, "DATASET_SHA256", sha256(payload).hexdigest()), \
         patch.object(task, "N_INSTANCES", 1), patch.object(task, "N_ITERATIONS", 1), \
         patch.object(Path, "read_bytes", read), patch.object(np, "load", side_effect=load), \
         patch.object(task, "emit", side_effect=events.append), \
         patch.object(sys, "argv", ["eval.py", size, "unused", "train"]):
        if name == "coordinates":
            task.main(deepaco_model=model, k_sparse=k_sparse)
        else:
            task.main(deepaco_model=model)
    return events


class TestBPPAssignment(unittest.TestCase):
    def test_bin_count_and_internal_fitness_are_distinct(self):
        demand = np.array([50, 25] * 4)
        four = [0, 1, 2, 0, 3, 4, 0, 5, 6, 0, 7, 8, 0]
        five = [0, 1, 2, 3, 4, 0, 5, 0, 7, 0, 6, 0, 8, 0]
        a = bpp_task.deepaco_assignment(four, demand)
        b = bpp_task.deepaco_assignment(five, demand)
        self.assertEqual(bpp_task.check_solution(a, demand), 4)
        self.assertEqual(bpp_task.check_solution(b, demand), 5)
        def fitness(assignment):
            loads = np.bincount(assignment, weights=demand)
            return np.mean((loads / bpp_task.CAPACITY) ** 2)
        self.assertGreater(fitness(b), fitness(a))

    def test_invalid_or_incomplete_sequences_are_rejected(self):
        demand = np.array([50, 25])
        for path in ([0, 1, 0], [0, 1, 1, 2, 0], [0, 1, 2], [0, 1, 3, 0], [0, 1, 2, True]):
            with self.subTest(path=path), self.assertRaises(ValueError):
                bpp_task.deepaco_assignment(path, demand)


@unittest.skipUnless(HAS_PYG, "真实 DeepACO 网络需要已授权的 PyG 目录")
class TestOtherTasksFloat64(unittest.TestCase):
    def test_tsp_cvrp_and_mkp_reject_float32_input(self):
        rng = np.random.RandomState(8123)
        points = rng.rand(20, 2)
        prizes = rng.rand(20)
        weights = rng.rand(20, 5) / 20
        cases = (
            (DeepACOReference("tsp", "tsp20"), {"coordinates": points.astype(np.float32), "k_sparse": 5}),
            (DeepACOReference("cvrp", "cvrp20"),
             {"coordinates": points.astype(np.float32), "demands": np.arange(20)}),
            (DeepACOReference("cvrp", "cvrp20"),
             {"coordinates": points, "demands": np.arange(20, dtype=np.float32)}),
            (DeepACOReference("mkp", "mkp300"), {"prize": prizes.astype(np.float32), "weight": weights}),
            (DeepACOReference("mkp", "mkp300"), {"prize": prizes, "weight": weights.astype(np.float32)}),
            (DeepACOReference("bpp", "bpp120"), {"demands": np.arange(20, dtype=np.float32)}),
        )
        for model, inputs in cases:
            with self.subTest(task=model.task), self.assertRaises(ValueError):
                model.predict(**inputs)

    def test_sparse_networks_use_source_edge_index_reshape(self):
        points = np.random.RandomState(8124).rand(20, 2)
        previous_dtype = torch.get_default_dtype()
        try:
            for task, checkpoint in (("tsp", "tsp20"), ("op", "op100")):
                with self.subTest(task=task):
                    model = DeepACOReference(task, checkpoint)
                    graph = model.utils.gen_pyg_data(torch.from_numpy(points.copy()), 5)[0]
                    torch.set_default_dtype(torch.float64)
                    with torch.no_grad():
                        expected = model.model.reshape(graph, model.model(graph)) + 1e-10
                    torch.set_default_dtype(torch.float32)
                    actual = model.predict(coordinates=points, k_sparse=5)
                    np.testing.assert_array_equal(actual, expected.numpy())
        finally:
            torch.set_default_dtype(previous_dtype)


@unittest.skipUnless(HAS_PYG, "真实 DeepACO 网络需要已授权的 PyG 目录")
class TestRealOP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DeepACOReference("op", "op100")
        rng = np.random.RandomState(7451)
        rng.rand(20, 2)
        cls.positions = rng.rand(20, 2)

    def test_float64_graph_uses_exact_project_prizes_on_rounding_boundary(self):
        prizes, distances = op_task.instance(self.positions)
        legacy = self.model.utils.gen_prizes(torch.as_tensor(self.positions, dtype=torch.float32)).numpy()
        self.assertGreater(np.max(np.abs(legacy - prizes.numpy())), 1e-4)
        captured = {}
        original = self.model.utils.gen_pyg_data

        def graph(*args, **kwargs):
            result = original(*args, **kwargs)
            captured["graph"] = result[0]
            return result

        with patch.object(self.model.utils, "gen_pyg_data", side_effect=graph):
            scores = self.model.predict(coordinates=self.positions, prize=prizes.numpy(), k_sparse=10)
        self.assertEqual(prizes.dtype, torch.float64)
        self.assertEqual(distances.dtype, torch.float64)
        self.assertEqual(captured["graph"].x.dtype, torch.float64)
        self.assertEqual(captured["graph"].edge_attr.dtype, torch.float64)
        self.assertEqual(next(self.model.model.parameters()).dtype, torch.float64)
        self.assertEqual(scores.dtype, np.float64)
        np.testing.assert_array_equal(captured["graph"].x[:, 1].numpy(), prizes.numpy())
        wrong = prizes.numpy().copy()
        wrong[0] += 0.1
        with self.assertRaisesRegex(ValueError, "收益与项目评价实例不一致"):
            self.model.predict(coordinates=self.positions, prize=wrong, k_sparse=10)

    def test_worker_and_supervisor_freeze_identical_op_sources(self):
        source, implementation = supervisor.reference_implementation_snapshot("op_aco", "op100")
        self.assertEqual(source, self.model.source_snapshot())
        self.assertIn(str(Path(op_task.__file__).resolve()), implementation)

    def test_float32_inputs_are_rejected_before_precision_is_lost(self):
        positions = self.positions.astype(np.float32)
        with self.assertRaisesRegex(ValueError, "float64 原始坐标"):
            op_task.instance(positions)
        with self.assertRaisesRegex(ValueError, "float64 原始坐标"):
            self.model.predict(coordinates=positions, k_sparse=10)
        with self.assertRaises(ValueError):
            op_task.solve_deepaco(positions, self.model, k_sparse=10, n_iterations=1)

    def test_reference_solver_matches_original_with_float64_default(self):
        prizes, distances = op_task.instance(self.positions)
        scores = self.model.predict(coordinates=self.positions, prize=prizes.numpy(), k_sparse=10)
        source = load_source("op", "aco")
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            torch.manual_seed(991)
            original = source.ACO(distances.clone(), prizes.clone(), 3.0, n_ants=op_task.N_ANTS,
                                  heuristic=torch.from_numpy(scores.copy()), device="cpu")
            old_obj, old_sol = original.run(2)
            torch.set_default_dtype(torch.float32)
            torch.manual_seed(991)
            objective, solution, distance, iterations = op_task.solve_deepaco(
                self.positions, self.model, k_sparse=10, n_iterations=2, max_len=3.0)
        finally:
            torch.set_default_dtype(original_dtype)
        self.assertEqual(solution, old_sol.tolist())
        self.assertEqual(objective, float(old_obj))
        self.assertEqual(iterations, 2)
        expected, checked_distance = op_task.check_solution(solution, self.positions, prizes.numpy(), 3.0)
        self.assertAlmostEqual(objective, expected, places=12)
        self.assertEqual(distance, checked_distance)

    def test_dummy_distance_is_zero_with_float32_default(self):
        source = load_source("op", "aco")
        original_pick = source.ACO.pick_node
        positions = np.array([[0.0, 0.0], [1e-10, 0.0], [0.4, 0.1]])
        _prizes, expected_distances = op_task.instance(positions)
        observed = []

        def pick(solver, *args):
            observed.append(solver.distances.clone())
            return original_pick(solver, *args)

        def source_for_test(task, filename):
            return source if (task, filename) == ("op", "aco") else load_source(task, filename)

        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            torch.manual_seed(991)
            with patch.object(source.ACO, "pick_node", pick), \
                 patch("baselines.deepaco.reference.load_source", side_effect=source_for_test):
                op_task.solve_deepaco(positions, self.model, k_sparse=2, n_iterations=1)
        finally:
            torch.set_default_dtype(original_dtype)
        self.assertTrue(observed)
        self.assertTrue(all(matrix.dtype == torch.float64 and torch.count_nonzero(matrix[:, -1]) == 0
                            and torch.equal(matrix[:-1, :-1], expected_distances) for matrix in observed))

    def test_float64_path_accumulation_rejects_boundary_overrun(self):
        positions = np.array([[0.0, 0.0], [0.499999982, 0.0], [0.5, 0.0001]])
        full = sum(math.hypot(*(positions[a] - positions[b])) for a, b in ((0, 1), (1, 2), (2, 0)))
        limit = full - 5e-9
        source = load_source("op", "aco")
        real_load = load_source

        def choose(solver, mask, _current, _require_prob):
            nodes = [1 if mask[i, 1] else 2 if mask[i, 2] else 3 for i in range(solver.n_ants)]
            return torch.tensor(nodes, dtype=torch.int64), None

        def source_for_test(task, filename):
            return source if (task, filename) == ("op", "aco") else real_load(task, filename)

        with patch.object(source.ACO, "pick_node", choose), \
             patch("baselines.deepaco.reference.load_source", side_effect=source_for_test):
            _objective, solution, distance, _iterations = op_task.solve_deepaco(
                positions, self.model, k_sparse=2, n_iterations=1, max_len=limit)
        self.assertNotIn(2, solution)
        self.assertLessEqual(distance, limit)

    def test_project_evaluation_emits_reference_results_with_same_fitness(self):
        events = []

        def fake_solve(points, model, *, k_sparse, n_iterations, max_len):
            self.assertIs(model, self.model)
            self.assertEqual(points.dtype, np.float64)
            self.assertEqual(k_sparse, 10)
            self.assertEqual(max_len, op_task.MAX_LEN)
            prizes, _ = op_task.instance(points)
            return float(prizes[0]), [0], 0.0, n_iterations

        with patch.object(sys, "argv", ["eval.py", "50", "unused", "train"]), \
             patch.object(op_task, "emit", side_effect=events.append), \
             patch.object(op_task, "load_heuristic", side_effect=AssertionError("不可加载候选")), \
             patch.object(op_task, "solve_deepaco", side_effect=fake_solve):
            op_task.main(deepaco_model=self.model, k_sparse=10)
        self.assertEqual([event["event"] for event in events], ["start"] + ["instance"] * 5 + ["complete"])
        self.assertEqual(events[0]["reference"]["id"], "deepaco_op64_project_eval")
        self.assertEqual(events[-1]["fitness"], events[-1]["average"])

    def test_real_model_reaches_project_evaluation_events_on_synthetic_data(self):
        points = np.random.RandomState(661).rand(1, 50, 2)
        events = run_synthetic_training_main(op_task, self.model, points, "coordinates", k_sparse=10)
        self.assertEqual([event["event"] for event in events], ["start", "instance", "complete"])
        prizes, _ = op_task.instance(points[0])
        expected, _distance = op_task.check_solution(events[1]["solution"], points[0], prizes.numpy(), 3.0)
        self.assertAlmostEqual(events[1]["objective"], expected, places=12)
        self.assertEqual(events[-1]["fitness"], events[-1]["average"])


@unittest.skipUnless(HAS_PYG and HAS_NUMBA, "真实 DeepACO BPP 需要已授权的 PyG 与 Numba 目录")
class TestRealBPP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DeepACOReference("bpp", "bpp120")

    def test_graph_and_model_use_float64(self):
        captured = {}
        original = self.model.utils.gen_pyg_data

        def graph(*args, **kwargs):
            result = original(*args, **kwargs)
            captured["graph"] = result
            return result

        with patch.object(self.model.utils, "gen_pyg_data", side_effect=graph):
            scores = self.model.predict(demands=np.array([50, 25, 100]))
        self.assertEqual(captured["graph"].x.dtype, torch.float64)
        self.assertEqual(captured["graph"].edge_attr.dtype, torch.float64)
        self.assertEqual(scores.dtype, np.float64)
        self.assertTrue(all(p.dtype == torch.float64 for p in self.model.model.parameters()))

    def test_network_matrix_uses_source_flattened_edge_order(self):
        demand = np.array([50, 25, 100])
        source_demand = torch.as_tensor(np.r_[0, demand], dtype=torch.float64)
        graph = self.model.utils.gen_pyg_data(source_demand, "cpu")
        graph.edge_attr = graph.edge_attr.to(dtype=torch.float64)
        with torch.no_grad():
            expected = self.model.model(graph).reshape((len(source_demand), len(source_demand))) + 1e-10
        actual = self.model.predict(demands=demand)
        np.testing.assert_array_equal(actual, expected.numpy())

    def test_worker_and_supervisor_freeze_identical_bpp_sources(self):
        source, implementation = supervisor.reference_implementation_snapshot("bpp_offline_aco", "bpp120")
        self.assertEqual(source, self.model.source_snapshot())
        self.assertIn(str(Path(bpp_task.__file__).resolve()), implementation)

    def test_source_selection_is_preserved_and_reported_as_project_bin_count(self):
        demand = np.array([50, 25] * 4)
        scores = self.model.predict(demands=demand)
        self.assertEqual(scores.dtype, np.float64)
        self.assertEqual(next(self.model.model.parameters()).dtype, torch.float64)
        source = load_source("bpp", "aco")
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            torch.manual_seed(12001)
            original = source.ACO(torch.as_tensor(np.r_[0, demand], dtype=torch.float64),
                                  n_ants=bpp_task.N_ANTS, heuristic=torch.from_numpy(scores.copy()),
                                  device="cpu", capacity=bpp_task.CAPACITY)
            old_fitness = float(original.run(2))
            old_assignment = bpp_task.deepaco_assignment(original.shortest_path.tolist(), demand)
            torch.manual_seed(12001)
            count, assignment, iterations, source_fitness = bpp_task.solve_deepaco(
                demand, self.model, n_iterations=2)
        finally:
            torch.set_default_dtype(original_dtype)
        self.assertEqual(assignment, old_assignment.tolist())
        self.assertEqual(source_fitness, old_fitness)
        self.assertEqual(count, bpp_task.check_solution(np.asarray(assignment), demand))
        self.assertEqual(iterations, 2)

    def test_project_evaluation_keeps_source_fitness_separate_from_bin_count(self):
        events = []

        def fake_solve(demand, model, *, n_iterations):
            self.assertIs(model, self.model)
            return len(demand), list(range(len(demand))), n_iterations, 0.625

        with patch.object(sys, "argv", ["eval.py", "500", "unused", "train"]), \
             patch.object(bpp_task, "emit", side_effect=events.append), \
             patch.object(bpp_task, "load_heuristic", side_effect=AssertionError("不可加载候选")), \
             patch.object(bpp_task, "solve_deepaco", side_effect=fake_solve):
            bpp_task.main(deepaco_model=self.model)
        self.assertEqual([event["event"] for event in events], ["start"] + ["instance"] * 5 + ["complete"])
        self.assertEqual(events[0]["reference"]["id"], "deepaco_bpp64_project_eval")
        self.assertTrue(all(event["source_fitness"] == 0.625 and event["objective"] == 500
                            for event in events[1:-1]))
        self.assertEqual(events[-1]["average"], 500)
        self.assertEqual(events[-1]["fitness"], -500)

    def test_real_model_reaches_project_evaluation_events_on_synthetic_data(self):
        demand = np.random.RandomState(662).randint(20, 101, size=(1, 500), dtype=np.int32)
        events = run_synthetic_training_main(bpp_task, self.model, demand, "demands")
        self.assertEqual([event["event"] for event in events], ["start", "instance", "complete"])
        assignment = np.asarray(events[1]["solution"], dtype=np.int64)
        self.assertEqual(events[1]["objective"], bpp_task.check_solution(assignment, demand[0]))
        self.assertIn("source_fitness", events[1])
        self.assertEqual(events[-1]["fitness"], -events[-1]["average"])


if __name__ == "__main__":
    unittest.main()
