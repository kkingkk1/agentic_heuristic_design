"""速度评价的停止、计时汇总和共同排序回归。"""

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "problems/tsp_aco"))
from aco import ACO  # noqa: E402
from baselines.roco.paper_controller import RoCoController  # noqa: E402
from scripts.p0_controller import Controller  # noqa: E402
from scripts.p0_tsp_aco_evaluate import read_complete  # noqa: E402
from scripts import p1_formal_matrix as matrix  # noqa: E402
from scripts.p1_test_evaluate import read_complete as read_test_complete  # noqa: E402

spec = importlib.util.spec_from_file_location("p1_speed_tsp_eval", ROOT / "problems/tsp_aco/eval.py")
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


class ScriptedACO(ACO):
    def __init__(self, costs):
        super().__init__(torch.ones((2, 2)), torch.ones((2, 2)), n_ants=1)
        self.costs = iter(costs)
        self.calls = 0

    def gen_path(self, require_prob=False):
        self.calls += 1
        return torch.tensor([[0], [1]])

    def gen_path_costs(self, paths):
        return torch.tensor([next(self.costs)], dtype=torch.float64)

    def update_pheronome(self, paths, costs):
        pass


class MemoryLog:
    def __init__(self):
        self.events = []

    def event(self, name, **fields):
        self.events.append((name, fields))


class FakeBackend:
    cfg = {"task": {"evaluation_mode": "quality_per_time", "stagnation_rounds": 2}}

    def __init__(self):
        self.results = iter([(5.0, 2.0), (6.0, 1.0)])

    def prepare_proposal(self, candidate):
        pass

    def release_proposal(self, candidate):
        pass

    def model(self, stage, prompt):
        return stage

    def evaluate(self, candidate):
        objective, elapsed = next(self.results)
        return {"status": "passed", "average": objective, "average_solve_wall_seconds": elapsed,
                "fitness": 1.0 / (objective * elapsed)}


def output_events():
    start = {"event": "start", "n_ants": 30, "n_iterations": 200, "seed_base": 20261103,
             "evaluation_mode": "quality_per_time", "stagnation_rounds": 2,
             "time_scope": "solve_wall_seconds_including_preparation_and_validation"}
    entries = [{"event": "instance", "index": index, "seed": 20261103 + index,
                "objective": float(index + 4), "iterations": 4, "tour": list(range(50)),
                "solve_wall_seconds": float(index + 1), "stop_reason": "stagnation"}
               for index in range(5)]
    final = {"event": "complete", "instance_count": 5, "average": 6.0,
             "average_solve_wall_seconds": 3.0, "fitness": 1.0 / 18.0}
    return [start, *entries, final]


def read_events(events):
    content = "\n".join("P0_JSON " + json.dumps(event) for event in events)
    with patch.object(Path, "read_text", return_value=content):
        return read_complete(Path("unused"), stagnation_rounds=2)


class TestSpeedEvaluation(unittest.TestCase):
    def test_stagnation_and_iteration_cap(self):
        solver = ScriptedACO([10, 8, 8, 8, 7, 7])
        self.assertEqual(solver.run(6, stagnation_rounds=2).item(), 8)
        self.assertEqual((solver.calls, solver.stop_reason), (4, "stagnation"))
        solver = ScriptedACO([10, 8, 8, 8])
        self.assertEqual(solver.run(4, stagnation_rounds=2).item(), 8)
        self.assertEqual((solver.calls, solver.stop_reason), (4, "iteration_limit"))
        solver = ScriptedACO([10, 8, 8, 8, 7, 7])
        self.assertEqual(solver.run(6).item(), 7)
        self.assertEqual(solver.calls, 6)
        with self.assertRaises(ValueError):
            solver.run(1, stagnation_rounds=0)

    def test_aggregate_first_then_divide_and_check_raw_evidence(self):
        result = read_events(output_events())
        self.assertEqual(result["average"], 6.0)
        self.assertEqual(result["average_solve_wall_seconds"], 3.0)
        self.assertEqual(result["fitness"], 1.0 / 18.0)
        self.assertEqual(result["solve_wall_seconds"], [1, 2, 3, 4, 5])
        events = output_events()
        events[2]["solve_wall_seconds"] = 9.0
        with self.assertRaises(ValueError):
            read_events(events)
        events = output_events()
        events[2]["stop_reason"] = "iteration_limit"
        with self.assertRaises(ValueError):
            read_events(events)

    def test_training_entrypoint_reports_per_instance_time_and_score(self):
        stream = io.StringIO()
        values = [(float(i + 4), list(range(50)), 4) for i in range(5)]
        clock = [value for i in range(5) for value in (10.0 * (i + 1), 10.0 * (i + 1) + i + 1)]
        args = ["eval.py", "50", "candidate", "train", "--stagnation-rounds", "2"]
        with patch.object(sys, "argv", args), patch.object(task.np, "load", return_value=np.zeros((5, 50, 2))), \
                patch.object(task, "load_heuristic", return_value=object()), \
                patch.object(task, "solve", side_effect=values) as solve, \
                patch.object(task, "perf_counter", side_effect=clock), contextlib.redirect_stdout(stream):
            task.main()
        events = [json.loads(line.removeprefix("P0_JSON ")) for line in stream.getvalue().splitlines()
                  if line.startswith("P0_JSON ")]
        result = read_events(events)
        self.assertEqual(result["fitness"], 1.0 / 18.0)
        self.assertTrue(all(call.kwargs["stagnation_rounds"] == 2 for call in solve.call_args_list))

    def test_test_entrypoint_validates_all_64_instance_times(self):
        dataset = Path("example_dataset.npy").resolve()
        entry = {"evaluation_seed_base": 100, "iterations": 500}
        manifest = {"task": "tsp_aco", "candidate_sha256": "abc"}
        start = {"event": "start", "dataset": str(dataset), "candidate_sha256": "abc", "seed_base": 100,
                 "n_iterations": 500, "n_ants": 30, "evaluation_mode": "quality_per_time",
                 "stagnation_rounds": 2, "time_scope": "solve_wall_seconds_including_preparation_and_validation"}
        instances = [{"event": "instance", "index": i, "seed": 100 + i, "objective": 6.0,
                      "iterations": 4, "stop_reason": "stagnation", "solve_wall_seconds": 2.0}
                     for i in range(64)]
        complete = {"event": "complete", "instance_count": 64, "average": 6.0,
                    "average_solve_wall_seconds": 2.0, "fitness": 1.0 / 12.0}
        content = "\n".join("P1_JSON " + json.dumps(row) for row in (start, *instances, complete))
        with patch.object(Path, "read_text", return_value=content):
            result = read_test_complete(Path("unused"), entry, manifest, "min", dataset, 30,
                                        stagnation_rounds=2)
        self.assertEqual(result["fitness"], 1.0 / 12.0)
        self.assertEqual(len(result["solve_wall_seconds"]), 64)

    def test_real_tsp_solver_stops_on_equal_triangle_tours(self):
        triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        torch.manual_seed(2026)
        objective, tour, iterations = task.solve(triangle, lambda distances: 1.0 / distances, 10,
                                                 stagnation_rounds=2)
        self.assertAlmostEqual(objective, 2.0 + math.sqrt(2.0))
        self.assertEqual((sorted(tour), iterations), ([0, 1, 2], 3))

    def test_both_controllers_select_and_describe_same_speed_score(self):
        for controller_type in (Controller, RoCoController):
            with self.subTest(controller=controller_type.__name__):
                controller = controller_type(FakeBackend(), MemoryLog(), population_size=2, initial_count=2)
                controller.parse_proposal = lambda response: {"thought": response, "code": "x = " + response[-1]}
                first = controller.propose(controller.register("init_1", ()), "prompt")
                second = controller.propose(controller.register("init_2", ()), "prompt")
                self.assertTrue(math.isclose(first.objective, 0.1))
                self.assertTrue(math.isclose(second.objective, 1.0 / 6.0))
                self.assertEqual(controller.select([first, second], 1), [second])
                self.assertEqual(controller.direction, "max")
                self.assertIn("Higher scores are better", controller.task)
                self.assertIn("higher objective is better", controller.branch_critic_prompt(first, second).lower())

    def test_speed_is_pending_until_its_stopping_rule_is_frozen(self):
        configuration = json.loads(matrix.MATRIX_PATH.read_text(encoding="utf-8"))
        self.assertIsNone(matrix.speed_stagnation(configuration, "tsp_aco", "final_quality"))
        with self.assertRaisesRegex(ValueError, "尚未单独冻结"):
            matrix.speed_stagnation(configuration, "tsp_aco", "quality_per_time")
        quality = matrix.training_config(configuration, "tsp_aco", "white_box", "roco_paper_adapted", 1)
        self.assertNotIn("evaluation_mode", quality["task"])
        with self.assertRaisesRegex(ValueError, "尚未单独冻结"):
            matrix.training_config(configuration, "tsp_aco", "white_box", "roco_paper_adapted", 1,
                                   evaluation_mode="quality_per_time")


if __name__ == "__main__":
    unittest.main()
