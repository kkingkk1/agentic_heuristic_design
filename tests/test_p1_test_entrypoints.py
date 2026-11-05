import contextlib
from unittest.mock import patch
import sys
import unittest

import numpy as np

from tests import test_p1_bpp_aco, test_p1_cvrp_aco, test_p1_mkp_aco, test_p1_op_aco


CASES = (
    ("mkp", test_p1_mkp_aco.task, (100, 200, 300, 500, 700), 100),
    ("cvrp", test_p1_cvrp_aco.task, (50, 100, 200, 500), 500),
    ("op", test_p1_op_aco.task, (50, 100, 200, 500), 200),
    ("bpp", test_p1_bpp_aco.task, (500, 700, 1000), 100),
)


def fake_solve(kind, iterations, max_len, black_box):
    def solve(*args, **kwargs):
        if kind in ("mkp", "cvrp"):
            assert args[3] == iterations
        else:
            assert args[2] == iterations
        if kind == "op":
            assert args[3] == max_len
            return 1.0, [0], 0.0, iterations
        if kind == "cvrp":
            assert kwargs["black_box"] is black_box
            return 1.0, [0, 0], iterations
        if kind == "mkp":
            return 1.0, [], iterations
        return 1, [], iterations
    return solve


class TestCandidateTestEntrypoints(unittest.TestCase):
    def run_entry(self, kind, task, size, iterations, black_box, hashes=("a" * 64, "a" * 64), events=None):
        if events is None:
            events = []
        max_len = task.TEST_MAX_LEN[size] if kind == "op" else None
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["eval.py", str(size), "unused", "test"]))
            stack.enter_context(patch.object(task, "candidate_hash", side_effect=hashes))
            loader = stack.enter_context(patch.object(task, "load_heuristic", return_value=object()))
            solver = stack.enter_context(patch.object(task, "solve", side_effect=fake_solve(kind, iterations, max_len, black_box)))
            stack.enter_context(patch.object(task, "emit", side_effect=events.append))
            stack.enter_context(patch.object(task.random, "seed"))
            stack.enter_context(patch.object(np.random, "seed"))
            if hasattr(task, "torch"):
                stack.enter_context(patch.object(task.torch, "manual_seed"))
            if kind == "bpp":
                stack.enter_context(patch.object(task.source_aco, "uniform_generator", task.source_aco.uniform_generator))
            task.main(black_box=black_box)
            self.assertIs(loader.call_args.kwargs["black_box"], black_box)
        return events, solver.call_count

    def test_all_sizes_and_prompt_conditions_use_fixed_test_contract(self):
        for kind, task, sizes, iterations in CASES:
            for size in sizes:
                for black_box in (False, True):
                    with self.subTest(task=kind, size=size, black_box=black_box):
                        events, calls = self.run_entry(kind, task, size, iterations, black_box)
                        self.assertEqual(calls, 64)
                        self.assertEqual([event["event"] for event in events],
                                         ["start"] + ["instance"] * 64 + ["complete"])
                        self.assertEqual(events[0]["n_iterations"], iterations)
                        self.assertEqual(events[0]["candidate_sha256"], "a" * 64)
                        self.assertEqual(events[1]["seed"], 2026110400 + size * 100)
                        self.assertEqual(events[-2]["seed"], 2026110400 + size * 100 + 63)
                        self.assertEqual(events[-1]["instance_count"], 64)
                        self.assertEqual(events[-1]["fitness"], 1.0 if kind in ("mkp", "op") else -1.0)
                        if kind == "op":
                            self.assertEqual(events[0]["max_len"], {50: 3.0, 100: 4.0, 200: 5.0, 500: 8.0}[size])

    def test_modified_candidate_stops_before_complete_event(self):
        for kind, task, sizes, iterations in CASES:
            events = []
            with self.subTest(task=kind), self.assertRaisesRegex(ValueError, "候选代码发生变化"):
                self.run_entry(kind, task, sizes[0], iterations, False,
                               hashes=("a" * 64, "b" * 64), events=events)
            self.assertNotIn("complete", [event["event"] for event in events])

    def test_wrong_dataset_hash_fails_before_loading_candidate(self):
        for kind, task, sizes, _iterations in CASES:
            size = sizes[0]
            with self.subTest(task=kind), patch.object(sys, "argv", ["eval.py", str(size), "unused", "test"]), \
                    patch.dict(task.TEST_DATASET_SHA256, {size: "0" * 64}), \
                    patch.object(task, "load_heuristic") as loader:
                with self.assertRaisesRegex(ValueError, "哈希不匹配"):
                    task.main()
                loader.assert_not_called()

    def test_op_extended_length_is_applied_to_constraint(self):
        task = test_p1_op_aco.task
        positions = np.array([[0.0, 0.0], [2.0, 0.0]])
        prizes = np.array([0.01, 1.0])
        with self.assertRaises(ValueError):
            task.check_solution([0, 1], positions, prizes, max_len=3.0)
        reward, distance = task.check_solution([0, 1], positions, prizes, max_len=8.0)
        self.assertEqual(distance, 4.0)
        self.assertAlmostEqual(reward, 1.01)


if __name__ == "__main__":
    unittest.main()
