"""检查训练入口的逐例计时及监督器到速度反馈的传递。"""

from contextlib import ExitStack
import math
import sys
import unittest
from unittest.mock import patch

from scripts import p0_controller as control
from tests import (test_p1_bpp_aco as bpp, test_p1_cvrp_aco as cvrp,
                   test_p1_mkp_aco as mkp, test_p1_op_aco as op,
                   test_p1_tsp_gls as gls)


CASES = (
    ("op_aco", op, "50", ("objective", "solution", "distance", "iterations"), True),
    ("cvrp_aco", cvrp, "50", ("objective", "path", "iterations"), True),
    ("mkp_aco", mkp, "100", ("objective", "solution", "iterations"), True),
    ("bpp_offline_aco", bpp, "500", ("objective", "solution", "iterations"), True),
    ("tsp_gls", gls, "200", ("objective", "tour", "iterations"), False),
)


class TestTrainingSolveTiming(unittest.TestCase):
    def test_training_entrypoints_supply_validated_per_instance_times(self):
        for name, test_module, size, fields, supports_black_box in CASES:
            for black_box in ((False, True) if supports_black_box else (False,)):
                with self.subTest(task=name, black_box=black_box):
                    task = test_module.task
                    rows = iter(test_module.TestCompletion.events()[1:-1])
                    count = task.N_INSTANCES
                    timeline, emitted = [], []
                    clock = iter(value for i in range(count)
                                 for value in (float(i * 100), float(i * 100 + i + 1)))

                    def tick():
                        timeline.append("clock")
                        return next(clock)

                    def solve(*_args, **_kwargs):
                        timeline.append("solve")
                        row = next(rows)
                        return tuple(row[field] for field in fields)

                    with ExitStack() as stack:
                        stack.enter_context(patch.object(sys, "argv", ["eval.py", size, "unused", "train"]))
                        stack.enter_context(patch.object(task, "load_heuristic", return_value=lambda *_args: None))
                        stack.enter_context(patch.object(task, "perf_counter", side_effect=tick))
                        stack.enter_context(patch.object(task, "solve", side_effect=solve))
                        stack.enter_context(patch.object(task, "emit", side_effect=emitted.append))
                        stack.enter_context(patch.object(task.random, "seed"))
                        stack.enter_context(patch.object(task.np.random, "seed"))
                        if hasattr(task, "torch"):
                            stack.enter_context(patch.object(task.torch, "manual_seed"))
                        if name == "bpp_offline_aco":
                            stack.enter_context(patch.object(task.source_aco, "uniform_generator"))
                        if supports_black_box:
                            task.main(black_box=black_box)
                        else:
                            task.main()

                    self.assertEqual(timeline, ["clock", "solve", "clock"] * count)
                    self.assertEqual(emitted[0]["time_scope"], control.SOLVER_TIME_SCOPE)
                    expected = [float(i + 1) for i in range(count)]
                    self.assertEqual([row["solve_wall_seconds"] for row in emitted[1:-1]], expected)
                    self.assertEqual(emitted[-1]["average_solve_wall_seconds"], math.fsum(expected) / count)
                    checked = test_module.TestCompletion.read(emitted)
                    self.assertEqual(checked["solve_wall_seconds"], expected)
                    self.assertEqual(control.timing_from_result(checked),
                                     (math.fsum(expected) / count, control.SOLVER_TIME_SCOPE))

    def test_incomplete_or_inconsistent_timing_is_rejected_for_every_task(self):
        for name, test_module, _size, _fields, _black_box in CASES:
            with self.subTest(task=name):
                original = test_module.TestCompletion.events()
                self.assertNotIn("average_solve_wall_seconds", test_module.TestCompletion.read(original))

                def timed_events():
                    events = test_module.TestCompletion.events()
                    events[0]["time_scope"] = control.SOLVER_TIME_SCOPE
                    for row in events[1:-1]:
                        row["solve_wall_seconds"] = 2.0
                    events[-1]["average_solve_wall_seconds"] = 2.0
                    return events

                valid = timed_events()
                self.assertEqual(test_module.TestCompletion.read(valid)["average_solve_wall_seconds"], 2.0)
                for field, value in (("solve_wall_seconds", None), ("solve_wall_seconds", 0),
                                     ("solve_wall_seconds", True), ("solve_wall_seconds", float("nan"))):
                    bad = timed_events()
                    bad[1][field] = value
                    with self.subTest(task=name, field=field, value=value), self.assertRaises(ValueError):
                        test_module.TestCompletion.read(bad)
                for mutation in ("mean", "scope", "missing_scope", "missing_mean"):
                    bad = timed_events()
                    if mutation == "mean":
                        bad[-1]["average_solve_wall_seconds"] = 3.0
                    elif mutation == "scope":
                        bad[0]["time_scope"] = "whole_training_evaluation_wall_seconds"
                    elif mutation == "missing_scope":
                        bad[0].pop("time_scope")
                    else:
                        bad[-1].pop("average_solve_wall_seconds")
                    with self.subTest(task=name, mutation=mutation), self.assertRaises(ValueError):
                        test_module.TestCompletion.read(bad)


if __name__ == "__main__":
    unittest.main()
