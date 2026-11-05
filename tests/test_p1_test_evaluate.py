import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import p1_test_evaluate as test_runner
from scripts.p0_runtime import digest


class FakeJob:
    def __init__(self, values):
        self.values, self.index, self.terminated = values, 0, False

    def accounting(self):
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value

    def terminate(self):
        self.terminated = True


class FakeProcess:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class TestP1TestSupervisor(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(test_runner.DATA_CONTRACT.read_text(encoding="utf-8"))
        self.manifest = {"status": "frozen_after_training", "task": "tsp_aco", "prompt_condition": "white_box",
                         "candidate_sha256": "a" * 64}
        self.config = {"status": "frozen", "task": "tsp_aco", "prompt_condition": "white_box", "size": 50,
                       "data_contract_sha256": digest(test_runner.DATA_CONTRACT), "solver_repeats_per_instance": 1,
                       "whole_dataset_wall_timeout_seconds": 1200, "process_tree_cpu_seconds": 5000}
        self.entry = test_runner.load_test_contract(self.config, self.manifest, self.data)
        self.dataset = test_runner.ROOT / self.entry["path"]

    def events(self):
        events = [{"event": "start", "dataset": str(self.dataset), "candidate_sha256": "a" * 64,
                   "seed_base": self.entry["evaluation_seed_base"], "n_iterations": self.entry["iterations"], "n_ants": 30}]
        events.extend({"event": "instance", "index": i, "seed": self.entry["evaluation_seed_base"] + i,
                       "iterations": self.entry["iterations"], "objective": float(i + 1)} for i in range(64))
        events.append({"event": "complete", "instance_count": 64, "average": 32.5})
        return events

    def read(self, events):
        text = "\n".join(test_runner.PREFIX + json.dumps(event) for event in events)
        with patch.object(Path, "read_text", return_value=text):
            return test_runner.read_complete(Path("fixture"), self.entry, self.manifest, "min", self.dataset, 30)

    def test_frozen_identity_and_complete_events(self):
        result = self.read(self.events())
        self.assertEqual((result["average"], result["fitness"], len(result["objectives"])), (32.5, -32.5, 64))
        for key, value in (("status", "engineering"), ("data_contract_sha256", "0" * 64),
                           ("solver_repeats_per_instance", 2), ("size", 49)):
            config = dict(self.config, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                test_runner.load_test_contract(config, self.manifest, self.data)

    def test_missing_or_inconsistent_test_evidence_is_rejected(self):
        variants = []
        events = self.events(); events[0]["candidate_sha256"] = "0" * 64; variants.append(events)
        events = self.events(); events[0]["n_ants"] = 20; variants.append(events)
        events = self.events(); events[5]["seed"] = -1; variants.append(events)
        events = self.events(); events[-1]["average"] = 1.0; variants.append(events)
        events = self.events(); events.pop(4); variants.append(events)
        for events in variants:
            with self.subTest(events=events[:2]), self.assertRaises(ValueError):
                self.read(events)

    def test_process_tree_cpu_limit_including_completed_job(self):
        job = FakeJob([{"active_processes": 1, "cpu_seconds": 2}, {"active_processes": 0, "cpu_seconds": 6}])
        process = FakeProcess()
        with patch.object(test_runner.time, "monotonic", return_value=1), patch.object(test_runner.time, "sleep"):
            reason, _usage = test_runner.wait_bounded(job, process, deadline=10, cpu_limit=5)
        self.assertEqual(reason, "cpu_budget")
        self.assertTrue(process.waited)

    def test_active_job_is_terminated_on_cpu_limit(self):
        job = FakeJob([{"active_processes": 1, "cpu_seconds": 6}, {"active_processes": 0, "cpu_seconds": 6}])
        process = FakeProcess()
        with patch.object(test_runner.time, "monotonic", return_value=1), patch.object(test_runner.time, "sleep"):
            reason, _usage = test_runner.wait_bounded(job, process, deadline=10, cpu_limit=5)
        self.assertEqual(reason, "cpu_budget")
        self.assertTrue(job.terminated)

    def test_gls_reference_stop_counts_and_negative_gap(self):
        manifest = dict(self.manifest, task="tsp_gls")
        config = dict(self.config, task="tsp_gls", size=20)
        entry = test_runner.load_test_contract(config, manifest, self.data)
        dataset = test_runner.ROOT / entry["path"]
        reference = test_runner.load_reference(20, entry["sha256"])
        start = {"event": "start", "dataset": str(dataset), "candidate_sha256": "a" * 64,
                 "seed_base": entry["evaluation_seed_base"], "n_iterations": entry["iterations"],
                 "perturbation_moves": entry["perturbation_moves"],
                 "lkh_reference_sha256": reference["sha256"], "lkh_reference_mean": reference["mean"],
                 "lkh_absolute_tolerance": test_runner.ABS_TOLERANCE,
                 "lkh_relative_tolerance": test_runner.REL_TOLERANCE}
        rows = []
        for index, ref in enumerate(reference["objectives"]):
            reached = index != 0
            rows.append({"event": "instance", "index": index, "seed": entry["evaluation_seed_base"] + index,
                         "iterations": 1 if reached else entry["iterations"],
                         "objective": ref * (0.99 if reached else 1.01), "lkh_reference": ref,
                         "stop_reason": "reference_reached" if reached else "iteration_limit"})
        average = sum(row["objective"] for row in rows) / 64
        complete = {"event": "complete", "instance_count": 64, "average": average, "fitness": -average,
                    "lkh_reference_mean": reference["mean"],
                    "lkh_gap_percent": (average / reference["mean"] - 1.0) * 100.0}
        events = [start, *rows, complete]

        def read(items):
            text = "\n".join(test_runner.PREFIX + json.dumps(item) for item in items)
            with patch.object(Path, "read_text", return_value=text):
                return test_runner.read_complete(Path("fixture"), entry, manifest, "min", dataset, None, reference)

        result = read(events)
        self.assertLess(result["lkh_gap_percent"], 0)
        self.assertEqual(result["actual_iterations"][0], entry["iterations"])
        self.assertEqual(result["stopping_reasons"].count("reference_reached"), 63)
        bad_start = [dict(start, lkh_reference_sha256="0" * 64), *rows, complete]
        bad_stop = [start, dict(rows[0], iterations=1), *rows[1:], complete]
        bad_reason = [start, rows[0], dict(rows[1], stop_reason="iteration_limit"), *rows[2:], complete]
        bad_gap = [start, *rows, dict(complete, lkh_gap_percent=0.0)]
        for bad in (bad_start, bad_stop, bad_reason, bad_gap):
            with self.subTest(first=bad[0]), self.assertRaises(ValueError):
                read(bad)


if __name__ == "__main__":
    unittest.main()
