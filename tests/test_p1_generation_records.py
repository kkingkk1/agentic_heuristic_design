import io
import json
import unittest

from baselines.roco.paper_controller import RoCoController
from scripts.p0_controller import Candidate, Controller, StopRun
from scripts.p1_freeze_candidate import original_code
from scripts.p1_runtime import TrainingLog, _evaluation_evidence
from scripts.p1_task_contract import load_task_contract
from tests.test_p0_controller import FixtureBackend


class MemoryPath:
    """供测试使用的内存文件系统；不创建或清理本地文件。"""

    def __init__(self, files, name):
        self.files, self.name_string = files, name

    def __truediv__(self, child):
        return MemoryPath(self.files, self.name_string + "/" + str(child))

    @property
    def parent(self):
        return MemoryPath(self.files, self.name_string.rsplit("/", 1)[0])

    @property
    def name(self):
        return self.name_string.rsplit("/", 1)[-1]

    def mkdir(self, **kwargs):
        pass

    def is_file(self):
        return self.name_string in self.files

    def read_text(self, **kwargs):
        return self.files[self.name_string]

    def open(self, mode, **kwargs):
        if mode == "rb":
            return io.BytesIO(self.files[self.name_string].encode("utf-8"))
        if mode == "x" and self.is_file():
            raise FileExistsError(self.name_string)
        initial = self.files.get(self.name_string, "") if mode == "a" else ""
        files, name = self.files, self.name_string
        files[name] = initial

        class Writer(io.StringIO):
            def close(self):
                files[name] = self.getvalue()
                super().close()

        writer = Writer(initial)
        writer.seek(0, io.SEEK_END)
        return writer


class RecordingBackend(FixtureBackend):
    def __init__(self, log, run_dir, **kwargs):
        super().__init__(**kwargs)
        self.log, self.run_dir = log, run_dir
        self.completed = 0
        self.corrupt_stdout = False
        self.corrupt_code = False
        self.duplicate_initial_pair = False

    def model(self, stage, prompt):
        self.log.event("model_attempt", stage=stage, retry=False, prompt=prompt)
        response = super().model(stage, prompt)
        if self.duplicate_initial_pair and stage == "init_02":
            response = response.replace("init_02", "init_01")
        rows = [{"type": "item.completed", "item": {"type": "agent_message", "text": response}},
                {"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 3}}]
        self.log.event("model_process", stage=stage, retry=False,
                       result={"exit_code": 0, "stdout": "\n".join(map(json.dumps, rows)),
                               "stderr": "", "elapsed_seconds": 0.5})
        return response

    def evaluate(self, candidate):
        result = super().evaluate(candidate)
        with self.lock:
            self.completed += 1
            completed = self.completed
        result["candidate_id"] = candidate.candidate_id
        output = [{"event": "start", "seed_base": 20261103,
                   "time_scope": result.get("start", {}).get("time_scope", "fixture")}]
        if result["status"] == "passed":
            objective = result["average"]
            elapsed = result["average_solve_wall_seconds"]
            result.update(objectives=[objective], solve_wall_seconds=[elapsed],
                          completed_monotonic_ns=completed, elapsed_seconds=1.0,
                          process_tree_cpu={"cpu_seconds": 0.5})
            output.extend([{"event": "instance", "index": 0, "seed": 20261103,
                            "objective": objective + int(self.corrupt_stdout), "solve_wall_seconds": elapsed},
                           {"event": "complete", "instance_count": 1}])
        else:
            output.append({"event": "instance", "index": 0, "seed": 20261103,
                           "objective": 7.0, "solve_wall_seconds": 0.5})
        root = self.run_dir / "candidates" / candidate.candidate_id
        with (root / "problems" / "tsp_aco" / "gpt.py").open("x", encoding="utf-8") as stream:
            stream.write(candidate.code + ("\n# changed" if self.corrupt_code else "\n"))
        for filename, contents in (("result.json", json.dumps(result)),
                                   ("stdout.txt", "\n".join("P0_JSON " + json.dumps(item) for item in output)),
                                   ("stderr.txt", "fixture error" if result["status"] != "passed" else "")):
            with (root / filename).open("x", encoding="utf-8") as stream:
                stream.write(contents)
        return result


class TestGenerationRecords(unittest.TestCase):
    def setUp(self):
        self.files = {}
        self.run_dir = MemoryPath(self.files, "run")
        self.contract = load_task_contract("tsp_aco")
        self.log = TrainingLog(self.run_dir / "events.jsonl", self.contract)

    def record(self, generation):
        return json.loads(self.files[f"run/generation_records/generation_{generation:04d}.json"])

    def events(self):
        return [json.loads(line) for line in self.files["run/events.jsonl"].splitlines()]

    def test_both_controllers_write_every_generation_without_changing_selection(self):
        for controller_class, method in ((Controller, "proposed_solution_quality_and_speed_agents"),
                                         (RoCoController, "roco_paper_adapted")):
            with self.subTest(method=method):
                self.setUp()
                backend = RecordingBackend(self.log, self.run_dir)
                controller = controller_class(backend, self.log, task_contract=self.contract,
                                              population_size=2, initial_count=3, rounds=2, generations=2)
                population = controller.run()
                events = self.events()
                by_id = {candidate.candidate_id: candidate for candidate in controller.candidates}
                for generation in range(3):
                    record = self.record(generation)
                    registrations = [event for event in events if event["event"] == "candidate_registered"
                                     and event["generation"] == generation]
                    self.assertEqual(record["status"], "completed")
                    self.assertEqual({row["candidate_id"] for row in record["candidates"]},
                                     {event["candidate_id"] for event in registrations})
                    for row in record["candidates"]:
                        candidate = by_id[row["candidate_id"]]
                        self.assertEqual(row["generation_order"], candidate.generation_order)
                        self.assertEqual((row["status"], row["valid"]), ("evaluation_passed", True))
                        self.assertEqual(row["instances"][0]["objective"], row["evaluation"]["average"])
                        self.assertEqual(row["instances"][0]["solve_wall_seconds"],
                                         row["evaluation"]["average_solve_wall_seconds"])
                        self.assertNotIn("objectives", row["evaluation"])
                        if row["selected"] or row["operator"] in (
                                "startup", "continuation", "round_fusion", "elite_fusion"):
                            self.assertEqual(row["code"], candidate.code)
                            self.assertEqual(row["thought"], candidate.thought)
                    for entry in record["trace"]:
                        if entry["event"] == "model_process" and "proposal_code_sha256" in entry["result"]:
                            self.assertNotIn("proposal", entry["result"])
                    if generation:
                        self.assertTrue(any(row["operator"] in ("E1", "E2", "M1", "M2")
                                            and not row["selected"] and "code" not in row
                                            for row in record["candidates"]))
                        self.assertTrue(any(entry["event"] == "model_response" and
                                            entry["stage"].startswith("critic") for entry in record["trace"]))
                        self.assertEqual(sum(entry["event"] == "history_appended" for entry in record["trace"]), 3)
                        expected = {item["candidate_id"]: item["evaluation_origin"]
                                    for item in record["selection"]["selected"]}
                        self.assertEqual({row["candidate_id"]: row["evaluation_origin"]
                                          for row in record["candidates"] if row["selected"]}, expected)
                selected = [item for item in events if item["event"] == "selection"][-1]["selected"]
                for entry, candidate in zip(selected, population):
                    self.assertEqual(original_code(events, method, entry["candidate_id"], self.contract), candidate.code)
                earlier_ids = {row["candidate_id"] for generation in (0, 1)
                               for row in self.record(generation)["candidates"]}
                self.assertTrue(any(item["evaluation_origin"] in earlier_ids for item in selected))
                self.assertEqual((len(backend.calls), len(backend.evaluations)), (61, 49))
                self.assertFalse(any(key.endswith("generation_0003.json") for key in self.files))

    def test_initial_duplicate_records_earliest_completed_evaluation(self):
        backend = RecordingBackend(self.log, self.run_dir)
        backend.duplicate_initial_pair = True
        controller = Controller(backend, self.log, task_contract=self.contract,
                                population_size=2, initial_count=3, rounds=1)
        controller.run()
        rows = {row["candidate_id"]: row for row in self.record(0)["candidates"]}
        pair = [rows["c0001"], rows["c0002"]]
        self.assertEqual(pair[0]["code_sha256"], pair[1]["code_sha256"])
        selected = next(row for row in pair if row["selected"])
        earliest = min(pair, key=lambda row: (row["evaluation"]["completed_monotonic_ns"], row["sequence"]))
        self.assertEqual(selected["evaluation_origin"], earliest["candidate_id"])

    def test_failed_initialization_records_partial_outputs_and_errors(self):
        backend = RecordingBackend(self.log, self.run_dir, invalid={"init_01"}, runtime_invalid={"init_02"})
        controller = Controller(backend, self.log, task_contract=self.contract,
                                population_size=2, initial_count=3, rounds=1)
        with self.assertRaises(StopRun):
            controller.run()
        record = self.record(0)
        rows = {row["stage"]: row for row in record["candidates"]}
        self.assertEqual(record["status"], "stopped")
        self.assertIsNone(record["selection"])
        self.assertIn("invalid syntax", rows["init_01"]["raw_response"])
        self.assertEqual((rows["init_01"]["status"], rows["init_01"]["valid"]), ("invalid_proposal", False))
        self.assertEqual(rows["init_02"]["evaluation"]["status"], "candidate_error")
        self.assertEqual((rows["init_02"]["status"], rows["init_02"]["valid"]), ("candidate_error", False))
        self.assertEqual(rows["init_02"]["instances"][0]["objective"], 7.0)
        self.assertEqual(rows["init_02"]["failure_stderr"], "fixture error")
        self.assertNotIn("fitness", rows["init_02"]["evaluation"])
        self.assertIsNone(rows["init_02"]["selected"])
        self.assertTrue(any(event["event"] == "finish" and event["status"] == "stopped"
                            for event in record["trace"]))

    def test_received_proposal_without_controller_response_is_preserved_on_stop(self):
        self.log.event("candidate_registered", candidate_id="c0001", sequence=1, stage="init_01",
                       source_role="initializer", operator="initialization", generation=0, round=None, parents=[])
        backend = RecordingBackend(self.log, self.run_dir)
        response = backend.model("init_01", "prompt")
        self.log.event("finish", status="stopped", reason="interrupted")
        row = self.record(0)["candidates"][0]
        self.assertEqual(row["code"], self.contract.parse_proposal(response)["code"])
        self.assertNotIn("generation_order", row)
        self.assertNotIn("evaluation", row)
        self.assertEqual((row["status"], row["valid"]), ("incomplete", None))

    def test_failure_during_generation_keeps_previous_record_and_failed_call(self):
        class FailedCriticBackend(RecordingBackend):
            def model(self, stage, prompt):
                if stage == "critic_initial":
                    self.log.event("model_attempt", stage=stage, prompt=prompt)
                    self.log.event("model_process", stage=stage, result={
                        "exit_code": 1, "stdout": "failed request", "stderr": "connection failed"})
                    raise StopRun("fixture connection failure")
                return super().model(stage, prompt)

        backend = FailedCriticBackend(self.log, self.run_dir)
        controller = Controller(backend, self.log, task_contract=self.contract,
                                population_size=2, initial_count=3, rounds=1)
        with self.assertRaises(StopRun):
            controller.run()
        self.assertEqual(self.record(0)["status"], "completed")
        partial = self.record(1)
        self.assertEqual(partial["status"], "stopped")
        self.assertIsNone(partial["selection"])
        failed = next(item for item in partial["trace"]
                      if item["event"] == "model_process" and item["stage"] == "critic_initial")
        self.assertEqual(failed["result"]["stderr"], "connection failed")
        self.assertIn("failed_prompt", failed)
        self.assertTrue(all(row["valid"] is None for row in partial["candidates"]))

    def test_evaluation_file_without_event_is_marked_unconfirmed(self):
        self.log.event("candidate_registered", candidate_id="c0001", sequence=1, stage="init_01",
                       source_role="initializer", operator="initialization", generation=0, round=None, parents=[])
        backend = RecordingBackend(self.log, self.run_dir)
        response = backend.model("init_01", "prompt")
        self.log.event("proposal_response", candidate_id="c0001", stage="init_01", generation_order=1,
                       prompt="prompt", response=response)
        candidate = Candidate("c0001", 1, "init_01", (), code=self.contract.parse_proposal(response)["code"])
        backend.evaluate(candidate)
        self.log.event("finish", status="stopped", reason="interrupted")
        row = self.record(0)["candidates"][0]
        self.assertTrue(row["evaluation_event_missing"])
        self.assertEqual(row["evaluation"]["status"], "passed")
        self.assertIsNone(row["valid"])
        self.assertEqual(row["code"], candidate.code)

    def test_p1_instance_events_preserve_timing_scope_and_stopping_reason(self):
        root = self.run_dir / "candidates" / "c0001"
        events = [{"event": "start", "time_scope": "full_solve", "seed_base": 42},
                  {"event": "instance", "index": 0, "seed": 42, "objective": 2.5,
                   "solve_wall_seconds": 0.75, "stop_reason": "stagnation"},
                  {"event": "complete", "instance_count": 1}]
        with (root / "stdout.txt").open("x", encoding="utf-8") as stream:
            stream.write("\n".join("P1_JSON " + json.dumps(item) for item in events))
        result = {"status": "passed", "objectives": [2.5], "solve_wall_seconds": [0.75],
                  "start": {"time_scope": "full_solve", "seed_base": 42}, "stopping_reasons": ["stagnation"]}
        evidence = _evaluation_evidence(root, result)
        self.assertEqual(evidence["evaluation_start"]["time_scope"], "full_solve")
        self.assertEqual(evidence["instances"][0]["stop_reason"], "stagnation")
        result["start"]["time_scope"] = "incorrect_scope"
        with self.assertRaisesRegex(ValueError, "计时范围"):
            _evaluation_evidence(root, result)

    def test_corrupt_evidence_stops_before_next_generation(self):
        backend = RecordingBackend(self.log, self.run_dir)
        backend.corrupt_stdout = True
        controller = Controller(backend, self.log, task_contract=self.contract,
                                population_size=2, initial_count=3, rounds=1)
        with self.assertRaisesRegex(StopRun, "代次归并失败"):
            controller.run()
        self.assertEqual(backend.generation_checks, 0)
        self.assertNotIn("run/generation_records/generation_0000.json", self.files)
        self.assertTrue(self.log.consolidation_failed)
        self.assertEqual(self.events()[-1]["status"], "stopped")

    def test_existing_generation_record_is_not_overwritten(self):
        target = "run/generation_records/generation_0000.json"
        self.files[target] = "existing record"
        backend = RecordingBackend(self.log, self.run_dir)
        controller = Controller(backend, self.log, task_contract=self.contract,
                                population_size=2, initial_count=3, rounds=1)
        with self.assertRaisesRegex(StopRun, "代次归并失败"):
            controller.run()
        self.assertEqual(self.files[target], "existing record")
        self.assertEqual(backend.generation_checks, 0)

    def test_modified_candidate_code_stops_consolidation(self):
        backend = RecordingBackend(self.log, self.run_dir)
        backend.corrupt_code = True
        controller = Controller(backend, self.log, task_contract=self.contract,
                                population_size=2, initial_count=3, rounds=1)
        with self.assertRaisesRegex(StopRun, "候选代码与提案不一致"):
            controller.run()
        self.assertEqual(backend.generation_checks, 0)
        self.assertNotIn("run/generation_records/generation_0000.json", self.files)


if __name__ == "__main__":
    unittest.main()
