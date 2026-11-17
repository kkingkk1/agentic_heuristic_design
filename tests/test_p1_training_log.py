import json
import threading
import unittest
from unittest.mock import patch

from baselines.roco.paper_controller import RoCoController
from scripts.p0_controller import Controller, code_hash
from scripts.p0_runtime import JsonLog
from scripts.p1_freeze_candidate import original_code
from scripts.p1_runtime import TrainingLog
from scripts.p1_task_contract import load_task_contract
from tests.test_p0_controller import FixtureBackend


class TestTrainingLog(unittest.TestCase):
    def setUp(self):
        self.contract = load_task_contract("tsp_aco")
        self.code = "import numpy as np\n" + self.contract.signature + "\n    return np.ones_like(distance_matrix)"
        self.response = f"<thought>idea</thought><code>{self.code}</code>"
        self.log = TrainingLog.__new__(TrainingLog)
        self.log.contract = self.contract
        self.log.pending_prompts = {}
        self.log.prompt_lock = threading.Lock()
        self.events = []
        self.original_event = patch.object(JsonLog, "event", autospec=True, side_effect=self.record)
        self.original_event.start()
        self.addCleanup(self.original_event.stop)

    def record(self, _log, kind, **fields):
        self.events.append({"event": kind, **fields})

    @staticmethod
    def process(message):
        rows = [{"type": "item.completed", "item": {"type": "agent_message", "text": message}},
                {"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 3}}]
        return {"exit_code": 0, "stdout": "\n".join(map(json.dumps, rows)), "stderr": "",
                "elapsed_seconds": 1.5}

    def test_compact_candidate_stays_freezable_for_both_methods(self):
        self.log.event("candidate_registered", candidate_id="c0001", stage="init_01")
        self.log.event("model_attempt", stage="init_01", prompt="private repeated prompt")
        process_result = self.process(self.response)
        self.log.event("model_process", stage="init_01", result=process_result)
        self.log.event("proposal_response", candidate_id="c0001", stage="init_01", generation_order=1,
                       prompt="private repeated prompt", response=self.response)
        self.assertNotIn("prompt", self.events[1])
        self.assertEqual(self.events[1]["prompt_sha256"], code_hash("private repeated prompt"))
        self.assertIn("stdout", process_result)
        self.assertNotIn("stdout", self.events[2]["result"])
        self.assertEqual(self.events[2]["result"]["proposal"], {"thought": "idea", "code": self.code})
        self.assertEqual(self.events[2]["result"]["usage"], {"input_tokens": 7, "output_tokens": 3})
        self.assertNotIn("response", self.events[3])
        self.assertNotIn("prompt", self.events[3])
        for method in ("proposed_solution_quality_and_speed_agents", "roco_paper_adapted"):
            with self.subTest(method=method):
                self.assertEqual(original_code(self.events, method, "c0001", self.contract), self.code)
        self.events[3]["code_sha256"] = "tampered"
        with self.assertRaisesRegex(ValueError, "不一致"):
            original_code(self.events, "roco_paper_adapted", "c0001", self.contract)

    def test_failed_process_and_invalid_proposal_keep_diagnostics(self):
        failure = {"exit_code": 1, "stdout": '{"type":"turn.failed"}', "stderr": "connection failed"}
        self.log.event("model_attempt", stage="init_01", prompt="failed request prompt")
        self.log.event("model_process", stage="init_01", result=failure)
        self.log.event("proposal_response", candidate_id="c0001", stage="init_01",
                       prompt="repeated", response="unparseable proposal")
        self.assertEqual(self.events[1]["result"], failure)
        self.assertEqual(self.events[1]["failed_prompt"], "failed request prompt")
        self.assertEqual(self.events[2]["response"], "unparseable proposal")
        self.assertNotIn("prompt", self.events[2])

    def test_feedback_and_completed_process_survive_interruption(self):
        feedback = "<quality_ref>reason</quality_ref><quality_ans>advice</quality_ans>"
        self.log.event("model_process", stage="critic_1", result=self.process(feedback))
        self.log.event("model_response", stage="critic_1", generation=1, prompt="repeated", response=feedback)
        self.assertEqual(self.events[0]["result"]["message"], feedback)
        self.assertEqual(self.events[1]["response"], feedback)
        self.assertNotIn("prompt", self.events[1])
        self.log.event("model_process", stage="init_02", result=self.process(self.response))
        self.assertEqual(self.events[-1]["result"]["proposal"]["code"], self.code)

    def test_two_generation_controller_selection_uses_recorded_proposals(self):
        outer = self

        class RecordingBackend(FixtureBackend):
            def model(self, stage, prompt):
                outer.log.event("model_attempt", stage=stage, prompt=prompt)
                response = super().model(stage, prompt)
                outer.log.event("model_process", stage=stage, result=outer.process(response))
                return response

        for controller_class, method in ((Controller, "proposed_solution_quality_and_speed_agents"),
                                         (RoCoController, "roco_paper_adapted")):
            with self.subTest(method=method):
                self.events.clear()
                controller = controller_class(RecordingBackend(), self.log, task_contract=self.contract,
                                              population_size=2, initial_count=3, rounds=2, generations=2)
                population = controller.run()
                selected = [event for event in self.events if event["event"] == "selection"][-1]["selected"]
                for row, candidate in zip(selected, population):
                    self.assertEqual(original_code(self.events, method, row["candidate_id"], self.contract),
                                     candidate.code)
                self.assertTrue(any(event["event"] == "model_response" and event["stage"].startswith("critic")
                                    and event.get("response") for event in self.events))


if __name__ == "__main__":
    unittest.main()
