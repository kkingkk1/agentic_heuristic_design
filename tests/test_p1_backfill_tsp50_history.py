import json
import unittest
from unittest.mock import patch

from scripts import p1_backfill_tsp50_history as backfill
from scripts.p0_controller import code_hash


def source(index, name="outputs/example/events.jsonl"):
    return {"path": name, "event_index": index}


class VirtualPath:
    """只用于核验目标冲突的内存路径。"""

    def __init__(self, files, name):
        self.files, self.value = files, name.rstrip("/")

    def __truediv__(self, name):
        return VirtualPath(self.files, self.value + "/" + str(name))

    def __eq__(self, other):
        return isinstance(other, VirtualPath) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    @property
    def parent(self):
        return VirtualPath(self.files, self.value.rsplit("/", 1)[0])

    def resolve(self):
        return self

    def exists(self):
        return self.value in self.files

    def is_file(self):
        return self.value in self.files

    def read_bytes(self):
        return self.files[self.value]

    def __str__(self):
        return self.value


class FakeCandidatePath:
    def __init__(self, files, name="candidate/c0001"):
        self.files, self.value = files, name

    def __truediv__(self, name):
        return FakeCandidatePath(self.files, self.value + "/" + str(name))

    @property
    def name(self):
        return self.value.rsplit("/", 1)[-1]

    def resolve(self):
        return self

    def is_relative_to(self, _other):
        return True

    def is_file(self):
        return self.value in self.files

    def read_text(self, **_kwargs):
        return self.files[self.value]

    def __str__(self):
        return self.value


class TestHistoricalBackfill(unittest.TestCase):
    def test_manual_recovery_sources_are_marked(self):
        events = [
            {"event": "candidate_registered", "event_index": 1, "source_event": source(1),
             "candidate_id": "c0079", "sequence": 79, "stage": "fusion_2_manual_retry",
             "generation": 1, "round": 2, "parents": ["c0076", "c0077"]},
            {"event": "reused_or_retried_response", "event_index": 2, "source_event": source(2),
             "candidate_id": "c0082", "sequence": 82, "stage": "round_3_solution_quality_agent_manual_retry",
             "parents": ["c0076"], "generation_order": 80, "response": "original response"},
            {"event": "branch_transition", "event_index": 3, "source_event": source(3),
             "proposal": "c0082", "previous": "c0076", "generation": 1, "round": 3,
             "role": "solution_quality_agent"},
            {"event": "evaluation", "event_index": 4, "source_event": source(4), "candidate_id": "c0082"},
        ]
        rows = backfill.registrations(events, "proposed_solution_quality_and_speed_agents", 1)
        self.assertEqual((rows["c0079"]["source_role"], rows["c0079"]["operator"]),
                         ("integrator", "round_fusion"))
        self.assertEqual(rows["c0079"]["historical_derivations"]["source_role"], "stage_and_fusion_contract")
        self.assertIsNone(rows["c0082"]["source_events"]["registration"])
        self.assertEqual(rows["c0082"]["source_events"]["branch_transition"], source(3))
        self.assertEqual((rows["c0082"]["generation"], rows["c0082"]["round"]), (1, 3))
        events[2]["previous"] = "c9999"
        with self.assertRaisesRegex(ValueError, "身份与分支记录不符"):
            backfill.registrations(events, "proposed_solution_quality_and_speed_agents", 1)

    def test_phase_boundaries_handle_reused_population_and_reject_conflict(self):
        rows = {"c0001": {"generation": 0}}
        events = [{"event": "candidate_registered", "event_index": 1, "generation": 0},
                  {"event": "initial_population", "event_index": 2, "candidate_ids": ["c0001"]},
                  {"event": "initial_population", "event_index": 3, "candidate_ids": ["c0001"]}]
        for generation in range(1, 7):
            events.extend([{"event": "selection", "event_index": generation * 10, "generation": generation},
                           {"event": "history_appended", "event_index": generation * 10 + 1,
                            "generation": generation, "role": "Integrator"}])
        boundaries, selected = backfill.phase_boundaries(events, rows)
        self.assertEqual(boundaries, [2, 11, 21, 31, 41, 51, 61])
        self.assertEqual(selected[0]["event_index"], 2)
        conflicting = [dict(event) for event in events]
        conflicting[0]["generation"] = 1
        with self.assertRaisesRegex(ValueError, "事件代次与筛选边界不一致"):
            backfill.phase_boundaries(conflicting, rows)

    def test_earlier_skip_does_not_override_later_evaluation(self):
        code = "import numpy as np\ndef heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:\n    return np.ones_like(distance_matrix)"
        response = f"<thought>idea</thought><code>{code}</code>"
        result = {"candidate_id": "c0001", "status": "passed", "average": 1.0, "fitness": -1.0,
                  "command": ["python", "-B", "-u", "eval.py", "50", "candidate/c0001", "train"]}
        files = {"candidate/c0001/result.json": json.dumps(result),
                 "candidate/c0001/problems/tsp_aco/gpt.py": code + "\n"}
        candidate = FakeCandidatePath(files)
        events = [{"event": "candidate_skipped", "event_index": 1, "candidate_id": "c0001",
                   "source_event": source(1)},
                  {"event": "proposal_response", "event_index": 2, "candidate_id": "c0001",
                   "stage": "init_01", "response": response, "generation_order": 1, "source_event": source(2)},
                  {"event": "evaluation", "event_index": 3, "candidate_id": "c0001",
                   "result": result, "source_event": source(3)}]
        rows = {"c0001": {"candidate_id": "c0001", "source_events": {}}}
        with patch.object(backfill, "Path", return_value=candidate), \
                patch.object(backfill, "relative", return_value="outputs/fake/c0001"), \
                patch.object(backfill, "_evaluation_evidence", return_value={"instances": []}):
            parsed, evaluations = backfill.proposal_and_evaluation(events, rows)
        self.assertEqual((rows["c0001"]["status"], rows["c0001"]["valid"]), ("evaluation_passed", True))
        self.assertEqual(rows["c0001"]["issues"][0]["event"], "candidate_skipped")
        self.assertEqual(parsed["c0001"]["code"], code)
        self.assertEqual(evaluations["c0001"]["source_event"], source(3))
        events[0]["event_index"] = 4
        with patch.object(backfill, "Path", return_value=candidate), \
                patch.object(backfill, "relative", return_value="outputs/fake/c0001"), \
                patch.object(backfill, "_evaluation_evidence", return_value={"instances": []}):
            with self.assertRaisesRegex(ValueError, "顺序或类型冲突"):
                backfill.proposal_and_evaluation(events, {"c0001": {"candidate_id": "c0001", "source_events": {}}})

    def test_trace_compacts_success_and_keeps_failed_prompt(self):
        code = "import numpy as np\ndef heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:\n    return np.ones_like(distance_matrix)"
        response = f"<thought>idea</thought><code>{code}</code>"
        completed = [{"type": "item.completed", "item": {"type": "agent_message", "text": response}},
                     {"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 3}}]
        events = [
            {"event": "model_attempt", "event_index": 1, "stage": "init_01", "prompt": "prompt",
             "source_event": source(1)},
            {"event": "model_process", "event_index": 2, "stage": "init_01", "source_event": source(2),
             "result": {"exit_code": 0, "stdout": "\n".join(map(json.dumps, completed)), "stderr": ""}},
            {"event": "proposal_response", "event_index": 3, "stage": "init_01", "response": response,
             "source_event": source(3)},
            {"event": "model_attempt", "event_index": 4, "stage": "critic_1", "prompt": "failed prompt",
             "source_event": source(4)},
            {"event": "model_process", "event_index": 5, "stage": "critic_1", "source_event": source(5),
             "result": {"exit_code": 1, "stdout": "failure", "stderr": "connection failed"}},
        ]
        trace = backfill.compact_trace(events)
        self.assertEqual(trace[0]["prompt_sha256"], code_hash("prompt"))
        self.assertNotIn("prompt", trace[0])
        self.assertEqual(trace[1]["result"]["proposal_code_sha256"], code_hash(code))
        self.assertNotIn("stdout", trace[1]["result"])
        self.assertNotIn("proposal", trace[1]["result"])
        self.assertEqual(trace[-1]["failed_prompt"], "failed prompt")
        self.assertEqual(trace[-1]["source_event"], source(5))
        self.assertEqual(trace[-1]["result"]["stderr"], "connection failed")

    def test_target_preflight_never_overwrites_different_content(self):
        files = {}
        prepared = [(None, VirtualPath(files, f"root/run{number}"),
                     [{"generation": generation} for generation in range(7)], None, None)
                    for number in range(6)]
        payloads = backfill.target_payloads(prepared)
        self.assertEqual(len(payloads), 42)
        path, data = payloads[0]
        files[path.value] = data
        self.assertEqual(len(backfill.target_payloads(prepared)), 42)
        files[path.value] = b"existing different content"
        with self.assertRaisesRegex(ValueError, "目标已存在且内容不同"):
            backfill.target_payloads(prepared)
        self.assertEqual(files[path.value], b"existing different content")


if __name__ == "__main__":
    unittest.main()
