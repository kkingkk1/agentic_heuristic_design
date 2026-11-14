import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import p1_freeze_candidate as freeze
from scripts.p1_task_contract import load_task_contract


class TestCandidateFreeze(unittest.TestCase):
    def test_formal_preflight_needs_no_subscription_gate_but_still_requires_start_record(self):
        method = "proposed_solution_quality_and_speed_agents"
        config = {"method": {"id": method}, "matrix_contract_sha256": "matrix",
                  "formal_replicate": 1,
                  "task": {"name": "tsp_aco", "prompt_condition": "white_box", "selection_seed": 1}}
        preflight = {"event": "preflight", "code_sha256": {}}
        started = {"event": "formal_run_started", "matrix_contract_sha256": "matrix",
                   "task": "tsp_aco", "prompt_condition": "white_box", "method": method,
                   "formal_replicate": 1, "selection_seed": 1}
        events = [preflight, started]

        def read(path, *args, **kwargs):
            if path.name == "config.json":
                return json.dumps(config)
            if path.name == "events.jsonl":
                return "\n".join(json.dumps(event) for event in events)
            raise AssertionError(f"unexpected read: {path}")

        directory = freeze.ROOT / "outputs/p1_control/fixture_no_quota_gate"
        with patch.object(Path, "read_text", autospec=True, side_effect=read), \
             patch.dict(freeze.METHODS, {method: ("p1_control", lambda _config: None)}), \
             patch.object(freeze, "verify_training_config"), \
             patch.object(freeze, "final_candidate", side_effect=ValueError("reached final candidate")) as final:
            with self.assertRaisesRegex(ValueError, "reached final candidate"):
                freeze.freeze(directory)
            final.assert_called_once()
            events[:] = [preflight]
            with self.assertRaisesRegex(ValueError, "启动记录不一致"):
                freeze.freeze(directory)
            events[:] = [started]
            with self.assertRaisesRegex(ValueError, "启动检查"):
                freeze.freeze(directory)

    def test_each_method_reads_its_actual_final_selection(self):
        for method in ("proposed_solution_quality_and_speed_agents", "roco_paper_adapted", "eoh_paper_adapted"):
            events = [{"event": "selection", "selected": [{"candidate_id": "c0007"}]},
                      {"event": "finish", "status": "completed"}]
            self.assertEqual(freeze.final_candidate(events, method), "c0007")
        for method in ("reevo_2f70042_adapted", "hsevo_fb569d6_adapted"):
            self.assertEqual(freeze.final_candidate([{"event": "finish", "status": "completed", "elite": "c0002"}], method), "c0002")
        self.assertEqual(freeze.final_candidate([{"event": "finish", "status": "completed", "best": "c0003"}],
                                                "mcts_ahd_ee9c4f4_adapted"), "c0003")
        for events in ([], [{"event": "finish", "status": "stopped"}],
                       [{"event": "finish", "status": "completed", "best": "../other"}]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                freeze.final_candidate(events, "mcts_ahd_ee9c4f4_adapted")

    def test_proposal_and_harmony_code_must_come_from_logged_source(self):
        contract = load_task_contract("tsp_aco")
        code = "import numpy as np\n" + contract.signature + "\n    return np.ones_like(distance_matrix)"
        registration = {"event": "candidate_registered", "candidate_id": "c0001", "stage": "init_01"}
        for method, response in (("proposed_solution_quality_and_speed_agents", f"<thought>x</thought><code>{code}</code>"),
                                 ("proposed_solution_quality_and_speed_agents", f"<thought>x</thought>\n{code}"),
                                 ("roco_paper_adapted", f"<thought>x</thought><code>{code}</code>"),
                                 ("roco_paper_adapted", f"<thought>x</thought>\n{code}"),
                                 ("eoh_paper_adapted", "{idea}\n" + code),
                                 ("mcts_ahd_ee9c4f4_adapted", "Useful ideas.\n{idea}\n" + code),
                                 ("reevo_2f70042_adapted", "```python\n" + code + "\n```"),
                                 ("hsevo_fb569d6_adapted", "```python\n" + code + "\n```")):
            events = [registration, {"event": "proposal_response", "candidate_id": "c0001", "response": response}]
            with self.subTest(method=method):
                self.assertEqual(freeze.original_code(events, method, "c0001", contract), code)
                with self.assertRaises(ValueError):
                    freeze.original_code(events + [events[-1]], method, "c0001", contract)
        events = [{"event": "candidate_registered", "candidate_id": "c0002", "stage": "harmony_01"},
                  {"event": "harmony_candidate", "candidate_id": "c0002", "code": code}]
        self.assertEqual(freeze.original_code(events, "hsevo_fb569d6_adapted", "c0002", contract), code)

    def test_seed_is_taken_from_fixed_source_and_no_engineering_run_is_accepted(self):
        contract = load_task_contract("tsp_aco", black_box=True)
        events = [{"event": "candidate_registered", "candidate_id": "c0001", "stage": "seed"}]
        code = freeze.original_code(events, "reevo_2f70042_adapted", "c0001", contract)
        self.assertIn(contract.signature, code)
        with self.assertRaises(ValueError):
            freeze.freeze(freeze.ROOT / "outputs/p1_runtime_checks/20261104_reevo_op_black_seed_01")


if __name__ == "__main__":
    unittest.main()
