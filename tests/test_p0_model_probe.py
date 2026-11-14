import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import p0_model_probe as probe


VALID_CODE = """import numpy as np
def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    return np.ones_like(distance_matrix)
"""


class ProtocolTests(unittest.TestCase):
    def test_proposal_accepts_complete_module_and_keeps_content(self):
        result = probe.parse_proposal(f"<thought>Uniform scores.</thought>\n<code>{VALID_CODE}</code>")
        self.assertEqual(result["thought"], "Uniform scores.")
        self.assertEqual(result["code"], VALID_CODE.strip())

    def test_proposal_rejects_wrong_interface_and_incomplete_response(self):
        for text in (
            f"<thought>Idea.</thought><code>{VALID_CODE.replace('distance_matrix', 'dist')}</code>",
            f"<thought>Idea.</thought><code>{VALID_CODE}",
            f"Extra< thought><thought>Idea.</thought><code>{VALID_CODE}</code>",
            "<thought>Idea.</thought><code>import numpy as np\n"
            "def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:\n"
            "    def helper():\n        return distance_matrix\n    pass\n</code>",
        ):
            with self.subTest(text=text), self.assertRaises(probe.ProbeError):
                probe.parse_proposal(text)

    def test_critic_keeps_full_feedback_and_requires_tags(self):
        reflection = " ".join(f"reflection{i}" for i in range(22))
        advice = " ".join(f"advice{i}" for i in range(45))
        parsed = probe.parse_critic(f"<ref>{reflection}</ref><ans>{advice}</ans>")
        self.assertEqual(parsed, {"reflection": reflection, "advice": advice})
        for invalid in ("<ans>Missing reflection.</ans>", "<ref> </ref><ans>Advice.</ans>"):
            with self.subTest(invalid=invalid), self.assertRaises(probe.ProbeError):
                probe.parse_critic(invalid)

    def test_long_term_and_merge_protocols(self):
        self.assertEqual(probe.parse_long_term("<ans>Retain useful hints.</ans>"), "Retain useful hints.")
        self.assertEqual(probe.parse_merge("Combine both insights."), "Combine both insights.")
        long_reflection = " ".join(f"word{i}" for i in range(55))
        self.assertEqual(probe.parse_long_term(f"<ans>{long_reflection}</ans>"), long_reflection)
        with self.assertRaises(probe.ProbeError):
            probe.parse_long_term("<ans> </ans>")
        with self.assertRaises(probe.ProbeError):
            probe.parse_merge("<ans>Unexpected wrapper.</ans>")

    def test_events_require_completion_and_reject_tool_actions(self):
        def result(*events):
            return {"exit_code": 0, "stdout": "\n".join(json.dumps(event) for event in events)}

        message = {"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}}
        completed = {"type": "turn.completed", "usage": {"input_tokens": 10}}
        self.assertEqual(probe.extract_message(result(message, completed))[:2], ("answer", {"input_tokens": 10}))
        with self.assertRaises(probe.ProbeError):
            probe.extract_message(result(message))
        with self.assertRaises(probe.ProbeError):
            probe.extract_message(result(message, {"type": "turn.failed"}, completed))
        with self.assertRaises(probe.ProbeError):
            probe.extract_message(result(message, {"type": "item.completed", "item": {"type": "command_execution"}}, completed))
        with self.assertRaises(probe.ProbeError):
            probe.extract_message(result({"type": "item.started", "item": {"type": "command_execution"}}, message, completed))
        retry = {"type": "error", "message": "Reconnecting... 2/5 (request timed out)"}
        fallback = {"type": "item.completed", "item": {"type": "error", "message": "Falling back from WebSockets to HTTPS transport. request timed out"}}
        self.assertEqual(len(probe.extract_message(result(retry, fallback, message, completed))[3]), 2)
        network_retry = {"type": "error", "message": "Reconnecting... waiting for network (Connection failed: error sending request)"}
        self.assertEqual(probe.extract_message(result(network_retry, message, completed))[3], [network_retry["message"]])
        with self.assertRaises(probe.ProbeError):
            probe.extract_message(result(network_retry, message))
        with self.assertRaises(probe.ProbeError):
            probe.extract_message(result(network_retry, message, {"type": "turn.failed"}))
        with self.assertRaises(probe.ProbeError):
            probe.extract_message(result({"type": "error", "message": "model unavailable"}, message, completed))

    def test_context_comparison_preserves_structure_without_instruction_text(self):
        prompt_input = [
            {"role": "developer", "content": [{"type": "input_text", "text": "Private AGENTS.md skills instructions"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "test prompt"}]},
        ]
        catalog = {"models": [{"slug": probe.MODEL, "context_window": 123, "truncation_policy": {"mode": "tokens", "limit": 42}}]}
        replies = [
            {"exit_code": 0, "stdout": json.dumps(prompt_input)},
            {"exit_code": 0, "stdout": json.dumps(catalog)},
        ]
        with patch.object(probe, "run_command", side_effect=replies):
            audit = probe.audit_context(["codex"], Path("."), "test prompt")
        self.assertTrue(audit["explicit_prompt_present_in_comparison"])
        self.assertTrue(audit["prompt_input_parts"][0]["mentions_agents_md"])
        self.assertEqual(audit["bundled_model_catalog"]["context_window"], 123)
        self.assertNotIn("Private AGENTS.md skills instructions", json.dumps(audit))

    def test_call_uses_chatgpt_https_without_changing_model_or_auth(self):
        message = {"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}}
        completed = {"type": "turn.completed", "usage": {"input_tokens": 10}}
        result = {"exit_code": 0, "stdout": "\n".join(json.dumps(event) for event in (message, completed))}
        with patch.object(probe, "run_command", return_value=result) as runner, patch.object(probe, "append_record"):
            self.assertEqual(probe.call(["codex"], Path("record"), Path("run"), "proposal", "prompt"), "answer")
        command, prompt = runner.call_args.args
        self.assertEqual(prompt, "prompt")
        self.assertIn("--ignore-user-config", command)
        self.assertNotIn("--ephemeral", command)
        self.assertNotIn("resume", command)
        self.assertEqual(command[command.index("--model") + 1], "gpt-5.6-luna")
        self.assertIn('model_providers.p0_chatgpt_https.base_url="https://chatgpt.com/backend-api/codex"', command)
        self.assertIn("model_providers.p0_chatgpt_https.requires_openai_auth=true", command)
        self.assertIn("model_providers.p0_chatgpt_https.supports_websockets=false", command)
        self.assertFalse(any("reasoning_effort" in part or "env_key" in part for part in command))

    def test_call_records_and_rejects_unexpected_websocket_fallback(self):
        events = [
            {"type": "error", "message": "Reconnecting... 2/5 (request timed out)"},
            {"type": "item.completed", "item": {"type": "error", "message": "Falling back from WebSockets to HTTPS transport. request timed out"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}},
            {"type": "turn.completed", "usage": {"input_tokens": 10}},
        ]
        result = {"exit_code": 0, "stdout": "\n".join(json.dumps(event) for event in events)}
        with patch.object(probe, "run_command", return_value=result), patch.object(probe, "append_record") as recorder:
            with self.assertRaisesRegex(probe.ProbeError, "HTTPS 配置未生效"):
                probe.call(["codex"], Path("record"), Path("run"), "proposal", "prompt")
        self.assertEqual([call.args[1]["event"] for call in recorder.call_args_list], ["prompt", "process", "response"])


if __name__ == "__main__":
    unittest.main()
