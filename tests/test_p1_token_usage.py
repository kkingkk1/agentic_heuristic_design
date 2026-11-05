import io
import sys
import unittest
from unittest.mock import patch

from scripts import p1_token_usage
from scripts.p1_token_usage import summarize


def event(kind, stage, **fields):
    return {"event": kind, "stage": stage, **fields}


class TestTokenUsage(unittest.TestCase):
    def test_complete_usage_keeps_cached_and_reasoning_as_subsets(self):
        usage = {"input_tokens": 100, "cached_input_tokens": 40,
                 "output_tokens": 20, "reasoning_output_tokens": 3}
        events = [event("model_attempt", "a"), event("model_process", "a"),
                  event("model_usage", "a", usage=usage, warnings=[])]
        result = summarize([events])
        self.assertTrue(result["local_records_complete"])
        self.assertEqual(result["observed_tokens"], usage)
        self.assertEqual(result["attempts_without_usage"], 0)

    def test_failed_attempt_and_missing_fields_remain_unknown(self):
        events = [event("model_attempt", "a"), event("model_process", "a"),
                  event("model_attempt", "b"), event("model_process", "b"),
                  event("model_usage", "b", usage={"input_tokens": 50}, warnings=[])]
        result = summarize([events])
        self.assertFalse(result["local_records_complete"])
        self.assertEqual(result["attempts_without_usage"], 1)
        self.assertEqual(result["observed_tokens"]["input_tokens"], 50)
        self.assertEqual(result["usage_events_missing_field"]["output_tokens"], 1)

    def test_invalid_values_inconsistent_subsets_and_warnings_are_reported(self):
        usage = {"input_tokens": 10, "cached_input_tokens": 11,
                 "output_tokens": True, "reasoning_output_tokens": -1}
        events = [event("model_attempt", "a"), event("model_process", "a"),
                  event("model_usage", "a", usage=usage, warnings=["retry"])]
        result = summarize([events])
        self.assertFalse(result["local_records_complete"])
        self.assertEqual(result["usage_events_invalid_field"]["output_tokens"], 1)
        self.assertEqual(result["usage_events_invalid_field"]["reasoning_output_tokens"], 1)
        self.assertEqual(result["usage_events_with_inconsistent_subsets"], 1)
        self.assertEqual(result["usage_events_with_warnings"], 1)

    def test_events_from_different_logs_do_not_cancel_missing_attempts(self):
        first = [event("model_attempt", "same"), event("model_process", "same")]
        second = [event("model_usage", "same", usage={}, warnings=[])]
        result = summarize([first, second])
        self.assertFalse(result["local_records_complete"])
        self.assertEqual(result["attempts_without_usage"], 1)
        self.assertEqual(result["usage_events_without_process"], 1)

    def test_missing_reasoning_breakdown_is_not_reported_as_complete(self):
        usage = {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 20}
        events = [event("model_attempt", "a"), event("model_process", "a"),
                  event("model_usage", "a", usage=usage, warnings=[])]
        result = summarize([events])
        self.assertFalse(result["local_records_complete"])
        self.assertEqual(result["usage_events_missing_field"]["reasoning_output_tokens"], 1)

    def test_empty_or_unfinished_logs_do_not_claim_complete(self):
        self.assertFalse(summarize([[]])["local_records_complete"])
        result = summarize([[event("model_attempt", "a")]])
        self.assertFalse(result["local_records_complete"])
        self.assertEqual(result["attempt_process_count_mismatches"], 1)

    def test_malformed_event_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "event 类型"):
            summarize([[{"event": {}}]])

    def test_cli_json_output_is_utf8_when_windows_pipe_default_is_not(self):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252")
        usage = {"input_tokens": 10, "cached_input_tokens": 2,
                 "output_tokens": 3, "reasoning_output_tokens": 1}
        events = [event("model_attempt", "a"), event("model_process", "a"),
                  event("model_usage", "a", usage=usage, warnings=[])]
        with patch.object(sys, "stdout", stream), patch.object(sys, "argv", ["token", __file__]), \
             patch.object(p1_token_usage, "read_events", return_value=iter(events)):
            self.assertEqual(p1_token_usage.main(), 0)
        stream.flush()
        self.assertIn("仅统计本地", raw.getvalue().decode("utf-8"))
        stream.detach()


if __name__ == "__main__":
    unittest.main()
