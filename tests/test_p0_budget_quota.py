import json
import unittest

from scripts import p0_budget_quota as quota


class QuotaSnapshotTests(unittest.TestCase):
    def test_snapshot_keeps_percent_and_removes_account_identifiers(self):
        bucket = {
            "limitId": "codex",
            "primary": {"usedPercent": 95, "windowDurationMins": 10080, "resetsAt": 123},
            "secondary": None,
            "credits": {"hasCredits": False, "balance": "0"},
        }
        result = quota.sanitize_limits({
            "accountId": "private-account-id",
            "ordinaryUsageAllowed": True,
            "rateLimits": bucket,
            "rateLimitsByLimitId": {"codex": bucket},
            "rateLimitResetCredits": {"availableCount": 1, "credits": [{"id": "private-reset-id"}]},
        })
        self.assertEqual(result["buckets"][0]["primary"]["used_percent"], 95)
        self.assertEqual(result["available_reset_credit_count"], 1)
        self.assertIsNone(result["remaining_model_calls"])
        self.assertFalse(result["model_specific_bucket_confirmed"])
        self.assertNotIn("private-", json.dumps(result))

    def test_missing_or_invalid_windows_fail_closed(self):
        invalid = [
            {},
            {"rateLimits": {}},
            {"rateLimits": {"primary": {"usedPercent": 101}}},
            {"rateLimits": {"primary": {"usedPercent": True}}},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(quota.QuotaProbeError):
                quota.sanitize_limits(value)

    def test_legacy_single_bucket_is_supported(self):
        result = quota.sanitize_limits({"rateLimits": {
            "limitId": "codex", "primary": {"usedPercent": 3}, "secondary": None,
        }})
        self.assertEqual(result["buckets"][0]["limit_id"], "codex")
        self.assertIsNone(result["buckets"][0]["primary"]["duration_minutes"])


if __name__ == "__main__":
    unittest.main()
