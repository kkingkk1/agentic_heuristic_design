import json
import unittest
from unittest.mock import patch

from scripts import p1_freeze_data as data


class TestP1FrozenData(unittest.TestCase):
    def test_frozen_contract_matches_current_data_and_solver_parameters(self):
        frozen = json.loads(data.CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(data.build_contract(), frozen)
        self.assertEqual(frozen["status"], "frozen")
        self.assertEqual(len(frozen["tasks"]), 6)
        self.assertEqual(sum(len(task["test"]) for task in frozen["tasks"].values()), 23)
        self.assertTrue(all(test["instances"] == 64 for task in frozen["tasks"].values() for test in task["test"]))

    def test_changed_data_or_seed_rejected(self):
        real_identity = data.file_identity

        def altered(path):
            result = real_identity(path)
            if path.name == "train50_dataset.npy":
                result["sha256"] = "0" * 64
            return result

        with patch.object(data, "file_identity", side_effect=altered), self.assertRaisesRegex(ValueError, "哈希"):
            data.build_contract()
        real_constants = data.test_constants

        def wrong_seed(name):
            result = real_constants(name)
            if name == "op_aco":
                result["TEST_SEED_BASE"] = 1
            return result

        with patch.object(data, "test_constants", side_effect=wrong_seed), self.assertRaisesRegex(ValueError, "随机种子"):
            data.build_contract()


if __name__ == "__main__":
    unittest.main()
