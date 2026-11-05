from io import BytesIO
import unittest

import numpy as np

from scripts import p1_generate_test_datasets as generator


class TestCandidateDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = generator.build_datasets()

    def test_all_source_datasets_match_git_blobs(self):
        for key, expected in generator.SOURCE_BLOBS.items():
            with self.subTest(dataset=key):
                self.assertEqual(generator.git_blob(self.datasets[key]), expected)

    def test_written_files_match_generator_and_expected_shapes(self):
        expected_sizes = {
            "op_aco": (50, 100, 200, 500),
            "cvrp_aco": (50, 100, 200, 500),
            "mkp_aco": (100, 200, 300, 500, 700),
            "bpp_offline_aco": (500, 700, 1000),
        }
        self.assertEqual(set(self.datasets), {(task, size) for task, sizes in expected_sizes.items() for size in sizes})
        for (task, size), content in self.datasets.items():
            with self.subTest(task=task, size=size):
                self.assertEqual(generator.dataset_path(task, size).read_bytes(), content)
                values = np.load(BytesIO(content), allow_pickle=False)
                if task == "op_aco":
                    self.assertEqual(values["coordinates"].shape, (64, size, 2))
                elif task == "cvrp_aco":
                    self.assertEqual(values.shape, (64, size + 1, 3))
                    np.testing.assert_array_equal(values[:, 0, 0], np.zeros(64))
                elif task == "mkp_aco":
                    self.assertEqual(values["prizes"].shape, (64, size))
                    self.assertEqual(values["weights"].shape, (64, size, 5))
                else:
                    self.assertEqual(values["demands"].shape, (64, size))
                    self.assertTrue(((values["demands"] >= 20) & (values["demands"] <= 100)).all())
                if hasattr(values, "close"):
                    values.close()

    def test_generator_refuses_to_overwrite_any_existing_file(self):
        with self.assertRaisesRegex(FileExistsError, "不覆盖"):
            generator.main()
        for key, content in self.datasets.items():
            self.assertEqual(generator.dataset_path(*key).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
