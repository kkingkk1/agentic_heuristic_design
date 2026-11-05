import hashlib
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.deepaco.reference import DeepACOReference


ROOT = Path(__file__).resolve().parents[1] / "baselines/deepaco/verified"
SOURCE_BLOBS = {
    "tsp/aco.py": "817d991334e132d792a0d958e4e6a606e36403b0",
    "op/aco.py": "22bf42ff9f094f3045d8af609c7d3ee80b51f943",
    "cvrp/aco.py": "c2fa4fe9de8e6b8114c8446db31c8555a15c63e0",
    "mkp/aco.py": "ccf3bda6ad2101f6802025e481151ebb60da2052",
    "bpp/aco.py": "3865830d68d5f8efe0ac9827461a3d54482a4e4e",
    "pretrained/tsp/tsp100.pt": "1537b3773867383066b6739638f8ec5ce0e6f912",
    "pretrained/tsp/tsp20.pt": "bac843d797d1955c8e9a68c3780101dbb75af489",
    "pretrained/tsp/tsp500.pt": "5d3aa2b73df06099e69eb9db79d207d30888f390",
    "pretrained/op/op100.pt": "5096b9d5da972e2aff290484346c23b445b27f86",
    "pretrained/op/op200.pt": "17270b7cf860a859b8bfd97189715e87736193ed",
    "pretrained/op/op300.pt": "9c81e4f9b9bc620fff105de367d521f97595f16e",
    "pretrained/cvrp/cvrp100.pt": "490fa8e24a7690c29d56c8b83d7af88460bed4ae",
    "pretrained/cvrp/cvrp20.pt": "4bbbe4de117666a9352770da520207bb870dcf51",
    "pretrained/cvrp/cvrp500.pt": "24cb14ee5f19945414f807a0cc6905a7d99d1350",
    "pretrained/mkp/mkp300.pt": "3c2cada9e651ead37440b290e80463704105c7c8",
    "pretrained/mkp/mkp500.pt": "92b70fae786956e4a51ff7ee15c73d2d1523e235",
    "pretrained/bpp/bpp120.pt": "567a7dcac9a1ca173d3cca91500ad39594048aaa",
    "tsp/net.py": "657cdaf042ca4ebf81c755cbc572489b2167c730",
    "tsp/utils.py": "a2fb237f23f76f94902d802899171cfd711f28a7",
    "op/net.py": "c6cb8d15a4c52b5e4911e3a561bdd31eec9ddacc",
    "op/utils.py": "721f82afd4dc6b7b7f31e4f5575996a28a3ffec8",
    "cvrp/net.py": "03304dc0535a3fb9799976efd37354f034b71289",
    "cvrp/utils.py": "ccccaf2cd79c2bb390ba6071fd391272585b0e81",
    "mkp/net.py": "4d1b4f37ff91a73a0a443182898e68b672509635",
    "mkp/utils.py": "886ce7e208583f4cffc07a903a05245f136c2388",
    "bpp/net.py": "03304dc0535a3fb9799976efd37354f034b71289",
    "bpp/utils.py": "992f592be347b7ec8d23525b5100f59cf9fc4029",
    "LICENSE.txt": "242a799b18c283c6afe3415f46fa1c6173cec3d2",
}


class TestDeepACOSources(unittest.TestCase):
    def test_official_source_and_checkpoint_blob_identities(self):
        for name, expected in SOURCE_BLOBS.items():
            with self.subTest(name=name):
                content = (ROOT / name).read_bytes()
                actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
                self.assertEqual(actual, expected)

    def test_checkpoint_is_tensor_state_dictionary_loaded_without_pickle_objects(self):
        for path in sorted((ROOT / "pretrained").rglob("*.pt")):
            with self.subTest(path=path):
                state = torch.load(path, map_location="cpu", weights_only=True)
                self.assertTrue(state)
                self.assertTrue(all(isinstance(key, str) and isinstance(value, torch.Tensor)
                                    for key, value in state.items()))
                self.assertTrue(all(torch.isfinite(value).all().item() for value in state.values()))
                self.assertTrue(any(key.startswith("emb_net.") for key in state))
                self.assertTrue(any(key.startswith("par_net_heu.") for key in state))

    def test_all_networks_load_and_infer_with_simulated_graph_primitives(self):
        # 只检查权重、数据流和形状；真实 PyG 兼容性必须在安装后另行验证。
        pyg, pyg_data, pyg_nn = ModuleType("torch_geometric"), ModuleType("torch_geometric.data"), ModuleType("torch_geometric.nn")

        class Data:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class BatchNorm(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.module = torch.nn.BatchNorm1d(width)

            def forward(self, value):
                return self.module(value)

        def global_mean_pool(value, index):
            count = int(index.max().item()) + 1
            sums = value.new_zeros((count, value.shape[-1]))
            sums.index_add_(0, index, value)
            divisor = value.new_zeros(count)
            divisor.index_add_(0, index, value.new_ones(len(index)))
            return sums / divisor.clamp(min=1).unsqueeze(-1)

        pyg_data.Data = Data
        pyg_nn.BatchNorm, pyg_nn.global_mean_pool = BatchNorm, global_mean_pool
        pyg.data, pyg.nn = pyg_data, pyg_nn
        rng = np.random.RandomState(7451)
        points, demands, prizes, weights = rng.rand(20, 2), rng.randint(1, 10, 20), rng.rand(20), rng.rand(20, 5) / 20
        cases = (("tsp", "tsp100", {"coordinates": points, "k_sparse": 5}, 20),
                 ("op", "op100", {"coordinates": points, "k_sparse": 5}, 20),
                 ("cvrp", "cvrp100", {"coordinates": points, "demands": demands}, 20),
                 ("mkp", "mkp300", {"prize": prizes, "weight": weights}, 20),
                 ("bpp", "bpp120", {"demands": demands}, 21))
        modules = {"torch_geometric": pyg, "torch_geometric.data": pyg_data, "torch_geometric.nn": pyg_nn}
        with patch.dict(sys.modules, modules):
            for task, checkpoint, kwargs, size in cases:
                with self.subTest(task=task):
                    model = DeepACOReference(task, checkpoint)
                    first, second = model.predict(**kwargs), model.predict(**kwargs)
                    self.assertEqual(first.shape, (size, size))
                    self.assertEqual(first.dtype, np.float64)
                    self.assertTrue(all(p.dtype == torch.float64 for p in model.model.parameters()))
                    self.assertTrue(np.isfinite(first).all())
                    np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
