import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from baselines.gnngls import reference


class SourceScalerUnpickler(reference.RestrictedScalerUnpickler):
    def find_class(self, module, name):
        if (module, name) == ("sklearn.preprocessing._data", "MinMaxScaler"):
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler
        return super().find_class(module, name)


torch_spec = importlib.util.find_spec("torch")
isolated_torch = bool(torch_spec and torch_spec.origin
                      and Path(torch_spec.origin).resolve().is_relative_to(reference.ISOLATED.resolve()))


@unittest.skipUnless(isolated_torch, "GNNGLS 源码一致性检查须使用已授权的隔离 PyTorch 环境。")
class TestGNNGLSSourceParity(unittest.TestCase):
    def test_source_feature_conversion_and_model_predictions(self):
        handle = os.add_dll_directory(str(reference.ISOLATED / "Library/bin"))
        self.addCleanup(handle.close)
        reference.verify_sources()
        torch, _, _, models = reference.load_source_modules()
        import networkx as nx

        size = 20
        points = np.random.RandomState(20800020).random_sample((size, 2))
        graph = nx.complete_graph(size)
        for a, b in graph.edges:
            weight = float(np.linalg.norm(points[a] - points[b]))
            graph.edges[a, b]["features"] = np.r_[weight, np.zeros(14)].astype(np.float32)
            graph.edges[a, b]["regret"] = 0.0
            graph.edges[a, b]["in_solution"] = False
        line_graph, edges = reference.load_line_graph(size)
        package = sys.modules["p1_verified_gnngls"]
        with patch.object(package, "fixed_edge_tour", create=True), patch.object(package, "optimal_cost", create=True):
            source_dataset = importlib.import_module("p1_verified_gnngls.datasets")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scalers = SourceScalerUnpickler(
                io.BytesIO((reference.ROOT / "data/tsp20/scalers.pkl").read_bytes())).load()["edges"]
        dataset = source_dataset.TSPDataset.__new__(source_dataset.TSPDataset)
        dataset.G, dataset.scalers, dataset.feat_drop_idx = line_graph, scalers, list(range(1, 15))
        source_features = dataset.get_scaled_features(graph).ndata["features"]
        scales, _ = reference.load_scalers(size)
        values = np.asarray([graph.edges[edge]["features"][0] for edge in edges], dtype=np.float64)
        adapted_features = torch.as_tensor((values * scales[0] + scales[1])[:, None], dtype=torch.float64)
        np.testing.assert_allclose(source_features.numpy(), adapted_features.numpy(), rtol=1e-6, atol=1e-7)

        params = json.loads((reference.ROOT / "models/tsp20/params.json").read_text(encoding="utf-8"))
        source_model = models.EdgePropertyPredictionModel(1, params["embed_dim"], 1, params["n_layers"],
                                                          n_heads=params["n_heads"]).float().eval()
        checkpoint = torch.load(reference.ROOT / "models/tsp20/checkpoint_best_val.pt", map_location="cpu", weights_only=True)
        source_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        _, _, _, adapted_model, _, _ = reference.load_model(size, size)
        with torch.no_grad():
            source = source_model(line_graph, source_features).numpy()
            adapted = adapted_model(line_graph, adapted_features).numpy()
        np.testing.assert_allclose(source, adapted, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
