"""DeepACO 固定来源的预训练网络推理；不把矩阵接口冒充物品向量。"""

from hashlib import sha1, sha256
import importlib.metadata
import importlib.util
from pathlib import Path
import platform

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent / "verified"
CHECKPOINT_BLOBS = {
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
}
CODE_BLOBS = {
    "tsp/aco.py": "817d991334e132d792a0d958e4e6a606e36403b0",
    "tsp/net.py": "657cdaf042ca4ebf81c755cbc572489b2167c730",
    "tsp/utils.py": "a2fb237f23f76f94902d802899171cfd711f28a7",
    "op/aco.py": "22bf42ff9f094f3045d8af609c7d3ee80b51f943",
    "op/net.py": "c6cb8d15a4c52b5e4911e3a561bdd31eec9ddacc",
    "op/utils.py": "721f82afd4dc6b7b7f31e4f5575996a28a3ffec8",
    "cvrp/aco.py": "c2fa4fe9de8e6b8114c8446db31c8555a15c63e0",
    "cvrp/net.py": "03304dc0535a3fb9799976efd37354f034b71289",
    "cvrp/utils.py": "ccccaf2cd79c2bb390ba6071fd391272585b0e81",
    "mkp/aco.py": "ccf3bda6ad2101f6802025e481151ebb60da2052",
    "mkp/net.py": "4d1b4f37ff91a73a0a443182898e68b672509635",
    "mkp/utils.py": "886ce7e208583f4cffc07a903a05245f136c2388",
    "bpp/aco.py": "3865830d68d5f8efe0ac9827461a3d54482a4e4e",
    "bpp/net.py": "03304dc0535a3fb9799976efd37354f034b71289",
    "bpp/utils.py": "992f592be347b7ec8d23525b5100f59cf9fc4029",
}
TASKS = frozenset({"tsp", "op", "cvrp", "mkp", "bpp"})


def software_versions(task):
    packages = ("torch-geometric", "pyparsing", "xxhash")
    if task == "bpp":
        packages += ("numba", "llvmlite")
    if task in ("tsp", "cvrp"):
        packages += ("scipy",)
    try:
        versions = {name: importlib.metadata.version(name) for name in packages}
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError(f"DeepACO 运行依赖缺失：{exc.name}") from exc
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__), **versions}


def verify_blob(path, expected):
    data = path.read_bytes()
    actual = sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if actual != expected:
        raise ValueError(f"DeepACO 来源文件哈希不符：{path}")


def load_source(task, filename):
    name = f"{task}/{filename}.py"
    path = ROOT / name
    verify_blob(path, CODE_BLOBS[name])
    spec = importlib.util.spec_from_file_location(f"deepaco_{task}_{filename}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeepACOReference:
    """预训练网络输出原方法的边矩阵；调用方须明确权重与稀疏度。"""

    def __init__(self, task, checkpoint):
        if task not in TASKS:
            raise ValueError("未知 DeepACO 任务。")
        name = f"pretrained/{task}/{checkpoint}.pt"
        if name not in CHECKPOINT_BLOBS or not checkpoint.startswith(task):
            raise ValueError("权重不属于已核验的任务及版本。")
        path = ROOT / name
        verify_blob(path, CHECKPOINT_BLOBS[name])
        self.task, self.checkpoint, self.checkpoint_path = task, checkpoint, path
        self.network, self.utils = load_source(task, "net"), load_source(task, "utils")
        self.model = self.network.Net()
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        # 所有任务统一以 float64 计算；权重值仍来自原始 float32 检查点。
        self.dtype = torch.float64
        self.model.to(dtype=self.dtype)
        self.model.eval()

    def source_snapshot(self):
        paths = [self.checkpoint_path, *(ROOT / self.task / f"{name}.py" for name in ("aco", "net", "utils")),
                 Path(__file__).resolve()]
        verify_blob(self.checkpoint_path, CHECKPOINT_BLOBS[self.checkpoint_path.relative_to(ROOT).as_posix()])
        for name in ("aco", "net", "utils"):
            verify_blob(ROOT / self.task / f"{name}.py", CODE_BLOBS[f"{self.task}/{name}.py"])
        return {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}

    def predict(self, *, coordinates=None, demands=None, prize=None, weight=None, k_sparse=None):
        if self.task in ("tsp", "op"):
            coordinates = np.asarray(coordinates)
            if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] < 2
                    or not np.isfinite(coordinates).all() or type(k_sparse) is not int
                    or not 1 <= k_sparse < coordinates.shape[0]):
                raise ValueError("坐标或稀疏邻居数无效。")
            if coordinates.dtype != np.float64:
                raise ValueError(f"DeepACO {self.task.upper()} 必须直接接收项目的 float64 原始坐标。")
            points = torch.as_tensor(coordinates, dtype=self.dtype)
            graph = self.utils.gen_pyg_data(points, k_sparse)[0]
            if self.task == "op" and prize is not None:
                canonical = torch.as_tensor(np.asarray(prize), dtype=self.dtype)
                if (canonical.shape != (len(points),) or not torch.isfinite(canonical).all()
                        or not torch.equal(graph.x[:, 1], canonical)):
                    raise ValueError("DeepACO OP 图中的收益与项目评价实例不一致。")
                graph.x[:, 1] = canonical
            expected_size = len(points)
        elif self.task == "cvrp":
            coordinates, demands = np.asarray(coordinates), np.asarray(demands)
            if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or demands.shape != (len(coordinates),)
                    or coordinates.dtype != np.float64
                    or (demands.dtype.kind not in "iu" and demands.dtype != np.float64)
                    or len(coordinates) < 2 or not np.isfinite(coordinates).all() or not np.isfinite(demands).all()):
                raise ValueError("CVRP 坐标或需求无效。")
            points = torch.as_tensor(coordinates, dtype=self.dtype)
            demand = torch.as_tensor(demands, dtype=self.dtype)
            graph = self.utils.gen_pyg_data(demand, self.utils.gen_distance_matrix(points), "cpu")
            expected_size = len(points)
        elif self.task == "mkp":
            prize, weight = np.asarray(prize), np.asarray(weight)
            if (prize.ndim != 1 or len(prize) < 2 or weight.shape != (len(prize), 5)
                    or prize.dtype != np.float64 or weight.dtype != np.float64
                    or not np.isfinite(prize).all() or not np.isfinite(weight).all()
                    or (prize < 0).any() or (weight < 0).any()):
                raise ValueError("MKP 价值或五维归一化重量无效。")
            prizes = torch.as_tensor(prize, dtype=self.dtype)
            # DeepACO 的官方生成器将同一归一化容量从 1 缩放为 n//2。
            weights = torch.as_tensor(weight, dtype=self.dtype) * (len(prize) // 2)
            graph = self.utils.gen_pyg_data(prizes, weights)
            expected_size = len(prize)
        else:
            demands = np.asarray(demands)
            if (demands.ndim != 1 or len(demands) < 1 or not np.isfinite(demands).all()
                    or (demands.dtype.kind not in "iu" and demands.dtype != np.float64)
                    or (demands < 0).any()):
                raise ValueError("BPP 需求无效。")
            demand = torch.cat((torch.zeros(1, dtype=self.dtype), torch.as_tensor(demands, dtype=self.dtype)))
            graph = self.utils.gen_pyg_data(demand, "cpu")
            graph.edge_attr = graph.edge_attr.to(dtype=self.dtype)
            expected_size = len(demand)
        with torch.no_grad():
            vector = self.model(graph)
            if self.task in ("tsp", "op"):
                # 稀疏图按来源 reshape 的边索引恢复；显式保留 float64。
                matrix = torch.zeros((expected_size, expected_size), dtype=vector.dtype, device=vector.device)
                matrix[graph.edge_index[0], graph.edge_index[1]] = vector
                matrix = matrix + 1e-10
            else:
                # 来源的稠密图测试直接按边列表顺序 reshape，不按 edge_index 回填。
                matrix = vector.reshape((expected_size, expected_size))
                if self.task == "mkp":
                    matrix = matrix / (matrix.min() + 1e-10) + 1e-10
                else:
                    matrix = matrix + 1e-10
        values = matrix.detach().cpu().numpy()
        if values.shape != (expected_size, expected_size) or not np.isfinite(values).all():
            raise ValueError("DeepACO 网络输出的形状或数值无效。")
        return values
