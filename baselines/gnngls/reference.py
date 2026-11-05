"""GNNGLS 作者源码和权重的独立 CPU／float64 测试适配。"""

import ast
from functools import lru_cache
from hashlib import sha1, sha256
import importlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parent / "verified"
ISOLATED = Path(r"D:\Tools\agentic-heuristic-p1-dgl")
SOURCE_COMMIT = "05b5aa1b166815f1404d7f01b40b59f6d0554237"
SOURCE_BLOBS = {
    "gnngls/__init__.py": "1ff5039f44811b755429fe68a48f72fa3d78b750",
    "gnngls/algorithms.py": "f57e8fd62e81d4f7befbf65679f17ae478086d91",
    "gnngls/models.py": "a7ed42e27a984a8335460b67324baafd9df72803",
    "gnngls/operators.py": "554e58d562466bf8efc753c6ae06b4818a64b316",
    "gnngls/datasets.py": "ed1dc15a8674867ec30e29db05fdf7293fe8aec0",
    "README.md": "546308fffdf1cb8dc3ae6f1085907b563b94844c",
    "scripts/test.py": "c01713558f4227e563cc58d8f2623762f8e12f9a",
    "scripts/preprocess_dataset.py": "19330ac1222c1b80b8bc7a0461a652ca59947614",
}
ASSET_SHA256 = {
    "models/tsp20/checkpoint_best_val.pt": "e1f2a0511479f249c779908601b3cdd4466052ccc5cca875a29fe3e3c63d2bf3",
    "models/tsp20/params.json": "f4033460123521752ea9706dd7c40405024ff62fdfe382e7686e7f8e33fbcb6c",
    "data/tsp20/scalers.pkl": "71ed6908111e2ace8f2bc1a5b9939bbb7ba21400d0ce29ccea29dd0e2147a0c6",
    "models/tsp50/checkpoint_best_val.pt": "b5e4f43b99ef0b47f1e67bde22adc872ad878cfb86d40a4cbd35711215412089",
    "models/tsp50/params.json": "e4959a8fc2d01b688dd539d5615e64882cafb35690a58fee40a3f4f3ed3c8174",
    "data/tsp50/scalers.pkl": "87b57287cf365f834c912a5d560827f4437295886f4dccaf4173cf28f93640bb",
    "models/tsp100/checkpoint_best_val.pt": "fc8dc46fc9fd3095972a34a2668cead5f69c131aff8db167ccb0a4c5e02b9b2d",
    "models/tsp100/params.json": "99c6a754502dafcf73525753db4da33d353cc53edd769bd23c55a600391bcccf",
    "data/tsp100/scalers.pkl": "74d231253c55469a5ccd75f6f2c2792352111abd3c817dd7ecaaf944d6dc46b7",
}


def verify_sources():
    for name, expected in SOURCE_BLOBS.items():
        data = (ROOT / name).read_bytes()
        if sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() != expected:
            raise ValueError(f"GNNGLS 来源文件哈希不符：{name}")
    for name, expected in ASSET_SHA256.items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"GNNGLS 权重或缩放器哈希不符：{name}")


class StoredMinMaxScaler:
    """只保存来源缩放器的已拟合数值，不执行 sklearn 反序列化逻辑。"""

    def __setstate__(self, state):
        if not isinstance(state, dict):
            raise pickle.UnpicklingError("来源缩放器状态无效。")
        self.__dict__.update(state)


class RestrictedScalerUnpickler(pickle.Unpickler):
    """只允许来源缩放器所需的 NumPy 类型和数值容器。"""

    def find_class(self, module, name):
        allowed = {("sklearn.preprocessing._data", "MinMaxScaler"): StoredMinMaxScaler,
                   ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
                   ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"来源缩放器包含不允许的类型：{module}.{name}")
        return allowed[module, name]


def load_scalers(size):
    if size not in (20, 50, 100):
        raise ValueError("GNNGLS 缩放器规模无效。")
    path = ROOT / f"data/tsp{size}/scalers.pkl"
    data = path.read_bytes()
    if sha256(data).hexdigest() != ASSET_SHA256[path.relative_to(ROOT).as_posix()]:
        raise ValueError("GNNGLS 缩放器哈希不符。")
    scalers = RestrictedScalerUnpickler(io.BytesIO(data)).load()
    if "edges" in scalers:
        scalers = scalers["edges"]
    feature, regret = scalers["features"], scalers["regret"]
    if (feature.n_features_in_ != 15 or regret.n_features_in_ != 1
            or feature.clip or regret.clip or len(feature.scale_) != 15 or len(regret.scale_) != 1):
        raise ValueError("GNNGLS 缩放器结构与来源模型不符。")
    values = tuple(float(value) for value in (feature.scale_[0], feature.min_[0], regret.scale_[0], regret.min_[0]))
    if not all(math.isfinite(value) for value in values) or values[0] <= 0 or values[2] <= 0:
        raise ValueError("GNNGLS 缩放器包含无效参数。")
    versions = {getattr(feature, "_sklearn_version", None), getattr(regret, "_sklearn_version", None)}
    if len(versions) != 1 or None in versions:
        raise ValueError("GNNGLS 缩放器的来源版本无效。")
    return values, versions.pop()


def load_source_modules():
    """只加载推理所需源码，避免导入仅用于生成标签的 Concorde 与 LKH。"""
    import torch
    from torch.utils._import_utils import dill_available
    from torch.utils.data.datapipes.utils import common
    if not hasattr(common, "DILL_AVAILABLE"):
        common.DILL_AVAILABLE = dill_available()
    os.environ.setdefault("DGLBACKEND", "pytorch")
    import dgl

    package_name = "p1_verified_gnngls"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(ROOT / "gnngls")]
        source = ROOT / "gnngls/__init__.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "tour_cost"]
        if len(functions) != 1:
            raise ValueError("GNNGLS 路线成本函数身份不明确。")
        module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
        exec(compile(module, str(source), "exec"), package.__dict__)
        sys.modules[package_name] = package
    algorithms = importlib.import_module(f"{package_name}.algorithms")
    models = importlib.import_module(f"{package_name}.models")
    return torch, dgl, algorithms, models


class ReferenceReached(Exception):
    def __init__(self, tour, objective):
        self.tour, self.objective = tour, objective


@lru_cache(maxsize=9)
def load_model(checkpoint_size, scaler_size):
    torch, dgl, algorithms, models = load_source_modules()
    params = json.loads((ROOT / f"models/tsp{checkpoint_size}/params.json").read_text(encoding="utf-8"))
    if params.get("efeat_drop_idx") != list(range(1, 15)):
        raise ValueError("GNNGLS 来源模型的特征筛选方式发生变化。")
    scales, scaler_version = load_scalers(scaler_size)
    model = models.EdgePropertyPredictionModel(1, params["embed_dim"], 1, params["n_layers"],
                                               n_heads=params["n_heads"]).double().eval()
    checkpoint = torch.load(ROOT / f"models/tsp{checkpoint_size}/checkpoint_best_val.pt",
                            map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return torch, dgl, algorithms, model, scales, scaler_version


@lru_cache(maxsize=4)
def load_line_graph(size):
    """来源在同一规模的数据集内复用线图；节点顺序由 DGL 返回。"""
    _, dgl, _, _ = load_source_modules()
    import networkx as nx

    line_graph = nx.line_graph(nx.complete_graph(size))
    for edge in line_graph.nodes:
        line_graph.nodes[edge]["e"] = edge
    line_graph = dgl.from_networkx(line_graph, node_attrs=["e"])
    edges = tuple(tuple(int(node) for node in edge) for edge in line_graph.ndata["e"].tolist())
    return line_graph, edges


def solve(points, *, checkpoint_size, scaler_size, time_limit, perturbation_moves, reference=None):
    """执行一例；时间上限含特征生成、推理和局部搜索。"""
    points = np.asarray(points)
    if (points.ndim != 2 or points.shape[1] != 2 or points.shape[0] not in (20, 50, 100, 200)
            or points.dtype != np.float64 or not np.isfinite(points).all()
            or (points < 0).any() or (points >= 1).any() or checkpoint_size not in (20, 50, 100)
            or scaler_size not in (20, 50, 100) or isinstance(time_limit, bool)
            or not isinstance(time_limit, (int, float)) or not math.isfinite(time_limit) or time_limit <= 0
            or type(perturbation_moves) is not int or perturbation_moves < 1):
        raise ValueError("GNNGLS 实例、权重、缩放器或预算无效。")
    if reference is not None and (isinstance(reference, bool) or not isinstance(reference, (int, float))
                                  or not math.isfinite(reference) or reference <= 0):
        raise ValueError("GNNGLS LKH 参考无效。")
    verify_sources()
    torch, _, algorithms, model, scales, scaler_version = load_model(checkpoint_size, scaler_size)
    import networkx as nx
    feature_scale, feature_min, regret_scale, regret_min = scales

    graph = nx.complete_graph(len(points))
    for a, b in graph.edges:
        graph.edges[a, b]["weight"] = math.dist(points[a], points[b])
    if any(graph.edges[edge]["weight"] <= 0 for edge in graph.edges):
        raise ValueError("GNNGLS 实例包含重复节点。")

    setup_start = time.monotonic()
    previous_hits = load_line_graph.cache_info().hits
    line_graph, edges = load_line_graph(len(points))
    graph_setup_seconds = time.monotonic() - setup_start
    graph_cache_hit = load_line_graph.cache_info().hits > previous_hits
    started_wall, started_clock = time.time(), time.monotonic()
    features = np.asarray([graph.edges[edge]["weight"] for edge in edges], dtype=np.float64)
    # MinMaxScaler 按列独立缩放；其余 14 列被来源参数删除。
    features = torch.as_tensor((features * feature_scale + feature_min)[:, None], dtype=torch.float64)
    with torch.no_grad():
        prediction = model(line_graph, features).detach().cpu().numpy().reshape(-1)
    prediction = np.maximum((prediction - regret_min) / regret_scale, 0)
    if prediction.shape != (len(edges),) or not np.isfinite(prediction).all():
        raise ValueError("GNNGLS 模型预测无效。")
    for edge, value in zip(edges, prediction):
        graph.edges[edge]["regret_pred"] = float(value)
    initial_tour = algorithms.nearest_neighbor(graph, 0, weight="regret_pred")
    initial_cost = math.fsum(graph.edges[edge]["weight"] for edge in zip(initial_tour[:-1], initial_tour[1:]))
    prediction_seconds = time.monotonic() - started_clock
    threshold = None if reference is None else float(reference) + max(1e-12, 1e-12 * abs(float(reference)))
    source_local_search = algorithms.local_search
    checks = 0

    def checked_local_search(*args, **kwargs):
        nonlocal checks
        result = source_local_search(*args, **kwargs)
        checks += 1
        tour = result[0]
        objective = math.fsum(graph.edges[edge]["weight"] for edge in zip(tour[:-1], tour[1:]))
        if threshold is not None and objective <= threshold:
            raise ReferenceReached(tour, objective)
        return result

    algorithms.local_search = checked_local_search
    try:
        try:
            tour, source_cost, _ = algorithms.guided_local_search(
                graph, initial_tour, initial_cost, started_wall + float(time_limit), weight="weight",
                guides=["regret_pred"], perturbation_moves=perturbation_moves, first_improvement=False)
            stop_reason = "time_limit"
        except ReferenceReached as found:
            tour, source_cost, stop_reason = found.tour, found.objective, "reference_reached"
    finally:
        algorithms.local_search = source_local_search
    if (len(tour) != len(points) + 1 or tour[0] != 0 or tour[-1] != 0
            or sorted(tour[:-1]) != list(range(len(points)))):
        raise ValueError("GNNGLS 路线不是完整闭合回路。")
    objective = math.fsum(math.dist(points[a], points[b]) for a, b in zip(tour[:-1], tour[1:]))
    if not math.isclose(objective, source_cost, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("GNNGLS 路线与来源目标不一致。")
    return {"route": tour[:-1], "objective": objective, "source_cost": source_cost,
            "size": len(points), "checkpoint_size": checkpoint_size, "scaler_size": scaler_size,
            "compute_dtype": "float64", "device": "cpu", "time_limit": float(time_limit),
            "perturbation_moves": perturbation_moves, "stop_reason": stop_reason,
            "iterations": max(0, checks - 1), "prediction_seconds": prediction_seconds,
            "elapsed_seconds": time.monotonic() - started_clock, "graph_setup_seconds": graph_setup_seconds,
            "graph_cache_hit": graph_cache_hit, "scaler_source_sklearn": scaler_version}
