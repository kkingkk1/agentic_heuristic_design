"""TSP-GLS 训练与候选测试评价；测试达到 LKH 参考时提前停止。"""

from hashlib import sha256
import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
from scipy.spatial import distance_matrix

from gls import guided_local_search, guided_local_search_until_reference
from lkh_reference import (ABS_TOLERANCE, REL_TOLERANCE, load_reference,
                           reached_reference, reference_digest, stop_threshold)


DATASET = Path(__file__).resolve().parent / "dataset" / "train200_dataset.npy"
DATASET_SHA256 = "f8da974c2a9e935b86c05991f0ed722d01bd1e3a4a18ddde65c381b224086235"
N_INSTANCES = 10
N_NODES = 200
N_ITERATIONS = 1200
PERTURBATION_MOVES = 30
SEED_BASE = 20261103
TEST_SEED_BASE = 2026110400
TEST_PARAMETERS = {20: (5, 73), 50: (30, 175), 100: (40, 1800), 200: (40, 800)}
TEST_DATASET_SHA256 = {
    20: "8f6673ec5c99a3bc40dc5d09e12c190eb2c52f3465f59fca2588e45aca4a9a59",
    50: "d1890e17f00b625f01047e03c4aa7d200d30cb3e2645592f261c15e44094aa11",
    100: "7910b44860ec93d340d3490d0993fcae2a8ee14a587547f753c7720b6be29e28",
    200: "daa71baff8beeee06f5cf5500eb07da62f754e1fb3945d9ffb2e7cdf15a3bbcc",
}
EVENT_PREFIX = "P1_JSON "


def load_heuristic(root):
    candidate_dir = Path(root).resolve() / "problems" / "tsp_gls"
    candidate = candidate_dir / "gpt.py"
    if not candidate.is_file():
        raise ValueError(f"缺少候选文件：{candidate}")
    spec = importlib.util.spec_from_file_location("gpt", candidate)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法导入候选文件：{candidate}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(candidate_dir))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    heuristic = getattr(module, "heuristics_v2", None)
    if not inspect.isfunction(heuristic):
        raise ValueError("候选必须定义函数 heuristics_v2。")
    signature = inspect.signature(heuristic)
    parameters = list(signature.parameters.values())
    if (len(parameters) != 1 or parameters[0].name != "distance_matrix"
            or parameters[0].kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
            or parameters[0].default is not inspect.Parameter.empty
            or parameters[0].annotation not in (np.ndarray, "np.ndarray")
            or signature.return_annotation not in (np.ndarray, "np.ndarray")):
        raise ValueError("候选签名与 distance_matrix NumPy 数组接口不一致。")
    return heuristic


def checked_heuristic(heuristic, distances):
    scores = heuristic(distances.copy())
    if not isinstance(scores, np.ndarray) or scores.shape != distances.shape:
        raise ValueError("启发式必须返回与距离矩阵同形状的 NumPy 数组。")
    if scores.dtype.kind not in "iuf" or not np.isfinite(scores).all():
        raise ValueError("启发式必须返回有限实数。")
    with np.errstate(over="raise", invalid="raise"):
        scores = scores.astype(np.float64, copy=True)
    if not np.isfinite(scores).all():
        raise ValueError("启发式转换后含非有限值。")
    return scores


def solve(positions, heuristic, n_iterations=N_ITERATIONS, perturbation_moves=PERTURBATION_MOVES, reference=None):
    positions = np.asarray(positions)
    if positions.dtype != np.float64:
        raise ValueError("TSP-GLS 必须直接使用 float64 原始坐标。")
    distances = distance_matrix(positions, positions) + np.eye(len(positions)) * 1e-5
    if not np.isfinite(distances).all() or (distances <= 0).any():
        raise ValueError("实例包含重复节点或无效距离。")
    scores = checked_heuristic(heuristic, distances)
    if reference is None:
        tour = guided_local_search(distances, scores, perturbation_moves, n_iterations)
        completed_iterations = n_iterations
    else:
        tour, completed_iterations = guided_local_search_until_reference(
            distances, scores, perturbation_moves, n_iterations, stop_threshold(reference))
    if (not isinstance(tour, np.ndarray) or tour.shape != (len(positions),)
            or tour.dtype.kind not in "iu" or not np.array_equal(np.sort(tour), np.arange(len(positions)))):
        raise ValueError("GLS 输出不是完整城市排列。")
    objective = float(distances[tour, np.roll(tour, 1)].sum())
    if not math.isfinite(objective) or objective <= 0:
        raise ValueError("GLS 路线长度无效。")
    if reference is not None and completed_iterations < n_iterations and not reached_reference(objective, reference):
        raise ValueError("提前停止的路线未达到 float64 参考阈值。")
    return objective, tour.astype(np.int64).tolist(), completed_iterations


def emit(event):
    print(EVENT_PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def candidate_hash(root):
    return sha256((Path(root).resolve() / "problems" / "tsp_gls" / "gpt.py").read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 4:
        raise ValueError("用法：eval.py <规模> <候选任务运行根目录> <train|test>")
    size_text, root, phase = sys.argv[1:]
    if phase == "train" and size_text == "200":
        size, count, iterations, moves, seed_base = N_NODES, N_INSTANCES, N_ITERATIONS, PERTURBATION_MOVES, SEED_BASE
        dataset, expected_hash = DATASET, DATASET_SHA256
    elif phase == "test" and size_text in {str(size) for size in TEST_PARAMETERS}:
        size, count = int(size_text), 64
        moves, iterations = TEST_PARAMETERS[size]
        seed_base = TEST_SEED_BASE + size * 100
        dataset = DATASET.parent / f"test{size}_p1_dataset.npy"
        expected_hash = TEST_DATASET_SHA256[size]
    else:
        raise ValueError("训练只支持 TSP200；测试支持 TSP20、50、100、200。")
    if sha256(dataset.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("实例集哈希不匹配。")
    instances = np.load(dataset, allow_pickle=False)
    if (instances.shape != (count, size, 2) or instances.dtype != np.float64
            or not np.isfinite(instances).all() or (instances < 0).any() or (instances >= 1).any()):
        raise ValueError("实例集内容不符合 TSP-GLS 合同。")
    reference = load_reference(size, expected_hash) if phase == "test" else None
    code_hash = candidate_hash(root) if phase == "test" else None
    heuristic = load_heuristic(root)
    start = {"event": "start", "dataset": str(dataset), "seed_base": seed_base, "n_nodes": size,
             "n_iterations": iterations, "perturbation_moves": moves, "numpy": np.__version__}
    if code_hash is not None:
        start["candidate_sha256"] = code_hash
        start.update({"lkh_reference_sha256": reference["sha256"], "lkh_reference_mean": reference["mean"],
                      "lkh_absolute_tolerance": ABS_TOLERANCE, "lkh_relative_tolerance": REL_TOLERANCE})
    emit(start)
    objectives = []
    for index, positions in enumerate(instances):
        seed = seed_base + index
        random.seed(seed)
        np.random.seed(seed)
        reference_value = reference["objectives"][index] if reference is not None else None
        objective, tour, completed_iterations = solve(positions, heuristic, iterations, moves, reference_value)
        objectives.append(objective)
        event = {"event": "instance", "index": index, "seed": seed, "objective": objective,
                 "iterations": completed_iterations, "tour": tour}
        if reference is not None:
            event.update({"lkh_reference": reference_value,
                          "stop_reason": "reference_reached" if reached_reference(objective, reference_value)
                          else "iteration_limit"})
        emit(event)
    if sha256(dataset.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("评价期间实例集发生变化；不生成完成事件。")
    if code_hash is not None and candidate_hash(root) != code_hash:
        raise ValueError("评价期间候选代码发生变化；不生成完成事件。")
    if reference is not None and reference_digest() != reference["sha256"]:
        raise ValueError("评价期间 LKH 参考发生变化；不生成完成事件。")
    average = float(np.mean(objectives))
    complete = {"event": "complete", "instance_count": len(objectives), "average": average, "fitness": -average}
    if reference is not None:
        complete.update({"lkh_reference_mean": reference["mean"],
                         "lkh_gap_percent": (average / reference["mean"] - 1.0) * 100.0})
    emit(complete)


if __name__ == "__main__":
    main()
