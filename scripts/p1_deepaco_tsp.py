"""通过已验收的 TSP 评价函数运行 DeepACO 网络和项目 ACO 适配参考。"""

from hashlib import sha256
import json
import random
from time import perf_counter

import numpy as np
import torch

from baselines.deepaco.reference import software_versions
from scripts.p1_runtime import DATA_CONTRACT, ROOT


REFERENCE_ID = "deepaco_tsp64_project_aco_adapted"
SCALE = 10.0


def solve(evaluator, positions, model, *, k_sparse, n_iterations):
    """将来源边权等比例送入 P0 评价函数，不改变 P0 文件。"""
    positions = np.asarray(positions).copy()
    n = len(positions) if positions.ndim == 2 else 0
    if (positions.shape != (n, 2) or n < 2 or positions.dtype != np.float64
            or not np.isfinite(positions).all() or getattr(model, "task", None) != "tsp"
            or getattr(model, "dtype", None) != torch.float64
            or type(k_sparse) is not int or not 1 <= k_sparse < n
            or type(n_iterations) is not int or n_iterations <= 0):
        raise ValueError("DeepACO TSP 实例、模型、稀疏度或求解预算无效。")
    scores = model.predict(coordinates=positions, k_sparse=k_sparse)
    if (not isinstance(scores, np.ndarray) or scores.dtype != np.float64 or scores.shape != (n, n)
            or not np.isfinite(scores).all() or (scores < 1e-10).any()):
        raise ValueError("DeepACO TSP 网络输出不是有效的 float64 矩阵。")
    # P0 白盒入口会加 1e-9 再截断；等比例缩放避免改变来源边权的相对概率。
    adjusted = scores * SCALE - 1e-9
    if not np.isfinite(adjusted).all():
        raise ValueError("DeepACO TSP 边权转换后含非有限值。")
    return evaluator.solve(positions, lambda _distances: adjusted, n_iterations)


def run(evaluator, model, *, size, phase, k_sparse):
    if phase == "train" and size == evaluator.PROBLEM_SIZE:
        contract = json.loads(DATA_CONTRACT.read_text(encoding="utf-8"))
        entry = contract["tasks"]["tsp_aco"]["train"]
        dataset = evaluator.DATASET
        if contract.get("status") != "frozen" or dataset.resolve() != (ROOT / entry["path"]).resolve():
            raise ValueError("TSP 训练数据合同或路径无效。")
        expected_hash = entry["sha256"]
        count, iterations, seed_base = evaluator.N_INSTANCES, evaluator.N_ITERATIONS, evaluator.SEED_BASE
        positions = np.load(dataset, allow_pickle=False)
    elif phase == "test" and size in evaluator.TEST_SIZES:
        dataset, positions = evaluator.test_dataset(size)
        expected_hash = evaluator.TEST_DATASET_SHA256[size]
        count, iterations = evaluator.TEST_INSTANCES, evaluator.TEST_ITERATIONS
        seed_base = evaluator.TEST_SEED_BASE + size * 100
    else:
        raise ValueError("训练只支持 TSP50；测试只支持项目登记的 TSP 规模。")
    if (sha256(dataset.read_bytes()).hexdigest() != expected_hash
            or positions.shape != (count, size, 2) or positions.dtype != np.float64
            or not np.isfinite(positions).all() or (positions < 0).any() or (positions >= 1).any()):
        raise ValueError("TSP 实例的哈希、形状或 float64 精度无效。")
    if (getattr(model, "task", None) != "tsp" or getattr(model, "dtype", None) != torch.float64
            or type(k_sparse) is not int or not 1 <= k_sparse < size):
        raise ValueError("DeepACO TSP 模型或稀疏度无效。")
    snapshot = model.source_snapshot()
    prefix = "P1_JSON "
    evaluator.emit({"event": "start", "dataset": str(dataset), "seed_base": seed_base,
                    "n_ants": evaluator.N_ANTS, "n_iterations": iterations, "n_nodes": size,
                    "numpy": np.__version__, "torch": str(torch.__version__),
                    "torch_threads": torch.get_num_threads(),
                    "time_scope": "solve_wall_seconds_including_preparation_and_validation",
                    "reference": {"id": REFERENCE_ID, "checkpoint": model.checkpoint,
                                  "k_sparse": k_sparse, "compute_dtype": "float64",
                                  "heuristic_scale": SCALE, "source_sha256": snapshot,
                                  "software_versions": software_versions("tsp")}}, prefix)
    objectives, solve_times = [], []
    for index, points in enumerate(positions):
        seed = seed_base + index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        started = perf_counter()
        objective, tour, completed = solve(evaluator, points, model, k_sparse=k_sparse, n_iterations=iterations)
        elapsed = perf_counter() - started
        if not np.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("DeepACO TSP 单例求解时间无效。")
        objectives.append(objective)
        solve_times.append(elapsed)
        evaluator.emit({"event": "instance", "index": index, "seed": seed, "objective": objective,
                        "iterations": completed, "tour": tour, "solve_wall_seconds": elapsed}, prefix)
    if sha256(dataset.read_bytes()).hexdigest() != expected_hash or model.source_snapshot() != snapshot:
        raise ValueError("评价期间 TSP 数据或 DeepACO 来源文件发生变化。")
    average = float(np.mean(objectives))
    evaluator.emit({"event": "complete", "instance_count": len(objectives),
                    "average": average, "fitness": -average,
                    "average_solve_wall_seconds": float(np.mean(solve_times))}, prefix)
