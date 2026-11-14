"""CVRP-ACO 训练与候选测试评价；核对路线、容量和实际路径长度。"""

from hashlib import sha256
import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np
from scipy.spatial import distance_matrix
import torch

from aco import ACO


DATASET = Path(__file__).resolve().parent / "dataset" / "train50_dataset.npy"
DATASET_SHA256 = "caf38e123b53da4fefda9a988da5f029267d78daf0995d2b68721f6ca8b46452"
N_INSTANCES = 10
N_CUSTOMERS = 50
N_ANTS = 20
N_ITERATIONS = 100
CAPACITY = 50
SEED_BASE = 20261103
EVENT_PREFIX = "P1_JSON "
TEST_ITERATIONS = 500
TEST_SEED_BASE = 2026110400
TEST_DATASET_SHA256 = {
    50: "3370bac48fa37c6d6f23c122a3803e9500fdaaaa57de20c519fe90f57bfd7fa8",
    100: "7fcf40e6bee3d9b798513a92017eab592539f8226529b696b95475a47a43167d",
    200: "2ba909893eeb440b8cea16591deac6bde5b11ce45fad6057d5931495dca46c79",
    500: "d35936a260f554cfe8b75ceb8f0118a0a8b9974a0bbc123cf3f1c8fa2a099c8a",
}


def load_heuristic(root, *, black_box=False):
    candidate_dir = Path(root).resolve() / "problems" / "cvrp_aco"
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
    names = ("edge_attr", "node_attr") if black_box else ("distance_matrix", "coordinates", "demands", "capacity")
    annotations = (np.ndarray, np.ndarray) if black_box else (np.ndarray, np.ndarray, np.ndarray, int)
    if (len(parameters) != len(names) or tuple(p.name for p in parameters) != names
            or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD or p.default is not inspect.Parameter.empty
                   or p.annotation not in (kind, "int" if kind is int else "np.ndarray")
                   for p, kind in zip(parameters, annotations))
            or signature.return_annotation not in (np.ndarray, "np.ndarray")):
        raise ValueError(f"候选签名与 {names} 不一致。")
    return heuristic


def checked_heuristic(heuristic, distances, coordinates, demands, *, black_box=False):
    if black_box:
        scores = heuristic(distances.copy(), demands.copy() / CAPACITY)
    else:
        scores = heuristic(distances.copy(), coordinates.copy(), demands.copy(), CAPACITY)
    if not isinstance(scores, np.ndarray) or scores.shape != distances.shape:
        raise ValueError("启发式必须返回与距离矩阵同形状的 NumPy 数组。")
    if scores.dtype.kind not in "iuf" or not np.isfinite(scores).all():
        raise ValueError("启发式必须返回有限实数。")
    with np.errstate(over="raise", invalid="raise"):
        scores = scores.astype(np.float64, copy=False) + 1e-9
    if not np.isfinite(scores).all():
        raise ValueError("启发式转换后含非有限值。")
    scores[scores < 1e-9] = 1e-9
    return scores


def check_path(path, coordinates, demands):
    path = list(path)
    n = len(demands)
    if (len(path) < 2 or path[0] != 0 or path[-1] != 0
            or any(type(node) is not int or node < 0 or node >= n for node in path)):
        raise ValueError("路线的端点或节点编号无效。")
    if sorted(node for node in path if node != 0) != list(range(1, n)):
        raise ValueError("路线未恰好访问每个客户一次。")
    used = 0.0
    for node in path:
        if node == 0:
            used = 0.0
        else:
            used += float(demands[node])
            if used > CAPACITY + 1e-12:
                raise ValueError("路线超过车辆容量。")
    return math.fsum(math.hypot(*(coordinates[a] - coordinates[b])) for a, b in zip(path, path[1:]))


class CheckedACO(ACO):
    """沿用来源构造与信息素更新，逐轮核对每条路线和真实路长。"""

    def __init__(self, distances, demands, heuristic, coordinates):
        self.source_coordinates = coordinates.copy()
        self.source_demands = demands.copy()
        self.checked_iterations = 0
        super().__init__(distances, demands, heuristic, CAPACITY, n_ants=N_ANTS)

    def gen_path(self):
        paths = super().gen_path()
        if (paths.ndim != 2 or paths.shape[1] != self.n_ants
                or paths.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError("蚂蚁路线的形状或类型无效。")
        for path in paths.T.detach().cpu().tolist():
            check_path(path, self.source_coordinates, self.source_demands)
        self.checked_iterations += 1
        return paths

    def gen_path_costs(self, paths):
        costs = super().gen_path_costs(paths)
        if costs.shape != (self.n_ants,) or not torch.isfinite(costs).all() or not (costs > 0).all():
            raise ValueError("蚂蚁路线代价无效。")
        for actual, path in zip(costs.detach().cpu().tolist(), paths.T.detach().cpu().tolist()):
            expected = check_path(path, self.source_coordinates, self.source_demands)
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("蚂蚁路径长度与坐标重算不一致。")
        return costs


def solve(coordinates, demands, heuristic, n_iterations=N_ITERATIONS, *, black_box=False):
    distances = distance_matrix(coordinates, coordinates)
    if not np.isfinite(distances).all():
        raise ValueError("距离矩阵含非有限值。")
    np.fill_diagonal(distances, 1)
    if (distances <= 0).any():
        raise ValueError("实例含重复节点或无效距离。")
    scores = checked_heuristic(heuristic, distances, coordinates, demands, black_box=black_box)
    # 来源框架用仓库零填充已结束蚂蚁；仓库自环的真实距离为零。
    cost_distances = distances.copy()
    np.fill_diagonal(cost_distances, 0)
    aco = CheckedACO(cost_distances, demands, scores, coordinates)
    objective = float(aco.run(n_iterations).item())
    if aco.checked_iterations != n_iterations or aco.shortest_path is None:
        raise ValueError("求解未完成全部迭代。")
    path = aco.shortest_path.detach().cpu().tolist()
    expected = check_path(path, coordinates, demands)
    if not math.isfinite(objective) or objective <= 0 or not math.isclose(objective, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("最优路线目标无效或与坐标重算不一致。")
    return objective, path, aco.checked_iterations


def solve_deepaco(coordinates, demands, model, *, n_iterations=N_ITERATIONS):
    """使用来源 DeepACO，并按项目的容量和路线长度检查结果。"""
    from baselines.deepaco.reference import load_source

    coordinates, demands = np.asarray(coordinates).copy(), np.asarray(demands).copy()
    if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2
            or coordinates.dtype != np.float64 or demands.dtype != np.float64
            or demands.shape != (len(coordinates),) or not np.isfinite(coordinates).all()
            or not np.isfinite(demands).all() or demands[0] != 0
            or (demands[1:] <= 0).any() or (demands[1:] > CAPACITY).any()
            or getattr(model, "task", None) != "cvrp"
            or type(n_iterations) is not int or n_iterations <= 0):
        raise ValueError("DeepACO CVRP 实例、模型或求解预算无效。")
    distances = distance_matrix(coordinates, coordinates)
    if not np.isfinite(distances).all():
        raise ValueError("CVRP 距离矩阵含非有限值。")
    np.fill_diagonal(distances, 1)
    if (distances <= 0).any():
        raise ValueError("CVRP 实例含重复节点或无效距离。")
    np.fill_diagonal(distances, 0)
    scores = model.predict(coordinates=coordinates, demands=demands)
    if (not isinstance(scores, np.ndarray) or scores.dtype != np.float64
            or scores.shape != distances.shape or not np.isfinite(scores).all() or (scores <= 0).any()):
        raise ValueError("DeepACO CVRP 网络输出不是有效的 float64 矩阵。")
    source = load_source("cvrp", "aco")

    class CheckedDeepACO(source.ACO):
        def __init__(self):
            super().__init__(torch.from_numpy(distances.copy()), torch.from_numpy(demands.copy()),
                             n_ants=N_ANTS, heuristic=torch.from_numpy(scores.copy()),
                             device="cpu", capacity=CAPACITY)
            self.checked_iterations = 0

        def gen_path(self, require_prob=False):
            if require_prob:
                raise ValueError("正式参考仅支持来源求解，不支持训练采样。")
            paths = super().gen_path(require_prob=False)
            if paths.ndim != 2 or paths.shape[1] != self.n_ants:
                raise ValueError("DeepACO CVRP 蚂蚁路线数量无效。")
            for path in paths.T.detach().cpu().tolist():
                check_path(path, coordinates, demands)
            self.checked_iterations += 1
            return paths

        def gen_path_costs(self, paths):
            costs = super().gen_path_costs(paths)
            if (costs.shape != (self.n_ants,) or costs.dtype != torch.float64
                    or not torch.isfinite(costs).all() or not (costs > 0).all()):
                raise ValueError("DeepACO CVRP 路线代价无效。")
            for actual, path in zip(costs.detach().cpu().tolist(), paths.T.detach().cpu().tolist()):
                expected = check_path(path, coordinates, demands)
                if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("DeepACO CVRP 路长与项目评价不一致。")
            return costs

    solver = CheckedDeepACO()
    objective = float(solver.run(n_iterations).item())
    if solver.checked_iterations != n_iterations or solver.shortest_path is None:
        raise ValueError("DeepACO CVRP 未完成全部迭代。")
    path = solver.shortest_path.detach().cpu().tolist()
    expected = check_path(path, coordinates, demands)
    if not math.isfinite(objective) or not math.isclose(objective, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("DeepACO CVRP 最优路长与项目评价不一致。")
    return objective, path, solver.checked_iterations


def emit(event):
    print(EVENT_PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def candidate_hash(root):
    return sha256((Path(root).resolve() / "problems" / "cvrp_aco" / "gpt.py").read_bytes()).hexdigest()


def main(*, black_box=False, deepaco_model=None):
    if len(sys.argv) != 4:
        raise ValueError("用法：eval.py <规模> <候选任务运行根目录> <train|test>")
    size_text, root, phase = sys.argv[1:]
    if phase == "train" and size_text == "50":
        size, count, iterations, seed_base = N_CUSTOMERS, N_INSTANCES, N_ITERATIONS, SEED_BASE
        dataset, expected_hash = DATASET, DATASET_SHA256
    elif phase == "test" and size_text in {str(size) for size in TEST_DATASET_SHA256}:
        size, count, iterations = int(size_text), 64, TEST_ITERATIONS
        seed_base = TEST_SEED_BASE + size * 100
        dataset = DATASET.parent / f"test{size}_p1_dataset.npy"
        expected_hash = TEST_DATASET_SHA256[size]
    else:
        raise ValueError("训练只支持 CVRP50；测试支持 CVRP50、100、200、500。")
    if sha256(dataset.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("实例集哈希不匹配。")
    instances = np.load(dataset, allow_pickle=False)
    if (instances.shape != (count, size + 1, 3) or instances.dtype != np.float64
            or not np.isfinite(instances).all() or not (instances[:, 0, 0] == 0).all()
            or not (instances[:, 0, 1:] == 0.5).all()
            or (instances[:, 1:, 0] < 1).any() or (instances[:, 1:, 0] > 9).any()):
        raise ValueError("实例集内容不符合 CVRP 合同。")
    if deepaco_model is None:
        code_hash = candidate_hash(root) if phase == "test" else None
        heuristic = load_heuristic(root, black_box=black_box)
        reference_snapshot = None
    else:
        if (black_box or getattr(deepaco_model, "task", None) != "cvrp"
                or getattr(deepaco_model, "dtype", None) != torch.float64):
            raise ValueError("DeepACO CVRP 模型或提示条件无效。")
        code_hash, heuristic = None, None
        reference_snapshot = deepaco_model.source_snapshot()
    start = {"event": "start", "dataset": str(dataset), "seed_base": seed_base, "n_ants": N_ANTS,
             "n_iterations": iterations, "n_customers": size, "capacity": CAPACITY,
             "numpy": np.__version__, "torch": torch.__version__, "torch_threads": torch.get_num_threads()}
    if phase == "train":
        start["time_scope"] = "solve_wall_seconds_including_preparation_and_validation"
    if code_hash is not None:
        start["candidate_sha256"] = code_hash
    if deepaco_model is not None:
        from baselines.deepaco.reference import software_versions
        start["reference"] = {"id": "deepaco_cvrp64_project_eval", "checkpoint": deepaco_model.checkpoint,
                              "compute_dtype": "float64", "source_sha256": reference_snapshot,
                              "software_versions": software_versions("cvrp")}
    emit(start)
    objectives, solve_times = [], []
    for index, instance in enumerate(instances):
        seed = seed_base + index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if phase == "train":
            started = perf_counter()
        if deepaco_model is None:
            objective, path, completed_iterations = solve(instance[:, 1:], instance[:, 0], heuristic,
                                                          iterations, black_box=black_box)
        else:
            objective, path, completed_iterations = solve_deepaco(
                instance[:, 1:], instance[:, 0], deepaco_model, n_iterations=iterations)
        if phase == "train":
            elapsed = perf_counter() - started
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("单例完整求解时间无效。")
            solve_times.append(elapsed)
        objectives.append(objective)
        event = {"event": "instance", "index": index, "seed": seed, "objective": objective,
                 "iterations": completed_iterations, "path": path}
        if phase == "train":
            event["solve_wall_seconds"] = elapsed
        emit(event)
    if sha256(dataset.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("评价期间实例集发生变化；不生成完成事件。")
    if code_hash is not None and candidate_hash(root) != code_hash:
        raise ValueError("评价期间候选代码发生变化；不生成完成事件。")
    if reference_snapshot is not None and deepaco_model.source_snapshot() != reference_snapshot:
        raise ValueError("评价期间 DeepACO 来源文件发生变化；不生成完成事件。")
    average = float(np.mean(objectives))
    complete = {"event": "complete", "instance_count": len(objectives), "average": average, "fitness": -average}
    if phase == "train":
        complete["average_solve_wall_seconds"] = math.fsum(solve_times) / len(solve_times)
    emit(complete)


if __name__ == "__main__":
    main()
