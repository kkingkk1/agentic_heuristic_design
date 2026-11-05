"""离线 BPP-ACO 训练与候选测试评价；核对装箱合法性和原始箱数。"""

from hashlib import sha256
import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

import aco as source_aco
from aco import ACO


DATASET = Path(__file__).resolve().parent / "dataset" / "train500_dataset.npz"
DATASET_SHA256 = "1119ff4455d01d61559c6cbb967fa9919c266712f67cf008d6438847ed73d433"
N_INSTANCES = 5
N_ITEMS = 500
N_ANTS = 20
N_ITERATIONS = 100
CAPACITY = 150
SEED_BASE = 20261103
EVENT_PREFIX = "P1_JSON "
TEST_ITERATIONS = 100
TEST_SEED_BASE = 2026110400
TEST_DATASET_SHA256 = {
    500: "b8035a12fdae1000cb37cff7dfb421fb7f93d4d95b36acaa665de7488c94f440",
    700: "74b354f4aeaedec718cd0ab64ba628d9eb3ec3c274bd75b306d78be26be31ff2",
    1000: "73c38066b27d5c1bffb47f38bb17278eccd0bf61ed2cd7c55e5a1cce9f136da4",
}


def load_heuristic(root, *, black_box=False):
    candidate_dir = Path(root).resolve() / "problems" / "bpp_offline_aco"
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
    names = ("node_attr", "node_constraint") if black_box else ("demand", "capacity")
    kinds = (np.ndarray, int)
    if (len(parameters) != 2 or tuple(p.name for p in parameters) != names
            or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD or p.default is not inspect.Parameter.empty
                   or p.annotation not in (kind, "int" if kind is int else "np.ndarray")
                   for p, kind in zip(parameters, kinds))
            or signature.return_annotation not in (np.ndarray, "np.ndarray")):
        raise ValueError(f"候选签名与 {names} 不一致。")
    return heuristic


def checked_heuristic(heuristic, demand):
    scores = heuristic(demand.copy(), CAPACITY)
    n = len(demand)
    if not isinstance(scores, np.ndarray) or scores.shape != (n, n):
        raise ValueError("启发式必须返回物品数平方形状的 NumPy 数组。")
    if scores.dtype.kind not in "iuf" or not np.isfinite(scores).all() or scores.max() <= 0:
        raise ValueError("启发式必须返回有限实数，且至少有一个正值。")
    return scores.astype(np.float64, copy=True)


def check_solution(solution, demand):
    if not isinstance(solution, np.ndarray) or solution.shape != demand.shape or solution.dtype.kind not in "iu":
        raise ValueError("装箱方案必须为每件物品提供整数箱号。")
    if (solution < 0).any():
        raise ValueError("装箱方案含负箱号。")
    bins, inverse = np.unique(solution, return_inverse=True)
    loads = np.bincount(inverse, weights=demand)
    if not np.isfinite(loads).all() or (loads > CAPACITY).any():
        raise ValueError("装箱方案超过箱容量。")
    return len(bins)


class CheckedACO(ACO):
    """沿用来源采样与信息素更新，逐轮检查每只蚂蚁的装箱。"""

    def __init__(self, demand, heuristic):
        self.source_demand = demand.copy()
        self.checked_iterations = 0
        super().__init__(demand, heuristic, CAPACITY, n_ants=N_ANTS, greedy=False)

    def gen_paths(self, count, prob):
        paths, costs, fitnesses = super().gen_paths(count, prob)
        if len(paths) != self.n_ants or costs.shape != (self.n_ants,) or fitnesses.shape != (self.n_ants,):
            raise ValueError("蚂蚁装箱结果的数量或形状无效。")
        if not np.isfinite(fitnesses).all():
            raise ValueError("来源信息素适应度含非有限值。")
        for path, cost in zip(paths, costs):
            if int(cost) != check_solution(path, self.source_demand):
                raise ValueError("箱数与装箱方案不一致。")
        self.checked_iterations += 1
        return paths, costs, fitnesses


def solve(demand, heuristic, n_iterations=N_ITERATIONS):
    scores = checked_heuristic(heuristic, demand)
    aco = CheckedACO(demand, scores)
    objective, solution = aco.run(n_iterations)
    if aco.checked_iterations != n_iterations:
        raise ValueError("求解未完成全部迭代。")
    if int(objective) != check_solution(solution, demand):
        raise ValueError("最优箱数与装箱方案不一致。")
    return int(objective), solution.tolist(), aco.checked_iterations


def deepaco_assignment(path, demand):
    """将来源求解器以 0 分隔的物品序列转换为项目的逐物品箱号。"""
    values = list(path)
    n = len(demand)
    if (len(values) < 2 or values[0] != 0 or values[-1] != 0
            or any(type(node) is not int or node < 0 or node > n for node in values)):
        raise ValueError("DeepACO BPP 序列的端点或物品编号无效。")
    assignment = np.full(n, -1, dtype=np.int64)
    bin_index, in_bin = -1, False
    for node in values[1:]:
        if node == 0:
            in_bin = False
            continue
        if assignment[node - 1] != -1:
            raise ValueError("DeepACO BPP 序列重复物品。")
        if not in_bin:
            bin_index += 1
            in_bin = True
        assignment[node - 1] = bin_index
    if (assignment < 0).any():
        raise ValueError("DeepACO BPP 序列未包含全部物品。")
    return assignment


def solve_deepaco(demand, model, *, n_iterations=N_ITERATIONS):
    """保留来源装载率选解，使用项目检查器报告箱数。"""
    import torch
    from baselines.deepaco.reference import load_source

    demand = np.asarray(demand).copy()
    if (demand.ndim != 1 or len(demand) < 1 or demand.dtype.kind not in "iu"
            or (demand <= 0).any() or (demand > CAPACITY).any()
            or getattr(model, "task", None) != "bpp"
            or type(n_iterations) is not int or n_iterations <= 0):
        raise ValueError("DeepACO BPP 实例、模型或求解预算无效。")
    scores = model.predict(demands=demand)
    size = len(demand) + 1
    if (not isinstance(scores, np.ndarray) or scores.dtype != np.float64
            or scores.shape != (size, size) or not np.isfinite(scores).all() or (scores <= 0).any()):
        raise ValueError("DeepACO BPP 网络输出不是有效的 float64 矩阵。")
    source = load_source("bpp", "aco")

    class CheckedDeepACO(source.ACO):
        def __init__(self):
            super().__init__(torch.as_tensor(np.r_[0, demand], dtype=torch.float64), n_ants=N_ANTS,
                             pheromone=torch.ones((size, size), dtype=torch.float64),
                             heuristic=torch.from_numpy(scores.copy()), device="cpu", capacity=CAPACITY)
            self.checked_iterations = 0

        def gen_path(self, require_prob=False):
            if require_prob:
                raise ValueError("正式参考仅支持来源求解，不支持训练采样。")
            paths = super().gen_path(require_prob=False)
            if paths.ndim != 2 or paths.shape[1] != self.n_ants:
                raise ValueError("DeepACO BPP 蚂蚁序列数量无效。")
            for path in paths.T.detach().cpu().tolist():
                check_solution(deepaco_assignment(path, demand), demand)
            self.checked_iterations += 1
            return paths

        def gen_path_costs(self, paths):
            costs = super().gen_path_costs(paths)
            if costs.shape != (self.n_ants,) or not torch.isfinite(costs).all():
                raise ValueError("DeepACO BPP 来源适应度无效。")
            for cost, path in zip(costs.detach().cpu().tolist(), paths.T.detach().cpu().tolist()):
                assignment = deepaco_assignment(path, demand)
                count = check_solution(assignment, demand)
                loads = np.bincount(assignment, weights=demand)
                expected = math.fsum((float(load) / CAPACITY) ** 2 for load in loads) / count
                if not math.isclose(-cost, expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("DeepACO BPP 来源适应度与装箱方案不一致。")
            return costs

    solver = CheckedDeepACO()
    source_fitness = float(solver.run(n_iterations))
    if solver.checked_iterations != n_iterations or solver.shortest_path is None or not math.isfinite(source_fitness):
        raise ValueError("DeepACO BPP 未完成全部迭代。")
    assignment = deepaco_assignment(solver.shortest_path.detach().cpu().tolist(), demand)
    count = check_solution(assignment, demand)
    return count, assignment.tolist(), solver.checked_iterations, source_fitness


def emit(event):
    print(EVENT_PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def candidate_hash(root):
    return sha256((Path(root).resolve() / "problems" / "bpp_offline_aco" / "gpt.py").read_bytes()).hexdigest()


def main(*, black_box=False, deepaco_model=None):
    if len(sys.argv) != 4:
        raise ValueError("用法：eval.py <规模> <候选任务运行根目录> <train|test>")
    size_text, root, phase = sys.argv[1:]
    if phase == "train" and size_text == "500":
        size, count, iterations, seed_base = N_ITEMS, N_INSTANCES, N_ITERATIONS, SEED_BASE
        dataset, expected_hash = DATASET, DATASET_SHA256
    elif phase == "test" and size_text in {str(size) for size in TEST_DATASET_SHA256}:
        size, count, iterations = int(size_text), 64, TEST_ITERATIONS
        seed_base = TEST_SEED_BASE + size * 100
        dataset = DATASET.parent / f"test{size}_p1_dataset.npz"
        expected_hash = TEST_DATASET_SHA256[size]
    else:
        raise ValueError("训练只支持 BPP500；测试支持 BPP500、700、1000。")
    if sha256(dataset.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("实例集哈希不匹配。")
    with np.load(dataset, allow_pickle=False) as data:
        demands = data["demands"].copy()
    if (demands.shape != (count, size) or demands.dtype.kind not in "iu"
            or (demands < 20).any() or (demands > 100).any()):
        raise ValueError("实例集内容不符合离线 BPP 合同。")
    if deepaco_model is None:
        code_hash = candidate_hash(root) if phase == "test" else None
        heuristic = load_heuristic(root, black_box=black_box)
        reference_snapshot = None
    else:
        import torch
        if (black_box or getattr(deepaco_model, "task", None) != "bpp"
                or getattr(deepaco_model, "dtype", None) != torch.float64):
            raise ValueError("DeepACO BPP 模型或提示条件无效。")
        code_hash, heuristic = None, None
        reference_snapshot = deepaco_model.source_snapshot()
    start = {"event": "start", "dataset": str(dataset), "seed_base": seed_base, "n_ants": N_ANTS,
             "n_iterations": iterations, "n_items": size, "capacity": CAPACITY, "numpy": np.__version__}
    if code_hash is not None:
        start["candidate_sha256"] = code_hash
    if deepaco_model is not None:
        from baselines.deepaco.reference import software_versions
        start["reference"] = {"id": "deepaco_bpp64_project_eval", "checkpoint": deepaco_model.checkpoint,
                              "compute_dtype": "float64", "source_sha256": reference_snapshot,
                              "software_versions": software_versions("bpp")}
    emit(start)
    objectives = []
    for index, demand in enumerate(demands):
        seed = seed_base + index
        random.seed(seed)
        np.random.seed(seed)
        if deepaco_model is None:
            source_aco.uniform_generator = source_aco.uniform_number_generator()
            objective, solution, completed_iterations = solve(demand, heuristic, iterations)
            source_fitness = None
        else:
            torch.manual_seed(seed)
            objective, solution, completed_iterations, source_fitness = solve_deepaco(
                demand, deepaco_model, n_iterations=iterations)
        objectives.append(objective)
        event = {"event": "instance", "index": index, "seed": seed, "objective": objective,
                 "iterations": completed_iterations, "solution": solution}
        if source_fitness is not None:
            event["source_fitness"] = source_fitness
        emit(event)
    if sha256(dataset.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("评价期间实例集发生变化；不生成完成事件。")
    if code_hash is not None and candidate_hash(root) != code_hash:
        raise ValueError("评价期间候选代码发生变化；不生成完成事件。")
    if reference_snapshot is not None and deepaco_model.source_snapshot() != reference_snapshot:
        raise ValueError("评价期间 DeepACO 来源文件发生变化；不生成完成事件。")
    average = float(np.mean(objectives))
    if not math.isfinite(average):
        raise ValueError("平均箱数无效。")
    emit({"event": "complete", "instance_count": len(objectives), "average": average, "fitness": -average})


if __name__ == "__main__":
    main()
