"""OP-ACO 训练与候选测试评价；核对收益和路径长度。"""

from hashlib import sha256
import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

from aco import ACO


DATASET = Path(__file__).resolve().parent / "dataset" / "train50_dataset.npz"
DATASET_SHA256 = "5371358192a23f4b56c91c0e72b3c9fa5ee186b287501f80b642a94711a981d8"
N_INSTANCES = 5
N_NODES = 50
N_ANTS = 30
N_ITERATIONS = 100
MAX_LEN = 3.0
SEED_BASE = 20261103
EVENT_PREFIX = "P1_JSON "
TEST_ITERATIONS = 200
TEST_SEED_BASE = 2026110400
TEST_MAX_LEN = {50: 3.0, 100: 4.0, 200: 5.0, 500: 8.0}
TEST_DATASET_SHA256 = {
    50: "a39a528bac23ecbd5bf825b4bd4a1e27936cd5b6cb38ef3642ee9d48d57d23dd",
    100: "acb9d1d08632fcf253f985586f590386f50b7b7d00bb6a070bf7697c5c8a5459",
    200: "c6b4b997e90540ebff17187bf6a5c2f32a8c311b5d7cf194a4e1944ebed9764f",
    500: "274178c6f62b2720b36ffb19da65fb739b54bf84d79a8990c411dd639c2cba8c",
}


def load_heuristic(root, *, black_box=False):
    candidate_dir = Path(root).resolve() / "problems" / "op_aco"
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
    names = ("node_attr", "edge_attr", "node_constraint") if black_box else ("prize", "distance", "maxlen")
    types = (np.ndarray, np.ndarray, float)
    if (len(parameters) != 3 or tuple(p.name for p in parameters) != names
            or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD or p.default is not inspect.Parameter.empty
                   or p.annotation not in (kind, "float" if kind is float else "np.ndarray")
                   for p, kind in zip(parameters, types))
            or signature.return_annotation not in (np.ndarray, "np.ndarray")):
        raise ValueError(f"候选签名与 {names} 不一致。")
    return heuristic


def instance(positions):
    positions = np.asarray(positions)
    if positions.dtype != np.float64:
        raise ValueError("OP 必须直接使用 float64 原始坐标。")
    coords = torch.from_numpy(positions.copy())
    from_depot = (coords - coords[0]).norm(p=2, dim=-1)
    if not torch.isfinite(from_depot).all() or from_depot.max() <= 0:
        raise ValueError("实例坐标无效。")
    prizes = 1 + torch.floor(99 * from_depot / from_depot.max())
    prizes /= prizes.max()
    distances = torch.norm(coords[:, None] - coords, dim=2, p=2)
    distances[torch.arange(len(coords)), torch.arange(len(coords))] = 1e9
    return prizes, distances


def checked_heuristic(heuristic, prizes, distances, max_len=MAX_LEN):
    scores = heuristic(prizes.copy(), distances.copy(), max_len)
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


def check_solution(solution, positions, prizes, max_len=MAX_LEN):
    solution = list(solution)
    n = len(prizes)
    if (not solution or solution[0] != 0
            or any(type(node) is not int or node < 0 or node > n for node in solution)):
        raise ValueError("路线的起点或节点编号无效。")
    if n in solution and any(node != n for node in solution[solution.index(n):]):
        raise ValueError("结束标记后出现节点。")
    nodes = [node for node in solution if node != n]
    if 0 in nodes[1:] or len(nodes) != len(set(nodes)):
        raise ValueError("路线重复访问节点。")
    closed = [*nodes, 0]
    distance = math.fsum(math.hypot(*(positions[a] - positions[b])) for a, b in zip(closed, closed[1:]))
    if not math.isfinite(distance) or distance > max_len + 1e-12:
        raise ValueError("路线超过长度限制。")
    # 来源求解器将起点收益计入目标；固定保留该口径并单独登记。
    return math.fsum(float(prizes[node]) for node in nodes), distance


class CheckedACO(ACO):
    """沿用来源构造与更新，逐轮核对每只蚂蚁的闭合路线和收益。"""

    def __init__(self, prizes, distances, heuristic, positions, max_len=MAX_LEN):
        self.source_positions = positions.copy()
        self.source_prizes = prizes.detach().cpu().numpy().copy()
        self.source_max_len = max_len
        self.checked_iterations = 0
        super().__init__(prizes, distances, max_len, heuristic, N_ANTS)
        # 与参考求解器使用同一原始距离，并精确清零虚拟终点列。
        self.distances[:self.n, :self.n] = distances
        self.distances[:, self.n] = 0

    def gen_sol(self):
        solutions = super().gen_sol()
        if (solutions.ndim != 2 or solutions.shape[1] != self.n_ants
                or solutions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError("蚂蚁解的形状或类型无效。")
        for solution in solutions.T.detach().cpu().tolist():
            check_solution(solution, self.source_positions, self.source_prizes, self.source_max_len)
        self.checked_iterations += 1
        return solutions

    def gen_sol_obj(self, solutions):
        objectives = super().gen_sol_obj(solutions)
        if objectives.shape != (self.n_ants,) or not torch.isfinite(objectives).all():
            raise ValueError("蚂蚁收益无效。")
        for actual, solution in zip(objectives.detach().cpu().tolist(), solutions.T.detach().cpu().tolist()):
            expected, _ = check_solution(solution, self.source_positions, self.source_prizes, self.source_max_len)
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("蚂蚁收益与所访问节点不一致。")
        return objectives


def solve(positions, heuristic, n_iterations=N_ITERATIONS, max_len=MAX_LEN):
    positions = np.asarray(positions).copy()
    prizes, distances = instance(positions)
    scores = checked_heuristic(heuristic, prizes.numpy(), distances.numpy(), max_len)
    aco = CheckedACO(prizes, distances, torch.from_numpy(scores), positions, max_len)
    objective, best = aco.run(n_iterations)
    if aco.checked_iterations != n_iterations or best is None:
        raise ValueError("求解未完成全部迭代。")
    solution = best.detach().cpu().tolist()
    expected, distance = check_solution(solution, positions, prizes.numpy(), max_len)
    objective = float(objective.item())
    if not math.isfinite(objective) or not math.isclose(objective, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("最优收益与所访问节点不一致。")
    return objective, solution, distance, aco.checked_iterations


def solve_deepaco(positions, model, *, k_sparse, n_iterations=N_ITERATIONS, max_len=MAX_LEN):
    """使用来源 DeepACO 求解器，并复用本任务的收益与路线检查。"""
    from baselines.deepaco.reference import load_source

    positions = np.asarray(positions).copy()
    if (positions.ndim != 2 or positions.shape[1] != 2 or len(positions) < 2
            or positions.dtype != np.float64 or not np.isfinite(positions).all()
            or getattr(model, "task", None) != "op"
            or type(n_iterations) is not int or n_iterations <= 0
            or isinstance(max_len, bool) or not isinstance(max_len, (int, float))
            or not math.isfinite(max_len) or max_len <= 0):
        raise ValueError("DeepACO OP 实例、模型或求解预算无效。")
    prizes, distances = instance(positions)
    scores = model.predict(coordinates=positions, prize=prizes.numpy(), k_sparse=k_sparse)
    if (not isinstance(scores, np.ndarray) or scores.dtype != np.float64
            or scores.shape != distances.shape or not np.isfinite(scores).all() or (scores <= 0).any()):
        raise ValueError("DeepACO OP 网络输出不是有效的 float64 矩阵。")
    source = load_source("op", "aco")

    class CheckedDeepACO(source.ACO):
        def __init__(self):
            super().__init__(distances, prizes, max_len, n_ants=N_ANTS,
                             heuristic=torch.from_numpy(scores.copy()), device="cpu")
            # 来源用默认精度构造终点列并按 1e-10 全局置零；恢复真实距离且精确清零终点。
            self.distances[:self.n, :self.n] = distances
            self.distances[:, self.n] = 0
            self.checked_iterations = 0

        def gen_sol(self, require_prob=False):
            if require_prob:
                raise ValueError("正式参考仅支持来源求解，不支持训练采样。")
            solutions = [torch.zeros(size=(self.n_ants,), device=self.device, dtype=torch.int64)]
            mask = torch.ones(size=(self.n_ants, self.n + 1), device=self.device, dtype=distances.dtype)
            # 来源将路径长度累加到默认 float32；这里使用冻结距离的 float64 精度。
            travel_dis = torch.zeros(size=(self.n_ants,), device=self.device, dtype=distances.dtype)
            cur_node = torch.zeros(size=(self.n_ants,), dtype=torch.int64, device=self.device)
            mask = self.update_mask(travel_dis, cur_node, mask)
            while not self.check_done(mask):
                nxt_node, _ = self.pick_node(mask, cur_node, False)
                solutions.append(nxt_node)
                travel_dis += self.distances[cur_node, nxt_node]
                cur_node = nxt_node
                mask = self.update_mask(travel_dis, cur_node, mask)
            result = torch.stack(solutions)
            for solution in result.T.detach().cpu().tolist():
                check_solution(solution, positions, prizes.numpy(), max_len)
            self.checked_iterations += 1
            return result

        def gen_sol_obj(self, solutions):
            objectives = super().gen_sol_obj(solutions)
            if (objectives.shape != (self.n_ants,) or objectives.dtype != torch.float64
                    or not torch.isfinite(objectives).all()):
                raise ValueError("DeepACO OP 收益不是有限的 float64。")
            for actual, solution in zip(objectives.detach().cpu().tolist(), solutions.T.detach().cpu().tolist()):
                expected, _ = check_solution(solution, positions, prizes.numpy(), max_len)
                if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("DeepACO OP 收益与项目评价不一致。")
            return objectives

    solver = CheckedDeepACO()
    objective, best = solver.run(n_iterations)
    if solver.checked_iterations != n_iterations or best is None:
        raise ValueError("DeepACO OP 未完成全部迭代。")
    solution = best.detach().cpu().tolist()
    expected, distance = check_solution(solution, positions, prizes.numpy(), max_len)
    objective = float(objective.item())
    if not math.isfinite(objective) or not math.isclose(objective, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("DeepACO OP 最优收益与项目评价不一致。")
    return objective, solution, distance, solver.checked_iterations


def emit(event):
    print(EVENT_PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def candidate_hash(root):
    return sha256((Path(root).resolve() / "problems" / "op_aco" / "gpt.py").read_bytes()).hexdigest()


def main(*, black_box=False, deepaco_model=None, k_sparse=None):
    if len(sys.argv) != 4:
        raise ValueError("用法：eval.py <规模> <候选任务运行根目录> <train|test>")
    size_text, root, phase = sys.argv[1:]
    if phase == "train" and size_text == "50":
        size, count, iterations, seed_base, max_len = N_NODES, N_INSTANCES, N_ITERATIONS, SEED_BASE, MAX_LEN
        dataset, expected_hash = DATASET, DATASET_SHA256
    elif phase == "test" and size_text in {str(size) for size in TEST_DATASET_SHA256}:
        size, count, iterations = int(size_text), 64, TEST_ITERATIONS
        seed_base, max_len = TEST_SEED_BASE + size * 100, TEST_MAX_LEN[size]
        dataset = DATASET.parent / f"test{size}_p1_dataset.npz"
        expected_hash = TEST_DATASET_SHA256[size]
    else:
        raise ValueError("训练只支持 OP50；测试支持 OP50、100、200、500。")
    if sha256(dataset.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("实例集哈希不匹配。")
    with np.load(dataset, allow_pickle=False) as data:
        positions = data["coordinates"].copy()
    if (positions.shape != (count, size, 2) or positions.dtype != np.float64
            or not np.isfinite(positions).all() or (positions < 0).any() or (positions >= 1).any()):
        raise ValueError("实例集内容不符合 OP 合同。")
    if deepaco_model is None:
        code_hash = candidate_hash(root) if phase == "test" else None
        heuristic = load_heuristic(root, black_box=black_box)
        reference_snapshot = None
    else:
        if (black_box or getattr(deepaco_model, "task", None) != "op"
                or getattr(deepaco_model, "dtype", None) != torch.float64
                or type(k_sparse) is not int or not 1 <= k_sparse < size):
            raise ValueError("DeepACO OP 模型、提示条件或稀疏度无效。")
        code_hash, heuristic = None, None
        reference_snapshot = deepaco_model.source_snapshot()
    start = {"event": "start", "dataset": str(dataset), "seed_base": seed_base, "n_ants": N_ANTS,
             "n_iterations": iterations, "n_nodes": size, "max_len": max_len,
             "numpy": np.__version__, "torch": torch.__version__, "torch_threads": torch.get_num_threads()}
    if code_hash is not None:
        start["candidate_sha256"] = code_hash
    if deepaco_model is not None:
        from baselines.deepaco.reference import software_versions
        start["reference"] = {"id": "deepaco_op64_project_eval", "checkpoint": deepaco_model.checkpoint,
                              "k_sparse": k_sparse, "compute_dtype": "float64", "source_sha256": reference_snapshot,
                              "software_versions": software_versions("op")}
    emit(start)
    objectives = []
    for index, points in enumerate(positions):
        seed = seed_base + index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if deepaco_model is None:
            objective, solution, distance, completed_iterations = solve(points, heuristic, iterations, max_len)
        else:
            objective, solution, distance, completed_iterations = solve_deepaco(
                points, deepaco_model, k_sparse=k_sparse, n_iterations=iterations, max_len=max_len)
        objectives.append(objective)
        emit({"event": "instance", "index": index, "seed": seed, "objective": objective,
              "iterations": completed_iterations, "solution": solution, "distance": distance})
    if sha256(dataset.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("评价期间实例集发生变化；不生成完成事件。")
    if code_hash is not None and candidate_hash(root) != code_hash:
        raise ValueError("评价期间候选代码发生变化；不生成完成事件。")
    if reference_snapshot is not None and deepaco_model.source_snapshot() != reference_snapshot:
        raise ValueError("评价期间 DeepACO 来源文件发生变化；不生成完成事件。")
    average = float(np.mean(objectives))
    emit({"event": "complete", "instance_count": len(objectives), "average": average, "fitness": average})


if __name__ == "__main__":
    main()
