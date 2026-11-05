"""MKP-ACO 训练与候选测试评价；检查解与原始目标。"""

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


DATASET = Path(__file__).resolve().parent / "dataset" / "train100_dataset.npz"
DATASET_SHA256 = "6dc1ffc07b50b8ee7bba919443a6840ed82a04a817e039e05da809877bf21073"
N_INSTANCES = 5
N_ITEMS = 100
N_DIMS = 5
N_ANTS = 10
N_ITERATIONS = 50
SEED_BASE = 20261103
EVENT_PREFIX = "P1_JSON "
TEST_ITERATIONS = 100
TEST_SEED_BASE = 2026110400
TEST_DATASET_SHA256 = {
    100: "1be223e7c34b414ffd358dd1f060b7dad815118d39ae2b62aabf5d03cab2a366",
    200: "7bbb642a2b9693028ce26505e2d50f20120d798ca05a4ddbf354ec279b46e4d2",
    300: "72f4a51e775c714b2b71783733b34165d2686c88f1830bd2f9aa42a3cb619a17",
    500: "171fb450c14e371fd5b989df8be03a08ed6ab620135d6c84b97a9bba001b8db9",
    700: "77132411d3d5279cce38342aead5e12b9a75ebb6dac8ceae3d559145ad645d27",
}


def load_heuristic(root, *, black_box=False):
    candidate_dir = Path(root).resolve() / "problems" / "mkp_aco"
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
    expected = ("item_attr1", "item_attr2") if black_box else ("prize", "weight")
    if (len(parameters) != 2 or tuple(p.name for p in parameters) != expected
            or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
                   or p.default is not inspect.Parameter.empty
                   or p.annotation not in (np.ndarray, "np.ndarray") for p in parameters)
            or signature.return_annotation not in (np.ndarray, "np.ndarray")):
        raise ValueError(f"候选签名必须使用两个 NumPy 数组参数 {expected} 并返回 NumPy 数组。")
    return heuristic


def checked_heuristic(heuristic, prize, weight):
    scores = heuristic(prize.copy(), weight.copy())
    if not isinstance(scores, np.ndarray) or scores.shape != prize.shape:
        raise ValueError("启发式必须返回与物品价值同形状的 NumPy 数组。")
    if scores.dtype.kind not in "iuf" or not np.isfinite(scores).all():
        raise ValueError("启发式必须返回有限实数。")
    with np.errstate(over="raise", invalid="raise"):
        scores = scores.astype(np.float64, copy=False) + 1e-9
    if not np.isfinite(scores).all():
        raise ValueError("启发式转换后含非有限值。")
    scores[scores < 1e-9] = 1e-9
    return scores


def check_solution(solution, prize, weight):
    values = list(solution)
    n = len(prize)
    if any(type(i) is not int or i < 0 or i > n for i in values):
        raise ValueError("解包含无效物品编号。")
    items = [i for i in values if i != n]
    if len(items) != len(set(items)):
        raise ValueError("解重复选择物品。")
    if n in values and any(i != n for i in values[values.index(n):]):
        raise ValueError("结束标记后出现物品。")
    used = weight[items].sum(axis=0) if items else np.zeros(weight.shape[1])
    if not np.isfinite(used).all() or (used > 1 + 1e-12).any():
        raise ValueError("解违反容量约束。")
    return math.fsum(float(prize[i]) for i in items)


class CheckedACO(ACO):
    """沿用来源求解步骤，逐轮检查每只蚂蚁的解和目标。"""

    def __init__(self, prize, weight, heuristic, n_ants):
        self.source_prize = prize.detach().cpu().numpy().copy()
        self.source_weight = weight.detach().cpu().numpy().copy()
        self.checked_iterations = 0
        super().__init__(prize, weight, heuristic, n_ants)

    def gen_sol(self):
        solutions = super().gen_sol()
        if (solutions.ndim != 2 or solutions.shape[1] != self.n_ants
                or solutions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError("蚂蚁解的形状或类型无效。")
        for solution in solutions.T.detach().cpu().tolist():
            check_solution(solution, self.source_prize, self.source_weight)
        self.checked_iterations += 1
        return solutions

    def gen_sol_obj(self, solutions):
        objectives = super().gen_sol_obj(solutions)
        if objectives.shape != (self.n_ants,) or not torch.isfinite(objectives).all():
            raise ValueError("蚂蚁目标值无效。")
        for actual, solution in zip(objectives.detach().cpu().tolist(), solutions.T.detach().cpu().tolist()):
            expected = check_solution(solution, self.source_prize, self.source_weight)
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("蚂蚁目标值与所选物品价值不一致。")
        return objectives


def solve(prize, weight, heuristic, n_iterations=N_ITERATIONS):
    scores = checked_heuristic(heuristic, prize, weight)
    aco = CheckedACO(torch.from_numpy(prize), torch.from_numpy(weight), torch.from_numpy(scores), N_ANTS)
    objective, best = aco.run(n_iterations)
    if aco.checked_iterations != n_iterations or best is None:
        raise ValueError("求解未完成全部迭代。")
    solution = best.detach().cpu().tolist()
    expected = check_solution(solution, prize, weight)
    objective = float(objective.item())
    if not math.isfinite(objective) or not math.isclose(objective, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("最优目标与所选物品价值不一致。")
    return objective, solution, aco.checked_iterations


def solve_deepaco(prize, weight, model, *, n_iterations=N_ITERATIONS):
    """保留 DeepACO 的物品间启发式，以项目口径检查多维容量和收益。"""
    from baselines.deepaco.reference import load_source

    prize, weight = np.asarray(prize).copy(), np.asarray(weight).copy()
    n = len(prize) if prize.ndim == 1 else 0
    if (n < 2 or prize.dtype != np.float64 or weight.dtype != np.float64
            or weight.shape != (n, N_DIMS) or not np.isfinite(prize).all()
            or not np.isfinite(weight).all() or (prize < 0).any() or prize.sum() <= 0
            or (weight < 0).any() or (weight > 1).any()
            or getattr(model, "task", None) != "mkp"
            or type(n_iterations) is not int or n_iterations <= 0):
        raise ValueError("DeepACO MKP 实例、模型或求解预算无效。")
    scores = model.predict(prize=prize, weight=weight)
    if (not isinstance(scores, np.ndarray) or scores.dtype != np.float64
            or scores.shape != (n, n) or not np.isfinite(scores).all() or (scores <= 0).any()):
        raise ValueError("DeepACO MKP 网络输出不是有效的 float64 矩阵。")
    scale = n // 2
    source = load_source("mkp", "aco")

    class CheckedDeepACO(source.ACO):
        def __init__(self):
            super().__init__(torch.from_numpy(prize.copy()), torch.from_numpy(weight * scale),
                             n_ants=N_ANTS, pheromone=torch.ones((n + 1, n + 1), dtype=torch.float64),
                             heuristic=torch.from_numpy(scores.copy()), device="cpu")
            self.heuristic[:, -1] = 1e-10
            self.checked_iterations = 0

        def gen_sol(self, require_prob=False):
            if require_prob:
                raise ValueError("正式参考仅支持来源求解，不支持训练采样。")
            items = torch.randint(low=0, high=self.n, size=(self.n_ants,), device=self.device)
            solutions = [items]
            # 来源原地累加到默认 float32；这里使用原始重量的 float64 精度。
            knapsack = torch.zeros((self.n_ants, self.m), device=self.device, dtype=self.weight.dtype)
            mask = torch.ones((self.n_ants, self.n + 1), device=self.device, dtype=self.weight.dtype)
            dummy_mask = torch.ones_like(mask)
            dummy_mask[:, -1] = 0
            mask, knapsack = self.update_knapsack(mask, knapsack, items)
            dummy_mask = self.update_dummy_state(mask, dummy_mask)
            while not self.check_done(mask):
                items, _ = self.pick_item(items, mask, dummy_mask, False)
                solutions.append(items)
                mask, knapsack = self.update_knapsack(mask, knapsack, items)
                dummy_mask = self.update_dummy_state(mask, dummy_mask)
            result = torch.stack(solutions)
            for solution in result.T.detach().cpu().tolist():
                check_solution(solution, prize, weight)
            self.checked_iterations += 1
            return result

        def gen_sol_obj(self, solutions):
            objectives = super().gen_sol_obj(solutions)
            if (objectives.shape != (self.n_ants,) or objectives.dtype != torch.float64
                    or not torch.isfinite(objectives).all()):
                raise ValueError("DeepACO MKP 收益不是有限的 float64。")
            for actual, solution in zip(objectives.detach().cpu().tolist(), solutions.T.detach().cpu().tolist()):
                expected = check_solution(solution, prize, weight)
                if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("DeepACO MKP 收益与项目评价不一致。")
            return objectives

    solver = CheckedDeepACO()
    objective, best = solver.run(n_iterations)
    if solver.checked_iterations != n_iterations or best is None:
        raise ValueError("DeepACO MKP 未完成全部迭代。")
    solution = best.detach().cpu().tolist()
    expected = check_solution(solution, prize, weight)
    objective = float(objective.item())
    if not math.isfinite(objective) or not math.isclose(objective, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("DeepACO MKP 最优收益与项目评价不一致。")
    return objective, solution, solver.checked_iterations


def emit(event):
    print(EVENT_PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def candidate_hash(root):
    return sha256((Path(root).resolve() / "problems" / "mkp_aco" / "gpt.py").read_bytes()).hexdigest()


def main(*, black_box=False, deepaco_model=None):
    if len(sys.argv) != 4:
        raise ValueError("用法：eval.py <规模> <候选任务运行根目录> <train|test>")
    size_text, root, phase = sys.argv[1:]
    if phase == "train" and size_text == "100":
        size, count, iterations, seed_base = N_ITEMS, N_INSTANCES, N_ITERATIONS, SEED_BASE
        dataset, expected_hash = DATASET, DATASET_SHA256
    elif phase == "test" and size_text in {str(size) for size in TEST_DATASET_SHA256}:
        size, count, iterations = int(size_text), 64, TEST_ITERATIONS
        seed_base = TEST_SEED_BASE + size * 100
        dataset = DATASET.parent / f"test{size}_p1_dataset.npz"
        expected_hash = TEST_DATASET_SHA256[size]
    else:
        raise ValueError("训练只支持 MKP100；测试支持 MKP100、200、300、500、700。")
    if sha256(dataset.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("实例集哈希不匹配。")
    with np.load(dataset, allow_pickle=False) as data:
        prizes, weights = data["prizes"].copy(), data["weights"].copy()
    if (prizes.shape != (count, size) or weights.shape != (count, size, N_DIMS)
            or prizes.dtype != np.float64 or weights.dtype != np.float64
            or not np.isfinite(prizes).all() or not np.isfinite(weights).all()
            or (prizes < 0).any() or (weights < 0).any()):
        raise ValueError("实例集内容不符合 MKP 合同。")
    if deepaco_model is None:
        code_hash = candidate_hash(root) if phase == "test" else None
        heuristic = load_heuristic(root, black_box=black_box)
        reference_snapshot = None
    else:
        if (black_box or getattr(deepaco_model, "task", None) != "mkp"
                or getattr(deepaco_model, "dtype", None) != torch.float64):
            raise ValueError("DeepACO MKP 模型或提示条件无效。")
        code_hash, heuristic = None, None
        reference_snapshot = deepaco_model.source_snapshot()
    start = {"event": "start", "dataset": str(dataset), "seed_base": seed_base,
             "n_ants": N_ANTS, "n_iterations": iterations, "n_items": size,
             "n_dims": N_DIMS, "numpy": np.__version__, "torch": torch.__version__,
             "torch_threads": torch.get_num_threads()}
    if code_hash is not None:
        start["candidate_sha256"] = code_hash
    if deepaco_model is not None:
        from baselines.deepaco.reference import software_versions
        start["reference"] = {"id": "deepaco_mkp64_project_eval", "checkpoint": deepaco_model.checkpoint,
                              "compute_dtype": "float64", "source_sha256": reference_snapshot,
                              "software_versions": software_versions("mkp")}
    emit(start)
    objectives = []
    for index, (prize, weight) in enumerate(zip(prizes, weights)):
        seed = seed_base + index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if deepaco_model is None:
            objective, solution, completed_iterations = solve(prize, weight, heuristic, iterations)
        else:
            objective, solution, completed_iterations = solve_deepaco(
                prize, weight, deepaco_model, n_iterations=iterations)
        objectives.append(objective)
        emit({"event": "instance", "index": index, "seed": seed, "objective": objective,
              "iterations": completed_iterations, "solution": solution})
    if sha256(dataset.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("评价期间实例集发生变化；不生成完成事件。")
    if code_hash is not None and candidate_hash(root) != code_hash:
        raise ValueError("评价期间候选代码发生变化；不生成完成事件。")
    if reference_snapshot is not None and deepaco_model.source_snapshot() != reference_snapshot:
        raise ValueError("评价期间 DeepACO 来源文件发生变化；不生成完成事件。")
    average = float(np.mean(objectives))
    emit({"event": "complete", "instance_count": len(objectives), "average": average,
          "fitness": average})


if __name__ == "__main__":
    main()
