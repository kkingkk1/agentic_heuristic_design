"""P0 TSP-ACO 训练评价；求解器来自固定版本的上游任务。"""

import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
from scipy.spatial import distance_matrix
import torch

from aco import ACO


N_ITERATIONS = 200
N_ANTS = 30
N_INSTANCES = 5
PROBLEM_SIZE = 50
SEED_BASE = 20261103
DATASET = Path(__file__).resolve().parent / "dataset" / "train50_dataset.npy"
EVENT_PREFIX = "P0_JSON "


def load_heuristic(root_dir):
    """从候选独立运行根目录导入 gpt，固定求解器仍从任务目录导入。"""
    candidate_dir = Path(root_dir).resolve() / "problems" / "tsp_aco"
    candidate_file = candidate_dir / "gpt.py"
    if not candidate_file.is_file():
        raise ValueError(f"缺少候选文件：{candidate_file}")
    spec = importlib.util.spec_from_file_location("gpt", candidate_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法导入候选文件：{candidate_file}")
    module = importlib.util.module_from_spec(spec)
    # 允许候选的局部辅助模块；每次评价使用单独的解释器进程。
    sys.path.insert(0, str(candidate_dir))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    heuristic = getattr(module, "heuristics_v2", None)
    if not inspect.isfunction(heuristic):
        raise ValueError("候选必须定义函数 heuristics_v2。")
    signature = inspect.signature(heuristic)
    parameters = list(signature.parameters.values())
    if (
        len(parameters) != 1
        or parameters[0].name != "distance_matrix"
        or parameters[0].kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
        or parameters[0].default is not inspect.Parameter.empty
        or parameters[0].annotation not in (np.ndarray, "np.ndarray")
        or signature.return_annotation not in (np.ndarray, "np.ndarray")
    ):
        raise ValueError("候选签名必须为 heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray。")
    return heuristic


def checked_heuristic(heuristic, distances):
    """检查原始输出，保留上游对有限负值的截断规则。"""
    scores = heuristic(distances.copy())
    if not isinstance(scores, np.ndarray) or scores.shape != distances.shape:
        raise ValueError("启发式必须返回与距离矩阵同形状的 NumPy 数组。")
    if scores.dtype.kind not in "iuf" or not np.isfinite(scores).all():
        raise ValueError("启发式矩阵必须是有限实数。")
    with np.errstate(over="raise", invalid="raise"):
        scores = scores.astype(np.float64, copy=False) + 1e-9
    if not np.isfinite(scores).all():
        raise ValueError("启发式矩阵转换后含非有限值。")
    scores[scores < 1e-9] = 1e-9
    return scores


class CheckedACO(ACO):
    """不改求解步骤，检查每轮全部蚂蚁的路径和代价。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checked_iterations = 0

    def gen_path(self, require_prob=False):
        paths = super().gen_path(require_prob=require_prob)
        if require_prob:
            raise ValueError("P0 求解不使用路径概率输出。")
        if paths.shape != (self.problem_size, self.n_ants) or paths.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        ):
            raise ValueError("蚂蚁路径的形状或类型无效。")
        expected = torch.arange(self.problem_size, device=paths.device).unsqueeze(1).expand_as(paths)
        if not torch.equal(torch.sort(paths.to(torch.int64), dim=0).values, expected):
            raise ValueError("蚂蚁路径不是城市排列。")
        self.checked_iterations += 1
        return paths

    def gen_path_costs(self, paths):
        costs = super().gen_path_costs(paths)
        if costs.shape != (self.n_ants,) or not torch.isfinite(costs).all() or not (costs > 0).all():
            raise ValueError("蚂蚁路径代价无效。")
        return costs


def solve(node_pos, heuristic):
    dist_mat = distance_matrix(node_pos, node_pos)
    if not np.isfinite(dist_mat).all():
        raise ValueError("距离矩阵含非有限值。")
    np.fill_diagonal(dist_mat, 1)
    if (dist_mat <= 0).any():
        raise ValueError("实例含重复城市或无效距离。")
    heu = checked_heuristic(heuristic, dist_mat)
    aco = CheckedACO(dist_mat, heu, n_ants=N_ANTS)
    obj = float(aco.run(N_ITERATIONS).item())
    if aco.checked_iterations != N_ITERATIONS or aco.shortest_path is None:
        raise ValueError("求解未完成全部迭代。")
    tour = aco.shortest_path.detach().cpu().numpy().astype(np.int64)
    if tour.shape != (len(node_pos),) or not np.array_equal(np.sort(tour), np.arange(len(node_pos))):
        raise ValueError("最优路径不是城市排列。")
    recomputed = float(dist_mat[tour, np.roll(tour, 1)].sum())
    if not math.isfinite(obj) or obj <= 0 or not math.isclose(obj, recomputed, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("最优路径的原始目标无效或与重算结果不一致。")
    return obj, tour.tolist(), aco.checked_iterations


def emit(event):
    print(EVENT_PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    if len(sys.argv) != 4 or sys.argv[1] != "50" or sys.argv[3] != "train":
        raise ValueError("用法：eval.py 50 <候选任务运行根目录> train")
    heuristic = load_heuristic(sys.argv[2])
    node_positions = np.load(DATASET, allow_pickle=False)
    if node_positions.shape != (N_INSTANCES, PROBLEM_SIZE, 2) or not np.isfinite(node_positions).all():
        raise ValueError("训练集不是 5 个有效的 50 节点二维实例。")
    print("[*] Running ...", flush=True)
    print(f"[*] Dataset loaded: {DATASET} with {N_INSTANCES} instances.", flush=True)
    emit({"event": "start", "dataset": str(DATASET), "seed_base": SEED_BASE,
          "n_ants": N_ANTS, "n_iterations": N_ITERATIONS,
          "numpy": np.__version__, "torch": torch.__version__,
          "torch_threads": torch.get_num_threads()})
    objs = []
    for index, node_pos in enumerate(node_positions):
        seed = SEED_BASE + index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        obj, tour, iterations = solve(node_pos, heuristic)
        objs.append(obj)
        print(f"[*] Instance {index}: {obj}", flush=True)
        emit({"event": "instance", "index": index, "seed": seed, "objective": obj,
              "iterations": iterations, "tour": tour})
    average = float(np.mean(objs))
    print("[*] Average:", flush=True)
    print(average, flush=True)
    emit({"event": "complete", "instance_count": len(objs), "average": average})


if __name__ == "__main__":
    main()
