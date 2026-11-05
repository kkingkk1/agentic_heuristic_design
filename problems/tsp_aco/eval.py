"""TSP-ACO 训练与 P1 测试评价；求解器来自固定版本的上游任务。"""

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
import torch

from aco import ACO


N_ITERATIONS = 200
N_ANTS = 30
N_INSTANCES = 5
PROBLEM_SIZE = 50
SEED_BASE = 20261103
DATASET = Path(__file__).resolve().parent / "dataset" / "train50_dataset.npy"
EVENT_PREFIX = "P0_JSON "
TEST_SIZES = (50, 100, 200)
TEST_ITERATIONS = 500
TEST_INSTANCES = 64
TEST_SEED_BASE = 2026110400
TEST_DATASET_SHA256 = {
    50: "a853e426a52e6beecbf2002fb6ddb69a2f7fbbc7c49cc1297f70171498abca15",
    100: "2124f6679df3788965c4822673eb335b4b129b88247c8c30a4bb1215f9cf696c",
    200: "bad03ad270daa513bb4b3cad6dc4fa7bea2cadd930c60ff8047c2ee828cfb2ab",
}


def load_heuristic(root_dir, parameter="distance_matrix"):
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
        or parameters[0].name != parameter
        or parameters[0].kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
        or parameters[0].default is not inspect.Parameter.empty
        or parameters[0].annotation not in (np.ndarray, "np.ndarray")
        or signature.return_annotation not in (np.ndarray, "np.ndarray")
    ):
        raise ValueError(f"候选签名必须为 heuristics_v2({parameter}: np.ndarray) -> np.ndarray。")
    return heuristic


def checked_heuristic(heuristic, distances, *, black_box=False):
    """检查启发式输出，并保留各上游入口的截断规则。"""
    if black_box:
        edge_count = distances.size
        scores = heuristic(distances.reshape(edge_count, 1).copy())
        if not isinstance(scores, np.ndarray) or scores.shape not in ((edge_count,), (edge_count, 1)):
            raise ValueError("黑盒启发式必须返回每条边一个分数。")
        scores = scores.reshape(distances.shape)
    else:
        scores = heuristic(distances.copy())
        if not isinstance(scores, np.ndarray) or scores.shape != distances.shape:
            raise ValueError("启发式必须返回与距离矩阵同形状的 NumPy 数组。")
    if scores.dtype.kind not in "iuf" or not np.isfinite(scores).all():
        raise ValueError("启发式矩阵必须是有限实数。")
    with np.errstate(over="raise", invalid="raise"):
        if black_box:
            scores = np.where(scores < 1e-9, 1e-9, scores)
        else:
            scores = scores.astype(np.float64, copy=False) + 1e-9
    if not np.isfinite(scores).all():
        raise ValueError("启发式矩阵转换后含非有限值。")
    if not black_box:
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


def solve(node_pos, heuristic, n_iterations=N_ITERATIONS, *, black_box=False):
    dist_mat = distance_matrix(node_pos, node_pos)
    if not np.isfinite(dist_mat).all():
        raise ValueError("距离矩阵含非有限值。")
    np.fill_diagonal(dist_mat, 1)
    if (dist_mat <= 0).any():
        raise ValueError("实例含重复城市或无效距离。")
    heu = checked_heuristic(heuristic, dist_mat, black_box=black_box)
    aco = CheckedACO(dist_mat, heu, n_ants=N_ANTS)
    obj = float(aco.run(n_iterations).item())
    if aco.checked_iterations != n_iterations or aco.shortest_path is None:
        raise ValueError("求解未完成全部迭代。")
    tour = aco.shortest_path.detach().cpu().numpy().astype(np.int64)
    if tour.shape != (len(node_pos),) or not np.array_equal(np.sort(tour), np.arange(len(node_pos))):
        raise ValueError("最优路径不是城市排列。")
    recomputed = float(dist_mat[tour, np.roll(tour, 1)].sum())
    if not math.isfinite(obj) or obj <= 0 or not math.isclose(obj, recomputed, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("最优路径的原始目标无效或与重算结果不一致。")
    return obj, tour.tolist(), aco.checked_iterations


def emit(event, prefix=EVENT_PREFIX):
    print(prefix + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def test_dataset(size):
    """读取指定测试集，核对登记文件及其形状。"""
    if size not in TEST_SIZES:
        raise ValueError("P1 TSP-ACO 测试只支持 50、100、200 节点。")
    path = DATASET.parent / f"test{size}_p1_dataset.npy"
    if sha256(path.read_bytes()).hexdigest() != TEST_DATASET_SHA256[size]:
        raise ValueError(f"测试集哈希不匹配：{path}")
    data = np.load(path, allow_pickle=False)
    if (data.shape != (TEST_INSTANCES, size, 2) or data.dtype.kind != "f"
            or not np.isfinite(data).all() or (data < 0).any() or (data >= 1).any()):
        raise ValueError(f"测试集内容不符合合同：{path}")
    return path, data


def candidate_hash(root):
    candidate = Path(root).resolve() / "problems" / "tsp_aco" / "gpt.py"
    return sha256(candidate.read_bytes()).hexdigest()


def main(*, black_box=False):
    if len(sys.argv) != 4:
        raise ValueError("用法：eval.py <规模> <候选任务运行根目录> <train|test>")
    size_text, root, phase = sys.argv[1:]
    if phase == "train" and size_text == "50":
        dataset, count, iterations, seed_base, prefix = DATASET, N_INSTANCES, N_ITERATIONS, SEED_BASE, EVENT_PREFIX
        node_positions = np.load(dataset, allow_pickle=False)
        if node_positions.shape != (count, PROBLEM_SIZE, 2) or not np.isfinite(node_positions).all():
            raise ValueError("训练集不是 5 个有效的 50 节点二维实例。")
    elif phase == "test" and size_text in {str(size) for size in TEST_SIZES}:
        size = int(size_text)
        dataset, node_positions = test_dataset(size)
        count, iterations, seed_base, prefix = TEST_INSTANCES, TEST_ITERATIONS, TEST_SEED_BASE + size * 100, "P1_JSON "
    else:
        raise ValueError("训练只支持 TSP50；测试只支持 TSP50、TSP100 和 TSP200。")
    code_hash = candidate_hash(root) if phase == "test" else None
    heuristic = load_heuristic(root, "edge_attr" if black_box else "distance_matrix")
    print("[*] Running ...", flush=True)
    print(f"[*] Dataset loaded: {dataset} with {count} instances.", flush=True)
    start = {"event": "start", "dataset": str(dataset), "seed_base": seed_base,
             "n_ants": N_ANTS, "n_iterations": iterations,
             "numpy": np.__version__, "torch": torch.__version__,
             "torch_threads": torch.get_num_threads()}
    if code_hash is not None:
        start["candidate_sha256"] = code_hash
    emit(start, prefix)
    objs = []
    for index, node_pos in enumerate(node_positions):
        seed = seed_base + index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if black_box:
            obj, tour, completed_iterations = solve(node_pos, heuristic, iterations, black_box=True)
        else:
            obj, tour, completed_iterations = solve(node_pos, heuristic, iterations)
        objs.append(obj)
        print(f"[*] Instance {index}: {obj}", flush=True)
        emit({"event": "instance", "index": index, "seed": seed, "objective": obj,
              "iterations": completed_iterations, "tour": tour}, prefix)
    if phase == "test":
        if sha256(dataset.read_bytes()).hexdigest() != TEST_DATASET_SHA256[size]:
            raise ValueError("评价期间测试集发生变化；不生成完成事件。")
        if candidate_hash(root) != code_hash:
            raise ValueError("评价期间候选代码发生变化；不生成完成事件。")
    average = float(np.mean(objs))
    print("[*] Average:", flush=True)
    print(average, flush=True)
    emit({"event": "complete", "instance_count": len(objs), "average": average}, prefix)


if __name__ == "__main__":
    main()
