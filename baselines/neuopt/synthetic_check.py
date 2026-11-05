"""NeuOpt 官方权重的独立合成实例工程检查，不读取正式测试集。"""

import argparse
from hashlib import sha1
import json
import math
from pathlib import Path
import random
import sys
import time
from types import ModuleType

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent / "verified"
SOURCE_BLOBS = {
    "agent/ppo.py": "879e0f952d28e6bafd72a29aa766a3f346e135cc",
    "agent/utils.py": "98cd2886a72c06f94f117d0f5cce6797198c5056",
    "nets/actor_network.py": "fc20c5c43e7e609c31efbc56fc4f3db2a0acd587",
    "nets/critic_network.py": "25a57ca42d1ed3c6e4d652c82c31f2a464ccc7be",
    "nets/graph_layers.py": "75b04d218fbc2db272407f2d21e009d568578469",
    "options.py": "97e32bc5367515dc9e703a32d562fd35ee2c7dc8",
    "problems/problem_cvrp.py": "78e97e819e67925f5fc21e7b1064e80cfdd28bac",
    "problems/problem_tsp.py": "199e381b078f846ed8c754bc84ec17b4f3bde703",
    "run.py": "4ac059550b0b5f11f3e075cadd90b7d2cc6ba53a",
    "utils/__init__.py": "90f60fdd89ad8575faafe45188bd1d968852fc67",
    "utils/logger.py": "26deaefb40bca48246e8057387e63a9b9418ac81",
    "utils/utils.py": "02dc42283aeccebaaf869f00608abe4163e7f3a4",
    "LICENSE": "eb2ad7b1ca6607490fab05b55be499bda3dcfd16",
    "pre-trained/tsp100.pt": "930ffeb8d30c92a51a862cfcbba7ce4d85c94b4f",
    "pre-trained/tsp20.pt": "9fb6a4665f11bd7e1941866b50645d5497be9ea1",
    "pre-trained/tsp200.pt": "9226aac2397e11d76fae58bbd78b5609954b5ab7",
    "pre-trained/tsp50.pt": "0ae4c68d942fc1e37693694ebaa002d6f66cf10d",
}


def verify_sources():
    for name, expected in SOURCE_BLOBS.items():
        data = (ROOT / name).read_bytes()
        actual = sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != expected:
            raise ValueError(f"NeuOpt 来源文件哈希不符：{name}")


def route_from_successor(successor):
    successor = list(successor)
    if not successor or any(type(value) is not int or value < 0 or value >= len(successor) for value in successor):
        raise ValueError("NeuOpt 后继表示无效。")
    route, node = [], 0
    for _ in successor:
        if node in route:
            raise ValueError("NeuOpt 路线提前重复。")
        route.append(node)
        node = successor[node]
    if node != 0 or len(set(route)) != len(successor):
        raise ValueError("NeuOpt 未返回完整闭合路线。")
    return route


def solve_batch(points, steps, seed, device, compute_dtype="float32"):
    """在独立进程中运行来源模型；每批至少两个实例。"""
    if compute_dtype not in ("float32", "float64"):
        raise ValueError("计算精度无效。")
    previous_dtype = torch.get_default_dtype()
    try:
        # 来源增强函数创建的旋转矩阵依赖默认精度；仅在本次调用期间设置。
        torch.set_default_dtype(torch.float64 if compute_dtype == "float64" else torch.float32)
        return _solve_batch(points, steps, seed, device, compute_dtype)
    finally:
        torch.set_default_dtype(previous_dtype)


def _solve_batch(points, steps, seed, device, compute_dtype):
    points = np.asarray(points)
    if (points.ndim != 3 or points.shape[0] < 2 or points.shape[2] != 2
            or points.shape[1] not in (20, 50, 100, 200) or points.dtype.kind != "f"
            or not np.isfinite(points).all() or (points < 0).any() or (points >= 1).any()
            or type(steps) is not int or steps < 1 or type(seed) is not int or seed < 0
            or device not in ("cpu", "cuda") or compute_dtype not in ("float32", "float64")):
        raise ValueError("实例、规模、步数、种子或设备无效。")
    batch_size, size = points.shape[:2]
    verify_sources()
    # 官方源代码在无 TensorBoard 模式下仍顶层导入 Logger；显式禁用该未使用功能。
    disabled_logger = ModuleType("tensorboard_logger")

    class DisabledLogger:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("本工程检查不启用 TensorBoard。")

    disabled_logger.Logger = DisabledLogger
    sys.modules["tensorboard_logger"] = disabled_logger
    sys.path.insert(0, str(ROOT))
    from options import get_options
    from problems.problem_tsp import TSP
    from agent.ppo import PPO

    options = get_options(["--problem", "tsp", "--graph_size", str(size), "--eval_only", "--no_tb",
                           "--no_saving", "--no_DDP", "--no_progress_bar", "--k", "4"]
                          + (["--no_cuda"] if device == "cpu" else []))
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("要求 CUDA，但当前不可用。")
    options.device = torch.device(device)
    options.use_cuda = device == "cuda"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    problem = TSP(p_size=size, init_val_met="random", k=4)
    agent = PPO(problem, options)
    state = torch.load(ROOT / "pre-trained" / f"tsp{size}.pt", map_location="cpu", weights_only=True)
    agent.actor.load_state_dict(state["actor"], strict=True)
    tensor_dtype = torch.float64 if compute_dtype == "float64" else torch.float32
    agent.actor.to(device=options.device, dtype=tensor_dtype)
    # 来源的位置编码不是 buffer，不会跟随 actor.to 转换精度。
    agent.actor.embedder.pattern = agent.actor.embedder.pattern.to(dtype=tensor_dtype)
    agent.actor.eval()
    batch = {"coordinates": torch.as_tensor(points, dtype=tensor_dtype).to(options.device)}
    captured = {}
    get_initial = problem.get_initial_solutions
    step = problem.step

    def capture_initial(input_batch):
        initial = get_initial(input_batch)
        captured["best"] = initial.clone().to(options.device)
        return initial

    def capture_step(*args, **kwargs):
        result = step(*args, **kwargs)
        improved = result[1][:, 0] > 0
        captured["best"][improved] = result[0][improved].clone()
        return result

    # TSP 的 step 返回空 feasibility_history；来源 record=True 会错误索引它。
    problem.get_initial_solutions, problem.step = capture_initial, capture_step
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.no_grad():
        result = agent.rollout(problem, T=steps, val_m=1, stall_limit=10, batch=batch, record=False)
    if device == "cuda":
        torch.cuda.synchronize()
    search_elapsed = time.monotonic() - started
    peak_gpu = torch.cuda.max_memory_allocated() if device == "cuda" else None
    successors = captured["best"].detach().cpu().tolist()
    observations = []
    for index, successor in enumerate(successors):
        route = route_from_successor(successor)
        length = math.fsum(math.hypot(*(points[index, a] - points[index, b]))
                           for a, b in zip(route, route[1:] + route[:1]))
        reported = float(result[0][index].detach().cpu())
        tolerance = 1e-12 if compute_dtype == "float64" else 1e-5
        if not math.isfinite(length) or not math.isclose(length, reported, rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError("NeuOpt 的路线与报告长度不一致。")
        observations.append({"route": route, "float64_recomputed_length": length,
                             "solver_reported_length": reported,
                             "source_float32_length": reported if compute_dtype == "float32" else None})
    return {"status": "passed", "size": size, "steps": steps, "seed": seed, "device": device,
            "compute_dtype": compute_dtype, "checkpoint": f"tsp{size}.pt", "batch_size": batch_size,
            "instances": observations,
            "search_elapsed_seconds": search_elapsed, "peak_gpu_allocated_bytes": peak_gpu,
            "adaptation": "不启用 TensorBoard；安全读取权重；使用至少两个实例；在 record=False 下记录来源最优路线。"}


def solve_one(points, steps, seed, device):
    """以逐实例种子求解；第二份副本仅满足来源的最小批量限制。"""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2 or points.dtype != np.float64:
        raise ValueError("NeuOpt 单实例输入必须为 float64 坐标矩阵。")
    batch = np.stack((points, points))
    result = solve_batch(batch, steps, seed, device, compute_dtype="float64")
    return {"route": result["instances"][0]["route"],
            "objective": result["instances"][0]["float64_recomputed_length"],
            "solver_reported_length": result["instances"][0]["solver_reported_length"],
            "search_elapsed_seconds": result["search_elapsed_seconds"],
            "peak_gpu_allocated_bytes": result["peak_gpu_allocated_bytes"]}


def check(size, steps, seed, device, compute_dtype="float32"):
    if size not in (20, 50, 100, 200) or type(seed) is not int or seed < 0:
        raise ValueError("合成规模或种子无效。")
    # 来源解码器对 batch_size=1 会将布尔张量压成标量；使用两个独立合成实例。
    points = np.random.RandomState(seed).random_sample((2, size, 2))
    result = solve_batch(points, steps, seed, device, compute_dtype)
    result["formal_test_data_used"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20790020)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compute-dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    print("P1_NEUOPT_JSON " + json.dumps(check(args.size, args.steps, args.seed, args.device, args.compute_dtype),
                                       ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
