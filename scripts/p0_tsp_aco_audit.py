"""独立核算任务 2 的验收记录；不参与候选评价或筛选。"""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "problems" / "tsp_aco" / "dataset" / "train50_dataset.npy"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_is_running(pid):
    if os.name != "nt":
        raise RuntimeError("进程树现场验收需要 Windows。")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER：PID 已不存在。
            return False
        raise OSError(error, f"无法核实 PID {pid}")
    try:
        exit_code = ctypes.c_uint32()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return exit_code.value == 259  # STILL_ACTIVE
    finally:
        kernel.CloseHandle(handle)


def audit(run_dir, check_live=False):
    dataset = np.load(DATASET, allow_pickle=False)
    if dataset.shape != (5, 50, 2) or not np.array_equal(dataset, np.random.RandomState(1234).rand(5, 50, 2)):
        raise ValueError("训练集与上游 seed=1234 的生成结果不一致。")
    if sha256(DATASET) != "49cfd10c6d21efd39736e5baf57510b4ce6945c3a057cdbe8dcad398c728599a":
        raise ValueError("训练集与固定来源的文件哈希不一致。")
    aco = PROJECT / "problems" / "tsp_aco" / "aco.py"
    if sha256(aco) != "233fdb43f4cba571c92a06134167830cdb2260d9f316650f4775d38e2848cb70":
        raise ValueError("ACO 求解器与固定来源的文件哈希不一致。")

    valid = {}
    for name in ("reference_final_a", "reference_final_b", "model_final"):
        directory = run_dir / name
        record = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        events = [json.loads(line[len("P0_JSON "):]) for line in
                  (directory / "stdout.txt").read_text(encoding="utf-8").splitlines()
                  if line.startswith("P0_JSON ")]
        if record["status"] != "passed" or [x["event"] for x in events] != ["start"] + ["instance"] * 5 + ["complete"]:
            raise ValueError(f"{name} 未完整评价。")
        objectives = []
        for index, event in enumerate(events[1:-1]):
            tour = event["tour"]
            if event["index"] != index or event["seed"] != 20261103 + index or event["iterations"] != 200:
                raise ValueError(f"{name} 实例 {index} 的索引、种子或迭代数错误。")
            if len(tour) != 50 or sorted(tour) != list(range(50)):
                raise ValueError(f"{name} 实例 {index} 的路径不是城市排列。")
            length = math.fsum(
                math.hypot(float(dataset[index, tour[j], 0] - dataset[index, tour[(j + 1) % 50], 0]),
                           float(dataset[index, tour[j], 1] - dataset[index, tour[(j + 1) % 50], 1]))
                for j in range(50)
            )
            if not math.isclose(length, event["objective"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{name} 实例 {index} 的路径长度不一致。")
            objectives.append(event["objective"])
        mean = math.fsum(objectives) / 5
        if (record["objectives"] != objectives or not math.isclose(mean, record["average"], abs_tol=1e-12)
                or not math.isclose(-mean, record["fitness"], abs_tol=1e-12)
                or not math.isclose(mean, events[-1]["average"], abs_tol=1e-12)):
            raise ValueError(f"{name} 的平均值或 Fitness 不一致。")
        valid[name] = {"objectives": objectives, "average": mean,
                       "elapsed_seconds": record["elapsed_seconds"], "cpu_seconds": record["cpu_seconds"]}
    if valid["reference_final_a"]["objectives"] != valid["reference_final_b"]["objectives"]:
        raise ValueError("参考函数的逐实例结果不能重放。")

    expected_errors = {"bad_signature": "候选签名必须", "bad_number": "启发式矩阵必须是有限实数",
                       "bad_shape": "同形状的 NumPy 数组"}
    for name, message in expected_errors.items():
        directory = run_dir / name
        record = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if (record["status"] != "failed" or record.get("exit_code") == 0 or "fitness" in record
                or message not in (directory / "stderr.txt").read_text(encoding="utf-8")):
            raise ValueError(f"{name} 没有按预期拒绝。")

    timeout_dir = run_dir / "timeout_latest"
    timeout = json.loads((timeout_dir / "result.json").read_text(encoding="utf-8"))
    child_pid = int((timeout_dir / "problems" / "tsp_aco" / "child_pid.txt").read_text(encoding="utf-8"))
    if (timeout["status"] != "timeout" or timeout["elapsed_seconds"] < 60
            or timeout["termination"]["exit_code"] != 0 or "fitness" in timeout
            or str(child_pid) not in timeout["termination"]["stdout"]):
        raise ValueError("超时、进程树终止或无 Fitness 的检查失败。")
    if check_live and (process_is_running(timeout["pid"]) or process_is_running(child_pid)):
        raise ValueError("超时进程或其子进程仍在运行。")

    seed_text = (PROJECT / "prompts" / "tsp_aco" / "seed_func.txt").read_text(encoding="utf-8")
    expected_reference = "import numpy as np\n" + seed_text.replace("heuristics_v1", "heuristics_v2")
    for name in ("reference_final_a", "reference_final_b"):
        if (run_dir / name / "problems" / "tsp_aco" / "gpt.py").read_text(encoding="utf-8") != expected_reference:
            raise ValueError("参考函数与上游工程种子不一致。")
    proposal_log = PROJECT / "outputs" / "p0_model_call" / "20261103_172345_07f94464" / "calls.jsonl"
    proposal = next(entry["code"] for entry in map(json.loads, proposal_log.read_text(encoding="utf-8").splitlines())
                    if entry.get("event") == "parsed" and entry.get("stage") == "proposal")
    if (run_dir / "model_final" / "problems" / "tsp_aco" / "gpt.py").read_text(encoding="utf-8").strip() != proposal.strip():
        raise ValueError("模型候选与任务 1 的原始记录不一致。")

    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": "4ad8cfc55056f2d2da94689366ac346eaa620505",
        "dataset_sha256": sha256(DATASET), "dataset_generation_seed": 1234,
        "aco_sha256": sha256(aco),
        "eval_sha256": sha256(PROJECT / "problems" / "tsp_aco" / "eval.py"),
        "supervisor_sha256_at_audit": sha256(PROJECT / "scripts" / "p0_tsp_aco_evaluate.py"),
        "independent_path_length_check": "math.hypot + math.fsum；5 实例 × 3 次评价",
        "reference_replay_exact": True, "valid_evaluations": valid,
        "invalid_candidates_rejected": list(expected_errors),
        "timeout": {"seconds": timeout["elapsed_seconds"], "pid": timeout["pid"],
                    "child_pid": child_pid, "live_pid_checked": check_live},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="本次验收根目录")
    parser.add_argument("--check-live", action="store_true", help="现场检查超时 PID；历史 PID 可能已复用")
    options = parser.parse_args()
    result = audit(options.run_dir.resolve(), check_live=options.check_live)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
