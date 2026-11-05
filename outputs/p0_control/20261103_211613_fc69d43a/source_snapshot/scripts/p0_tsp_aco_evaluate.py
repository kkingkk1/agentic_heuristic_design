"""监督一次 P0 TSP-ACO 评价，保留原始输出、耗时和失败证据。"""

import ctypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "problems" / "tsp_aco" / "eval.py"
TIMEOUT_SECONDS = 60
EVENT_PREFIX = "P0_JSON "


def process_cpu_seconds(process):
    """记录评价主进程的用户态和内核态 CPU 时间；不含后代进程。"""
    if os.name != "nt":
        return None

    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    get_times = ctypes.windll.kernel32.GetProcessTimes
    get_times.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(FileTime)] * 4
    get_times.restype = ctypes.c_int
    created, exited, kernel, user = (FileTime() for _ in range(4))
    if not get_times(int(process._handle), *(ctypes.byref(x) for x in (created, exited, kernel, user))):
        return None

    def seconds(value):
        return ((value.high << 32) | value.low) / 10_000_000

    return {"user": seconds(user), "kernel": seconds(kernel)}


def read_complete(stdout_path):
    """只接受五个完整实例和平均值；部分输出不产生 Fitness。"""
    events = []
    for line in stdout_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(EVENT_PREFIX):
            events.append(json.loads(line[len(EVENT_PREFIX):]))
    if [event.get("event") for event in events] != ["start"] + ["instance"] * 5 + ["complete"]:
        raise ValueError("评价事件缺失、重复或顺序错误。")
    start, complete = events[0], events[-1]
    instances = events[1:-1]
    if (start.get("n_ants") != 30 or start.get("n_iterations") != 200
            or start.get("seed_base") != 20261103 or complete.get("instance_count") != 5):
        raise ValueError("评价配置或实例数量不符合 P0 合同。")
    for index, instance in enumerate(instances):
        obj = instance.get("objective")
        tour = instance.get("tour")
        if (
            instance.get("index") != index
            or instance.get("seed") != start.get("seed_base") + index
            or instance.get("iterations") != 200
            or not isinstance(obj, (int, float)) or isinstance(obj, bool)
            or not math.isfinite(obj) or obj <= 0
            or not isinstance(tour, list) or len(tour) != 50
            or any(type(city) is not int for city in tour) or sorted(tour) != list(range(50))
        ):
            raise ValueError(f"实例 {index} 的输出不完整或无效。")
    average = complete.get("average")
    if (
        not isinstance(average, (int, float)) or isinstance(average, bool)
        or not math.isfinite(average)
        or not math.isclose(average, sum(item["objective"] for item in instances) / 5,
                            rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError("平均目标与逐实例结果不一致。")
    return {"objectives": [item["objective"] for item in instances],
            "average": average, "fitness": -average, "start": start}


def terminate_tree(process):
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, text=True, errors="replace", check=False,
            )
        except OSError as exc:
            if process.poll() is None:
                process.kill()
            process.wait()
            return {"error": f"进程树终止失败，仅尝试终止主进程：{type(exc).__name__}: {exc}"}
        if result.returncode != 0 and process.poll() is None:
            process.kill()
        process.wait()
        return {"command": "taskkill /PID <pid> /T /F", "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    process.kill()
    process.wait()
    return {"error": "当前实现只在 Windows 使用 taskkill 终止进程树；此处仅终止主进程。"}


def evaluate(root_dir):
    root_dir = Path(root_dir).resolve()
    candidate = root_dir / "problems" / "tsp_aco" / "gpt.py"
    if not candidate.is_file():
        raise ValueError(f"找不到候选文件：{candidate}")
    stdout_path, stderr_path, record_path = (root_dir / name for name in ("stdout.txt", "stderr.txt", "result.json"))
    if any(path.exists() for path in (stdout_path, stderr_path, record_path)):
        raise ValueError("运行目录已含评价输出；请使用新的独立运行目录。")
    command = [sys.executable, "-B", "-u", str(EVAL), "50", str(root_dir), "train"]
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    record = {
        "command": command, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "timeout_seconds": TIMEOUT_SECONDS, "pythonpath": env.get("PYTHONPATH"),
        "omp_num_threads": env.get("OMP_NUM_THREADS"), "mkl_num_threads": env.get("MKL_NUM_THREADS"),
        "cpu_scope": "仅启动的主进程，不包含后代；若 Python 是启动器，实际求解可能在后代进程", "status": "failed",
    }
    begin = time.monotonic()
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr)
            record["pid"] = process.pid
            try:
                process.wait(timeout=max(0, TIMEOUT_SECONDS - (time.monotonic() - begin)))
            except subprocess.TimeoutExpired:
                record["status"] = "timeout"
                record["termination"] = terminate_tree(process)
            record["exit_code"] = process.returncode
            record["cpu_seconds"] = process_cpu_seconds(process)
        except OSError as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = time.monotonic() - begin
    if record["status"] != "timeout" and record.get("exit_code") == 0:
        try:
            record.update(read_complete(stdout_path))
            record["status"] = "passed"
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            record["error"] = f"输出不完整或无效：{exc}"
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def main():
    if len(sys.argv) != 2:
        print("用法：python -B scripts/p0_tsp_aco_evaluate.py <候选任务运行根目录>", file=sys.stderr)
        return 2
    try:
        result = evaluate(sys.argv[1])
    except (OSError, ValueError) as exc:
        print(f"评价无法启动：{exc}", file=sys.stderr)
        return 2
    print(f"评价状态：{result['status']}；记录：{Path(sys.argv[1]).resolve() / 'result.json'}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
