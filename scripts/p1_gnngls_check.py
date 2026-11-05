"""监督 GNNGLS 真实权重的合成实例接入检查，不读取正式测试集。"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.gnngls.reference import ASSET_SHA256, ISOLATED, ROOT as SOURCE_ROOT, SOURCE_BLOBS  # noqa: E402
from scripts.p0_runtime import digest, memory_and_priority  # noqa: E402
from scripts.p0_windows_job import WindowsJob  # noqa: E402
from scripts.p1_job_supervision import wait_for_job  # noqa: E402
from scripts.p1_gnngls_test_evaluate import read_complete, run_instances, synthetic_inputs  # noqa: E402


PREFIX = "P1_GNNGLS_JSON "
def audit(stdout_path, size, seed, mode, time_limit, perturbations, checkpoint_size=20, scaler_size=20):
    events = [json.loads(line[len(PREFIX):]) for line in stdout_path.read_text(encoding="utf-8").splitlines()
              if line.startswith(PREFIX)]
    if len(events) != 1 or not isinstance(events[0], dict):
        raise ValueError("GNNGLS 合成检查事件缺失或重复。")
    event = events[0]
    if (event.get("status") != "passed" or event.get("formal_test_data_used") is not False
            or event.get("size") != size or event.get("seed") != seed or event.get("mode") != mode
            or event.get("checkpoint_size") != checkpoint_size or event.get("scaler_size") != scaler_size
            or event.get("compute_dtype") != "float64" or event.get("device") != "cpu"
            or event.get("time_limit") != time_limit or event.get("perturbation_moves") != perturbations):
        raise ValueError("GNNGLS 合成检查身份或参数不符。")
    route = event.get("route")
    if (not isinstance(route, list) or len(route) != size or any(type(node) is not int for node in route)
            or sorted(route) != list(range(size)) or route[0] != 0):
        raise ValueError("GNNGLS 合成检查的路线无效。")
    points = np.random.RandomState(seed).random_sample((size, 2))
    expected = math.fsum(math.dist(points[a], points[b]) for a, b in zip(route, route[1:] + route[:1]))
    for value in (event.get("objective"), event.get("source_cost")):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("GNNGLS 合成路线长度不一致。")
    if (mode == "initial_stop" and (event.get("stop_reason") != "reference_reached" or event.get("iterations") != 0)
            or mode == "timed" and event.get("stop_reason") != "time_limit"):
        raise ValueError("GNNGLS 合成检查的停止原因不符。")
    for key in ("prediction_seconds", "elapsed_seconds"):
        value = event.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("GNNGLS 合成检查的耗时无效。")
    return event


def worker(size, seed, mode, time_limit, perturbations, checkpoint_size, scaler_size, sequence):
    dll_dir = ISOLATED / "Library/bin"
    dll_handle = os.add_dll_directory(str(dll_dir)) if dll_dir.is_dir() else None
    from baselines.gnngls.reference import solve
    if sequence:
        run_instances(*synthetic_inputs(size, seed, mode, time_limit, perturbations, checkpoint_size, scaler_size))
        if dll_handle is not None:
            dll_handle.close()
        return
    points = np.random.RandomState(seed).random_sample((size, 2))
    # 单位正方形内每边小于 sqrt(2)，用于验证初始局部搜索后的提前停止。
    threshold = size * math.sqrt(2) + 1 if mode == "initial_stop" else None
    result = solve(points, checkpoint_size=checkpoint_size, scaler_size=scaler_size, time_limit=time_limit,
                   perturbation_moves=perturbations, reference=threshold)
    result.update({"status": "passed", "seed": seed, "mode": mode, "formal_test_data_used": False})
    print(PREFIX + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    if dll_handle is not None:
        dll_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=(20, 50, 100, 200), default=20)
    parser.add_argument("--seed", type=int, default=20800020)
    parser.add_argument("--mode", choices=("initial_stop", "timed"), default="initial_stop")
    parser.add_argument("--time-limit", type=float, default=3.0)
    parser.add_argument("--perturbations", type=int, default=3)
    parser.add_argument("--checkpoint-size", type=int, choices=(20, 50, 100), default=20)
    parser.add_argument("--scaler-size", type=int, choices=(20, 50, 100), default=20)
    parser.add_argument("--sequence", action="store_true", help="检查逐实例种子与线图复用")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not 0 <= args.seed < 2**32 or not math.isfinite(args.time_limit) or args.time_limit <= 0
            or args.perturbations < 1 or args.timeout_seconds <= 0):
        parser.error("种子、时间或扰动参数无效。")
    if args.worker:
        worker(args.size, args.seed, args.mode, args.time_limit, args.perturbations, args.checkpoint_size,
               args.scaler_size, args.sequence)
        return 0
    config = json.loads((ROOT / "cfg/p0_tsp_aco.json").read_text(encoding="utf-8"))["resources"]
    if Path(sys.executable).resolve() != Path(config["python_executable"]).resolve():
        raise ValueError("GNNGLS 检查必须使用项目 Python。")
    resources = memory_and_priority()
    if resources["available_memory_bytes"] < config["min_available_system_memory_bytes"] or resources["priority"] != "Normal":
        raise ValueError("可用内存或进程优先级不满足工程检查条件。")
    protected = [Path(__file__).resolve(), ROOT / "scripts/p1_gnngls_test_evaluate.py", ROOT / "baselines/gnngls/reference.py",
                 *(SOURCE_ROOT / name for name in SOURCE_BLOBS), *(SOURCE_ROOT / name for name in ASSET_SHA256)]
    before = {str(path): digest(path) for path in protected}
    command = [config["python_executable"], "-B", "-u", str(Path(__file__).resolve()), "--worker",
               "--size", str(args.size), "--seed", str(args.seed), "--mode", args.mode,
               "--time-limit", str(args.time_limit), "--perturbations", str(args.perturbations),
               "--checkpoint-size", str(args.checkpoint_size), "--scaler-size", str(args.scaler_size)]
    if args.sequence:
        command.append("--sequence")
    env = os.environ.copy()
    env.update(config["thread_environment"])
    env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "DGLBACKEND": "pytorch",
                "PYTHONPATH": os.pathsep.join([str(ROOT), str(ISOLATED), *config["pythonpath"]])})
    root = ROOT / "outputs/p1_gnngls" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    record = {"status": "failed", "command": command, "resources": resources, "protected_sha256": before,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "timeout_seconds": args.timeout_seconds,
              "formal_test_data_used": False}
    begin = time.monotonic()
    stdout_path, stderr_path = root / "stdout.txt", root / "stderr.txt"
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        try:
            with WindowsJob() as job:
                process = job.start(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                record["pid"] = process.pid
                timed_out, record["process_tree"] = wait_for_job(job, process, begin + args.timeout_seconds)
                record["exit_code"] = process.returncode
                if timed_out:
                    record["status"] = "timeout"
        except (OSError, RuntimeError) as exc:
            record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = time.monotonic() - begin
    try:
        if any(digest(Path(path)) != expected for path, expected in before.items()):
            raise ValueError("GNNGLS 工程检查期间受保护文件发生变化。")
        if record["status"] == "failed" and record.get("exit_code") == 0:
            if args.sequence:
                record["measurement"] = read_complete(stdout_path, *synthetic_inputs(args.size, args.seed, args.mode,
                    args.time_limit, args.perturbations, args.checkpoint_size, args.scaler_size))
            else:
                record["measurement"] = audit(stdout_path, args.size, args.seed, args.mode, args.time_limit,
                                               args.perturbations, args.checkpoint_size, args.scaler_size)
            record["status"] = "passed"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    with (root / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"GNNGLS 工程检查：{record['status']}；记录：{root}")
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
