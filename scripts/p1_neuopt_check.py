"""监督 NeuOpt 的合成实例工程检查；不读取正式测试集。"""

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

from scripts.p0_runtime import digest, memory_and_priority  # noqa: E402
from scripts.p0_windows_job import WindowsJob  # noqa: E402
from scripts.p1_job_supervision import wait_for_job  # noqa: E402
from baselines.neuopt.synthetic_check import ROOT as NEUOPT_ROOT, SOURCE_BLOBS  # noqa: E402


def audit(stdout_path, size, steps, seed, device):
    prefix = "P1_NEUOPT_JSON "
    events = [json.loads(line[len(prefix):]) for line in stdout_path.read_text(encoding="utf-8").splitlines()
              if line.startswith(prefix)]
    if len(events) != 1:
        raise ValueError("NeuOpt 工程输出事件缺失或重复。")
    event = events[0]
    if (event.get("status") != "passed" or event.get("size") != size or event.get("steps") != steps
            or event.get("seed") != seed or event.get("device") != device or event.get("batch_size") != 2
            or event.get("formal_test_data_used") is not False or len(event.get("instances", [])) != 2):
        raise ValueError("NeuOpt 工程检查参数不符。")
    elapsed, peak = event.get("search_elapsed_seconds"), event.get("peak_gpu_allocated_bytes")
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0
            or (device == "cuda" and (type(peak) is not int or peak <= 0))
            or (device == "cpu" and peak is not None)):
        raise ValueError("NeuOpt 搜索耗时或 GPU 峰值记录无效。")
    points = np.random.RandomState(seed).random_sample((2, size, 2))
    for index, result in enumerate(event["instances"]):
        route = result.get("route")
        if (not isinstance(route, list) or len(route) != size or any(type(node) is not int for node in route)
                or sorted(route) != list(range(size))):
            raise ValueError("NeuOpt 路线不是城市排列。")
        length = math.fsum(math.hypot(*(points[index, a] - points[index, b]))
                           for a, b in zip(route, route[1:] + route[:1]))
        if (not math.isclose(length, result.get("float64_recomputed_length"), rel_tol=1e-12, abs_tol=1e-12)
                or not math.isclose(length, result.get("source_float32_length"), rel_tol=1e-5, abs_tol=1e-5)):
            raise ValueError("NeuOpt 路线长度与独立重算不一致。")
    return event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=(20, 50, 100, 200), default=20)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20790020)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.steps <= 0 or args.seed < 0 or args.timeout_seconds <= 0:
        parser.error("步数和保护时限必须为正，种子不能为负。")
    resources = memory_and_priority()
    if resources["available_memory_bytes"] < 8589934592 or resources["priority"] != "Normal":
        raise RuntimeError("可用内存或进程优先级不满足工程检查条件。")
    gpu = None
    if args.device == "cuda":
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError("无法读取 GPU 状态。")
        gpu = result.stdout.strip()
    root = ROOT / "outputs/p1_neuopt" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    script = ROOT / "baselines/neuopt/synthetic_check.py"
    protected_paths = [script, Path(__file__).resolve(), *(NEUOPT_ROOT / name for name in SOURCE_BLOBS)]
    protected = {str(path): digest(path) for path in protected_paths}
    command = [sys.executable, "-B", "-u", str(script), "--size", str(args.size), "--steps", str(args.steps),
               "--seed", str(args.seed), "--device", args.device]
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    record = {"status": "failed", "command": command, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "timeout_seconds": args.timeout_seconds, "resources": resources, "gpu_before": gpu, "protected_sha256": protected,
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
        if any(digest(Path(path)) != expected for path, expected in protected.items()):
            raise ValueError("工程检查脚本在运行期间发生变化。")
        if record["status"] == "failed" and record.get("exit_code") == 0:
            record["measurement"] = audit(stdout_path, args.size, args.steps, args.seed, args.device)
            record["status"] = "passed"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    with (root / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"NeuOpt 工程检查：{record['status']}；记录：{root}")
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
