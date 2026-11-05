"""按 ReEvo 笔记的 elkai/LKH 协议检查 GLS 训练集参考路线。"""

from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import distance_matrix


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.p0_runtime import memory_and_priority  # noqa: E402
from scripts.p0_windows_job import WindowsJob  # noqa: E402
from scripts.p1_job_supervision import wait_for_job  # noqa: E402
from scripts.p1_tsp_gls_evaluate import DATASET, DATASET_SHA256  # noqa: E402


PREFIX = "LKH_JSON "
SCALE = 1_000_000
RUNS = 10
TIMEOUT_SECONDS = 120


def solve_instance(item):
    import elkai

    index, points = item
    distances = distance_matrix(points, points) + np.eye(len(points)) * 1e-5
    matrix = (distances * SCALE).astype(np.int64).tolist()
    begin = time.monotonic()
    route = elkai.DistanceMatrix(matrix).solve_tsp(runs=RUNS)
    if (not isinstance(route, list) or len(route) != len(points) + 1 or route[0] != route[-1]
            or any(type(node) is not int for node in route) or sorted(route[:-1]) != list(range(len(points)))):
        raise ValueError(f"实例 {index} 的 LKH 路线无效。")
    objective = math.fsum(math.hypot(*(points[a] - points[b])) for a, b in zip(route, route[1:]))
    return {"event": "instance", "index": index, "tour": route,
            "objective": objective, "elapsed_seconds": time.monotonic() - begin}


def emit(event):
    print(PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def child():
    if sha256(DATASET.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("训练集哈希不匹配。")
    points = np.load(DATASET, allow_pickle=False)
    if points.shape != (10, 200, 2) or not np.isfinite(points).all():
        raise ValueError("GLS 训练集形状或内容无效。")
    version = importlib.metadata.version("elkai")
    if version != "2.0.1":
        raise ValueError(f"elkai 版本不符：{version}")
    workers = min(len(points), os.cpu_count() or 1)
    emit({"event": "start", "instances": len(points), "nodes": 200, "scale": SCALE, "runs": RUNS,
          "workers": workers, "elkai_version": version, "optimality_proven": False})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(solve_instance, enumerate(points)))
    for event in results:
        emit(event)
    if sha256(DATASET.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("参考求解期间训练集发生变化。")
    emit({"event": "complete", "instances": len(results),
          "average": math.fsum(event["objective"] for event in results) / len(results),
          "optimality_proven": False})


def read_complete(path):
    events = [json.loads(line[len(PREFIX):]) for line in path.read_text(encoding="utf-8").splitlines()
              if line.startswith(PREFIX)]
    if [event.get("event") for event in events] != ["start"] + ["instance"] * 10 + ["complete"]:
        raise ValueError("LKH 参考事件不完整。")
    start, complete = events[0], events[-1]
    if (start.get("instances") != 10 or start.get("nodes") != 200 or start.get("scale") != SCALE
            or start.get("runs") != RUNS or start.get("elkai_version") != "2.0.1"
            or start.get("optimality_proven") is not False or complete.get("optimality_proven") is not False):
        raise ValueError("LKH 参考配置不一致。")
    if sha256(DATASET.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("训练集哈希不匹配。")
    data = np.load(DATASET, allow_pickle=False)
    values = []
    for index, (event, points) in enumerate(zip(events[1:-1], data)):
        tour, value = event.get("tour"), event.get("objective")
        if (event.get("index") != index or not isinstance(tour, list) or len(tour) != 201
                or tour[0] != tour[-1] or any(type(node) is not int for node in tour)
                or sorted(tour[:-1]) != list(range(200))
                or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"实例 {index} 的 LKH 参考路线无效。")
        expected = math.fsum(math.hypot(*(points[a] - points[b])) for a, b in zip(tour, tour[1:]))
        if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"实例 {index} 的 LKH 路长不一致。")
        values.append(value)
    if complete.get("instances") != 10 or not math.isclose(complete.get("average", math.nan), math.fsum(values) / 10,
                                                          rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("LKH 参考平均值不一致。")
    return {"objectives": values, "average": math.fsum(values) / 10, "start": start,
            "optimality_proven": False}


def supervise(run_dir):
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    resources = memory_and_priority()
    if resources["available_memory_bytes"] < 8 * 1024**3:
        raise RuntimeError("可用内存低于 8 GiB 工程检查门槛。")
    protected = [Path(__file__).resolve(), DATASET, ROOT / "scripts/p1_job_supervision.py"]
    before = {str(path): sha256(path.read_bytes()).hexdigest() for path in protected}
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--child"]
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    start = time.monotonic()
    record = {"status": "failed", "command": command, "timeout_seconds": TIMEOUT_SECONDS,
              "protected_sha256": before, "resources": resources, "optimality_proven": False}
    stdout_path, stderr_path = run_dir / "stdout.txt", run_dir / "stderr.txt"
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        try:
            with WindowsJob() as job:
                process = job.start(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                record["pid"] = process.pid
                timeout, accounting = wait_for_job(job, process, start + TIMEOUT_SECONDS)
                record["exit_code"], record["process_tree"] = process.returncode, accounting
                if timeout:
                    record["status"] = "timeout"
        except (OSError, RuntimeError) as exc:
            record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = time.monotonic() - start
    if any(not path.is_file() or sha256(path.read_bytes()).hexdigest() != before[str(path)] for path in protected):
        record["status"], record["error"] = "infrastructure_error", "受保护文件发生变化。"
    if record["status"] == "failed" and record.get("exit_code") == 0:
        try:
            record.update(read_complete(stdout_path))
            record["status"] = "passed"
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            record["error"] = f"LKH 输出不完整或无效：{exc}"
    with (run_dir / "result.json").open("x", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    return record


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--child":
        child()
        return 0
    if len(sys.argv) != 2:
        print("用法：p1_tsp_gls_lkh_train.py <新工程运行目录>", file=sys.stderr)
        return 2
    try:
        result = supervise(sys.argv[1])
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"参考求解无法启动：{exc}", file=sys.stderr)
        return 2
    print(f"参考求解状态：{result['status']}；记录：{Path(sys.argv[1]).resolve() / 'result.json'}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
