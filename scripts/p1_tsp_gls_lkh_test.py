"""按 ReEvo 的 LKH 协议生成已冻结 GLS 测试集的逐实例参考。"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
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
import scipy
from scipy.spatial import distance_matrix


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.p0_runtime import memory_and_priority  # noqa: E402
from scripts.p0_windows_job import WindowsJob  # noqa: E402
from scripts.p1_tsp_gls_lkh_train import RUNS, SCALE, solve_instance  # noqa: E402


CONTRACT = ROOT / "cfg" / "p1_data_contract.json"
OUTPUT_ROOT = ROOT / "outputs" / "p1_tsp_gls"
SIZES = (20, 50, 100, 200)
COUNT = 64
PREFIX = "LKH_TEST_JSON "


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load_datasets():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen":
        raise ValueError("P1 数据合同未冻结。")
    entries = contract["tasks"]["tsp_gls"]["test"]
    if sorted(item["size"] for item in entries) != list(SIZES):
        raise ValueError("GLS 测试规模与数据合同不符。")
    datasets = {}
    for item in entries:
        size = item["size"]
        path = ROOT / "problems" / "tsp_gls" / "dataset" / f"test{size}_p1_dataset.npy"
        if item["path"] != path.relative_to(ROOT).as_posix() or item["instances"] != COUNT:
            raise ValueError(f"TSP{size} 的数据路径或实例数不符。")
        if digest(path) != item["sha256"]:
            raise ValueError(f"TSP{size} 的数据哈希不符。")
        points = np.load(path, allow_pickle=False)
        if (points.shape != (COUNT, size, 2) or points.dtype != np.float64
                or not np.isfinite(points).all() or (points < 0).any() or (points >= 1).any()):
            raise ValueError(f"TSP{size} 的坐标类型、形状或范围不符。")
        datasets[size] = {"path": path, "sha256": item["sha256"], "points": points}
    return datasets


def reference_objective(points, tour):
    size = len(points)
    if (points.dtype != np.float64 or not isinstance(tour, list) or len(tour) != size + 1
            or tour[0] != tour[-1] or any(type(node) is not int for node in tour)
            or sorted(tour[:-1]) != list(range(size))):
        raise ValueError("参考路线或 float64 坐标无效。")
    distances = distance_matrix(points, points) + np.eye(size) * 1e-5
    nodes = np.asarray(tour[:-1], dtype=np.int64)
    objective = float(distances[nodes, np.roll(nodes, 1)].sum())
    if not math.isfinite(objective) or objective <= 0:
        raise ValueError("参考路线长度无效。")
    return objective


def emit(event):
    print(PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def child():
    if (SCALE, RUNS) != (1_000_000, 10) or importlib.metadata.version("elkai") != "2.0.1":
        raise ValueError("LKH 协议或 elkai 版本不符。")
    datasets = load_datasets()
    contract_hash = digest(CONTRACT)
    workers = min(COUNT * len(SIZES), os.cpu_count() or 1)
    emit({"event": "start", "sizes": list(SIZES), "instances_per_size": COUNT,
          "scale": SCALE, "runs": RUNS, "workers": workers,
          "elkai_version": importlib.metadata.version("elkai"), "numpy_version": np.__version__,
          "scipy_version": scipy.__version__, "seed": "elkai 默认；接口未显式指定",
          "source": "https://github.com/ai4co/reevo/blob/2f70042e6014d4d5d1255732dbfb312bc5d3abab/problems/tsp_gls/test.ipynb",
          "integerization": "float64 距离乘 1000000 后向零取整为 int64",
          "objective": "按项目 GLS 评价器用原始 float64 距离计算闭合路线长度",
          "data_contract_sha256": contract_hash, "optimality_proven": False})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(solve_instance, (index, points)): (size, index)
                   for size in reversed(SIZES)
                   for index, points in enumerate(datasets[size]["points"])}
        for future in as_completed(futures):
            size, index = futures[future]
            result = future.result()
            objective = reference_objective(datasets[size]["points"][index], result["tour"])
            if result["index"] != index or not math.isclose(result["objective"], objective,
                                                            rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"TSP{size} 实例 {index} 的路线或长度不一致。")
            emit({"event": "instance", "size": size, "index": index, "tour": result["tour"],
                  "objective": objective, "elapsed_seconds": result["elapsed_seconds"]})
    if digest(CONTRACT) != contract_hash or any(digest(d["path"]) != d["sha256"] for d in datasets.values()):
        raise ValueError("参考求解期间数据合同或实例文件发生变化。")
    emit({"event": "complete", "instances": COUNT * len(SIZES), "optimality_proven": False})


def read_complete(path, datasets):
    events = [json.loads(line[len(PREFIX):]) for line in path.read_text(encoding="utf-8").splitlines()
              if line.startswith(PREFIX)]
    if (len(events) != COUNT * len(SIZES) + 2 or events[0].get("event") != "start"
            or events[-1] != {"event": "complete", "instances": COUNT * len(SIZES),
                              "optimality_proven": False}):
        raise ValueError("参考求解事件不完整。")
    start = events[0]
    if (start.get("sizes") != list(SIZES) or start.get("instances_per_size") != COUNT
            or start.get("scale") != SCALE or start.get("runs") != RUNS
            or start.get("elkai_version") != "2.0.1" or start.get("optimality_proven") is not False
            or start.get("data_contract_sha256") != digest(CONTRACT)):
        raise ValueError("参考求解配置或数据合同不一致。")
    grouped = {size: {} for size in SIZES}
    for event in events[1:-1]:
        size, index = event.get("size"), event.get("index")
        if (event.get("event") != "instance" or type(size) is not int or size not in grouped
                or type(index) is not int or not 0 <= index < COUNT or index in grouped[size]):
            raise ValueError("参考实例缺失、重复或索引无效。")
        expected = reference_objective(datasets[size]["points"][index], event.get("tour"))
        value, elapsed = event.get("objective"), event.get("elapsed_seconds")
        if (type(value) not in (float, int) or value != expected or type(elapsed) not in (float, int)
                or not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError(f"TSP{size} 实例 {index} 的参考长度或耗时无效。")
        grouped[size][index] = {"index": index, "tour": event["tour"], "objective": value,
                                "elapsed_seconds": elapsed}
    output = {}
    for size in SIZES:
        if sorted(grouped[size]) != list(range(COUNT)) or digest(datasets[size]["path"]) != datasets[size]["sha256"]:
            raise ValueError(f"TSP{size} 的结果或数据不完整。")
        records = [grouped[size][index] for index in range(COUNT)]
        output[str(size)] = {"dataset": datasets[size]["path"].relative_to(ROOT).as_posix(),
                             "dataset_sha256": datasets[size]["sha256"], "instances": records,
                             "mean_objective": float(np.mean([record["objective"] for record in records]))}
    return start, output


def supervise(run_dir):
    run_dir = Path(run_dir).resolve()
    if run_dir.parent != OUTPUT_ROOT.resolve():
        raise ValueError("运行目录须是 outputs/p1_tsp_gls 下的新目录。")
    datasets = load_datasets()
    resources = memory_and_priority()
    if resources["available_memory_bytes"] < 8 * 1024**3:
        raise RuntimeError("可用内存低于已登记的 8 GiB 启动门槛。")
    protected = [Path(__file__).resolve(), ROOT / "scripts" / "p1_tsp_gls_lkh_train.py", CONTRACT,
                 *(datasets[size]["path"] for size in SIZES)]
    before = {str(path): digest(path) for path in protected}
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--child"]
    environment = os.environ.copy()
    environment.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    record = {"status": "failed", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "command": command, "resources": resources, "protected_sha256": before,
              "optimality_proven": False}
    begin = time.monotonic()
    stdout_path, stderr_path = run_dir / "stdout.txt", run_dir / "stderr.txt"
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        try:
            with WindowsJob() as job:
                process = job.start(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                    stdout=stdout, stderr=stderr)
                record["pid"] = process.pid
                process.wait()
                while job.accounting()["active_processes"]:
                    time.sleep(0.05)
                record["exit_code"], record["process_tree"] = process.returncode, job.accounting()
        except (OSError, RuntimeError) as exc:
            record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = time.monotonic() - begin
    changed = [str(path) for path in protected if not path.is_file() or digest(path) != before[str(path)]]
    if changed:
        record["status"], record["error"] = "infrastructure_error", f"受保护文件发生变化：{changed}"
    if record["status"] == "failed" and record.get("exit_code") == 0:
        try:
            record["protocol"], record["datasets"] = read_complete(stdout_path, datasets)
            record["status"] = "passed"
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            record["error"] = f"参考输出不完整或无效：{exc}"
    with (run_dir / "result.json").open("x", encoding="utf-8", newline="\n") as output:
        json.dump(record, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    return record


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--child":
        child()
        return 0
    if len(sys.argv) != 2:
        print("用法：p1_tsp_gls_lkh_test.py <新的 GLS 运行目录>", file=sys.stderr)
        return 2
    try:
        result = supervise(sys.argv[1])
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"参考求解无法启动：{exc}", file=sys.stderr)
        return 2
    print(f"参考求解状态：{result['status']}；记录：{Path(sys.argv[1]).resolve() / 'result.json'}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
