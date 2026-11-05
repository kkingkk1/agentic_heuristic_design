"""使用独立合成实例测量测试求解开销，不读取正式测试集。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import random
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
from scripts.p1_freeze_data import TEST_SIZES, test_constants  # noqa: E402
from scripts.p1_job_supervision import wait_for_job  # noqa: E402


TIMEOUT_SECONDS = 120
WORKERS = 4


def worker(name, size):
    task_dir = ROOT / "problems" / name
    sys.path.insert(0, str(task_dir))
    spec = importlib.util.spec_from_file_location("p1_calibration_evaluator", task_dir / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference_dir = "gls_reference" if name == "tsp_gls" else "aco_reference"
    reference = importlib.util.spec_from_file_location("p1_calibration_reference", ROOT / "baselines" / reference_dir / f"{name}.py")
    reference_module = importlib.util.module_from_spec(reference)
    reference.loader.exec_module(reference_module)
    heuristic = reference_module.heuristics_v2
    seed = 20770000 + list(TEST_SIZES).index(name) * 10000 + size
    rng = np.random.RandomState(seed)
    random.seed(seed)
    np.random.seed(seed)
    if name != "tsp_gls" and name != "bpp_offline_aco":
        import torch
        torch.manual_seed(seed)
    settings = test_constants(name)
    iterations = settings["TEST_PARAMETERS"][size][1] if name == "tsp_gls" else settings["TEST_ITERATIONS"]
    print(json.dumps({"event": "calibration_start", "task": name, "size": size, "seed": seed,
                      "iterations": iterations, "formal_test_data_used": False}), flush=True)
    started = time.monotonic()
    if name == "tsp_aco":
        result = module.solve(rng.rand(size, 2), heuristic, iterations, black_box=False)
    elif name == "op_aco":
        result = module.solve(rng.rand(size, 2), heuristic, iterations, settings["TEST_MAX_LEN"][size])
    elif name == "cvrp_aco":
        locations = np.concatenate((np.array([[0.5, 0.5]]), rng.rand(size, 2)))
        demands = np.concatenate((np.zeros(1), rng.randint(1, 10, size)))
        result = module.solve(locations, demands, heuristic, iterations, black_box=False)
    elif name == "mkp_aco":
        prizes, weights = rng.rand(size), rng.rand(size, 5)
        capacity = rng.uniform(low=weights.max(0), high=weights.sum(0))
        result = module.solve(prizes, weights / capacity.reshape(1, 5), heuristic, iterations)
    elif name == "bpp_offline_aco":
        module.source_aco.uniform_generator = module.source_aco.uniform_number_generator()
        result = module.solve(rng.randint(20, 101, size=size, dtype=np.int32), heuristic, iterations)
    elif name == "tsp_gls":
        result = module.solve(rng.rand(size, 2), heuristic, iterations, settings["TEST_PARAMETERS"][size][0])
    else:
        raise ValueError("未知任务。")
    print(json.dumps({"event": "calibration_complete", "task": name, "size": size, "seed": seed,
                      "iterations": iterations, "objective": float(result[0]),
                      "solution": result[1],
                      "solve_elapsed_seconds": time.monotonic() - started}, allow_nan=False), flush=True)


def calibrate(name, size, root, timeout):
    directory = root / name / str(size)
    directory.mkdir(parents=True, exist_ok=False)
    stdout_path, stderr_path = directory / "stdout.txt", directory / "stderr.txt"
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker", name, str(size)]
    begin = time.monotonic()
    record = {"task": name, "size": size, "status": "failed", "command": command,
              "timeout_seconds": timeout, "formal_test_data_used": False,
              "calibration_sha256": digest(Path(__file__).resolve()),
              "reference_sha256": digest(ROOT / "baselines" /
                  ("gls_reference" if name == "tsp_gls" else "aco_reference") / f"{name}.py"),
              "solver_sha256": digest(ROOT / "problems" / name / ("gls.py" if name == "tsp_gls" else "aco.py")),
              "evaluator_sha256": digest(ROOT / "problems" / name / "eval.py"),
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        try:
            with WindowsJob() as job:
                process = job.start(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                record["pid"] = process.pid
                timed_out, record["process_tree"] = wait_for_job(job, process, begin + timeout)
                record["exit_code"] = process.returncode
                if timed_out:
                    record["status"] = "timeout"
        except (OSError, RuntimeError) as exc:
            record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = time.monotonic() - begin
    protected = {Path(__file__).resolve(): record["calibration_sha256"],
                 ROOT / "baselines" / ("gls_reference" if name == "tsp_gls" else "aco_reference") / f"{name}.py": record["reference_sha256"],
                 ROOT / "problems" / name / ("gls.py" if name == "tsp_gls" else "aco.py"): record["solver_sha256"],
                 ROOT / "problems" / name / "eval.py": record["evaluator_sha256"]}
    if any(digest(path) != expected for path, expected in protected.items()):
        record["status"], record["error"] = "infrastructure_error", "校准期间受保护的源码发生变化。"
    if record["status"] == "failed" and record.get("exit_code") == 0:
        try:
            events = [json.loads(line) for line in stdout_path.read_text(encoding="utf-8").splitlines() if line.startswith("{")]
            if len(events) != 2 or [event.get("event") for event in events] != ["calibration_start", "calibration_complete"]:
                raise ValueError("校准事件不完整。")
            record["measurement"], record["status"] = events[-1], "passed"
        except (ValueError, json.JSONDecodeError) as exc:
            record["error"] = f"输出无效：{exc}"
    with (directory / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="执行 23 个单实例校准；默认仅输出方案")
    parser.add_argument("--case", action="append", help="只校准 TASK:SIZE；可重复")
    parser.add_argument("--timeout-seconds", type=int, default=TIMEOUT_SECONDS, help="每个完整实例的保护时限")
    parser.add_argument("--worker", nargs=2, metavar=("TASK", "SIZE"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        name, size = args.worker[0], int(args.worker[1])
        if name not in TEST_SIZES or size not in TEST_SIZES[name]:
            parser.error("未知任务或规模。")
        worker(name, size)
        return 0
    if args.timeout_seconds <= 0:
        parser.error("时限必须为正整数。")
    jobs = []
    for value in args.case or []:
        try:
            name, size = value.split(":", 1)
            size = int(size)
            if name not in TEST_SIZES or size not in TEST_SIZES[name] or (name, size) in jobs:
                raise ValueError
        except ValueError:
            parser.error(f"无效或重复的任务规模：{value}")
        jobs.append((name, size))
    if not jobs:
        jobs = [(name, size) for name, sizes in TEST_SIZES.items() for size in sizes]
    if not args.run:
        print(json.dumps({"instances": len(jobs), "workers": WORKERS,
                          "timeout_seconds_each": args.timeout_seconds, "formal_test_data_used": False}, ensure_ascii=False))
        return 0
    resources = memory_and_priority()
    if resources["available_memory_bytes"] < 8589934592 or resources["priority"] != "Normal":
        raise RuntimeError("资源或进程优先级不满足校准条件。")
    root = ROOT / "outputs/p1_test_calibration" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(calibrate, name, size, root, args.timeout_seconds) for name, size in jobs]
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda result: (list(TEST_SIZES).index(result["task"]), result["size"]))
    with (root / "summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"resources": resources, "results": results}, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"校准完成：{sum(result['status'] == 'passed' for result in results)}/{len(results)}；记录：{root}")
    return 0 if all(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
