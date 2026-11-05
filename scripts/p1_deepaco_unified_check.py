"""以完整训练迭代预算检查已接入的 DeepACO；只使用独立合成实例。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.p0_runtime import digest, memory_and_priority  # noqa: E402
from scripts.p0_windows_job import WindowsJob  # noqa: E402
from scripts.p1_deepaco_worker import load_task  # noqa: E402
from scripts.p1_job_supervision import wait_for_job  # noqa: E402


CONFIG = ROOT / "cfg/p0_tsp_aco.json"
CASES = {
    "tsp": {"task_dir": "tsp_aco", "size": 50, "ants": 30, "iterations": 200,
            "checkpoint": "tsp100", "k_sparse": 20, "seed": 20783050},
    "op": {"task_dir": "op_aco", "size": 50, "ants": 30, "iterations": 100,
           "checkpoint": "op100", "k_sparse": 20, "seed": 20780050},
    "bpp": {"task_dir": "bpp_offline_aco", "size": 500, "ants": 20, "iterations": 100,
            "checkpoint": "bpp120", "seed": 20780500},
    "cvrp": {"task_dir": "cvrp_aco", "size": 50, "ants": 20, "iterations": 100,
             "checkpoint": "cvrp20", "seed": 20781050},
    "mkp": {"task_dir": "mkp_aco", "size": 100, "ants": 10, "iterations": 50,
            "checkpoint": "mkp300", "seed": 20782100},
}


def worker(name):
    import numpy as np
    import torch
    from baselines.deepaco.reference import DeepACOReference

    case = CASES[name]
    task = load_task(name)
    model = DeepACOReference(name, case["checkpoint"])
    rng = np.random.RandomState(case["seed"])
    torch.manual_seed(case["seed"])
    started = time.monotonic()
    if name == "tsp":
        from scripts.p1_deepaco_tsp import SCALE, solve
        data = rng.rand(case["size"], 2)
        objective, solution, iterations = solve(task, data, model, k_sparse=case["k_sparse"],
                                                 n_iterations=case["iterations"])
        detail = {"heuristic_scale": SCALE, "fitness": -objective}
    elif name == "op":
        data = rng.rand(case["size"], 2)
        objective, solution, distance, iterations = task.solve_deepaco(
            data, model, k_sparse=case["k_sparse"], n_iterations=case["iterations"], max_len=3.0)
        detail = {"distance": distance, "max_len": 3.0, "fitness": objective}
    elif name == "bpp":
        data = rng.randint(20, 101, size=case["size"], dtype=np.int32)
        objective, solution, iterations, source_fitness = task.solve_deepaco(
            data, model, n_iterations=case["iterations"])
        detail = {"source_fitness": source_fitness, "fitness": -objective}
    elif name == "cvrp":
        data = np.c_[np.r_[0, rng.randint(1, 10, size=case["size"])], rng.rand(case["size"] + 1, 2)]
        data[0, 1:] = (0.5, 0.5)
        objective, solution, iterations = task.solve_deepaco(
            data[:, 1:], data[:, 0], model, n_iterations=case["iterations"])
        detail = {"capacity": task.CAPACITY, "fitness": -objective}
    else:
        data = np.c_[rng.rand(case["size"]), rng.rand(case["size"], 5) / 50]
        objective, solution, iterations = task.solve_deepaco(
            data[:, 0], data[:, 1:], model, n_iterations=case["iterations"])
        detail = {"capacity": 1, "fitness": objective}
    result = {"status": "passed", "task": name, "size": case["size"], "seed": case["seed"],
              "ants": case["ants"], "iterations": iterations, "checkpoint": case["checkpoint"],
              "source_model_compute_dtype": str(model.dtype), "input_dtype": str(data.dtype),
              "input_sha256": hashlib.sha256(data.tobytes()).hexdigest(), "objective": objective,
              "solution": solution, "formal_test_data_used": False, "detail": detail,
              "solve_elapsed_seconds": time.monotonic() - started}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


def check_one(name, root, environment):
    case = CASES[name]
    directory = root / name
    directory.mkdir(exist_ok=False)
    checkpoint = ROOT / "baselines/deepaco/verified/pretrained" / name / f"{case['checkpoint']}.pt"
    protected = [Path(__file__).resolve(), CONFIG, checkpoint,
                 ROOT / "scripts/p1_deepaco_worker.py",
                 ROOT / "baselines/deepaco/reference.py",
                 *(ROOT / "baselines/deepaco/verified" / name / f"{part}.py" for part in ("aco", "net", "utils")),
                 ROOT / "problems" / case["task_dir"] / "eval.py",
                 ROOT / "problems" / case["task_dir"] / "aco.py"]
    if name == "tsp":
        protected.append(ROOT / "scripts/p1_deepaco_tsp.py")
    before = {str(path): digest(path) for path in protected}
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker", name]
    stdout_path, stderr_path = directory / "stdout.txt", directory / "stderr.txt"
    record = {"task": name, "status": "failed", "command": command, "timeout_seconds": 60,
              "formal_test_data_used": False, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "protected_sha256": before}
    begin = time.monotonic()
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        try:
            with WindowsJob() as job:
                process = job.start(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                    stdout=stdout, stderr=stderr)
                record["pid"] = process.pid
                timed_out, record["process_tree"] = wait_for_job(job, process, begin + 60)
                record["exit_code"] = process.returncode
                if timed_out:
                    record["status"] = "timeout"
        except (OSError, RuntimeError) as exc:
            record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = time.monotonic() - begin
    try:
        if any(digest(Path(path)) != expected for path, expected in before.items()):
            raise ValueError("工程检查期间受保护文件发生变化。")
        if record["status"] == "failed" and record.get("exit_code") == 0:
            measurement = json.loads(stdout_path.read_text(encoding="utf-8"))
            if (measurement.get("status") != "passed" or measurement.get("task") != name
                    or measurement.get("size") != case["size"] or measurement.get("ants") != case["ants"]
                    or measurement.get("iterations") != case["iterations"]
                    or measurement.get("checkpoint") != case["checkpoint"]
                    or measurement.get("source_model_compute_dtype") != "torch.float64"
                    or measurement.get("formal_test_data_used") is not False):
                raise ValueError("工程检查的任务、精度或预算记录不一致。")
            record["measurement"], record["status"] = measurement, "passed"
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    with (directory / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="启动已接入任务的合成实例受监督检查")
    parser.add_argument("--worker", choices=tuple(CASES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    if not args.run:
        print(json.dumps({"cases": CASES, "timeout_seconds": 60, "formal_test_data_used": False}, ensure_ascii=False))
        return 0
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if Path(sys.executable).resolve() != Path(config["resources"]["python_executable"]).resolve():
        raise ValueError("工程检查必须使用冻结的项目 Python。")
    resources = memory_and_priority()
    if resources["available_memory_bytes"] < config["resources"]["min_available_system_memory_bytes"]:
        raise RuntimeError("可用内存低于冻结阈值。")
    environment = os.environ.copy()
    environment.update(config["resources"]["thread_environment"])
    environment.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.pathsep.join([
        str(ROOT), r"D:\Tools\agentic-heuristic-p1-pyg", r"D:\Tools\agentic-heuristic-p1-numba",
        *config["resources"]["pythonpath"]])})
    root = ROOT / "outputs/p1_deepaco_unified" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    print(f"合成检查输出：{root}", flush=True)
    with ThreadPoolExecutor(max_workers=len(CASES)) as executor:
        futures = [executor.submit(check_one, name, root, environment) for name in CASES]
        results = [future.result() for future in as_completed(futures)]
    results.sort(key=lambda item: item["task"])
    with (root / "summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"resources": resources, "results": results}, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print("检查结果：" + ", ".join(f"{item['task']}={item['status']}" for item in results), flush=True)
    return 0 if all(item["status"] == "passed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
