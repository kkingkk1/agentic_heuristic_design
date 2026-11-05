"""监督已冻结配置的 DeepACO 正式测试，并复用项目评价口径。"""

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

from baselines.deepaco.reference import CHECKPOINT_BLOBS, CODE_BLOBS, ROOT as SOURCE_ROOT, software_versions, verify_blob  # noqa: E402
from scripts.p0_runtime import digest, memory_and_priority  # noqa: E402
from scripts.p0_windows_job import WindowsJob  # noqa: E402
from scripts.p1_deepaco_tsp import SCALE as TSP_HEURISTIC_SCALE  # noqa: E402
from scripts.p1_deepaco_worker import load_task  # noqa: E402
from scripts.p1_runtime import DATA_CONTRACT  # noqa: E402
from scripts.p1_test_evaluate import wait_bounded  # noqa: E402


RESOURCE_CONFIG = ROOT / "cfg/p0_tsp_aco.json"
PREFIX = "P1_JSON "
SOURCE_COMMIT = "9a756a3fe6cf9627ecc166110a02fdbb61f2a0d7"
TASKS = {"tsp_aco": ("tsp", "deepaco_tsp64_project_aco_adapted", "min"),
         "op_aco": ("op", "deepaco_op64_project_eval", "max"),
         "bpp_offline_aco": ("bpp", "deepaco_bpp64_project_eval", "min"),
         "cvrp_aco": ("cvrp", "deepaco_cvrp64_project_eval", "min"),
         "mkp_aco": ("mkp", "deepaco_mkp64_project_eval", "max")}


def load_reference_contract(config, data_contract, data_sha256):
    if (not isinstance(config, dict) or config.get("status") != "frozen"
            or not isinstance(data_contract, dict) or data_contract.get("status") != "frozen"
            or data_contract.get("contract_id") != "p1_data_20261104_v1"
            or config.get("data_contract_sha256") != data_sha256
            or config.get("resource_config_sha256") != digest(RESOURCE_CONFIG)):
        raise ValueError("参考方法运行配置或数据合同尚未冻结。")
    task = config.get("task")
    if task not in TASKS:
        raise ValueError("DeepACO 正式参考只支持已接入的任务。")
    name, method, _direction = TASKS[task]
    size, checkpoint = config.get("size"), config.get("checkpoint")
    if type(size) is not int or size < 2 or not isinstance(checkpoint, str):
        raise ValueError("参考方法规模或权重无效。")
    key = f"pretrained/{name}/{checkpoint}.pt"
    if (config.get("method") != method or config.get("source_commit") != SOURCE_COMMIT
            or key not in CHECKPOINT_BLOBS
            or config.get("checkpoint_git_blob") != CHECKPOINT_BLOBS[key]):
        raise ValueError("参考方法或来源权重身份不一致。")
    if not isinstance(config.get("implementation_sha256"), dict) or not config["implementation_sha256"]:
        raise ValueError("参考方法的调用代码与评价代码哈希尚未冻结。")
    version_keys = {"python", "numpy", "torch", "torch-geometric", "pyparsing", "xxhash"}
    if name == "bpp":
        version_keys |= {"numba", "llvmlite"}
    if name in ("tsp", "cvrp"):
        version_keys.add("scipy")
    versions = config.get("software_versions")
    if (not isinstance(versions, dict) or set(versions) != version_keys
            or any(not isinstance(value, str) or not value for value in versions.values())):
        raise ValueError("参考方法的软件版本尚未完整冻结。")
    checkpoint_size = int(checkpoint.removeprefix(name))
    scale_mode = "matched" if checkpoint_size == size else "cross_scale"
    if config.get("checkpoint_scale_mode") != scale_mode:
        raise ValueError("必须明确登记权重规模是否与测试规模一致。")
    k_sparse = config.get("k_sparse")
    if ((name in ("tsp", "op") and (type(k_sparse) is not int or not 1 <= k_sparse < size))
            or (name not in ("tsp", "op") and k_sparse is not None)):
        raise ValueError("TSP／OP 稀疏度或其他任务参数无效。")
    if type(config.get("solver_repeats_per_instance")) is not int or config["solver_repeats_per_instance"] != 1:
        raise ValueError("当前测试入口只支持每个实例求解一次。")
    for key in ("whole_dataset_wall_timeout_seconds", "process_tree_cpu_seconds"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("完整数据集的墙钟或进程树 CPU 预算无效。")
    entries = [item for item in data_contract.get("tasks", {}).get(task, {}).get("test", [])
               if item.get("size") == size]
    if len(entries) != 1 or entries[0].get("instances") != 64:
        raise ValueError("测试规模未在冻结数据合同中唯一登记。")
    return entries[0]


def expected_source_snapshot(name, checkpoint):
    path = SOURCE_ROOT / "pretrained" / name / f"{checkpoint}.pt"
    verify_blob(path, CHECKPOINT_BLOBS[path.relative_to(SOURCE_ROOT).as_posix()])
    paths = [path]
    for part in ("aco", "net", "utils"):
        path = SOURCE_ROOT / name / f"{part}.py"
        verify_blob(path, CODE_BLOBS[f"{name}/{part}.py"])
        paths.append(path)
    paths.append(ROOT / "baselines/deepaco/reference.py")
    return {str(path): digest(path) for path in paths}


def reference_implementation_snapshot(task, checkpoint):
    name = TASKS[task][0]
    source = expected_source_snapshot(name, checkpoint)
    task_dir = ROOT / "problems" / task
    paths = [task_dir / "eval.py", task_dir / "aco.py", Path(__file__).resolve(),
             *(ROOT / "scripts" / name for name in ("p1_deepaco_worker.py", "p1_test_evaluate.py",
                                                     "p0_windows_job.py", "p0_runtime.py", "p1_runtime.py"))]
    if name == "tsp":
        paths.append(ROOT / "scripts/p1_deepaco_tsp.py")
    return source, {**source, **{str(path): digest(path) for path in paths}}


def check_reference_implementation(config, implementation):
    if config.get("implementation_sha256") != implementation:
        raise ValueError("参考方法的调用代码或评价代码与冻结配置不一致。")


def load_reference_instances(name, dataset, size):
    if name in ("tsp", "cvrp"):
        instances = np.load(dataset, allow_pickle=False).copy()
        shape = (64, size, 2) if name == "tsp" else (64, size + 1, 3)
        if instances.shape != shape or instances.dtype != np.float64:
            raise ValueError("TSP／CVRP 测试实例的形状或精度不符合项目合同。")
    else:
        with np.load(dataset, allow_pickle=False) as data:
            if name == "mkp":
                instances = (data["prizes"].copy(), data["weights"].copy())
            else:
                instances = data["coordinates" if name == "op" else "demands"].copy()
        if name == "mkp":
            prizes, weights = instances
            if (prizes.shape != (64, size) or weights.shape != (64, size, 5)
                    or prizes.dtype != np.float64 or weights.dtype != np.float64):
                raise ValueError("MKP 测试实例的形状或精度不符合项目合同。")
        elif (instances.shape != ((64, size, 2) if name == "op" else (64, size))
              or (name == "op" and instances.dtype != np.float64)
              or (name == "bpp" and instances.dtype.kind not in "iu")):
            raise ValueError("测试实例的形状或精度不符合项目合同。")
    return instances


def read_reference_complete(stdout_path, config, entry, dataset, instances, evaluator, snapshot, ants):
    events = [json.loads(line[len(PREFIX):]) for line in stdout_path.read_text(encoding="utf-8").splitlines()
              if line.startswith(PREFIX)]
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("DeepACO 测试事件不是 JSON 对象。")
    if [event.get("event") for event in events] != ["start"] + ["instance"] * 64 + ["complete"]:
        raise ValueError("DeepACO 测试事件缺失、重复或顺序错误。")
    name, method, direction = TASKS[config["task"]]
    start, complete = events[0], events[-1]
    reference = start.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("DeepACO 参考来源记录无效。")
    if (Path(start.get("dataset", "")).resolve() != dataset.resolve()
            or start.get("seed_base") != entry["evaluation_seed_base"]
            or start.get("n_iterations") != entry["iterations"] or start.get("n_ants") != ants
            or start.get({"tsp": "n_nodes", "op": "n_nodes", "bpp": "n_items",
                          "cvrp": "n_customers", "mkp": "n_items"}[name])
            != config["size"]
            or complete.get("instance_count") != 64
            or reference.get("id") != method or reference.get("checkpoint") != config["checkpoint"]
            or reference.get("compute_dtype") != "float64"
            or reference.get("software_versions") != config["software_versions"]
            or reference.get("source_sha256") != snapshot):
        raise ValueError("DeepACO 测试数据、来源身份或预算记录不一致。")
    if name == "op" and (reference.get("k_sparse") != config["k_sparse"]
                         or start.get("max_len") != entry["max_path_length"]):
        raise ValueError("DeepACO OP 稀疏度或路线长度约束不一致。")
    if name == "tsp" and (reference.get("k_sparse") != config["k_sparse"]
                          or reference.get("heuristic_scale") != TSP_HEURISTIC_SCALE):
        raise ValueError("DeepACO TSP 稀疏度或边权适配不一致。")
    if ((name == "cvrp" and start.get("capacity") != evaluator.CAPACITY)
            or (name == "mkp" and start.get("n_dims") != evaluator.N_DIMS)):
        raise ValueError("DeepACO 任务的容量或维度不一致。")
    objectives = []
    for index, event in enumerate(events[1:-1]):
        objective = event.get("objective")
        if (event.get("index") != index or event.get("seed") != entry["evaluation_seed_base"] + index
                or event.get("iterations") != entry["iterations"]
                or isinstance(objective, bool) or not isinstance(objective, (int, float)) or not math.isfinite(objective)):
            raise ValueError(f"DeepACO 测试实例 {index} 的结果无效或不完整。")
        if name == "tsp":
            tour = event.get("tour")
            if (not isinstance(tour, list) or len(tour) != config["size"]
                    or any(type(node) is not int for node in tour)
                    or sorted(tour) != list(range(config["size"]))):
                raise ValueError(f"DeepACO TSP 实例 {index} 的路径无效。")
            distances = evaluator.distance_matrix(instances[index], instances[index])
            np.fill_diagonal(distances, 1)
            expected = float(distances[tour, np.roll(tour, 1)].sum())
        elif name == "op":
            prizes, _ = evaluator.instance(instances[index])
            expected, distance = evaluator.check_solution(event.get("solution", []), instances[index],
                                                           prizes.numpy(), entry["max_path_length"])
            reported_distance = event.get("distance")
            if (isinstance(reported_distance, bool) or not isinstance(reported_distance, (int, float))
                    or not math.isfinite(reported_distance)
                    or not math.isclose(reported_distance, distance, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f"DeepACO OP 实例 {index} 的路径长度不一致。")
        elif name == "bpp":
            solution = event.get("solution")
            if (not isinstance(solution, list) or any(type(item) is not int or not 0 <= item < config["size"]
                                                       for item in solution)):
                raise ValueError(f"DeepACO BPP 实例 {index} 的箱号无效。")
            assignment = np.asarray(solution, dtype=np.int64)
            expected = evaluator.check_solution(assignment, instances[index])
            loads = np.bincount(assignment, weights=instances[index])
            source_fitness = math.fsum((float(load) / evaluator.CAPACITY) ** 2 for load in loads) / expected
            reported_fitness = event.get("source_fitness")
            if (type(objective) is not int or isinstance(reported_fitness, bool)
                    or not isinstance(reported_fitness, (int, float)) or not math.isfinite(reported_fitness)
                    or not math.isclose(reported_fitness, source_fitness, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f"DeepACO BPP 实例 {index} 的箱数或来源适应度无效。")
        elif name == "cvrp":
            instance = instances[index]
            expected = evaluator.check_path(event.get("path", []), instance[:, 1:], instance[:, 0])
        else:
            prizes, weights = instances
            expected = evaluator.check_solution(event.get("solution", []), prizes[index], weights[index])
        if not math.isclose(objective, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"DeepACO 测试实例 {index} 的目标与项目评价不一致。")
        objectives.append(float(objective))
    average = complete.get("average")
    if (isinstance(average, bool) or not isinstance(average, (int, float)) or not math.isfinite(average)
            or not math.isclose(average, math.fsum(objectives) / 64, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("DeepACO 测试平均目标与逐实例结果不一致。")
    fitness = -average if direction == "min" else average
    supplied_fitness = complete.get("fitness")
    if (isinstance(supplied_fitness, bool) or not isinstance(supplied_fitness, (int, float))
            or not math.isfinite(supplied_fitness) or supplied_fitness != fitness):
        raise ValueError("DeepACO 测试 Fitness 与项目目标方向不一致。")
    return {"objectives": objectives, "average": average, "fitness": fitness, "start": start}


def evaluate_reference(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_contract = json.loads(DATA_CONTRACT.read_text(encoding="utf-8"))
    entry = load_reference_contract(config, data_contract, digest(DATA_CONTRACT))
    name, _method, _direction = TASKS[config["task"]]
    dataset = (ROOT / entry["path"]).resolve()
    expected_directory = (ROOT / "problems" / config["task"] / "dataset").resolve()
    if not dataset.is_relative_to(expected_directory) or digest(dataset) != entry["sha256"]:
        raise ValueError("测试数据位置或哈希与冻结合同不符。")
    resource_config = json.loads(RESOURCE_CONFIG.read_text(encoding="utf-8"))["resources"]
    if Path(sys.executable).resolve() != Path(resource_config["python_executable"]).resolve():
        raise ValueError("参考测试必须使用项目冻结的 Python。")
    if software_versions(name) != config["software_versions"]:
        raise ValueError("当前软件版本与冻结配置不一致。")
    resources = memory_and_priority()
    if resources["available_memory_bytes"] < resource_config["min_available_system_memory_bytes"]:
        raise ValueError("可用内存低于项目运行阈值。")
    snapshot, implementation = reference_implementation_snapshot(config["task"], config["checkpoint"])
    check_reference_implementation(config, implementation)
    protected = [config_path, DATA_CONTRACT, RESOURCE_CONFIG, dataset, *(Path(path) for path in implementation)]
    before = {str(path): digest(path) for path in protected}
    command = [resource_config["python_executable"], "-B", "-u", str(ROOT / "scripts/p1_deepaco_worker.py"),
               "--task", name, "--size", str(config["size"]), "--phase", "test",
               "--checkpoint", config["checkpoint"]]
    if name in ("tsp", "op"):
        command += ["--k-sparse", str(config["k_sparse"])]
    environment = os.environ.copy()
    environment.update(resource_config["thread_environment"])
    environment.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.pathsep.join([
        str(ROOT), r"D:\Tools\agentic-heuristic-p1-pyg", r"D:\Tools\agentic-heuristic-p1-numba",
        *resource_config["pythonpath"]])})
    root = ROOT / "outputs/p1_deepaco_test" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    record = {"status": "failed", "method": config["method"], "task": config["task"],
              "checkpoint": config["checkpoint"], "checkpoint_scale_mode": config["checkpoint_scale_mode"],
              "size": config["size"], "command": command, "resources": resources, "protected_sha256": before,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "timeout_seconds": config["whole_dataset_wall_timeout_seconds"],
              "process_tree_cpu_limit_seconds": config["process_tree_cpu_seconds"]}
    begin = time.monotonic()
    stdout_path, stderr_path = root / "stdout.txt", root / "stderr.txt"
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        try:
            with WindowsJob() as job:
                process = job.start(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                    stdout=stdout, stderr=stderr)
                record["pid"] = process.pid
                reason, record["process_tree"] = wait_bounded(job, process,
                    deadline=begin + config["whole_dataset_wall_timeout_seconds"],
                    cpu_limit=config["process_tree_cpu_seconds"])
                record["exit_code"] = process.returncode
                if reason:
                    record["status"] = reason
        except (OSError, RuntimeError) as exc:
            record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = time.monotonic() - begin
    audit_started, audit_cpu_started = time.monotonic(), time.process_time()
    try:
        if any(digest(path) != before[str(path)] for path in protected):
            raise ValueError("测试期间受保护文件发生变化。")
        if record["status"] == "failed" and record.get("exit_code") == 0:
            instances = load_reference_instances(name, dataset, config["size"])
            ants = data_contract["tasks"][config["task"]]["train"]["ants"]
            record.update(read_reference_complete(stdout_path, config, entry, dataset, instances,
                                                  load_task(name), snapshot, ants))
            record["status"] = "passed"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    record["audit_wall_seconds"] = time.monotonic() - audit_started
    record["audit_parent_cpu_seconds"] = time.process_time() - audit_cpu_started
    record["total_wall_seconds"] = time.monotonic() - begin
    with (root / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return root, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="已冻结的 DeepACO 单规模测试配置")
    args = parser.parse_args()
    try:
        root, record = evaluate_reference(args.config)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"参考测试无法启动：{exc}", file=sys.stderr)
        return 1
    print(f"参考测试状态：{record['status']}；记录：{root}")
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
