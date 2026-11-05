"""监督 GNNGLS 单规模参考测试；仅接受任务 4 冻结后的配置。"""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.gnngls.reference import ASSET_SHA256, ISOLATED, ROOT as SOURCE_ROOT, SOURCE_BLOBS, SOURCE_COMMIT, solve, verify_sources  # noqa: E402
from problems.tsp_gls.lkh_reference import REFERENCE_PATH, load_reference, reached_reference  # noqa: E402
from scripts.p0_runtime import digest, memory_and_priority  # noqa: E402
from scripts.p0_windows_job import WindowsJob  # noqa: E402
from scripts.p1_runtime import DATA_CONTRACT  # noqa: E402
from scripts.p1_test_evaluate import wait_bounded  # noqa: E402


PREFIX = "P1_JSON "
METHOD = "gnngls_05b5aa1b_cpu_float64_adapted"
RESOURCE_CONFIG = ROOT / "cfg/p0_tsp_aco.json"
VERSION_NAMES = ("numpy", "scipy", "torch", "torchdata", "dgl", "networkx", "pandas")


def software_versions(resources_config):
    found = {}
    paths = [str(ISOLATED), *resources_config["pythonpath"]]
    for distribution in metadata.distributions(path=paths):
        name = distribution.metadata.get("Name", "").lower().replace("_", "-")
        found.setdefault(name, distribution.version)
    if any(name not in found for name in VERSION_NAMES):
        raise ValueError("GNNGLS 隔离环境缺少软件版本信息。")
    return {"python": platform.python_version(), **{name: found[name] for name in VERSION_NAMES}}


def implementation_snapshot():
    verify_sources()
    paths = [Path(__file__).resolve(), ROOT / "baselines/gnngls/reference.py", ROOT / "scripts/p1_gnngls_check.py",
             ROOT / "problems/tsp_gls/lkh_reference.py",
             ROOT / "scripts/p0_runtime.py", ROOT / "scripts/p0_windows_job.py", ROOT / "scripts/p1_runtime.py",
             ROOT / "scripts/p1_test_evaluate.py", *(SOURCE_ROOT / name for name in SOURCE_BLOBS),
             *(SOURCE_ROOT / name for name in ASSET_SHA256)]
    return {str(path): digest(path) for path in paths}


def load_reference_contract(config, data_contract, data_sha256, resource_sha256):
    if (not isinstance(config, dict) or config.get("status") != "frozen"
            or not isinstance(data_contract, dict) or data_contract.get("status") != "frozen"
            or data_contract.get("contract_id") != "p1_data_20261104_v1"
            or config.get("task") != "tsp_gls" or config.get("method") != METHOD
            or config.get("source_commit") != SOURCE_COMMIT or config.get("compute_dtype") != "float64"
            or config.get("device") != "cpu" or config.get("data_contract_sha256") != data_sha256
            or config.get("resource_config_sha256") != resource_sha256
            or config.get("solver_repeats_per_instance") != 1):
        raise ValueError("GNNGLS 方法、数据、资源或测试配置尚未冻结。")
    size, checkpoint, scaler = config.get("size"), config.get("checkpoint_size"), config.get("scaler_size")
    if (type(size) is not int or size not in (20, 50, 100, 200)
            or type(checkpoint) is not int or checkpoint not in (20, 50, 100)
            or type(scaler) is not int or scaler not in (20, 50, 100)
            or config.get("checkpoint_scale_mode") != ("matched" if checkpoint == size else "cross_scale")
            or config.get("scaler_scale_mode") != ("matched" if scaler == size else "cross_scale")
            or type(config.get("perturbation_moves")) is not int or config["perturbation_moves"] < 1):
        raise ValueError("GNNGLS 规模、权重、缩放器或扰动参数无效。")
    for key in ("search_time_limit_seconds", "whole_dataset_wall_timeout_seconds", "process_tree_cpu_seconds"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("GNNGLS 搜索和完整数据集资源预算必须为有限正数。")
    if (not isinstance(config.get("implementation_sha256"), dict) or not config["implementation_sha256"]
            or not isinstance(config.get("software_versions"), dict)):
        raise ValueError("GNNGLS 实现哈希或软件版本尚未冻结。")
    entries = [item for item in data_contract.get("tasks", {}).get("tsp_gls", {}).get("test", [])
               if item.get("size") == size]
    if len(entries) != 1 or entries[0].get("instances") != 64:
        raise ValueError("数据合同中没有唯一的 64 实例测试规模。")
    seed = entries[0].get("evaluation_seed_base")
    if type(seed) is not int or seed < 0 or seed + 63 >= 2**32:
        raise ValueError("逐实例随机种子无效。")
    return entries[0]


def load_instances(dataset, size):
    instances = np.load(dataset, allow_pickle=False)
    if (instances.shape != (64, size, 2) or instances.dtype != np.float64
            or not np.isfinite(instances).all() or (instances < 0).any() or (instances >= 1).any()):
        raise ValueError("GNNGLS 测试坐标的形状、范围或精度不符合合同。")
    return instances


def emit(event):
    print(PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def synthetic_inputs(size, seed, mode, time_limit, perturbations, checkpoint_size, scaler_size):
    instances = np.random.RandomState(seed).random_sample((2, size, 2))
    entry = {"sha256": f"synthetic:{size}:{seed}", "evaluation_seed_base": seed}
    config = {"checkpoint_size": checkpoint_size, "scaler_size": scaler_size,
              "search_time_limit_seconds": time_limit, "perturbation_moves": perturbations}
    value = size * math.sqrt(2) + 1 if mode == "initial_stop" else 1e-12
    reference = {"sha256": "synthetic", "mean": value, "objectives": [value, value]}
    return instances, entry, config, reference


def run_instances(instances, entry, config, reference, emit_event=emit):
    import torch
    count, size = instances.shape[:2]
    emit_event({"event": "start", "method": METHOD, "source_commit": SOURCE_COMMIT,
                "dataset_sha256": entry["sha256"], "size": size, "instance_count": count,
                "seed_base": entry["evaluation_seed_base"], "checkpoint_size": config["checkpoint_size"],
                "scaler_size": config["scaler_size"], "compute_dtype": "float64", "device": "cpu",
                "time_limit": config["search_time_limit_seconds"], "perturbation_moves": config["perturbation_moves"],
                "reference_sha256": reference["sha256"], "reference_mean": reference["mean"]})
    objectives = []
    for index, points in enumerate(instances):
        seed = entry["evaluation_seed_base"] + index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        begin = time.monotonic()
        result = solve(points, checkpoint_size=config["checkpoint_size"], scaler_size=config["scaler_size"],
                       time_limit=config["search_time_limit_seconds"], perturbation_moves=config["perturbation_moves"],
                       reference=reference["objectives"][index])
        objectives.append(result["objective"])
        emit_event({"event": "instance", "index": index, "seed": seed,
                    "reference": reference["objectives"][index], "instance_wall_seconds": time.monotonic() - begin,
                    **result})
    average = math.fsum(objectives) / count
    emit_event({"event": "complete", "instance_count": count, "average": average,
                "reference_mean": reference["mean"],
                "lkh_gap_percent": (average / reference["mean"] - 1.0) * 100.0})


def read_complete(stdout_path, instances, entry, config, reference):
    events = [json.loads(line[len(PREFIX):]) for line in stdout_path.read_text(encoding="utf-8").splitlines()
              if line.startswith(PREFIX)]
    count, size = instances.shape[:2]
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("GNNGLS 测试事件不是 JSON 对象。")
    if [event.get("event") for event in events] != ["start"] + ["instance"] * count + ["complete"]:
        raise ValueError("GNNGLS 测试事件缺失、重复或顺序错误。")
    start, complete = events[0], events[-1]
    expected = {"method": METHOD, "source_commit": SOURCE_COMMIT, "dataset_sha256": entry["sha256"],
                "size": size, "instance_count": count, "seed_base": entry["evaluation_seed_base"],
                "checkpoint_size": config["checkpoint_size"], "scaler_size": config["scaler_size"],
                "compute_dtype": "float64", "device": "cpu", "time_limit": config["search_time_limit_seconds"],
                "perturbation_moves": config["perturbation_moves"], "reference_sha256": reference["sha256"],
                "reference_mean": reference["mean"]}
    if any(start.get(key) != value for key, value in expected.items()):
        raise ValueError("GNNGLS 方法、数据、种子或参考身份不符。")
    objectives, stopping_reasons = [], []
    for index, event in enumerate(events[1:-1]):
        route, objective = event.get("route"), event.get("objective")
        ref = reference["objectives"][index]
        if (event.get("index") != index or event.get("seed") != entry["evaluation_seed_base"] + index
                or event.get("reference") != ref or event.get("checkpoint_size") != config["checkpoint_size"]
                or event.get("scaler_size") != config["scaler_size"] or event.get("compute_dtype") != "float64"
                or event.get("scaler_source_sklearn") != "0.24.2"
                or not isinstance(route, list) or len(route) != size or any(type(node) is not int for node in route)
                or sorted(route) != list(range(size)) or route[0] != 0):
            raise ValueError(f"GNNGLS 实例 {index} 的身份、参考或路线无效。")
        actual = math.fsum(math.dist(instances[index, a], instances[index, b])
                           for a, b in zip(route, route[1:] + route[:1]))
        for value in (objective, event.get("source_cost")):
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or not math.isclose(value, actual, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f"GNNGLS 实例 {index} 的目标与路线不一致。")
        reason = "reference_reached" if reached_reference(objective, ref) else "time_limit"
        if (event.get("stop_reason") != reason or type(event.get("iterations")) is not int or event["iterations"] < 0
                or type(event.get("graph_cache_hit")) is not bool or event["graph_cache_hit"] != (index > 0)):
            raise ValueError(f"GNNGLS 实例 {index} 的停止原因或迭代数无效。")
        for key in ("prediction_seconds", "elapsed_seconds", "instance_wall_seconds", "graph_setup_seconds"):
            value = event.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"GNNGLS 实例 {index} 的耗时无效。")
        objectives.append(float(objective))
        stopping_reasons.append(reason)
    average = math.fsum(objectives) / count
    gap = (average / reference["mean"] - 1.0) * 100.0
    if (complete.get("instance_count") != count or complete.get("reference_mean") != reference["mean"]
            or not math.isclose(complete.get("average"), average, rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(complete.get("lkh_gap_percent"), gap, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("GNNGLS 平均目标或 LKH 参考差距不一致。")
    return {"objectives": objectives, "average": average, "lkh_gap_percent": gap,
            "stopping_reasons": stopping_reasons, "start": start}


def resolve_input(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = json.loads(DATA_CONTRACT.read_text(encoding="utf-8"))
    entry = load_reference_contract(config, data, digest(DATA_CONTRACT), digest(RESOURCE_CONFIG))
    dataset = (ROOT / entry["path"]).resolve()
    if (not dataset.is_relative_to((ROOT / "problems/tsp_gls/dataset").resolve())
            or digest(dataset) != entry["sha256"]):
        raise ValueError("GNNGLS 测试集位置或哈希不符。")
    reference = load_reference(config["size"], entry["sha256"])
    if config.get("lkh_reference_sha256") != reference["sha256"]:
        raise ValueError("GNNGLS 测试的 LKH 参考身份不符。")
    return config_path, config, entry, dataset, reference


def evaluate_reference(config_path):
    config_path, config, entry, dataset, reference = resolve_input(config_path)
    resource_config = json.loads(RESOURCE_CONFIG.read_text(encoding="utf-8"))["resources"]
    if Path(sys.executable).resolve() != Path(resource_config["python_executable"]).resolve():
        raise ValueError("参考测试必须使用项目冻结的 Python。")
    if software_versions(resource_config) != config["software_versions"] or implementation_snapshot() != config["implementation_sha256"]:
        raise ValueError("GNNGLS 软件版本或实现哈希与冻结配置不符。")
    resources = memory_and_priority()
    if resources["available_memory_bytes"] < resource_config["min_available_system_memory_bytes"]:
        raise ValueError("可用内存低于项目运行阈值。")
    protected = [config_path, DATA_CONTRACT, RESOURCE_CONFIG, dataset, REFERENCE_PATH,
                 *(Path(path) for path in config["implementation_sha256"])]
    before = {str(path): digest(path) for path in protected}
    command = [resource_config["python_executable"], "-B", "-u", str(Path(__file__).resolve()),
               "--worker", str(config_path)]
    env = os.environ.copy()
    env.update(resource_config["thread_environment"])
    env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "DGLBACKEND": "pytorch",
                "PYTHONPATH": os.pathsep.join([str(ROOT), str(ISOLATED), *resource_config["pythonpath"]])})
    root = ROOT / "outputs/p1_gnngls_test" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    record = {"status": "failed", "method": METHOD, "task": "tsp_gls", "size": config["size"],
              "command": command, "resources": resources, "protected_sha256": before,
              "checkpoint_size": config["checkpoint_size"], "scaler_size": config["scaler_size"],
              "checkpoint_scale_mode": config["checkpoint_scale_mode"], "scaler_scale_mode": config["scaler_scale_mode"],
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "timeout_seconds": config["whole_dataset_wall_timeout_seconds"],
              "process_tree_cpu_limit_seconds": config["process_tree_cpu_seconds"]}
    started = time.monotonic()
    stdout_path, stderr_path = root / "stdout.txt", root / "stderr.txt"
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        try:
            with WindowsJob() as job:
                process = job.start(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                record["pid"] = process.pid
                reason, record["process_tree"] = wait_bounded(job, process,
                    deadline=started + config["whole_dataset_wall_timeout_seconds"], cpu_limit=config["process_tree_cpu_seconds"])
                record["exit_code"] = process.returncode
                if reason:
                    record["status"] = reason
        except (OSError, RuntimeError) as exc:
            record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = time.monotonic() - started
    audit_start, audit_cpu = time.monotonic(), time.process_time()
    try:
        if any(digest(path) != before[str(path)] for path in protected):
            raise ValueError("GNNGLS 参考测试期间受保护文件发生变化。")
        if record["status"] == "failed" and record.get("exit_code") == 0:
            record.update(read_complete(stdout_path, load_instances(dataset, config["size"]), entry, config, reference))
            record["status"] = "passed"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    record["audit_wall_seconds"] = time.monotonic() - audit_start
    record["audit_parent_cpu_seconds"] = time.process_time() - audit_cpu
    record["total_wall_seconds"] = time.monotonic() - started
    with (root / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return root, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="已冻结的 GNNGLS 单规模测试配置")
    group.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            dll_dir = ISOLATED / "Library/bin"
            dll_handle = os.add_dll_directory(str(dll_dir)) if dll_dir.is_dir() else None
            _, config, entry, dataset, reference = resolve_input(args.worker)
            run_instances(load_instances(dataset, config["size"]), entry, config, reference)
            if dll_handle is not None:
                dll_handle.close()
            return 0
        root, record = evaluate_reference(args.config)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"GNNGLS 参考测试无法启动：{exc}", file=sys.stderr)
        return 1
    print(f"GNNGLS 参考测试：{record['status']}；记录：{root}")
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
