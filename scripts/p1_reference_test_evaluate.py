"""监督 ReEvo 专家启发式与 KGLS 适配的独立参考测试。"""

import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha1
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems.tsp_gls.lkh_reference import REFERENCE_PATH, load_reference  # noqa: E402
from scripts.p0_controller import check_modification_scope  # noqa: E402
from scripts.p0_runtime import digest, memory_and_priority  # noqa: E402
from scripts.p0_windows_job import WindowsJob  # noqa: E402
from scripts.p1_formal_matrix import MATRIX_PATH, verify_reference_config  # noqa: E402
from scripts.p1_runtime import DATA_CONTRACT  # noqa: E402
from scripts.p1_task_contract import load_task_contract  # noqa: E402
from scripts.p1_test_evaluate import PREFIX, read_complete, wait_bounded  # noqa: E402


RESOURCE_CONFIG = ROOT / "cfg/p0_tsp_aco.json"
SOURCE_COMMIT = "2f70042e6014d4d5d1255732dbfb312bc5d3abab"
SEED_BLOBS = {"tsp_aco": "98ac834fd7880c7730ad34e3ad825ecfc2824753",
              "op_aco": "ad1dcd8759268b3eba1cdb279c3f9fa958f5fb18",
              "cvrp_aco": "bb9df8e69694017ca74538bd365a028fc5a82015",
              "mkp_aco": "26a5373c82d4deb7ce329ed3dd4b4e244eea2e1a",
              "bpp_offline_aco": "5894db80cbc9e46fd1a261d646a96876021e6a4b",
              "tsp_gls": "4b42109cabae3a488210c2569a3d3b6a26923847"}


def method_for(task):
    if task not in SEED_BLOBS:
        raise ValueError("参考任务无效。")
    return "kgls_reevo_float64_adapted" if task == "tsp_gls" else "aco_reevo_seed_float64_adapted"


def source_paths(task):
    if task not in SEED_BLOBS:
        raise ValueError("参考任务无效。")
    reference_dir = "gls_reference" if task == "tsp_gls" else "aco_reference"
    return (ROOT / "baselines" / reference_dir / f"{task}.py",
            ROOT / "baselines/reevo/upstream/prompts" / task / "seed_func.txt")


def verify_source(task):
    reference, seed = source_paths(task)
    data = seed.read_bytes()
    if sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() != SEED_BLOBS[task]:
        raise ValueError("ReEvo 来源种子 Git blob 不符。")

    def function(text, name):
        nodes = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(nodes) != 1:
            raise ValueError("参考启发式函数身份不明确。")
        nodes[0].name = "heuristics"
        return ast.dump(nodes[0], include_attributes=False)

    code = reference.read_text(encoding="utf-8")
    if function(code, "heuristics_v2") != function(data.decode("utf-8"), "heuristics_v1"):
        raise ValueError("参考启发式与 ReEvo 来源种子不一致。")
    load_task_contract(task).parse_proposal(f"<thought>reference</thought><code>{code}</code>")
    check_modification_scope(code)
    return reference, seed


def implementation_snapshot(task):
    reference, seed = verify_source(task)
    directory = ROOT / "problems" / task
    solver = directory / ("gls.py" if task == "tsp_gls" else "aco.py")
    paths = [reference, seed, directory / "eval.py", solver, Path(__file__).resolve(),
             ROOT / "scripts/p1_test_evaluate.py", ROOT / "scripts/p1_task_contract.py",
             ROOT / "scripts/p0_controller.py", ROOT / "scripts/p0_runtime.py",
             ROOT / "scripts/p0_windows_job.py", ROOT / "scripts/p1_runtime.py",
             ROOT / "cfg/problem" / f"{task}.yaml",
             ROOT / "prompts" / task / "func_desc.txt", ROOT / "prompts" / task / "func_signature.txt"]
    if task == "tsp_gls":
        paths.append(directory / "lkh_reference.py")
    return {str(path): digest(path) for path in paths}


def software_versions(task, resources):
    names = ["numpy", "scipy", "torch"]
    if task in ("tsp_gls", "bpp_offline_aco"):
        names += ["numba", "llvmlite"]
    paths = [r"D:\Tools\agentic-heuristic-p1-numba", *resources["pythonpath"]]
    found = {}
    for distribution in metadata.distributions(path=paths):
        name = distribution.metadata.get("Name", "").lower().replace("_", "-")
        found.setdefault(name, distribution.version)
    for name in names:
        if name not in found:
            raise ValueError(f"参考测试运行环境缺少软件版本信息：{name}")
    return {"python": platform.python_version(), **{name: found[name] for name in names}}


def load_reference_contract(config, data_contract, data_sha256, resource_sha256):
    task = config.get("task") if isinstance(config, dict) else None
    if (not isinstance(config, dict) or task not in SEED_BLOBS or config.get("status") != "frozen"
            or config.get("method") != method_for(task) or config.get("source_commit") != SOURCE_COMMIT
            or config.get("compute_dtype") != "float64" or config.get("solver_repeats_per_instance") != 1
            or config.get("data_contract_sha256") != data_sha256 or config.get("resource_config_sha256") != resource_sha256
            or not isinstance(data_contract, dict) or data_contract.get("status") != "frozen"
            or data_contract.get("contract_id") != "p1_data_20261104_v1"):
        raise ValueError("专家参考的方法、数据、资源或配置尚未冻结。")
    size = config.get("size")
    if type(size) is not int:
        raise ValueError("参考测试规模无效。")
    for key in ("whole_dataset_wall_timeout_seconds", "process_tree_cpu_seconds"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("完整数据集墙钟和进程树 CPU 预算必须为有限正数。")
    if (not isinstance(config.get("implementation_sha256"), dict) or not config["implementation_sha256"]
            or not isinstance(config.get("software_versions"), dict)):
        raise ValueError("参考实现哈希或软件版本尚未冻结。")
    entries = [item for item in data_contract.get("tasks", {}).get(task, {}).get("test", []) if item.get("size") == size]
    if len(entries) != 1 or entries[0].get("instances") != 64:
        raise ValueError("数据合同中没有唯一的 64 实例测试规模。")
    return entries[0]


def verify_tsp_routes(stdout_path, dataset, size):
    points = np.load(dataset, allow_pickle=False)
    if points.shape != (64, size, 2) or points.dtype != np.float64 or not np.isfinite(points).all():
        raise ValueError("参考测试坐标无效。")
    events = [json.loads(line[len(PREFIX):]) for line in stdout_path.read_text(encoding="utf-8").splitlines()
              if line.startswith(PREFIX)]
    rows = [event for event in events if isinstance(event, dict) and event.get("event") == "instance"]
    if len(rows) != 64:
        raise ValueError("参考测试路线数量无效。")
    for index, row in enumerate(rows):
        tour = row.get("tour")
        if (not isinstance(tour, list) or len(tour) != size or any(type(node) is not int for node in tour)
                or sorted(tour) != list(range(size))):
            raise ValueError(f"参考测试实例 {index} 的路线无效。")
        expected = math.fsum(math.dist(points[index, a], points[index, b])
                             for a, b in zip(tour, tour[1:] + tour[:1]))
        if not math.isclose(row["objective"], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"参考测试实例 {index} 的路线长度不一致。")


def evaluate_reference(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_reference_config(config, "expert")
    data = json.loads(DATA_CONTRACT.read_text(encoding="utf-8"))
    entry = load_reference_contract(config, data, digest(DATA_CONTRACT), digest(RESOURCE_CONFIG))
    task = config["task"]
    dataset = (ROOT / entry["path"]).resolve()
    if not dataset.is_relative_to((ROOT / "problems" / task / "dataset").resolve()) or digest(dataset) != entry["sha256"]:
        raise ValueError("参考测试集位置或哈希不符。")
    resource_config = json.loads(RESOURCE_CONFIG.read_text(encoding="utf-8"))["resources"]
    if Path(sys.executable).resolve() != Path(resource_config["python_executable"]).resolve():
        raise ValueError("参考测试必须使用项目冻结的 Python。")
    snapshot = implementation_snapshot(task)
    if config["implementation_sha256"] != snapshot or config["software_versions"] != software_versions(task, resource_config):
        raise ValueError("参考测试实现或软件版本与冻结配置不符。")
    reference, _ = source_paths(task)
    lkh = load_reference(config["size"], entry["sha256"]) if task == "tsp_gls" else None
    if lkh is not None and config.get("lkh_reference_sha256") != lkh["sha256"]:
        raise ValueError("KGLS 的 LKH 参考身份不符。")
    resources = memory_and_priority()
    if resources["available_memory_bytes"] < resource_config["min_available_system_memory_bytes"]:
        raise ValueError("可用内存低于项目运行阈值。")
    protected = [config_path, DATA_CONTRACT, RESOURCE_CONFIG, MATRIX_PATH, dataset, *(Path(path) for path in snapshot)]
    if lkh is not None:
        protected.append(REFERENCE_PATH)
    before = {str(path): digest(path) for path in protected}
    root = ROOT / "outputs/p1_reference_test" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    module_dir = root / "problems" / task
    module_dir.mkdir(parents=True, exist_ok=False)
    copied = module_dir / "gpt.py"
    with copied.open("xb") as stream:
        stream.write(reference.read_bytes())
    if digest(copied) != digest(reference):
        raise ValueError("参考函数副本不一致。")
    evaluator = ROOT / "problems" / task / "eval.py"
    command = [resource_config["python_executable"], "-B", "-u", str(evaluator), str(config["size"]), str(root), "test"]
    env = os.environ.copy()
    env.update(resource_config["thread_environment"])
    env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join([str(ROOT), r"D:\Tools\agentic-heuristic-p1-numba", *resource_config["pythonpath"]])})
    record = {"status": "failed", "method": config["method"], "task": task, "size": config["size"],
              "matrix_contract_sha256": config["matrix_contract_sha256"],
              "source_commit": SOURCE_COMMIT, "reference_sha256": digest(reference), "command": command,
              "resources": resources, "protected_sha256": before, "started_at_utc": datetime.now(timezone.utc).isoformat(),
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
        if any(digest(path) != before[str(path)] for path in protected) or digest(copied) != digest(reference):
            raise ValueError("参考测试期间受保护文件发生变化。")
        if record["status"] == "failed" and record.get("exit_code") == 0:
            manifest = {"task": task, "candidate_sha256": digest(reference)}
            ants = data["tasks"][task]["train"].get("ants")
            record.update(read_complete(stdout_path, entry, manifest, load_task_contract(task).direction, dataset, ants, lkh))
            if task in ("tsp_aco", "tsp_gls"):
                verify_tsp_routes(stdout_path, dataset, config["size"])
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
    parser.add_argument("--config", required=True, help="已冻结的专家参考单规模测试配置")
    args = parser.parse_args()
    try:
        root, record = evaluate_reference(args.config)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"专家参考测试无法启动：{exc}", file=sys.stderr)
        return 1
    print(f"专家参考测试：{record['status']}；记录：{root}")
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
