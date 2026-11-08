"""监督已冻结候选的完整测试集评价；不参与研发选优。"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.p0_controller import check_modification_scope  # noqa: E402
from scripts.p0_runtime import digest, memory_and_priority  # noqa: E402
from scripts.p0_windows_job import WindowsJob  # noqa: E402
from scripts.p1_formal_matrix import MATRIX_PATH, verify_test_config, verify_training_config  # noqa: E402
from scripts.p1_runtime import DATA_CONTRACT  # noqa: E402
from scripts.p1_task_contract import load_task_contract  # noqa: E402
from problems.tsp_gls.lkh_reference import (ABS_TOLERANCE, REL_TOLERANCE, REFERENCE_PATH,
                                            load_reference, reached_reference)  # noqa: E402


PREFIX = "P1_JSON "


def load_test_contract(config, manifest, data_contract):
    if (not isinstance(config, dict) or config.get("status") != "frozen"
            or data_contract.get("status") != "frozen" or data_contract.get("contract_id") != "p1_data_20261104_v1"
            or manifest.get("status") != "frozen_after_training"
            or config.get("data_contract_sha256") != digest(DATA_CONTRACT)
            or config.get("task") != manifest.get("task")
            or config.get("prompt_condition") != manifest.get("prompt_condition")
            or config.get("solver_repeats_per_instance") != 1):
        raise ValueError("候选、数据或测试合同未冻结或身份不一致。")
    size = config.get("size")
    if type(size) is not int:
        raise ValueError("测试规模无效。")
    entries = [entry for entry in data_contract.get("tasks", {}).get(config["task"], {}).get("test", [])
               if entry.get("size") == size]
    if len(entries) != 1 or entries[0].get("instances") != 64:
        raise ValueError("测试规模未在冻结数据合同中唯一登记。")
    timeout, cpu = config.get("whole_dataset_wall_timeout_seconds"), config.get("process_tree_cpu_seconds")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0
            or isinstance(cpu, bool) or not isinstance(cpu, (int, float)) or not math.isfinite(cpu) or cpu <= 0):
        raise ValueError("完整测试集时限或进程树 CPU 预算无效。")
    return entries[0]


def read_complete(stdout_path, entry, manifest, direction, dataset, expected_ants, lkh_reference=None):
    events = [json.loads(line[len(PREFIX):]) for line in stdout_path.read_text(encoding="utf-8").splitlines()
              if line.startswith(PREFIX)]
    if [event.get("event") for event in events] != ["start"] + ["instance"] * 64 + ["complete"]:
        raise ValueError("测试事件缺失、重复或顺序错误。")
    start, complete = events[0], events[-1]
    if (Path(start.get("dataset", "")).resolve() != dataset.resolve()
            or start.get("candidate_sha256") != manifest["candidate_sha256"]
            or start.get("seed_base") != entry["evaluation_seed_base"]
            or start.get("n_iterations") != entry["iterations"]
            or complete.get("instance_count") != 64):
        raise ValueError("测试数据、候选、种子、迭代或实例数量不符。")
    if expected_ants is not None and start.get("n_ants") != expected_ants:
        raise ValueError("ACO 测试蚂蚁数量不符。")
    if "perturbation_moves" in entry and start.get("perturbation_moves") != entry["perturbation_moves"]:
        raise ValueError("GLS 测试扰动次数不符。")
    if "max_path_length" in entry and start.get("max_len") != entry["max_path_length"]:
        raise ValueError("OP 测试路径长度限制不符。")
    is_gls = manifest.get("task") == "tsp_gls"
    if is_gls and (lkh_reference is None or direction != "min"
                   or start.get("lkh_reference_sha256") != lkh_reference["sha256"]
                   or start.get("lkh_reference_mean") != lkh_reference["mean"]
                   or start.get("lkh_absolute_tolerance") != ABS_TOLERANCE
                   or start.get("lkh_relative_tolerance") != REL_TOLERANCE):
        raise ValueError("GLS 的 LKH 参考或停止容差不符。")
    objectives, actual_iterations, stopping_reasons = [], [], []
    for index, event in enumerate(events[1:-1]):
        objective = event.get("objective")
        if (event.get("index") != index or event.get("seed") != entry["evaluation_seed_base"] + index
                or isinstance(objective, bool) or not isinstance(objective, (int, float)) or not math.isfinite(objective)):
            raise ValueError(f"测试实例 {index} 的结果无效或不完整。")
        iterations = event.get("iterations")
        if is_gls:
            reference_value = lkh_reference["objectives"][index]
            if (type(iterations) is not int or not 0 <= iterations <= entry["iterations"]
                    or event.get("lkh_reference") != reference_value):
                raise ValueError(f"GLS 实例 {index} 的迭代数或参考无效。")
            reached = reached_reference(objective, reference_value)
            reason = "reference_reached" if reached else "iteration_limit"
            if event.get("stop_reason") != reason or (not reached and iterations != entry["iterations"]):
                raise ValueError(f"GLS 实例 {index} 的停止原因不符。")
            actual_iterations.append(iterations)
            stopping_reasons.append(reason)
        elif iterations != entry["iterations"]:
            raise ValueError(f"测试实例 {index} 的迭代数不符。")
        objectives.append(float(objective))
    average = complete.get("average")
    if (isinstance(average, bool) or not isinstance(average, (int, float)) or not math.isfinite(average)
            or not math.isclose(average, math.fsum(objectives) / 64, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("测试平均目标与逐实例结果不一致。")
    fitness = -average if direction == "min" else average
    if "fitness" in complete:
        supplied = complete["fitness"]
        if (isinstance(supplied, bool) or not isinstance(supplied, (int, float))
                or not math.isfinite(supplied) or supplied != fitness):
            raise ValueError("测试 Fitness 与目标方向不一致。")
    output = {"objectives": objectives, "average": average, "fitness": fitness, "start": start}
    if is_gls:
        gap = (average / lkh_reference["mean"] - 1.0) * 100.0
        supplied_gap = complete.get("lkh_gap_percent")
        if (complete.get("lkh_reference_mean") != lkh_reference["mean"]
                or isinstance(supplied_gap, bool) or not isinstance(supplied_gap, (int, float))
                or not math.isfinite(supplied_gap) or not math.isclose(supplied_gap, gap, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("GLS 的 LKH 参考均值或差距不符。")
        output.update({"lkh_reference_mean": lkh_reference["mean"], "lkh_gap_percent": gap,
                       "actual_iterations": actual_iterations, "stopping_reasons": stopping_reasons})
    return output


def wait_bounded(job, process, *, deadline, cpu_limit):
    reason = None
    while True:
        usage = job.accounting()
        if usage["active_processes"] == 0:
            if usage["cpu_seconds"] >= cpu_limit:
                reason = "cpu_budget"
            elif time.monotonic() >= deadline:
                reason = "timeout"
            break
        if usage["cpu_seconds"] >= cpu_limit:
            reason = "cpu_budget"
        elif time.monotonic() >= deadline:
            reason = "timeout"
        if reason:
            job.terminate()
            break
        time.sleep(0.05)
    if reason:
        terminate_deadline = time.monotonic() + 10
        while job.accounting()["active_processes"]:
            if time.monotonic() >= terminate_deadline:
                raise RuntimeError("终止后测试进程树仍有进程存活。")
            time.sleep(0.05)
    process.wait()
    return reason, job.accounting()


def evaluate(candidate_dir, config_path):
    candidate_dir, config_path = Path(candidate_dir).resolve(), Path(config_path).resolve()
    if not candidate_dir.is_relative_to((ROOT / "outputs/p1_frozen_candidates").resolve()):
        raise ValueError("候选不在正式冻结目录。")
    manifest_path, candidate_path = candidate_dir / "manifest.json", candidate_dir / "candidate.py"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_id = manifest.get("source_candidate_id")
    if not isinstance(candidate_id, str) or re.fullmatch(r"c[0-9]{4,}", candidate_id) is None:
        raise ValueError("冻结候选来源 ID 无效。")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_test_config(config)
    data_contract = json.loads(DATA_CONTRACT.read_text(encoding="utf-8"))
    entry = load_test_contract(config, manifest, data_contract)
    if digest(candidate_path) != manifest.get("candidate_sha256"):
        raise ValueError("冻结候选代码哈希不符。")
    source_run = Path(manifest["source_run"]).resolve()
    train_config = source_run / "config.json"
    if digest(train_config) != manifest.get("source_config_sha256"):
        raise ValueError("原研发运行配置已变化。")
    training = json.loads(train_config.read_text(encoding="utf-8"))
    verify_training_config(training)
    if training.get("status") != "frozen" or training.get("method", {}).get("id") != manifest.get("method"):
        raise ValueError("原研发运行的状态或方法身份不符。")
    if (manifest.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
            or manifest.get("matrix_contract_sha256") != training["matrix_contract_sha256"]
            or manifest.get("formal_replicate") != training["formal_replicate"]
            or manifest.get("selection_seed") != training["task"]["selection_seed"]):
        raise ValueError("候选、研发和测试配置的矩阵或重复索引不一致。")
    train_events = source_run / "events.jsonl"
    source_candidate = source_run / "candidates" / manifest["source_candidate_id"]
    train_result, train_stdout = source_candidate / "result.json", source_candidate / "stdout.txt"
    if (digest(train_events) != manifest.get("source_events_sha256")
            or digest(train_result) != manifest.get("source_result_sha256")
            or digest(train_stdout) != manifest.get("source_stdout_sha256")):
        raise ValueError("原研发事件或候选评价记录已变化。")
    contract = load_task_contract(manifest["task"], black_box=manifest["prompt_condition"] == "black_box")
    code = candidate_path.read_text(encoding="utf-8")
    contract.parse_proposal(f"<thought>frozen candidate</thought><code>{code}</code>")
    check_modification_scope(code)
    dataset = ROOT / entry["path"]
    if digest(dataset) != entry["sha256"]:
        raise ValueError("测试集哈希与冻结合同不符。")
    task_dir = ROOT / "problems" / manifest["task"]
    evaluator = task_dir / ("eval_black_box.py" if contract.black_box else "eval.py")
    common_evaluator = task_dir / "eval.py"
    solver = task_dir / ("gls.py" if manifest["task"] == "tsp_gls" else "aco.py")
    lkh_reference = load_reference(entry["size"], entry["sha256"]) if manifest["task"] == "tsp_gls" else None
    reference_code = task_dir / "lkh_reference.py" if lkh_reference is not None else None
    events = [json.loads(line) for line in train_events.read_text(encoding="utf-8").splitlines() if line.strip()]
    preflight = [event for event in events if event.get("event") == "preflight"]
    source_hashes = preflight[0].get("code_sha256", {}) if len(preflight) == 1 else {}
    training_sources = [DATA_CONTRACT, evaluator, common_evaluator, solver]
    if reference_code is not None:
        training_sources.append(reference_code)
    if any(source_hashes.get(str(path)) != digest(path) for path in training_sources):
        raise ValueError("测试评价代码、求解器或数据合同与研发运行时不一致。")
    protected = [candidate_path, manifest_path, config_path, DATA_CONTRACT, MATRIX_PATH, train_config, train_events,
                 train_result, train_stdout, dataset, evaluator, common_evaluator, solver,
                 Path(__file__).resolve()]
    if lkh_reference is not None:
        protected.extend([reference_code, REFERENCE_PATH])
    before = {str(path): digest(path) for path in protected}
    resources = memory_and_priority()
    if resources["available_memory_bytes"] < training["resources"]["min_available_system_memory_bytes"]:
        raise ValueError("可用内存低于研发运行的资源阈值。")
    root = ROOT / "outputs/p1_test" / manifest["method"] / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    module_dir = root / "problems" / manifest["task"]
    module_dir.mkdir(parents=True, exist_ok=False)
    copied = module_dir / "gpt.py"
    with copied.open("xb") as stream:
        stream.write(candidate_path.read_bytes())
    if digest(copied) != manifest["candidate_sha256"]:
        raise ValueError("测试副本与冻结候选不一致。")
    command = [training["resources"]["python_executable"], "-B", "-u", str(evaluator), str(config["size"]), str(root), "test"]
    env = os.environ.copy()
    env.update(training["resources"]["thread_environment"])
    env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join(training["resources"]["pythonpath"])})
    stdout_path, stderr_path = root / "stdout.txt", root / "stderr.txt"
    record = {"status": "failed", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "method": manifest["method"], "candidate_id": manifest["source_candidate_id"],
              "matrix_contract_sha256": manifest["matrix_contract_sha256"],
              "task": manifest["task"], "prompt_condition": manifest["prompt_condition"],
              "formal_replicate": manifest["formal_replicate"], "source_run": manifest["source_run"],
              "candidate_sha256": manifest["candidate_sha256"], "size": config["size"],
              "command": command, "resources": resources, "protected_sha256": before,
              "timeout_seconds": config["whole_dataset_wall_timeout_seconds"],
              "process_tree_cpu_limit_seconds": config["process_tree_cpu_seconds"]}
    begin = time.monotonic()
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        try:
            with WindowsJob() as job:
                process = job.start(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                record["pid"] = process.pid
                reason, record["process_tree"] = wait_bounded(job, process,
                    deadline=begin + config["whole_dataset_wall_timeout_seconds"], cpu_limit=config["process_tree_cpu_seconds"])
                record["exit_code"] = process.returncode
                if reason:
                    record["status"] = reason
        except (OSError, RuntimeError) as exc:
            record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = time.monotonic() - begin
    try:
        changed = [str(path) for path in protected if digest(path) != before[str(path)]]
        if digest(copied) != manifest["candidate_sha256"]:
            changed.append(str(copied))
        if changed:
            raise ValueError(f"受保护文件发生变化：{changed}")
        if record["status"] == "failed" and record.get("exit_code") == 0:
            expected_ants = data_contract["tasks"][manifest["task"]]["train"]["ants"]
            record.update(read_complete(stdout_path, entry, manifest, contract.direction, dataset,
                                        expected_ants, lkh_reference))
            record["status"] = "passed"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    with (root / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return root, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True, help="训练完成后冻结的候选目录")
    parser.add_argument("--config", required=True, help="已冻结的单规模测试预算配置")
    args = parser.parse_args()
    try:
        root, record = evaluate(args.candidate_dir, args.config)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"测试无法启动：{exc}", file=sys.stderr)
        return 1
    print(f"测试状态：{record['status']}；记录：{root}")
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
