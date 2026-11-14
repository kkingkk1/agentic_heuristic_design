"""保留休眠前完成的测试实例，继续 RoCo 首个候选的 TSP50 测试。"""

import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
ORIGINAL = ROOT / "outputs/p1_test/roco_paper_adapted/20261114_003857_23054d62"
FROZEN = ROOT / "outputs/p1_frozen_candidates/roco_paper_adapted/20261112_183109_ae37c237_resume"
CONFIG = ROOT / "outputs/p1_tsp50_roco_scale_20261112_135601_c2bfdab0/test.json"
SLEEP_UTC = datetime.fromisoformat("2026-11-13T16:40:50.307880+00:00")
WAKE_UTC = "2026-11-14T02:05:25.853900+00:00"
PREFIX = "P1_JSON "
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.spatial import distance_matrix

from scripts import p1_test_evaluate as test, p1_formal_matrix as matrix
from scripts.p0_runtime import digest, memory_and_priority
from scripts.p1_runtime import DATA_CONTRACT


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_events(path):
    return [json.loads(line[len(PREFIX):]) for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith(PREFIX)]


def validate_instances(instances, data, seed_base, iterations):
    require(len(instances) == 64, "测试实例总数不符。")
    for index, event in enumerate(instances):
        tour = event.get("tour")
        objective = event.get("objective")
        solve_time = event.get("solve_wall_seconds")
        heuristic_time = event.get("heuristic_wall_seconds")
        require(event.get("event") == "instance" and event.get("index") == index
                and event.get("seed") == seed_base + index and event.get("iterations") == iterations
                and isinstance(tour, list) and len(tour) == 50 and all(type(x) is int for x in tour)
                and sorted(tour) == list(range(50)), f"实例 {index} 的种子、迭代或路径不符。")
        require(type(objective) in (int, float) and math.isfinite(objective) and objective > 0
                and type(solve_time) in (int, float) and math.isfinite(solve_time) and solve_time > 0
                and type(heuristic_time) in (int, float) and math.isfinite(heuristic_time)
                and 0 <= heuristic_time <= solve_time, f"实例 {index} 的目标或计时无效。")
        distances = distance_matrix(data[index], data[index])
        recomputed = float(distances[tour, np.roll(tour, 1)].sum())
        require(math.isclose(objective, recomputed, rel_tol=1e-12, abs_tol=1e-12),
                f"实例 {index} 的目标与路径不一致。")


def assembled_events(start, instances):
    return [start, *instances, {"event": "complete", "instance_count": 64,
            "average": math.fsum(x["objective"] for x in instances) / 64,
            "average_solve_wall_seconds": math.fsum(x["solve_wall_seconds"] for x in instances) / 64,
            "average_heuristic_wall_seconds": math.fsum(x["heuristic_wall_seconds"] for x in instances) / 64,
            "source": "derived_from_recorded_instance_events"}]


def encoded(events):
    return "".join(PREFIX + json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in events)


class TextSource:
    def __init__(self, text):
        self.text = text

    def read_text(self, encoding=None):
        return self.text


def inspect():
    manifest = json.loads((FROZEN / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    old = json.loads((ORIGINAL / "result.json").read_text(encoding="utf-8"))
    contract = json.loads(DATA_CONTRACT.read_text(encoding="utf-8"))
    matrix.verify_test_config(config)
    entry = test.load_test_contract(config, manifest, contract)
    require(old.get("status") == "timeout" and old.get("method") == "roco_paper_adapted"
            and old.get("formal_replicate") == 1 and old.get("candidate_id") == "c0341"
            and old.get("matrix_contract_sha256") == config["matrix_contract_sha256"]
            and old.get("size") == 50 and old.get("timeout_seconds") == 900, "原测试身份或停止状态不符。")
    protected = old.get("protected_sha256")
    require(isinstance(protected, dict) and protected, "原测试受保护文件清单缺失。")
    for filename, expected in protected.items():
        require(Path(filename).is_file() and digest(Path(filename)) == expected, f"受保护文件变化：{filename}")
    candidate = FROZEN / "candidate.py"
    require(digest(candidate) == manifest["candidate_sha256"] == old["candidate_sha256"]
            and digest(ORIGINAL / "problems/tsp_aco/gpt.py") == manifest["candidate_sha256"], "候选代码不一致。")
    tree = ast.parse(candidate.read_text(encoding="utf-8"))
    require(len(tree.body) == 2 and isinstance(tree.body[0], ast.Import)
            and [(x.name, x.asname) for x in tree.body[0].names] == [("numpy", "np")]
            and isinstance(tree.body[1], ast.FunctionDef) and tree.body[1].name == "heuristics_v2"
            and not any(isinstance(x, (ast.Global, ast.Nonlocal)) for x in ast.walk(tree)),
            "候选存在不能确认的跨实例状态。")
    events = parse_events(ORIGINAL / "stdout.txt")
    require(len(events) == 15 and [x.get("event") for x in events] == ["start"] + ["instance"] * 14,
            "原测试已完成实例记录不完整或不唯一。")
    start, prefix = events[0], events[1:]
    dataset = ROOT / entry["path"]
    require(digest(dataset) == entry["sha256"] and Path(start["dataset"]).resolve() == dataset.resolve()
            and start["candidate_sha256"] == manifest["candidate_sha256"]
            and start["seed_base"] == entry["evaluation_seed_base"] and start["n_iterations"] == entry["iterations"]
            and start["n_ants"] == contract["tasks"]["tsp_aco"]["train"]["ants"]
            and start["time_scope"] == "solve_wall_seconds_including_preparation_and_validation"
            and start["heuristic_time_scope"] == "heuristic_function_only", "原始测试合同不符。")
    data = np.load(dataset, allow_pickle=False)
    require(data.shape == (64, 50, 2) and np.isfinite(data).all(), "测试数据无效。")
    # 用真实路径重算已完成的 14 个结果；余下模拟条目只用于复用完整校验器。
    placeholder = []
    for index in range(14, 64):
        tour = list(range(50))
        distances = distance_matrix(data[index], data[index])
        placeholder.append({"event": "instance", "index": index, "seed": entry["evaluation_seed_base"] + index,
                            "objective": float(distances[tour, np.roll(tour, 1)].sum()), "iterations": entry["iterations"],
                            "tour": tour, "solve_wall_seconds": 1.0, "heuristic_wall_seconds": 0.0})
    validate_instances(prefix + placeholder, data, entry["evaluation_seed_base"], entry["iterations"])
    begun = datetime.fromisoformat(old["started_at_utc"])
    active_before_sleep = (SLEEP_UTC - begun).total_seconds()
    require(0 < active_before_sleep < old["timeout_seconds"] and old["elapsed_seconds"] > 9 * 3600,
            "睡眠时间或原测试经过时间不符。")
    # 向上取整并预留十秒，覆盖短暂唤醒及监督器终止子进程的时间。
    previous_active_ceiling = math.ceil(active_before_sleep) + 10
    wall_remaining = old["timeout_seconds"] - previous_active_ceiling
    previous_cpu = old["process_tree"]["cpu_seconds"]
    cpu_remaining = old["process_tree_cpu_limit_seconds"] - previous_cpu
    require(wall_remaining > 0 and cpu_remaining > 0, "原测试运行预算已用完。")
    source_run = Path(manifest["source_run"])
    training = json.loads((source_run / "config.json").read_text(encoding="utf-8"))
    matrix.verify_training_config(training)
    return {"manifest": manifest, "config": config, "contract": contract, "entry": entry,
            "old": old, "start": start, "prefix": prefix, "data": data, "dataset": dataset,
            "training": training, "previous_active_ceiling": previous_active_ceiling,
            "wall_remaining": wall_remaining, "previous_cpu": previous_cpu, "cpu_remaining": cpu_remaining,
            "source_stdout_sha256": digest(ORIGINAL / "stdout.txt"),
            "source_result_sha256": digest(ORIGINAL / "result.json")}


def validate_complete(state, instances):
    validate_instances(instances, state["data"], state["entry"]["evaluation_seed_base"], state["entry"]["iterations"])
    events = assembled_events(state["start"], instances)
    checked = test.read_complete(TextSource(encoded(events)), state["entry"], state["manifest"], "min",
                                 state["dataset"], state["contract"]["tasks"]["tsp_aco"]["train"]["ants"])
    return events, checked


def simulate():
    state = inspect()
    fake = []
    for index in range(14, 64):
        tour = list(range(50))
        distances = distance_matrix(state["data"][index], state["data"][index])
        fake.append({"event": "instance", "index": index, "seed": state["entry"]["evaluation_seed_base"] + index,
                     "objective": float(distances[tour, np.roll(tour, 1)].sum()), "iterations": state["entry"]["iterations"],
                     "tour": tour, "solve_wall_seconds": 1.0, "heuristic_wall_seconds": 0.0})
    _, checked = validate_complete(state, state["prefix"] + fake)
    require(len(checked["objectives"]) == 64 and state["wall_remaining"] + state["previous_active_ceiling"] == 900,
            "续跑合并或墙钟预算不符。")
    try:
        validate_complete(state, state["prefix"] + [fake[0], *fake[:-1]])
    except ValueError:
        pass
    else:
        raise ValueError("重复实例夹具没有被拒绝。")
    print(json.dumps({"status": "passed", "completed_instances": 14, "remaining_instances": 50,
                      "previous_active_wall_ceiling_seconds": state["previous_active_ceiling"],
                      "remaining_wall_seconds": state["wall_remaining"], "previous_cpu_seconds": state["previous_cpu"],
                      "remaining_cpu_seconds": state["cpu_remaining"], "repeated_instance_rejected": True}, ensure_ascii=False))


def worker(root):
    state = inspect()
    evaluator = ROOT / "problems/tsp_aco/eval.py"
    sys.path.insert(0, str(evaluator.parent))
    spec = importlib.util.spec_from_file_location("resumed_tsp_aco_eval", evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset, data = module.test_dataset(50)
    require(dataset.resolve() == state["dataset"].resolve() and module.candidate_hash(root) == state["manifest"]["candidate_sha256"]
            and np.__version__ == state["start"]["numpy"] and torch.__version__ == state["start"]["torch"]
            and torch.get_num_threads() == state["start"]["torch_threads"], "续跑软件、数据或候选不符。")
    heuristic = module.load_heuristic(root)
    print("RESUME_JSON " + json.dumps({"event": "resume_start", "source_stdout_sha256": state["source_stdout_sha256"],
                                      "first_index": 14, "last_index": 63}, ensure_ascii=False), flush=True)
    for index in range(14, 64):
        seed = state["entry"]["evaluation_seed_base"] + index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        timings = {}
        begun = time.perf_counter()
        obj, tour, iterations = module.solve(data[index], heuristic, state["entry"]["iterations"], timings=timings)
        elapsed = time.perf_counter() - begun
        event = {"event": "instance", "index": index, "seed": seed, "objective": obj,
                 "iterations": iterations, "tour": tour, "solve_wall_seconds": elapsed,
                 "heuristic_wall_seconds": timings.get("heuristic_wall_seconds")}
        print(PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)
    require(module.candidate_hash(root) == state["manifest"]["candidate_sha256"]
            and digest(dataset) == state["entry"]["sha256"], "续跑期间候选或测试数据发生变化。")


def resume():
    state = inspect()
    resources = memory_and_priority()
    require(resources["available_memory_bytes"] >= state["training"]["resources"]["min_available_system_memory_bytes"], "内存不足。")
    root = ROOT / "outputs/p1_test/roco_paper_adapted" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8] + "_resume")
    module_dir = root / "problems/tsp_aco"
    module_dir.mkdir(parents=True, exist_ok=False)
    copy = module_dir / "gpt.py"
    with copy.open("xb") as out:
        out.write((FROZEN / "candidate.py").read_bytes())
    require(digest(copy) == state["manifest"]["candidate_sha256"], "续跑候选副本不符。")
    protected = dict(state["old"]["protected_sha256"])
    for path in (ORIGINAL / "stdout.txt", ORIGINAL / "result.json", Path(__file__).resolve(), copy):
        protected[str(path)] = digest(path)
    env = os.environ.copy()
    env.update(state["training"]["resources"]["thread_environment"])
    env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join(state["training"]["resources"]["pythonpath"])})
    command = [state["training"]["resources"]["python_executable"], "-B", "-u", str(Path(__file__).resolve()), "--worker", str(root)]
    record = {"status": "failed", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "method": state["manifest"]["method"], "candidate_id": state["manifest"]["source_candidate_id"],
              "matrix_contract_sha256": state["manifest"]["matrix_contract_sha256"], "task": "tsp_aco",
              "prompt_condition": "white_box", "formal_replicate": 1, "source_run": state["manifest"]["source_run"],
              "candidate_sha256": state["manifest"]["candidate_sha256"], "size": 50, "command": command,
              "resources": resources, "protected_sha256": protected, "timeout_seconds": 900,
              "process_tree_cpu_limit_seconds": state["old"]["process_tree_cpu_limit_seconds"],
              "resumed_from": str(ORIGINAL), "source_stdout_sha256": state["source_stdout_sha256"],
              "source_result_sha256": state["source_result_sha256"], "retained_instance_count": 14,
              "resumed_instance_indexes": [14, 63], "sleep_utc": SLEEP_UTC.isoformat(), "wake_utc": WAKE_UTC,
              "sleep_evidence": "Windows System: Kernel-Power 42 and Power-Troubleshooter 1",
              "previous_active_wall_ceiling_seconds": state["previous_active_ceiling"],
              "remaining_active_wall_seconds": state["wall_remaining"],
              "previous_process_tree_cpu_seconds": state["previous_cpu"],
              "note": "连续测试被系统休眠打断；保留已完成实例，重启未完成实例，休眠时间不计入续跑预算。"}
    print(f"RoCo 第 1 项测试断点检查通过；记录：{root}", flush=True)
    begin = time.monotonic()
    with (root / "stdout.txt").open("x", encoding="utf-8") as stdout, (root / "stderr.txt").open("x", encoding="utf-8") as stderr:
        try:
            with test.WindowsJob() as job:
                process = job.start(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                record["pid"] = process.pid
                reason, record["process_tree"] = test.wait_bounded(job, process,
                    deadline=begin + state["wall_remaining"], cpu_limit=state["cpu_remaining"])
                record["exit_code"] = process.returncode
                if reason:
                    record["status"] = reason
        except (OSError, RuntimeError) as exc:
            record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    record["segment_elapsed_seconds"] = time.monotonic() - begin
    record["elapsed_seconds"] = state["previous_active_ceiling"] + record["segment_elapsed_seconds"]
    try:
        changed = [name for name, expected in protected.items() if digest(Path(name)) != expected]
        require(not changed, f"受保护文件发生变化：{changed}")
        if record["status"] == "failed" and record.get("exit_code") == 0:
            suffix = parse_events(root / "stdout.txt")
            require(len(suffix) == 50 and [x.get("index") for x in suffix] == list(range(14, 64)), "续跑实例缺失或重复。")
            combined, checked = validate_complete(state, state["prefix"] + suffix)
            with (root / "combined_stdout.txt").open("x", encoding="utf-8", newline="\n") as out:
                out.write(encoded(combined))
            record.update(checked)
            record["status"] = "passed"
            record["combined_stdout_sha256"] = digest(root / "combined_stdout.txt")
            record["resume_stdout_sha256"] = digest(root / "stdout.txt")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        record["status"], record["error"] = "infrastructure_error", f"{type(exc).__name__}: {exc}"
    with (root / "result.json").open("x", encoding="utf-8", newline="\n") as out:
        json.dump(record, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    print(f"测试状态：{record['status']}；记录：{root}", flush=True)
    return 0 if record["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--simulate", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.simulate:
        simulate()
    elif args.run:
        return resume()
    elif args.worker is not None:
        worker(args.worker)
    else:
        state = inspect()
        print(json.dumps({"status": "passed", "completed_instances": len(state["prefix"]),
                          "remaining_instances": 50, "remaining_wall_seconds": state["wall_remaining"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"RoCo 第 1 项测试恢复检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
