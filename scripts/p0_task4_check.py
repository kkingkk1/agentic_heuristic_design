"""使用任务2的参考函数，独立检查任务4的真实评价接入；不调用模型。"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import p0_runtime  # noqa: E402
from scripts.p0_controller import Candidate  # noqa: E402
from scripts.p0_windows_job import WindowsJob  # noqa: E402


REFERENCE = ROOT / "outputs/p0_tsp_aco/20261103_174821_76657347/reference_final_a"


def prior_check_cpu(current_root):
    total = 0.0
    for path in (ROOT / "outputs/p0_control_checks").glob("*/events.jsonl"):
        if path.parent == current_root:
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        started = {item["candidate_id"] for item in records if item.get("event") == "evaluation_budget_reserved"}
        finished = {item["candidate_id"] for item in records if item.get("event") == "evaluation_cpu_final"}
        if started != finished:
            raise RuntimeError(f"此前检查的 CPU 记录不完整：{path}")
        total += sum(item["accounting"]["cpu_seconds"] for item in records
                     if item.get("event") == "evaluation_cpu_final")
    return total


def checked_evaluate(runtime, candidate):
    runtime.prepare_proposal(candidate)
    try:
        return runtime.evaluate(candidate)
    finally:
        runtime.release_proposal(candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures-only", action="store_true", help="实际检查非法数值和超时，不重跑参考评价")
    args = parser.parse_args()
    root = ROOT / "outputs" / "p0_control_checks" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    log = p0_runtime.JsonLog(root / "events.jsonl")
    config = json.loads(p0_runtime.CONFIG.read_text(encoding="utf-8"))
    config["budget"]["future_initialization_and_generation_eval_process_tree_cpu_seconds"] = config["budget"]["future_probes_and_checks_eval_process_tree_cpu_seconds"]
    cores = os.cpu_count() or 1
    limited_cores = min(8, cores)
    rate = min(10000, int(limited_cores * 10000 / cores))

    class CheckJob(WindowsJob):
        def __init__(self):
            super().__init__()
            try:
                self.set_cpu_rate(rate)
            except BaseException:
                self.close()
                raise

    p0_runtime.WindowsJob = CheckJob
    report = {"status": "failed", "model_calls": 0, "source": str(REFERENCE),
              "cpu_budget_seconds": 600, "check_job_cpu_rate": rate,
              "check_job_max_logical_cores": limited_cores, "whole_training_wall_timeout_seconds": 60}
    try:
        previous_cpu = prior_check_cpu(root)
        report["previous_check_cpu_seconds"] = previous_cpu
        config["budget"]["future_initialization_and_generation_eval_process_tree_cpu_seconds"] = 600 - previous_cpu
        runtime = p0_runtime.Runtime(root, config, log)
        # 检查作业采用操作系统硬 CPU 配额，故其最坏并发 CPU 数不超过此值。
        runtime.budget.cores = limited_cores
        if args.failures_only:
            invalid = Candidate("invalid_number", 1, "invalid_number", (), code=(
                "import numpy as np\ndef heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:\n"
                "    return np.full_like(distance_matrix, np.nan)\n"))
            bad = checked_evaluate(runtime, invalid)
            if bad["status"] != "candidate_error" or "fitness" in bad:
                raise RuntimeError(f"非法数值处理错误：{bad}")
            # 只缩短故障 fixture 的超时；正式评价参数不变。
            config["task"]["whole_training_wall_timeout_seconds"] = 3
            slow = Candidate("timeout", 2, "timeout", (), code=(
                "import numpy as np\ndef heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:\n"
                "    while True:\n        pass\n    return np.ones_like(distance_matrix)\n"))
            timed = checked_evaluate(runtime, slow)
            if timed["status"] != "timeout" or "fitness" in timed:
                raise RuntimeError(f"超时处理错误：{timed}")
            if timed["process_tree_cpu"]["active_processes"] != 0:
                raise RuntimeError("超时后仍有作业内进程。")
            report.update({"status": "passed", "case": "failures_only",
                           "invalid_number_status": bad["status"], "timeout_status": timed["status"],
                           "fixture_wall_timeout_seconds": 3,
                           "total_process_tree_cpu_seconds": runtime.budget.cpu_finished})
        else:
            code = (REFERENCE / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8")
            candidate = Candidate("reference", 1, "reference_check", (), code=code)
            actual = checked_evaluate(runtime, candidate)
            baseline = json.loads((REFERENCE / "result.json").read_text(encoding="utf-8"))
            if actual.get("status") != "passed":
                raise RuntimeError(f"参考评价失败：{actual.get('error') or actual.get('status')}")
            if len(actual["objectives"]) != len(baseline["objectives"]) or not all(
                math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)
                for x, y in zip(actual["objectives"], baseline["objectives"])
            ):
                raise RuntimeError("五个实例目标与任务2参考记录不一致。")
            accounting = actual["process_tree_cpu"]
            if accounting["active_processes"] != 0 or accounting["total_processes"] < 1:
                raise RuntimeError("进程树未完整退出或计数无效。")
            report.update({"status": "passed", "objectives": actual["objectives"], "average": actual["average"],
                           "process_tree_cpu": accounting, "elapsed_seconds": actual["elapsed_seconds"],
                           "baseline_objectives_match": True})
        if previous_cpu + runtime.budget.cpu_finished > 600:
            raise RuntimeError("检查使用的评价 CPU 超过检查预算。")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    with (root / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"任务4真实评价检查：{report['status']}；记录：{root}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
