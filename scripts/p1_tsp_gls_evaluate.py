"""监督 TSP-GLS 单候选完整训练集评价并保留失败证据。"""

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.p0_controller import check_modification_scope, read_solve_timing  # noqa: E402
from scripts.p0_windows_job import WindowsJob  # noqa: E402
from scripts.p1_job_supervision import wait_for_job  # noqa: E402


TASK = ROOT / "problems" / "tsp_gls"
DATASET = TASK / "dataset" / "train200_dataset.npy"
DATASET_SHA256 = "f8da974c2a9e935b86c05991f0ed722d01bd1e3a4a18ddde65c381b224086235"
EVENT_PREFIX = "P1_JSON "
TIMEOUT_SECONDS = 60


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_complete(stdout_path):
    events = [json.loads(line[len(EVENT_PREFIX):])
              for line in stdout_path.read_text(encoding="utf-8").splitlines()
              if line.startswith(EVENT_PREFIX)]
    if [event.get("event") for event in events] != ["start"] + ["instance"] * 10 + ["complete"]:
        raise ValueError("评价事件缺失、重复或顺序错误。")
    start, complete = events[0], events[-1]
    if (start.get("seed_base") != 20261103 or start.get("n_nodes") != 200
            or start.get("n_iterations") != 1200 or start.get("perturbation_moves") != 30
            or complete.get("instance_count") != 10):
        raise ValueError("评价配置或实例数量不符合 TSP-GLS 合同。")
    if digest(DATASET) != DATASET_SHA256:
        raise ValueError("训练集哈希不匹配。")
    positions = np.load(DATASET, allow_pickle=False)
    objectives = []
    for index, (event, points) in enumerate(zip(events[1:-1], positions)):
        tour, objective = event.get("tour"), event.get("objective")
        if (event.get("index") != index or event.get("seed") != 20261103 + index
                or event.get("iterations") != 1200 or not isinstance(tour, list) or len(tour) != 200
                or any(type(node) is not int for node in tour) or sorted(tour) != list(range(200))
                or not isinstance(objective, (int, float)) or isinstance(objective, bool)
                or not math.isfinite(objective) or objective <= 0):
            raise ValueError(f"实例 {index} 的输出不完整或无效。")
        expected = math.fsum(math.hypot(*(points[a] - points[b])) for a, b in zip(tour, tour[1:] + tour[:1]))
        if not math.isclose(objective, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"实例 {index} 的路线长度与坐标不一致。")
        objectives.append(objective)
    average, fitness = complete.get("average"), complete.get("fitness")
    if (not isinstance(average, (int, float)) or isinstance(average, bool) or not math.isfinite(average)
            or not isinstance(fitness, (int, float)) or isinstance(fitness, bool) or not math.isfinite(fitness)
            or fitness != -average or not math.isclose(average, math.fsum(objectives) / 10, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("平均目标或 Fitness 与逐实例结果不一致。")
    result = {"objectives": objectives, "average": average, "fitness": fitness, "start": start}
    result.update(read_solve_timing(start, events[1:-1], complete))
    return result


def evaluate(root_dir):
    root_dir = Path(root_dir).resolve()
    candidate = root_dir / "problems" / "tsp_gls" / "gpt.py"
    if not candidate.is_file():
        raise ValueError(f"找不到候选文件：{candidate}")
    check_modification_scope(candidate.read_text(encoding="utf-8"))
    stdout_path, stderr_path, record_path = (root_dir / name for name in ("stdout.txt", "stderr.txt", "result.json"))
    if any(path.exists() for path in (stdout_path, stderr_path, record_path)):
        raise ValueError("运行目录已含评价输出；请使用新的独立运行目录。")
    protected = [candidate, TASK / "gls.py", TASK / "eval.py", TASK / "lkh_reference.py", DATASET,
                 ROOT / "scripts" / "p0_controller.py", ROOT / "scripts" / "p1_job_supervision.py",
                 Path(__file__).resolve()]
    before = {str(path): digest(path) for path in protected}
    if before[str(DATASET)] != DATASET_SHA256:
        raise ValueError("训练集哈希不匹配。")
    command = [sys.executable, "-B", "-u", str(TASK / "eval.py"), "200", str(root_dir), "train"]
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    record = {"command": command, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "timeout_seconds": TIMEOUT_SECONDS, "protected_sha256": before, "status": "failed",
              "cpu_scope": "Windows 作业中的完整进程树", "prompt_condition": "white_box"}
    begin = time.monotonic()
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        try:
            with WindowsJob() as job:
                process = job.start(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                record["pid"] = process.pid
                timed_out, accounting = wait_for_job(job, process, begin + TIMEOUT_SECONDS)
                if timed_out:
                    record["status"] = "timeout"
                record["exit_code"] = process.returncode
                record["process_tree"] = accounting
        except (OSError, RuntimeError) as exc:
            record["status"] = "infrastructure_error"
            record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = time.monotonic() - begin
    try:
        changed = [str(path) for path in protected if digest(path) != before[str(path)]]
    except OSError as exc:
        changed = [f"无法读取受保护文件：{exc}"]
    if changed:
        record["status"] = "infrastructure_error"
        record["error"] = f"受保护文件发生变化：{changed}"
    if record["status"] == "failed" and record.get("exit_code") == 0:
        try:
            record.update(read_complete(stdout_path))
            record["status"] = "passed"
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            record["error"] = f"输出不完整或无效：{exc}"
    with record_path.open("x", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    return record


def main():
    if len(sys.argv) != 2:
        print("用法：p1_tsp_gls_evaluate.py <候选运行根目录>", file=sys.stderr)
        return 2
    try:
        result = evaluate(sys.argv[1])
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"评价无法启动：{exc}", file=sys.stderr)
        return 2
    print(f"评价状态：{result['status']}；记录：{Path(sys.argv[1]).resolve() / 'result.json'}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
