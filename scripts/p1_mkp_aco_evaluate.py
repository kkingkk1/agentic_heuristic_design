"""监督 MKP-ACO 单候选整训练集评价，保存进程树、输出和失败证据。"""

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

from scripts.p0_controller import check_modification_scope  # noqa: E402
from scripts.p0_windows_job import WindowsJob  # noqa: E402
from scripts.p1_job_supervision import wait_for_job  # noqa: E402


TASK = ROOT / "problems" / "mkp_aco"
DATASET = TASK / "dataset" / "train100_dataset.npz"
DATASET_SHA256 = "6dc1ffc07b50b8ee7bba919443a6840ed82a04a817e039e05da809877bf21073"
EVENT_PREFIX = "P1_JSON "
TIMEOUT_SECONDS = 60


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_complete(stdout_path):
    events = [json.loads(line[len(EVENT_PREFIX):])
              for line in stdout_path.read_text(encoding="utf-8").splitlines()
              if line.startswith(EVENT_PREFIX)]
    if [event.get("event") for event in events] != ["start"] + ["instance"] * 5 + ["complete"]:
        raise ValueError("评价事件缺失、重复或顺序错误。")
    start, complete = events[0], events[-1]
    if (start.get("seed_base") != 20261103 or start.get("n_ants") != 10
            or start.get("n_iterations") != 50 or start.get("n_items") != 100
            or start.get("n_dims") != 5 or complete.get("instance_count") != 5):
        raise ValueError("评价配置或实例数量不符合 MKP 合同。")
    if digest(DATASET) != DATASET_SHA256:
        raise ValueError("训练集哈希不匹配。")
    with np.load(DATASET, allow_pickle=False) as data:
        prizes, weights = data["prizes"], data["weights"]
    objectives = []
    for index, event in enumerate(events[1:-1]):
        solution = event.get("solution")
        objective = event.get("objective")
        if (event.get("index") != index or event.get("seed") != 20261103 + index
                or event.get("iterations") != 50 or not isinstance(solution, list)
                or any(type(item) is not int or not 0 <= item <= 100 for item in solution)
                or not isinstance(objective, (int, float)) or isinstance(objective, bool)
                or not math.isfinite(objective) or objective < 0):
            raise ValueError(f"实例 {index} 的输出不完整或无效。")
        items = [item for item in solution if item != 100]
        if len(items) != len(set(items)) or (100 in solution and any(x != 100 for x in solution[solution.index(100):])):
            raise ValueError(f"实例 {index} 的解重复选取或结束后继续选取。")
        if items and (weights[index, items].sum(axis=0) > 1 + 1e-12).any():
            raise ValueError(f"实例 {index} 的解违反容量约束。")
        expected = math.fsum(float(prizes[index, item]) for item in items)
        if not math.isclose(objective, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"实例 {index} 的目标值不等于所选物品价值。")
        objectives.append(float(objective))
    average = complete.get("average")
    fitness = complete.get("fitness")
    if (not isinstance(average, (int, float)) or isinstance(average, bool)
            or not math.isfinite(average) or not isinstance(fitness, (int, float))
            or isinstance(fitness, bool) or not math.isfinite(fitness) or fitness != average
            or not math.isclose(average, math.fsum(objectives) / 5, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("平均目标或 Fitness 与逐实例结果不一致。")
    return {"objectives": objectives, "average": average, "fitness": fitness, "start": start}


def evaluate(root_dir, *, black_box=False):
    root_dir = Path(root_dir).resolve()
    candidate = root_dir / "problems" / "mkp_aco" / "gpt.py"
    if not candidate.is_file():
        raise ValueError(f"找不到候选文件：{candidate}")
    check_modification_scope(candidate.read_text(encoding="utf-8"))
    stdout_path, stderr_path, record_path = (root_dir / name for name in ("stdout.txt", "stderr.txt", "result.json"))
    if any(path.exists() for path in (stdout_path, stderr_path, record_path)):
        raise ValueError("运行目录已含评价输出；请使用新的独立运行目录。")
    protected = [candidate, TASK / "aco.py", TASK / "eval.py", TASK / "eval_black_box.py", DATASET,
                 ROOT / "scripts" / "p1_job_supervision.py", Path(__file__).resolve()]
    before = {str(path): digest(path) for path in protected}
    if before[str(DATASET)] != DATASET_SHA256:
        raise ValueError("训练集哈希不匹配。")
    eval_script = TASK / ("eval_black_box.py" if black_box else "eval.py")
    command = [sys.executable, "-B", "-u", str(eval_script), "100", str(root_dir), "train"]
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    record = {"command": command, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "timeout_seconds": TIMEOUT_SECONDS, "protected_sha256": before, "status": "failed",
              "cpu_scope": "Windows 作业中的完整进程树", "prompt_condition": "black_box" if black_box else "white_box"}
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
    if len(sys.argv) != 3 or sys.argv[2] not in ("white", "black"):
        print("用法：p1_mkp_aco_evaluate.py <候选运行根目录> <white|black>", file=sys.stderr)
        return 2
    try:
        result = evaluate(sys.argv[1], black_box=sys.argv[2] == "black")
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"评价无法启动：{exc}", file=sys.stderr)
        return 2
    print(f"评价状态：{result['status']}；记录：{Path(sys.argv[1]).resolve() / 'result.json'}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
