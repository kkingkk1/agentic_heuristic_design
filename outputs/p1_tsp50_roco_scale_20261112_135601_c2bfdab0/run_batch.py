"""本批 TSP50 对照：依次研发，冻结候选，再进行测试。"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


BATCH = Path(__file__).resolve().parent
ROOT = BATCH.parents[1]
sys.path.insert(0, str(ROOT))

from scripts import p1_formal_matrix as matrix
from scripts import p1_freeze_candidate, p1_test_evaluate, p1_token_usage

METHODS = {
    "proposed": ("proposed_solution_quality_and_speed_agents", "p1_control", "p1_run_generation.py"),
    "roco": ("roco_paper_adapted", "p1_roco", "p1_run_roco.py"),
}
CELLS = [(short, repeat) for repeat in range(1, 4) for short in METHODS]
LEDGER = BATCH / "ledger.jsonl"


def record(event, **fields):
    item = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with LEDGER.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(item, ensure_ascii=False, allow_nan=False), flush=True)


def check():
    contract, data = matrix.load_matrix()
    matrix.verify_expansion(contract, data)
    for short, repeat in CELLS:
        method = METHODS[short][0]
        actual = json.loads((BATCH / f"{short}_{repeat}.json").read_text(encoding="utf-8"))
        expected = matrix.training_config(contract, "tsp_aco", "white_box", method, repeat)
        if actual != expected:
            raise ValueError(f"研发配置不符：{short}_{repeat}")
        matrix.verify_training_config(actual)
    actual = json.loads((BATCH / "test.json").read_text(encoding="utf-8"))
    if actual != matrix.test_config(contract, "tsp_aco", "white_box", 50):
        raise ValueError("测试配置不符。")
    matrix.verify_test_config(actual)
    return contract


def run_training(short, repeat):
    method, output_dir, runner = METHODS[short]
    parent = ROOT / "outputs" / output_dir
    before = {path.resolve() for path in parent.iterdir() if path.is_dir()}
    log_path = BATCH / f"{short}_{repeat}.log"
    cfg_path = BATCH / f"{short}_{repeat}.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    command = [sys.executable, "-B", "-u", str(ROOT / "scripts" / runner),
               "--config", str(cfg_path), "--run"]
    record("training_started", method=method, repeat=repeat, stdout=str(log_path))
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    new = [path for path in parent.iterdir() if path.is_dir() and path.resolve() not in before]
    matches = []
    for path in new:
        try:
            if json.loads((path / "config.json").read_text(encoding="utf-8")) == cfg:
                matches.append(path)
        except (OSError, ValueError):
            continue
    output = log_path.read_text(encoding="utf-8", errors="replace")
    if len(matches) != 1:
        record("training_identity_error", method=method, repeat=repeat, matching_runs=len(matches),
               exit_code=result.returncode, output=output[-1000:])
        return None, False, True
    run_dir = matches[0]
    record("training_finished", method=method, repeat=repeat, exit_code=result.returncode,
           run_dir=str(run_dir), output=output[-1000:])
    fatal = any(reason in output for reason in (
        "额度或限流", "模型连接失败", "模型调用失败", "无法确认 Codex",
        "可用内存低于", "配置无效", "受保护文件", "已有 P1 正式研发"))
    return run_dir, result.returncode == 0, fatal


def main():
    check()
    if LEDGER.exists() or (BATCH / "summary.json").exists():
        raise ValueError("批次已有执行记录；不能重复启动。")
    if sys.argv[1:] == ["--check"]:
        print("批次配置核验通过；尚未发起模型调用。")
        return 0
    if sys.argv[1:] != ["--run"]:
        raise ValueError("使用 --check 或 --run。")
    record("batch_started", matrix_sha256=matrix.digest(matrix.MATRIX_PATH), cells=len(CELLS))
    completed = []
    all_runs = []
    stopped = False
    for short, repeat in CELLS:
        if stopped:
            record("training_not_started", method=METHODS[short][0], repeat=repeat,
                   reason="此前运行发生服务或启动门禁错误。")
            continue
        run_dir, succeeded, stopped = run_training(short, repeat)
        if run_dir is not None:
            all_runs.append((short, repeat, run_dir))
            if succeeded:
                completed.append((short, repeat, run_dir))

    frozen = []
    for short, repeat, run_dir in completed:
        try:
            candidate = p1_freeze_candidate.freeze(run_dir)
            frozen.append((short, repeat, candidate))
            record("candidate_frozen", method=METHODS[short][0], repeat=repeat, candidate_dir=str(candidate))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            record("candidate_freeze_failed", method=METHODS[short][0], repeat=repeat,
                   error=f"{type(exc).__name__}: {exc}")

    passed = {short: [] for short in METHODS}
    for short, repeat, candidate in frozen:
        try:
            path, result = p1_test_evaluate.evaluate(candidate, BATCH / "test.json")
            record("test_finished", method=METHODS[short][0], repeat=repeat, status=result["status"],
                   result_path=str(path / "result.json"), average=result.get("average"),
                   average_solve_wall_seconds=result.get("average_solve_wall_seconds"),
                   error=result.get("error"))
            if result["status"] == "passed":
                passed[short].append(result["average"])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            record("test_failed_to_start", method=METHODS[short][0], repeat=repeat,
                   error=f"{type(exc).__name__}: {exc}")

    summaries = {}
    for short, values in passed.items():
        summaries[short] = {"values": values,
                            "mean": statistics.mean(values) if len(values) == 3 else None,
                            "sample_std": statistics.stdev(values) if len(values) == 3 else None}
    usage_logs = [p1_token_usage.read_events(run_dir / "events.jsonl")
                  for _, _, run_dir in all_runs if (run_dir / "events.jsonl").is_file()]
    usage = p1_token_usage.summarize(usage_logs) if usage_logs else None
    summary = {"status": "complete" if all(len(values) == 3 for values in passed.values()) else "incomplete",
               "stop_due_to_service_or_gate": stopped, "methods": summaries, "completed_training_runs": len(completed),
               "token_usage_for_all_started_runs": usage}
    with (BATCH / "summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    record("batch_finished", status=summary["status"], summary=str(BATCH / "summary.json"))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"批次停止：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
