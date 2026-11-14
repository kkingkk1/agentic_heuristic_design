"""首组续跑完成后，执行其余五组并测试冻结候选。"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
sys.path.insert(0, str(ROOT))

from scripts import p1_formal_matrix as matrix, p1_freeze_candidate, p1_test_evaluate, p1_token_usage

ORIGINAL_BATCH = ROOT / "outputs/p1_tsp50_roco_scale_20261112_135601_c2bfdab0"
ORIGINAL_RUN = ROOT / "outputs/p1_control/20261112_135816_854bdc72"
FUSION_RETRY = ROOT / "outputs/p1_manual_retry/fusion_2_20261112_143551_66d3a6ad"
FIRST_RESUME = ROOT / "outputs/p1_control/20261112_144932_43904aa4_resume"
SECOND_RESUME = ROOT / "outputs/p1_control/20261112_145816_48da5a36_resume"
THIRD_RESUME = ROOT / "outputs/p1_control/20261112_161733_a2221c51_resume"
ROCO_SEGMENTS = [ROOT / "outputs/p1_roco" / name for name in (
    "20261112_170516_a9eab10e", "20261112_172536_af604137_resume",
    "20261112_174300_b791da58_resume", "20261112_180259_94bd200c_resume",
    "20261112_180809_e040f878_resume", "20261112_181257_1a92db4f_resume",
    "20261112_181554_cd08d9e6_resume", "20261112_183109_ae37c237_resume")]
FIRST_ROCO = ROCO_SEGMENTS[-1]
ROCO_SAVED_EVALUATIONS = [ROOT / "outputs/p1_manual_retry/roco_generation2_saved_20261112" / name
                          for name in ("20261112_175444_2229cb9a_evaluation", "20261112_181024_8854108b_evaluation")]
PROPOSED_SECOND_SEGMENTS = [ROOT / "outputs/p1_control" / name for name in
                            ("20261112_195246_68d519ff", "20261112_200325_f23e0414_resume",
                             "20261112_201912_0dacd7eb_resume", "20261112_204019_c8ca46d5_resume",
                             "20261112_205629_801e167f_resume", "20261112_210054_1819d656_resume",
                             "20261112_211734_d7c5ee74_resume")]
PROPOSED_SECOND = PROPOSED_SECOND_SEGMENTS[-1]
ROCO_SECOND_SEGMENTS = [ROOT / "outputs/p1_roco" / name for name in
                        ("20261112_214857_dd30b74a", "20261112_215400_819f82c2_resume")]
ROCO_SECOND = ROCO_SECOND_SEGMENTS[-1]
ROCO_SECOND_SAVED_EVALUATIONS = [ROOT / "outputs/p1_manual_retry/roco_2_quota_saved_20261112/20261112_221401_cedfe544_evaluation"]
LEDGER = RECOVERY / "continuation_ledger.jsonl"
METHODS = {
    "proposed": ("proposed_solution_quality_and_speed_agents", "p1_control", "p1_run_generation.py"),
    "roco": ("roco_paper_adapted", "p1_roco", "p1_run_roco.py"),
}
PENDING = [("roco", 1), ("proposed", 2), ("roco", 2), ("proposed", 3), ("roco", 3)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_events(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def record(event, **fields):
    item = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with LEDGER.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(item, ensure_ascii=False, allow_nan=False), flush=True)


def check(*, require_complete):
    config, data = matrix.load_matrix()
    matrix.verify_expansion(config, data)
    for short, repeat in [("proposed", 1), *PENDING]:
        actual = json.loads((ORIGINAL_BATCH / f"{short}_{repeat}.json").read_text(encoding="utf-8"))
        expected = matrix.training_config(config, "tsp_aco", "white_box", METHODS[short][0], repeat)
        require(actual == expected, f"研发配置不符：{short}_{repeat}")
        matrix.verify_training_config(actual)
    test = json.loads((ORIGINAL_BATCH / "test.json").read_text(encoding="utf-8"))
    matrix.verify_test_config(test)
    events = read_events(SECOND_RESUME / "events.jsonl")
    start = [x for x in events if x["event"] == "formal_run_started"]
    require(len(start) == 1 and start[0].get("resume_of") == str(FIRST_RESUME)
            and start[0].get("total_model_call_cap") == 421, "首组续跑身份或预算记录不符。")
    require(events[-1]["event"] == "preflight_or_run_stopped", "第二次续跑尚未停止。")
    continuation = read_events(THIRD_RESUME / "events.jsonl")
    continuation_start = [x for x in continuation if x["event"] == "formal_run_started"]
    require(len(continuation_start) == 1 and continuation_start[0].get("resume_of") == str(SECOND_RESUME)
            and continuation_start[0].get("total_model_call_cap") == 421, "第三次续跑身份或预算记录不符。")
    selections = [x["generation"] for x in (*events, *continuation) if x["event"] == "selection"]
    first_complete = (continuation[-1]["event"] == "finish" and continuation[-1].get("status") == "completed"
                      and selections == list(range(1, 7)))
    budget_stop = (continuation[-1]["event"] == "preflight_or_run_stopped"
                   and "剩余正常模型调用不足以覆盖一代" in continuation[-1].get("error", "")
                   and selections == list(range(1, len(selections) + 1)) and bool(selections))
    if require_complete:
        require(first_complete or budget_stop, "首组仍在运行，或因非预算原因停止。")
    return config, first_complete, budget_stop


def train(short, repeat):
    method, output_name, _ = METHODS[short]
    parent = ROOT / "outputs" / output_name
    before = {p.resolve() for p in parent.iterdir() if p.is_dir()}
    config_path = ORIGINAL_BATCH / f"{short}_{repeat}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stdout_path = RECOVERY / f"continuation_{short}_{repeat}.log"
    command = [sys.executable, "-B", "-u", str(ROOT / "scripts/p1_classified_runner.py"),
               "--method", short, "--config", str(config_path), "--run"]
    record("training_started", method=method, repeat=repeat, stdout=str(stdout_path))
    with stdout_path.open("x", encoding="utf-8", newline="\n") as stdout:
        process = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                 stdout=stdout, stderr=subprocess.STDOUT)
    candidates = []
    for path in parent.iterdir():
        if not path.is_dir() or path.resolve() in before:
            continue
        try:
            if json.loads((path / "config.json").read_text(encoding="utf-8")) == config:
                candidates.append(path)
        except (OSError, ValueError):
            continue
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    require(len(candidates) == 1, f"研发输出身份不唯一：{short}_{repeat}")
    record("training_finished", method=method, repeat=repeat, exit_code=process.returncode,
           run_dir=str(candidates[0]), output=text[-1000:])
    stop = any(reason in text for reason in ("安全审核拒绝", "额度或限流", "模型连接失败", "模型调用失败",
                                             "无法确认 Codex", "可用内存低于", "配置无效",
                                             "受保护文件", "已有 P1 正式研发"))
    return candidates[0], process.returncode == 0, stop


def prior_test_exists(manifest):
    parent = ROOT / "outputs/p1_test" / manifest["method"]
    if not parent.exists():
        return False
    for path in parent.glob("*/result.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        if (result.get("matrix_contract_sha256") == manifest["matrix_contract_sha256"]
                and result.get("formal_replicate") == manifest["formal_replicate"]
                and result.get("task") == manifest["task"] and result.get("size") == 50):
            return True
    return False


def completed_roco_first():
    expected = json.loads((ORIGINAL_BATCH / "roco_1.json").read_text(encoding="utf-8"))
    all_events = []
    for index, path in enumerate(ROCO_SEGMENTS):
        require(json.loads((path / "config.json").read_text(encoding="utf-8")) == expected,
                f"RoCo 首组片段配置不符：{path}")
        events = read_events(path / "events.jsonl")
        started = [x for x in events if x["event"] == "formal_run_started"]
        require(len(started) == 1 and started[0].get("method") == METHODS["roco"][0]
                and started[0].get("formal_replicate") == 1, f"RoCo 首组片段身份不符：{path}")
        if index:
            previous = ROCO_SEGMENTS[index - 1]
            require(started[0].get("resume_of") == str(previous)
                    and started[0].get("interrupted_events_sha256") == matrix.digest(previous / "events.jsonl"),
                    f"RoCo 首组断点链不符：{path}")
        if index < len(ROCO_SEGMENTS) - 1:
            require(events[-1]["event"] == "preflight_or_run_stopped", f"历史片段未停止：{path}")
        else:
            require(events[-1]["event"] == "finish" and events[-1].get("status") == "completed",
                    "RoCo 首组未完整结束。")
        all_events.extend(events)
    require([x["generation"] for x in all_events if x["event"] == "selection"] == list(range(1, 7))
            and sum(x["event"] == "model_attempt" for x in all_events) == 420,
            "RoCo 首组代数或实际模型调用数不符。")
    for path in ROCO_SAVED_EVALUATIONS:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        source = Path(summary["source_events"])
        require(summary["status"] == "completed" and summary["new_model_calls"] == 0
                and summary["source_events_sha256"] == matrix.digest(source), "RoCo 保存响应评价身份不符。")
        all_events.extend(read_events(path / "evaluation_events.jsonl"))
    evaluations = [x["candidate_id"] for x in all_events if x["event"] == "evaluation_budget_reserved"]
    require(len(evaluations) == 368 and len(set(evaluations)) == len(evaluations),
            "RoCo 首组评价数不符或发生重复评价。")
    return FIRST_ROCO


def completed_existing_second(short, segments):
    expected = json.loads((ORIGINAL_BATCH / f"{short}_2.json").read_text(encoding="utf-8"))
    all_events = []
    for index, path in enumerate(segments):
        require(json.loads((path / "config.json").read_text(encoding="utf-8")) == expected,
                f"{short} 第 2 次研发配置不符。")
        events = read_events(path / "events.jsonl")
        started = [x for x in events if x["event"] == "formal_run_started"]
        require(len(started) == 1 and started[0].get("method") == METHODS[short][0]
                and started[0].get("formal_replicate") == 2, f"{short} 第 2 次研发身份不符。")
        if index:
            previous = segments[index - 1]
            require(started[0].get("resume_of") == str(previous)
                    and started[0].get("interrupted_events_sha256") == matrix.digest(previous / "events.jsonl"),
                    f"{short} 第 2 次研发断点链不符。")
        if index < len(segments) - 1:
            require(events[-1]["event"] == "preflight_or_run_stopped", "原研发片段未停止。")
        else:
            require(events[-1]["event"] == "finish" and events[-1].get("status") == "completed",
                    f"{short} 第 2 次研发尚未完整结束，不能重复启动。")
        all_events.extend(events)
    if short == "roco":
        for path in ROCO_SECOND_SAVED_EVALUATIONS:
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            require(summary["status"] == "completed" and summary["new_model_calls"] == 0
                    and summary["source_events_sha256"] == matrix.digest(Path(summary["source_events"])),
                    "RoCo 第 2 次研发保存响应评价身份不符。")
            all_events.extend(read_events(path / "evaluation_events.jsonl"))
    evaluations = [x["candidate_id"] for x in all_events if x["event"] == "evaluation_budget_reserved"]
    require([x["generation"] for x in all_events if x["event"] == "selection"] == list(range(1, 7))
            and sum(x["event"] == "model_attempt" for x in all_events) <= 420
            and len(evaluations) <= 378 and len(set(evaluations)) == len(evaluations),
            f"{short} 第 2 次研发代数、预算或评价身份不符。")
    return segments[-1]


def completed_proposed_second():
    return completed_existing_second("proposed", PROPOSED_SECOND_SEGMENTS)


def completed_roco_second():
    return completed_existing_second("roco", ROCO_SECOND_SEGMENTS)


def frozen_or_freeze(short, repeat, run_dir):
    method = METHODS[short][0]
    destination = ROOT / "outputs/p1_frozen_candidates" / method / run_dir.name
    if not destination.exists():
        return p1_freeze_candidate.freeze(run_dir)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    events = read_events(run_dir / "events.jsonl")
    candidate_id = p1_freeze_candidate.final_candidate(events, method)
    source = run_dir / "candidates" / candidate_id
    require(manifest["status"] == "frozen_after_training" and manifest["method"] == method
            and manifest["formal_replicate"] == repeat and manifest["source_run"] == str(run_dir)
            and manifest["source_candidate_id"] == candidate_id
            and manifest["source_config_sha256"] == matrix.digest(run_dir / "config.json")
            and manifest["source_events_sha256"] == matrix.digest(run_dir / "events.jsonl")
            and manifest["source_result_sha256"] == matrix.digest(source / "result.json")
            and manifest["source_stdout_sha256"] == matrix.digest(source / "stdout.txt")
            and manifest["candidate_sha256"] == matrix.digest(destination / "candidate.py"),
            f"已冻结候选与原运行不一致：{destination}")
    return destination


def run():
    _, first_complete, budget_stop = check(require_complete=True)
    require(first_complete, "本方法首组尚未完成；正式测试须等待首组预算决定及研发结果。")
    first_roco = completed_roco_first()
    proposed_second = completed_proposed_second()
    roco_second = completed_roco_second()
    require(not LEDGER.exists() and not (RECOVERY / "continuation_summary.json").exists(),
            "后续批次已有记录，不能重复启动。")
    classified_sources = (ROOT / "scripts/p1_classified_runtime.py", ROOT / "scripts/p1_classified_runner.py")
    source_hashes = {str(path): matrix.digest(path) for path in classified_sources}
    record("batch_continued", first_run=str(THIRD_RESUME), first_run_complete=first_complete,
           first_run_stopped_at_budget=budget_stop, remaining_cells=len(PENDING),
           classified_sources_sha256=source_hashes, batch_script_sha256=matrix.digest(Path(__file__).resolve()))
    all_runs = [("proposed", 1, THIRD_RESUME, first_complete)]
    stop = False
    for short, repeat in PENDING:
        if stop:
            record("training_not_started", method=METHODS[short][0], repeat=repeat,
                   reason="此前运行发生服务或启动门禁错误。")
            continue
        if any(matrix.digest(Path(name)) != expected for name, expected in source_hashes.items()):
            stop = True
            record("training_not_started", method=METHODS[short][0], repeat=repeat,
                   reason="分类入口源码在批次中发生变化。")
            continue
        if (short, repeat) == ("roco", 1):
            all_runs.append((short, repeat, first_roco, True))
            record("training_reused", method=METHODS[short][0], repeat=repeat, run_dir=str(first_roco))
            continue
        if (short, repeat) == ("proposed", 2):
            all_runs.append((short, repeat, proposed_second, True))
            record("training_reused", method=METHODS[short][0], repeat=repeat, run_dir=str(proposed_second))
            continue
        if (short, repeat) == ("roco", 2):
            all_runs.append((short, repeat, roco_second, True))
            record("training_reused", method=METHODS[short][0], repeat=repeat, run_dir=str(roco_second))
            continue
        path, success, stop = train(short, repeat)
        all_runs.append((short, repeat, path, success))
        if any(matrix.digest(Path(name)) != expected for name, expected in source_hashes.items()):
            stop = True
            record("classified_sources_changed", method=METHODS[short][0], repeat=repeat)
    if stop:
        record("batch_stopped_before_tests", reason="有研发未启动；测试集尚未使用。")
    else:
        frozen = []
        for short, repeat, path, success in all_runs:
            if not success:
                continue
            try:
                candidate = frozen_or_freeze(short, repeat, path)
                frozen.append((short, repeat, candidate))
                record("candidate_frozen", method=METHODS[short][0], repeat=repeat, candidate_dir=str(candidate))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                record("candidate_freeze_failed", method=METHODS[short][0], repeat=repeat,
                       error=f"{type(exc).__name__}: {exc}")
        for short, repeat, candidate in frozen:
            try:
                manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
                require(not prior_test_exists(manifest), "相同矩阵、方法和重复索引已有测试记录。")
                path, result = p1_test_evaluate.evaluate(candidate, ORIGINAL_BATCH / "test.json")
                record("test_finished", method=METHODS[short][0], repeat=repeat, status=result["status"],
                       result_path=str(path / "result.json"), average=result.get("average"),
                       average_solve_wall_seconds=result.get("average_solve_wall_seconds"),
                       error=result.get("error"))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                record("test_failed_to_start", method=METHODS[short][0], repeat=repeat,
                       error=f"{type(exc).__name__}: {exc}")
    events = read_events(LEDGER)
    by_method = {}
    for short, (method, _, _) in METHODS.items():
        rows = sorted((x for x in events if x["event"] == "test_finished"
                       and x["method"] == method and x["status"] == "passed"), key=lambda x: x["repeat"])
        values = [x["average"] for x in rows]
        by_method[short] = {"runs": [{"repeat": x["repeat"], "average": x["average"],
                                       "result_path": x["result_path"]} for x in rows],
                            "mean": statistics.mean(values) if len(values) == 3 else None,
                            "sample_std": statistics.stdev(values) if len(values) == 3 else None}
    usage_paths = [ORIGINAL_RUN / "events.jsonl", FUSION_RETRY / "events.jsonl",
                   FIRST_RESUME / "events.jsonl", RECOVERY / "events.jsonl", SECOND_RESUME / "events.jsonl",
                   THIRD_RESUME / "events.jsonl", *(path / "events.jsonl" for path in ROCO_SEGMENTS),
                   *(path / "events.jsonl" for path in PROPOSED_SECOND_SEGMENTS[:-1]),
                   *(path / "events.jsonl" for path in ROCO_SECOND_SEGMENTS[:-1])]
    usage_paths += [path / "events.jsonl" for short, repeat, path, _ in all_runs[1:] if (short, repeat) != ("roco", 1)]
    usage = p1_token_usage.summarize(p1_token_usage.read_events(path) for path in usage_paths)
    complete = all(len(row["runs"]) == 3 for row in by_method.values())
    summary = {"status": "complete" if complete else "incomplete", "methods": by_method,
               "actual_training_runs_with_records": len(all_runs), "stopped_before_remaining_runs": stop,
               "first_run_extra_model_cli_invocations": 1, "token_usage_all_segments": usage}
    summary_path = RECOVERY / "continuation_summary.json"
    with summary_path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(summary, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    record("batch_finished", status=summary["status"], summary=str(summary_path))
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.check:
        check(require_complete=False)
        print("后续批次配置与首组续跑身份核验通过；未发起模型调用。")
        return 0
    return run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"后续批次停止：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
