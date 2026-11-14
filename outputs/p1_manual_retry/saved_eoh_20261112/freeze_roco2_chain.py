"""核验 RoCo 第二次研发的断点链并冻结最终候选。"""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
RUN = ROOT / "outputs/p1_roco/20261113_134209_41bdbcb7_resume"
sys.path.insert(0, str(ROOT))

from scripts import p1_freeze_candidate, p1_formal_matrix as matrix
from scripts.p0_runtime import digest


spec = importlib.util.spec_from_file_location("roco_gen6_audit", RECOVERY / "resume_roco_generation6.py")
recovery = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recovery
spec.loader.exec_module(recovery)
require, read_events = recovery.require, recovery.read_events


def audit():
    terminal = RUN.resolve()
    config = json.loads((terminal / "config.json").read_text(encoding="utf-8"))
    require(config["method"]["id"] == "roco_paper_adapted" and config["formal_replicate"] == 2,
            "研发方法或重复索引不符。")
    matrix.verify_training_config(config)
    segments, seen, current = [], set(), terminal
    while True:
        require(current.is_relative_to((ROOT / "outputs/p1_roco").resolve()) and current not in seen,
                "断点链越界或成环。")
        seen.add(current)
        require(json.loads((current / "config.json").read_text(encoding="utf-8")) == config, "断点配置不一致。")
        rows = read_events(current / "events.jsonl")
        require([x["event_index"] for x in rows] == list(range(1, len(rows) + 1)), "断点事件编号不连续。")
        starts = [x for x in rows if x["event"] == "formal_run_started"]
        preflights = [x for x in rows if x["event"] == "preflight"]
        require(len(starts) == len(preflights) == 1, "断点启动或预检记录不唯一。")
        start = starts[0]
        require(start.get("method") == config["method"]["id"] and start.get("formal_replicate") == 2
                and start.get("matrix_contract_sha256") == config["matrix_contract_sha256"]
                and start.get("selection_seed") == config["task"]["selection_seed"], "断点身份不符。")
        require(rows[-1]["event"] == ("finish" if current == terminal else "preflight_or_run_stopped"),
                "断点结束状态不符。")
        if current == terminal:
            require(rows[-1].get("status") == "completed", "最终研发未完成。")
        for filename, expected in preflights[0]["code_sha256"].items():
            require(Path(filename).is_file() and digest(Path(filename)) == expected, f"受保护文件变化：{filename}")
        segments.append((current, rows))
        if not start.get("resume_of"):
            break
        parent = Path(start["resume_of"]).resolve()
        require(start.get("interrupted_events_sha256") == digest(parent / "events.jsonl"), "断点父记录哈希不符。")
        current = parent
    segments.reverse()
    all_rows, saved_dirs = [], []
    for directory, rows in segments:
        all_rows.extend(rows)
        for saved_dir, saved_rows, _ in recovery.chain.find_saved(directory):
            saved_dirs.append(saved_dir)
            all_rows.extend(saved_rows)
    model_calls = sum(x["event"] == "model_attempt" for x in all_rows)
    eval_ids = [x["candidate_id"] for x in all_rows if x["event"] == "evaluation_budget_reserved"]
    results = [x for x in all_rows if x["event"] == "evaluation"]
    require(model_calls == 420 and len(eval_ids) == 358 and len(eval_ids) == len(set(eval_ids)),
            "累计预算或评价 ID 不符。")
    require(len(results) == len(eval_ids) and {x["candidate_id"] for x in results} == set(eval_ids),
            "累计评价结果缺失或重复。")
    require([x["generation"] for x in all_rows if x["event"] == "selection"] == list(range(1, 7)),
            "六代筛选记录不完整。")
    terminal_rows = segments[-1][1]
    candidate_id = p1_freeze_candidate.final_candidate(terminal_rows, config["method"]["id"])
    selected = [x for x in terminal_rows if x["event"] == "selection"][-1]["selected"][0]
    require(selected["evaluation_origin"] == candidate_id, "最终候选的评价来源需要单独处理。")
    registrations = [(directory, x) for directory, rows in segments for x in rows
                     if x["event"] == "candidate_registered" and x.get("candidate_id") == candidate_id]
    reused = [x for x in terminal_rows if x["event"] == "reused_candidate_registration"
              and x.get("candidate_id") == candidate_id]
    require(len(registrations) == len(reused) == 1, "候选注册来源缺失或重复。")
    source_dir, registration = registrations[0]
    require((registration["generation"], registration["stage"], registration["parents"])
            == (reused[0]["generation"], reused[0]["stage"], reused[0]["parents"]), "候选注册与复用记录不符。")
    preflight = next(x for x in terminal_rows if x["event"] == "preflight")
    source_events = source_dir / "events.jsonl"
    require(preflight["code_sha256"].get(str(source_events)) == digest(source_events), "候选注册来源未受终点保护。")
    return {"segments": len(segments), "saved_evaluations": len(saved_dirs), "model_calls": model_calls,
            "evaluation_starts": len(eval_ids), "evaluation_statuses": dict(Counter(x["result"]["status"] for x in results)),
            "candidate_id": candidate_id, "registration_events": str(source_events), "registration": registration}


def freeze(state):
    original = p1_freeze_candidate.original_code
    registration = state["registration"]

    def with_registration(events, method, candidate_id, contract):
        require(candidate_id == state["candidate_id"] and not any(
            x.get("event") == "candidate_registered" and x.get("candidate_id") == candidate_id for x in events),
            "终点候选注册状态变化。")
        return original([registration, *events], method, candidate_id, contract)

    with patch.object(p1_freeze_candidate, "original_code", side_effect=with_registration):
        return p1_freeze_candidate.freeze(RUN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    state = audit()
    output = {k: v for k, v in state.items() if k != "registration"}
    output["status"] = "passed"
    if args.freeze:
        output["frozen_dir"] = str(freeze(state))
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
