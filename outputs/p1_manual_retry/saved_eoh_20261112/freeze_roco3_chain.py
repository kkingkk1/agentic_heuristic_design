"""核验 RoCo 第 3 次研发的完整断点链并冻结最终候选。"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from unittest.mock import patch


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
TERMINAL = ROOT / "outputs/p1_roco/20261113_235648_34d12148_resume"
sys.path.insert(0, str(ROOT))

from scripts import p1_freeze_candidate, p1_formal_matrix as matrix
from scripts.p0_runtime import digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_events(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit():
    terminal = TERMINAL.resolve()
    config = json.loads((terminal / "config.json").read_text(encoding="utf-8"))
    require(config["method"]["id"] == "roco_paper_adapted" and config["formal_replicate"] == 3,
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
        require(start.get("method") == config["method"]["id"] and start.get("formal_replicate") == 3
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
    require(len(segments) == 10, "断点片段数量不符。")
    model_calls = evaluation_starts = 0
    all_rows = []
    for i, (_, rows) in enumerate(segments):
        preflight = next(x for x in rows if x["event"] == "preflight")
        if i:
            require(preflight.get("physical_model_cli_invocations_used") == model_calls
                    and preflight.get("evaluation_starts_used") == evaluation_starts, "累计预算不连续。")
        model_calls += sum(x["event"] == "model_attempt" for x in rows)
        evaluation_starts += sum(x["event"] == "evaluation_budget_reserved" for x in rows)
        all_rows.extend(rows)
    eval_ids = [x["candidate_id"] for x in all_rows if x["event"] == "evaluation_budget_reserved"]
    results = [x for x in all_rows if x["event"] == "evaluation"]
    require(model_calls == 417 and evaluation_starts == 362 and len(eval_ids) == len(set(eval_ids))
            and len(results) == len(eval_ids) and {x["candidate_id"] for x in results} == set(eval_ids),
            "累计预算或评价记录不符。")
    require([x["generation"] for x in all_rows if x["event"] == "selection"] == list(range(1, 7)),
            "六代筛选记录不完整。")
    terminal_rows = segments[-1][1]
    candidate_id = p1_freeze_candidate.final_candidate(terminal_rows, config["method"]["id"])
    selected = [x for x in terminal_rows if x["event"] == "selection"][-1]["selected"][0]
    require(candidate_id == "c0324" and selected["evaluation_origin"] == candidate_id,
            "最终候选或评价来源变化。")
    registrations = [(directory, x) for directory, rows in segments for x in rows
                     if x["event"] == "candidate_registered" and x.get("candidate_id") == candidate_id]
    producers = [(directory, rows) for directory, rows in segments
                 if any(x["event"] == "proposal_response" and x.get("candidate_id") == candidate_id for x in rows)]
    evaluations = [(directory, x) for directory, rows in segments for x in rows
                   if x["event"] == "evaluation" and x.get("candidate_id") == candidate_id]
    require(len(registrations) == len(producers) == len(evaluations) == 1, "最终候选来源缺失或重复。")
    producer, producer_rows = producers[0]
    require(registrations[0][0] == producer == evaluations[0][0] and evaluations[0][1]["result"]["status"] == "passed",
            "候选注册、生成或评价来源不符。")
    protected = next(x for x in terminal_rows if x["event"] == "preflight")["code_sha256"]
    require(protected.get(str(producer / "events.jsonl")) == digest(producer / "events.jsonl"),
            "最终候选来源未受终点保护。")
    return {"segments": len(segments), "model_calls": model_calls, "evaluation_starts": evaluation_starts,
            "evaluation_statuses": dict(Counter(x["result"]["status"] for x in results)),
            "candidate_id": candidate_id, "evaluation_origin": selected["evaluation_origin"],
            "producer": str(producer), "terminal": str(terminal)}


def freeze(state):
    original_dump = json.dump

    def final_candidate(events, method):
        require(method == "roco_paper_adapted" and any(x.get("event") == "proposal_response"
                and x.get("candidate_id") == state["candidate_id"] for x in events), "候选生成来源变化。")
        return state["candidate_id"]

    def augmented_dump(obj, stream, *args, **kwargs):
        require(obj.get("status") == "frozen_after_training"
                and obj.get("source_candidate_id") == state["candidate_id"], "冻结清单身份不符。")
        data = dict(obj, training_completion_run=state["terminal"],
                    training_completion_events_sha256=digest(Path(state["terminal"]) / "events.jsonl"),
                    selection_evaluation_origin=state["evaluation_origin"])
        return original_dump(data, stream, *args, **kwargs)

    with patch.object(p1_freeze_candidate, "final_candidate", side_effect=final_candidate), \
            patch.object(p1_freeze_candidate.json, "dump", side_effect=augmented_dump):
        return p1_freeze_candidate.freeze(Path(state["producer"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    state = audit()
    state["status"] = "passed"
    if args.freeze:
        state["frozen_dir"] = str(freeze(state))
    print(json.dumps(state, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
