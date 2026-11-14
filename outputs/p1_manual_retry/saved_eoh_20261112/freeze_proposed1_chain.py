"""核验本方法首组断点链，并冻结带去重评价来源的最终候选。"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from unittest.mock import patch


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
RUN = ROOT / "outputs/p1_control/20261113_151011_1e444253_resume"
sys.path.insert(0, str(ROOT))

from scripts import p0_tsp_aco_evaluate, p1_freeze_candidate, p1_formal_matrix as matrix
from scripts.p0_runtime import digest


EXTRA_EVENTS = [
    ROOT / "outputs/p1_manual_retry/fusion_2_20261112_143551_66d3a6ad/events.jsonl",
    ROOT / "outputs/p1_manual_retry/fusion_2_20261112_143551_66d3a6ad/evaluation_events.jsonl",
    ROOT / "outputs/p1_manual_retry/round_3_quality_20261112_145256_3c8fe086/events.jsonl",
    ROOT / "outputs/p1_manual_retry/round_3_quality_20261112_145256_3c8fe086/evaluation_events.jsonl",
    RECOVERY / "20261112_230934_proposed_9dcc64a8_evaluation/evaluation_events.jsonl",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_events(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit():
    terminal = RUN.resolve()
    config = json.loads((terminal / "config.json").read_text(encoding="utf-8"))
    require(config["method"]["id"] == "proposed_solution_quality_and_speed_agents"
            and config["formal_replicate"] == 1, "研发方法或重复索引不符。")
    matrix.verify_training_config(config)
    segments, seen, current = [], set(), terminal
    while True:
        require(current.is_relative_to((ROOT / "outputs/p1_control").resolve()) and current not in seen,
                "断点链越界或成环。")
        seen.add(current)
        require(json.loads((current / "config.json").read_text(encoding="utf-8")) == config, "断点配置不一致。")
        rows = read_events(current / "events.jsonl")
        require([x["event_index"] for x in rows] == list(range(1, len(rows) + 1)), "断点事件编号不连续。")
        starts = [x for x in rows if x["event"] == "formal_run_started"]
        preflights = [x for x in rows if x["event"] == "preflight"]
        require(len(starts) == len(preflights) == 1, "断点启动或预检记录不唯一。")
        start = starts[0]
        require(start.get("method") == config["method"]["id"] and start.get("formal_replicate") == 1
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
        parent_hash = start.get("interrupted_events_sha256", start.get("original_events_sha256"))
        require(parent_hash == digest(parent / "events.jsonl"), "断点父记录哈希不符。")
        current = parent
    segments.reverse()
    terminal_rows = segments[-1][1]
    protected = next(x for x in terminal_rows if x["event"] == "preflight")["code_sha256"]
    all_rows = [row for _, rows in segments for row in rows]
    for path in EXTRA_EVENTS:
        path = path.resolve()
        require(protected.get(str(path)) == digest(path), f"手动恢复记录未受终点保护：{path}")
        all_rows.extend(read_events(path))
    model_calls = sum(x["event"] == "model_attempt" for x in all_rows)
    eval_ids = [x["candidate_id"] for x in all_rows if x["event"] == "evaluation_budget_reserved"]
    results = [x for x in all_rows if x["event"] == "evaluation"]
    require(model_calls == 421 and len(eval_ids) == 370 and len(eval_ids) == len(set(eval_ids)),
            "累计预算或评价 ID 不符。")
    require(len(results) == len(eval_ids) and {x["candidate_id"] for x in results} == set(eval_ids),
            "累计评价结果缺失或重复。")
    require([x["generation"] for x in all_rows if x["event"] == "selection"] == list(range(1, 7)),
            "六代筛选记录不完整。")
    selected = [x for x in terminal_rows if x["event"] == "selection"][-1]["selected"][0]
    candidate_id, origin_id = selected["candidate_id"], selected["evaluation_origin"]
    require(candidate_id == "c0358" and origin_id == "c0302", "最终候选或去重评价来源变化。")
    registrations = [(directory, x) for directory, rows in segments for x in rows
                     if x["event"] == "candidate_registered" and x.get("candidate_id") == candidate_id]
    producers = [(directory, rows) for directory, rows in segments
                 if any(x["event"] == "proposal_response" and x.get("candidate_id") == candidate_id for x in rows)]
    origins = [(directory, rows) for directory, rows in segments
               if any(x["event"] == "evaluation" and x.get("candidate_id") == origin_id for x in rows)]
    require(len(registrations) == len(producers) == len(origins) == 1, "候选注册、生成或评价来源不唯一。")
    registration_dir, registration = registrations[0]
    producer, producer_rows = producers[0]
    origin, origin_rows = origins[0]
    producer_responses = [x for x in producer_rows if x["event"] == "proposal_response" and x.get("candidate_id") == candidate_id]
    producer_evaluations = [x for x in producer_rows if x["event"] == "evaluation" and x.get("candidate_id") == candidate_id]
    origin_evaluations = [x for x in origin_rows if x["event"] == "evaluation" and x.get("candidate_id") == origin_id]
    require(len(producer_responses) == len(producer_evaluations) == len(origin_evaluations) == 1,
            "候选或去重评价证据缺失。")
    candidate_dir, origin_dir = producer / "candidates" / candidate_id, origin / "candidates" / origin_id
    candidate_code = candidate_dir / "problems/tsp_aco/gpt.py"
    origin_code = origin_dir / "problems/tsp_aco/gpt.py"
    candidate_result = json.loads((candidate_dir / "result.json").read_text(encoding="utf-8"))
    origin_result = json.loads((origin_dir / "result.json").read_text(encoding="utf-8"))
    require(digest(candidate_code) == digest(origin_code) and candidate_result == producer_evaluations[0]["result"]
            and origin_result == origin_evaluations[0]["result"]
            and candidate_result["status"] == origin_result["status"] == "passed", "去重代码或评价证据不符。")
    for field in ("average", "fitness", "objectives"):
        require(candidate_result[field] == origin_result[field], f"去重评价字段不符：{field}")
    checked = p0_tsp_aco_evaluate.read_complete(origin_dir / "stdout.txt")
    require(all(checked[field] == origin_result[field] for field in ("average", "fitness", "objectives")),
            "去重评价输出独立重读不一致。")
    for directory in (registration_dir, producer, origin):
        path = directory / "events.jsonl"
        require(protected.get(str(path)) == digest(path), f"候选来源未受终点保护：{path}")
    return {"segments": len(segments), "manual_event_files": len(EXTRA_EVENTS), "model_calls": model_calls,
            "evaluation_starts": len(eval_ids), "evaluation_statuses": dict(Counter(x["result"]["status"] for x in results)),
            "candidate_id": candidate_id, "evaluation_origin": origin_id, "registration": registration,
            "registration_events": str(registration_dir / "events.jsonl"), "producer": str(producer),
            "origin": str(origin), "terminal": str(terminal), "origin_result": str(origin_dir / "result.json"),
            "origin_stdout": str(origin_dir / "stdout.txt")}


def freeze(state):
    original_code = p1_freeze_candidate.original_code
    original_dump = json.dump
    registration = state["registration"]
    candidate_id, method = state["candidate_id"], "proposed_solution_quality_and_speed_agents"

    def final_candidate(events, actual_method):
        require(actual_method == method and any(x.get("event") == "proposal_response"
                and x.get("candidate_id") == candidate_id for x in events), "候选生成来源变化。")
        return candidate_id

    def with_registration(events, actual_method, ident, contract):
        require(ident == candidate_id and not any(x.get("event") == "candidate_registered"
                and x.get("candidate_id") == ident for x in events), "候选注册状态变化。")
        return original_code([registration, *events], actual_method, ident, contract)

    def augmented_dump(obj, stream, *args, **kwargs):
        require(obj.get("status") == "frozen_after_training" and obj.get("source_candidate_id") == candidate_id,
                "冻结清单身份不符。")
        data = dict(obj, training_completion_run=state["terminal"],
                    training_completion_events_sha256=digest(Path(state["terminal"]) / "events.jsonl"),
                    source_registration_events=state["registration_events"],
                    source_registration_events_sha256=digest(Path(state["registration_events"])),
                    selection_evaluation_origin=state["evaluation_origin"],
                    selection_evaluation_origin_run=state["origin"],
                    selection_evaluation_origin_events_sha256=digest(Path(state["origin"]) / "events.jsonl"),
                    selection_evaluation_origin_result_sha256=digest(Path(state["origin_result"])),
                    selection_evaluation_origin_stdout_sha256=digest(Path(state["origin_stdout"])))
        return original_dump(data, stream, *args, **kwargs)

    with patch.object(p1_freeze_candidate, "final_candidate", side_effect=final_candidate), \
            patch.object(p1_freeze_candidate, "original_code", side_effect=with_registration), \
            patch.object(p1_freeze_candidate.json, "dump", side_effect=augmented_dump):
        return p1_freeze_candidate.freeze(Path(state["producer"]))


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
