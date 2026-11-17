"""将已完成的 TSP50 断点链回填为按代记录；只读取历史来源，不调用模型。"""

from __future__ import annotations

import argparse
import ast
import bisect
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe  # noqa: E402
from scripts.p0_controller import code_hash  # noqa: E402
from scripts.p0_runtime import digest  # noqa: E402
from scripts.p1_runtime import _evaluation_evidence  # noqa: E402
from scripts.p1_task_contract import load_task_contract  # noqa: E402


EVIDENCE = ROOT / "vault/Results/Evidence/tsp50-roco-20261114"
HISTORICAL_INDEX = ROOT / "outputs/p1_manual_retry/saved_eoh_20261112/analyze_tsp50_roco.py"
CONTRACT = load_task_contract("tsp_aco")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), f"来源或目标超出项目：{path}")
    return path.relative_to(ROOT).as_posix()


def source_ref(event):
    return event["source_event"]


def verified_inputs():
    """核验原证据清单；只解析历史脚本中的静态路径，不执行该脚本。"""
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("experiment_id") == "tsp50-roco-20261114", "证据包身份不符。")
    entries = [*manifest["source_files"], manifest["canonical_table"], manifest["historical_record"]]
    hashes = {}
    for entry in entries:
        path = Path(entry["path"]).resolve()
        relative(path)
        require(path.is_file() and digest(path) == entry["sha256"], f"证据文件缺失或哈希不符：{path}")
        if path in hashes:
            require(hashes[path] == entry["sha256"], f"来源哈希冲突：{path}")
        hashes[path] = entry["sha256"]
    require(HISTORICAL_INDEX.resolve() in hashes, "历史路径索引未纳入证据清单。")
    literals = {}
    for node in ast.parse(HISTORICAL_INDEX.read_text(encoding="utf-8")).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("RUNS", "EXTRAS")):
            name = node.targets[0].id
            require(name not in literals, f"历史路径索引重复：{name}")
            literals[name] = ast.literal_eval(node.value)
    require(set(literals) == {"RUNS", "EXTRAS"}, "历史路径索引不完整。")
    with (EVIDENCE / "data.csv").open(encoding="utf-8", newline="") as stream:
        table = list(csv.DictReader(stream))
    expected = {(row["method"], int(row["replicate"])): row for row in table}
    require(len(expected) == len(table) == 6, "运行级证据包含缺失或重复的研发。")
    require({(row[0], row[1]) for row in literals["RUNS"]} == set(expected), "研发路径与证据矩阵不符。")
    return literals["RUNS"], literals["EXTRAS"], expected, hashes


def read_events(path, hashes):
    path = Path(path).resolve()
    require(path in hashes, f"事件文件未纳入来源清单：{path}")
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    previous_index, previous_time = 0, None
    for event in events:
        index, timestamp = event.get("event_index"), event.get("at_utc")
        require(type(index) is int and index > previous_index and isinstance(timestamp, str),
                f"事件索引或时间缺失：{path}")
        when = datetime.fromisoformat(timestamp)
        require(when.utcoffset() is not None and (previous_time is None or when >= previous_time),
                f"同一事件文件的时间倒序或缺少时区：{path}")
        previous_index, previous_time = index, when
    return events


def history_chain(terminal, hashes):
    """沿已登记的续跑指针回溯，并验证每个父片段的哈希。"""
    terminal = Path(terminal).resolve()
    require(terminal.is_relative_to((ROOT / "outputs").resolve()), "研发终点不在项目输出目录。")
    paths, seen = [], set()
    while True:
        require(terminal not in seen and terminal.is_relative_to((ROOT / "outputs").resolve()),
                "研发断点链成环或越界。")
        seen.add(terminal)
        path = terminal / "events.jsonl"
        events = read_events(path, hashes)
        starts = [event for event in events if event.get("event") == "formal_run_started"]
        require(len(starts) == 1, f"研发启动记录不唯一：{path}")
        paths.append(path)
        parent = starts[0].get("resume_of")
        if not parent:
            return list(reversed(paths))
        parent_path = Path(parent).resolve() / "events.jsonl"
        expected_hash = starts[0].get("interrupted_events_sha256") or starts[0].get("original_events_sha256")
        require(parent_path in hashes and hashes[parent_path] == expected_hash, f"断点来源哈希不符：{parent_path}")
        terminal = parent_path.parent


def merge_events(paths, hashes):
    """按 UTC 时间合并；相同时间按来源顺序及原编号稳定排序。"""
    merged = []
    for source_order, path in enumerate(paths):
        for event in read_events(path, hashes):
            require("source_event" not in event, "历史事件占用了来源字段。")
            merged.append((datetime.fromisoformat(event["at_utc"]), source_order, event["event_index"], path, event))
    merged.sort(key=lambda item: item[:3])
    result = []
    for logical_index, (_, _, original_index, path, event) in enumerate(merged, 1):
        item = dict(event)
        item["source_event"] = {"path": relative(path), "event_index": original_index}
        item["event_index"] = logical_index
        result.append(item)
    return result


def registrations(events, method, replicate):
    rows = {}
    for event in events:
        if event.get("event") != "candidate_registered":
            continue
        ident = event.get("candidate_id")
        require(isinstance(ident, str) and ident.startswith("c") and ident[1:].isdigit()
                and ident not in rows and event.get("generation") in range(7), "候选注册缺失、重复或无效。")
        keys = ("candidate_id", "sequence", "stage", "source_role", "operator", "generation", "round", "parents")
        row = {key: event.get(key) for key in keys}
        row["source_events"] = {"registration": source_ref(event)}
        if row["source_role"] is None or row["operator"] is None:
            # 该手动融合登记未记录角色；阶段名与项目融合合同共同确定这两个字段。
            require(method == "proposed_solution_quality_and_speed_agents" and replicate == 1
                    and ident == "c0079" and row["stage"] == "fusion_2_manual_retry" and row["round"] == 2,
                    f"候选角色或算子缺失：{ident}")
            row.update(source_role="integrator", operator="round_fusion")
            row["historical_derivations"] = {"source_role": "stage_and_fusion_contract",
                                             "operator": "stage_and_fusion_contract"}
        rows[ident] = row

    evaluations = [event for event in events if event.get("event") == "evaluation"]
    missing = {event.get("candidate_id") for event in evaluations} - set(rows)
    if missing:
        # 唯一的手动重试直接记录了复用响应，没有常规 candidate_registered。
        require(method == "proposed_solution_quality_and_speed_agents" and replicate == 1
                and missing == {"c0082"}, f"评价缺少候选注册：{missing}")
        reused = [event for event in events if event.get("event") == "reused_or_retried_response"
                  and event.get("candidate_id") == "c0082"]
        transitions = [event for event in events if event.get("event") == "branch_transition"
                       and event.get("proposal") == "c0082"]
        require(len(reused) == len(transitions) == 1, "手动候选的响应或分支来源不唯一。")
        item, transition = reused[0], transitions[0]
        require(item.get("sequence") == 82 and item.get("parents") == [transition.get("previous")]
                and item.get("stage") == "round_3_solution_quality_agent_manual_retry"
                and transition.get("generation") == 1 and transition.get("round") == 3
                and transition.get("role") == "solution_quality_agent", "手动候选身份与分支记录不符。")
        rows["c0082"] = {"candidate_id": "c0082", "sequence": 82, "stage": item["stage"],
                         "source_role": transition["role"], "operator": "continuation", "generation": 1,
                         "round": transition["round"], "parents": item["parents"],
                         "source_events": {"registration": None, "reused_response": source_ref(item),
                                           "branch_transition": source_ref(transition)},
                         "historical_derivations": {"registration": "not_recorded",
                                                    "generation_role_round": "branch_transition",
                                                    "operator": "stage_and_branch_contract"}}
    return rows


def phase_boundaries(events, rows):
    initial = [event for event in events if event.get("event") == "initial_population"
               and event.get("candidate_ids")
               and all(rows.get(ident, {}).get("generation") == 0 for ident in event["candidate_ids"])]
    require(initial, "缺少初始化种群。")
    initial.sort(key=lambda event: event["event_index"])
    require(all(event["candidate_ids"] == initial[0]["candidate_ids"] for event in initial),
            "恢复记录中的初始化种群不一致。")
    selections = {0: initial[0]}
    ends = {0: initial[0]["event_index"]}
    for event in events:
        kind, generation = event.get("event"), event.get("generation")
        if kind == "selection":
            require(type(generation) is int and generation in range(1, 7) and generation not in selections,
                    "代末筛选缺失、重复或代次无效。")
            selections[generation] = event
        if kind == "history_appended" and str(event.get("role", "")).lower() == "integrator":
            require(type(generation) is int and generation in range(1, 7) and generation not in ends,
                    "Integrator 反思的代次无效或重复。")
            ends[generation] = event["event_index"]
    require(set(selections) == set(ends) == set(range(7)), "研发未完成初始化及六代筛选和反思。")
    boundaries = [ends[generation] for generation in range(7)]
    require(boundaries == sorted(boundaries) and len(set(boundaries)) == 7, "阶段边界时间不一致。")
    for generation in range(1, 7):
        require(selections[generation]["event_index"] < ends[generation], "筛选未先于角色历史写入。")
    for event in events:
        phase = min(6, bisect.bisect_left(boundaries, event["event_index"]))
        if (event.get("event") in ("candidate_registered", "proposal_response", "selection", "model_response",
                                  "branch_transition", "history_appended", "slot_skipped")
                and type(event.get("generation")) is int):
            require(event["generation"] == phase, "事件代次与筛选边界不一致。")
        if event.get("event") in ("evaluation", "candidate_invalid", "candidate_skipped",
                                  "candidate_infrastructure_error", "reused_or_retried_response"):
            ident = event.get("candidate_id")
            if ident in rows:
                require(rows[ident]["generation"] == phase, f"候选事件跨代：{ident}")
    return boundaries, selections


def proposal_and_evaluation(events, rows):
    proposals, evaluations, issues = defaultdict(list), {}, defaultdict(list)
    for event in events:
        kind, ident = event.get("event"), event.get("candidate_id")
        if kind in ("proposal_response", "reused_or_retried_response") and ident is not None:
            require(ident in rows and isinstance(event.get("response"), str), f"候选响应缺失或未注册：{ident}")
            proposals[ident].append(event)
        elif kind == "evaluation":
            require(ident in rows and ident not in evaluations and isinstance(event.get("result"), dict),
                    f"候选评价未注册或重复：{ident}")
            evaluations[ident] = event
        elif kind in ("candidate_invalid", "candidate_skipped", "candidate_infrastructure_error"):
            require(ident in rows, f"候选失败记录未注册：{ident}")
            issues[ident].append(event)
    parsed = {}
    for ident, items in proposals.items():
        response = items[0]["response"]
        require(type(items[0].get("generation_order")) is int and items[0]["generation_order"] > 0,
                f"候选生成顺序缺失：{ident}")
        require(all(item["response"] == response and item.get("generation_order") == items[0].get("generation_order")
                    for item in items), f"复用响应或生成顺序不一致：{ident}")
        rows[ident]["generation_order"] = items[0].get("generation_order")
        rows[ident]["source_events"]["proposals"] = [source_ref(item) for item in items]
        try:
            parsed[ident] = CONTRACT.parse_proposal(response)
        except (probe.ProbeError, TypeError, ValueError, SyntaxError):
            rows[ident]["raw_response"] = response
        else:
            rows[ident]["code_sha256"] = code_hash(parsed[ident]["code"])
    for ident, items in issues.items():
        rows[ident]["issues"] = items
    for ident, event in evaluations.items():
        result = event["result"]
        command = result.get("command")
        require(result.get("candidate_id") == ident and isinstance(command, list) and len(command) >= 7,
                f"候选评价身份或路径无效：{ident}")
        candidate_dir = Path(command[5]).resolve()
        require(candidate_dir.name == ident and candidate_dir.is_relative_to((ROOT / "outputs").resolve()),
                f"候选评价路径越界或身份不符：{ident}")
        result_path = candidate_dir / "result.json"
        source_path = candidate_dir / "problems/tsp_aco/gpt.py"
        require(result_path.is_file() and json.loads(result_path.read_text(encoding="utf-8")) == result,
                f"候选评价事件与文件不一致：{ident}")
        require(ident in parsed and source_path.is_file()
                and source_path.read_text(encoding="utf-8") == parsed[ident]["code"] + "\n",
                f"候选代码与原始提案不一致：{ident}")
        omitted = {"command", "resources"}
        if result.get("status") == "passed":
            omitted.update(("objectives", "solve_wall_seconds", "heuristic_wall_seconds", "stopping_reasons", "start"))
        rows[ident]["evaluation"] = {key: value for key, value in result.items() if key not in omitted}
        rows[ident].update(_evaluation_evidence(candidate_dir, result))
        rows[ident]["source_events"]["evaluation"] = source_ref(event)
        rows[ident]["source_events"]["candidate_directory"] = relative(candidate_dir)
    for ident, row in rows.items():
        row.setdefault("instances", [])
        result = evaluations.get(ident, {}).get("result")
        kinds = {item["event"] for item in issues.get(ident, [])}
        if result and kinds:
            # 断点片段可先跳过未启动槽位，后续片段再完成同一候选。
            require(kinds == {"candidate_skipped"} and
                    all(item["event_index"] < evaluations[ident]["event_index"] for item in issues[ident]),
                    f"候选失败与后续评价的顺序或类型冲突：{ident}")
        if result and result.get("status") == "passed":
            row["status"], row["valid"] = "evaluation_passed", True
        elif result and result.get("status") in ("candidate_error", "timeout"):
            row["status"], row["valid"] = result["status"], False
        elif result and result.get("status") == "infrastructure_error":
            row["status"], row["valid"] = "infrastructure_error", None
        elif "candidate_invalid" in kinds:
            row["status"], row["valid"] = "invalid_proposal", False
        elif "candidate_skipped" in kinds:
            row["status"], row["valid"] = "skipped", None
        elif kinds:
            row["status"], row["valid"] = "infrastructure_error", None
        else:
            row["status"], row["valid"] = "incomplete", None
    return parsed, evaluations


def compact_trace(events):
    """按新记录器的口径压缩成功请求，保留失败证据与原事件来源。"""
    known_messages = {(event.get("stage"), event.get("response")) for event in events
                      if event.get("event") in ("model_response", "proposal_response", "reused_or_retried_response")
                      and isinstance(event.get("response"), str)}
    prompts, trace = {}, []
    skip = {"candidate_registered", "proposal_response", "evaluation", "candidate_invalid", "candidate_skipped",
            "candidate_infrastructure_error", "selection", "preflight", "formal_run_started"}
    for event in events:
        kind, stage = event.get("event"), event.get("stage")
        if kind in skip:
            continue
        entry = dict(event)
        if kind == "model_attempt":
            prompt = entry.pop("prompt", None)
            if isinstance(prompt, str):
                entry["prompt_sha256"] = code_hash(prompt)
            prompts[(source_ref(event)["path"], stage)] = prompt
        elif kind == "model_response":
            entry.pop("prompt", None)
        elif kind == "reused_or_retried_response":
            response = entry.pop("response", None)
            if isinstance(response, str):
                entry["response_sha256"] = code_hash(response)
        elif kind == "model_process" and isinstance(entry.get("result"), dict):
            result = entry["result"]
            try:
                message, usage, event_types, warnings = probe.extract_message(result)
            except (probe.ProbeError, TypeError, ValueError, AttributeError, KeyError):
                prompt = prompts.get((source_ref(event)["path"], stage))
                if prompt is not None:
                    entry["failed_prompt"] = prompt
            else:
                compact = {key: value for key, value in result.items() if key != "stdout"}
                compact.update(usage=usage, event_types=event_types, warnings=warnings)
                try:
                    proposal = CONTRACT.parse_proposal(message)
                except (probe.ProbeError, TypeError, ValueError, SyntaxError):
                    if (stage, message) not in known_messages:
                        compact["message"] = message
                else:
                    compact["proposal_code_sha256"] = code_hash(proposal["code"])
                    if (stage, message) not in known_messages:
                        compact["proposal"] = proposal
                entry["result"] = compact
                if warnings:
                    prompt = prompts.get((source_ref(event)["path"], stage))
                    if prompt is not None:
                        entry["failed_prompt"] = prompt
        if kind == "initial_population":
            entry["historical_role"] = "resumed_population_checkpoint"
        trace.append(entry)
    return trace


def make_records(run, extra_paths, expected, hashes):
    method, replicate, _, terminal, _, expected_calls, expected_evaluations = run
    terminal = (ROOT / terminal).resolve()
    required_parent = "p1_control" if method == "proposed_solution_quality_and_speed_agents" else "p1_roco"
    require(terminal.parent == (ROOT / "outputs" / required_parent).resolve(), "研发终点与方法不符。")
    paths = history_chain(terminal, hashes) + [(ROOT / path).resolve() for path in extra_paths]
    require(len(paths) == len(set(paths)), "研发断点或手动恢复来源重复。")
    events = merge_events(paths, hashes)
    rows = registrations(events, method, replicate)
    boundaries, selections = phase_boundaries(events, rows)
    parsed, evaluations = proposal_and_evaluation(events, rows)
    require(sum(event.get("event") == "model_attempt" for event in events) == expected_calls == int(expected["model_calls"])
            and len(evaluations) == expected_evaluations == int(expected["training_evaluations"]),
            "研发的模型调用或评价计数与证据表不符。")
    statuses = Counter(event["result"].get("status") for event in evaluations.values())
    for status, column in (("passed", "training_passed"), ("candidate_error", "training_candidate_errors"),
                           ("infrastructure_error", "training_infrastructure_errors")):
        require(statuses[status] == int(expected[column]), f"研发评价状态计数不符：{status}")

    output, origins = [], {}
    for generation in range(7):
        phase = [event for event in events
                 if min(6, bisect.bisect_left(boundaries, event["event_index"])) == generation]
        require(phase, f"阶段事件缺失：{generation}")
        selection_event = selections[generation]
        if generation == 0:
            selected = selection_event["candidate_ids"]
            selection = {"event_index": selection_event["event_index"], "candidate_ids": selected,
                         "source_event": source_ref(selection_event)}
            selection_origins = {}
        else:
            selected_entries = selection_event.get("selected")
            require(isinstance(selected_entries, list) and all(isinstance(item, dict) for item in selected_entries),
                    f"代末入选记录无效：{generation}")
            selected = [item.get("candidate_id") for item in selected_entries]
            selection_origins = {item.get("candidate_id"): item.get("evaluation_origin") for item in selected_entries}
            selection = {"event_index": selection_event["event_index"], "pool": selection_event.get("pool"),
                         "selected": selected_entries, "source_event": source_ref(selection_event)}
        require(len(selected) == 10 and len(set(selected)) == 10
                and all(ident in rows and rows[ident]["generation"] == generation for ident in selected),
                f"阶段入选数量或来源无效：{generation}")
        candidates = sorted((row for row in rows.values() if row["generation"] == generation),
                            key=lambda row: row["sequence"])
        for row in candidates:
            ident = row["candidate_id"]
            row["selected"] = ident in selected
            result = evaluations.get(ident, {}).get("result")
            if row["valid"] is True:
                completed = result.get("completed_monotonic_ns")
                require(type(completed) is int and row.get("code_sha256"), f"有效候选缺少去重证据：{ident}")
                key = row["code_sha256"]
                value = (completed, row["sequence"], ident)
                origins[key] = min(origins.get(key, value), value)
            keep = (row["selected"] or row["valid"] is not True or
                    row.get("operator") in ("startup", "continuation", "round_fusion", "elite_fusion"))
            if keep and ident in parsed:
                row["thought"], row["code"] = parsed[ident]["thought"], parsed[ident]["code"]
        for ident in selected:
            row = rows[ident]
            require(row["valid"] is True and "code" in row, f"入选候选缺少有效评价或代码：{ident}")
            origin = origins[row["code_sha256"]][2]
            if generation:
                require(selection_origins.get(ident) == origin, f"入选候选去重评价来源不一致：{ident}")
            row["evaluation_origin"] = origin
        if generation == 6:
            require(selected[0] == expected["candidate_id"], "末代候选与已冻结证据不一致。")
        fragments = {}
        for event in phase:
            ref = source_ref(event)
            info = fragments.setdefault(ref["path"], {"path": ref["path"],
                                "sha256": hashes[(ROOT / ref["path"]).resolve()], "event_indexes": []})
            info["event_indexes"].append(ref["event_index"])
        source_fragments = []
        for info in fragments.values():
            indexes = info.pop("event_indexes")
            info.update(first_event_index=min(indexes), last_event_index=max(indexes), event_count=len(indexes))
            source_fragments.append(info)
        phase_trace = [event for event in phase if event is not selection_event]
        record = {"schema_version": 1, "generation": generation, "status": "completed",
                  "source_event_range": [phase[0]["event_index"], phase[-1]["event_index"]],
                  "source_event_range_scope": "historical_utc_merge; ties use source order and original event index",
                  "selection": selection, "candidates": candidates, "trace": compact_trace(phase_trace),
                  "source_fragments": source_fragments}
        output.append(record)
    return terminal, output, events, rows


def prepare():
    runs, extras, expected, hashes = verified_inputs()
    prepared = []
    for run in runs:
        key = (run[0], run[1])
        terminal, records, events, rows = make_records(run, extras.get(key, []), expected[key], hashes)
        config = json.loads(terminal.joinpath("config.json").read_text(encoding="utf-8"))
        require(config.get("task", {}).get("selection_seed") == int(expected[key]["selection_seed"])
                and config.get("method", {}).get("id") == key[0] and config.get("formal_replicate") == key[1],
                f"研发身份或种子与终点配置不符：{terminal}")
        prepared.append((key, terminal, records, events, rows))
    return prepared


def target_payloads(prepared):
    payloads = []
    for _, terminal, records, _, _ in prepared:
        target_dir = terminal / "generation_records"
        require(target_dir.resolve().parent == terminal.resolve(), f"归并目录越界：{target_dir}")
        for record in records:
            path = target_dir / f"generation_{record['generation']:04d}.json"
            require(path.resolve().parent == target_dir.resolve(), f"归并目标越界：{path}")
            data = (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
            if path.exists():
                require(path.is_file() and path.read_bytes() == data, f"目标已存在且内容不同：{path}")
            payloads.append((path, data))
    require(len(payloads) == 42 and len({path for path, _ in payloads}) == 42, "归并目标数量或路径不唯一。")
    return payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="只读重建并核验，不写入文件")
    action.add_argument("--write", action="store_true", help="核验后只创建缺失的按代文件")
    args = parser.parse_args()
    prepared = prepare()
    payloads = target_payloads(prepared)
    created = 0
    if args.write:
        for path, data in payloads:
            path.parent.mkdir(exist_ok=True)
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(data)
                created += 1
            require(path.read_bytes() == data, f"归并文件读回不一致：{path}")
    print(json.dumps({"status": "written" if args.write else "checked", "research_runs": len(prepared),
                      "generation_files": len(payloads), "created": created,
                      "candidates": sum(len(rows) for _, _, _, _, rows in prepared),
                      "evaluations": sum(sum(event.get("event") == "evaluation" for event in events)
                                         for _, _, _, events, _ in prepared),
                      "directories": [relative(terminal / "generation_records")
                                      for _, terminal, _, _, _ in prepared]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, IndexError, SyntaxError, json.JSONDecodeError) as exc:
        print(f"历史代次回填失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
