"""独立核验 RoCo 两代试跑的来源、路线、筛选与共享反馈。"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
REFERENCE = ROOT / "outputs/p1_control/20261111_172813_speed_feedback_i3_g2_r2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(first, second):
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config_path, events_path = RUN / "config.json", RUN / "events.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    reference = json.loads((REFERENCE / "config.json").read_text(encoding="utf-8"))
    require(config.get("method", {}).get("id") == "roco_paper_adapted", "方法身份不符。")
    for field in ("task", "model", "resources", "budget", "trial_adjustment"):
        require(config[field] == reference[field], f"与参照试跑的 {field} 不一致。")
    task = config["task"]
    require((task["initial_proposals"], task["population_size"], task["generations"],
             task["collaboration_rounds"], task["evaluation_mode"]) == (3, 2, 2, 2, "final_quality"),
            "试跑合同不符。")
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    require([row["event_index"] for row in events] == list(range(1, len(events) + 1)), "事件序号不连续。")
    finished = bool(events and events[-1]["event"] == "pilot_completed")
    stopped = bool(events and events[-1]["event"] == "pilot_stopped")
    require(bool(events), "试跑尚无事件记录。")
    if finished:
        require(any(row["event"] == "finish" and row["status"] == "completed" for row in events),
                "控制器未完成。")
    preflight = next(row for row in events if row["event"] == "preflight")
    require(preflight["reference_config_sha256"] == digest(REFERENCE / "config.json")
            and preflight["reference_events_sha256"] == digest(REFERENCE / "events.jsonl"),
            "参照试跑配置或事件记录变化。")
    checked = 0
    for filename, expected in preflight["code_sha256"].items():
        source = Path(filename).resolve()
        if source.is_relative_to(RUN):
            preserved = source
        elif source.is_relative_to(ROOT) and source.suffix in (".py", ".json", ".yaml", ".txt"):
            preserved = RUN / "source_snapshots" / source.relative_to(ROOT)
        else:
            preserved = source
        require(preserved.is_file() and digest(preserved) == expected, f"输入或源码快照不符：{source}")
        checked += 1

    dataset = ROOT / task["dataset"]
    require(digest(dataset) == task["dataset_sha256"], "训练集哈希不符。")
    instances = np.load(dataset, allow_pickle=False)
    require(instances.shape == (5, 50, 2), "训练集形状不符。")
    evaluation_events = [row for row in events if row["event"] == "evaluation"]
    evaluations = {row["candidate_id"]: row for row in evaluation_events}
    require(len(evaluations) == len(evaluation_events), "存在候选重复评价记录。")
    registrations = {row["candidate_id"]: row for row in events if row["event"] == "candidate_registered"}
    proposals = {row["candidate_id"]: row for row in events if row["event"] == "proposal_response"}
    invalid = {row["candidate_id"]: row["error"] for row in events if row["event"] == "candidate_invalid"}
    passed = {cid: row for cid, row in evaluations.items() if row["result"]["status"] == "passed"}
    code_hashes, max_error = {}, 0.0
    for cid, event in evaluations.items():
        folder, result = RUN / "candidates" / cid, event["result"]
        require(json.loads((folder / "result.json").read_text(encoding="utf-8")) == result,
                f"{cid} 的结果文件和事件不一致。")
        if result["status"] != "passed":
            continue
        code = (folder / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8")
        require(code.endswith("\n"), f"{cid} 的候选代码不完整。")
        code_hashes[cid] = hashlib.sha256(code[:-1].encode("utf-8")).hexdigest()
        rows = [json.loads(line[len("P0_JSON "):]) for line in (folder / "stdout.txt").read_text(
            encoding="utf-8").splitlines() if line.startswith("P0_JSON ")]
        require([row.get("event") for row in rows] == ["start"] + ["instance"] * 5 + ["complete"],
                f"{cid} 的逐例记录不完整。")
        require(rows[0].get("time_scope") == "solve_wall_seconds_including_preparation_and_validation",
                f"{cid} 的计时范围不符。")
        objectives, times = [], []
        for index, (row, points) in enumerate(zip(rows[1:-1], instances)):
            tour = row["tour"]
            require(row["index"] == index and row["seed"] == task["evaluation_seeds"][index]
                    and row["iterations"] == task["iterations_per_instance"]
                    and sorted(tour) == list(range(50)), f"{cid} 实例 {index} 的路线或合同无效。")
            actual = math.fsum(math.hypot(float(points[a, 0] - points[b, 0]),
                                          float(points[a, 1] - points[b, 1]))
                               for a, b in zip(tour, tour[1:] + tour[:1]))
            max_error = max(max_error, abs(actual - row["objective"]))
            require(same(actual, row["objective"]), f"{cid} 实例 {index} 的路径长度重算不符。")
            elapsed = row.get("solve_wall_seconds")
            require(isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                    and math.isfinite(elapsed) and elapsed > 0, f"{cid} 实例 {index} 的计时无效。")
            objectives.append(row["objective"])
            times.append(elapsed)
        average, average_time = math.fsum(objectives) / 5, math.fsum(times) / 5
        require(same(average, result["average"]) and same(average, rows[-1]["average"])
                and same(result["fitness"], -average), f"{cid} 的平均目标或 Fitness 不符。")
        require(same(average_time, result["average_solve_wall_seconds"])
                and same(average_time, rows[-1]["average_solve_wall_seconds"])
                and times == result["solve_wall_seconds"], f"{cid} 的平均耗时不符。")

    def replay(pool, at):
        available = [cid for cid, row in passed.items() if row["event_index"] < at]
        earliest = {}
        for cid in sorted(available, key=lambda item: passed[item]["result"]["completed_monotonic_ns"]):
            earliest.setdefault(code_hashes[cid], cid)
        unique = {}
        valid = [cid for cid in pool if cid in passed and passed[cid]["event_index"] < at]
        for cid in sorted(valid, key=lambda item: proposals[item]["generation_order"]):
            unique.setdefault(code_hashes[cid], cid)
        ranked = sorted(unique.values(), key=lambda cid: (
            -passed[earliest[code_hashes[cid]]]["result"]["fitness"], proposals[cid]["generation_order"]))
        require(len(ranked) >= 2, "可重放的有效不同候选不足。")
        return [(cid, earliest[code_hashes[cid]]) for cid in ranked[:2]]

    groups = []
    initial = next((row for row in events if row["event"] == "initial_population"), None)
    if initial:
        pool = [cid for cid, row in registrations.items() if row["stage"].startswith("init_")]
        selected = replay(pool, initial["event_index"])
        require([cid for cid, _ in selected] == initial["candidate_ids"], "初始种群重放不符。")
        groups.append({"generation": 0, "selected": selected})
    selections = [row for row in events if row["event"] == "selection"]
    for selection in selections:
        expected = replay(selection["pool"], selection["event_index"])
        actual = [(row["candidate_id"], row["evaluation_origin"]) for row in selection["selected"]]
        require(expected == actual, f"第 {selection['generation']} 代筛选重放不符。")
        eligible = [cid for cid, row in registrations.items() if row["generation"] == selection["generation"]
                    and (row["stage"].startswith(("eoh_", "mutation_")) or row["stage"] in (
                        "round_2_explorer", "round_2_exploiter", "fusion_2", "fusion_final"))]
        require(set(selection["pool"]) == set(eligible), "固定代末候选池不符。")
        groups.append({"generation": selection["generation"], "selected": actual})
    if finished:
        require(len(groups) == 3 and len(selections) == 2, "未完成初始筛选及两代筛选。")

    def details(cid, origin=None):
        result = passed[origin or cid]["result"]
        return {"candidate_id": cid, "evaluation_origin": origin or cid, "stage": registrations[cid]["stage"],
                "average_objective": result["average"],
                "average_solve_wall_seconds": result["average_solve_wall_seconds"]}

    populations = [{"generation": group["generation"],
                    "members": [details(cid, origin) for cid, origin in group["selected"]]} for group in groups]
    elite_pairs = {row["generation"]: row for row in events if row["event"] == "elite_pair"}
    transitions = [row for row in events if row["event"] == "branch_transition"]
    branch_results = []
    group_map = {group["generation"]: dict(group["selected"]) for group in groups}
    for generation in (1, 2):
        if generation not in elite_pairs:
            continue
        for role in ("Explorer", "Exploiter"):
            steps = [row for row in transitions if row["generation"] == generation and row["role"] == role]
            require([step["round"] for step in steps] == list(range(len(steps))), "分支轮次不连续。")
            previous = elite_pairs[generation]["b"]
            origin = group_map[generation - 1][previous]
            for step in steps:
                require(step["previous"] == previous and registrations[step["proposal"]]["parents"] == [previous],
                        "分支基底或续写父代不符。")
                if step["proposal"] in passed:
                    require(step["current"] == step["proposal"], "有效提案没有续写。")
                    origin = step["proposal"]
                else:
                    require(step["current"] == previous, "无效提案没有正确回退。")
                item = details(step["current"], origin)
                require(same(step["to_objective"], item["average_objective"]), "分支目标变化不符。")
                branch_results.append({"generation": generation, "role": role, "round": step["round"],
                                       "previous": previous, "proposal": step["proposal"], **item})
                previous = step["current"]
    if finished:
        require(len(branch_results) == 12, "两代分支状态不完整。")

    critics = [row for row in events if row["event"] == "model_response"
               and row["stage"] in ("critic_initial", "critic_1", "critic_2")]
    routes, input_pairs = 0, 0
    branches = {(row["generation"], row["role"], row["round"]): row for row in branch_results}

    def candidate_code(cid):
        return (RUN / "candidates" / cid / "problems/tsp_aco/gpt.py").read_text(
            encoding="utf-8").removesuffix("\n")

    for event in critics:
        match = re.fullmatch(r"\s*<ref>\s*(.*?)\s*</ref>\s*<ans>\s*(.*?)\s*</ans>\s*",
                             event["response"], re.DOTALL)
        if match is None and not finished:
            continue
        require(match is not None and all(value.strip() for value in match.groups()), "共享 Critic 标签不完整。")
        reflection, advice = (value.strip() for value in match.groups())
        prompt = event["prompt"]
        require("measured timing" not in prompt and "benefit per solve second" not in prompt,
                "RoCo 共享 Critic 意外读取了速度反馈字段。")
        generation = event["generation"]
        if event["stage"] == "critic_initial":
            pair = elite_pairs[generation]
            first = details(pair["a"], group_map[generation - 1][pair["a"]])
            second = details(pair["b"], group_map[generation - 1][pair["b"]])
            labels = (("First", "Second") if first["average_objective"] == second["average_objective"]
                      else ("Better", "Worse"))
            for label, candidate in zip(labels, (first, second)):
                require(f"{label} code:\n{candidate_code(candidate['candidate_id'])}\n"
                        f"{label} objective score: {candidate['average_objective']}" in prompt,
                        "初始 Critic 输入与对应精英不符。")
        else:
            previous_round = int(event["stage"].split("_")[1]) - 1
            for label, role in (("A", "Explorer"), ("B", "Exploiter")):
                candidate = branches[generation, role, previous_round]
                require(f"Candidate {label} code: {candidate_code(candidate['candidate_id'])}\n"
                        f"Candidate {label} objective score: {candidate['average_objective']}" in prompt,
                        "逐轮 Critic 输入与对应分支不符。")
        input_pairs += 1
        for role in ("explorer", "exploiter"):
            stage = ("start_" if event["stage"] == "critic_initial" else
                     "round_" + event["stage"].split("_")[1] + "_") + role
            matches = [row for row in proposals.values() if row["generation"] == event["generation"]
                       and row["stage"] == stage]
            if not matches and not finished:
                continue
            require(len(matches) == 1, "角色提案缺失或重复。")
            branch_prompt = matches[0]["prompt"]
            require(f"Critic's feedback: {advice}" in branch_prompt
                    and f"Reflection: {reflection}" in branch_prompt, "共享反馈路由不符。")
            routes += 1
    if finished:
        require(len(critics) == 6 and input_pairs == 6 and routes == 12, "Critic 输入或共享反馈路由数量不符。")

    labels = {"explorer": "Explorer", "exploiter": "Exploiter", "integrator": "Integrator"}
    mapping = next(row for row in events if row["event"] == "role_mapping")
    require(mapping["internal"] == labels, "RoCo 角色映射不符。")
    history = [row for row in events if row["event"] == "history_appended"]
    reflections = {(row["generation"], row["stage"]): row["response"] for row in events
                   if row["event"] == "model_response" and row["stage"].startswith("reflection_")}
    mutation_checks = 0
    for row in proposals.values():
        if not row["stage"].startswith("mutation_"):
            continue
        role = row["stage"].split("_", 2)[2]
        require(role in labels, "变异角色不属于 RoCo 合同。")
        previous = [item["reflection"] for item in history
                    if item["generation"] < row["generation"] and item["role"] == labels[role]]
        current = reflections[row["generation"], f"reflection_{role}"].strip()
        if role != "integrator":
            match = re.fullmatch(r"\s*<ans>\s*(.*?)\s*</ans>\s*", current, re.DOTALL)
            require(match is not None, "角色长期反思的格式不符。")
            current = match.group(1).strip()
        require(f"[History reflection as reference]:\n{' | '.join(previous)}\n" in row["prompt"]
                and f"[Reflection insights]:\n{current}\n" in row["prompt"], "变异没有读取本角色反思。")
        mutation_checks += 1
    if finished:
        require(mutation_checks == 12 and len(history) == 6, "两代变异或角色历史不完整。")

    attempts = [row for row in events if row["event"] == "model_attempt"]
    require(all(row["retry"] is False for row in attempts), "出现未经合同允许的自动重试。")
    require(len(attempts) <= 61 and len(evaluations) <= 49, "超出模型或评价预算。")
    usages = [row["usage"] for row in events if row["event"] == "model_usage"]
    totals = {}
    for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
        values = [entry.get(key) for entry in usages]
        totals[key] = sum(values) if len(usages) == len(attempts) and all(type(value) is int for value in values) else None
    cpu_records = [row for row in events if row["event"] == "evaluation_cpu_final"]
    cpu_total = events[-1].get("process_tree_cpu_seconds")
    if cpu_total is None and cpu_records:
        cpu_total = cpu_records[-1]["total_cpu_seconds"]
    output = {
        "status": "passed" if finished else "partial", "scope": config["scope"],
        "source_events_sha256": digest(events_path), "source_config_sha256": digest(config_path),
        "verified_protected_inputs": checked, "dataset_sha256": digest(dataset), "model_calls": len(attempts),
        "model_process_failures": sum(row["result"].get("exit_code") != 0 for row in events
                                      if row["event"] == "model_process"),
        "evaluation_statuses": dict(Counter(row["result"]["status"] for row in evaluations.values())),
        "invalid_proposals": invalid, "candidate_evaluations": len(evaluations),
        "independently_recomputed_tours": len(passed) * 5, "maximum_route_absolute_error": max_error,
        "population_replay": populations,
        "elite_pairs": {str(g): {"a": row["a"], "b": row["b"]} for g, row in elite_pairs.items()},
        "branch_transitions_verified": len(branch_results), "branch_results": branch_results,
        "critic_comparisons": len(critics), "critic_input_pairs_verified": input_pairs,
        "critic_output_routes_verified": routes, "mutation_history_checks": mutation_checks,
        "model_usage_records": len(usages), "model_usage_totals": totals,
        "process_tree_cpu_seconds": cpu_total,
        "elapsed_seconds": events[-1].get("elapsed_seconds"),
        "stop_reason": events[-1].get("error"),
    }
    if not finished and not stopped:
        print(f"进行中只读核验：{len(passed)} 个有效候选，{len(passed) * 5} 条路线，"
              f"{input_pairs} 次 Critic 输入及 {routes} 条反馈路由。")
        return 0
    with (RUN / "audit_roco_pilot.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"RoCo 试跑审计 {output['status']}：{len(passed)} 个有效候选，"
          f"{len(passed) * 5} 条路线，{len(critics)} 次 Critic 比较。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
