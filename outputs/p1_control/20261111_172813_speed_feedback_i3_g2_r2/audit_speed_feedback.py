"""独立重算本次两代试跑的路线、筛选与 Critic 收益时间比。"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def critic_tags(response):
    names = ("quality_ref", "quality_ans", "speed_ref", "speed_ans")
    pattern = r"\s*" + r"\s*".join(rf"<{name}>\s*(.*?)\s*</{name}>" for name in names) + r"\s*"
    match = re.fullmatch(pattern, response, re.DOTALL)
    require(match is not None and all(value.strip() for value in match.groups()), "Critic 标签不完整。")
    return dict(zip(names, (value.strip() for value in match.groups())))


def prompt_metrics(prompt, letter):
    pattern = (rf"(?m)^Candidate {letter} objective score: {NUMBER}; raw task objective: {NUMBER}; "
               rf"Fitness: {NUMBER}; measured timing: mean complete solver wall time: {NUMBER} "
               rf"seconds per instance \([^\n]*?\); final-solution benefit: {NUMBER}; "
               rf"benefit per solve second: {NUMBER}\.\s*$")
    match = re.search(pattern, prompt)
    require(match is not None, f"Critic 提示缺少候选 {letter} 的原始目标、时间或收益时间比。")
    objective, raw, fitness, seconds, benefit, rate = (float(value) for value in match.groups())
    require(raw > 0 and seconds > 0 and same(objective, raw) and same(fitness, -raw), "目标或 Fitness 不符。")
    require(benefit == 1.0 / raw and rate == 1.0 / (raw * seconds), "收益时间比重算不符。")
    return {"average_objective": raw, "average_solve_wall_seconds": seconds, "benefit_per_second": rate}


def verify_prompt_sources(audit):
    """独立核对六次 Critic 输入数值与当时精英或分支状态的一致性。"""
    populations = {group["generation"]: {m["candidate_id"]: m for m in group["members"]}
                   for group in audit["population_replay"]}
    branches = {(row["generation"], row["role"], row["round"]): row for row in audit["branch_results"]}
    for critic in audit["critic_results"]:
        generation, stage = critic["generation"], critic["stage"]
        if stage == "critic_initial":
            pair = audit["elite_pairs"][str(generation)]
            a = populations[generation - 1][pair["a"]]
            b = populations[generation - 1][pair["b"]]
        else:
            previous_round = int(stage.split("_")[1]) - 1
            a = branches[generation, "solution_quality_agent", previous_round]
            b = branches[generation, "speed_agent", previous_round]
        for label, expected in (("candidate_a", a), ("candidate_b", b)):
            actual = critic[label]
            for key in ("average_objective", "average_solve_wall_seconds", "benefit_per_second"):
                require(same(actual[key], expected[key]),
                        f"第 {generation} 代 {stage} 的 {label} {key} 与候选记录不符。")
    return len(audit["critic_results"])


def main():
    config_path, events_path = RUN / "config.json", RUN / "events.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task = config["task"]
    require((task["initial_proposals"], task["population_size"], task["generations"],
             task["collaboration_rounds"], task["evaluation_mode"]) == (3, 2, 2, 2, "final_quality"),
            "试跑合同不符。")
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    require([e["event_index"] for e in events] == list(range(1, len(events) + 1)), "事件序号不连续。")
    require(events[-1]["event"] == "pilot_completed", "试跑没有完成记录。")
    require(any(e["event"] == "finish" and e["status"] == "completed" for e in events), "控制器未完成。")
    preflight = next(e for e in events if e["event"] == "preflight")
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
    evaluations = {e["candidate_id"]: e for e in events if e["event"] == "evaluation"}
    registrations = {e["candidate_id"]: e for e in events if e["event"] == "candidate_registered"}
    proposals = {e["candidate_id"]: e for e in events if e["event"] == "proposal_response"}
    invalid = {e["candidate_id"]: e["error"] for e in events if e["event"] == "candidate_invalid"}
    passed = {cid: event for cid, event in evaluations.items() if event["result"]["status"] == "passed"}
    code_hashes = {}
    max_error = 0.0
    for cid, event in evaluations.items():
        folder = RUN / "candidates" / cid
        result = event["result"]
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
        available = [cid for cid, e in passed.items() if e["event_index"] < at]
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

    initial = next(e for e in events if e["event"] == "initial_population")
    init_pool = [cid for cid, e in registrations.items() if e["stage"].startswith("init_")]
    groups = [{"generation": 0, "selected": replay(init_pool, initial["event_index"])}]
    require([cid for cid, _ in groups[0]["selected"]] == initial["candidate_ids"], "初始种群重放不符。")
    selections = [e for e in events if e["event"] == "selection"]
    require(len(selections) == 2, "未完成两代筛选。")
    for selection in selections:
        expected = replay(selection["pool"], selection["event_index"])
        actual = [(e["candidate_id"], e["evaluation_origin"]) for e in selection["selected"]]
        require(expected == actual, f"第 {selection['generation']} 代筛选重放不符。")
        eligible = [cid for cid, e in registrations.items() if e["generation"] == selection["generation"]
                    and (e["stage"].startswith(("eoh_", "mutation_")) or e["stage"] in (
                        "round_2_solution_quality_agent", "round_2_speed_agent", "fusion_2", "fusion_final"))]
        require(set(selection["pool"]) == set(eligible), "固定代末候选池不符。")
        groups.append({"generation": selection["generation"], "selected": actual})

    def details(cid, origin=None):
        result = passed[origin or cid]["result"]
        value, seconds = result["average"], result["average_solve_wall_seconds"]
        return {"candidate_id": cid, "evaluation_origin": origin or cid, "stage": registrations[cid]["stage"],
                "average_objective": value, "average_solve_wall_seconds": seconds,
                "benefit_per_second": 1.0 / (value * seconds)}

    populations = [{"generation": group["generation"],
                    "members": [details(cid, origin) for cid, origin in group["selected"]]} for group in groups]
    elite_pairs = {e["generation"]: e for e in events if e["event"] == "elite_pair"}
    transitions = [e for e in events if e["event"] == "branch_transition"]
    branch_results = []
    for generation in (1, 2):
        parent_origins = dict(groups[generation - 1]["selected"])
        for role in ("solution_quality_agent", "speed_agent"):
            steps = [e for e in transitions if e["generation"] == generation and e["role"] == role]
            require([step["round"] for step in steps] == [0, 1, 2], "分支轮次不完整。")
            previous = elite_pairs[generation]["b"]
            origin = parent_origins[previous]
            for step in steps:
                require(step["previous"] == previous and registrations[step["proposal"]]["parents"] == [previous],
                        "分支基底或续写父代不符。")
                if step["proposal"] in passed:
                    require(step["current"] == step["proposal"], "有效提案没有续写。")
                    origin = step["proposal"]
                else:
                    require(step["current"] == previous, "无效提案没有正确回退。")
                row = details(step["current"], origin)
                require(same(step["to_objective"], row["average_objective"]), "分支目标变化不符。")
                branch_results.append({"generation": generation, "role": role, "round": step["round"],
                                       "previous": previous, "proposal": step["proposal"], **row})
                previous = step["current"]

    critics = [e for e in events if e["event"] == "model_response"
               and e["stage"] in ("critic_initial", "critic_1", "critic_2")]
    require(len(critics) == 6, "Critic 比较数不符。")
    critic_results = []
    for event in critics:
        content, prompt = critic_tags(event["response"]), event["prompt"]
        a, b = prompt_metrics(prompt, "A"), prompt_metrics(prompt, "B")
        expected = "tie" if a["benefit_per_second"] == b["benefit_per_second"] else (
            "A" if a["benefit_per_second"] > b["benefit_per_second"] else "B")
        require(f"Benefit per solve second comparison: {expected}" in prompt
                and "shorter time alone is not a better speed result" in prompt
                and "does not prove convergence" in prompt, "Critic 的速度比较或停止边界不符。")
        for role, prefix in (("solution_quality_agent", "quality"), ("speed_agent", "speed")):
            stage = ("start_" if event["stage"] == "critic_initial" else
                     "round_" + event["stage"].split("_")[1] + "_") + role
            matches = [e for e in proposals.values() if e["generation"] == event["generation"]
                       and e["stage"] == stage]
            require(len(matches) == 1, "角色提案缺失或重复。")
            branch_prompt = matches[0]["prompt"]
            require(f"Critic's feedback: {content[prefix + '_ans']}" in branch_prompt
                    and f"Reflection: {content[prefix + '_ref']}" in branch_prompt, "角色反馈路由不符。")
            if role == "speed_agent":
                require("benefit per solve second:" in branch_prompt, "速度分支缺少自身收益时间比。")
        critic_results.append({"generation": event["generation"], "stage": event["stage"],
                               "candidate_a": a, "candidate_b": b, "rate_winner": expected, **content})

    attempts = [e for e in events if e["event"] == "model_attempt"]
    require(all(e["retry"] is False for e in attempts), "出现未经合同允许的自动重试。")
    usages = [e["usage"] for e in events if e["event"] == "model_usage"]
    totals = {}
    for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
        values = [entry.get(key) for entry in usages]
        totals[key] = sum(values) if all(type(value) is int for value in values) else None
    output = {
        "status": "passed", "scope": config["scope"], "source_events_sha256": digest(events_path),
        "source_config_sha256": digest(config_path), "verified_protected_inputs": checked,
        "dataset_sha256": digest(dataset), "model_calls": len(attempts),
        "model_process_failures": sum(e["result"].get("exit_code") != 0 for e in events
                                      if e["event"] == "model_process"),
        "evaluation_statuses": dict(Counter(e["result"]["status"] for e in evaluations.values())),
        "invalid_proposals": invalid, "candidate_evaluations": len(evaluations),
        "independently_recomputed_tours": len(passed) * 5, "maximum_route_absolute_error": max_error,
        "population_replay": populations,
        "elite_pairs": {str(g): {"a": e["a"], "b": e["b"]} for g, e in elite_pairs.items()},
        "branch_transitions_verified": len(branch_results), "branch_results": branch_results,
        "critic_output_routes_verified": len(critics) * 2, "critic_results": critic_results,
        "model_usage_records": len(usages), "model_usage_totals": totals,
        "process_tree_cpu_seconds": events[-1]["process_tree_cpu_seconds"],
        "elapsed_seconds": events[-1]["elapsed_seconds"],
    }
    with (RUN / "audit_speed_feedback.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"独立审计通过：{len(passed)} 个有效候选，{len(passed) * 5} 条路线，6 次 Critic 比较。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
