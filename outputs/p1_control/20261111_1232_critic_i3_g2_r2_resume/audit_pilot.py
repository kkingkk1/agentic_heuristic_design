"""串联原运行与断点续跑，重算路线、计时汇总和两代筛选。"""

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ORIGINAL = RUN_DIR.parent / "20261111_1217_critic_i3_g2_r2_02"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(first, second):
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tags(response):
    names = ("quality_ref", "quality_ans", "speed_ref", "speed_ans")
    pattern = r"\s*" + r"\s*".join(rf"<{name}>\s*(.*?)\s*</{name}>" for name in names) + r"\s*"
    match = re.fullmatch(pattern, response, re.DOTALL)
    require(match is not None and all(value.strip() for value in match.groups()), "Critic 输出格式无效。")
    return dict(zip(names, (value.strip() for value in match.groups())))


def main():
    config_path, events_path = RUN_DIR / "config.json", RUN_DIR / "events.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task = config["task"]
    require((task["initial_proposals"], task["population_size"], task["generations"],
             task["collaboration_rounds"]) == (3, 2, 2, 2), "运行规模不符。")
    original_path = ORIGINAL / "events.jsonl"
    original = [json.loads(line) for line in original_path.read_text(encoding="utf-8").splitlines()]
    continuation = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    for records in (original, continuation):
        require([event["event_index"] for event in records] == list(range(1, len(records) + 1)),
                "事件序号不连续。")
    require(original[-2]["event"] == "finish" and original[-2]["status"] == "stopped"
            and original[-1]["event"] == "pilot_stopped" and "模型连接失败" in original[-1]["error"],
            "原运行的停止记录不符。")
    require(continuation[-1]["event"] == "pilot_completed", "续跑没有完成记录。")
    resume_preflight = next(event for event in continuation if event["event"] == "resume_preflight")
    require(resume_preflight["source_events_sha256"] == sha256(original_path), "原事件记录已变化。")
    events = [{**event, "source_event_index": event["event_index"], "event_index": index}
              for index, event in enumerate(original[:-2] + continuation, 1)]
    preflight = next(event for event in events if event["event"] == "preflight")
    verified_sources = 0
    for snapshot in (preflight, resume_preflight):
        for filename, expected_hash in snapshot["code_sha256"].items():
            source = Path(filename).resolve()
            if source.is_relative_to(RUN_DIR) or source.is_relative_to(ORIGINAL):
                preserved = source
            elif source.is_relative_to(ROOT) and source.suffix in (".py", ".json", ".yaml", ".txt"):
                preserved = ORIGINAL / "source_snapshots" / source.relative_to(ROOT)
            else:
                preserved = source
            require(preserved.is_file() and sha256(preserved) == expected_hash,
                    f"运行输入或源码快照哈希不符：{source}")
            verified_sources += 1
    attempts = [event for event in events if event["event"] == "model_attempt"]
    evaluations = {event["candidate_id"]: event for event in events if event["event"] == "evaluation"}
    registration = {event["candidate_id"]: event for event in events if event["event"] == "candidate_registered"}
    proposals = {event["candidate_id"]: event for event in events if event["event"] == "proposal_response"}
    require(len(attempts) == 62 and all(event["retry"] is False for event in attempts),
            "模型调用数或重试情况与合同不符。")
    failures = [event for event in events if event["event"] == "model_process"
                and event["result"].get("exit_code") != 0]
    require(len(failures) == 1 and failures[0]["stage"] == "fusion_final"
            and "Connection failed" in failures[0]["result"].get("stdout", ""), "失败调用记录不符。")
    failed_attempt = next(event for event in reversed(original) if event["event"] == "model_attempt")
    resumed_attempt = next(event for event in continuation if event["event"] == "model_attempt")
    failed_registration = next(event for event in original if event["event"] == "candidate_registered"
                               and event["candidate_id"] == "c0043")
    resumed_registration = next(event for event in continuation if event["event"] == "candidate_registered")
    require(failed_attempt["stage"] == resumed_attempt["stage"] == "fusion_final"
            and failed_attempt["prompt"] == resumed_attempt["prompt"]
            and failed_registration["parents"] == resumed_registration["parents"]
            and resumed_registration["candidate_id"] == "c0044", "断点融合提示或父代不一致。")
    require(len(evaluations) == 49 and len(registration) == 50 and len(proposals) == 49,
            "候选生成或评价数量不符。")
    require(all(event["result"]["status"] == "passed" for event in evaluations.values()),
            "存在未通过的候选评价。")
    dataset = ROOT / task["dataset"]
    require(sha256(dataset) == task["dataset_sha256"], "训练集哈希不符。")
    instances = np.load(dataset, allow_pickle=False)
    require(instances.shape == (5, 50, 2), "训练集形状不符。")
    code_hashes = {}
    max_route_error = 0.0
    for candidate_id, evaluation in evaluations.items():
        root = (ORIGINAL if (ORIGINAL / "candidates" / candidate_id).is_dir() else RUN_DIR) / "candidates" / candidate_id
        result = evaluation["result"]
        require(json.loads((root / "result.json").read_text(encoding="utf-8")) == result,
                f"{candidate_id} 结果文件与事件不一致。")
        code = (root / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8")
        require(code.endswith("\n"), f"{candidate_id} 缺少候选代码。")
        code_hashes[candidate_id] = hashlib.sha256(code[:-1].encode("utf-8")).hexdigest()
        rows = [json.loads(line[len("P0_JSON "):])
                for line in (root / "stdout.txt").read_text(encoding="utf-8").splitlines()
                if line.startswith("P0_JSON ")]
        require([row.get("event") for row in rows] == ["start"] + ["instance"] * 5 + ["complete"],
                f"{candidate_id} 的逐例事件不完整。")
        require(rows[0].get("time_scope") == "solve_wall_seconds_including_preparation_and_validation",
                f"{candidate_id} 的计时范围无效。")
        objectives, times = [], []
        for index, (row, points) in enumerate(zip(rows[1:-1], instances)):
            tour = row["tour"]
            require(row["index"] == index and row["seed"] == task["evaluation_seeds"][index]
                    and row["iterations"] == 200 and sorted(tour) == list(range(50)),
                    f"{candidate_id} 实例 {index} 的索引、种子、迭代或路线无效。")
            expected = math.fsum(
                math.hypot(float(points[a, 0] - points[b, 0]), float(points[a, 1] - points[b, 1]))
                for a, b in zip(tour, tour[1:] + tour[:1])
            )
            error = abs(expected - row["objective"])
            max_route_error = max(max_route_error, error)
            require(same(expected, row["objective"]), f"{candidate_id} 实例 {index} 的目标重算不符。")
            seconds = row.get("solve_wall_seconds")
            require(isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
                    and math.isfinite(seconds) and seconds > 0, f"{candidate_id} 实例 {index} 的时间无效。")
            objectives.append(row["objective"])
            times.append(seconds)
        average = math.fsum(objectives) / 5
        average_time = math.fsum(times) / 5
        require(same(average, rows[-1]["average"]) and same(average, result["average"])
                and same(result["fitness"], -average), f"{candidate_id} 的平均目标或 Fitness 不符。")
        require(same(average_time, rows[-1]["average_solve_wall_seconds"])
                and same(average_time, result["average_solve_wall_seconds"])
                and times == result["solve_wall_seconds"], f"{candidate_id} 的时间汇总不符。")

    def replay(pool, at_index):
        available = [cid for cid, event in evaluations.items() if event["event_index"] < at_index]
        earliest = {}
        for cid in sorted(available, key=lambda item: evaluations[item]["result"]["completed_monotonic_ns"]):
            earliest.setdefault(code_hashes[cid], cid)
        unique = {}
        for cid in sorted(pool, key=lambda item: proposals[item]["generation_order"]):
            if cid in evaluations and evaluations[cid]["event_index"] < at_index:
                unique.setdefault(code_hashes[cid], cid)
        ranked = sorted(unique.values(), key=lambda cid: (
            -evaluations[earliest[code_hashes[cid]]]["result"]["fitness"],
            proposals[cid]["generation_order"],
        ))
        return [(cid, earliest[code_hashes[cid]]) for cid in ranked[:2]]

    initial = next(event for event in events if event["event"] == "initial_population")
    initial_pool = [cid for cid, event in registration.items() if event["stage"].startswith("init_")]
    initial_selected = replay(initial_pool, initial["event_index"])
    require([cid for cid, _origin in initial_selected] == initial["candidate_ids"], "初始种群筛选不符。")
    selections = [event for event in events if event["event"] == "selection"]
    require(len(selections) == 2, "完整代数不符。")
    populations = [{"generation": 0, "selected": initial_selected}]
    for selection in selections:
        expected = replay(selection["pool"], selection["event_index"])
        recorded = [(entry["candidate_id"], entry["evaluation_origin"]) for entry in selection["selected"]]
        require(expected == recorded, f"第 {selection['generation']} 代筛选重放不符。")
        expected_ids = [cid for cid, event in registration.items()
                        if cid != "c0043" and event["generation"] == selection["generation"]
                        and (event["stage"].startswith(("eoh_", "mutation_"))
                             or event["stage"] in ("round_2_solution_quality_agent",
                                                   "round_2_speed_agent", "fusion_2", "fusion_final"))]
        require(set(selection["pool"]) == set(expected_ids), f"第 {selection['generation']} 代候选池不符。")
        populations.append({"generation": selection["generation"], "selected": recorded})

    transitions = [event for event in events if event["event"] == "branch_transition"]
    elite_pairs = {event["generation"]: event for event in events if event["event"] == "elite_pair"}
    branch_results = []
    for generation in (1, 2):
        for role in ("solution_quality_agent", "speed_agent"):
            steps = [event for event in transitions if event["generation"] == generation and event["role"] == role]
            require([step["round"] for step in steps] == [0, 1, 2], "分支轮数不符。")
            previous = elite_pairs[generation]["b"]
            for step in steps:
                require(step["previous"] == previous
                        and registration[step["proposal"]]["parents"] == [previous],
                        f"第 {generation} 代 {role} 第 {step['round']} 轮继承不符。")
                require(step["current"] == step["proposal"]
                        and same(step["to_objective"], evaluations[step["proposal"]]["result"]["average"]),
                        "有效提案未按合同续写。")
                branch_results.append({"generation": generation, "role": role, "round": step["round"],
                                       "previous": previous, "candidate_id": step["current"],
                                       "average_objective": step["to_objective"],
                                       "average_solve_wall_seconds":
                                           evaluations[step["current"]]["result"]["average_solve_wall_seconds"]})
                previous = step["current"]

    critics = [event for event in events if event["event"] == "model_response"
               and event["stage"] in ("critic_initial", "critic_1", "critic_2")]
    require(len(critics) == 6, "Critic 调用数不符。")
    for critic in critics:
        response = tags(critic["response"])
        for role, prefix in (("solution_quality_agent", "quality"), ("speed_agent", "speed")):
            other = "speed" if prefix == "quality" else "quality"
            stage = ("start_" if critic["stage"] == "critic_initial"
                     else "round_" + critic["stage"].split("_")[1] + "_") + role
            matches = [event for event in proposals.values()
                       if event["generation"] == critic["generation"] and event["stage"] == stage]
            require(len(matches) == 1, "角色提案缺失或重复。")
            prompt = matches[0]["prompt"]
            require(response[prefix + "_ref"] in prompt and response[prefix + "_ans"] in prompt,
                    f"第 {critic['generation']} 代 {stage} 缺少本角色反馈。")
            require(response[other + "_ref"] not in prompt and response[other + "_ans"] not in prompt,
                    f"第 {critic['generation']} 代 {stage} 混入另一角色反馈。")

    histories = {(event["generation"], event["role"]): event["reflection"] for event in events
                 if event["event"] == "history_appended"}
    require(len(histories) == 6, "两代角色历史不完整。")
    for elite, parent in (("A", elite_pairs[2]["a"]), ("B", elite_pairs[2]["b"])):
        for role in ("solution_quality_agent", "speed_agent", "integrator"):
            stage = f"mutation_{elite}_{role}"
            matches = [event for event in proposals.values() if event["generation"] == 2 and event["stage"] == stage]
            require(len(matches) == 1 and registration[matches[0]["candidate_id"]]["parents"] == [parent],
                    f"{stage} 的父代不符。")
            prompt = matches[0]["prompt"]
            require(histories[(1, role)] in prompt and histories[(2, role)] in prompt,
                    f"{stage} 缺少此前历史或本代反思。")

    usage = [event["usage"] for event in events if event["event"] == "model_usage"]
    usage_totals = {}
    for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
        values = [item.get(key) for item in usage]
        usage_totals[key] = sum(values) if all(type(value) is int for value in values) else None
    output = {
        "status": "passed", "scope": config["scope"],
        "source_events_sha256": sha256(events_path), "original_events_sha256": sha256(original_path),
        "source_config_sha256": sha256(config_path),
        "verified_protected_inputs": verified_sources,
        "dataset_sha256": sha256(dataset), "model_calls": len(attempts), "failed_model_calls": len(failures),
        "candidate_evaluations": len(evaluations), "evaluation_statuses": {"passed": len(evaluations)},
        "independently_recomputed_tours": len(evaluations) * 5,
        "maximum_route_absolute_error": max_route_error,
        "population_replay": [
            {"generation": group["generation"], "members": [
                {"candidate_id": cid, "evaluation_origin": origin, "stage": registration[cid]["stage"],
                 "average_objective": evaluations[origin]["result"]["average"],
                 "fitness": evaluations[origin]["result"]["fitness"]}
                for cid, origin in group["selected"]]}
            for group in populations],
        "branch_transitions_verified": len(branch_results), "branch_results": branch_results,
        "resumed_mutation_histories_verified": 6,
        "critic_output_routes_verified": len(critics) * 2,
        "model_usage_records": len(usage), "model_usage_totals": usage_totals,
        "process_tree_cpu_seconds": events[-1]["process_tree_cpu_seconds"],
        "original_elapsed_seconds": original[-1]["elapsed_seconds"],
        "continuation_elapsed_seconds": events[-1]["elapsed_seconds"],
    }
    destination = RUN_DIR / "independent_audit.json"
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Independent audit passed: {len(evaluations)} candidates, {len(evaluations) * 5} tours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
