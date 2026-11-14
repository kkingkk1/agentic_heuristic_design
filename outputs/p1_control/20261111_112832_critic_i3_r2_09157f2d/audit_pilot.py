"""复核本次部分完成的 Critic 试跑，不调用模型或重新求解。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import p1_token_usage  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(first, second):
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)


def tagged(response):
    names = ("quality_ref", "quality_ans", "speed_ref", "speed_ans")
    pattern = r"\s*" + r"\s*".join(fr"<{name}>(.*?)</{name}>" for name in names) + r"\s*"
    match = re.fullmatch(pattern, response, flags=re.DOTALL)
    require(match is not None, "Critic 的四项输出标签不完整或顺序错误。")
    values = {name: value.strip() for name, value in zip(names, match.groups())}
    require(all(values.values()), "Critic 的反馈包含空字段。")
    for name in ("quality_ans", "speed_ans"):
        count = len(re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", values[name], flags=re.UNICODE))
        require(count <= 40, f"{name} 超出 40 词限制。")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="重新核验，不覆盖已有审计记录")
    args = parser.parse_args()
    config_path, events_path = RUN_DIR / "config.json", RUN_DIR / "events.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task = config["task"]
    require((task["initial_proposals"], task["population_size"], task["generations"],
             task["collaboration_rounds"], task["evaluation_mode"]) == (3, 2, 1, 2, "final_quality"),
            "运行规模或评价设置不符。")
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require([event["event_index"] for event in events] == list(range(1, len(events) + 1)), "事件序号不连续。")
    attempts = [event for event in events if event["event"] == "model_attempt"]
    evaluations = {event["candidate_id"]: event for event in events if event["event"] == "evaluation"}
    registrations = {event["candidate_id"]: event for event in events if event["event"] == "candidate_registered"}
    proposals = {event["candidate_id"]: event for event in events if event["event"] == "proposal_response"}
    require(len(attempts) == 25 and all(event["retry"] is False for event in attempts), "模型调用数或重试记录不符。")
    require(len(evaluations) == len(registrations) == len(proposals) == 20, "候选记录数量不符。")
    require(all(event["result"]["status"] == "passed" for event in evaluations.values()), "存在未通过的候选评价。")
    stopped = [event for event in events if event["event"] == "pilot_stopped"]
    require(len(stopped) == 1 and "长期反思须少于 50 个词" in stopped[0]["error"], "停止原因不符。")
    require(not any(event["event"] in ("selection", "pilot_completed") for event in events),
            "不完整代不能视为已完成筛选。")

    preflight = next(event for event in events if event["event"] == "preflight")
    snapshots = {
        (ROOT / "scripts/p0_runtime.py").resolve(): RUN_DIR / "source_snapshots/p0_runtime.py",
        (ROOT / "scripts/p1_run_generation.py").resolve(): RUN_DIR / "source_snapshots/p1_run_generation.py",
    }
    mismatched, used_snapshots = [], []
    for filename, expected in preflight["code_sha256"].items():
        path = Path(filename)
        if path.is_file() and digest(path) == expected:
            continue
        snapshot = snapshots.get(path.resolve())
        if snapshot and snapshot.is_file() and digest(snapshot) == expected:
            used_snapshots.append(snapshot.relative_to(RUN_DIR).as_posix())
        else:
            mismatched.append(filename)
    require(not mismatched, f"审计时受保护文件已变化：{mismatched}")
    dataset = ROOT / task["dataset"]
    require(digest(dataset) == task["dataset_sha256"], "训练集哈希不符。")
    instances = np.load(dataset, allow_pickle=False)
    require(instances.shape == (5, 50, 2), "训练集形状不符。")
    code_hashes, max_error, total_cpu = {}, 0.0, 0.0
    for candidate_id, event in evaluations.items():
        directory = RUN_DIR / "candidates" / candidate_id
        result = event["result"]
        require(json.loads((directory / "result.json").read_text(encoding="utf-8")) == result,
                f"{candidate_id} 的结果文件与事件不一致。")
        code_hashes[candidate_id] = digest(directory / "problems/tsp_aco/gpt.py")
        rows = [json.loads(line[len("P0_JSON "):])
                for line in (directory / "stdout.txt").read_text(encoding="utf-8").splitlines()
                if line.startswith("P0_JSON ")]
        require([row.get("event") for row in rows] == ["start"] + ["instance"] * 5 + ["complete"],
                f"{candidate_id} 的逐例记录不完整。")
        require(rows[0].get("time_scope") == "solve_wall_seconds_including_preparation_and_validation",
                f"{candidate_id} 的求解计时范围无效。")
        values, times = [], []
        for index, (row, points) in enumerate(zip(rows[1:-1], instances)):
            tour = row["tour"]
            require(row["index"] == index and row["seed"] == task["evaluation_seeds"][index]
                    and row["iterations"] == 200 and sorted(tour) == list(range(50)),
                    f"{candidate_id} 第 {index} 例的种子、迭代或路线无效。")
            expected = math.fsum(math.hypot(float(points[a, 0] - points[b, 0]),
                                            float(points[a, 1] - points[b, 1]))
                                 for a, b in zip(tour, tour[1:] + tour[:1]))
            max_error = max(max_error, abs(expected - row["objective"]))
            require(close(expected, row["objective"]), f"{candidate_id} 第 {index} 例目标重算不符。")
            seconds = row["solve_wall_seconds"]
            require(isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
                    and math.isfinite(seconds) and seconds > 0, f"{candidate_id} 第 {index} 例计时无效。")
            values.append(row["objective"])
            times.append(seconds)
        average, average_time = math.fsum(values) / 5, math.fsum(times) / 5
        require(close(average, rows[-1]["average"]) and close(average, result["average"])
                and close(-average, result["fitness"]), f"{candidate_id} 目标汇总不符。")
        require(close(average_time, rows[-1]["average_solve_wall_seconds"])
                and close(average_time, result["average_solve_wall_seconds"])
                and times == result["solve_wall_seconds"], f"{candidate_id} 计时汇总不符。")
        total_cpu += result["process_tree_cpu"]["cpu_seconds"]

    initial = next(event for event in events if event["event"] == "initial_population")
    pool = [candidate_id for candidate_id, event in registrations.items() if event["stage"].startswith("init_")]
    earliest = {}
    for candidate_id in sorted(pool, key=lambda item: evaluations[item]["result"]["completed_monotonic_ns"]):
        earliest.setdefault(code_hashes[candidate_id], candidate_id)
    unique = {}
    for candidate_id in sorted(pool, key=lambda item: proposals[item]["generation_order"]):
        unique.setdefault(code_hashes[candidate_id], candidate_id)
    ranked = sorted(unique.values(), key=lambda item: (
        -evaluations[earliest[code_hashes[item]]]["result"]["fitness"], proposals[item]["generation_order"]))
    require(initial["candidate_ids"] == ranked[:2], "初始精英无法按原规则重放。")

    elite = next(event for event in events if event["event"] == "elite_pair")
    transitions = {(event["round"], event["role"]): event for event in events
                   if event["event"] == "branch_transition"}
    require(len(transitions) == 6, "两条分支的启动及两轮续写记录不完整。")
    inputs = {
        "critic_initial": (elite["a"], elite["b"]),
        "critic_1": (transitions[(0, "solution_quality_agent")]["current"],
                     transitions[(0, "speed_agent")]["current"]),
        "critic_2": (transitions[(1, "solution_quality_agent")]["current"],
                     transitions[(1, "speed_agent")]["current"]),
    }
    responses = {event["stage"]: event for event in events
                 if event["event"] == "model_response" and event["stage"] in inputs}
    require(set(responses) == set(inputs), "三次 Critic 响应不完整。")
    critics = []
    for stage, (first, second) in inputs.items():
        response = responses[stage]
        values = tagged(response["response"])
        origins = (earliest[code_hashes[first]], earliest[code_hashes[second]]) if stage == "critic_initial" else (first, second)
        scores = [evaluations[origin]["result"] for origin in origins]
        for label, result in zip(("A", "B"), scores):
            expected = (f"Candidate {label} objective score: {result['average']}; raw task objective: {result['average']}; "
                        f"Fitness: {result['fitness']}; measured timing: mean complete solver wall time: "
                        f"{result['average_solve_wall_seconds']} seconds per instance")
            require(expected in response["prompt"], f"{stage} 的候选 {label} 输入与评价记录不一致。")
        winner = lambda a, b, lower: "tie" if a == b else ("A" if (a < b) == lower else "B")
        expected_comparisons = {
            "fitness": winner(scores[0]["fitness"], scores[1]["fitness"], False),
            "raw_objective": winner(scores[0]["average"], scores[1]["average"], True),
            "solve_time": winner(scores[0]["average_solve_wall_seconds"], scores[1]["average_solve_wall_seconds"], True),
        }
        for label, key in (("Fitness comparison", "fitness"), ("Raw objective comparison", "raw_objective"),
                           ("Observed solve time comparison", "solve_time")):
            require(f"{label}: {expected_comparisons[key]} " in response["prompt"], f"{stage} 的比较方向不符。")
        prefix = "start_" if stage == "critic_initial" else "round_" + stage.split("_")[1] + "_"
        for role, own, other in (("solution_quality_agent", "quality", "speed"),
                                 ("speed_agent", "speed", "quality")):
            next_attempt = next(event for event in attempts if event["stage"] == prefix + role)
            require(values[own + "_ref"] in next_attempt["prompt"] and values[own + "_ans"] in next_attempt["prompt"],
                    f"{stage} 的 {role} 未收到本角色反馈。")
            require(values[other + "_ref"] not in next_attempt["prompt"] and values[other + "_ans"] not in next_attempt["prompt"],
                    f"{stage} 的 {role} 混入另一角色反馈。")
        critics.append({"stage": stage, "candidate_a": first, "candidate_b": second,
                        "evaluation_origin_a": origins[0], "evaluation_origin_b": origins[1],
                        "objective_a": scores[0]["average"], "objective_b": scores[1]["average"],
                        "solve_seconds_a": scores[0]["average_solve_wall_seconds"],
                        "solve_seconds_b": scores[1]["average_solve_wall_seconds"],
                        "comparisons": expected_comparisons, "feedback": values})

    reflection = next(event["response"] for event in events
                      if event["event"] == "model_response" and event["stage"] == "reflection_speed_agent")
    words = len(re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", re.sub(r"</?ans>", "", reflection), flags=re.UNICODE))
    require(words >= 50, "速度角色反思未复现词数超限。")
    result = {
        "status": "passed_for_partial_run", "run_status": "stopped_before_selection",
        "config_sha256": digest(config_path), "events_sha256": digest(events_path),
        "protected_files_match_before_runtime_changes": not used_snapshots,
        "protected_files_verified": not mismatched, "source_snapshots_used": used_snapshots,
        "model_calls": len(attempts), "candidate_evaluations": len(evaluations),
        "validated_tours": 5 * len(evaluations), "max_route_absolute_error": max_error,
        "critic_comparisons_verified": len(critics), "role_feedback_routes_verified": 2 * len(critics),
        "critics": critics, "speed_reflection_word_count": words,
        "stop_reason": stopped[0]["error"], "elapsed_seconds": stopped[0]["elapsed_seconds"],
        "evaluation_process_tree_cpu_seconds": total_cpu,
        "usage": p1_token_usage.summarize([events]),
    }
    if not args.verify_only:
        with (RUN_DIR / "audit.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(f"部分运行审计通过：{len(critics)} 次 Critic 比较，{5 * len(evaluations)} 条路线。")


if __name__ == "__main__":
    main()
