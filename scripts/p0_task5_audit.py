"""独立重放 P0 真实运行的事件、预算、候选池和 TSP 路径。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "p0_control"
EVENT_PREFIX = "P0_JSON "


def require(condition, message):
    if not condition:
        raise ValueError(message)


def proposal_parts(text):
    match = re.fullmatch(r"\s*<thought>\s*(.*?)\s*</thought>\s*<code>\s*(.*?)\s*</code>\s*",
                         text, flags=re.DOTALL)
    if not match or not all(part.strip() for part in match.groups()):
        return None
    return tuple(part.strip() for part in match.groups())


def code_digest(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def close(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def audit_evaluation(run_dir, candidate_id, record, data):
    root = run_dir / "candidates" / candidate_id
    saved = json.loads((root / "result.json").read_text(encoding="utf-8"))
    require(saved == record, f"{candidate_id} 的评价事件与结果文件不一致。")
    stdout_path, stderr_path = root / "stdout.txt", root / "stderr.txt"
    require(stdout_path.is_file() and stderr_path.is_file(), f"{candidate_id} 缺少 stdout/stderr。")
    accounting = record.get("process_tree_cpu", {})
    require(accounting.get("active_processes") == 0, f"{candidate_id} 的进程树未确认退出。")
    require(isinstance(accounting.get("cpu_seconds"), (int, float)) and accounting["cpu_seconds"] >= 0,
            f"{candidate_id} 缺少有效进程树 CPU。")
    if record.get("status") != "passed":
        require("fitness" not in record and "average" not in record,
                f"{candidate_id} 的失败评价包含正常分数。")
        return 0
    require(record.get("exit_code") == 0, f"{candidate_id} 的成功评价退出码非零。")
    events = [json.loads(line[len(EVENT_PREFIX):]) for line in stdout_path.read_text(encoding="utf-8").splitlines()
              if line.startswith(EVENT_PREFIX)]
    require([event.get("event") for event in events] == ["start"] + ["instance"] * 5 + ["complete"],
            f"{candidate_id} 的实例输出不完整。")
    start, complete = events[0], events[-1]
    require(start.get("n_ants") == 30 and start.get("n_iterations") == 200
            and start.get("seed_base") == 20261103 and complete.get("instance_count") == 5,
            f"{candidate_id} 的固定评价参数不匹配。")
    objectives = []
    for index, instance in enumerate(events[1:-1]):
        tour = instance.get("tour")
        require(instance.get("index") == index and instance.get("seed") == 20261103 + index
                and instance.get("iterations") == 200 and isinstance(tour, list)
                and len(tour) == 50 and all(type(city) is int for city in tour)
                and sorted(tour) == list(range(50)), f"{candidate_id} 的实例 {index} 路径无效。")
        distances = [math.hypot(float(data[index][tour[i]][0] - data[index][tour[(i + 1) % 50]][0]),
                                float(data[index][tour[i]][1] - data[index][tour[(i + 1) % 50]][1]))
                     for i in range(50)]
        objective = math.fsum(distances)
        require(close(objective, instance.get("objective")), f"{candidate_id} 的实例 {index} 目标重算不符。")
        objectives.append(float(instance["objective"]))
    average = math.fsum(objectives) / 5
    require(close(average, complete.get("average")) and close(average, record.get("average"))
            and close(-average, record.get("fitness"))
            and all(close(a, b) for a, b in zip(objectives, record.get("objectives", [])))
            and len(record.get("objectives", [])) == 5, f"{candidate_id} 的汇总分数不一致。")
    return 5


def audit(run_dir):
    rows = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    require(rows and [row.get("event_index") for row in rows] == list(range(1, len(rows) + 1)),
            "事件序号缺失或不连续。")
    counts = Counter(row["event"] for row in rows)
    preflight = next(row for row in rows if row["event"] == "preflight")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    require(config == json.loads((ROOT / "cfg/p0_tsp_aco.json").read_text(encoding="utf-8")),
            "运行配置与冻结文件不一致。")
    snapshot = run_dir / "source_snapshot"
    for name, expected in preflight["code_sha256"].items():
        original = Path(name).resolve()
        require(original.is_relative_to(ROOT.resolve()), f"源码路径超出项目范围：{name}")
        source = snapshot / original.relative_to(ROOT.resolve()) if snapshot.is_dir() else original
        require(hashlib.sha256(source.read_bytes()).hexdigest() == expected, f"执行源码或快照哈希不符：{name}")
    dataset_path = ROOT / config["task"]["dataset"]
    require(hashlib.sha256(dataset_path.read_bytes()).hexdigest() == config["task"]["dataset_sha256"],
            "数据集哈希与冻结配置不一致。")
    data = np.load(dataset_path, allow_pickle=False)
    require(data.shape == (5, 50, 2) and np.isfinite(data).all(), "训练集形状或数值无效。")

    registered = [row for row in rows if row["event"] == "candidate_registered"]
    by_id = {row["candidate_id"]: row for row in registered}
    by_stage = {row["stage"]: row for row in registered}
    require(len(by_id) == len(registered) == len(by_stage), "候选 ID 或槽位重复。")
    require([row["sequence"] for row in registered] == list(range(1, len(registered) + 1)),
            "候选生成顺序不连续。")
    responses = {row["candidate_id"]: row for row in rows if row["event"] == "proposal_response"}
    invalid = {row["candidate_id"] for row in rows if row["event"] == "candidate_invalid"}
    evaluations = {row["candidate_id"]: row["result"] for row in rows if row["event"] == "evaluation"}
    require(len(evaluations) == counts["evaluation"] and len(responses) == counts["proposal_response"],
            "同一候选出现重复响应或评价。")
    recorded_generation_count = sum("generation_order" in row for row in responses.values())
    require(recorded_generation_count in (0, len(responses)), "实际生成顺序记录不完整。")
    has_generation_order = recorded_generation_count > 0
    if has_generation_order:
        response_order = sorted(responses.values(), key=lambda row: row["event_index"])
        require([row["generation_order"] for row in response_order] == list(range(1, len(response_order) + 1)),
                "实际生成顺序与响应事件顺序不一致。")
    require(set(evaluations).issubset(responses) and invalid.isdisjoint(evaluations),
            "候选响应、非法记录与评价关联异常。")
    attempts = [row for row in rows if row["event"] == "model_attempt"]
    normal = sum(not row.get("retry") for row in attempts)
    retries = len(attempts) - normal
    require(normal <= 95 and retries <= 20 and len(attempts) <= 115,
            "模型调用超出冻结预算。")
    require(counts["evaluation_budget_reserved"] <= 98, "评价启动次数超过预算。")
    usage_rows = [row.get("usage") for row in rows if row["event"] == "model_usage"]
    usage_fields = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
    observed_usage = {field: sum(value[field] for value in usage_rows
                                 if isinstance(value, dict) and isinstance(value.get(field), int)
                                 and not isinstance(value.get(field), bool)) for field in usage_fields}
    missing_usage = {field: sum(not isinstance(value, dict) or not isinstance(value.get(field), int)
                                or isinstance(value.get(field), bool) for value in usage_rows)
                     for field in usage_fields}
    for attempt in attempts:
        command = attempt["command"]
        required_options = {'model_provider="p0_chatgpt_https"',
                            'model_providers.p0_chatgpt_https.base_url="https://chatgpt.com/backend-api/codex"',
                            "model_providers.p0_chatgpt_https.requires_openai_auth=true",
                            "model_providers.p0_chatgpt_https.supports_websockets=false",
                            "model_providers.p0_chatgpt_https.request_max_retries=0",
                            "model_providers.p0_chatgpt_https.stream_max_retries=0",
                            "unbounded_connection_retries"}
        require("--model" in command and command[command.index("--model") + 1] == "gpt-5.6-luna"
                and "--ignore-user-config" in command and "--ephemeral" in command
                and "--sandbox" in command and command[command.index("--sandbox") + 1] == "read-only"
                and "--disable" in command and required_options.issubset(set(command))
                and not any("reasoning_effort" in str(item) for item in command),
                f"{attempt['stage']} 的模型调用不符合冻结合同。")

    paths_recomputed = 0
    valid = {}
    cpu_total = 0.0
    for candidate_id, record in evaluations.items():
        parts = proposal_parts(responses[candidate_id]["response"])
        require(parts is not None, f"{candidate_id} 有评价但响应格式不可解析。")
        code = (run_dir / "candidates" / candidate_id / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8")
        require(code == parts[1] + "\n", f"{candidate_id} 的评价代码与原响应不一致。")
        paths_recomputed += audit_evaluation(run_dir, candidate_id, record, data)
        cpu_total += record["process_tree_cpu"]["cpu_seconds"]
        if record.get("status") == "passed":
            valid[candidate_id] = {"code": parts[1], "thought": parts[0], "record": record,
                                   "sequence": by_id[candidate_id]["sequence"]}
    require(cpu_total <= 6000, f"评价进程树 CPU 已超预算：{cpu_total} 秒。")

    finish = [row for row in rows if row["event"] == "finish"]
    completed = len(finish) == 1 and finish[0].get("status") == "completed"
    report = {"status": "incomplete", "events": len(rows), "event_counts": dict(counts),
              "model_attempts": len(attempts), "normal_model_calls": normal, "model_retries": retries,
              "candidate_slots": len(registered), "evaluations": len(evaluations),
              "valid_evaluations": len(valid), "recomputed_tours": paths_recomputed,
              "evaluation_process_tree_cpu_seconds": cpu_total,
              "observed_model_usage": observed_usage, "missing_model_usage_fields": missing_usage,
              "failed_attempt_usage": "不可完整取得，不按零估算。",
              "source_verification": "source_snapshot" if snapshot.is_dir() else "current_workspace",
              "generation_order_recorded": has_generation_order,
              "finish": finish[-1] if finish else None,
              "quota_acceptance": preflight.get("unquantified_quota_accepted_for_this_run")}
    if not completed:
        return report

    require(counts["model_process"] == len(attempts), "已完成运行存在未记录的模型进程结果。")
    require(counts["evaluation_budget_reserved"] == len(evaluations) == counts["evaluation_cpu_final"],
            "已完成运行存在未闭合的评价记录。")
    require(sum(stage.startswith("init_") for stage in by_stage) == 30, "初始化提案不是 30 个。")
    require(all(identifier in invalid or identifier in evaluations for identifier in responses),
            "已完成运行存在没有评价或错误结论的提案。")

    earliest = {}
    for candidate_id, item in sorted(valid.items(), key=lambda pair: pair[1]["record"]["completed_monotonic_ns"]):
        earliest.setdefault(code_digest(item["code"]), candidate_id)

    def rank(ids, *, actual_generation_order=False):
        if actual_generation_order:
            order = lambda item: responses[item]["event_index"]
        elif has_generation_order:
            order = lambda item: responses[item]["generation_order"]
        else:
            order = lambda item: valid[item]["sequence"]
        unique = {}
        for identifier in sorted((item for item in ids if item in valid), key=order):
            digest = code_digest(valid[identifier]["code"])
            unique.setdefault(digest, identifier)
        return sorted(unique.values(), key=lambda identifier: (
            valid[earliest[code_digest(valid[identifier]["code"])]]["record"]["average"],
            order(identifier)))

    init = [row["candidate_id"] for row in registered if row["stage"].startswith("init_")]
    original = rank(init)[:10]
    population = next(row for row in rows if row["event"] == "initial_population")
    require(len(original) == 10 and population["candidate_ids"] == original, "初始化排序或去重不符。")
    require(original == rank(init, actual_generation_order=True)[:10],
            "初始化按登记顺序与实际响应生成顺序排序得到不同结果。")
    require(all(f"eoh_{op}_{i:02d}" in by_stage for op in ("E1", "E2", "M1", "M2") for i in range(1, 11)),
            "EoH 槽位不完整。")
    rng = random.Random(preflight["selection_seed"])
    for op in ("E1", "E2", "M1", "M2"):
        for number in range(1, 11):
            pool, chosen = list(original), []
            for _ in range(5 if op in ("E1", "E2") else 1):
                parent = rng.choices(pool, weights=[1 / (original.index(item) + 1 + len(original))
                                                    for item in pool], k=1)[0]
                pool.remove(parent)
                chosen.append(parent)
            require(by_stage[f"eoh_{op}_{number:02d}"]["parents"] == chosen, f"{op} 的父代重放不符。")
    first = rng.choices(range(9), weights=[(i + 1) ** -3 for i in range(9)], k=1)[0]
    second = rng.choice([i for i in (first + 1, first + 2) if i < 10])
    pair = next(row for row in rows if row["event"] == "elite_pair")
    require((pair["a"], pair["b"]) == (original[first], original[second]), "精英 A/B 重放不符。")

    transitions = {(row["role"], row["round"]): row for row in rows if row["event"] == "branch_transition"}
    for role in ("Q", "E"):
        current = pair["b"]
        for turn in range(4):
            stage = f"start_{role}" if turn == 0 else f"round_{turn}_{role}"
            proposal = by_stage[stage]["candidate_id"]
            transition = transitions[(role, turn)]
            require(transition["previous"] == current and transition["proposal"] == proposal,
                    f"{stage} 的父代或提案不符。")
            current = proposal if proposal in valid else current
            require(transition["current"] == current, f"{stage} 的有效续写或失败回退不符。")
    skipped = {row["stage"] for row in rows if row["event"] == "slot_skipped"}
    for turn in range(1, 4):
        inputs = [by_stage[f"round_{turn}_{role}"]["candidate_id"] for role in ("Q", "E")]
        stage = f"fusion_{turn}"
        if all(identifier in valid for identifier in inputs):
            require(stage in by_stage and by_stage[stage]["parents"] == inputs,
                    f"{stage} 的融合输入不符。")
        else:
            require(stage in skipped and stage not in by_stage, f"{stage} 未按无效输入跳过。")
    for elite, identifier in (("A", pair["a"]), ("B", pair["b"])):
        for role in ("Q", "E", "I"):
            require(by_stage[f"mutation_{elite}_{role}"]["parents"] == [identifier], "精英变异输入不符。")

    expected_stages = [f"eoh_{op}_{i:02d}" for op in ("E1", "E2", "M1", "M2") for i in range(1, 11)]
    expected_stages += ["round_3_Q", "round_3_E"]
    expected_stages += [stage for stage in ("fusion_3", "fusion_final") if stage in by_stage]
    expected_stages += [f"mutation_{elite}_{role}" for elite in ("A", "B") for role in ("Q", "E", "I")]
    expected_pool = [by_stage[stage]["candidate_id"] for stage in expected_stages]
    selection = next(row for row in rows if row["event"] == "selection")
    require(selection["pool"] == expected_pool, "代末候选池不符。")
    selected = rank(expected_pool)[:10]
    require(len(selected) == 10 and [row["candidate_id"] for row in selection["selected"]] == selected,
            "代末 Top N 排序或去重不符。")
    require(selected == rank(expected_pool, actual_generation_order=True)[:10],
            "代末按登记顺序与实际响应生成顺序排序得到不同结果。")
    for item in selection["selected"]:
        expected_origin = earliest[code_digest(valid[item["candidate_id"]]["code"])]
        require(item["evaluation_origin"] == expected_origin, "重复代码的评价来源不符。")
    history = [row for row in rows if row["event"] == "history_appended"]
    require({row["role"] for row in history} == {"Q", "E", "I"} and len(history) == 3
            and all(row["generation"] == 1 and row["event_index"] > selection["event_index"] for row in history),
            "角色经验追加时机或数量不符。")

    attempts_by_stage = {}
    for row in attempts:
        attempts_by_stage.setdefault(row["stage"], row)
    role_responses = {row["stage"]: row for row in rows if row["event"] == "model_response"}
    evaluation_events = {row["candidate_id"]: row for row in rows if row["event"] == "evaluation"}
    invalid_events = {row["candidate_id"]: row for row in rows if row["event"] == "candidate_invalid"}
    skip_events = {row["stage"]: row for row in rows if row["event"] == "slot_skipped"}

    def terminal(identifier):
        return (evaluation_events.get(identifier) or invalid_events[identifier])["event_index"]

    def candidate_text(identifier, *, selected_from_population=False):
        item = valid[identifier]
        origin = earliest[code_digest(item["code"])] if selected_from_population else identifier
        objective = float(valid[origin]["record"]["average"])
        return (f"Algorithm description: {item['thought']}\nCode:\n{item['code']}\n"
                f"Objective score: {objective}\n")

    def tagged(text, names):
        pattern = r"\s*" + r"\s*".join(rf"<{name}>\s*(.*?)\s*</{name}>" for name in names) + r"\s*"
        match = re.fullmatch(pattern, text, flags=re.DOTALL)
        require(match is not None and all(value.strip() for value in match.groups()),
                f"角色响应未包含预期标签：{names}")
        return tuple(value.strip() for value in match.groups())

    def words(value):
        return len(re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", value, flags=re.UNICODE))

    for number in range(1, 31):
        require("Code:\n" not in attempts_by_stage[f"init_{number:02d}"]["prompt"],
                "初始化提示含已有候选代码。")
    for op in ("E1", "E2", "M1", "M2"):
        for number in range(1, 11):
            stage = f"eoh_{op}_{number:02d}"
            prompt = attempts_by_stage[stage]["prompt"]
            for parent in by_stage[stage]["parents"]:
                require(f"Algorithm description: {valid[parent]['thought']}\nCode:\n{valid[parent]['code']}\n" in prompt,
                        f"{stage} 未读取登记的父代。")

    initial_critic = attempts_by_stage["critic_initial"]["prompt"]
    require(f"Better code:\n{valid[pair['a']]['code']}\n" in initial_critic
            and f"Worse code:\n{valid[pair['b']]['code']}\n" in initial_critic,
            "初始 Critic 未读取原始精英代码。")
    short_reflections = []
    for stage in ("critic_initial", "critic_1", "critic_2", "critic_3"):
        reflection, advice = tagged(role_responses[stage]["response"], ("ref", "ans"))
        require(words(advice) <= 40 and (stage != "critic_initial" or words(reflection) < 20),
                f"{stage} 的词数超过提示合同。")
        short_reflections.append(reflection)
    require(population["event_index"] < attempts_by_stage["eoh_E1_01"]["event_index"]
            and max(terminal(by_stage[f"eoh_{op}_{i:02d}"]["candidate_id"])
                    for op in ("E1", "E2", "M1", "M2") for i in range(1, 11)) < pair["event_index"]
            < attempts_by_stage["critic_initial"]["event_index"], "初始化、EoH 与精英选择时序不符。")
    for role in ("Q", "E"):
        stage = f"start_{role}"
        require(by_stage[stage]["parents"] == [pair["b"]]
                and candidate_text(pair["b"], selected_from_population=True) in attempts_by_stage[stage]["prompt"]
                and role_responses["critic_initial"]["event_index"] < attempts_by_stage[stage]["event_index"],
                f"{stage} 的输入或时序不符。")
    require(max(terminal(by_stage[f"start_{role}"]["candidate_id"]) for role in ("Q", "E"))
            < attempts_by_stage["critic_1"]["event_index"], "分支启动尚未评价即进入正式轮次。")
    for turn in range(1, 4):
        critic = f"critic_{turn}"
        q_prev, e_prev = (transitions[(role, turn)]["previous"] for role in ("Q", "E"))
        critic_prompt = attempts_by_stage[critic]["prompt"]
        require(candidate_text(q_prev, selected_from_population=q_prev == pair["b"]) in critic_prompt
                and candidate_text(e_prev, selected_from_population=e_prev == pair["b"]) in critic_prompt,
                f"{critic} 未读取当前两支候选。")
        inputs = []
        for role in ("Q", "E"):
            stage = f"round_{turn}_{role}"
            candidate_id = by_stage[stage]["candidate_id"]
            inputs.append(candidate_id)
            require(by_stage[stage]["parents"] == [transitions[(role, turn)]["previous"]]
                    and candidate_text(transitions[(role, turn)]["previous"],
                                       selected_from_population=transitions[(role, turn)]["previous"] == pair["b"])
                    in attempts_by_stage[stage]["prompt"]
                    and role_responses[critic]["event_index"] < attempts_by_stage[stage]["event_index"],
                    f"{stage} 的角色输入或时序不符。")
        stage = f"fusion_{turn}"
        if stage in by_stage:
            require(all(terminal(identifier) < attempts_by_stage[stage]["event_index"] for identifier in inputs)
                    and all(candidate_text(identifier) in attempts_by_stage[stage]["prompt"] for identifier in inputs),
                    f"{stage} 未在两项评价完成后读取本轮候选。")
            end = terminal(by_stage[stage]["candidate_id"])
        else:
            end = skip_events[stage]["event_index"]
        if turn < 3:
            require(end < attempts_by_stage[f"critic_{turn + 1}"]["event_index"], "下一轮早于上轮融合结束。")
    formal_best = []
    for role in ("Q", "E"):
        formal = [by_stage[f"round_{turn}_{role}"]["candidate_id"] for turn in range(1, 4)]
        ranked = rank(formal)
        if ranked:
            require(ranked[0] == rank(formal, actual_generation_order=True)[0],
                    f"{role} 支正式轮次按登记顺序与实际响应顺序选出的最佳候选不同。")
            formal_best.append(ranked[0])
    if len(formal_best) == 2:
        require("fusion_final" in by_stage and by_stage["fusion_final"]["parents"] == formal_best,
                "末轮精英融合没有使用各支正式轮次最佳候选。")
        final_prompt = attempts_by_stage["fusion_final"]["prompt"]
        require("Reflection summary: " + " | ".join(short_reflections) in final_prompt,
                "末轮精英融合未按顺序读取短期反思。")
        for identifier in formal_best:
            origin = earliest[code_digest(valid[identifier]["code"])]
            require(f"Code:\n{valid[identifier]['code']}\nObjective score: "
                    f"{float(valid[origin]['record']['average'])}\n" in final_prompt,
                    "末轮精英融合未读取入选代码和目标值。")
    else:
        require("fusion_final" in skipped, "缺少正式分支时未跳过末轮精英融合。")
    lessons = {}
    for role in ("Q", "E"):
        (lesson,) = tagged(role_responses[f"reflection_{role}"]["response"], ("ans",))
        require(words(lesson) < 50, f"{role} 的本代经验超过 50 词。")
        lessons[role] = lesson
    lessons["I"] = role_responses["reflection_I"]["response"].strip()
    merge_prompt = ("Merge these two reflections into one concise experience. Return plain text only.\n"
                    f"Q: {lessons['Q']}\nE: {lessons['E']}")
    require(attempts_by_stage["reflection_I"]["prompt"] == merge_prompt, "整合反思读取了额外字段。")
    for item in history:
        require(item["reflection"] == lessons[item["role"]], "追加的角色历史与本代反思不一致。")
    for elite, identifier in (("A", pair["a"]), ("B", pair["b"])):
        for role in ("Q", "E", "I"):
            prompt = attempts_by_stage[f"mutation_{elite}_{role}"]["prompt"]
            require("History reflection: \n" in prompt and f"Reflection insights: {lessons[role]}\n" in prompt
                    and f"Current elite code:\n{valid[identifier]['code']}\n" in prompt,
                    f"mutation_{elite}_{role} 的反思或精英代码不符。")
    require(role_responses["reflection_I"]["event_index"]
            < min(attempts_by_stage[f"mutation_{elite}_{role}"]["event_index"]
                  for elite in ("A", "B") for role in ("Q", "E", "I")),
            "精英变异早于本代反思完成。")
    require(max(terminal(by_stage[f"mutation_{elite}_{role}"]["candidate_id"])
                for elite in ("A", "B") for role in ("Q", "E", "I")) < selection["event_index"],
            "代末筛选早于精英变异结束。")
    report.update({"status": "passed", "initial_population": original, "elites": [pair["a"], pair["b"]],
                   "final_pool_size": len(expected_pool), "final_population": selected,
                   "skipped_slots": sorted(skipped), "history_roles": [row["role"] for row in history]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--label", default="", help="为重放记录指定后缀，避免覆盖既有记录")
    args = parser.parse_args()
    if args.label and not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("--label 只允许字母、数字、下划线和连字符。")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(OUTPUT_ROOT.resolve()) or not run_dir.is_dir():
        parser.error("run_dir 必须是 outputs/p0_control 下已有运行目录。")
    output = run_dir / (f"independent_audit_{args.label}.json" if args.label else "independent_audit.json")
    if output.exists():
        parser.error("独立核验记录已存在，不覆盖；请先检查已有记录。")
    try:
        report = audit(run_dir)
    except (OSError, ValueError, KeyError, TypeError, StopIteration, json.JSONDecodeError) as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"P0 真实运行独立重放：{report['status']}；记录：{output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
