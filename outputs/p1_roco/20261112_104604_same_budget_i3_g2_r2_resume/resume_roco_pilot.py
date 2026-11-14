"""从第二代 Critic 断点续跑，复用原响应并保持原预算。"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ORIGINAL = RUN_DIR.parent / "20261112_104604_same_budget_i3_g2_r2"
PARSER_POLICY_HASHES = {
    "scripts/p0_model_probe.py": "0f291de0fec28329101c213cf9f9a89dfbe82be6ada0bf0370eb02c59046f029",
    "scripts/p0_controller.py": "90dad5865b816867bf77e2f0f969c1dba9f4df020a726a72a44ccc2e0e43aa3a",
}
sys.path.insert(0, str(ROOT))

from baselines.roco.paper_controller import RoCoController  # noqa: E402
from baselines.roco.runtime import RoCoRuntime  # noqa: E402
from scripts import p0_model_probe as probe  # noqa: E402
from scripts.p0_controller import Candidate, StopRun, parse_round_critic, timing_from_result  # noqa: E402
from scripts.p0_runtime import JsonLog, digest, memory_and_priority  # noqa: E402
from scripts.p1_formal_matrix import verify_training_software  # noqa: E402
from scripts.p1_task_contract import load_task_contract  # noqa: E402


class PilotRoCoController(RoCoController):
    def long_reflection_prompt(self, role, reflections, changes):
        return (super().long_reflection_prompt(role, reflections, changes)
                + "\nFor this run, aim for at most 40 words in <ans>.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_checkpoint():
    config = json.loads((ORIGINAL / "config.json").read_text(encoding="utf-8"))
    events_path = ORIGINAL / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    require([row["event_index"] for row in events] == list(range(1, len(events) + 1)), "事件序号不连续。")
    require(events[-1]["event"] == "pilot_stopped"
            and "critic_initial 的角色输出无效" in events[-1]["error"]
            and "词数限制" in events[-1]["error"], "原运行不是预期的 Critic 格式失败。")
    audit = json.loads((ORIGINAL / "audit_roco_pilot.json").read_text(encoding="utf-8"))
    require(audit.get("status") == "partial" and audit.get("source_events_sha256") == digest(events_path),
            "原运行缺少对应的部分审计。")
    attempts = [row for row in events if row["event"] == "model_attempt"]
    evaluations = [row for row in events if row["event"] == "evaluation"]
    cpu_records = [row for row in events if row["event"] == "evaluation_cpu_final"]
    require(len(attempts) == 41 and len(evaluations) == len(cpu_records) == 34,
            "原调用或评价数量与断点不符。")
    require(attempts[-1]["stage"] == "critic_initial" and all(not row["retry"] for row in attempts),
            "失败调用不是预期的初始 Critic。")
    selections = [row for row in events if row["event"] == "selection"]
    require(len(selections) == 1 and selections[0]["generation"] == 1, "原运行已完成代数不符。")
    preflight = next(row for row in events if row["event"] == "preflight")
    for filename, expected in preflight["code_sha256"].items():
        source = Path(filename).resolve()
        relative = source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else None
        expected_current = PARSER_POLICY_HASHES.get(relative, expected)
        require(source.is_file() and digest(source) == expected_current, f"原运行输入变化超出许可范围：{source}")
        if source.is_relative_to(ROOT) and not source.is_relative_to(ORIGINAL) and source.suffix in (
                ".py", ".json", ".yaml", ".txt"):
            snapshot = ORIGINAL / "source_snapshots" / source.relative_to(ROOT)
            require(snapshot.is_file() and digest(snapshot) == expected, f"原运行源码快照不一致：{source}")
    return config, events


def restore_checkpoint(config, events, backend, log):
    task = config["task"]
    require((task["initial_proposals"], task["population_size"], task["generations"],
             task["collaboration_rounds"], task["evaluation_mode"]) == (3, 2, 2, 2, "final_quality"),
            "原运行范围不符。")
    contract = load_task_contract("tsp_aco", black_box=False)
    controller = PilotRoCoController(backend, log, population_size=2, initial_count=3, rounds=2, generations=2,
                                     seed=task["selection_seed"], task_contract=contract)
    controller.generation = 2
    registrations = [row for row in events if row["event"] == "candidate_registered"]
    require(len(registrations) == 34, "原候选注册数量不符。")
    by_id = {}
    for row in registrations:
        candidate = Candidate(row["candidate_id"], row["sequence"], row["stage"], tuple(row["parents"]))
        require(candidate.sequence == len(controller.candidates) + 1 and candidate.candidate_id not in by_id,
                "原候选编号不连续或重复。")
        controller.candidates.append(candidate)
        by_id[candidate.candidate_id] = candidate
    for row in (item for item in events if item["event"] == "proposal_response"):
        candidate = by_id[row["candidate_id"]]
        parsed = contract.parse_proposal(row["response"])
        candidate.thought, candidate.code = parsed["thought"], parsed["code"]
        candidate.generation_order = row["generation_order"]
        code_path = ORIGINAL / "candidates" / candidate.candidate_id / "problems/tsp_aco/gpt.py"
        require(code_path.read_text(encoding="utf-8") == candidate.code + "\n", "原候选代码与模型响应不符。")
    for row in (item for item in events if item["event"] == "evaluation"):
        candidate, result = by_id[row["candidate_id"]], row["result"]
        if result["status"] != "passed":
            candidate.error = result.get("error", result["status"])
            continue
        require(result["fitness"] == -result["average"], "原候选评价不可信。")
        candidate.objective, candidate.fitness = float(result["average"]), float(result["fitness"])
        candidate.raw_objective = float(result["average"])
        candidate.timing_seconds, candidate.timing_scope = timing_from_result(result)
        candidate.completed_order = result["completed_monotonic_ns"]
        candidate.evaluation_origin = candidate.candidate_id
    controller._generated = max(candidate.generation_order or 0 for candidate in controller.candidates)

    role_names = {"Explorer": "explorer", "Exploiter": "exploiter", "Integrator": "integrator"}
    for row in (item for item in events if item["event"] == "history_appended"):
        require(row["generation"] == 1 and row["role"] in role_names, "原角色历史不符。")
        controller.history[role_names[row["role"]]].append(row["reflection"])
    require(all(len(controller.history[role]) == 1 for role in controller.history), "第一代角色历史不完整。")

    selection = next(row for row in events if row["event"] == "selection" and row["generation"] == 1)
    population = controller.select([by_id[cid] for cid in selection["pool"]], 2)
    require([(c.candidate_id, c.evaluation_origin) for c in population]
            == [(row["candidate_id"], row["evaluation_origin"]) for row in selection["selected"]],
            "第一代种群与原筛选记录不符。")
    pair = next(row for row in events if row["event"] == "elite_pair" and row["generation"] == 2)
    chosen = {candidate.candidate_id: candidate for candidate in population}
    a, b = chosen[pair["a"]], chosen[pair["b"]]
    prompt = controller.initial_critic_prompt(a, b)
    attempt = next(row for row in reversed(events) if row["event"] == "model_attempt")
    require(prompt == attempt["prompt"], "重建的 Critic 提示与原请求不一致。")
    response = next(row for row in reversed(events) if row["event"] == "model_response")
    require(response["stage"] == "critic_initial" and response["generation"] == 2
            and response["prompt"] == prompt, "原 Critic 响应与断点不符。")
    parsed = probe.parse_critic(response["response"])
    require(probe.english_words(parsed["reflection"]) == 22, "原 Critic 反馈不是预期的 22 词。")
    eoh = [by_id[row["candidate_id"]] for row in registrations
           if row["generation"] == 2 and row["stage"].startswith("eoh_")]
    require(len(eoh) == 8, "第二代 EoH 候选数量不符。")
    return controller, {"a": a, "b": b, "prompt": prompt, "eoh": eoh,
                        "critic_response": response["response"], "critic_event_index": response["event_index"]}


def resume_config(original, events):
    config = copy.deepcopy(original)
    config["scope"] = "RoCo 两代工程试跑的人工断点续跑；原运行失败记录保留。"
    response = next(row for row in reversed(events) if row["event"] == "model_response")
    config["resume"] = {"source_run": str(ORIGINAL), "source_events_sha256": digest(ORIGINAL / "events.jsonl"),
                        "reason": "用户要求词数仅作建议；完整复用原第二代 Critic 反馈。",
                        "source_response_event_index": response["event_index"],
                        "source_response_sha256": hashlib.sha256(response["response"].encode("utf-8")).hexdigest(),
                        "prior_model_attempts": 41, "prior_evaluations": 34,
                        "extra_model_invocations": 0, "automatic_retries": 0,
                        "parser_policy_sha256": PARSER_POLICY_HASHES}
    require(config["budget"]["normal_model_calls"] == 61
            and config["budget"]["max_initialization_and_generation_model_cli_invocations"] == 61
            and config["budget"]["additional_infrastructure_retry_calls"] == 0
            and config["budget"]["additional_infrastructure_evaluation_starts"] == 0
            and config["budget"]["max_initialization_and_generation_evaluation_starts"] == 49,
            "续跑不得添加自动重试或评价次数。")
    require(len([row for row in events if row["event"] == "evaluation_cpu_final"]) == 34,
            "原运行 CPU 记录不完整。")
    return config


def continue_generation(controller, state):
    critic = probe.parse_critic(state["critic_response"])
    controller.event("reused_model_response", stage="critic_initial", generation=2,
                     source_event_index=state["critic_event_index"],
                     source_events_sha256=digest(ORIGINAL / "events.jsonl"),
                     response=state["critic_response"], policy="词数为建议，完整保留反馈，不追加调用。")
    views = controller.critic_views(critic)
    reflections = controller.fusion_reflections(critic)
    role_reflections = {role: [views[role]["reflection"]] for role in controller.design_roles}
    first_role, second_role = controller.design_roles
    all_roles = (*controller.design_roles, controller.integrator_role)
    b = state["b"]
    starts = controller.batch([(f"start_{role}", (b,), controller.branch_prompt(role, b, views[role]))
                               for role in controller.design_roles])
    current = {role: result if result.valid else b for role, result in zip(controller.design_roles, starts)}
    changes = {role: [{"from_objective": b.objective, "proposal_objective": result.objective,
                       "to_objective": current[role].objective}]
               for role, result in zip(controller.design_roles, starts)}
    for role, result in zip(controller.design_roles, starts):
        controller.event("branch_transition", generation=2, role=role, round=0, previous=b.candidate_id,
                         proposal=result.candidate_id, current=current[role].candidate_id, **changes[role][-1])
    formal = {role: [] for role in controller.design_roles}
    last = []
    for turn in range(1, 3):
        critic = controller.request(f"critic_{turn}", controller.branch_critic_prompt(current[first_role],
                                                                                      current[second_role]),
                                    parse_round_critic)
        views = controller.critic_views(critic)
        reflections.extend(controller.fusion_reflections(critic))
        for role in controller.design_roles:
            role_reflections[role].append(views[role]["reflection"])
        previous = current.copy()
        proposals = controller.batch([(f"round_{turn}_{role}", (previous[role],),
                                       controller.branch_prompt(role, previous[role], views[role]))
                                      for role in controller.design_roles])
        for role, result in zip(controller.design_roles, proposals):
            if result.valid:
                formal[role].append(result)
                current[role] = result
            change = {"from_objective": previous[role].objective, "proposal_objective": result.objective,
                      "to_objective": current[role].objective}
            changes[role].append(change)
            controller.event("branch_transition", generation=2, role=role, round=turn,
                             previous=previous[role].candidate_id, proposal=result.candidate_id,
                             current=current[role].candidate_id, **change)
        fusion = None
        if all(candidate.valid for candidate in proposals):
            prompt = controller.round_fusion_prompt(proposals, "Lower")
            fusion = controller.batch([(f"fusion_{turn}", proposals, prompt)])[0]
        else:
            controller.event("slot_skipped", generation=2, stage=f"fusion_{turn}", reason="invalid_new_input")
        last = [*proposals, fusion]
    elite_fusion = None
    if all(formal[role] for role in controller.design_roles):
        best = [controller.select(formal[role], 1)[0] for role in controller.design_roles]
        prompt = controller.final_fusion_prompt(best, reflections)
        elite_fusion = controller.batch([("fusion_final", best, prompt)])[0]
    else:
        controller.event("slot_skipped", generation=2, stage="fusion_final", reason="missing_formal_branch")
    lessons = {}
    for role in controller.design_roles:
        prompt = controller.long_reflection_prompt(role, role_reflections[role], changes)
        lessons[role] = controller.request(f"reflection_{role}", prompt, probe.parse_long_term)
    lessons[controller.integrator_role] = controller.request(
        f"reflection_{controller.integrator_role}", controller.merge_reflections_prompt(lessons), probe.parse_merge)
    specs = []
    for elite_name, elite in (("A", state["a"]), ("B", b)):
        for role in all_roles:
            prompt = (f"{controller.task}\n[History reflection as reference]:\n{' | '.join(controller.history[role])}\n"
                      f"[Reflection insights]:\n{lessons[role]}\n[Current best code]:\n{elite.code}\n"
                      "Based on the history reflections and mainly based on reflection insights and "
                      "the current best code, propose a mutation that addresses the key insights "
                      f"and improves upon the previous solutions.\n{controller.output}")
            specs.append((f"mutation_{elite_name}_{role}", (elite,), prompt))
    mutations = controller.batch(specs)
    pool = [*state["eoh"], *last, elite_fusion, *mutations]
    selected = controller.select(pool, controller.n)
    controller.event("selection", generation=2, pool=[candidate.candidate_id for candidate in pool if candidate],
                     selected=[{"candidate_id": candidate.candidate_id, "evaluation_origin": candidate.evaluation_origin}
                               for candidate in selected])
    for role in all_roles:
        controller.history[role].append(lessons[role])
        controller.event("history_appended", role=role, generation=2, reflection=lessons[role])
    controller.event("finish", status="completed")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="执行续跑；省略时只核验断点。")
    args = parser.parse_args()
    original, events = read_checkpoint()
    config = resume_config(original, events)
    if not args.run:
        class CheckBackend:
            cfg = config
        controller, state = restore_checkpoint(original, events, CheckBackend(), None)
        print(f"断点验证通过：{len(controller.candidates)} 个候选；精英 "
              f"{state['a'].candidate_id}/{state['b'].candidate_id}；Critic 提示与原请求逐字一致，"
              "原反馈完整可用，无须额外调用。")
        return 0
    config_path = RUN_DIR / "config.json"
    with config_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    log = JsonLog(RUN_DIR / "events.jsonl")
    started = time.monotonic()
    try:
        matrix = json.loads((ROOT / "cfg/p1_formal_matrix.json").read_text(encoding="utf-8"))
        verify_training_software(matrix)
        runtime = RoCoRuntime(RUN_DIR, config_path, log)
        for source in (Path(__file__).resolve(), ORIGINAL / "config.json", ORIGINAL / "events.jsonl",
                       ORIGINAL / "audit_roco_pilot.json"):
            runtime.protected.append(source)
            runtime.protected_hashes[str(source)] = digest(source)
        for source in runtime.protected:
            source = source.resolve()
            if source.is_relative_to(ROOT) and not source.is_relative_to(RUN_DIR) and source.suffix in (
                    ".py", ".json", ".yaml", ".txt"):
                snapshot = RUN_DIR / "source_snapshots" / source.relative_to(ROOT)
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as src, snapshot.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                require(digest(snapshot) == runtime.protected_hashes[str(source)], f"源码快照不一致：{source}")
        resources = memory_and_priority()
        if resources["available_memory_bytes"] < config["resources"]["min_available_system_memory_bytes"]:
            raise StopRun("可用内存低于资源合同阈值。")
        cli = probe.cli_command()
        version, login = probe.run_command(cli + ["--version"]), probe.run_command(cli + ["login", "status"])
        if (version.get("exit_code") != 0 or login.get("exit_code") != 0
                or "Logged in using ChatGPT" not in (login.get("stdout", "") + login.get("stderr", ""))):
            raise StopRun("无法确认 Codex CLI 版本或 ChatGPT 登录。")
        runtime.cli = cli
        cpu_records = [row for row in events if row["event"] == "evaluation_cpu_final"]
        runtime.budget.model_normal = 41
        runtime.budget.eval_normal = runtime.budget.eval_starts = 34
        runtime.budget.cpu_finished = cpu_records[-1]["total_cpu_seconds"]
        runtime.budget.eval_cpu_samples = [row["accounting"]["cpu_seconds"] for row in cpu_records]
        controller, state = restore_checkpoint(original, events, runtime, log)
        runtime.assert_integrity()
        log.event("resume_preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
                  source_run=str(ORIGINAL), source_events_sha256=digest(ORIGINAL / "events.jsonl"),
                  prior_model_attempts=41, prior_evaluations=34, prior_process_tree_cpu_seconds=runtime.budget.used_cpu(),
                  extra_model_invocations=0, automatic_retries=0, parser_policy_sha256=PARSER_POLICY_HASHES,
                  code_sha256=runtime.protected_hashes)
        selected = continue_generation(controller, state)
        runtime.assert_integrity()
        log.event("pilot_completed", population=[
            {"candidate_id": candidate.candidate_id, "evaluation_origin": candidate.evaluation_origin,
             "stage": candidate.stage, "fitness": candidate.fitness,
             "average_objective": candidate.raw_objective, "average_solve_wall_seconds": candidate.timing_seconds}
            for candidate in selected], model_calls=runtime.budget.model_normal,
            evaluation_starts=runtime.budget.eval_starts, process_tree_cpu_seconds=runtime.budget.used_cpu(),
            elapsed_seconds=time.monotonic() - started)
        print(f"RoCo 断点续跑完成：{RUN_DIR}", flush=True)
        return 0
    except Exception as exc:
        log.event("pilot_stopped", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - started)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
