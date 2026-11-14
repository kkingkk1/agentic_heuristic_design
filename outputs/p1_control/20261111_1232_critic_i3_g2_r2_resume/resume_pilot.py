"""从已记录的第二代融合失败处手动续跑，不重放模型调用或候选评价。"""

import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ORIGINAL = RUN_DIR.parent / "20261111_1217_critic_i3_g2_r2_02"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe  # noqa: E402
from scripts.p0_controller import Candidate, Controller, StopRun, parse_split_critic, timing_from_result  # noqa: E402
from scripts.p0_runtime import JsonLog, digest, memory_and_priority  # noqa: E402
from scripts.p1_formal_matrix import verify_training_software  # noqa: E402
from scripts.p1_runtime import TaskRuntime  # noqa: E402
from scripts.p1_task_contract import load_task_contract  # noqa: E402


class PilotController(Controller):
    def long_reflection_prompt(self, role, reflections, changes):
        return (super().long_reflection_prompt(role, reflections, changes)
                + "\nFor this run, aim for at most 40 words in <ans>; "
                  "the hard limit remains strictly fewer than 50 words.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_checkpoint():
    events_path = ORIGINAL / "events.jsonl"
    config = json.loads((ORIGINAL / "config.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require([e["event_index"] for e in events] == list(range(1, len(events) + 1)), "原事件序号不连续。")
    require(events[-1]["event"] == "pilot_stopped" and "模型连接失败" in events[-1]["error"],
            "原运行不是预期的连接失败状态。")
    attempts = [e for e in events if e["event"] == "model_attempt"]
    usages = [e for e in events if e["event"] == "model_usage"]
    evaluations = [e for e in events if e["event"] == "evaluation"]
    require(len(attempts) == 52 and len(usages) == 51 and len(evaluations) == 42,
            "原调用或评价数量与断点不符。")
    require(attempts[-1]["stage"] == "fusion_final" and all(not e["retry"] for e in attempts),
            "失败调用不是预期的最终融合。")
    require(all(e["result"]["status"] == "passed" for e in evaluations), "原评价包含未通过结果。")
    selections = [e for e in events if e["event"] == "selection"]
    require(len(selections) == 1 and selections[0]["generation"] == 1, "原运行已完成代数不符。")
    preflight = next(e for e in events if e["event"] == "preflight")
    for filename, expected in preflight["code_sha256"].items():
        source = Path(filename).resolve()
        require(source.is_file() and digest(source) == expected, f"原运行输入已变化：{source}")
        if source.is_relative_to(ROOT) and not source.is_relative_to(ORIGINAL) and source.suffix in (".py", ".json", ".yaml", ".txt"):
            snapshot = ORIGINAL / "source_snapshots" / source.relative_to(ROOT)
            require(snapshot.is_file() and digest(snapshot) == expected, f"原运行源码快照不一致：{source}")
    return config, events


def restore_checkpoint(config, events, backend, log):
    task = config["task"]
    require((task["initial_proposals"], task["population_size"], task["generations"],
             task["collaboration_rounds"], task["evaluation_mode"]) == (3, 2, 2, 2, "final_quality"),
            "原运行范围不符。")
    contract = load_task_contract("tsp_aco", black_box=False)
    controller = PilotController(backend, log, population_size=2, initial_count=3, rounds=2, generations=2,
                                 seed=task["selection_seed"], task_contract=contract)
    controller.generation = 2
    registrations = [e for e in events if e["event"] == "candidate_registered"]
    require(len(registrations) == 43, "原候选注册数量不符。")
    by_id = {}
    for event in registrations:
        candidate = Candidate(event["candidate_id"], event["sequence"], event["stage"], tuple(event["parents"]))
        require(candidate.sequence == len(controller.candidates) + 1 and candidate.candidate_id not in by_id,
                "原候选编号不连续或重复。")
        controller.candidates.append(candidate)
        by_id[candidate.candidate_id] = candidate
    for event in (e for e in events if e["event"] == "proposal_response"):
        candidate = by_id[event["candidate_id"]]
        parsed = contract.parse_proposal(event["response"])
        candidate.thought, candidate.code = parsed["thought"], parsed["code"]
        candidate.generation_order = event["generation_order"]
        code_path = ORIGINAL / "candidates" / candidate.candidate_id / "problems/tsp_aco/gpt.py"
        require(code_path.read_text(encoding="utf-8") == candidate.code + "\n", "原候选代码与模型响应不符。")
    for event in (e for e in events if e["event"] == "evaluation"):
        candidate, result = by_id[event["candidate_id"]], event["result"]
        require(result["status"] == "passed" and result["fitness"] == -result["average"], "原候选评价不可信。")
        candidate.objective, candidate.fitness = float(result["average"]), float(result["fitness"])
        candidate.raw_objective = float(result["average"])
        candidate.timing_seconds, candidate.timing_scope = timing_from_result(result)
        candidate.completed_order = result["completed_monotonic_ns"]
        candidate.evaluation_origin = candidate.candidate_id
    controller._generated = max(c.generation_order or 0 for c in controller.candidates)
    failed = by_id["c0043"]
    require(failed.stage == "fusion_final" and not failed.valid and not failed.code,
            "失败提案状态不符。")
    failed.error = "模型连接失败，原调用未产生提案。"

    for event in (e for e in events if e["event"] == "history_appended"):
        require(event["generation"] == 1, "出现非第一代的历史记录。")
        controller.history[event["role"]].append(event["reflection"])
    require(all(len(controller.history[role]) == 1 for role in controller.history), "第一代角色历史不完整。")

    pair = next(e for e in events if e["event"] == "elite_pair" and e["generation"] == 2)
    a, b = by_id[pair["a"]], by_id[pair["b"]]
    critics = [e for e in events if e["event"] == "model_response" and e.get("generation") == 2
               and e["stage"] in ("critic_initial", "critic_1", "critic_2")]
    require([e["stage"] for e in critics] == ["critic_initial", "critic_1", "critic_2"], "第二代 Critic 记录不完整。")
    role_reflections = {role: [] for role in controller.design_roles}
    reflections = []
    for event in critics:
        critic = parse_split_critic(event["response"])
        views = controller.critic_views(critic)
        reflections.extend(controller.fusion_reflections(critic))
        for role in controller.design_roles:
            role_reflections[role].append(views[role]["reflection"])
    changes, formal = {}, {}
    for role in controller.design_roles:
        transitions = [e for e in events if e["event"] == "branch_transition"
                       and e["generation"] == 2 and e["role"] == role]
        require([e["round"] for e in transitions] == [0, 1, 2], "第二代分支记录不完整。")
        changes[role] = [{key: e[key] for key in ("from_objective", "proposal_objective", "to_objective")}
                         for e in transitions]
        formal[role] = [by_id[e["proposal"]] for e in transitions[1:] if by_id[e["proposal"]].valid]
        require(len(formal[role]) == 2, "第二代分支缺少有效候选。")
    best = [controller.select(formal[role], 1)[0] for role in controller.design_roles]
    require([c.candidate_id for c in best] == list(failed.parents), "最终融合父代与原记录不一致。")
    prompt = controller.final_fusion_prompt(best, reflections)
    attempt = next(e for e in reversed(events) if e["event"] == "model_attempt")
    require(prompt == attempt["prompt"], "重建的最终融合提示与原请求不一致。")

    gen2 = [e for e in registrations if e["generation"] == 2]
    eoh = [by_id[e["candidate_id"]] for e in gen2 if e["stage"].startswith("eoh_")]
    require(len(eoh) == 8, "第二代 EoH 候选数量不符。")
    last = []
    for stage in ("round_2_solution_quality_agent", "round_2_speed_agent", "fusion_2"):
        matches = [by_id[e["candidate_id"]] for e in gen2 if e["stage"] == stage]
        require(len(matches) == 1 and matches[0].valid, f"第二代缺少有效候选：{stage}")
        last.append(matches[0])
    return controller, {"a": a, "b": b, "best": best, "prompt": prompt, "eoh": eoh, "last": last,
                        "role_reflections": role_reflections, "changes": changes}


def build_resume_config(original):
    config = copy.deepcopy(original)
    config["scope"] = "两代 Critic 工程试跑的手动断点续跑；不属于 P1 正式对照。"
    config["resume"] = {"source_run": str(ORIGINAL), "source_events_sha256": digest(ORIGINAL / "events.jsonl"),
                        "reason": "最终融合的原模型调用连接失败；保留原记录并手动再调用一次。",
                        "prior_model_attempts": 52, "prior_evaluations": 42,
                        "extra_model_invocations": 1, "automatic_retries": 0}
    config["model"]["retry_policy"] = "仅手动恢复已失败的最终融合一次；续跑过程不自动重试。"
    config["budget"]["normal_model_calls"] = 62
    config["budget"]["max_initialization_and_generation_model_cli_invocations"] = 62
    require(config["budget"]["additional_infrastructure_retry_calls"] == 0, "续跑不得自动重试。")
    require(config["budget"]["normal_code_generation_slots"] == 49, "评价上限不符。")
    return config


def continue_generation(controller, state):
    fusion = controller.batch([("fusion_final", state["best"], state["prompt"])])[0]
    lessons = {}
    for role in controller.design_roles:
        prompt = controller.long_reflection_prompt(role, state["role_reflections"][role], state["changes"])
        lessons[role] = controller.request(f"reflection_{role}", prompt, probe.parse_long_term)
    lessons[controller.integrator_role] = controller.request(
        f"reflection_{controller.integrator_role}", controller.merge_reflections_prompt(lessons), probe.parse_merge)
    specs = []
    all_roles = (*controller.design_roles, controller.integrator_role)
    for elite_name, elite in (("A", state["a"]), ("B", state["b"])):
        for role in all_roles:
            prompt = (f"{controller.task}\n[History reflection as reference]:\n{' | '.join(controller.history[role])}\n"
                      f"[Reflection insights]:\n{lessons[role]}\n[Current best code]:\n{elite.code}\n"
                      "Based on the history reflections and mainly based on reflection insights and "
                      "the current best code, propose a mutation that addresses the key insights "
                      f"and improves upon the previous solutions.\n{controller.output}")
            specs.append((f"mutation_{elite_name}_{role}", (elite,), prompt))
    mutations = controller.batch(specs)
    pool = [*state["eoh"], *state["last"], fusion, *mutations]
    selected = controller.select(pool, controller.n)
    controller.event("selection", generation=2, pool=[c.candidate_id for c in pool if c is not None],
                     selected=[{"candidate_id": c.candidate_id, "evaluation_origin": c.evaluation_origin}
                               for c in selected])
    for role in all_roles:
        controller.history[role].append(lessons[role])
        controller.event("history_appended", role=role, generation=2, reflection=lessons[role])
    controller.event("finish", status="completed")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="执行续跑；省略时只校验断点。")
    args = parser.parse_args()
    original, events = read_checkpoint()
    config = build_resume_config(original)
    if not args.run:
        class CheckBackend:
            cfg = config
        controller, state = restore_checkpoint(original, events, CheckBackend(), None)
        print(f"断点验证通过：{len(controller.candidates)} 个注册候选；融合父代 "
              f"{[c.candidate_id for c in state['best']]}；提示与失败请求逐字一致。")
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
        runtime = TaskRuntime(RUN_DIR, config_path, log)
        driver = Path(__file__).resolve()
        runtime.protected.append(driver)
        runtime.protected_hashes[str(driver)] = digest(driver)
        resources = memory_and_priority()
        if resources["available_memory_bytes"] < config["resources"]["min_available_system_memory_bytes"]:
            raise StopRun("可用内存低于资源合同阈值。")
        cli = probe.cli_command()
        version, login = probe.run_command(cli + ["--version"]), probe.run_command(cli + ["login", "status"])
        if (version.get("exit_code") != 0 or login.get("exit_code") != 0
                or "Logged in using ChatGPT" not in (login.get("stdout", "") + login.get("stderr", ""))):
            raise StopRun("无法确认 Codex CLI 版本或 ChatGPT 登录。")
        runtime.cli = cli
        cpu_records = [e for e in events if e["event"] == "evaluation_cpu_final"]
        runtime.budget.model_normal = 52
        runtime.budget.eval_normal = runtime.budget.eval_starts = 42
        runtime.budget.cpu_finished = cpu_records[-1]["total_cpu_seconds"]
        runtime.budget.eval_cpu_samples = [e["accounting"]["cpu_seconds"] for e in cpu_records]
        controller, state = restore_checkpoint(original, events, runtime, log)
        runtime.assert_integrity()
        log.event("resume_preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
                  source_run=str(ORIGINAL), source_events_sha256=digest(ORIGINAL / "events.jsonl"),
                  prior_model_attempts=52, prior_evaluations=42, prior_process_tree_cpu_seconds=runtime.budget.used_cpu(),
                  extra_model_invocations=1, automatic_retries=0, code_sha256=runtime.protected_hashes)
        selected = continue_generation(controller, state)
        runtime.assert_integrity()
        log.event("pilot_completed", population=[
            {"candidate_id": c.candidate_id, "evaluation_origin": c.evaluation_origin, "stage": c.stage,
             "fitness": c.fitness, "average_objective": c.raw_objective,
             "average_solve_wall_seconds": c.timing_seconds} for c in selected],
            model_calls=runtime.budget.model_normal, evaluation_starts=runtime.budget.eval_starts,
            process_tree_cpu_seconds=runtime.budget.used_cpu(), elapsed_seconds=time.monotonic() - started)
        print(f"断点续跑完成：{RUN_DIR}", flush=True)
        return 0
    except Exception as exc:
        log.event("pilot_stopped", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - started)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
