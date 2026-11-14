"""按原合同恢复首组第 3 代第 2 轮融合；旧事件和失败候选只读。"""

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from unittest.mock import patch


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
PREVIOUS = ROOT / "outputs/p1_manual_retry/round_3_quality_20261112_145256_3c8fe086"
INTERRUPTED = ROOT / "outputs/p1_control/20261112_145816_48da5a36_resume"
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate, p1_formal_matrix as matrix
from scripts.p0_controller import Candidate, Controller, StopRun, check_modification_scope, parse_split_critic, timing_from_result
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from scripts.p1_classified_runtime import ClassifiedTaskRuntime, classify_failure

spec = importlib.util.spec_from_file_location("p1_previous_recovery", PREVIOUS / "resume_after_connection.py")
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
base = previous.base
require = base.require
read_jsonl = base.read_jsonl


class ContinueController(previous.RecoveryController):
    def event(self, kind, **fields):
        if kind == "initial_population" and self._resume_initial_event:
            self._resume_initial_event = False
            return Controller.event(self, "resume_population", source_generation=self._generation_offset, **fields)
        return Controller.event(self, kind, **fields)

    def run_remaining(self, population, generations):
        require(self.generation == 3 and generations == 3, "续跑代次不符。")
        self._generation_offset = self.generation
        self._resume_population = population
        self.generations = generations
        return Controller.run(self)


def reconstruct(backend, log):
    """逐候选回读评价，并重放两代筛选及随机父代，恢复第 3 代分支。"""
    with patch.object(previous, "RecoveryController", ContinueController):
        controller, earlier = previous.reconstruct(backend, log)
    controller._saved_critic = None
    controller._saved_round_candidates = None
    config = earlier["config"]
    require((INTERRUPTED / "config.json").read_bytes() == (previous.INTERRUPTED / "config.json").read_bytes(),
            "第二次续跑的配置发生变化。")
    events = read_jsonl(INTERRUPTED / "events.jsonl")
    require([x["event_index"] for x in events] == list(range(1, len(events) + 1)), "第二次续跑事件不连续。")
    preflight = [x for x in events if x["event"] == "preflight"]
    require(len(preflight) == 1, "第二次续跑启动检查不唯一。")
    for name, expected in preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"历史受保护文件变化：{name}")
    starts = [x for x in events if x["event"] == "formal_run_started"]
    require(len(starts) == 1 and starts[0]["resume_of"] == str(previous.INTERRUPTED)
            and starts[0]["total_model_call_cap"] == 421, "第二次续跑身份或预算不符。")
    attempts = [x for x in events if x["event"] == "model_attempt"]
    failures = [x for x in events if x["event"] == "model_process" and x["stage"] == "fusion_2"]
    require(len(attempts) == 126 and len(failures) == 1 and attempts[-1]["stage"] == "fusion_2"
            and classify_failure(failures[0]["result"])["category"] == "policy"
            and events[-1]["event"] == "preflight_or_run_stopped", "停止点不是预期的审核拒绝。")

    registrations = [x for x in events if x["event"] == "candidate_registered"]
    responses = {x["candidate_id"]: x for x in events if x["event"] == "proposal_response"}
    evaluations = {x["candidate_id"]: x["result"] for x in events if x["event"] == "evaluation"}
    require(len(registrations) == 113 and len(responses) == len(evaluations) == 112,
            "第二次续跑候选或评价数量不符。")
    candidates = {x.candidate_id: x for x in controller.candidates}
    require(len(candidates) == 82 and controller._generated == 80, "历史候选起点不符。")
    for row in registrations:
        ident, seq = row["candidate_id"], row["sequence"]
        require(ident == f"c{seq:04d}" and seq == len(controller.candidates) + 1,
                f"候选编号不连续：{ident}")
        candidate = Candidate(ident, seq, row["stage"], tuple(row["parents"]))
        controller.candidates.append(candidate)
        candidates[ident] = candidate
        if ident == "c0195":
            require(ident not in responses and ident not in evaluations and row["stage"] == "fusion_2",
                    "失败融合候选状态不符。")
            candidate.error = "模型请求被安全审核拒绝，未产生提案；用户指示断点续跑。"
            continue
        response, result = responses[ident], evaluations[ident]
        parsed = controller.parse_proposal(response["response"])
        check_modification_scope(parsed["code"])
        candidate.thought, candidate.code = parsed["thought"], parsed["code"]
        candidate.generation_order = response["generation_order"]
        directory = INTERRUPTED / "candidates" / ident
        require(json.loads((directory / "result.json").read_text(encoding="utf-8")) == result
                and (directory / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8") == candidate.code + "\n",
                f"候选或评价记录不一致：{ident}")
        if result["status"] != "passed":
            candidate.error = str(result.get("error") or result["status"])
            continue
        checked = p0_tsp_aco_evaluate.read_complete(directory / "stdout.txt")
        require(checked["average"] == result["average"] and checked["fitness"] == result["fitness"]
                and checked["objectives"] == result["objectives"], f"候选评价输出不一致：{ident}")
        candidate.objective = candidate.raw_objective = float(result["average"])
        candidate.fitness = float(result["fitness"])
        candidate.timing_seconds, candidate.timing_scope = timing_from_result(result)
        candidate.completed_order = result["completed_monotonic_ns"]
        candidate.evaluation_origin = ident
    orders = sorted(x["generation_order"] for x in responses.values())
    require(orders == list(range(81, 193)), "生成顺序不连续。")
    controller._generated = 192
    require(max(c.completed_order for c in controller.candidates if c.valid) < time.monotonic_ns(),
            "当前单调时钟早于历史评价，不能重建去重顺序。")

    selections = [x for x in events if x["event"] == "selection"]
    require([x["generation"] for x in selections] == [1, 2], "历史代末筛选不完整。")
    populations = {}
    for row in selections:
        selected = controller.select([candidates[x] for x in row["pool"]], controller.n)
        require([{"candidate_id": x.candidate_id, "evaluation_origin": x.evaluation_origin} for x in selected]
                == row["selected"], f"第 {row['generation']} 代筛选重建不一致。")
        populations[row["generation"]] = selected
    pairs = {}
    for generation in (2, 3):
        population = populations[generation - 1]
        eoh = [x for x in registrations if x.get("generation") == generation and x["stage"].startswith("eoh_")]
        require(len(eoh) == 40, f"第 {generation} 代 EoH 槽位不符。")
        for row in eoh:
            operator = row["stage"].split("_")[1]
            parents = controller.eoh_parents(population, 5 if operator in ("E1", "E2") else 1)
            require([x.candidate_id for x in parents] == row["parents"], f"随机父代不一致：{row['candidate_id']}")
        a, b = controller.elite_pair(population)
        pair = [x for x in events if x["event"] == "elite_pair" and x["generation"] == generation]
        require(len(pair) == 1 and (a.candidate_id, b.candidate_id) == (pair[0]["a"], pair[0]["b"]),
                f"第 {generation} 代双精英不一致。")
        pairs[generation] = (a, b)
    history = [x for x in events if x["event"] == "history_appended"]
    require(len(history) == 6, "前两代角色历史不完整。")
    for row in history:
        controller.history[row["role"]].append(row["reflection"])
    require(all(len(x) == 2 for x in controller.history.values()), "前两代角色历史不符。")
    controller.generation = 3
    a, b = pairs[3]
    critic_rows = {x["stage"]: x for x in events if x["event"] == "model_response" and x.get("generation") == 3}
    require(set(critic_rows) == {"critic_initial", "critic_1", "critic_2"}, "第 3 代 Critic 记录不符。")
    require(critic_rows["critic_initial"]["prompt"] == controller.initial_critic_prompt(a, b),
            "第 3 代初始 Critic 提示不一致。")
    critic = parse_split_critic(critic_rows["critic_initial"]["response"])
    views = controller.critic_views(critic)
    reflections = controller.fusion_reflections(critic)
    role_reflections = {role: [views[role]["reflection"]] for role in controller.design_roles}
    formal = {role: [] for role in controller.design_roles}
    changes = {role: [] for role in controller.design_roles}
    current = {}
    transitions = {(x["role"], x["round"]): x for x in events
                   if x["event"] == "branch_transition" and x.get("generation") == 3}
    require(len(transitions) == 6, "第 3 代分支转移不完整。")

    def restore_round(turn, feedback):
        before = {role: b if turn == 0 else current[role] for role in controller.design_roles}
        proposals = []
        for role in controller.design_roles:
            row = transitions[(role, turn)]
            proposal = candidates[row["proposal"]]
            expected_stage = f"start_{role}" if turn == 0 else f"round_{turn}_{role}"
            require(proposal.stage == expected_stage and proposal.parents == (before[role].candidate_id,)
                    and row["previous"] == before[role].candidate_id
                    and responses[proposal.candidate_id]["prompt"] == controller.branch_prompt(role, before[role], feedback[role]),
                    f"分支输入或提示不一致：{expected_stage}")
            current[role] = proposal if proposal.valid else before[role]
            if turn and proposal.valid:
                formal[role].append(proposal)
            change = {"from_objective": before[role].objective, "proposal_objective": proposal.objective,
                      "to_objective": current[role].objective}
            require(row["current"] == current[role].candidate_id and all(row[k] == v for k, v in change.items()),
                    f"分支转移不一致：{expected_stage}")
            changes[role].append(change)
            proposals.append(proposal)
        return proposals

    restore_round(0, views)
    proposals = None
    for turn in (1, 2):
        row = critic_rows[f"critic_{turn}"]
        require(row["prompt"] == controller.branch_critic_prompt(*(current[r] for r in controller.design_roles)),
                f"第 {turn} 轮 Critic 提示不一致。")
        critic = parse_split_critic(row["response"])
        views = controller.critic_views(critic)
        reflections.extend(controller.fusion_reflections(critic))
        for role in controller.design_roles:
            role_reflections[role].append(views[role]["reflection"])
        proposals = restore_round(turn, views)
        if turn == 1 and all(x.valid for x in proposals):
            prior = [x for x in responses.values() if x["generation"] == 3 and x["stage"] == "fusion_1"]
            require(len(prior) == 1 and prior[0]["prompt"] == controller.round_fusion_prompt(proposals, "Lower"),
                    "第 1 轮融合提示不一致。")
    failed = candidates["c0195"]
    require(all(x.valid for x in proposals) and failed.parents == tuple(x.candidate_id for x in proposals)
            and attempts[-1]["prompt"] == controller.round_fusion_prompt(proposals, "Lower"),
            "被拒融合的父代或提示与原记录不一致。")
    cpu = [x["accounting"]["cpu_seconds"] for x in events if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == 112 and len(earlier["cpu_samples"]) == 79, "累计 CPU 记录不完整。")
    return controller, {"config": config, "a": a, "b": b, "current": current, "formal": formal,
                        "role_reflections": role_reflections, "reflections": reflections, "changes": changes,
                        "proposals": proposals, "retry_prompt": attempts[-1]["prompt"],
                        "eoh": [candidates[x["candidate_id"]] for x in registrations
                                if x.get("generation") == 3 and x["stage"].startswith("eoh_")],
                        "cpu_samples": earlier["cpu_samples"] + cpu,
                        "prior_hashes": preflight[0]["code_sha256"]}


def finish_after_round2(controller, state):
    """按原控制器从第 3 轮继续；融合提案无效时仍保留分支。"""
    generation = controller.generation
    first, second = controller.design_roles
    all_roles = (*controller.design_roles, controller.integrator_role)
    current, formal = state["current"], state["formal"]
    role_reflections, reflections, changes = state["role_reflections"], state["reflections"], state["changes"]
    critic = controller.request("critic_3", controller.branch_critic_prompt(current[first], current[second]),
                                parse_split_critic)
    views = controller.critic_views(critic)
    reflections.extend(controller.fusion_reflections(critic))
    for role in controller.design_roles:
        role_reflections[role].append(views[role]["reflection"])
    previous_current = current.copy()
    proposals = controller.batch([(f"round_3_{role}", (previous_current[role],),
                                   controller.branch_prompt(role, previous_current[role], views[role]))
                                  for role in controller.design_roles])
    for role, proposal in zip(controller.design_roles, proposals):
        if proposal.valid:
            formal[role].append(proposal)
            current[role] = proposal
        change = {"from_objective": previous_current[role].objective, "proposal_objective": proposal.objective,
                  "to_objective": current[role].objective}
        changes[role].append(change)
        controller.event("branch_transition", generation=generation, role=role, round=3,
                         previous=previous_current[role].candidate_id, proposal=proposal.candidate_id,
                         current=current[role].candidate_id, **change)
    fusion = None
    if all(x.valid for x in proposals):
        better = "Lower" if controller.direction == "min" else "Higher"
        fusion = controller.batch([("fusion_3", proposals, controller.round_fusion_prompt(proposals, better))])[0]
    else:
        controller.event("slot_skipped", generation=generation, stage="fusion_3", reason="invalid_new_input")
    elite_fusion = None
    if all(formal[role] for role in controller.design_roles):
        best = [controller.select(formal[role], 1)[0] for role in controller.design_roles]
        elite_fusion = controller.batch([("fusion_final", best, controller.final_fusion_prompt(best, reflections))])[0]
    else:
        controller.event("slot_skipped", generation=generation, stage="fusion_final", reason="missing_formal_branch")
    lessons = {}
    for role in controller.design_roles:
        lessons[role] = controller.request(f"reflection_{role}",
                                           controller.long_reflection_prompt(role, role_reflections[role], changes),
                                           probe.parse_long_term)
    lessons[controller.integrator_role] = controller.request(
        f"reflection_{controller.integrator_role}", controller.merge_reflections_prompt(lessons), probe.parse_merge)
    specs = []
    for elite_name, elite in (("A", state["a"]), ("B", state["b"])):
        for role in all_roles:
            prompt = (f"{controller.task}\n[History reflection as reference]:\n{' | '.join(controller.history[role])}\n"
                      f"[Reflection insights]:\n{lessons[role]}\n[Current best code]:\n{elite.code}\n"
                      "Based on the history reflections and mainly based on reflection insights and "
                      "the current best code, propose a mutation that addresses the key insights "
                      f"and improves upon the previous solutions.\n{controller.output}")
            specs.append((f"mutation_{elite_name}_{role}", (elite,), prompt))
    mutations = controller.batch(specs)
    pool = [*state["eoh"], *proposals, fusion, elite_fusion, *mutations]
    selected = controller.select(pool, controller.n)
    controller.event("selection", generation=generation, pool=[x.candidate_id for x in pool if x is not None],
                     selected=[{"candidate_id": x.candidate_id, "evaluation_origin": x.evaluation_origin} for x in selected])
    for role in all_roles:
        controller.history[role].append(lessons[role])
        controller.event("history_appended", role=role, generation=generation, reflection=lessons[role])
    return selected


@contextmanager
def recovery_guard(config, run_dir, log):
    require(os.name == "nt", "正式研发互斥要求 Windows。")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateMutexW(None, False, r"Local\agentic_heuristic_p1_formal")
    if not handle:
        raise OSError(ctypes.get_last_error(), "无法创建 P1 研发互斥锁。")
    acquired = False
    try:
        result = kernel.WaitForSingleObject(handle, 0)
        if result == 0x102:
            raise ValueError("已有 P1 正式研发运行，不能并发续跑。")
        if result not in (0, 0x80):
            raise OSError(ctypes.get_last_error(), "无法取得 P1 研发互斥锁。")
        acquired = True
        matrix.verify_training_config(config)
        started = set()
        for path in (ROOT / "outputs/p1_control").glob("*/config.json"):
            if path.parent.resolve() == run_dir.resolve():
                continue
            prior = json.loads(path.read_text(encoding="utf-8"))
            if (prior.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                    or matrix.run_identity(prior) != matrix.run_identity(config)):
                continue
            event_path = path.parent / "events.jsonl"
            if event_path.is_file() and any(x["event"] == "formal_run_started" for x in read_jsonl(event_path)):
                started.add(path.parent.resolve())
        require(started == {base.SOURCE.resolve(), previous.INTERRUPTED.resolve(), INTERRUPTED.resolve()},
                "该重复索引存在未知已启动运行，禁止续跑。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=config["formal_replicate"], selection_seed=task["selection_seed"],
                  resume_of=str(INTERRUPTED), interrupted_events_sha256=digest(INTERRUPTED / "events.jsonl"),
                  extra_model_cli_invocations=1, total_model_call_cap=421,
                  note="用户指示断点续跑；本次重试计入现有正常调用额度，不增加总上限。")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def simulate():
    from tests.test_p0_controller import FixtureBackend, MemoryLog
    first_backend, first_log = FixtureBackend(), MemoryLog()
    second_backend, second_log = FixtureBackend(), MemoryLog()
    original, original_state = base.reconstruct(first_backend, first_log)
    replacement, replacement_state = base.reconstruct(second_backend, second_log)
    selected_original = base.finish_first_generation(original, original_state)
    selected_replacement = finish_after_round2(replacement, replacement_state)
    require(sorted(first_backend.calls) == sorted(second_backend.calls)
            and [x.candidate_id for x in selected_original] == [x.candidate_id for x in selected_replacement]
            and original.history == replacement.history, "第 3 轮续写与冻结控制器基线不一致。")
    backend, log = FixtureBackend(), MemoryLog()
    controller, state = reconstruct(backend, log)
    with recovery_guard(state["config"], RECOVERY / "unused_guard_fixture", MemoryLog()):
        code = ("from pathlib import Path\nfrom scripts import p1_formal_matrix as m\ntry:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED", "续跑互斥锁不一致。")
    retried = controller.batch([("fusion_2", state["proposals"], state["retry_prompt"])])[0]
    require(retried.candidate_id == "c0196", "重试候选身份不符。")
    selected = finish_after_round2(controller, state)
    require((len(backend.calls), len(backend.evaluations), len(selected)) == (15, 11, 10),
            "第 3 代剩余调用或评价槽位不符。")
    selected = controller.run_remaining(selected, 3)
    require(len(selected) == 10 and controller.generation == 6
            and [x["generation"] for x in log.items if x["event"] == "selection"] == [3, 4, 5, 6],
            "后续代次或筛选不符。")
    print(json.dumps({"status": "passed", "checkpoint": "generation_3_fusion_2",
                      "prior_physical_calls": 212, "prior_evaluations": 191,
                      "same_contract_comparator": True, "formal_mutex_shared": True}, ensure_ascii=False))


def resume():
    config = json.loads((INTERRUPTED / "config.json").read_text(encoding="utf-8"))
    matrix.verify_training_software(matrix.load_matrix()[0])
    class CheckBackend:
        cfg = config
    reconstruct(CheckBackend(), None)
    resources = memory_and_priority()
    require(resources["available_memory_bytes"] >= config["resources"]["min_available_system_memory_bytes"],
            "可用内存不足。")
    cli = probe.cli_command()
    version, login = probe.run_command(cli + ["--version"]), probe.run_command(cli + ["login", "status"])
    require(version.get("exit_code") == 0 and login.get("exit_code") == 0
            and "Logged in using ChatGPT" in (login.get("stdout", "") + login.get("stderr", "")),
            "无法确认 Codex CLI 版本或 ChatGPT 登录。")
    run_dir = ROOT / "outputs/p1_control" / (datetime.now().strftime("%Y%m%d_%H%M%S")
                                               + "_" + uuid.uuid4().hex[:8] + "_resume")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("xb") as out:
        out.write((INTERRUPTED / "config.json").read_bytes())
    log = JsonLog(run_dir / "events.jsonl")
    runtime = ClassifiedTaskRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    controller, state = reconstruct(runtime, log)
    for path in [*(Path(x) for x in state["prior_hashes"]), INTERRUPTED / "events.jsonl",
                 INTERRUPTED / "config.json", Path(__file__).resolve()]:
        if str(path) not in runtime.protected_hashes:
            runtime.protected.append(path)
            runtime.protected_hashes[str(path)] = digest(path)
    budget = runtime.budget
    budget.model_normal = 211
    budget.eval_starts = budget.eval_normal = 191
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
              code_sha256=runtime.protected_hashes, selection_seed=config["task"]["selection_seed"],
              generations=6, continuation=True, normal_model_slots_used=211,
              physical_model_cli_invocations_used=212, extra_manual_invocations=1,
              evaluation_starts_used=191, evaluation_cpu_seconds_used=budget.cpu_finished)
    print(f"第 3 代恢复启动前检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(config, run_dir, log):
            log.event("resume_checkpoint", interrupted_run=str(INTERRUPTED), generation=3, next_stage="fusion_2",
                      failed_candidate_id="c0195", next_candidate_id="c0196", parent_ids=[x.candidate_id for x in state["proposals"]],
                      physical_model_calls_already_used=212, normal_model_slots_already_used=211,
                      evaluation_starts_already_used=191, total_model_call_cap=421)
            controller.batch([("fusion_2", state["proposals"], state["retry_prompt"])])
            selected = finish_after_round2(controller, state)
            selected = controller.run_remaining(selected, 3)
            require(len(selected) == 10 and controller.generation == 6, "续跑未完成全部六代。")
        print(f"续跑完成；记录：{run_dir}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"续跑停止：{exc}；记录：{run_dir}", file=sys.stderr, flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--simulate", action="store_true")
    actions.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.simulate:
        simulate()
        return 0
    if args.run:
        return resume()
    class CheckBackend:
        cfg = json.loads((INTERRUPTED / "config.json").read_text(encoding="utf-8"))
    controller, state = reconstruct(CheckBackend(), None)
    print(json.dumps({"status": "passed", "generation": controller.generation,
                      "candidates": len(controller.candidates), "generated": controller._generated,
                      "parents": [x.candidate_id for x in state["proposals"]], "next_candidate": "c0196",
                      "model_calls_used": 212, "evaluation_starts": len(state["cpu_samples"]),
                      "cpu_seconds_used": sum(state["cpu_samples"]), "total_model_call_cap": 421},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"恢复检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
