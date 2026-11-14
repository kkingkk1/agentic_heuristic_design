"""核对并恢复本批次第 1 次研发的第 1 代断点。"""

import argparse
from collections import Counter
from contextlib import contextmanager
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


RETRY = Path(__file__).resolve().parent
ROOT = RETRY.parents[2]
SOURCE = ROOT / "outputs/p1_control/20261112_135816_854bdc72"
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate, p1_formal_matrix as matrix
from scripts.p0_controller import Candidate, Controller, StopRun, check_modification_scope, parse_split_critic, timing_from_result
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from scripts.p1_runtime import TaskRuntime
from scripts.p1_task_contract import load_task_contract


class ResumeController(Controller):
    """复用原控制器执行后续整代，只将已有种群接入 run 的初始化位置。"""

    def __init__(self, *args, **kwargs):
        self._generation_offset = 0
        self._resume_population = None
        self._resume_initial_event = False
        super().__init__(*args, **kwargs)

    @property
    def generation(self):
        return self._generation

    @generation.setter
    def generation(self, value):
        self._generation = value + self._generation_offset if value else value

    def batch(self, specs):
        if self._resume_population is not None:
            require(len(specs) == self.initial_count
                    and all(stage.startswith("init_") and not parents for stage, parents, _ in specs),
                    "续跑初始化位置与原控制器不符。")
            population, self._resume_population = self._resume_population, None
            self._resume_initial_event = True
            return population
        return super().batch(specs)

    def event(self, kind, **fields):
        if kind == "initial_population" and self._resume_initial_event:
            self._resume_initial_event = False
            return super().event("resume_population", source_generation=1, **fields)
        return super().event(kind, **fields)

    def run_remaining(self, population, generations):
        require(self.generation == 1 and type(generations) is int and generations > 0,
                "续跑起点或剩余代数无效。")
        self._generation_offset = 1
        self._resume_population = population
        self.generations = generations
        return super().run()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def reconstruct(backend, log):
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    matrix.verify_training_config(config)
    source_events = read_jsonl(SOURCE / "events.jsonl")
    retry_events = read_jsonl(RETRY / "events.jsonl")
    evaluation_events = read_jsonl(RETRY / "evaluation_events.jsonl")
    retry_summary = json.loads((RETRY / "summary.json").read_text(encoding="utf-8"))
    require(retry_summary["status"] == "proposal_parsed"
            and retry_summary["source_events_sha256"] == digest(SOURCE / "events.jsonl"),
            "重试来源或响应状态不符。")
    require((SOURCE / "config.json").read_bytes() == (RETRY / "config.json").read_bytes(),
            "重试评价配置与原配置不一致。")
    require([x["event_index"] for x in source_events] == list(range(1, len(source_events) + 1)),
            "原运行事件序号不连续。")
    preflight = [x for x in source_events if x["event"] == "preflight"]
    require(len(preflight) == 1, "原运行启动检查不唯一。")
    for name, expected in preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"原运行受保护文件发生变化：{name}")
    require(source_events[-1]["event"] == "preflight_or_run_stopped"
            and any(x["event"] == "finish" and x.get("status") == "stopped" for x in source_events),
            "原运行不是已停止状态。")
    attempts = [x for x in source_events if x["event"] == "model_attempt"]
    usages = [x for x in source_events if x["event"] == "model_usage"]
    require(len(attempts) == 81 and len(usages) == 80 and attempts[-1]["stage"] == "fusion_2",
            "原运行模型调用断点不符。")
    require(len([x for x in retry_events if x["event"] == "model_attempt"]) == 1
            and len([x for x in retry_events if x["event"] == "model_usage"]) == 1,
            "手动重试不是唯一成功调用。")
    records = [x for x in source_events if x["event"] == "candidate_registered"]
    require(len(records) == 78, "原运行候选数量不符。")
    config_task = config["task"]
    require((config_task["initial_proposals"], config_task["population_size"], config_task["generations"],
             config_task["collaboration_rounds"]) == (30, 10, 6, 3), "原运行规模不符。")
    contract = load_task_contract("tsp_aco")
    controller = ResumeController(backend, log, population_size=10, initial_count=30, rounds=3,
                                  seed=config_task["selection_seed"], generations=6, task_contract=contract)
    controller.generation = 1
    candidates = {}
    for event in records:
        ident, sequence = event["candidate_id"], event["sequence"]
        require(ident == f"c{sequence:04d}" and sequence == len(candidates) + 1,
                "原运行候选身份或顺序不符。")
        candidate = Candidate(ident, sequence, event["stage"], tuple(event["parents"]))
        controller.candidates.append(candidate)
        candidates[ident] = candidate
    invalid = {x["candidate_id"]: x["error"] for x in source_events if x["event"] == "candidate_invalid"}
    for event in (x for x in source_events if x["event"] == "proposal_response"):
        candidate = candidates[event["candidate_id"]]
        candidate.generation_order = event["generation_order"]
        try:
            parsed = contract.parse_proposal(event["response"])
            candidate.thought, candidate.code = parsed["thought"], parsed["code"]
            check_modification_scope(candidate.code)
        except (probe.ProbeError, ValueError, SyntaxError):
            require(candidate.candidate_id in invalid, "原提案解析与失败记录不符。")
            candidate.error = invalid[candidate.candidate_id]
    for event in (x for x in source_events if x["event"] == "evaluation"):
        candidate, result = candidates[event["candidate_id"]], event["result"]
        directory = SOURCE / "candidates" / candidate.candidate_id
        require(json.loads((directory / "result.json").read_text(encoding="utf-8")) == result,
                f"原评价结果不一致：{candidate.candidate_id}")
        require((directory / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8") == candidate.code + "\n",
                f"原候选代码不一致：{candidate.candidate_id}")
        if result["status"] != "passed":
            candidate.error = str(result.get("error") or result["status"])
            continue
        checked = p0_tsp_aco_evaluate.read_complete(directory / "stdout.txt")
        require(checked["average"] == result["average"] and checked["fitness"] == result["fitness"]
                and checked["objectives"] == result["objectives"], f"原评价输出不一致：{candidate.candidate_id}")
        candidate.objective = candidate.raw_objective = float(result["average"])
        candidate.fitness = float(result["fitness"])
        candidate.timing_seconds, candidate.timing_scope = timing_from_result(result)
        candidate.completed_order = result["completed_monotonic_ns"]
        candidate.evaluation_origin = candidate.candidate_id
    original_failed = candidates["c0078"]
    require(original_failed.stage == "fusion_2" and original_failed.parents == ("c0076", "c0077")
            and not original_failed.valid and not original_failed.code, "原失败候选不符。")
    original_failed.error = "模型接口拒绝，未产生提案；用户授权手动重试一次。"
    new_registration = [x for x in evaluation_events if x["event"] == "candidate_registered"]
    new_response = [x for x in retry_events if x["event"] == "proposal_response"]
    new_evaluation = [x for x in evaluation_events if x["event"] == "evaluation"]
    require(len(new_registration) == len(new_response) == len(new_evaluation) == 1,
            "手动重试候选或评价记录不唯一。")
    require(new_registration[0]["candidate_id"] == "c0079" and new_evaluation[0]["candidate_id"] == "c0079",
            "手动重试候选身份不符。")
    parsed = contract.parse_proposal(new_response[0]["response"])
    check_modification_scope(parsed["code"])
    candidate = Candidate("c0079", 79, "fusion_2_manual_retry", ("c0076", "c0077"),
                          thought=parsed["thought"], code=parsed["code"], generation_order=78)
    result = new_evaluation[0]["result"]
    candidate_dir = RETRY / "candidates/c0079"
    require(result["status"] == "passed"
            and json.loads((candidate_dir / "result.json").read_text(encoding="utf-8")) == result
            and (candidate_dir / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8") == candidate.code + "\n",
            "重试候选或评价结果不一致。")
    checked = p0_tsp_aco_evaluate.read_complete(candidate_dir / "stdout.txt")
    require(checked["average"] == result["average"] and checked["fitness"] == result["fitness"]
            and checked["objectives"] == result["objectives"], "重试评价输出不一致。")
    candidate.objective = candidate.raw_objective = float(result["average"])
    candidate.fitness = float(result["fitness"])
    candidate.timing_seconds, candidate.timing_scope = timing_from_result(result)
    candidate.completed_order = result["completed_monotonic_ns"]
    candidate.evaluation_origin = candidate.candidate_id
    controller.candidates.append(candidate)
    candidates[candidate.candidate_id] = candidate
    controller._generated = 78
    require(max(x.completed_order for x in controller.candidates if x.valid) < time.monotonic_ns(),
            "当前单调时钟早于原评价记录，不能重建评价先后。")
    initial = controller.select(controller.candidates[:30], controller.n)
    initial_event = next(x for x in source_events if x["event"] == "initial_population")
    require([x.candidate_id for x in initial] == initial_event["candidate_ids"], "初始种群重建不一致。")
    eoh = [candidates[x["candidate_id"]] for x in records if x["stage"].startswith("eoh_")]
    require(len(eoh) == 40, "EoH 候选数量不符。")
    for proposal in eoh:
        operator = proposal.stage.split("_")[1]
        reconstructed = controller.eoh_parents(initial, 5 if operator in ("E1", "E2") else 1)
        require(tuple(x.candidate_id for x in reconstructed) == proposal.parents,
                f"选择种子重建失败：{proposal.stage}")
    a, b = controller.elite_pair(initial)
    pair = next(x for x in source_events if x["event"] == "elite_pair")
    require((a.candidate_id, b.candidate_id) == (pair["a"], pair["b"]), "双精英重建不一致。")
    critics = [x for x in source_events if x["event"] == "model_response"]
    require([x["stage"] for x in critics] == ["critic_initial", "critic_1", "critic_2"],
            "Critic 记录与断点不一致。")
    role_reflections = {role: [] for role in controller.design_roles}
    reflections = []
    for event in critics:
        critic = parse_split_critic(event["response"])
        views = controller.critic_views(critic)
        reflections.extend(controller.fusion_reflections(critic))
        for role in controller.design_roles:
            role_reflections[role].append(views[role]["reflection"])
    transitions = [x for x in source_events if x["event"] == "branch_transition"]
    changes, formal, current = {}, {}, {}
    for role in controller.design_roles:
        rows = [x for x in transitions if x["role"] == role]
        require([x["round"] for x in rows] == [0, 1, 2], "分支转移记录不完整。")
        changes[role] = [{k: x[k] for k in ("from_objective", "proposal_objective", "to_objective")} for x in rows]
        formal[role] = [candidates[x["proposal"]] for x in rows[1:] if candidates[x["proposal"]].valid]
        current[role] = b if rows[-1]["current"] == b.candidate_id else candidates[rows[-1]["current"]]
    previous_inputs = [candidates["c0076"], candidates["c0077"]]
    require(all(x.valid for x in previous_inputs)
            and controller.round_fusion_prompt(previous_inputs, "Lower") == attempts[-1]["prompt"],
            "重建的融合输入或提示与原记录不一致。")
    require(not any(x["event"] in ("selection", "history_appended") for x in source_events),
            "原运行已发生不预期的代末筛选或历史写入。")
    cpu = [x["accounting"]["cpu_seconds"] for x in source_events if x["event"] == "evaluation_cpu_final"]
    new_cpu = [x["accounting"]["cpu_seconds"] for x in evaluation_events if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == 76 and len(new_cpu) == 1, "CPU 成本记录不完整。")
    return controller, {"config": config, "a": a, "b": b, "eoh": eoh, "formal": formal, "current": current,
                        "role_reflections": role_reflections, "reflections": reflections, "changes": changes,
                        "retry_candidate": candidate, "cpu_samples": cpu + new_cpu}


def finish_first_generation(controller, state):
    """从已评价的第 2 轮融合之后继续原控制器的第 3 轮及代末。"""
    require(controller.generation == 1 and state["retry_candidate"].valid, "第 1 代恢复状态无效。")
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
    previous = current.copy()
    proposals = controller.batch([(f"round_3_{role}", (previous[role],),
                                   controller.branch_prompt(role, previous[role], views[role]))
                                  for role in controller.design_roles])
    for role, proposal in zip(controller.design_roles, proposals):
        if proposal.valid:
            formal[role].append(proposal)
            current[role] = proposal
        change = {"from_objective": previous[role].objective, "proposal_objective": proposal.objective,
                  "to_objective": current[role].objective}
        changes[role].append(change)
        controller.event("branch_transition", generation=1, role=role, round=3,
                         previous=previous[role].candidate_id, proposal=proposal.candidate_id,
                         current=current[role].candidate_id, **change)
    fusion = None
    if all(proposal.valid for proposal in proposals):
        better = "Lower" if controller.direction == "min" else "Higher"
        fusion = controller.batch([("fusion_3", proposals, controller.round_fusion_prompt(proposals, better))])[0]
    else:
        controller.event("slot_skipped", generation=1, stage="fusion_3", reason="invalid_new_input")
    last = [*proposals, fusion]
    elite_fusion = None
    if all(formal[role] for role in controller.design_roles):
        best = [controller.select(formal[role], 1)[0] for role in controller.design_roles]
        elite_fusion = controller.batch([("fusion_final", best,
                                          controller.final_fusion_prompt(best, reflections))])[0]
    else:
        controller.event("slot_skipped", generation=1, stage="fusion_final", reason="missing_formal_branch")
    lessons = {}
    for role in controller.design_roles:
        prompt = controller.long_reflection_prompt(role, role_reflections[role], changes)
        lessons[role] = controller.request(f"reflection_{role}", prompt, probe.parse_long_term)
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
    pool = [*state["eoh"], *last, elite_fusion, *mutations]
    selected = controller.select(pool, controller.n)
    controller.event("selection", generation=1, pool=[x.candidate_id for x in pool if x is not None],
                     selected=[{"candidate_id": x.candidate_id, "evaluation_origin": x.evaluation_origin}
                               for x in selected])
    for role in all_roles:
        controller.history[role].append(lessons[role])
        controller.event("history_appended", role=role, generation=1, reflection=lessons[role])
    return selected


def simulate():
    from tests.test_p0_controller import FixtureBackend, MemoryLog

    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    guard_log = MemoryLog()
    with continuation_guard(config, RETRY / "unused_guard_fixture", guard_log):
        code = ("from pathlib import Path\n"
                "from scripts import p1_formal_matrix as m\n"
                "try:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n"
                " print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED",
                "续跑与正式研发没有共用互斥锁。")
    require(len(guard_log.items) == 1 and guard_log.items[0]["event"] == "formal_run_started"
            and guard_log.items[0]["resume_of"] == str(SOURCE), "续跑身份记录不完整。")
    baseline_backend, baseline_log = FixtureBackend(), MemoryLog()
    baseline = Controller(baseline_backend, baseline_log, generations=2, seed=2026110601)
    baseline_selected = baseline.run()
    split_backend, split_log = FixtureBackend(), MemoryLog()
    split = ResumeController(split_backend, split_log, generations=1, seed=2026110601)
    split_first = split.run()
    split_selected = split.run_remaining(split_first, 1)
    def parents(log):
        return [(x["stage"], x["parents"]) for x in log.items
                if x["event"] == "candidate_registered" and x.get("generation") == 2
                and x["stage"].startswith("eoh_")]
    require(parents(baseline_log) == parents(split_log)
            and [x.candidate_id for x in baseline_selected] == [x.candidate_id for x in split_selected],
            "拆分运行的随机父代或最终筛选与原控制器不一致。")
    backend, log = FixtureBackend(), MemoryLog()
    controller, state = reconstruct(backend, log)
    first = finish_first_generation(controller, state)
    require((len(backend.calls), len(backend.evaluations), len(first)) == (14, 10, 10),
            "第 1 代续跑槽位与原流程不一致。")
    require(len([x for x in log.items if x["event"] == "selection" and x["generation"] == 1]) == 1,
            "第 1 代筛选记录不唯一。")
    selected = controller.run_remaining(first, 1)
    require((len(backend.calls), len(backend.evaluations), len(selected)) == (79, 68, 10),
            "下一完整代槽位与原流程不一致。")
    require([x["generation"] for x in log.items if x["event"] == "selection"] == [1, 2],
            "续跑的代数或筛选时点不一致。")
    require(all(len(controller.history[role]) == 2 for role in controller.history), "角色历史追加不一致。")
    print(json.dumps({"status": "passed", "first_generation_calls": 14, "first_generation_evaluations": 10,
                      "second_generation_calls": 65, "second_generation_evaluations": 58,
                      "split_run_matches_original": True, "formal_mutex_shared": True}, ensure_ascii=False))


@contextmanager
def continuation_guard(config, run_dir, log):
    """取得正式研发互斥锁，只允许恢复唯一的原始失败运行。"""
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
        started = []
        for path in (ROOT / "outputs/p1_control").glob("*/config.json"):
            if path.parent.resolve() == run_dir.resolve():
                continue
            previous = json.loads(path.read_text(encoding="utf-8"))
            if (previous.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                    or matrix.run_identity(previous) != matrix.run_identity(config)):
                continue
            events_path = path.parent / "events.jsonl"
            if events_path.is_file() and any(x["event"] == "formal_run_started" for x in read_jsonl(events_path)):
                started.append(path.parent.resolve())
        require(started == [SOURCE.resolve()], "该重复索引存在其他已启动运行，禁止重复续跑。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"],
                  method=config["method"]["id"], formal_replicate=config["formal_replicate"],
                  selection_seed=task["selection_seed"], resume_of=str(SOURCE),
                  original_events_sha256=digest(SOURCE / "events.jsonl"),
                  manual_retry_events_sha256=digest(RETRY / "events.jsonl"),
                  manual_retry_evaluation_sha256=digest(RETRY / "evaluation_events.jsonl"),
                  extra_model_cli_invocations=1, extra_model_reason="用户授权手动重试已失败的融合请求一次")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def resume():
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    matrix.verify_training_software(matrix.load_matrix()[0])
    class CheckBackend:
        cfg = config
    reconstruct(CheckBackend(), None)
    resources = memory_and_priority()
    require(resources["available_memory_bytes"] >= config["resources"]["min_available_system_memory_bytes"],
            "可用内存低于冻结阈值。")
    cli = probe.cli_command()
    version, login = probe.run_command(cli + ["--version"]), probe.run_command(cli + ["login", "status"])
    require(version.get("exit_code") == 0 and login.get("exit_code") == 0
            and "Logged in using ChatGPT" in (login.get("stdout", "") + login.get("stderr", "")),
            "无法确认 Codex CLI 版本或 ChatGPT 登录。")
    run_dir = ROOT / "outputs/p1_control" / (datetime.now().strftime("%Y%m%d_%H%M%S")
                                               + "_" + uuid.uuid4().hex[:8] + "_resume")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("xb") as stream:
        stream.write((SOURCE / "config.json").read_bytes())
    log = JsonLog(run_dir / "events.jsonl")
    runtime = TaskRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    extra_sources = [SOURCE / "events.jsonl", SOURCE / "config.json", RETRY / "events.jsonl",
                     RETRY / "evaluation_events.jsonl", RETRY / "summary.json",
                     RETRY / "candidates/c0079/result.json", RETRY / "candidates/c0079/stdout.txt",
                     RETRY / "candidates/c0079/problems/tsp_aco/gpt.py", Path(__file__).resolve()]
    for path in extra_sources:
        runtime.protected.append(path)
        runtime.protected_hashes[str(path)] = digest(path)
    controller, state = reconstruct(runtime, log)
    budget = runtime.budget
    budget.model_normal = 81
    budget.eval_starts = budget.eval_normal = 77
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
              code_sha256=runtime.protected_hashes, selection_seed=config["task"]["selection_seed"],
              generations=6, continuation=True, logical_model_calls_used=81,
              physical_model_cli_invocations_used=82, extra_manual_invocations=1,
              evaluation_starts_used=77, evaluation_cpu_seconds_used=budget.cpu_finished)
    print(f"续跑启动前检查通过；记录：{run_dir}", flush=True)
    try:
        with continuation_guard(config, run_dir, log):
            log.event("resume_checkpoint", source_run=str(SOURCE), retry_run=str(RETRY),
                      model_cli_invocations_already_used=82, normal_model_slots_already_used=81,
                      evaluation_starts_already_used=77, generation=1, next_stage="critic_3",
                      initial_candidate_ids=[x.candidate_id for x in controller.candidates[:30]],
                      eoh_candidate_ids=[x.candidate_id for x in state["eoh"]],
                      retry_candidate_id=state["retry_candidate"].candidate_id)
            first = finish_first_generation(controller, state)
            selected = controller.run_remaining(first, 5)
            require(len(selected) == 10 and controller.generation == 6, "续跑未完成全部六代。")
        print(f"续跑完成；记录：{run_dir}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"续跑停止：{exc}；记录：{run_dir}", file=sys.stderr, flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="只重建断点，不调用模型。")
    action.add_argument("--simulate", action="store_true", help="使用离线夹具核对续跑流程。")
    action.add_argument("--run", action="store_true", help="在互斥与来源核验后恢复正式研发。")
    args = parser.parse_args()
    if args.simulate:
        simulate()
        return 0
    if args.run:
        return resume()
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    class CheckBackend:
        cfg = config
    controller, state = reconstruct(CheckBackend(), None)
    print(json.dumps({"status": "passed", "candidates": len(controller.candidates),
                      "generated": controller._generated,
                      "elite_pair": [state["a"].candidate_id, state["b"].candidate_id],
                      "resumed_fusion": state["retry_candidate"].candidate_id,
                      "evaluation_starts": len(state["cpu_samples"]),
                      "evaluation_cpu_seconds": sum(state["cpu_samples"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
