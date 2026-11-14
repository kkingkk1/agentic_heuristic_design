"""离线准备 RoCo 第 2 次研发的限流断点；只有服务恢复后才能执行。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
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
SOURCE = ROOT / "outputs/p1_roco/20261112_215400_819f82c2_resume"
SAVED = ROOT / "outputs/p1_manual_retry/roco_2_quota_saved_20261112/20261112_221401_cedfe544_evaluation"
sys.path.insert(0, str(ROOT))

from baselines.roco.paper_controller import RoCoController, ROLE_LABELS
from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate, p1_formal_matrix as matrix
from scripts.p0_controller import Candidate, Controller, StopRun, check_modification_scope, parse_round_critic
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from scripts.p1_classified_runtime import ClassifiedRoCoRuntime

spec = importlib.util.spec_from_file_location("roco_second_initial_recovery",
    ROOT / "outputs/p1_manual_retry/roco_2_initial_20261112_recovery/resume_initial.py")
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
require, read_jsonl = previous.require, previous.read_jsonl


class ContinueController(previous.RecoveryController):
    def __init__(self, *args, **kwargs):
        self._generation_offset = 0
        self._future_population = None
        self._future_initial_event = False
        super().__init__(*args, **kwargs)

    @property
    def generation(self):
        return self._generation

    @generation.setter
    def generation(self, value):
        self._generation = value + self._generation_offset if value else value

    def event(self, kind, **fields):
        if kind == "initial_population" and self._future_initial_event:
            self._future_initial_event = False
            return RoCoController.event(self, "resume_population", source_generation=1, **fields)
        return RoCoController.event(self, kind, **fields)

    def batch(self, specs):
        if self._future_population is not None:
            require(len(specs) == self.initial_count and all(stage.startswith("init_") and not parents
                    for stage, parents, _ in specs), "后续种群接入位置不符。")
            population, self._future_population = self._future_population, None
            self._future_initial_event = True
            return population
        return Controller.batch(self, specs)

    def run_future(self, population):
        require(len(population) == 10, "第 1 代种群不完整。")
        self._generation_offset = 1
        self._future_population = population
        self.generations = 5
        return RoCoController.run(self)


def reconstruct(backend, log):
    with patch.object(previous, "RecoveryController", ContinueController):
        controller, state = previous.reconstruct(backend, log)
    require((SOURCE / "config.json").read_bytes() == (previous.SOURCE / "config.json").read_bytes()
            == (SAVED / "config.json").read_bytes(), "恢复配置不一致。")
    events = read_jsonl(SOURCE / "events.jsonl")
    require([x["event_index"] for x in events] == list(range(1, len(events) + 1)), "事件序号不连续。")
    preflight = [x for x in events if x["event"] == "preflight"]
    starts = [x for x in events if x["event"] == "formal_run_started"]
    require(len(preflight) == len(starts) == 1 and starts[0]["resume_of"] == str(previous.SOURCE)
            and starts[0]["interrupted_events_sha256"] == digest(previous.SOURCE / "events.jsonl")
            and events[-1]["event"] == "preflight_or_run_stopped", "限流断点身份不符。")
    for name, expected in preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
    failures = [x for x in events if x["event"] == "model_failure_classified"]
    require({(x["stage"], x["category"]) for x in failures} == {
        ("mutation_A_integrator", "quota"), ("mutation_B_explorer", "quota")}
        and len([x for x in events if x["event"] == "model_attempt"]) == 77, "限流或调用数不符。")
    skipped = [x["candidate_id"] for x in events if x["event"] == "candidate_skipped"]
    require(skipped == ["c0087", "c0088"], "未启动变异槽位不符。")
    saved_events = read_jsonl(SAVED / "evaluation_events.jsonl")
    summary = json.loads((SAVED / "summary.json").read_text(encoding="utf-8"))
    require(summary["status"] == "completed" and summary["source_events_sha256"] == digest(SOURCE / "events.jsonl")
            and summary["new_model_calls"] == 0 and summary["new_evaluations"] == 2,
            "已返回变异的首次评价身份不符。")
    saved_preflight = [x for x in saved_events if x["event"] == "preflight"]
    require(len(saved_preflight) == 1, "首次评价启动检查不唯一。")
    for name, expected in saved_preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"首次评价受保护文件变化：{name}")
    candidates = {x.candidate_id: x for x in controller.candidates}
    require(len(candidates) == 30 and controller._generated == 15, "初始化候选起点不符。")
    registrations = [x for x in events if x["event"] == "candidate_registered"]
    require(len(registrations) == 58, "第 1 代候选注册数不符。")
    for row in registrations:
        ident, seq = row["candidate_id"], row["sequence"]
        require(ident == f"c{seq:04d}" and seq == len(controller.candidates) + 1, "候选编号不连续。")
        candidate = Candidate(ident, seq, row["stage"], tuple(row["parents"]))
        if ident in ("c0085", "c0086"):
            candidate.error = "模型服务返回 HTTP 429，未产生提案；不自动重试。"
        controller.candidates.append(candidate)
        candidates[ident] = candidate
    responses = {x["candidate_id"]: x for x in events if x["event"] == "proposal_response"}
    require(len(responses) == 68 and sorted(x["generation_order"] for x in responses.values()) == list(range(16, 84)),
            "生成顺序不连续。")
    for ident, row in responses.items():
        candidate = candidates[ident]
        parsed = controller.parse_proposal(row["response"])
        check_modification_scope(parsed["code"])
        candidate.thought, candidate.code = parsed["thought"], parsed["code"]
        candidate.generation_order = row["generation_order"]
        if ident in {x.candidate_id for x in state["pending"]}:
            require(row["prompt"] == state["initial_prompt"], f"初始化提示不符：{ident}")
    controller._generated = 83
    evaluated = [(x, SOURCE) for x in events if x["event"] == "evaluation"]
    evaluated += [(x, SAVED) for x in saved_events if x["event"] == "evaluation"]
    require(len(evaluated) == 71 and {x["candidate_id"] for x, _ in evaluated}.issuperset({"c0083", "c0084"}),
            "首次评价记录不完整。")
    for row, origin in evaluated:
        ident, result = row["candidate_id"], row["result"]
        candidate = candidates[ident]
        directory = origin / "candidates" / ident
        require(json.loads((directory / "result.json").read_text(encoding="utf-8")) == result
                and (directory / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8") == candidate.code + "\n",
                f"候选或评价记录不一致：{ident}")
        if result["status"] != "passed":
            candidate.error = str(result.get("error") or result["status"])
            continue
        checked = p0_tsp_aco_evaluate.read_complete(directory / "stdout.txt")
        require(checked["average"] == result["average"] and checked["fitness"] == result["fitness"]
                and checked["objectives"] == result["objectives"], f"评价输出不一致：{ident}")
        candidate.objective, candidate.fitness = float(result["average"]), float(result["fitness"])
        candidate.completed_order, candidate.evaluation_origin = result["completed_monotonic_ns"], ident
    require(max(c.completed_order for c in controller.candidates if c.valid) < time.monotonic_ns(),
            "当前单调时钟早于历史评价。")
    initial = controller.select(controller.candidates[:30], controller.n)
    initial_events = [x for x in events if x["event"] == "initial_population"]
    require(len(initial_events) == 1 and [x.candidate_id for x in initial] == initial_events[0]["candidate_ids"],
            "初始种群筛选不一致。")
    for row in registrations[:40]:
        op = row["stage"].split("_")[1]
        parents = controller.eoh_parents(initial, 5 if op in ("E1", "E2") else 1)
        require([x.candidate_id for x in parents] == row["parents"], f"第 1 代随机父代不符：{row['candidate_id']}")
    a, b = controller.elite_pair(initial)
    pair = [x for x in events if x["event"] == "elite_pair"]
    require(len(pair) == 1 and (a.candidate_id, b.candidate_id) == (pair[0]["a"], pair[0]["b"]), "双精英不符。")
    controller.generation = 1
    role_rows = {x["stage"]: x for x in events if x["event"] == "model_response"}
    require(set(role_rows) == {"critic_initial", "critic_1", "critic_2", "critic_3", "reflection_explorer",
                               "reflection_exploiter", "reflection_integrator"}
            and role_rows["critic_initial"]["prompt"] == controller.initial_critic_prompt(a, b), "角色响应不完整。")
    critic = probe.parse_critic(role_rows["critic_initial"]["response"])
    views = controller.critic_views(critic)
    reflections = controller.fusion_reflections(critic)
    role_reflections = {role: [views[role]["reflection"]] for role in controller.design_roles}
    formal = {role: [] for role in controller.design_roles}
    changes = {role: [] for role in controller.design_roles}
    current = {}
    transitions = {(x["role"], x["round"]): x for x in events if x["event"] == "branch_transition"}
    require(len(transitions) == 8, "分支转移不完整。")

    def restore_round(turn, feedback):
        before = {role: b if turn == 0 else current[role] for role in controller.design_roles}
        proposals = []
        for role in controller.design_roles:
            row = transitions[(ROLE_LABELS[role], turn)]
            candidate = candidates[row["proposal"]]
            stage = f"start_{role}" if turn == 0 else f"round_{turn}_{role}"
            require(candidate.stage == stage and candidate.parents == (before[role].candidate_id,)
                    and row["previous"] == before[role].candidate_id
                    and responses[candidate.candidate_id]["prompt"] == controller.branch_prompt(role, before[role], feedback[role]),
                    f"分支输入不符：{stage}")
            current[role] = candidate if candidate.valid else before[role]
            if turn and candidate.valid:
                formal[role].append(candidate)
            change = {"from_objective": before[role].objective, "proposal_objective": candidate.objective,
                      "to_objective": current[role].objective}
            require(row["current"] == current[role].candidate_id and all(row[k] == v for k, v in change.items()),
                    f"分支转移不符：{stage}")
            changes[role].append(change)
            proposals.append(candidate)
        return proposals

    restore_round(0, views)
    last = None
    for turn in (1, 2, 3):
        row = role_rows[f"critic_{turn}"]
        require(row["prompt"] == controller.branch_critic_prompt(*(current[r] for r in controller.design_roles)),
                f"第 {turn} 轮 Critic 提示不符。")
        critic = parse_round_critic(row["response"])
        views = controller.critic_views(critic)
        reflections.extend(controller.fusion_reflections(critic))
        for role in controller.design_roles:
            role_reflections[role].append(views[role]["reflection"])
        proposals = restore_round(turn, views)
        require(all(x.valid for x in proposals), f"第 {turn} 轮融合输入无效。")
        fusion_row = [x for x in responses.values() if x["stage"] == f"fusion_{turn}"]
        require(len(fusion_row) == 1 and fusion_row[0]["prompt"] == controller.round_fusion_prompt(proposals, "Lower"),
                f"第 {turn} 轮融合提示不符。")
        last = [*proposals, candidates[fusion_row[0]["candidate_id"]]]
    best = [controller.select(formal[role], 1)[0] for role in controller.design_roles]
    final = [x for x in responses.values() if x["stage"] == "fusion_final"]
    require(len(final) == 1 and final[0]["prompt"] == controller.final_fusion_prompt(best, reflections), "末轮融合提示不符。")
    final_candidate = candidates[final[0]["candidate_id"]]
    lessons = {}
    for role in controller.design_roles:
        row = role_rows[f"reflection_{role}"]
        require(row["prompt"] == controller.long_reflection_prompt(role, role_reflections[role], changes),
                f"角色反思提示不符：{role}")
        lessons[role] = probe.parse_long_term(row["response"])
    row = role_rows["reflection_integrator"]
    require(row["prompt"] == controller.merge_reflections_prompt(lessons), "融合反思提示不符。")
    lessons[controller.integrator_role] = probe.parse_merge(row["response"])
    all_roles = (*controller.design_roles, controller.integrator_role)
    specs = []
    for elite_name, elite in (("A", a), ("B", b)):
        for role in all_roles:
            prompt = (f"{controller.task}\n[History reflection as reference]:\n{' | '.join(controller.history[role])}\n"
                      f"[Reflection insights]:\n{lessons[role]}\n[Current best code]:\n{elite.code}\n"
                      "Based on the history reflections and mainly based on reflection insights and "
                      "the current best code, propose a mutation that addresses the key insights "
                      f"and improves upon the previous solutions.\n{controller.output}")
            specs.append((f"mutation_{elite_name}_{role}", (elite,), prompt))
    mutations = [candidates[x["candidate_id"]] for x in registrations[-6:]]
    attempts = {x["stage"]: x for x in events if x["event"] == "model_attempt" and x["stage"].startswith("mutation_")}
    require(len(attempts) == 4, "变异模型调用数量不符。")
    for candidate, (stage, parents, prompt) in zip(mutations, specs):
        require(candidate.stage == stage and candidate.parents == tuple(x.candidate_id for x in parents),
                f"变异父代不符：{stage}")
        if stage in attempts:
            require(attempts[stage]["prompt"] == prompt, f"变异提示不符：{stage}")
    cpu = [x["accounting"]["cpu_seconds"] for x in (*events, *saved_events) if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == 71 and len(state["cpu_samples"]) == 12, "累计评价 CPU 不符。")
    return controller, {"config": state["config"], "eoh": [candidates[x["candidate_id"]] for x in registrations[:40]],
                        "last": last, "elite_fusion": final_candidate, "lessons": lessons,
                        "mutations": mutations, "pending_specs": specs[-2:], "cpu_samples": state["cpu_samples"] + cpu,
                        "prior_hashes": preflight[0]["code_sha256"], "saved_hashes": saved_preflight[0]["code_sha256"]}


def finish_generation(controller, state):
    candidates = state["mutations"][-2:]
    for candidate, (stage, parents, _) in zip(candidates, state["pending_specs"]):
        require(candidate.stage == stage and candidate.parents == tuple(x.candidate_id for x in parents), "待执行变异不符。")
        controller.event("reused_candidate_registration", generation=1, candidate_id=candidate.candidate_id,
                         stage=stage, parents=list(candidate.parents), source_events=str(SOURCE / "events.jsonl"))
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(controller.propose, candidate, spec[2])
                   for candidate, spec in zip(candidates, state["pending_specs"])]
        failures = []
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                controller.stop.set()
                failures.append(exc)
        if failures:
            raise StopRun(f"并发阶段停止：{failures[0]}") from failures[0]
    pool = [*state["eoh"], *state["last"], state["elite_fusion"], *state["mutations"]]
    selected = controller.select(pool, controller.n)
    controller.event("selection", generation=1, pool=[x.candidate_id for x in pool if x is not None],
                     selected=[{"candidate_id": x.candidate_id, "evaluation_origin": x.evaluation_origin} for x in selected])
    for role in (*controller.design_roles, controller.integrator_role):
        controller.history[role].append(state["lessons"][role])
        controller.event("history_appended", role=role, generation=1, reflection=state["lessons"][role])
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
        for path in (ROOT / "outputs/p1_roco").glob("*/config.json"):
            if path.parent.resolve() == run_dir.resolve():
                continue
            other = json.loads(path.read_text(encoding="utf-8"))
            if (other.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                    or matrix.run_identity(other) != matrix.run_identity(config)):
                continue
            event_path = path.parent / "events.jsonl"
            if event_path.is_file() and any(x["event"] == "formal_run_started" for x in read_jsonl(event_path)):
                started.add(path.parent.resolve())
        require(started == {previous.SOURCE.resolve(), SOURCE.resolve()}, "RoCo 第 2 次研发存在未知已启动运行。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=config["formal_replicate"], selection_seed=task["selection_seed"],
                  resume_of=str(SOURCE), interrupted_events_sha256=digest(SOURCE / "events.jsonl"),
                  saved_evaluation_sha256=digest(SAVED / "evaluation_events.jsonl"), total_model_call_cap=420,
                  note="服务恢复后继续未启动槽位；保留 429 失败及已完成的首次评价。")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def simulate():
    from tests.test_p0_controller import FixtureBackend, MemoryLog
    backend, log = FixtureBackend(), MemoryLog()
    controller, state = reconstruct(backend, log)
    with recovery_guard(state["config"], RECOVERY / "unused_guard_fixture", MemoryLog()):
        code = ("from pathlib import Path\nfrom scripts import p1_formal_matrix as m\ntry:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED", "续跑互斥锁不一致。")
    selected = finish_generation(controller, state)
    require(len(backend.calls) == len(backend.evaluations) == 2
            and all(stage not in ("mutation_A_integrator", "mutation_B_explorer") for stage, _ in backend.calls),
            "未启动变异或限流失败处理不符。")
    selected = controller.run_future(selected)
    require(len(selected) == 10 and controller.generation == 6 and len(backend.calls) == 327
            and len(backend.evaluations) == 292
            and [x["generation"] for x in log.items if x["event"] == "selection"] == list(range(1, 7)),
            "后续代数、预算槽位或筛选不符。")
    print(json.dumps({"status": "passed", "prior_model_calls": 93, "prior_evaluations": 83,
                      "unstarted_mutations": 2, "later_model_calls": 327, "later_evaluations": 292,
                      "model_call_cap": 420, "formal_mutex_shared": True}, ensure_ascii=False))


def resume():
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
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
    run_dir = ROOT / "outputs/p1_roco" / (datetime.now().strftime("%Y%m%d_%H%M%S")
                                            + "_" + uuid.uuid4().hex[:8] + "_resume")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("xb") as out:
        out.write((SOURCE / "config.json").read_bytes())
    log = JsonLog(run_dir / "events.jsonl")
    runtime = ClassifiedRoCoRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    controller, state = reconstruct(runtime, log)
    paths = [*(Path(x) for x in state["prior_hashes"]), *(Path(x) for x in state["saved_hashes"]),
             SOURCE / "events.jsonl", SOURCE / "config.json", SAVED / "evaluation_events.jsonl",
             SAVED / "summary.json", Path(previous.__file__).resolve(), Path(__file__).resolve()]
    for ident in ("c0083", "c0084"):
        paths.extend(SAVED / "candidates" / ident / item for item in ("result.json", "stdout.txt", "problems/tsp_aco/gpt.py"))
    for row in read_jsonl(SOURCE / "events.jsonl"):
        if row["event"] != "evaluation":
            continue
        directory = SOURCE / "candidates" / row["candidate_id"]
        paths.extend(directory / item for item in ("result.json", "problems/tsp_aco/gpt.py"))
        if (directory / "stdout.txt").is_file():
            paths.append(directory / "stdout.txt")
    for path in paths:
        if str(path) not in runtime.protected_hashes:
            runtime.protected.append(path)
            runtime.protected_hashes[str(path)] = digest(path)
    budget = runtime.budget
    budget.model_normal = 93
    budget.eval_starts = budget.eval_normal = 83
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
              code_sha256=runtime.protected_hashes, selection_seed=config["task"]["selection_seed"],
              generations=6, continuation=True, normal_model_slots_used=93,
              physical_model_cli_invocations_used=93, evaluation_starts_used=83,
              evaluation_cpu_seconds_used=budget.cpu_finished)
    print(f"RoCo 第 2 次研发限流断点恢复检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(config, run_dir, log):
            log.event("resume_checkpoint", interrupted_run=str(SOURCE), generation=1,
                      failed_candidate_ids=["c0085", "c0086"], saved_evaluation_run=str(SAVED),
                      unstarted_candidates=[x.candidate_id for x in state["mutations"][-2:]],
                      model_calls_already_used=93, evaluation_starts_already_used=83, total_model_call_cap=420)
            selected = finish_generation(controller, state)
            selected = controller.run_future(selected)
            require(len(selected) == 10 and controller.generation == 6, "RoCo 第 2 次研发未完成六代。")
        print(f"RoCo 第 2 次研发六代完成；记录：{run_dir}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"RoCo 第 2 次研发续跑停止：{exc}；记录：{run_dir}", file=sys.stderr, flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--simulate", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--service-restored", action="store_true", help="仅在限流解除后启用模型调用")
    args = parser.parse_args()
    if args.simulate:
        simulate()
        return 0
    if args.run:
        require(args.service_restored, "服务曾返回 HTTP 429；限流解除前不能继续模型调用。")
        return resume()
    class CheckBackend:
        cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    controller, state = reconstruct(CheckBackend(), None)
    print(json.dumps({"status": "passed", "generation": 1, "candidates": len(controller.candidates),
                      "model_calls_used": 93, "evaluation_starts": len(state["cpu_samples"]),
                      "unstarted_candidates": [x.candidate_id for x in state["mutations"][-2:]],
                      "model_call_cap": 420, "waiting_for_service_recovery": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"RoCo 第 2 次研发限流断点检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
