"""从 RoCo 第 2 代末轮融合连接失败处手动恢复，保留原模型预算。"""

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
SOURCE = ROOT / "outputs/p1_roco/20261112_181554_cd08d9e6_resume"
sys.path.insert(0, str(ROOT))

from baselines.roco.paper_controller import RoCoController, ROLE_LABELS
from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate, p1_formal_matrix as matrix
from scripts.p0_controller import Candidate, Controller, StopRun, check_modification_scope, parse_round_critic
from scripts.p0_runtime import JsonLog, digest, memory_and_priority

spec = importlib.util.spec_from_file_location("roco_before_fusion_final", RECOVERY / "resume_after_181257.py")
prior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prior
spec.loader.exec_module(prior)
require, read_jsonl = prior.require, prior.read_jsonl
original_recovery = prior.prior.prior.prior.prior.prior


class ContinueController(original_recovery.RecoveryController):
    def __init__(self, *args, **kwargs):
        self._future_population = None
        super().__init__(*args, **kwargs)

    def batch(self, specs):
        if self._future_population is not None:
            require(len(specs) == self.initial_count and all(stage.startswith("init_") and not parents
                    for stage, parents, _ in specs), "后续种群接入位置不符。")
            population, self._future_population = self._future_population, None
            return population
        return Controller.batch(self, specs)

    def run_future(self, population):
        require(len(population) == 10, "第 2 代种群不完整。")
        self._generation_offset = 2
        self._future_population = population
        self.generations = 4
        return RoCoController.run(self)


def reconstruct(backend, log):
    with patch.object(original_recovery, "RecoveryController", ContinueController):
        controller, state = prior.reconstruct(backend, log)
    require((SOURCE / "config.json").read_bytes() == (prior.SOURCE / "config.json").read_bytes(),
            "恢复配置不一致。")
    events = read_jsonl(SOURCE / "events.jsonl")
    require([x["event_index"] for x in events] == list(range(1, len(events) + 1)), "事件序号不连续。")
    preflight = [x for x in events if x["event"] == "preflight"]
    starts = [x for x in events if x["event"] == "formal_run_started"]
    require(len(preflight) == len(starts) == 1 and starts[0]["resume_of"] == str(prior.SOURCE)
            and events[-1]["event"] == "preflight_or_run_stopped", "断点身份不符。")
    for name, expected in preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
    failures = [x for x in events if x["event"] == "model_failure_classified"]
    attempts = [x for x in events if x["event"] == "model_attempt"]
    require([(x["stage"], x["category"]) for x in failures] == [("fusion_final", "network")]
            and len(attempts) == 34 and attempts[-1]["stage"] == "fusion_final", "失败点或调用数不符。")
    skipped = [x for x in events if x["event"] == "slot_skipped"]
    require(len(skipped) == 1 and skipped[0]["stage"] == "fusion_3"
            and skipped[0]["reason"] == "invalid_new_input", "留出的原槽位不符。")
    candidates = {x.candidate_id: x for x in controller.candidates}
    require(len(candidates) == 128 and controller._generated == 101, "历史候选起点不符。")
    registrations = [x for x in events if x["event"] == "candidate_registered"]
    require(len(registrations) == 11, "协作候选注册数量不符。")
    for row in registrations:
        ident, seq = row["candidate_id"], row["sequence"]
        require(ident == f"c{seq:04d}" and seq == len(controller.candidates) + 1, "候选编号不连续。")
        candidate = Candidate(ident, seq, row["stage"], tuple(row["parents"]))
        controller.candidates.append(candidate)
        candidates[ident] = candidate
    responses = {x["candidate_id"]: x for x in events if x["event"] == "proposal_response"}
    require(len(responses) == 29 and sorted(x["generation_order"] for x in responses.values()) == list(range(102, 131)),
            "候选生成顺序不符。")
    pending = {c.candidate_id: s for c, s in zip(state["pending"], state["pending_specs"])}
    for ident, row in responses.items():
        candidate = candidates[ident]
        if ident in pending:
            stage, parents, prompt = pending[ident]
            require(candidate.stage == stage == row["stage"] and list(candidate.parents) == parents
                    and row["prompt"] == prompt, f"EoH 提案来源不符：{ident}")
        parsed = controller.parse_proposal(row["response"])
        check_modification_scope(parsed["code"])
        candidate.thought, candidate.code = parsed["thought"], parsed["code"]
        candidate.generation_order = row["generation_order"]
    controller._generated = 130
    evaluations = {x["candidate_id"]: x["result"] for x in events if x["event"] == "evaluation"}
    require(len(evaluations) == 32 and set(evaluations) == set(responses) | {"c0106", "c0107", "c0109"},
            "候选首次评价不完整。")
    for ident, result in evaluations.items():
        candidate = candidates[ident]
        directory = SOURCE / "candidates" / ident
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

    population = state["population"]
    for stage, parent_ids, _ in controller._replay_specs:
        operator = stage.split("_")[1]
        parents = controller.eoh_parents(population, 5 if operator in ("E1", "E2") else 1)
        require([x.candidate_id for x in parents] == parent_ids, f"第 2 代随机父代不符：{stage}")
    a, b = controller.elite_pair(population)
    pair = [x for x in events if x["event"] == "elite_pair"]
    require(len(pair) == 1 and (a.candidate_id, b.candidate_id) == (pair[0]["a"], pair[0]["b"]),
            "第 2 代双精英不符。")
    controller.generation = 2
    critics = {x["stage"]: x for x in events if x["event"] == "model_response"}
    require(set(critics) == {"critic_initial", "critic_1", "critic_2", "critic_3"}
            and critics["critic_initial"]["prompt"] == controller.initial_critic_prompt(a, b), "Critic 初始记录不符。")
    critic = probe.parse_critic(critics["critic_initial"]["response"])
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
        row = critics[f"critic_{turn}"]
        require(row["prompt"] == controller.branch_critic_prompt(*(current[r] for r in controller.design_roles)),
                f"第 {turn} 轮 Critic 提示不符。")
        critic = parse_round_critic(row["response"])
        views = controller.critic_views(critic)
        reflections.extend(controller.fusion_reflections(critic))
        for role in controller.design_roles:
            role_reflections[role].append(views[role]["reflection"])
        proposals = restore_round(turn, views)
        if turn in (1, 2):
            fusion = [x for x in responses.values() if x["stage"] == f"fusion_{turn}"]
            require(len(fusion) == 1 and fusion[0]["prompt"] == controller.round_fusion_prompt(proposals, "Lower"),
                    f"第 {turn} 轮融合提示不符。")
        else:
            require(not all(x.valid for x in proposals), "第 3 轮融合跳过条件不符。")
            last = [*proposals, None]
    require(all(formal[role] for role in controller.design_roles), "末轮融合缺少有效分支。")
    best = [controller.select(formal[role], 1)[0] for role in controller.design_roles]
    failed = candidates["c0139"]
    require(failed.stage == "fusion_final" and failed.parents == tuple(x.candidate_id for x in best)
            and attempts[-1]["prompt"] == controller.final_fusion_prompt(best, reflections), "末轮融合父代或提示不符。")
    failed.error = "模型连接失败，未产生提案；用户指示断点续跑。"
    cpu = [x["accounting"]["cpu_seconds"] for x in events if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == 32 and len(state["cpu_samples"]) == 97, "累计评价 CPU 不符。")
    return controller, {"config": state["config"], "a": a, "b": b, "best": best,
                        "last": last, "eoh": controller._replay_eoh, "changes": changes,
                        "role_reflections": role_reflections, "retry_prompt": attempts[-1]["prompt"],
                        "cpu_samples": state["cpu_samples"] + cpu, "prior_hashes": preflight[0]["code_sha256"]}


def finish_generation(controller, state, elite_fusion):
    all_roles = (*controller.design_roles, controller.integrator_role)
    lessons = {}
    for role in controller.design_roles:
        lessons[role] = controller.request(f"reflection_{role}",
                                           controller.long_reflection_prompt(role, state["role_reflections"][role], state["changes"]),
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
    pool = [*state["eoh"], *state["last"], elite_fusion, *mutations]
    selected = controller.select(pool, controller.n)
    controller.event("selection", generation=2, pool=[x.candidate_id for x in pool if x is not None],
                     selected=[{"candidate_id": x.candidate_id, "evaluation_origin": x.evaluation_origin} for x in selected])
    for role in all_roles:
        controller.history[role].append(lessons[role])
        controller.event("history_appended", role=role, generation=2, reflection=lessons[role])
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
        expected = {(ROOT / "outputs/p1_roco" / name).resolve() for name in [*prior.HISTORY, SOURCE.name]}
        require(started == expected, "RoCo 首组存在未知已启动运行。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=config["formal_replicate"], selection_seed=task["selection_seed"],
                  resume_of=str(SOURCE), interrupted_events_sha256=digest(SOURCE / "events.jsonl"),
                  total_model_call_cap=420, note="按用户指示手动恢复连接失败；使用未调用的原槽位，不增加总上限。")
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
    fusion = controller.batch([("fusion_final", state["best"], state["retry_prompt"])])[0]
    require(fusion.candidate_id == "c0140", "重试候选身份不符。")
    selected = finish_generation(controller, state, fusion)
    require(len(backend.calls) == 10 and len(backend.evaluations) == 7 and len(selected) == 10,
            "第 2 代剩余槽位不符。")
    selected = controller.run_future(selected)
    require(len(selected) == 10 and controller.generation == 6 and len(backend.calls) == 270
            and len(backend.evaluations) == 239
            and [x["generation"] for x in log.items if x["event"] == "selection"] == [2, 3, 4, 5, 6],
            "后续代数、预算槽位或筛选不符。")
    print(json.dumps({"status": "passed", "prior_model_calls": 150, "prior_evaluations": 129,
                      "retry_candidate": "c0140", "later_model_calls": 270, "later_evaluations": 239,
                      "total_model_call_cap": 420, "formal_mutex_shared": True}, ensure_ascii=False))


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
    runtime = original_recovery.RecoveryRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    runtime.resume_first_generation = False
    controller, state = reconstruct(runtime, log)
    paths = [*(Path(x) for x in state["prior_hashes"]), SOURCE / "events.jsonl", SOURCE / "config.json",
             RECOVERY / "resume_after_181257.py", Path(__file__).resolve()]
    for candidate in controller.candidates:
        directory = SOURCE / "candidates" / candidate.candidate_id
        if not directory.is_dir():
            continue
        paths.extend(directory / item for item in ("result.json", "problems/tsp_aco/gpt.py"))
        if (directory / "stdout.txt").is_file():
            paths.append(directory / "stdout.txt")
    for path in paths:
        if str(path) not in runtime.protected_hashes:
            runtime.protected.append(path)
            runtime.protected_hashes[str(path)] = digest(path)
    budget = runtime.budget
    budget.model_normal = 150
    budget.eval_starts = budget.eval_normal = 129
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
              code_sha256=runtime.protected_hashes, selection_seed=config["task"]["selection_seed"],
              generations=6, continuation=True, normal_model_slots_used=150,
              physical_model_cli_invocations_used=150, evaluation_starts_used=129,
              evaluation_cpu_seconds_used=budget.cpu_finished)
    print(f"RoCo 末轮融合启动前检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(config, run_dir, log):
            log.event("resume_checkpoint", interrupted_run=str(SOURCE), generation=2, next_stage="fusion_final",
                      failed_candidate_id="c0139", next_candidate_id="c0140",
                      unused_original_slot="fusion_3", parent_ids=[x.candidate_id for x in state["best"]],
                      model_calls_already_used=150, evaluation_starts_already_used=129, total_model_call_cap=420)
            fusion = controller.batch([("fusion_final", state["best"], state["retry_prompt"])])[0]
            selected = finish_generation(controller, state, fusion)
            selected = controller.run_future(selected)
            require(len(selected) == 10 and controller.generation == 6, "RoCo 未完成六代。")
        print(f"RoCo 六代完成；记录：{run_dir}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"RoCo 末轮融合续跑停止：{exc}；记录：{run_dir}", file=sys.stderr, flush=True)
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
        cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    controller, state = reconstruct(CheckBackend(), None)
    print(json.dumps({"status": "passed", "generation": 2, "candidates": len(controller.candidates),
                      "model_calls_used": 150, "evaluation_starts": len(state["cpu_samples"]),
                      "next_candidate": "c0140", "parents": [x.candidate_id for x in state["best"]],
                      "model_call_cap": 420}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"RoCo 末轮融合恢复检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
