"""保留第 2 代第一轮的请求与评价失败，重放已有过程后继续。"""

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
import uuid
from unittest.mock import patch


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
SOURCE = ROOT / "outputs/p1_control/20261113_160445_22419217_resume"
PREVIOUS_SCRIPT = RECOVERY / "resume_proposed3_generation2.py"
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate, p1_formal_matrix as matrix
from scripts import p1_classified_runtime
from scripts.p0_controller import Candidate, StopRun, check_modification_scope, timing_from_result
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from tests.test_p0_controller import FixtureBackend, MemoryLog


spec = importlib.util.spec_from_file_location("proposed3_first_recovery", PREVIOUS_SCRIPT)
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
require, read_jsonl = previous.require, previous.read_jsonl


class ReplayController(previous.base.RecoveryController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_roles = {}
        self.cached_batches = {}

    def event(self, kind, **fields):
        if self.generation == 2 and kind == "elite_pair" and self.cached_roles:
            kind = "reused_elite_pair"
        elif self.generation == 2 and kind == "branch_transition" and fields.get("round") == 0 and self.cached_roles:
            kind = "reused_branch_transition"
        return super().event(kind, **fields)

    def request(self, stage, prompt, parser):
        if self.generation == 2 and stage in self.cached_roles:
            row = self.cached_roles.pop(stage)
            require(row["prompt"] == prompt, f"已完成角色请求的提示不符：{stage}")
            self.event("reused_role_response", generation=2, stage=stage, source_events=str(SOURCE / "events.jsonl"))
            return parser(row["response"])
        return super().request(stage, prompt, parser)

    def batch(self, specs):
        if self.generation == 2 and self._replay_phase == 2:
            key = tuple(stage for stage, _, _ in specs)
            if key in self.cached_batches:
                expected, candidates = self.cached_batches.pop(key)
                actual = [(stage, [p.candidate_id for p in parents], prompt) for stage, parents, prompt in specs]
                require(actual == expected, f"已完成或失败提案的提示不符：{key}")
                for candidate in candidates:
                    self.event("reused_candidate_registration", generation=2, candidate_id=candidate.candidate_id,
                               stage=candidate.stage, parents=list(candidate.parents),
                               source_events=str(SOURCE / "events.jsonl"))
                return candidates
        return super().batch(specs)


def apply_result(controller, candidate, response, result):
    parsed = controller.parse_proposal(response["response"])
    check_modification_scope(parsed["code"])
    candidate.thought, candidate.code = parsed["thought"], parsed["code"]
    candidate.generation_order = response["generation_order"]
    directory = SOURCE / "candidates" / candidate.candidate_id
    require(json.loads((directory / "result.json").read_text(encoding="utf-8")) == result
            and (directory / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8") == candidate.code + "\n",
            f"候选或评价记录不符：{candidate.candidate_id}")
    if result["status"] != "passed":
        require(candidate.candidate_id == "c0131" and result["status"] == "infrastructure_error"
                and result.get("error") == "其他并发任务停止运行。", "存在未预期的评价失败。")
        candidate.error = "并发任务停止导致评价中断；保留证据，不重评。"
        return
    checked = p0_tsp_aco_evaluate.read_complete(directory / "stdout.txt")
    require(all(checked[key] == result[key] for key in ("average", "fitness", "objectives")),
            f"评价输出不符：{candidate.candidate_id}")
    candidate.objective = candidate.raw_objective = float(result["average"])
    candidate.fitness = float(result["fitness"])
    candidate.timing_seconds, candidate.timing_scope = timing_from_result(result)
    candidate.completed_order = result["completed_monotonic_ns"]
    candidate.evaluation_origin = candidate.candidate_id


def reconstruct(backend, log):
    with patch.object(previous.base, "RecoveryController", ReplayController):
        controller, state = previous.reconstruct(backend, log)
    config = state["config"]
    require((SOURCE / "config.json").read_bytes() == (previous.SOURCE / "config.json").read_bytes(), "断点配置不一致。")
    rows = read_jsonl(SOURCE / "events.jsonl")
    require([x["event_index"] for x in rows] == list(range(1, len(rows) + 1)), "事件编号不连续。")
    starts = [x for x in rows if x["event"] == "formal_run_started"]
    preflight = [x for x in rows if x["event"] == "preflight"]
    require(len(starts) == len(preflight) == 1 and starts[0]["resume_of"] == str(previous.SOURCE)
            and starts[0]["interrupted_events_sha256"] == digest(previous.SOURCE / "events.jsonl")
            and starts[0]["total_model_call_cap"] == 420 and starts[0]["effective_model_concurrency"] == 1
            and preflight[0]["physical_model_cli_invocations_used"] == 96
            and preflight[0]["evaluation_starts_used"] == 88 and rows[-1]["event"] == "preflight_or_run_stopped",
            "断点身份、预算或停止状态不符。")
    for name, expected in preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
    attempts = [x for x in rows if x["event"] == "model_attempt"]
    failures = [x for x in rows if x["event"] == "model_failure_classified"]
    evaluations = {x["candidate_id"]: x["result"] for x in rows if x["event"] == "evaluation"}
    responses = {x["candidate_id"]: x for x in rows if x["event"] == "proposal_response"}
    eval_ids = [x["candidate_id"] for x in rows if x["event"] == "evaluation_budget_reserved"]
    require(len(attempts) == 45 and len(responses) == len(evaluations) == len(eval_ids) == 42
            and len(set(eval_ids)) == 42 and set(eval_ids) == set(responses) == set(evaluations)
            and len(failures) == 1 and failures[0]["stage"] == "round_1_speed_agent"
            and failures[0]["category"] == "network", "请求、评价或失败状态不符。")
    reused = [x for x in rows if x["event"] == "reused_candidate_registration"]
    require([x["candidate_id"] for x in reused] == [x.candidate_id for x in state["pending"]],
            "第 2 代待恢复候选不符。")
    by_stage = {x["stage"]: x for x in attempts}
    require(len(by_stage) == len(attempts), "第 2 代存在重复请求。")
    for candidate, (stage, parents, prompt) in zip(state["pending"], state["pending_specs"]):
        require(candidate.stage == stage and list(candidate.parents) == parents and by_stage[stage]["prompt"] == prompt,
                f"EoH 原始提示或父代不符：{candidate.candidate_id}")
        apply_result(controller, candidate, responses[candidate.candidate_id], evaluations[candidate.candidate_id])
    registrations = [x for x in rows if x["event"] == "candidate_registered"]
    require([x["candidate_id"] for x in registrations] == ["c0129", "c0130", "c0131", "c0132"],
            "协作候选注册不符。")
    candidates = {x.candidate_id: x for x in controller.candidates}
    for row in registrations:
        ident, seq = row["candidate_id"], row["sequence"]
        require(seq == len(controller.candidates) + 1 and ident == f"c{seq:04d}", "候选编号不连续。")
        candidate = Candidate(ident, seq, row["stage"], tuple(row["parents"]))
        controller.candidates.append(candidate)
        candidates[ident] = candidate
        if ident in responses:
            require(responses[ident]["prompt"] == by_stage[row["stage"]]["prompt"], f"协作提案提示不符：{ident}")
            apply_result(controller, candidate, responses[ident], evaluations[ident])
        else:
            require(ident == "c0132", "未返回提案的身份不符。")
            candidate.error = "模型连接失败；保留失败请求，不自动重试。"
    require(sorted(x["generation_order"] for x in responses.values()) == list(range(89, 131)), "生成顺序不连续。")
    controller._generated = 130
    role_rows = [x for x in rows if x["event"] == "model_response"]
    require([x["stage"] for x in role_rows] == ["critic_initial", "critic_1"], "已完成的 Critic 响应不符。")
    controller.cached_roles = {x["stage"]: x for x in role_rows}
    for ids in (("c0129", "c0130"), ("c0131", "c0132")):
        items = [candidates[x] for x in ids]
        key = tuple(x.stage for x in items)
        expected = [(x.stage, list(x.parents), by_stage[x.stage]["prompt"]) for x in items]
        controller.cached_batches[key] = (expected, items)
    pair = [x for x in rows if x["event"] == "elite_pair"]
    transitions = [x for x in rows if x["event"] == "branch_transition"]
    require(len(pair) == 1 and len(transitions) == 2
            and [x["proposal"] for x in transitions] == ["c0129", "c0130"]
            and all(x["round"] == 0 for x in transitions), "双精英或启动分支记录不符。")
    cpu = [x["accounting"]["cpu_seconds"] for x in rows if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == 42, "评价 CPU 记录不完整。")
    state.update(cpu_samples=state["cpu_samples"] + cpu, prior_hashes=preflight[0]["code_sha256"],
                 model_calls=141, evaluation_starts=130)
    return controller, state


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
        raise OSError(ctypes.get_last_error(), "无法创建研发互斥锁。")
    acquired = False
    try:
        result = kernel.WaitForSingleObject(handle, 0)
        require(result in (0, 0x80), "已有正式研发运行或无法取得互斥锁。")
        acquired = True
        matrix.verify_training_config(config)
        started = set()
        for path in (ROOT / "outputs/p1_control").glob("*/config.json"):
            if path.parent.resolve() == run_dir.resolve():
                continue
            other = json.loads(path.read_text(encoding="utf-8"))
            if (other.get("matrix_contract_sha256") == config["matrix_contract_sha256"]
                    and matrix.run_identity(other) == matrix.run_identity(config)):
                event_path = path.parent / "events.jsonl"
                if event_path.is_file() and any(x["event"] == "formal_run_started" for x in read_jsonl(event_path)):
                    started.add(path.parent.resolve())
        require(started == {previous.SOURCE.resolve(), SOURCE.resolve()}, "同一研发存在未知已启动运行。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=3, selection_seed=task["selection_seed"], resume_of=str(SOURCE),
                  interrupted_events_sha256=digest(SOURCE / "events.jsonl"), total_model_call_cap=420,
                  frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4,
                  note="两项未获有效评价的提案保留失败，沿用各分支上一有效状态并跳过本轮融合。")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def simulate():
    backend, log = FixtureBackend(), MemoryLog()
    backend.cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    controller, state = reconstruct(backend, log)
    with recovery_guard(state["config"], RECOVERY / "unused_proposed3_round1_fixture", MemoryLog()):
        code = ("from pathlib import Path\nfrom scripts import p1_formal_matrix as m\ntry:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED", "研发互斥锁不一致。")
    selected = controller.run_remaining(state["population"])
    transitions = [x for x in log.items if x["event"] == "branch_transition" and x.get("generation") == 2
                   and x.get("round") == 1]
    require(len(transitions) == 2 and [(x["proposal"], x["current"]) for x in transitions]
            == [("c0131", "c0129"), ("c0132", "c0130")], "无效分支没有沿用上一有效状态。")
    skipped = [x for x in log.items if x["event"] == "slot_skipped" and x.get("generation") == 2]
    require(len(skipped) == 1 and skipped[0]["stage"] == "fusion_1"
            and len(selected) == 10 and controller.generation == 6
            and len(backend.calls) == 278 and len(backend.evaluations) == 245
            and [x["generation"] for x in log.items if x["event"] == "selection"] == [2, 3, 4, 5, 6],
            "跳过融合、后续代次或预算模拟不符。")
    limited_log = MemoryLog()
    runtime = previous.RecoveryRuntime(SOURCE, SOURCE / "config.json", limited_log, cli=["offline-fixture"])
    limited_controller, limited_state = reconstruct(runtime, limited_log)
    with patch.object(probe, "run_command", return_value={"exit_code": 1, "stdout": "", "stderr": ""}), \
            patch.object(probe, "extract_message", side_effect=probe.ProbeError("offline 429")), \
            patch.object(p1_classified_runtime, "classify_failure",
                         return_value={"category": "quota", "source": "fixture", "plugin_warnings": []}):
        try:
            limited_controller.run_remaining(limited_state["population"])
        except StopRun:
            pass
        else:
            raise ValueError("限流夹具没有停止。")
    attempts = [x for x in limited_log.items if x["event"] == "model_attempt"]
    require(len(attempts) == 1 and attempts[0]["stage"] == "critic_2", "限流后启动了其他模型请求。")
    print(json.dumps({"status": "passed", "prior_model_calls": 141, "prior_evaluations": 130,
                      "remaining_model_calls": 278, "remaining_evaluations": 245,
                      "failed_candidates": ["c0131", "c0132"], "skipped_fusion": "fusion_1",
                      "formal_mutex_shared": True, "model_concurrency": 1, "stop_after_429": True}, ensure_ascii=False))


def resume():
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    matrix.verify_training_software(matrix.load_matrix()[0])
    check = FixtureBackend()
    check.cfg = config
    reconstruct(check, MemoryLog())
    resources = memory_and_priority()
    require(resources["available_memory_bytes"] >= config["resources"]["min_available_system_memory_bytes"], "内存不足。")
    cli = probe.cli_command()
    version, login = probe.run_command(cli + ["--version"]), probe.run_command(cli + ["login", "status"])
    require(version.get("exit_code") == 0 and login.get("exit_code") == 0
            and "Logged in using ChatGPT" in (login.get("stdout", "") + login.get("stderr", "")), "CLI 登录状态不符。")
    run_dir = ROOT / "outputs/p1_control" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8] + "_resume")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("xb") as out:
        out.write((SOURCE / "config.json").read_bytes())
    log = JsonLog(run_dir / "events.jsonl")
    runtime = previous.RecoveryRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    controller, state = reconstruct(runtime, log)
    previous.base.SOURCE = SOURCE
    for path in [*(Path(x) for x in state["prior_hashes"]), SOURCE / "events.jsonl", SOURCE / "config.json",
                 PREVIOUS_SCRIPT, Path(__file__).resolve()]:
        if str(path) not in runtime.protected_hashes:
            runtime.protected.append(path)
            runtime.protected_hashes[str(path)] = digest(path)
    budget = runtime.budget
    budget.model_normal = state["model_calls"]
    budget.eval_starts = budget.eval_normal = state["evaluation_starts"]
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
              code_sha256=runtime.protected_hashes, selection_seed=config["task"]["selection_seed"],
              generations=6, continuation=True, normal_model_slots_used=141, physical_model_cli_invocations_used=141,
              evaluation_starts_used=130, evaluation_cpu_seconds_used=budget.cpu_finished,
              frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4)
    print(f"本方法第 3 次研发协作断点检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(config, run_dir, log):
            log.event("resume_checkpoint", interrupted_run=str(SOURCE), generation=2,
                      unavailable_candidates=["c0131", "c0132"], skipped_fusion="fusion_1",
                      model_calls_already_used=141, evaluation_starts_already_used=130, total_model_call_cap=420)
            selected = controller.run_remaining(state["population"])
            require(len(selected) == 10 and controller.generation == 6, "六代未完成。")
        print(f"本方法第 3 次研发六代完成；记录：{run_dir}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"本方法第 3 次研发续跑停止：{exc}；记录：{run_dir}", file=sys.stderr, flush=True)
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
    elif args.run:
        return resume()
    else:
        backend = FixtureBackend()
        backend.cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
        _, state = reconstruct(backend, MemoryLog())
        print(json.dumps({"status": "passed", "model_calls_used": state["model_calls"],
                          "evaluation_starts": state["evaluation_starts"], "unavailable_candidates": ["c0131", "c0132"]},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"本方法第 3 次研发协作恢复检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
