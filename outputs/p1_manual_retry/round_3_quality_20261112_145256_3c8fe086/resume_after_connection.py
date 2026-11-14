"""复用已返回的第 3 轮分支，恢复连接失败后的正式研发。"""

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
FUSION_RETRY = ROOT / "outputs/p1_manual_retry/fusion_2_20261112_143551_66d3a6ad"
INTERRUPTED = ROOT / "outputs/p1_control/20261112_144932_43904aa4_resume"
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate, p1_formal_matrix as matrix
from scripts.p0_controller import Candidate, StopRun, check_modification_scope, timing_from_result
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from scripts.p1_runtime import TaskRuntime
from scripts.p1_task_contract import load_task_contract

spec = importlib.util.spec_from_file_location("p1_fusion_resume", FUSION_RETRY / "resume_formal.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
require = base.require
read_jsonl = base.read_jsonl


class RecoveryController(base.ResumeController):
    def set_reused_round(self, critic, attempts, candidates):
        self._saved_critic = critic
        self._saved_round_attempts = attempts
        self._saved_round_candidates = candidates

    def request(self, stage, prompt, parser):
        if stage == "critic_3" and self._saved_critic is not None:
            event, self._saved_critic = self._saved_critic, None
            require(prompt == event["prompt"], "重建的第 3 轮 Critic 提示不一致。")
            self.event("reused_model_response", stage=stage, generation=1,
                       source_events=str(INTERRUPTED / "events.jsonl"), response=event["response"])
            return parser(event["response"])
        return super().request(stage, prompt, parser)

    def batch(self, specs):
        if self._saved_round_candidates is not None:
            require([stage for stage, _, _ in specs] == ["round_3_solution_quality_agent", "round_3_speed_agent"],
                    "恢复位置不在第 3 轮双分支。")
            for stage, parents, prompt in specs:
                attempt = self._saved_round_attempts[stage]
                require(prompt == attempt["prompt"] and len(parents) == 1,
                        f"重建的分支提示不一致：{stage}")
                candidate = self._saved_round_candidates[0 if "quality" in stage else 1]
                require(candidate.parents == (parents[0].candidate_id,), f"重建的分支父代不一致：{stage}")
            candidates, self._saved_round_candidates = self._saved_round_candidates, None
            self.event("reused_evaluated_proposals", generation=1, round=3,
                       candidate_ids=[x.candidate_id for x in candidates],
                       source_events=str(RECOVERY / "evaluation_events.jsonl"))
            return candidates
        return super().batch(specs)


def reconstruct(backend, log):
    with patch.object(base, "ResumeController", RecoveryController):
        controller, state = base.reconstruct(backend, log)
    interrupted = read_jsonl(INTERRUPTED / "events.jsonl")
    manual = read_jsonl(RECOVERY / "events.jsonl")
    evaluation = read_jsonl(RECOVERY / "evaluation_events.jsonl")
    summary = json.loads((RECOVERY / "summary.json").read_text(encoding="utf-8"))
    require(summary["status"] == "proposal_parsed"
            and summary["source_events_sha256"] == digest(INTERRUPTED / "events.jsonl")
            and summary["prior_physical_calls"] == 85 and summary["total_model_call_cap"] == 421,
            "连接失败重试来源或预算不符。")
    require((INTERRUPTED / "config.json").read_bytes() == (RECOVERY / "config.json").read_bytes(),
            "本次重试评价配置与原配置不同。")
    preflight = [x for x in interrupted if x["event"] == "preflight"]
    require(len(preflight) == 1, "首次续跑启动检查不唯一。")
    for name, expected in preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"首次续跑受保护文件变化：{name}")
    require(interrupted[-1]["event"] == "preflight_or_run_stopped"
            and "模型连接失败" in interrupted[-1]["error"], "首次续跑不是预期的连接失败状态。")
    starts = [x for x in interrupted if x["event"] == "formal_run_started"]
    require(len(starts) == 1 and starts[0].get("resume_of") == str(base.SOURCE),
            "首次续跑身份不符。")
    attempts = {x["stage"]: x for x in interrupted if x["event"] == "model_attempt"}
    require(set(attempts) == {"critic_3", "round_3_solution_quality_agent", "round_3_speed_agent"},
            "首次续跑的模型调用阶段不符。")
    failures = [x for x in interrupted if x["event"] == "model_process"
                and x["stage"] == "round_3_solution_quality_agent"]
    require(len(failures) == 1 and failures[0]["result"]["exit_code"] == 1
            and "Connection failed: error sending request" in failures[0]["result"]["stdout"],
            "质量分支的原连接错误不符。")
    critic = [x for x in interrupted if x["event"] == "model_response" and x["stage"] == "critic_3"]
    speed_response = [x for x in interrupted if x["event"] == "proposal_response"
                      and x.get("candidate_id") == "c0081"]
    quality_response = [x for x in manual if x["event"] == "proposal_response"]
    require(len(critic) == len(speed_response) == len(quality_response) == 1, "原响应或重试响应不唯一。")
    manual_attempt = [x for x in manual if x["event"] == "model_attempt"]
    require(len(manual_attempt) == 1
            and manual_attempt[0]["prompt"] == attempts["round_3_solution_quality_agent"]["prompt"],
            "质量分支重试提示发生变化。")
    registered = [x for x in interrupted if x["event"] == "candidate_registered"]
    require([(x["candidate_id"], x["sequence"]) for x in registered] == [("c0080", 80), ("c0081", 81)],
            "原双分支候选编号不符。")
    failed = Candidate("c0080", 80, registered[0]["stage"], tuple(registered[0]["parents"]),
                       error="模型连接失败，未返回提案。")
    controller.candidates.append(failed)
    contract = load_task_contract("tsp_aco")
    restored = {}
    responses = {"c0081": speed_response[0]["response"], "c0082": quality_response[0]["response"]}
    details = {
        "c0081": (81, "round_3_speed_agent", ("c0077",), 79),
        "c0082": (82, "round_3_solution_quality_agent_manual_retry", ("c0076",), 80),
    }
    for ident, (seq, stage, parents, order) in details.items():
        parsed = contract.parse_proposal(responses[ident])
        check_modification_scope(parsed["code"])
        candidate = Candidate(ident, seq, stage, parents, thought=parsed["thought"], code=parsed["code"],
                              generation_order=order)
        rows = [x for x in evaluation if x["event"] == "evaluation" and x["candidate_id"] == ident]
        directory = RECOVERY / "candidates" / ident
        require(len(rows) == 1 and rows[0]["result"]["status"] == "passed"
                and json.loads((directory / "result.json").read_text(encoding="utf-8")) == rows[0]["result"]
                and (directory / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8") == candidate.code + "\n",
                f"分支候选或评价记录不一致：{ident}")
        result = rows[0]["result"]
        checked = p0_tsp_aco_evaluate.read_complete(directory / "stdout.txt")
        require(checked["average"] == result["average"] and checked["fitness"] == result["fitness"]
                and checked["objectives"] == result["objectives"], f"分支评价输出不一致：{ident}")
        candidate.objective = candidate.raw_objective = float(result["average"])
        candidate.fitness = float(result["fitness"])
        candidate.timing_seconds, candidate.timing_scope = timing_from_result(result)
        candidate.completed_order = result["completed_monotonic_ns"]
        candidate.evaluation_origin = ident
        controller.candidates.append(candidate)
        restored[ident] = candidate
    controller._generated = 80
    controller.set_reused_round(critic[0], attempts, [restored["c0082"], restored["c0081"]])
    cpu = [x["accounting"]["cpu_seconds"] for x in evaluation if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == 2, "第 3 轮两个候选的 CPU 记录不完整。")
    state["cpu_samples"] += cpu
    require(len(state["cpu_samples"]) == 79, "累计评价预算不符。")
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
            previous = json.loads(path.read_text(encoding="utf-8"))
            if (previous.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                    or matrix.run_identity(previous) != matrix.run_identity(config)):
                continue
            events = path.parent / "events.jsonl"
            if events.is_file() and any(x["event"] == "formal_run_started" for x in read_jsonl(events)):
                started.add(path.parent.resolve())
        require(started == {base.SOURCE.resolve(), INTERRUPTED.resolve()},
                "该重复索引存在未知已启动运行，禁止续跑。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=config["formal_replicate"], selection_seed=task["selection_seed"],
                  resume_of=str(INTERRUPTED), original_run=str(base.SOURCE),
                  interrupted_events_sha256=digest(INTERRUPTED / "events.jsonl"),
                  retry_events_sha256=digest(RECOVERY / "events.jsonl"),
                  evaluation_events_sha256=digest(RECOVERY / "evaluation_events.jsonl"),
                  extra_model_cli_invocations=1, total_model_call_cap=421,
                  note="保留先前获准的单次额外调用；本次连接补试计入原正常调用上限。")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def simulate():
    from tests.test_p0_controller import FixtureBackend, MemoryLog
    config = json.loads((INTERRUPTED / "config.json").read_text(encoding="utf-8"))
    guard_log = MemoryLog()
    with recovery_guard(config, RECOVERY / "unused_guard_fixture", guard_log):
        code = ("from pathlib import Path\nfrom scripts import p1_formal_matrix as m\ntry:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED", "续跑互斥锁不一致。")
    backend, log = FixtureBackend(), MemoryLog()
    controller, state = reconstruct(backend, log)
    first = base.finish_first_generation(controller, state)
    require((len(backend.calls), len(backend.evaluations), len(first)) == (11, 8, 10),
            "第 1 代剩余槽位与原流程不一致。")
    require(len([x for x in log.items if x["event"] == "reused_model_response"]) == 1
            and len([x for x in log.items if x["event"] == "reused_evaluated_proposals"]) == 1,
            "已返回结果复用记录缺失。")
    selected = controller.run_remaining(first, 1)
    require((len(backend.calls), len(backend.evaluations), len(selected)) == (76, 66, 10),
            "后续完整代槽位与原流程不一致。")
    require([x["generation"] for x in log.items if x["event"] == "selection"] == [1, 2]
            and all(len(controller.history[role]) == 2 for role in controller.history),
            "续跑筛选或角色历史不一致。")
    print(json.dumps({"status": "passed", "first_generation_calls": 11, "first_generation_evaluations": 8,
                      "second_generation_calls": 65, "second_generation_evaluations": 58,
                      "formal_mutex_shared": True}, ensure_ascii=False))


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
    runtime = TaskRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    extra_sources = [base.SOURCE / "events.jsonl", base.SOURCE / "config.json",
                     FUSION_RETRY / "events.jsonl", FUSION_RETRY / "evaluation_events.jsonl",
                     FUSION_RETRY / "summary.json", FUSION_RETRY / "resume_formal.py",
                     FUSION_RETRY / "candidates/c0079/result.json",
                     FUSION_RETRY / "candidates/c0079/stdout.txt",
                     FUSION_RETRY / "candidates/c0079/problems/tsp_aco/gpt.py",
                     INTERRUPTED / "events.jsonl", INTERRUPTED / "config.json",
                     RECOVERY / "events.jsonl", RECOVERY / "evaluation_events.jsonl", RECOVERY / "summary.json",
                     Path(__file__).resolve()]
    for ident in ("c0081", "c0082"):
        extra_sources.extend(RECOVERY / "candidates" / ident / item
                             for item in ("result.json", "stdout.txt", "problems/tsp_aco/gpt.py"))
    for path in extra_sources:
        runtime.protected.append(path)
        runtime.protected_hashes[str(path)] = digest(path)
    controller, state = reconstruct(runtime, log)
    budget = runtime.budget
    budget.model_normal = 85
    budget.eval_starts = budget.eval_normal = 79
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
              code_sha256=runtime.protected_hashes, selection_seed=config["task"]["selection_seed"],
              generations=6, continuation=True, normal_model_slots_used=85,
              physical_model_cli_invocations_used=86, extra_manual_invocations=1,
              evaluation_starts_used=79, evaluation_cpu_seconds_used=budget.cpu_finished)
    print(f"连接恢复启动前检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(config, run_dir, log):
            log.event("resume_checkpoint", original_run=str(base.SOURCE), interrupted_run=str(INTERRUPTED),
                      manual_recovery=str(RECOVERY), model_cli_invocations_already_used=86,
                      normal_model_slots_already_used=85, evaluation_starts_already_used=79,
                      generation=1, next_stage="fusion_3", reused_critic="critic_3",
                      reused_candidates=["c0082", "c0081"])
            first = base.finish_first_generation(controller, state)
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
    config = json.loads((INTERRUPTED / "config.json").read_text(encoding="utf-8"))
    class CheckBackend:
        cfg = config
    controller, state = reconstruct(CheckBackend(), None)
    print(json.dumps({"status": "passed", "candidates": len(controller.candidates),
                      "generated": controller._generated, "evaluation_starts": len(state["cpu_samples"]),
                      "normal_model_slots_used": 85, "physical_model_calls_used": 86,
                      "total_model_call_cap": 421}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
