"""恢复 RoCo 第 2 代：评价已返回的提案并继续未启动的原槽位。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
SOURCE = ROOT / "outputs/p1_roco/20261112_181257_1a92db4f_resume"
HISTORY = ["20261112_170516_a9eab10e", "20261112_172536_af604137_resume",
           "20261112_174300_b791da58_resume", "20261112_180259_94bd200c_resume",
           "20261112_180809_e040f878_resume", "20261112_181257_1a92db4f_resume"]
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p1_formal_matrix as matrix
from scripts.p0_controller import StopRun, check_modification_scope
from scripts.p0_runtime import JsonLog, digest, memory_and_priority

spec = importlib.util.spec_from_file_location("roco_before_181257", RECOVERY / "resume_after_180809.py")
prior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prior
spec.loader.exec_module(prior)
require, read_jsonl = prior.require, prior.read_jsonl


def reconstruct(backend, log):
    controller, state = prior.reconstruct(backend, log)
    require((SOURCE / "config.json").read_bytes() == (prior.SOURCE / "config.json").read_bytes(),
            "恢复配置不一致。")
    events = read_jsonl(SOURCE / "events.jsonl")
    preflight = [x for x in events if x["event"] == "preflight"]
    starts = [x for x in events if x["event"] == "formal_run_started"]
    require(len(preflight) == len(starts) == 1 and starts[0]["resume_of"] == str(prior.SOURCE)
            and events[-1]["event"] == "preflight_or_run_stopped", "断点身份不符。")
    for name, expected in preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
    failures = [x for x in events if x["event"] == "model_failure_classified"]
    require([(x["stage"], x["category"]) for x in failures] == [("eoh_E2_10", "unknown")]
            and len([x for x in events if x["event"] == "model_attempt"]) == 4,
            "模型失败或调用数不符。")
    skipped = [x["candidate_id"] for x in events if x["event"] == "candidate_skipped"]
    require(skipped == [f"c{x:04d}" for x in range(110, 129)], "未启动槽位不符。")
    responses = {x["candidate_id"]: x for x in events if x["event"] == "proposal_response"}
    require(set(responses) == {"c0106", "c0107", "c0109"}
            and sorted(x["generation_order"] for x in responses.values()) == [99, 100, 101], "已返回提案不符。")
    require(not any(x.get("candidate_id") in responses for x in events if x["event"] == "evaluation_budget_reserved")
            and not any(x["event"] == "evaluation" for x in events), "候选已有评价启动记录。")
    pending = {c.candidate_id: (c, s) for c, s in zip(state["pending"], state["pending_specs"])}
    saved = []
    for ident in ("c0106", "c0107", "c0109"):
        candidate, (stage, parents, prompt) = pending[ident]
        row = responses[ident]
        require(candidate.stage == stage == row["stage"] and list(candidate.parents) == parents
                and row["prompt"] == prompt and row["generation"] == 2, f"候选来源不符：{ident}")
        parsed = controller.parse_proposal(row["response"])
        check_modification_scope(parsed["code"])
        candidate.thought, candidate.code = parsed["thought"], parsed["code"]
        candidate.generation_order = row["generation_order"]
        saved.append(candidate)
    pending["c0108"][0].error = "模型服务返回 Bad Request，未产生提案；不自动重试。"
    controller._generated = 101
    require(len(state["cpu_samples"]) == 97, "累计评价开销不符。")
    return controller, {"config": state["config"], "population": state["population"], "saved": saved,
                        "pending": [pending[x][0] for x in skipped],
                        "pending_specs": [pending[x][1] for x in skipped],
                        "cpu_samples": state["cpu_samples"], "prior_hashes": preflight[0]["code_sha256"]}


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
        require(started == {(ROOT / "outputs/p1_roco" / name).resolve() for name in HISTORY},
                "RoCo 首组存在未知已启动运行。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=config["formal_replicate"], selection_seed=task["selection_seed"],
                  resume_of=str(SOURCE), interrupted_events_sha256=digest(SOURCE / "events.jsonl"),
                  total_model_call_cap=420, note="首次评价已返回提案；只执行尚未启动的原槽位。")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def evaluate_saved(controller, candidate):
    backend = controller.backend
    backend.prepare_proposal(candidate)
    try:
        controller.event("reused_model_proposal", generation=2, candidate_id=candidate.candidate_id,
                         source_events=str(SOURCE / "events.jsonl"))
        result = backend.evaluate(candidate)
        controller.event("evaluation", candidate_id=candidate.candidate_id, result=result)
        if result.get("status") == "infrastructure_error":
            raise StopRun(f"评价基础设施不可信：{result.get('error')}")
        if result.get("status") != "passed":
            candidate.error = str(result.get("error") or result["status"])
            return
        objective, fitness = result.get("average"), result.get("fitness")
        if (not isinstance(objective, (int, float)) or isinstance(objective, bool)
                or not isinstance(fitness, (int, float)) or isinstance(fitness, bool)
                or not math.isfinite(objective) or not math.isfinite(fitness)
                or fitness != (-objective if controller.direction == "min" else objective)):
            raise StopRun("评价器返回的目标或 Fitness 不可信。")
        candidate.objective, candidate.fitness = float(objective), float(fitness)
        candidate.completed_order = result.get("completed_monotonic_ns", time.monotonic_ns())
        candidate.evaluation_origin = candidate.candidate_id
    finally:
        backend.release_proposal(candidate)


def run_pending(controller, state):
    require(len(state["pending"]) == len(state["pending_specs"]) == 19, "未启动槽位数量不符。")
    controller.generation = 2
    entries = []
    for candidate, (stage, parents, prompt) in zip(state["pending"], state["pending_specs"]):
        require(candidate.stage == stage and list(candidate.parents) == parents, "候选或父代不符。")
        controller.event("reused_candidate_registration", generation=2, candidate_id=candidate.candidate_id,
                         stage=stage, parents=parents, source_events=str(SOURCE / "events.jsonl"))
        entries.append((candidate, prompt))
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(controller.propose, candidate, prompt) for candidate, prompt in entries]
        failures = []
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                controller.stop.set()
                failures.append(exc)
        if failures:
            raise StopRun(f"并发阶段停止：{failures[0]}") from failures[0]


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
    for candidate in state["saved"]:
        evaluate_saved(controller, candidate)
    require(not backend.calls and len(backend.evaluations) == 3, "保存提案首次评价不符。")
    run_pending(controller, state)
    require(len(backend.calls) == 19 and len(backend.evaluations) == 22
            and all(stage != "eoh_E2_10" for stage, _ in backend.calls), "恢复槽位或失败处理不符。")
    selected = controller.run_remaining(state["population"])
    require(len(selected) == 10 and controller.generation == 6 and len(backend.calls) == 304
            and len(backend.evaluations) == 272
            and [x["generation"] for x in log.items if x["event"] == "selection"] == [2, 3, 4, 5, 6],
            "后续代数、预算槽位或筛选不符。")
    print(json.dumps({"status": "passed", "prior_model_calls": 116, "prior_evaluations": 97,
                      "saved_candidates": [x.candidate_id for x in state["saved"]],
                      "unstarted_proposals": 19, "later_model_calls": 304, "later_evaluation_slots": 272,
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
    runtime = prior.prior.prior.prior.prior.RecoveryRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    controller, state = reconstruct(runtime, log)
    for path in [*(Path(x) for x in state["prior_hashes"]), SOURCE / "events.jsonl", SOURCE / "config.json",
                 RECOVERY / "resume_after_180809.py", Path(__file__).resolve()]:
        if str(path) not in runtime.protected_hashes:
            runtime.protected.append(path)
            runtime.protected_hashes[str(path)] = digest(path)
    budget = runtime.budget
    budget.model_normal = 116
    budget.eval_starts = budget.eval_normal = 97
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
              code_sha256=runtime.protected_hashes, selection_seed=config["task"]["selection_seed"],
              generations=6, continuation=True, normal_model_slots_used=116,
              physical_model_cli_invocations_used=116, evaluation_starts_used=97,
              evaluation_cpu_seconds_used=budget.cpu_finished)
    print(f"RoCo 第 2 代启动前检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(config, run_dir, log):
            log.event("resume_checkpoint", interrupted_run=str(SOURCE), generation=2,
                      saved_candidates=[x.candidate_id for x in state["saved"]], failed_candidate_ids=["c0108"],
                      unstarted_candidates=[x.candidate_id for x in state["pending"]],
                      model_calls_already_used=116, evaluation_starts_already_used=97, total_model_call_cap=420)
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(evaluate_saved, controller, candidate) for candidate in state["saved"]]
                for future in futures:
                    future.result()
            run_pending(controller, state)
            selected = controller.run_remaining(state["population"])
            require(len(selected) == 10 and controller.generation == 6, "RoCo 未完成六代。")
        print(f"RoCo 六代完成；记录：{run_dir}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"RoCo 第 2 代续跑停止：{exc}；记录：{run_dir}", file=sys.stderr, flush=True)
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
    print(json.dumps({"status": "passed", "candidates": len(controller.candidates), "model_calls_used": 116,
                      "evaluation_starts": len(state["cpu_samples"]),
                      "saved_candidates": [x.candidate_id for x in state["saved"]],
                      "unstarted_candidates": len(state["pending"]), "model_call_cap": 420}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"RoCo 剩余槽位检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
