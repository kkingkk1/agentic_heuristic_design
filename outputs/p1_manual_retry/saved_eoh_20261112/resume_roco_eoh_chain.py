"""按事件链恢复 RoCo 第 2 代未启动的 EoH 槽位，保留失败与评价。"""

import argparse
from collections import Counter
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
import threading
import time
import uuid
from unittest.mock import patch


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p1_formal_matrix as matrix
import scripts.p1_classified_runtime as classified_runtime
from scripts.p0_controller import StopRun, check_modification_scope
from scripts.p0_runtime import JsonLog, digest, memory_and_priority


spec = importlib.util.spec_from_file_location("roco_eoh_chain_base", RECOVERY / "resume_roco_generation2.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
require = base.require
read_events = base.read_events


def configure_model_concurrency(runtime):
    resources = runtime.cfg["resources"]
    require(resources["max_concurrent_model_calls"] == 4 and resources["max_concurrent_evaluations"] == 4,
            "冻结的并发合同不是预期的模型 4、评价 4。")
    runtime.model_slots = threading.BoundedSemaphore(1)


def unstarted_candidates(pending, attempted_ids, events):
    remaining = [x for x in pending if x.candidate_id not in attempted_ids]
    remaining_ids = {x.candidate_id for x in remaining}
    skipped = [x["candidate_id"] for x in events if x["event"] == "candidate_skipped"]
    reserved = [x["candidate_id"] for x in events if x["event"] == "proposal_budget_reserved"]
    released = [x["candidate_id"] for x in events if x["event"] == "proposal_budget_released"]
    require(len(skipped) == len(set(skipped)) and set(skipped) <= remaining_ids, "跳过的候选与未启动槽位不符。")
    require(Counter(reserved) == Counter(released) and len(reserved) == len(set(reserved)),
            "提案预算预留未正确释放或发生重复。")
    require(not any(x.get("candidate_id") in remaining_ids for x in events
                    if x["event"] in ("proposal_response", "evaluation_budget_reserved", "evaluation")),
            "未启动槽位已有响应或评价。")
    return remaining


def find_segments(config):
    """只接受单条续跑链；有分叉、活动进程或非 EoH 断点时停止。"""
    children = {}
    for path in (ROOT / "outputs/p1_roco").glob("*/config.json"):
        other = json.loads(path.read_text(encoding="utf-8"))
        if (other.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                or matrix.run_identity(other) != matrix.run_identity(config)):
            continue
        event_file = path.parent / "events.jsonl"
        if not event_file.is_file():
            continue
        events = read_events(event_file)
        starts = [x for x in events if x["event"] == "formal_run_started"]
        if not starts:
            continue
        require(len(starts) == 1, f"续跑身份不唯一：{event_file}")
        parent = starts[0].get("resume_of")
        if parent:
            children.setdefault(str(Path(parent).resolve()), []).append((path.parent, events))
    segments, parent, seen = [], base.SOURCE.resolve(), set()
    while str(parent) in children:
        options = children[str(parent)]
        require(len(options) == 1, f"续跑记录存在分叉：{parent}")
        directory, events = options[0]
        require(directory.resolve() not in seen, "续跑记录成环。")
        seen.add(directory.resolve())
        require(events[-1]["event"] == "preflight_or_run_stopped", "续跑仍在运行、已完成或不是可恢复断点。")
        segments.append((directory, events))
        parent = directory.resolve()
    require(segments, "未找到第 2 代后续断点。")
    return segments


def find_saved(source):
    """以来源文件哈希关联已完成的首次评价。"""
    found = []
    event_path = (source / "events.jsonl").resolve()
    for summary_path in (ROOT / "outputs/p1_manual_retry").glob("*/*/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        prior = summary.get("source_events")
        if not prior or Path(prior).resolve() != event_path:
            continue
        require(summary.get("status") == "completed" and summary.get("new_model_calls") == 0
                and summary.get("source_events_sha256") == digest(event_path), "保存响应评价身份不符。")
        directory = summary_path.parent
        require((directory / "config.json").read_bytes() == (source / "config.json").read_bytes(),
                "保存响应评价配置不一致。")
        rows = read_events(directory / "evaluation_events.jsonl")
        require([x["event_index"] for x in rows] == list(range(1, len(rows) + 1)), "评价事件序号不连续。")
        preflight = [x for x in rows if x["event"] == "preflight"]
        require(len(preflight) == 1, "保存响应评价预检不唯一。")
        for name, expected in preflight[0]["code_sha256"].items():
            require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
        found.append((directory, rows, preflight[0]["code_sha256"]))
    return found


def reconstruct(backend, log):
    controller, earlier = base.reconstruct(backend, log)
    config = earlier["config"]
    segments = find_segments(config)
    candidates = {x.candidate_id: x for x in controller.candidates}
    pending = list(earlier["pending"])
    initial_pending = earlier["pending"]
    spec_by_id = {candidate.candidate_id: spec
                  for candidate, spec in zip(initial_pending, earlier["pending_specs"])}
    require(len(spec_by_id) == len(initial_pending) == 36, "原未启动槽位不符。")
    cpu_samples = list(earlier["cpu_samples"])
    calls, evaluations = 99, 88
    generated = controller._generated
    prior_hashes = dict(earlier["prior_hashes"])
    prior_hashes.update(earlier["saved_hashes"])
    protected = [base.SOURCE / "events.jsonl", base.SOURCE / "config.json",
                 base.SAVED / "evaluation_events.jsonl", base.SAVED / "summary.json"]
    previous = base.SOURCE
    for directory, events in segments:
        require((directory / "config.json").read_bytes() == (base.SOURCE / "config.json").read_bytes(), "续跑配置不一致。")
        require([x["event_index"] for x in events] == list(range(1, len(events) + 1)), "续跑事件序号不连续。")
        starts = [x for x in events if x["event"] == "formal_run_started"]
        preflights = [x for x in events if x["event"] == "preflight"]
        require(len(starts) == len(preflights) == 1 and starts[0].get("resume_of") == str(previous)
                and starts[0].get("interrupted_events_sha256") == digest(previous / "events.jsonl")
                and starts[0].get("total_model_call_cap") == 420, "续跑链身份或预算不符。")
        require(preflights[0].get("physical_model_cli_invocations_used") == calls
                and preflights[0].get("evaluation_starts_used") == evaluations,
                "续跑前累计预算不符。")
        for name, expected in preflights[0]["code_sha256"].items():
            require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
            prior_hashes[name] = expected
        registrations = [x for x in events if x["event"] == "reused_candidate_registration"]
        require([x["candidate_id"] for x in registrations] == [x.candidate_id for x in pending],
                "续跑未启动槽位顺序不符。")
        by_stage = {x.stage: x for x in pending}
        require(len(by_stage) == len(pending), "未启动槽位阶段名重复。")
        attempts = [x for x in events if x["event"] == "model_attempt"]
        require(len({x["stage"] for x in attempts}) == len(attempts)
                and all(x["stage"] in by_stage for x in attempts), "存在重复或非 EoH 模型调用。")
        require(len([x for x in events if x["event"] == "model_budget_reserved"]) == len(attempts),
                "模型预算预留与请求启动不一致，不能自动续跑。")
        for attempt in attempts:
            candidate = by_stage[attempt["stage"]]
            stage, parents, prompt = spec_by_id[candidate.candidate_id]
            require(stage == candidate.stage and parents == list(candidate.parents) and attempt["prompt"] == prompt,
                    f"续跑父代或提示不符：{candidate.candidate_id}")
        responses = {x["candidate_id"]: x for x in events if x["event"] == "proposal_response"}
        source_results = {x["candidate_id"]: x["result"] for x in events if x["event"] == "evaluation"}
        started_eval = [x["candidate_id"] for x in events if x["event"] == "evaluation_budget_reserved"]
        require(len(set(started_eval)) == len(started_eval) and set(started_eval) == set(source_results),
                "已有评价启动缺少结果或发生重复。")
        saved_results = {}
        saved_count = 0
        for saved_dir, rows, hashes in find_saved(directory):
            for name, expected in hashes.items():
                if name in prior_hashes:
                    require(prior_hashes[name] == expected, f"历史文件哈希不一致：{name}")
                prior_hashes[name] = expected
            items = {x["candidate_id"]: x["result"] for x in rows if x["event"] == "evaluation"}
            saved_starts = [x["candidate_id"] for x in rows if x["event"] == "evaluation_budget_reserved"]
            require(set(items) == set(saved_starts) and len(saved_starts) == len(set(saved_starts))
                    and not set(items) & set(saved_results), "保存响应评价缺失或重复。")
            for ident, result in items.items():
                saved_results[ident] = (result, saved_dir)
            saved_count += len(saved_starts)
            cpu_samples.extend(x["accounting"]["cpu_seconds"] for x in rows if x["event"] == "evaluation_cpu_final")
            protected.extend((saved_dir / "evaluation_events.jsonl", saved_dir / "summary.json", saved_dir / "config.json"))
        require(not set(source_results) & set(saved_results)
                and set(responses) == set(source_results) | set(saved_results), "存在未评价、重复评价或缺失响应的候选。")
        require(sorted(x["generation_order"] for x in responses.values())
                == list(range(generated + 1, generated + len(responses) + 1)), "生成顺序不连续。")
        generated += len(responses)
        for ident, response in responses.items():
            require(ident in candidates and candidates[ident] in pending and response["generation"] == 2,
                    f"续跑候选身份不符：{ident}")
            if ident in source_results:
                result, result_dir = source_results[ident], directory / "candidates" / ident
            else:
                result, saved_dir = saved_results[ident]
                result_dir = saved_dir / "candidates" / ident
            base.apply_result(controller, candidates[ident], response, result, result_dir)
        failed = [x for x in events if x["event"] == "model_failure_classified"]
        for row in failed:
            require(row["stage"] in by_stage, "失败阶段不属于未启动槽位。")
            candidate = by_stage[row["stage"]]
            require(candidate.candidate_id not in responses, "失败槽位同时出现成功响应。")
            candidate.error = f"模型请求失败（{row['category']}）；保留失败，不自动重试。"
        attempted_ids = {by_stage[x["stage"]].candidate_id for x in attempts}
        require(attempted_ids == set(responses) | {by_stage[x["stage"]].candidate_id for x in failed},
                "模型调用没有对应响应或失败记录。")
        pending = unstarted_candidates(pending, attempted_ids, events)
        cpu = [x["accounting"]["cpu_seconds"] for x in events if x["event"] == "evaluation_cpu_final"]
        require(len(cpu) == len(started_eval), "评价 CPU 记录不完整。")
        cpu_samples.extend(cpu)
        calls += len(attempts)
        evaluations += len(started_eval) + saved_count
        require(calls <= 420 and evaluations <= 378 and len(cpu_samples) == evaluations, "累计预算不符。")
        protected.extend((directory / "events.jsonl", directory / "config.json"))
        previous = directory
    require(pending and max(c.completed_order for c in controller.candidates if c.valid) < time.monotonic_ns(),
            "没有可恢复槽位或历史单调时钟不符。")
    controller._generated = generated
    return controller, {"config": config, "population": earlier["population"], "pending": pending,
                        "pending_specs": [spec_by_id[x.candidate_id] for x in pending],
                        "cpu_samples": cpu_samples, "model_calls": calls, "evaluation_starts": evaluations,
                        "segments": segments, "latest": previous, "prior_hashes": prior_hashes,
                        "protected": protected}


def run_pending(controller, state):
    require(len(state["pending"]) == len(state["pending_specs"]) > 0, "待执行槽位数量不符。")
    controller.generation = 2
    entries = []
    for candidate, (stage, parents, prompt) in zip(state["pending"], state["pending_specs"]):
        require(candidate.stage == stage and list(candidate.parents) == parents, "待执行候选或父代不符。")
        controller.event("reused_candidate_registration", generation=2, candidate_id=candidate.candidate_id,
                         stage=stage, parents=parents, source_events=str(state["latest"] / "events.jsonl"))
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
    controller.generation = 1


@contextmanager
def recovery_guard(state, run_dir, log, resume_basis="offline_simulation"):
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
        config = state["config"]
        matrix.verify_training_config(config)
        started = set()
        for path in (ROOT / "outputs/p1_roco").glob("*/config.json"):
            if path.parent.resolve() == run_dir.resolve():
                continue
            other = json.loads(path.read_text(encoding="utf-8"))
            if (other.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                    or matrix.run_identity(other) != matrix.run_identity(config)):
                continue
            event_file = path.parent / "events.jsonl"
            if event_file.is_file() and any(x["event"] == "formal_run_started" for x in read_events(event_file)):
                started.add(path.parent.resolve())
        expected = {base.base.previous.SOURCE, base.base.SOURCE, base.SOURCE}
        expected.update(path for path, _ in state["segments"])
        require(started == {x.resolve() for x in expected}, "RoCo 研发存在未知已启动运行。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=config["formal_replicate"], selection_seed=task["selection_seed"],
                  resume_of=str(state["latest"]), interrupted_events_sha256=digest(state["latest"] / "events.jsonl"),
                  total_model_call_cap=420, resume_basis=resume_basis,
                  frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4,
                  note="保留失败与已启动评价；仅执行未启动的原槽位。")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def verify_serial_runtime():
    """用同一运行时的离线响应核验模型并发及限流后的等待槽位。"""
    from tests.test_p0_controller import MemoryLog

    runtime = base.ResumeRuntime(base.SOURCE, base.SOURCE / "config.json", MemoryLog(), cli=["offline-fixture"])

    def peak_model_calls():
        counter = {"active": 0, "peak": 0}
        lock, barrier = threading.Lock(), threading.Barrier(4)

        def fake_command(command, prompt, cwd=None):
            with lock:
                counter["active"] += 1
                counter["peak"] = max(counter["peak"], counter["active"])
            time.sleep(0.05)
            with lock:
                counter["active"] -= 1
            return {"exit_code": 0, "stdout": "", "stderr": ""}

        def call(index):
            barrier.wait()
            return runtime.model(f"fixture_{index}", "offline")

        with patch.object(probe, "run_command", side_effect=fake_command), \
             patch.object(probe, "extract_message", return_value=("offline", {}, [], [])), \
             patch.object(classified_runtime, "classify_failure",
                          return_value={"category": "unknown", "source": "fixture", "plugin_warnings": []}):
            with ThreadPoolExecutor(max_workers=4) as executor:
                require(list(executor.map(call, range(4))) == ["offline"] * 4, "并发夹具结果不符。")
        return counter["peak"]

    before = peak_model_calls()
    configure_model_concurrency(runtime)
    after = peak_model_calls()
    require(before == 4 and after == 1, "模型并发上限未按预期从 4 变为 1。")
    acquired = [runtime.eval_slots.acquire(blocking=False) for _ in range(5)]
    require(acquired == [True, True, True, True, False], "评价并发上限发生变化。")
    for _ in range(4):
        runtime.eval_slots.release()

    log = MemoryLog()
    stopped_runtime = base.ResumeRuntime(base.SOURCE, base.SOURCE / "config.json", log, cli=["offline-fixture"])
    configure_model_concurrency(stopped_runtime)
    controller, state = reconstruct(stopped_runtime, log)

    def fake_limited(command, prompt, cwd=None):
        time.sleep(0.05)
        return {"exit_code": 1, "stdout": "", "stderr": ""}

    with patch.object(probe, "run_command", side_effect=fake_limited), \
         patch.object(probe, "extract_message", side_effect=probe.ProbeError("offline 429")), \
         patch.object(classified_runtime, "classify_failure",
                      return_value={"category": "quota", "source": "fixture", "plugin_warnings": []}):
        try:
            run_pending(controller, state)
        except StopRun:
            pass
        else:
            raise ValueError("限流夹具没有停止运行。")
    attempts = [x for x in log.items if x["event"] == "model_attempt"]
    require(len(attempts) == 1, "单并发限流后仍启动了其他模型请求。")
    attempted = {x.candidate_id for x in state["pending"] if x.stage == attempts[0]["stage"]}
    remaining = unstarted_candidates(state["pending"], attempted, log.items)
    require(len(remaining) == len(state["pending"]) - 1, "等待模型的未启动槽位没有保留。")


def simulate():
    from tests.test_p0_controller import FixtureBackend, MemoryLog
    backend, log = FixtureBackend(), MemoryLog()
    controller, state = reconstruct(backend, log)
    verify_serial_runtime()
    with recovery_guard(state, RECOVERY / "unused_chain_guard_fixture", MemoryLog()):
        code = ("from pathlib import Path\nfrom scripts import p1_formal_matrix as m\ntry:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED", "续跑互斥锁不一致。")
    pending_count = len(state["pending"])
    run_pending(controller, state)
    require(len(backend.calls) == pending_count and len(backend.evaluations) == pending_count,
            "未启动 EoH 槽位数量不符。")
    selected = controller.run_future(state["population"])
    require(len(selected) == 10 and controller.generation == 6 and len(backend.calls) == pending_count + 285
            and len(backend.evaluations) == pending_count + 250
            and [x["generation"] for x in log.items if x["event"] == "selection"] == [2, 3, 4, 5, 6],
            "后续代数、预算槽位或筛选不符。")
    require(state["model_calls"] + len(backend.calls) <= 420
            and state["evaluation_starts"] + len(backend.evaluations) <= 378, "模拟超出预算。")
    print(json.dumps({"status": "passed", "prior_model_calls": state["model_calls"],
                      "prior_evaluations": state["evaluation_starts"], "unstarted_eoh": pending_count,
                      "remaining_model_calls": len(backend.calls), "remaining_evaluations": len(backend.evaluations),
                      "model_call_cap": 420, "formal_mutex_shared": True,
                      "model_concurrency_before": 4, "model_concurrency_after": 1,
                      "evaluation_concurrency": 4, "single_request_stop_verified": True}, ensure_ascii=False))


def resume(resume_basis):
    config = json.loads((base.SOURCE / "config.json").read_text(encoding="utf-8"))
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
        out.write((base.SOURCE / "config.json").read_bytes())
    log = JsonLog(run_dir / "events.jsonl")
    runtime = base.ResumeRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    configure_model_concurrency(runtime)
    controller, state = reconstruct(runtime, log)
    for path in [*(Path(x) for x in state["prior_hashes"]), *state["protected"], Path(__file__).resolve()]:
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
              generations=6, continuation=True, normal_model_slots_used=state["model_calls"],
              physical_model_cli_invocations_used=state["model_calls"], evaluation_starts_used=state["evaluation_starts"],
              evaluation_cpu_seconds_used=budget.cpu_finished, resume_basis=resume_basis,
              frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4)
    print(f"RoCo 第 2 代续跑链检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(state, run_dir, log, resume_basis=resume_basis):
            log.event("resume_checkpoint", interrupted_run=str(state["latest"]), generation=2,
                      unstarted_candidates=[x.candidate_id for x in state["pending"]],
                      model_calls_already_used=state["model_calls"], evaluation_starts_already_used=state["evaluation_starts"],
                      total_model_call_cap=420)
            run_pending(controller, state)
            selected = controller.run_future(state["population"])
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
    parser.add_argument("--service-restored", action="store_true", help="服务已恢复时续跑")
    parser.add_argument("--user-requested-resume", action="store_true", help="用户明确要求手动续跑；再次限流时停止")
    args = parser.parse_args()
    if args.simulate:
        simulate()
        return 0
    if args.run:
        require(args.service_restored != args.user_requested_resume, "须明确记录服务恢复或用户手动指示其中一项。")
        return resume("user_requested" if args.user_requested_resume else "service_restored")
    class CheckBackend:
        cfg = json.loads((base.SOURCE / "config.json").read_text(encoding="utf-8"))
    _, state = reconstruct(CheckBackend(), None)
    print(json.dumps({"status": "passed", "model_calls_used": state["model_calls"],
                      "evaluation_starts": state["evaluation_starts"], "unstarted_eoh": len(state["pending"]),
                      "total_model_call_cap": 420}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"RoCo 续跑链检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
