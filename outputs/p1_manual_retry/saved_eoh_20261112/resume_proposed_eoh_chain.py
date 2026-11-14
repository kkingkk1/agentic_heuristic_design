"""恢复本方法首组第 6 代未启动的 EoH 槽位。"""

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
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p1_formal_matrix as matrix
from scripts.p0_controller import StopRun
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from tests.test_p0_controller import FixtureBackend, MemoryLog


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, RECOVERY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


previous = load_module("proposed_generation6_base", "resume_proposed_generation6.py")
helper = load_module("roco_eoh_helpers", "resume_roco_eoh_chain.py")
require, read_events = previous.require, previous.read_events


def find_segments(config):
    children = {}
    for path in (ROOT / "outputs/p1_control").glob("*/config.json"):
        other = json.loads(path.read_text(encoding="utf-8"))
        if (other.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                or matrix.run_identity(other) != matrix.run_identity(config)):
            continue
        event_file = path.parent / "events.jsonl"
        if not event_file.is_file():
            continue
        rows = read_events(event_file)
        starts = [x for x in rows if x["event"] == "formal_run_started"]
        if not starts:
            continue
        require(len(starts) == 1, "续跑身份不唯一。")
        if starts[0].get("resume_of"):
            children.setdefault(str(Path(starts[0]["resume_of"]).resolve()), []).append((path.parent, rows))
    segments, parent, seen = [], previous.SOURCE.resolve(), set()
    while str(parent) in children:
        options = children[str(parent)]
        require(len(options) == 1, f"续跑记录存在分叉：{parent}")
        directory, rows = options[0]
        require(directory.resolve() not in seen and rows[-1]["event"] == "preflight_or_run_stopped",
                "续跑成环、仍在运行或已完成。")
        seen.add(directory.resolve())
        segments.append((directory, rows))
        parent = directory.resolve()
    require(segments, "未找到第 6 代后续断点。")
    return segments


def reconstruct(backend, log):
    controller, state = previous.reconstruct(backend, log)
    config = state["config"]
    segments = find_segments(config)
    pending = list(state["pending"])
    require(len(pending) == 36, "原未启动槽位数不符。")
    candidates = {x.candidate_id: x for x in controller.candidates}
    spec_by_id = {c.candidate_id: spec for c, spec in zip(pending, state["pending_specs"])}
    cpu_samples = list(state["cpu_samples"])
    model_calls, eval_starts = 361, 321
    prior_hashes = {**state["prior_hashes"], **state["saved_hashes"]}
    protected = [previous.SOURCE / "events.jsonl", previous.SOURCE / "config.json",
                 previous.SAVED / "evaluation_events.jsonl", previous.SAVED / "summary.json",
                 previous.SAVED / "config.json"]
    prior = previous.SOURCE
    for directory, rows in segments:
        require((directory / "config.json").read_bytes() == (previous.SOURCE / "config.json").read_bytes(), "续跑配置不一致。")
        require([x["event_index"] for x in rows] == list(range(1, len(rows) + 1)), "事件编号不连续。")
        starts = [x for x in rows if x["event"] == "formal_run_started"]
        preflights = [x for x in rows if x["event"] == "preflight"]
        require(len(starts) == len(preflights) == 1 and starts[0].get("resume_of") == str(prior)
                and starts[0].get("interrupted_events_sha256") == digest(prior / "events.jsonl")
                and starts[0].get("total_model_call_cap") == 422, "断点链或调用上限不符。")
        require(preflights[0].get("physical_model_cli_invocations_used") == model_calls
                and preflights[0].get("normal_model_slots_used") == model_calls - 1
                and preflights[0].get("evaluation_starts_used") == eval_starts, "断点预算不符。")
        for name, expected in preflights[0]["code_sha256"].items():
            require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
            prior_hashes[name] = expected
        registrations = [x["candidate_id"] for x in rows if x["event"] == "reused_candidate_registration"]
        require(registrations == [x.candidate_id for x in pending], "待执行槽位不符。")
        by_stage = {x.stage: x for x in pending}
        attempts = [x for x in rows if x["event"] == "model_attempt"]
        require(len({x["stage"] for x in attempts}) == len(attempts)
                and all(x["stage"] in by_stage for x in attempts), "存在重复或非 EoH 模型调用。")
        require(len([x for x in rows if x["event"] == "model_budget_reserved"]) == len(attempts),
                "模型预算预留与请求启动不一致。")
        for attempt in attempts:
            candidate = by_stage[attempt["stage"]]
            stage, parents, prompt = spec_by_id[candidate.candidate_id]
            require(stage == candidate.stage and parents == list(candidate.parents) and attempt["prompt"] == prompt,
                    f"父代或提示不符：{candidate.candidate_id}")
        responses = {x["candidate_id"]: x for x in rows if x["event"] == "proposal_response"}
        results = {x["candidate_id"]: x["result"] for x in rows if x["event"] == "evaluation"}
        eval_ids = [x["candidate_id"] for x in rows if x["event"] == "evaluation_budget_reserved"]
        require(len(eval_ids) == len(set(eval_ids)) and set(eval_ids) == set(results), "评价结果缺失或重复。")
        saved_results = {}
        saved_count = 0
        for saved_dir, saved_rows, hashes in helper.find_saved(directory):
            prior_hashes.update(hashes)
            items = {x["candidate_id"]: x["result"] for x in saved_rows if x["event"] == "evaluation"}
            saved_ids = [x["candidate_id"] for x in saved_rows if x["event"] == "evaluation_budget_reserved"]
            require(len(saved_ids) == len(set(saved_ids)) and set(saved_ids) == set(items)
                    and not set(items) & set(saved_results), "保存响应评价缺失或重复。")
            for ident, result in items.items():
                saved_results[ident] = (result, saved_dir / "candidates" / ident)
            saved_count += len(saved_ids)
            cpu_samples.extend(x["accounting"]["cpu_seconds"] for x in saved_rows if x["event"] == "evaluation_cpu_final")
            protected.extend((saved_dir / "evaluation_events.jsonl", saved_dir / "summary.json", saved_dir / "config.json"))
        invalid = {x["candidate_id"]: x["error"] for x in rows if x["event"] == "candidate_invalid"}
        require(not set(results) & set(saved_results) and not (set(results) | set(saved_results)) & set(invalid)
                and set(responses) == set(results) | set(saved_results) | set(invalid), "响应与首次评价不一致。")
        orders = sorted(x["generation_order"] for x in responses.values())
        require(orders == list(range(controller._generated + 1, controller._generated + len(orders) + 1)),
                "生成顺序不连续。")
        controller._generated += len(orders)
        for ident, response in responses.items():
            require(ident in candidates and candidates[ident] in pending and response["generation"] == 6,
                    "续跑响应候选不符。")
            if ident in invalid:
                candidates[ident].generation_order = response["generation_order"]
                candidates[ident].error = invalid[ident]
            else:
                result, candidate_dir = ((results[ident], directory / "candidates" / ident)
                                         if ident in results else saved_results[ident])
                helper.base.apply_result(controller, candidates[ident], response, result, candidate_dir)
        failures = [x for x in rows if x["event"] == "model_failure_classified"]
        for failure in failures:
            require(failure["stage"] in by_stage and by_stage[failure["stage"]].candidate_id not in responses,
                    "失败槽位不符。")
            by_stage[failure["stage"]].error = f"模型请求失败（{failure['category']}）；不自动重试。"
        attempted_ids = {by_stage[x["stage"]].candidate_id for x in attempts}
        require(attempted_ids == set(responses) | {by_stage[x["stage"]].candidate_id for x in failures},
                "模型调用没有对应响应或失败记录。")
        pending = helper.unstarted_candidates(pending, attempted_ids, rows)
        cpu = [x["accounting"]["cpu_seconds"] for x in rows if x["event"] == "evaluation_cpu_final"]
        require(len(cpu) == len(eval_ids), "评价 CPU 记录不完整。")
        cpu_samples.extend(cpu)
        model_calls += len(attempts)
        eval_starts += len(eval_ids) + saved_count
        require(model_calls <= 422 and eval_starts <= 378 and len(cpu_samples) == eval_starts, "累计预算不符。")
        protected.extend((directory / "events.jsonl", directory / "config.json"))
        prior = directory
    require(pending and max(c.completed_order for c in controller.candidates if c.valid) < time.monotonic_ns(),
            "没有可恢复槽位或历史单调时钟不符。")
    state.update(pending=pending, pending_specs=[spec_by_id[x.candidate_id] for x in pending],
                 cpu_samples=cpu_samples, model_calls=model_calls, evaluation_starts=eval_starts,
                 segments=segments, latest=prior, prior_hashes=prior_hashes, protected=protected)
    return controller, state


def run_pending(controller, state):
    controller.generation = 6
    entries = []
    for candidate, (stage, parents, prompt) in zip(state["pending"], state["pending_specs"]):
        require(candidate.stage == stage and list(candidate.parents) == parents, "待执行候选或父代不符。")
        controller.event("reused_candidate_registration", generation=6, candidate_id=candidate.candidate_id,
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
    controller.generation = 5


@contextmanager
def recovery_guard(state, run_dir, log):
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
        for path in (ROOT / "outputs/p1_control").glob("*/config.json"):
            if path.parent.resolve() == run_dir.resolve():
                continue
            other = json.loads(path.read_text(encoding="utf-8"))
            if (other.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                    or matrix.run_identity(other) != matrix.run_identity(config)):
                continue
            event_file = path.parent / "events.jsonl"
            if event_file.is_file() and any(x["event"] == "formal_run_started" for x in read_events(event_file)):
                started.add(path.parent.resolve())
        base = previous.base
        expected = {base.base.SOURCE, base.generation3.previous.INTERRUPTED,
                    base.generation3.INTERRUPTED, base.INTERRUPTED, previous.SOURCE}
        expected.update(path for path, _ in state["segments"])
        require(started == {x.resolve() for x in expected}, "首组存在未知已启动运行。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=config["formal_replicate"], selection_seed=task["selection_seed"],
                  resume_of=str(state["latest"]), interrupted_events_sha256=digest(state["latest"] / "events.jsonl"),
                  total_model_call_cap=422, extra_model_cli_invocations=2, resume_basis="user_requested",
                  frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4,
                  note="保留失败与已启动评价；仅执行未启动的原槽位。")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def verify_stop_after_limit():
    log = MemoryLog()
    runtime = previous.ResumeRuntime(previous.SOURCE, previous.SOURCE / "config.json", log, cli=["offline-fixture"])
    previous.configure_model_concurrency(runtime)
    controller, state = reconstruct(runtime, log)

    def fake_limited(command, prompt, cwd=None):
        time.sleep(0.05)
        return {"exit_code": 1, "stdout": "", "stderr": ""}

    with patch.object(probe, "run_command", side_effect=fake_limited), \
            patch.object(probe, "extract_message", side_effect=probe.ProbeError("offline 429")), \
            patch.object(previous.classified_runtime, "classify_failure",
                         return_value={"category": "quota", "source": "fixture", "plugin_warnings": []}):
        try:
            run_pending(controller, state)
        except StopRun:
            pass
        else:
            raise ValueError("限流夹具没有停止。")
    attempts = [x for x in log.items if x["event"] == "model_attempt"]
    require(len(attempts) == 1, "单并发限流后仍启动了其他请求。")
    attempted = {x.candidate_id for x in state["pending"] if x.stage == attempts[0]["stage"]}
    remaining = helper.unstarted_candidates(state["pending"], attempted, log.items)
    require(len(remaining) == len(state["pending"]) - 1, "限流后的等待槽位没有保留。")


def simulate():
    backend, log = FixtureBackend(), MemoryLog()
    backend.cfg = json.loads((previous.SOURCE / "config.json").read_text(encoding="utf-8"))
    controller, state = reconstruct(backend, log)
    previous.verify_serial_runtime()
    verify_stop_after_limit()
    with recovery_guard(state, RECOVERY / "unused_proposed_chain_fixture", MemoryLog()):
        code = ("from pathlib import Path\nfrom scripts import p1_formal_matrix as m\ntry:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED", "续跑互斥锁不一致。")
    count = len(state["pending"])
    run_pending(controller, state)
    require(len(backend.calls) == count and len(backend.evaluations) == count, "未启动槽位重放不符。")
    selected = controller.run_final(state["population"])
    require(len(selected) == 10 and controller.generation == 6
            and [x["generation"] for x in log.items if x["event"] == "selection"] == [6], "末代筛选不符。")
    require(state["model_calls"] + len(backend.calls) <= 422
            and state["evaluation_starts"] + len(backend.evaluations) <= 378, "模拟超出预算。")
    print(json.dumps({"status": "passed", "prior_model_calls": state["model_calls"],
                      "prior_evaluations": state["evaluation_starts"], "unstarted_eoh": count,
                      "remaining_model_calls": len(backend.calls), "remaining_evaluations": len(backend.evaluations),
                      "formal_mutex_shared": True, "model_concurrency": 1, "evaluation_concurrency": 4}, ensure_ascii=False))


def resume():
    config = json.loads((previous.SOURCE / "config.json").read_text(encoding="utf-8"))
    matrix.verify_training_software(matrix.load_matrix()[0])
    check_backend = FixtureBackend()
    check_backend.cfg = config
    reconstruct(check_backend, MemoryLog())
    resources = memory_and_priority()
    require(resources["available_memory_bytes"] >= config["resources"]["min_available_system_memory_bytes"], "内存不足。")
    cli = probe.cli_command()
    version, login = probe.run_command(cli + ["--version"]), probe.run_command(cli + ["login", "status"])
    require(version.get("exit_code") == 0 and login.get("exit_code") == 0
            and "Logged in using ChatGPT" in (login.get("stdout", "") + login.get("stderr", "")), "CLI 登录状态不符。")
    run_dir = ROOT / "outputs/p1_control" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8] + "_resume")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("xb") as out:
        out.write((previous.SOURCE / "config.json").read_bytes())
    log = JsonLog(run_dir / "events.jsonl")
    runtime = previous.ResumeRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    previous.configure_model_concurrency(runtime)
    controller, state = reconstruct(runtime, log)
    for path in [*(Path(x) for x in state["prior_hashes"]), *state["protected"], Path(__file__).resolve()]:
        if str(path) not in runtime.protected_hashes:
            runtime.protected.append(path)
            runtime.protected_hashes[str(path)] = digest(path)
    budget = runtime.budget
    budget.cfg = previous.base.extended_budget_config(config)
    budget.model_normal = state["model_calls"] - 1
    budget.eval_starts = budget.eval_normal = state["evaluation_starts"]
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
              code_sha256=runtime.protected_hashes, selection_seed=config["task"]["selection_seed"],
              generations=6, continuation=True, normal_model_slots_used=state["model_calls"] - 1,
              physical_model_cli_invocations_used=state["model_calls"], evaluation_starts_used=state["evaluation_starts"],
              evaluation_cpu_seconds_used=budget.cpu_finished, authorized_physical_call_cap=422,
              frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4)
    print(f"本方法第 6 代断点链检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(state, run_dir, log):
            log.event("resume_checkpoint", interrupted_run=str(state["latest"]), generation=6,
                      unstarted_candidates=[x.candidate_id for x in state["pending"]],
                      model_calls_already_used=state["model_calls"], evaluation_starts_already_used=state["evaluation_starts"],
                      total_model_call_cap=422)
            run_pending(controller, state)
            selected = controller.run_final(state["population"])
            require(len(selected) == 10 and controller.generation == 6, "首组六代未完成。")
        print(f"本方法首组六代完成；记录：{run_dir}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"本方法首组续跑停止：{exc}；记录：{run_dir}", file=sys.stderr, flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--simulate", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--user-requested-resume", action="store_true")
    args = parser.parse_args()
    if args.simulate:
        simulate()
        return 0
    if args.run:
        require(args.user_requested_resume, "正式续跑须记录用户指示。")
        return resume()
    backend, log = FixtureBackend(), MemoryLog()
    backend.cfg = json.loads((previous.SOURCE / "config.json").read_text(encoding="utf-8"))
    _, state = reconstruct(backend, log)
    print(json.dumps({"status": "passed", "model_calls_used": state["model_calls"],
                      "evaluation_starts": state["evaluation_starts"], "unstarted_eoh": len(state["pending"]),
                      "latest": str(state["latest"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
