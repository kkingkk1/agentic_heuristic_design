"""核验RoCo 第 3 次研发的第 3 代断点链，继续未启动的 EoH 槽位。"""

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
import uuid
from unittest.mock import patch


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
PREVIOUS_SCRIPT = RECOVERY / "resume_roco3_generation3.py"
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate, p1_formal_matrix as matrix
from scripts import p1_classified_runtime
from scripts.p0_controller import StopRun, check_modification_scope
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from tests.test_p0_controller import FixtureBackend, MemoryLog


spec = importlib.util.spec_from_file_location("roco3_generation3_recovery", PREVIOUS_SCRIPT)
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
require, read_jsonl = previous.require, previous.read_events


def find_segments(config):
    children = {}
    for path in (ROOT / "outputs/p1_roco").glob("*/config.json"):
        other = json.loads(path.read_text(encoding="utf-8"))
        if (other.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                or matrix.run_identity(other) != matrix.run_identity(config)):
            continue
        event_file = path.parent / "events.jsonl"
        if not event_file.is_file():
            continue
        rows = read_jsonl(event_file)
        starts = [x for x in rows if x["event"] == "formal_run_started"]
        if starts and starts[0].get("resume_of"):
            require(len(starts) == 1, "续跑身份不唯一。")
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
    require(segments, "未找到第 3 代后续断点。")
    return segments


def apply_result(controller, candidate, response, result, directory):
    parsed = controller.parse_proposal(response["response"])
    check_modification_scope(parsed["code"])
    candidate.thought, candidate.code = parsed["thought"], parsed["code"]
    candidate.generation_order = response["generation_order"]
    require(json.loads((directory / "result.json").read_text(encoding="utf-8")) == result
            and (directory / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8") == candidate.code + "\n",
            f"候选或评价记录不符：{candidate.candidate_id}")
    if result["status"] != "passed":
        require(result["status"] == "candidate_error" or
                (result["status"] == "infrastructure_error" and result.get("error") == "其他并发任务停止运行。"),
                f"存在未知评价基础设施故障：{candidate.candidate_id}")
        candidate.error = str(result.get("error") or result["status"])
        return
    checked = p0_tsp_aco_evaluate.read_complete(directory / "stdout.txt")
    require(all(checked[key] == result[key] for key in ("average", "fitness", "objectives")),
            f"评价输出不符：{candidate.candidate_id}")
    candidate.objective = float(result["average"])
    candidate.fitness = float(result["fitness"])
    candidate.completed_order = result["completed_monotonic_ns"]
    candidate.evaluation_origin = candidate.candidate_id


def reconstruct(backend, log):
    config = json.loads((previous.SOURCE / "config.json").read_text(encoding="utf-8"))
    earlier_segments = previous.previous.find_segments(config)
    positions = [i for i, (directory, _) in enumerate(earlier_segments)
                 if directory.resolve() == previous.SOURCE.resolve()]
    require(len(positions) == 1, "无法唯一定位第 3 代来源断点。")
    with patch.object(previous.previous, "find_segments", return_value=earlier_segments[:positions[0] + 1]):
        controller, state = previous.reconstruct(backend, log)
    config = state["config"]
    segments = find_segments(config)
    pending = list(state["pending"])
    specs = {c.candidate_id: item for c, item in zip(pending, state["pending_specs"])}
    model_calls, evaluation_starts = state["model_calls"], state["evaluation_starts"]
    cpu_samples, prior_hashes = list(state["cpu_samples"]), dict(state["prior_hashes"])
    protected, parent = [], previous.SOURCE
    failures_without_progress = []
    for directory, rows in segments:
        require((directory / "config.json").read_bytes() == (previous.SOURCE / "config.json").read_bytes()
                and [x["event_index"] for x in rows] == list(range(1, len(rows) + 1)), "断点配置或事件编号不符。")
        started = [x for x in rows if x["event"] == "formal_run_started"]
        preflight = [x for x in rows if x["event"] == "preflight"]
        require(len(started) == len(preflight) == 1 and started[0]["resume_of"] == str(parent)
                and started[0]["interrupted_events_sha256"] == digest(parent / "events.jsonl")
                and started[0]["total_model_call_cap"] == 420 and started[0]["effective_model_concurrency"] == 1
                and preflight[0]["physical_model_cli_invocations_used"] == model_calls
                and preflight[0]["evaluation_starts_used"] == evaluation_starts, "断点链、并发或预算不符。")
        for name, expected in preflight[0]["code_sha256"].items():
            require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
            prior_hashes[name] = expected
        registrations = [x["candidate_id"] for x in rows if x["event"] == "reused_candidate_registration"]
        require(registrations == [x.candidate_id for x in pending], "待执行槽位不符。")
        by_stage = {x.stage: x for x in pending}
        attempts = [x for x in rows if x["event"] == "model_attempt"]
        require(len({x["stage"] for x in attempts}) == len(attempts)
                and all(x["stage"] in by_stage for x in attempts), "存在重复或非 EoH 模型调用。")
        for attempt in attempts:
            candidate = by_stage[attempt["stage"]]
            stage, parents, prompt = specs[candidate.candidate_id]
            require(candidate.stage == stage and list(candidate.parents) == parents and attempt["prompt"] == prompt,
                    f"父代或提示不符：{candidate.candidate_id}")
        responses = {x["candidate_id"]: x for x in rows if x["event"] == "proposal_response"}
        results = {x["candidate_id"]: x["result"] for x in rows if x["event"] == "evaluation"}
        invalid = {x["candidate_id"]: x["error"] for x in rows if x["event"] == "candidate_invalid"}
        eval_ids = [x["candidate_id"] for x in rows if x["event"] == "evaluation_budget_reserved"]
        require(len(eval_ids) == len(set(eval_ids)) and set(eval_ids) == set(results)
                and not set(results) & set(invalid) and set(responses) == set(results) | set(invalid),
                "响应与首次评价不一致；可能需要另行处理已返回响应。")
        orders = sorted(x["generation_order"] for x in responses.values())
        require(orders == list(range(controller._generated + 1, controller._generated + len(orders) + 1)),
                "生成顺序不连续。")
        controller._generated += len(orders)
        candidates = {x.candidate_id: x for x in pending}
        for ident, response in responses.items():
            require(ident in candidates and response["generation"] == 3, "续跑响应候选不符。")
            if ident in invalid:
                candidates[ident].generation_order = response["generation_order"]
                candidates[ident].error = invalid[ident]
            else:
                apply_result(controller, candidates[ident], response, results[ident], directory / "candidates" / ident)
        failures = [x for x in rows if x["event"] == "model_failure_classified"]
        require(failures, "续跑停止但缺少模型失败分类。")
        for failure in failures:
            require(failure["stage"] in by_stage and failure["category"] in ("network", "unknown"),
                    "遇到限流、审核或未知停止原因，不能自动续跑。")
            if failure["category"] == "unknown":
                processes = [x for x in rows if x["event"] == "model_process" and x["stage"] == failure["stage"]]
                require(len(processes) == 1 and "Bad Request" in processes[0]["result"].get("stdout", ""),
                        "未知模型错误不是已核实的 Bad Request。")
            by_stage[failure["stage"]].error = f"模型请求失败（{failure['category']}）；不自动重试。"
        attempted_ids = {by_stage[x["stage"]].candidate_id for x in attempts}
        require(attempted_ids == set(responses) | {by_stage[x["stage"]].candidate_id for x in failures},
                "模型调用没有对应响应或失败。")
        if not responses and not eval_ids:
            failures_without_progress.append(failures[-1]["category"])
        else:
            failures_without_progress.clear()
        require(not (len(failures_without_progress) >= 3 and len(set(failures_without_progress[-3:])) == 1),
                "同类请求连续三次无进展，停止当前恢复策略。")
        pending = [candidate for candidate in pending if candidate.candidate_id not in attempted_ids]
        cpu = [x["accounting"]["cpu_seconds"] for x in rows if x["event"] == "evaluation_cpu_final"]
        require(len(cpu) == len(eval_ids), "评价 CPU 记录不完整。")
        cpu_samples.extend(cpu)
        model_calls += len(attempts)
        evaluation_starts += len(eval_ids)
        require(model_calls <= 420 and evaluation_starts <= 378 and len(cpu_samples) == evaluation_starts,
                "累计预算不符。")
        protected.extend((directory / "events.jsonl", directory / "config.json"))
        parent = directory
    require(pending, "没有未启动的 EoH 槽位。")
    state.update(pending=pending, pending_specs=[specs[x.candidate_id] for x in pending], cpu_samples=cpu_samples,
                 prior_hashes=prior_hashes, protected=protected, segments=segments, latest=parent,
                 model_calls=model_calls, evaluation_starts=evaluation_starts)
    return controller, state


def run_pending(controller, state):
    controller._generation_offset = 0
    controller.generation = 3
    entries = []
    for candidate, (stage, parents, prompt) in zip(state["pending"], state["pending_specs"]):
        require(candidate.stage == stage and list(candidate.parents) == parents, "待执行槽位不符。")
        controller.event("reused_candidate_registration", generation=3, candidate_id=candidate.candidate_id,
                         stage=stage, parents=parents, source_events=str(state["latest"] / "events.jsonl"))
        entries.append((candidate, prompt))
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(controller.propose, candidate, prompt) for candidate, prompt in entries]
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                controller.stop.set()
                errors.append(exc)
        if errors:
            raise StopRun(f"并发阶段停止：{errors[0]}") from errors[0]


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
        raise OSError(ctypes.get_last_error(), "无法创建研发互斥锁。")
    acquired = False
    try:
        result = kernel.WaitForSingleObject(handle, 0)
        require(result in (0, 0x80), "已有正式研发运行或无法取得互斥锁。")
        acquired = True
        config = state["config"]
        matrix.verify_training_config(config)
        started = set()
        for path in (ROOT / "outputs/p1_roco").glob("*/config.json"):
            if path.parent.resolve() == run_dir.resolve():
                continue
            other = json.loads(path.read_text(encoding="utf-8"))
            if (other.get("matrix_contract_sha256") == config["matrix_contract_sha256"]
                    and matrix.run_identity(other) == matrix.run_identity(config)):
                event_path = path.parent / "events.jsonl"
                if event_path.is_file() and any(x["event"] == "formal_run_started" for x in read_jsonl(event_path)):
                    started.add(path.parent.resolve())
        expected = previous.lineage()
        expected.update(path.resolve() for path, _ in state["segments"])
        require(started == expected, "同一研发存在未知已启动运行。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=3, selection_seed=task["selection_seed"], resume_of=str(state["latest"]),
                  interrupted_events_sha256=digest(state["latest"] / "events.jsonl"), total_model_call_cap=420,
                  frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4,
                  note="保留失败和已启动评价，仅执行第 3 代未启动的原槽位。")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def simulate():
    backend, log = FixtureBackend(), MemoryLog()
    backend.cfg = json.loads((previous.SOURCE / "config.json").read_text(encoding="utf-8"))
    controller, state = reconstruct(backend, log)
    with recovery_guard(state, RECOVERY / "unused_roco3_generation3_chain_fixture", MemoryLog()):
        code = ("from pathlib import Path\nfrom scripts import p1_formal_matrix as m\ntry:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED", "研发互斥锁不一致。")
    run_pending(controller, state)
    require(len(backend.calls) == len(backend.evaluations) == len(state["pending"]), "未启动槽位模拟不符。")
    selected = controller.run_from_third(state["population"])
    require(len(selected) == 10 and controller.generation == 6
            and state["model_calls"] + len(backend.calls) <= 420
            and state["evaluation_starts"] + len(backend.evaluations) <= 378
            and [x["generation"] for x in log.items if x["event"] == "selection"] == [3, 4, 5, 6],
            "后续代次或预算模拟不符。")
    limited_log = MemoryLog()
    runtime = previous.RecoveryRuntime(previous.SOURCE, previous.SOURCE / "config.json", limited_log, cli=["offline-fixture"])
    limited_controller, limited_state = reconstruct(runtime, limited_log)
    with patch.object(probe, "run_command", return_value={"exit_code": 1, "stdout": "", "stderr": ""}), \
            patch.object(probe, "extract_message", side_effect=probe.ProbeError("offline 429")), \
            patch.object(p1_classified_runtime, "classify_failure",
                         return_value={"category": "quota", "source": "fixture", "plugin_warnings": []}):
        try:
            run_pending(limited_controller, limited_state)
        except StopRun:
            pass
        else:
            raise ValueError("限流夹具没有停止。")
    require(len([x for x in limited_log.items if x["event"] == "model_attempt"]) == 1,
            "限流后启动了其他模型请求。")
    print(json.dumps({"status": "passed", "prior_model_calls": state["model_calls"],
                      "prior_evaluations": state["evaluation_starts"], "unstarted_eoh": len(state["pending"]),
                      "remaining_model_calls": len(backend.calls), "remaining_evaluations": len(backend.evaluations),
                      "formal_mutex_shared": True, "model_concurrency": 1, "stop_after_429": True}, ensure_ascii=False))


def resume():
    config = json.loads((previous.SOURCE / "config.json").read_text(encoding="utf-8"))
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
    run_dir = ROOT / "outputs/p1_roco" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8] + "_resume")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("xb") as out:
        out.write((previous.SOURCE / "config.json").read_bytes())
    log = JsonLog(run_dir / "events.jsonl")
    runtime = previous.RecoveryRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    controller, state = reconstruct(runtime, log)
    for path in [*(Path(x) for x in state["prior_hashes"]), *state["protected"], PREVIOUS_SCRIPT, Path(__file__).resolve()]:
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
              evaluation_cpu_seconds_used=budget.cpu_finished,
              frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4)
    print(f"RoCo 第 3 次研发第 3 代断点链检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(state, run_dir, log):
            log.event("resume_checkpoint", interrupted_run=str(state["latest"]), generation=3,
                      unstarted_candidates=[x.candidate_id for x in state["pending"]],
                      model_calls_already_used=state["model_calls"], evaluation_starts_already_used=state["evaluation_starts"],
                      total_model_call_cap=420)
            run_pending(controller, state)
            selected = controller.run_from_third(state["population"])
            require(len(selected) == 10 and controller.generation == 6, "六代未完成。")
        print(f"RoCo 第 3 次研发六代完成；记录：{run_dir}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"RoCo 第 3 次研发续跑停止：{exc}；记录：{run_dir}", file=sys.stderr, flush=True)
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
        backend.cfg = json.loads((previous.SOURCE / "config.json").read_text(encoding="utf-8"))
        _, state = reconstruct(backend, MemoryLog())
        print(json.dumps({"status": "passed", "model_calls_used": state["model_calls"],
                          "evaluation_starts": state["evaluation_starts"], "unstarted_eoh": len(state["pending"]),
                          "latest": str(state["latest"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"RoCo 第 3 次研发第 3 代断点链检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
