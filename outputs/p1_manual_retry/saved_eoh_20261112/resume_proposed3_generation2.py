"""核验本方法第 3 次研发的第 2 代断点，仅续跑未启动的 EoH 槽位。"""

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
import threading
import time
import uuid
from unittest.mock import patch


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
SOURCE = ROOT / "outputs/p1_control/20261113_151506_7fc2acab"
BASE_PATH = ROOT / "outputs/p1_manual_retry/proposed_2_generation2_20261112_recovery/resume_proposed_generation2.py"
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate, p1_formal_matrix as matrix
from scripts import p1_classified_runtime
from scripts.p0_controller import Candidate, StopRun, check_modification_scope, timing_from_result
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from scripts.p1_task_contract import load_task_contract
from tests.test_p0_controller import FixtureBackend, MemoryLog


spec = importlib.util.spec_from_file_location("proposed3_generation2_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
base.SOURCE = SOURCE
require, read_jsonl = base.require, base.read_jsonl


class RecoveryRuntime(base.RecoveryRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        resources = self.cfg["resources"]
        require(resources["max_concurrent_model_calls"] == resources["max_concurrent_evaluations"] == 4,
                "冻结并发合同变化。")
        self.model_slots = threading.BoundedSemaphore(1)


def reconstruct(backend, log):
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    matrix.verify_training_config(config)
    events = read_jsonl(SOURCE / "events.jsonl")
    require([x["event_index"] for x in events] == list(range(1, len(events) + 1)), "事件编号不连续。")
    started = [x for x in events if x["event"] == "formal_run_started"]
    preflights = [x for x in events if x["event"] == "preflight"]
    require(len(started) == len(preflights) == 1 and started[0]["method"] == config["method"]["id"]
            and started[0]["formal_replicate"] == config["formal_replicate"] == 3
            and config["method"]["id"] == "proposed_solution_quality_and_speed_agents"
            and started[0]["matrix_contract_sha256"] == config["matrix_contract_sha256"]
            and started[0]["selection_seed"] == config["task"]["selection_seed"]
            and events[-1]["event"] == "preflight_or_run_stopped", "研发身份或停止状态不符。")
    for name, expected in preflights[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
    failures = [x for x in events if x["event"] == "model_failure_classified"]
    failed_processes = [x for x in events if x["event"] == "model_process" and x["stage"] == "eoh_E1_01"
                        and x["event_index"] > next(y["event_index"] for y in events if y["event"] == "selection")]
    require(len(failures) == len(failed_processes) == 1 and failures[0]["category"] == "unknown"
            and '"detail":"Bad Request"' in failed_processes[0]["result"]["stdout"].replace('\\"', '"'),
            "断点不是预期的 Bad Request。")
    registrations = [x for x in events if x["event"] == "candidate_registered"]
    responses = {x["candidate_id"]: x for x in events if x["event"] == "proposal_response"}
    evaluations = {x["candidate_id"]: x["result"] for x in events if x["event"] == "evaluation"}
    eval_ids = [x["candidate_id"] for x in events if x["event"] == "evaluation_budget_reserved"]
    attempts = [x for x in events if x["event"] == "model_attempt"]
    require(len(registrations) == 128 and len(responses) == len(evaluations) == len(eval_ids) == 88
            and set(responses) == set(evaluations) == set(eval_ids) and len(set(eval_ids)) == len(eval_ids)
            and len(attempts) == 96 and sorted(x["generation_order"] for x in responses.values()) == list(range(1, 89)),
            "历史候选、模型调用或评价预算不符。")
    task = config["task"]
    controller = base.RecoveryController(backend, log, population_size=task["population_size"],
                                         initial_count=task["initial_proposals"], rounds=task["collaboration_rounds"],
                                         seed=task["selection_seed"], generations=task["generations"],
                                         task_contract=load_task_contract("tsp_aco"))
    candidates = {}
    for row in registrations:
        ident, sequence = row["candidate_id"], row["sequence"]
        require(ident == f"c{sequence:04d}" and sequence == len(controller.candidates) + 1, "候选编号不连续。")
        candidate = Candidate(ident, sequence, row["stage"], tuple(row["parents"]))
        controller.candidates.append(candidate)
        candidates[ident] = candidate
        if ident not in responses:
            require(sequence >= 89, "历史候选缺少响应。")
            if sequence == 89:
                candidate.error = "模型请求返回 Bad Request；不自动重试。"
            continue
        response, result = responses[ident], evaluations[ident]
        parsed = controller.parse_proposal(response["response"])
        check_modification_scope(parsed["code"])
        candidate.thought, candidate.code = parsed["thought"], parsed["code"]
        candidate.generation_order = response["generation_order"]
        directory = SOURCE / "candidates" / ident
        require(json.loads((directory / "result.json").read_text(encoding="utf-8")) == result
                and (directory / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8") == candidate.code + "\n",
                f"候选代码或评价记录不符：{ident}")
        if result["status"] != "passed":
            require(result["status"] == "candidate_error", f"存在其他评价中断：{ident}")
            candidate.error = str(result.get("error") or result["status"])
            continue
        checked = p0_tsp_aco_evaluate.read_complete(directory / "stdout.txt")
        require(all(checked[key] == result[key] for key in ("average", "fitness", "objectives")),
                f"评价输出不符：{ident}")
        candidate.objective = candidate.raw_objective = float(result["average"])
        candidate.fitness = float(result["fitness"])
        candidate.timing_seconds, candidate.timing_scope = timing_from_result(result)
        candidate.completed_order = result["completed_monotonic_ns"]
        candidate.evaluation_origin = ident
    controller._generated = len(responses)
    require(max(c.completed_order for c in controller.candidates if c.valid) < time.monotonic_ns(),
            "当前单调时钟早于历史评价。")
    initial = controller.select(controller.candidates[:30], controller.n)
    initial_rows = [x for x in events if x["event"] == "initial_population"]
    require(len(initial_rows) == 1 and [x.candidate_id for x in initial] == initial_rows[0]["candidate_ids"],
            "初始种群不一致。")
    selection_rows = [x for x in events if x["event"] == "selection"]
    require(len(selection_rows) == 1 and selection_rows[0]["generation"] == 1, "第 1 代筛选不完整。")
    selection = selection_rows[0]
    first_attempts = {x["stage"]: x for x in attempts if x["stage"].startswith("eoh_")
                      and x["event_index"] < selection["event_index"]}
    directions = {"E1": "Create a heuristic as different as possible from these parents.",
                  "E2": "Identify their shared idea and create a new structure based on it.",
                  "M1": "Modify this heuristic's structure to improve performance.",
                  "M2": "Adjust parameters while retaining this heuristic's overall logic."}

    def build_specs(rows, population, checked_attempts):
        specs = []
        for row in rows:
            operator = row["stage"].split("_")[1]
            parents = controller.eoh_parents(population, 5 if operator in ("E1", "E2") else 1)
            require([x.candidate_id for x in parents] == row["parents"], f"随机父代不符：{row['candidate_id']}")
            context = "\n".join(f"Algorithm description: {x.thought}\nCode:\n{x.code}\n" for x in parents)
            prompt = f"{controller.task}\n{context}{directions[operator]}\n{controller.output}"
            if row["stage"] in checked_attempts:
                require(checked_attempts[row["stage"]]["prompt"] == prompt, f"原提示不符：{row['candidate_id']}")
            specs.append((row["stage"], row["parents"], prompt))
        return specs

    require(len(first_attempts) == 40, "第 1 代 EoH 请求不完整。")
    build_specs(registrations[30:70], initial, first_attempts)
    a, b = controller.elite_pair(initial)
    pair = [x for x in events if x["event"] == "elite_pair"]
    require(len(pair) == 1 and (a.candidate_id, b.candidate_id) == (pair[0]["a"], pair[0]["b"]),
            "第 1 代双精英不一致。")
    selected = controller.select([candidates[x] for x in selection["pool"]], controller.n)
    require([{"candidate_id": x.candidate_id, "evaluation_origin": x.evaluation_origin} for x in selected]
            == selection["selected"], "第 1 代筛选不一致。")
    history = [x for x in events if x["event"] == "history_appended"]
    require(len(history) == 3 and {x["role"] for x in history} == set(controller.history), "角色历史不完整。")
    for row in history:
        controller.history[row["role"]].append(row["reflection"])
    random_state = controller.random.getstate()
    second_attempts = {x["stage"]: x for x in attempts if x["event_index"] > selection["event_index"]}
    require(list(second_attempts) == ["eoh_E1_01"], "第 2 代实际启动槽位不符。")
    specs = build_specs(registrations[88:], selected, second_attempts)
    controller.random.setstate(random_state)
    controller._replay_eoh = [candidates[x["candidate_id"]] for x in registrations[88:]]
    controller._replay_specs = specs
    cpu = [x["accounting"]["cpu_seconds"] for x in events if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == 88 and all(candidates[f"c{i:04d}"].stage not in second_attempts for i in range(90, 129)),
            "CPU 记录或未启动槽位不符。")
    return controller, {"config": config, "population": selected, "pending": controller._replay_eoh[1:],
                        "pending_specs": specs[1:], "cpu_samples": cpu, "prior_hashes": preflights[0]["code_sha256"]}


def run_pending(controller, state):
    require(len(state["pending"]) == len(state["pending_specs"]) == 39, "未启动槽位数量不符。")
    controller.generation = 2
    entries = []
    for candidate, (stage, parents, prompt) in zip(state["pending"], state["pending_specs"]):
        require(candidate.stage == stage and list(candidate.parents) == parents, "待执行槽位不符。")
        controller.event("reused_candidate_registration", generation=2, candidate_id=candidate.candidate_id,
                         stage=stage, parents=parents, source_events=str(SOURCE / "events.jsonl"))
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
    controller.generation = 0


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
        require(started == {SOURCE.resolve()}, "同一研发存在未知已启动运行。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=3, selection_seed=task["selection_seed"], resume_of=str(SOURCE),
                  interrupted_events_sha256=digest(SOURCE / "events.jsonl"), total_model_call_cap=420,
                  frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4,
                  note="保留失败，仅执行未启动的原槽位。")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def simulate():
    backend, log = FixtureBackend(), MemoryLog()
    backend.cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    controller, state = reconstruct(backend, log)
    with recovery_guard(state["config"], RECOVERY / "unused_proposed3_fixture", MemoryLog()):
        code = ("from pathlib import Path\nfrom scripts import p1_formal_matrix as m\ntry:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED", "研发互斥锁不一致。")
    run_pending(controller, state)
    require(len(backend.calls) == len(backend.evaluations) == 39
            and all(stage != "eoh_E1_01" for stage, _ in backend.calls), "未启动槽位模拟不符。")
    selected = controller.run_remaining(state["population"])
    require(len(selected) == 10 and controller.generation == 6 and len(backend.calls) == 324
            and len(backend.evaluations) == 289
            and [x["generation"] for x in log.items if x["event"] == "selection"] == [2, 3, 4, 5, 6],
            "后续代次或预算模拟不符。")
    serial_runtime = RecoveryRuntime(SOURCE, SOURCE / "config.json", MemoryLog(), cli=["offline-fixture"])
    counts, lock = {"active": 0, "peak": 0}, threading.Lock()

    def fake_success(command, prompt, cwd=None):
        with lock:
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
        time.sleep(0.05)
        with lock:
            counts["active"] -= 1
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    with patch.object(probe, "run_command", side_effect=fake_success), \
            patch.object(probe, "extract_message", return_value=("offline", {}, [], [])):
        with ThreadPoolExecutor(max_workers=4) as executor:
            output = list(executor.map(lambda i: serial_runtime.model(f"fixture_{i}", "offline"), range(4)))
    require(output == ["offline"] * 4 and counts["peak"] == 1, "实际模型并发不为 1。")
    limited_log = MemoryLog()
    runtime = RecoveryRuntime(SOURCE, SOURCE / "config.json", limited_log, cli=["offline-fixture"])
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
            "限流后仍启动其他模型请求。")
    acquired = [runtime.eval_slots.acquire(blocking=False) for _ in range(5)]
    require(acquired == [True, True, True, True, False], "评价并发上限变化。")
    for _ in range(4):
        runtime.eval_slots.release()
    print(json.dumps({"status": "passed", "prior_model_calls": 96, "prior_evaluations": 88,
                      "unstarted_eoh": 39, "remaining_model_calls": 324, "remaining_evaluations": 289,
                      "formal_mutex_shared": True, "model_concurrency": 1, "evaluation_concurrency": 4,
                      "stop_after_429": True}, ensure_ascii=False))


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
    runtime = RecoveryRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    controller, state = reconstruct(runtime, log)
    for path in [*(Path(x) for x in state["prior_hashes"]), SOURCE / "events.jsonl", SOURCE / "config.json",
                 BASE_PATH, Path(base.previous.__file__).resolve(), Path(__file__).resolve()]:
        if str(path) not in runtime.protected_hashes:
            runtime.protected.append(path)
            runtime.protected_hashes[str(path)] = digest(path)
    budget = runtime.budget
    budget.model_normal = 96
    budget.eval_starts = budget.eval_normal = 88
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
              code_sha256=runtime.protected_hashes, selection_seed=config["task"]["selection_seed"],
              generations=6, continuation=True, normal_model_slots_used=96, physical_model_cli_invocations_used=96,
              evaluation_starts_used=88, evaluation_cpu_seconds_used=budget.cpu_finished,
              frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4)
    print(f"本方法第 3 次研发断点检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(config, run_dir, log):
            log.event("resume_checkpoint", interrupted_run=str(SOURCE), generation=2, failed_candidate_id="c0089",
                      unstarted_candidates=[x.candidate_id for x in state["pending"]],
                      model_calls_already_used=96, evaluation_starts_already_used=88, total_model_call_cap=420)
            run_pending(controller, state)
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
        print(json.dumps({"status": "passed", "model_calls_used": 96, "evaluation_starts": len(state["cpu_samples"]),
                          "unstarted_eoh": len(state["pending"]), "model_call_cap": 420}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"本方法第 3 次研发恢复检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
