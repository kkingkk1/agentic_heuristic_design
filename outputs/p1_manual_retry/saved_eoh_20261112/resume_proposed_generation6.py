"""从首组第 6 代未启动的 EoH 槽位续跑，保留失败和已完成的首次评价。"""

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
SOURCE = ROOT / "outputs/p1_control/20261112_230214_43fec153_resume"
SAVED = RECOVERY / "20261112_230934_proposed_9dcc64a8_evaluation"
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate, p1_formal_matrix as matrix
import scripts.p1_classified_runtime as classified_runtime
from scripts.p0_controller import Candidate, StopRun, check_modification_scope, timing_from_result
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from scripts.p1_classified_runtime import ClassifiedTaskRuntime

spec = importlib.util.spec_from_file_location(
    "proposed_first_generation6_base",
    ROOT / "outputs/p1_manual_retry/fusion_2_generation3_20261112_recovery/prepare_generation6.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
require = base.require
read_events = base.read_jsonl


def configure_model_concurrency(runtime):
    resources = runtime.cfg["resources"]
    require(resources["max_concurrent_model_calls"] == 4 and resources["max_concurrent_evaluations"] == 4,
            "冻结的并发合同不是预期的模型 4、评价 4。")
    runtime.model_slots = threading.BoundedSemaphore(1)


class ResumeController(base.FinalController):
    def __init__(self, *args, **kwargs):
        self._replay_eoh = None
        self._replay_specs = None
        super().__init__(*args, **kwargs)

    def batch(self, specs):
        if self._resume_population is None and self._replay_eoh is not None:
            require(len(specs) == len(self._replay_specs) == 40, "第 6 代 EoH 重放槽位不符。")
            for actual, expected in zip(specs, self._replay_specs):
                stage, parents, prompt = actual
                require((stage, [x.candidate_id for x in parents], prompt) == expected,
                        f"第 6 代 EoH 提示或父代不符：{stage}")
            candidates, self._replay_eoh = self._replay_eoh, None
            self.event("reused_eoh_proposals", generation=6, source_events=str(SOURCE / "events.jsonl"),
                       saved_evaluation_events=str(SAVED / "evaluation_events.jsonl"))
            return candidates
        return super().batch(specs)


class ResumeRuntime(ClassifiedTaskRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resume_first_generation = True

    def before_generation(self):
        if self.resume_first_generation:
            self.resume_first_generation = False
            self.log.event("reused_generation_budget_check", generation=6, source_events=str(SOURCE / "events.jsonl"))
            return
        return super().before_generation()


def reconstruct(backend, log):
    with patch.object(base, "FinalController", ResumeController):
        controller, earlier = base.reconstruct(backend, log)
    config = earlier["config"]
    require((SOURCE / "config.json").read_bytes() == (base.INTERRUPTED / "config.json").read_bytes()
            == (SAVED / "config.json").read_bytes(), "续跑配置不一致。")
    events = read_events(SOURCE / "events.jsonl")
    saved = read_events(SAVED / "evaluation_events.jsonl")
    summary = json.loads((SAVED / "summary.json").read_text(encoding="utf-8"))
    require([x["event_index"] for x in events] == list(range(1, len(events) + 1))
            and [x["event_index"] for x in saved] == list(range(1, len(saved) + 1)), "事件序号不连续。")
    preflight = [x for x in events if x["event"] == "preflight"]
    saved_preflight = [x for x in saved if x["event"] == "preflight"]
    starts = [x for x in events if x["event"] == "formal_run_started"]
    require(len(preflight) == len(saved_preflight) == len(starts) == 1
            and starts[0]["resume_of"] == str(base.INTERRUPTED)
            and starts[0]["interrupted_events_sha256"] == digest(base.INTERRUPTED / "events.jsonl")
            and starts[0]["total_model_call_cap"] == 422
            and events[-1]["event"] == "preflight_or_run_stopped", "第 6 代断点身份不符。")
    require(summary["status"] == "completed" and summary["new_model_calls"] == 0
            and summary["new_evaluations"] == 3 and summary["prior_model_calls"] == 361
            and summary["source_events_sha256"] == digest(SOURCE / "events.jsonl"), "首次评价身份不符。")
    for row in (preflight[0], saved_preflight[0]):
        for name, expected in row["code_sha256"].items():
            require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
    failures = [x for x in events if x["event"] == "model_failure_classified"]
    processes = [x for x in events if x["event"] == "model_process" and x["stage"] == "eoh_E1_02"]
    require(len(failures) == len(processes) == 1 and failures[0]["stage"] == "eoh_E1_02"
            and failures[0]["category"] == "unknown" and "Bad Request" in processes[0]["result"]["stdout"]
            and len([x for x in events if x["event"] == "model_attempt"]) == 4,
            "模型失败或调用数不符。")
    registrations = [x for x in events if x["event"] == "candidate_registered"]
    responses = {x["candidate_id"]: x for x in events if x["event"] == "proposal_response"}
    evaluations = {x["candidate_id"]: x["result"] for x in saved if x["event"] == "evaluation"}
    skipped = [x["candidate_id"] for x in events if x["event"] == "candidate_skipped"]
    require(len(registrations) == 40 and set(responses) == set(evaluations) == {"c0323", "c0325", "c0326"}
            and skipped == [f"c{i:04d}" for i in range(327, 363)], "保存响应或未启动槽位不符。")
    candidates = {x.candidate_id: x for x in controller.candidates}
    for row in registrations:
        ident, seq = row["candidate_id"], row["sequence"]
        require(ident == f"c{seq:04d}" and seq == len(controller.candidates) + 1, "候选编号不连续。")
        candidate = Candidate(ident, seq, row["stage"], tuple(row["parents"]))
        if ident == "c0324":
            candidate.error = "模型返回 Bad Request；保留失败，不自动重试。"
        controller.candidates.append(candidate)
        candidates[ident] = candidate
    for ident, row in responses.items():
        candidate, result = candidates[ident], evaluations[ident]
        parsed = controller.parse_proposal(row["response"])
        check_modification_scope(parsed["code"])
        candidate.thought, candidate.code = parsed["thought"], parsed["code"]
        candidate.generation_order = row["generation_order"]
        directory = SAVED / "candidates" / ident
        require(json.loads((directory / "result.json").read_text(encoding="utf-8")) == result
                and (directory / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8") == candidate.code + "\n",
                f"候选或评价记录不一致：{ident}")
        if result["status"] != "passed":
            candidate.error = str(result.get("error") or result["status"])
            continue
        checked = p0_tsp_aco_evaluate.read_complete(directory / "stdout.txt")
        require(checked["average"] == result["average"] and checked["fitness"] == result["fitness"]
                and checked["objectives"] == result["objectives"], f"评价输出不一致：{ident}")
        candidate.objective = candidate.raw_objective = float(result["average"])
        candidate.fitness = float(result["fitness"])
        candidate.timing_seconds, candidate.timing_scope = timing_from_result(result)
        candidate.completed_order = result["completed_monotonic_ns"]
        candidate.evaluation_origin = ident
    require(sorted(x["generation_order"] for x in responses.values()) == [320, 321, 322], "生成顺序不连续。")
    controller._generated = 322
    require(max(c.completed_order for c in controller.candidates if c.valid) < time.monotonic_ns(),
            "当前单调时钟早于历史评价。")
    attempts = {x["stage"]: x for x in events if x["event"] == "model_attempt"}
    directions = {"E1": "Create a heuristic as different as possible from these parents.",
                  "E2": "Identify their shared idea and create a new structure based on it.",
                  "M1": "Modify this heuristic's structure to improve performance.",
                  "M2": "Adjust parameters while retaining this heuristic's overall logic."}
    random_state = controller.random.getstate()
    specs = []
    for row in registrations:
        op = row["stage"].split("_")[1]
        parents = controller.eoh_parents(earlier["population"], 5 if op in ("E1", "E2") else 1)
        require([x.candidate_id for x in parents] == row["parents"], f"随机父代不符：{row['candidate_id']}")
        context = "\n".join(f"Algorithm description: {x.thought}\nCode:\n{x.code}\n" for x in parents)
        prompt = f"{controller.task}\n{context}{directions[op]}\n{controller.output}"
        if row["stage"] in attempts:
            require(attempts[row["stage"]]["prompt"] == prompt, f"原提示不符：{row['stage']}")
        specs.append((row["stage"], row["parents"], prompt))
    controller.random.setstate(random_state)
    controller._replay_eoh = [candidates[x["candidate_id"]] for x in registrations]
    controller._replay_specs = specs
    cpu = [x["accounting"]["cpu_seconds"] for x in saved if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == 3 and len(earlier["cpu_samples"]) == 318, "累计评价 CPU 不符。")
    return controller, {"config": config, "population": earlier["population"],
                        "pending": [candidates[x] for x in skipped], "pending_specs": specs[4:],
                        "cpu_samples": earlier["cpu_samples"] + cpu,
                        "prior_hashes": preflight[0]["code_sha256"],
                        "saved_hashes": saved_preflight[0]["code_sha256"]}


def run_pending(controller, state):
    require(len(state["pending"]) == len(state["pending_specs"]) == 36, "待执行槽位数量不符。")
    controller.generation = 6
    entries = []
    for candidate, (stage, parents, prompt) in zip(state["pending"], state["pending_specs"]):
        require(candidate.stage == stage and list(candidate.parents) == parents, "待执行候选或父代不符。")
        controller.event("reused_candidate_registration", generation=6, candidate_id=candidate.candidate_id,
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
    controller.generation = 5


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
            other = json.loads(path.read_text(encoding="utf-8"))
            if (other.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                    or matrix.run_identity(other) != matrix.run_identity(config)):
                continue
            event_file = path.parent / "events.jsonl"
            if event_file.is_file() and any(x["event"] == "formal_run_started" for x in read_events(event_file)):
                started.add(path.parent.resolve())
        expected = {base.base.SOURCE, base.generation3.previous.INTERRUPTED,
                    base.generation3.INTERRUPTED, base.INTERRUPTED, SOURCE}
        require(started == {x.resolve() for x in expected}, "首组存在未知的已启动运行。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=config["formal_replicate"], selection_seed=task["selection_seed"],
                  resume_of=str(SOURCE), interrupted_events_sha256=digest(SOURCE / "events.jsonl"),
                  saved_evaluation_sha256=digest(SAVED / "evaluation_events.jsonl"),
                  total_model_call_cap=422, extra_model_cli_invocations=2,
                  frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4,
                  note="保留 Bad Request；复用首次评价，仅执行未启动的原槽位。")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def verify_serial_runtime():
    """比较同一运行时在原设置和单并发设置下的离线请求峰值。"""
    from tests.test_p0_controller import MemoryLog

    runtime = ResumeRuntime(SOURCE, SOURCE / "config.json", MemoryLog(), cli=["offline-fixture"])

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


def simulate():
    from tests.test_p0_controller import FixtureBackend, MemoryLog
    backend, log = FixtureBackend(), MemoryLog()
    controller, state = reconstruct(backend, log)
    verify_serial_runtime()
    with recovery_guard(state["config"], RECOVERY / "unused_guard_fixture", MemoryLog()):
        code = ("from pathlib import Path\nfrom scripts import p1_formal_matrix as m\ntry:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED", "续跑互斥锁不一致。")
    run_pending(controller, state)
    require(len(backend.calls) == 36 and len(backend.evaluations) == 36
            and all(stage != "eoh_E1_02" for stage, _ in backend.calls), "未启动槽位或失败保留不符。")
    selected = controller.run_final(state["population"])
    require(len(selected) == 10 and controller.generation == 6 and len(backend.calls) == 61
            and len(backend.evaluations) == 54
            and [x["generation"] for x in log.items if x["event"] == "selection"] == [6],
            "第 6 代预算槽位或筛选不符。")
    print(json.dumps({"status": "passed", "prior_model_calls": 361, "prior_evaluations": 321,
                      "unstarted_eoh": 36, "remaining_model_calls": 61,
                      "remaining_evaluation_slots": 54, "model_call_cap": 422,
                      "formal_mutex_shared": True, "model_concurrency_before": 4,
                      "model_concurrency_after": 1, "evaluation_concurrency": 4}, ensure_ascii=False))


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
    run_dir = ROOT / "outputs/p1_control" / (datetime.now().strftime("%Y%m%d_%H%M%S")
                                            + "_" + uuid.uuid4().hex[:8] + "_resume")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("xb") as out:
        out.write((SOURCE / "config.json").read_bytes())
    log = JsonLog(run_dir / "events.jsonl")
    runtime = ResumeRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    configure_model_concurrency(runtime)
    controller, state = reconstruct(runtime, log)
    for path in [*(Path(x) for x in state["prior_hashes"]), *(Path(x) for x in state["saved_hashes"]),
                 SOURCE / "events.jsonl", SOURCE / "config.json", SAVED / "evaluation_events.jsonl",
                 SAVED / "summary.json", Path(__file__).resolve()]:
        if str(path) not in runtime.protected_hashes:
            runtime.protected.append(path)
            runtime.protected_hashes[str(path)] = digest(path)
    budget = runtime.budget
    budget.cfg = base.extended_budget_config(config)
    budget.model_normal = 360
    budget.eval_starts = budget.eval_normal = 321
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
              code_sha256=runtime.protected_hashes, selection_seed=config["task"]["selection_seed"],
              generations=6, continuation=True, normal_model_slots_used=360,
              physical_model_cli_invocations_used=361, evaluation_starts_used=321,
              evaluation_cpu_seconds_used=budget.cpu_finished, authorized_physical_call_cap=422,
              frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4)
    print(f"本方法第 6 代恢复检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(config, run_dir, log):
            log.event("resume_checkpoint", interrupted_run=str(SOURCE), generation=6,
                      failed_candidate_id="c0324", saved_evaluation_run=str(SAVED),
                      unstarted_candidates=[x.candidate_id for x in state["pending"]],
                      model_calls_already_used=361, evaluation_starts_already_used=321,
                      total_model_call_cap=422)
            run_pending(controller, state)
            selected = controller.run_final(state["population"])
            require(len(selected) == 10 and controller.generation == 6, "首组第 6 代未完成。")
        print(f"本方法首组六代完成；记录：{run_dir}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"本方法第 6 代续跑停止：{exc}；记录：{run_dir}", file=sys.stderr, flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--simulate", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--service-restored", action="store_true", help="仅在最新限流解除后启用模型调用")
    args = parser.parse_args()
    if args.simulate:
        simulate()
        return 0
    if args.run:
        require(args.service_restored, "服务再次返回 HTTP 429；限流解除前不能继续模型调用。")
        return resume()
    class CheckBackend:
        cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    controller, state = reconstruct(CheckBackend(), None)
    print(json.dumps({"status": "passed", "generation": controller.generation,
                      "model_calls_used": 361, "evaluation_starts": len(state["cpu_samples"]),
                      "unstarted_eoh": len(state["pending"]), "total_model_call_cap": 422}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"本方法第 6 代恢复检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
