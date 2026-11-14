"""从本方法第 2 次研发的未启动 EoH 槽位续跑，保留连接和评价失败。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes
from datetime import datetime
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
SOURCE = ROOT / "outputs/p1_control/20261112_195246_68d519ff"
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate, p1_formal_matrix as matrix
from scripts.p0_controller import Candidate, Controller, StopRun, check_modification_scope, timing_from_result
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from scripts.p1_classified_runtime import ClassifiedTaskRuntime, classify_failure
from scripts.p1_task_contract import load_task_contract


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class RecoveryController(Controller):
    def __init__(self, *args, **kwargs):
        self._replay_phase = 0
        self._replay_eoh = None
        self._replay_specs = None
        super().__init__(*args, **kwargs)

    def batch(self, specs):
        if self._replay_phase == 0:
            require(len(specs) == self.initial_count and all(stage.startswith("init_") and not parents
                    for stage, parents, _ in specs), "初始种群重放位置不符。")
            self._replay_phase = 1
            self.event("reused_initial_proposals", source_events=str(SOURCE / "events.jsonl"))
            return self.candidates[:self.initial_count]
        if self._replay_phase == 1:
            require(len(specs) == len(self._replay_specs) == 40, "EoH 重放槽位不符。")
            for actual, expected in zip(specs, self._replay_specs):
                stage, parents, prompt = actual
                require((stage, [x.candidate_id for x in parents], prompt) == expected,
                        f"EoH 重建提示或父代不符：{stage}")
            self._replay_phase = 2
            self.event("reused_eoh_proposals", generation=1, source_events=str(SOURCE / "events.jsonl"))
            return self._replay_eoh
        return super().batch(specs)


class RecoveryRuntime(ClassifiedTaskRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resume_first_generation = True

    def before_generation(self):
        if self.resume_first_generation:
            self.resume_first_generation = False
            self.log.event("reused_generation_budget_check", generation=1, source_events=str(SOURCE / "events.jsonl"))
            return
        return super().before_generation()


def reconstruct(backend, log):
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    matrix.verify_training_config(config)
    events = read_jsonl(SOURCE / "events.jsonl")
    require([x["event_index"] for x in events] == list(range(1, len(events) + 1)), "原运行事件不连续。")
    preflight = [x for x in events if x["event"] == "preflight"]
    started = [x for x in events if x["event"] == "formal_run_started"]
    require(len(preflight) == len(started) == 1 and started[0]["method"] == "proposed_solution_quality_and_speed_agents"
            and started[0]["formal_replicate"] == 2, "本方法第 2 次研发身份不符。")
    for name, expected in preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
    failures = [x for x in events if x["event"] == "model_process" and x["stage"] == "eoh_M1_01"]
    require(len(failures) == 1 and classify_failure(failures[0]["result"])["category"] == "network"
            and events[-1]["event"] == "preflight_or_run_stopped", "断点不是预期的连接失败。")
    registrations = [x for x in events if x["event"] == "candidate_registered"]
    responses = {x["candidate_id"]: x for x in events if x["event"] == "proposal_response"}
    evaluations = {x["candidate_id"]: x["result"] for x in events if x["event"] == "evaluation"}
    invalid = {x["candidate_id"]: x["error"] for x in events if x["event"] == "candidate_invalid"}
    skipped = [x["candidate_id"] for x in events if x["event"] == "candidate_skipped"]
    require(len(registrations) == 70 and len(responses) == 50 and len(evaluations) == 48
            and skipped == [f"c{x:04d}" for x in range(52, 71)], "历史候选或未启动槽位不符。")
    task = config["task"]
    controller = RecoveryController(backend, log, population_size=task["population_size"],
                                    initial_count=task["initial_proposals"], rounds=task["collaboration_rounds"],
                                    seed=task["selection_seed"], generations=task["generations"],
                                    task_contract=load_task_contract("tsp_aco"))
    candidates = {}
    for row in registrations:
        ident, seq = row["candidate_id"], row["sequence"]
        require(ident == f"c{seq:04d}" and seq == len(controller.candidates) + 1, "候选编号不连续。")
        candidate = Candidate(ident, seq, row["stage"], tuple(row["parents"]))
        controller.candidates.append(candidate)
        candidates[ident] = candidate
        if ident not in responses:
            if ident == "c0051":
                candidate.error = "模型连接失败，未产生提案；不自动重试。"
            else:
                require(ident in skipped, f"候选缺少模型响应：{ident}")
            continue
        response = responses[ident]
        candidate.generation_order = response["generation_order"]
        try:
            parsed = controller.parse_proposal(response["response"])
            candidate.thought, candidate.code = parsed["thought"], parsed["code"]
            check_modification_scope(candidate.code)
        except (probe.ProbeError, ValueError, SyntaxError):
            require(ident in invalid, f"提案解析与失败记录不符：{ident}")
            candidate.error = invalid[ident]
            continue
        if ident not in evaluations:
            require(ident in ("c0049", "c0050"), f"未评价提案身份不符：{ident}")
            continue
        result = evaluations[ident]
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
        candidate.objective = candidate.raw_objective = float(result["average"])
        candidate.fitness = float(result["fitness"])
        candidate.timing_seconds, candidate.timing_scope = timing_from_result(result)
        candidate.completed_order = result["completed_monotonic_ns"]
        candidate.evaluation_origin = ident
    require(sorted(x["generation_order"] for x in responses.values()) == list(range(1, 51)), "生成顺序不连续。")
    controller._generated = 50
    require(max(c.completed_order for c in controller.candidates if c.valid) < time.monotonic_ns(),
            "当前单调时钟早于历史评价。")
    initial = controller.select(controller.candidates[:30], controller.n)
    initial_events = [x for x in events if x["event"] == "initial_population"]
    require(len(initial_events) == 1 and [x.candidate_id for x in initial] == initial_events[0]["candidate_ids"],
            "初始种群筛选不一致。")
    attempts = {x["stage"]: x for x in events if x["event"] == "model_attempt"}
    directions = {"E1": "Create a heuristic as different as possible from these parents.",
                  "E2": "Identify their shared idea and create a new structure based on it.",
                  "M1": "Modify this heuristic's structure to improve performance.",
                  "M2": "Adjust parameters while retaining this heuristic's overall logic."}
    saved_random = controller.random.getstate()
    specs = []
    for row in registrations[30:]:
        operator = row["stage"].split("_")[1]
        parents = controller.eoh_parents(initial, 5 if operator in ("E1", "E2") else 1)
        require([x.candidate_id for x in parents] == row["parents"], f"随机父代不符：{row['candidate_id']}")
        context = "\n".join(f"Algorithm description: {x.thought}\nCode:\n{x.code}\n" for x in parents)
        prompt = f"{controller.task}\n{context}{directions[operator]}\n{controller.output}"
        if row["stage"] in attempts:
            require(attempts[row["stage"]]["prompt"] == prompt, f"原 EoH 提示不符：{row['stage']}")
        specs.append((row["stage"], row["parents"], prompt))
    controller.random.setstate(saved_random)
    controller._replay_eoh = controller.candidates[30:70]
    controller._replay_specs = specs
    cpu = [x["accounting"]["cpu_seconds"] for x in events if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == 48 and len([x for x in events if x["event"] == "model_attempt"]) == 51,
            "历史模型调用或 CPU 记录不符。")
    require(not any(x.get("candidate_id") in ("c0049", "c0050") for x in events if x["event"] == "evaluation_budget_reserved"),
            "保存的响应已有评价启动记录，不能再次评价。")
    return controller, {"config": config, "pending": [candidates[x] for x in skipped],
                        "pending_specs": specs[21:], "unevaluated": [candidates["c0049"], candidates["c0050"]],
                        "cpu_samples": cpu, "prior_hashes": preflight[0]["code_sha256"]}


def evaluate_saved_response(controller, candidate):
    """复用已返回但从未启动评价的响应；不再次调用模型。"""
    backend = controller.backend
    backend.prepare_proposal(candidate)
    try:
        controller.event("reused_model_proposal", generation=1, candidate_id=candidate.candidate_id,
                         source_events=str(SOURCE / "events.jsonl"))
        result = backend.evaluate(candidate)
        controller.event("evaluation", candidate_id=candidate.candidate_id, result=result)
        if result.get("status") == "infrastructure_error":
            raise StopRun(f"评价基础设施不可信：{result.get('error')}")
        if result.get("status") != "passed":
            candidate.error = str(result.get("error") or result.get("status"))
            return
        objective, fitness = result.get("average"), result.get("fitness")
        require(isinstance(objective, (int, float)) and not isinstance(objective, bool)
                and isinstance(fitness, (int, float)) and not isinstance(fitness, bool)
                and math.isfinite(objective) and math.isfinite(fitness) and fitness == -objective,
                "复用候选的目标或 Fitness 不可信。")
        candidate.objective = candidate.raw_objective = float(objective)
        candidate.fitness = float(fitness)
        candidate.timing_seconds, candidate.timing_scope = timing_from_result(result)
        candidate.completed_order = result.get("completed_monotonic_ns", time.monotonic_ns())
        candidate.evaluation_origin = candidate.candidate_id
    finally:
        backend.release_proposal(candidate)


def run_pending(controller, state):
    require(len(state["pending"]) == len(state["pending_specs"]) == 19, "待执行槽位数量不符。")
    controller.generation = 1
    for candidate in state["unevaluated"]:
        evaluate_saved_response(controller, candidate)
    entries = []
    for candidate, (stage, parents, prompt) in zip(state["pending"], state["pending_specs"]):
        require(candidate.stage == stage and list(candidate.parents) == parents, "待执行候选或父代不符。")
        controller.event("reused_candidate_registration", generation=1, candidate_id=candidate.candidate_id,
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
            prior = json.loads(path.read_text(encoding="utf-8"))
            if (prior.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                    or matrix.run_identity(prior) != matrix.run_identity(config)):
                continue
            event_path = path.parent / "events.jsonl"
            if event_path.is_file() and any(x["event"] == "formal_run_started" for x in read_jsonl(event_path)):
                started.add(path.parent.resolve())
        require(started == {SOURCE.resolve()}, "本方法第 2 次研发存在未知已启动运行。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=config["formal_replicate"], selection_seed=task["selection_seed"],
                  resume_of=str(SOURCE), interrupted_events_sha256=digest(SOURCE / "events.jsonl"),
                  total_model_call_cap=420,
                  note="保留连接失败与一次中断评价；只评价未评价响应并执行尚未启动的原槽位。")
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
    run_pending(controller, state)
    require(len(backend.calls) == 19 and len(backend.evaluations) == 21
            and all(stage != "eoh_M1_01" for stage, _ in backend.calls), "未启动槽位或原响应复用不符。")
    selected = controller.run()
    require(len(selected) == 10 and controller.generation == 6 and len(backend.calls) == 369
            and len(backend.evaluations) == 329
            and [x["generation"] for x in log.items if x["event"] == "selection"] == list(range(1, 7)),
            "本方法后续代数、预算槽位或筛选不符。")
    print(json.dumps({"status": "passed", "prior_model_calls": 51, "prior_evaluations": 48,
                      "unstarted_proposals": 19, "saved_response_first_evaluation": 2,
                      "later_model_calls": 369, "later_evaluation_slots": 329,
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
    run_dir = ROOT / "outputs/p1_control" / (datetime.now().strftime("%Y%m%d_%H%M%S")
                                            + "_" + uuid.uuid4().hex[:8] + "_resume")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("xb") as out:
        out.write((SOURCE / "config.json").read_bytes())
    log = JsonLog(run_dir / "events.jsonl")
    runtime = RecoveryRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    controller, state = reconstruct(runtime, log)
    for path in [*(Path(x) for x in state["prior_hashes"]), SOURCE / "events.jsonl",
                 SOURCE / "config.json", Path(__file__).resolve()]:
        if str(path) not in runtime.protected_hashes:
            runtime.protected.append(path)
            runtime.protected_hashes[str(path)] = digest(path)
    budget = runtime.budget
    budget.model_normal = 51
    budget.eval_starts = budget.eval_normal = 48
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
              code_sha256=runtime.protected_hashes, selection_seed=config["task"]["selection_seed"],
              generations=6, continuation=True, normal_model_slots_used=51,
              physical_model_cli_invocations_used=51, evaluation_starts_used=48,
              evaluation_cpu_seconds_used=budget.cpu_finished)
    print(f"本方法第 2 次研发恢复启动前检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(config, run_dir, log):
            log.event("resume_checkpoint", interrupted_run=str(SOURCE), generation=1,
                      failed_candidate_id="c0051", interrupted_evaluations=["c0048"],
                      saved_response_first_evaluation=["c0049", "c0050"], unstarted_candidates=[x.candidate_id for x in state["pending"]],
                      model_calls_already_used=51, evaluation_starts_already_used=48, total_model_call_cap=420)
            run_pending(controller, state)
            selected = controller.run()
            require(len(selected) == 10 and controller.generation == 6, "本方法第 2 次研发未完成六代。")
        print(f"本方法第 2 次研发六代完成；记录：{run_dir}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"本方法第 2 次研发续跑停止：{exc}；记录：{run_dir}", file=sys.stderr, flush=True)
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
    print(json.dumps({"status": "passed", "candidates": len(controller.candidates),
                      "model_calls_used": 51, "evaluation_starts": len(state["cpu_samples"]),
                      "unstarted_candidates": len(state["pending"]), "reused_response": ["c0049", "c0050"],
                      "total_model_call_cap": 420}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"本方法第 2 次研发恢复检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
