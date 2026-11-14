"""只准备首组第 6 代恢复；执行须由调用者另行确认一次调用上限。"""

import argparse
import copy
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
INTERRUPTED = ROOT / "outputs/p1_control/20261112_161733_a2221c51_resume"
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate, p1_formal_matrix as matrix
from scripts.p0_controller import Candidate, Controller, StopRun, check_modification_scope, timing_from_result
from scripts.p0_runtime import Budget, JsonLog, digest, memory_and_priority
from scripts.p1_classified_runtime import ClassifiedTaskRuntime

spec = importlib.util.spec_from_file_location("p1_generation3_recovery", RECOVERY / "resume_generation3.py")
generation3 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = generation3
spec.loader.exec_module(generation3)
base = generation3.base
require = base.require
read_jsonl = base.read_jsonl


class FinalController(generation3.ContinueController):
    def run_final(self, population):
        require(self.generation == 5 and len(population) == 10, "末代起点不符。")
        self._generation_offset = 5
        self._resume_population = population
        self.generations = 1
        return Controller.run(self)


def reconstruct(backend, log):
    with patch.object(generation3, "ContinueController", FinalController):
        controller, earlier = generation3.reconstruct(backend, log)
    config = earlier["config"]
    require((INTERRUPTED / "config.json").read_bytes() == (generation3.INTERRUPTED / "config.json").read_bytes(),
            "第三次续跑配置不一致。")
    events = read_jsonl(INTERRUPTED / "events.jsonl")
    require([x["event_index"] for x in events] == list(range(1, len(events) + 1)), "事件序号不连续。")
    preflight = [x for x in events if x["event"] == "preflight"]
    start = [x for x in events if x["event"] == "formal_run_started"]
    require(len(preflight) == len(start) == 1 and start[0]["resume_of"] == str(generation3.INTERRUPTED)
            and start[0]["total_model_call_cap"] == 421, "第三次续跑身份不符。")
    for name, expected in preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"历史受保护文件变化：{name}")
    require(events[-1]["event"] == "preflight_or_run_stopped"
            and "剩余正常模型调用不足以覆盖一代" in events[-1]["error"], "当前不是预期的预算停止点。")
    require(len([x for x in events if x["event"] == "model_attempt"]) == 145
            and [x for x in events if x["event"] == "model_budget_reserved"][-1]["normal"] == 356,
            "模型调用累计不符。")
    registrations = [x for x in events if x["event"] == "candidate_registered"]
    responses = {x["candidate_id"]: x for x in events if x["event"] == "proposal_response"}
    evaluations = {x["candidate_id"]: x["result"] for x in events if x["event"] == "evaluation"}
    require(len(registrations) == len(responses) == len(evaluations) == 127,
            "续跑候选或评价数量不符。")
    candidates = {x.candidate_id: x for x in controller.candidates}
    for row in registrations:
        ident, seq = row["candidate_id"], row["sequence"]
        require(ident == f"c{seq:04d}" and seq == len(controller.candidates) + 1, f"候选编号不连续：{ident}")
        response, result = responses[ident], evaluations[ident]
        parsed = controller.parse_proposal(response["response"])
        check_modification_scope(parsed["code"])
        candidate = Candidate(ident, seq, row["stage"], tuple(row["parents"]),
                              thought=parsed["thought"], code=parsed["code"],
                              generation_order=response["generation_order"])
        controller.candidates.append(candidate)
        candidates[ident] = candidate
        directory = INTERRUPTED / "candidates" / ident
        require(result["status"] == "passed"
                and json.loads((directory / "result.json").read_text(encoding="utf-8")) == result
                and (directory / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8") == candidate.code + "\n",
                f"候选或评价记录不一致：{ident}")
        checked = p0_tsp_aco_evaluate.read_complete(directory / "stdout.txt")
        require(checked["average"] == result["average"] and checked["fitness"] == result["fitness"]
                and checked["objectives"] == result["objectives"], f"评价输出不一致：{ident}")
        candidate.objective = candidate.raw_objective = float(result["average"])
        candidate.fitness = float(result["fitness"])
        candidate.timing_seconds, candidate.timing_scope = timing_from_result(result)
        candidate.completed_order = result["completed_monotonic_ns"]
        candidate.evaluation_origin = ident
    require(sorted(x["generation_order"] for x in responses.values()) == list(range(193, 320)),
            "生成顺序不连续。")
    controller._generated = 319
    require(max(c.completed_order for c in controller.candidates if c.valid) < time.monotonic_ns(),
            "当前单调时钟早于历史评价，不能重建去重顺序。")
    selections = [x for x in events if x["event"] == "selection"]
    require([x["generation"] for x in selections] == [3, 4, 5], "第 3 至 5 代筛选不完整。")
    populations = {}
    for row in selections:
        selected = controller.select([candidates[x] for x in row["pool"]], controller.n)
        require([{"candidate_id": x.candidate_id, "evaluation_origin": x.evaluation_origin} for x in selected]
                == row["selected"], f"第 {row['generation']} 代筛选重建不一致。")
        populations[row["generation"]] = selected
    for generation in (4, 5):
        population = populations[generation - 1]
        rows = [x for x in registrations if x.get("generation") == generation and x["stage"].startswith("eoh_")]
        require(len(rows) == 40, f"第 {generation} 代 EoH 候选不完整。")
        for row in rows:
            op = row["stage"].split("_")[1]
            parents = controller.eoh_parents(population, 5 if op in ("E1", "E2") else 1)
            require([x.candidate_id for x in parents] == row["parents"], f"随机父代不一致：{row['candidate_id']}")
        a, b = controller.elite_pair(population)
        pair = [x for x in events if x["event"] == "elite_pair" and x["generation"] == generation]
        require(len(pair) == 1 and (a.candidate_id, b.candidate_id) == (pair[0]["a"], pair[0]["b"]),
                f"第 {generation} 代双精英不一致。")
    history = [x for x in events if x["event"] == "history_appended"]
    require(len(history) == 9, "第 3 至 5 代角色历史不完整。")
    for row in history:
        controller.history[row["role"]].append(row["reflection"])
    require(all(len(values) == 5 for values in controller.history.values()), "五代角色历史不符。")
    controller.generation = 5
    cpu = [x["accounting"]["cpu_seconds"] for x in events if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == 127 and len(earlier["cpu_samples"]) == 191, "评价 CPU 记录不完整。")
    return controller, {"config": config, "population": populations[5],
                        "cpu_samples": earlier["cpu_samples"] + cpu,
                        "prior_hashes": preflight[0]["code_sha256"]}


def extended_budget_config(config):
    adjusted = copy.deepcopy(config)
    adjusted["budget"]["normal_model_calls"] += 1
    adjusted["budget"]["max_initialization_and_generation_model_cli_invocations"] += 1
    return adjusted


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
        require(started == {base.SOURCE.resolve(), generation3.previous.INTERRUPTED.resolve(),
                            generation3.INTERRUPTED.resolve(), INTERRUPTED.resolve()},
                "该重复索引存在未知已启动运行。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=config["formal_replicate"], selection_seed=task["selection_seed"],
                  resume_of=str(INTERRUPTED), interrupted_events_sha256=digest(INTERRUPTED / "events.jsonl"),
                  total_model_call_cap=422, extra_model_cli_invocations=2,
                  note="首组总上限增加 1 次；配置文件不变，额外调用计入本运行预算。")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def simulate():
    from tests.test_p0_controller import FixtureBackend, MemoryLog
    backend, log = FixtureBackend(), MemoryLog()
    controller, state = reconstruct(backend, log)
    budget = Budget(state["config"], MemoryLog())
    budget.model_normal = 356
    budget.eval_starts = budget.eval_normal = 318
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    try:
        budget.before_generation()
    except StopRun as exc:
        require("剩余正常模型调用不足" in str(exc), "原预算停止原因不符。")
    else:
        raise ValueError("原预算不应允许启动完整末代。")
    budget.cfg = extended_budget_config(state["config"])
    budget.before_generation()
    require(budget.cfg["budget"]["normal_model_calls"] - budget.model_normal == 65,
            "增加一次调用后未恰好覆盖末代预算。")
    boundary = Budget(extended_budget_config(state["config"]), MemoryLog())
    boundary.model_normal = 420
    boundary.model_start()
    require(boundary.model_normal == 421, "补充调用未计入正常预算。")
    try:
        boundary.model_start()
    except StopRun:
        pass
    else:
        raise ValueError("补充预算允许超过 422 次物理调用。")
    selected = controller.run_final(state["population"])
    require(len(selected) == 10 and controller.generation == 6
            and len(backend.calls) == 65 and len(backend.evaluations) == 58
            and all(len(x) == 6 for x in controller.history.values()), "末代离线流程或槽位不符。")
    print(json.dumps({"status": "passed", "prior_physical_calls": 357, "prior_evaluations": 318,
                      "old_cap": 421, "proposed_cap": 422, "final_generation_calls": 65,
                      "final_generation_evaluation_slots": 58}, ensure_ascii=False))


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
    runtime = ClassifiedTaskRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    controller, state = reconstruct(runtime, log)
    for path in [*(Path(x) for x in state["prior_hashes"]), INTERRUPTED / "events.jsonl",
                 INTERRUPTED / "config.json", Path(__file__).resolve()]:
        if str(path) not in runtime.protected_hashes:
            runtime.protected.append(path)
            runtime.protected_hashes[str(path)] = digest(path)
    budget = runtime.budget
    budget.cfg = extended_budget_config(config)
    budget.model_normal = 356
    budget.eval_starts = budget.eval_normal = 318
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
              code_sha256=runtime.protected_hashes, selection_seed=config["task"]["selection_seed"],
              generations=6, continuation=True, normal_model_slots_used=356,
              physical_model_cli_invocations_used=357, extra_manual_invocations=2,
              evaluation_starts_used=318, evaluation_cpu_seconds_used=budget.cpu_finished,
              original_physical_call_cap=421, authorized_physical_call_cap=422)
    print(f"第 6 代恢复启动前检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(config, run_dir, log):
            log.event("resume_checkpoint", interrupted_run=str(INTERRUPTED), generation=5, next_generation=6,
                      physical_model_calls_already_used=357, normal_model_slots_already_used=356,
                      evaluation_starts_already_used=318, total_model_call_cap=422,
                      population=[x.candidate_id for x in state["population"]])
            selected = controller.run_final(state["population"])
            require(len(selected) == 10 and controller.generation == 6, "末代未完成。")
        print(f"首组六代完成；记录：{run_dir}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"末代续跑停止：{exc}；记录：{run_dir}", file=sys.stderr, flush=True)
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
        cfg = json.loads((INTERRUPTED / "config.json").read_text(encoding="utf-8"))
    controller, state = reconstruct(CheckBackend(), None)
    print(json.dumps({"status": "passed", "generation": controller.generation,
                      "population": [x.candidate_id for x in state["population"]],
                      "model_calls_used": 357, "evaluation_starts": len(state["cpu_samples"]),
                      "old_total_cap": 421, "proposed_total_cap": 422}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"恢复检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
