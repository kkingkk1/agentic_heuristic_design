"""首次评价 RoCo 中断时已返回的两个提案，不启动模型调用。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
PRIOR_RECOVERY = ROOT / "outputs/p1_manual_retry/roco_generation1_20261112_recovery"
SOURCE = ROOT / "outputs/p1_roco/20261112_174300_b791da58_resume"
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p1_formal_matrix as matrix
from scripts.p0_controller import StopRun, check_modification_scope
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from scripts.p1_classified_runtime import ClassifiedRoCoRuntime

spec = importlib.util.spec_from_file_location("roco_generation2_recovery", PRIOR_RECOVERY / "resume_roco_generation2.py")
prior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prior
spec.loader.exec_module(prior)
require = prior.require
read_jsonl = prior.read_jsonl


def reconstruct(backend, log):
    controller, state = prior.reconstruct(backend, log)
    config = state["config"]
    require((SOURCE / "config.json").read_bytes() == (prior.SOURCE / "config.json").read_bytes(),
            "RoCo 第 2 代续跑配置不一致。")
    events = read_jsonl(SOURCE / "events.jsonl")
    preflight = [x for x in events if x["event"] == "preflight"]
    started = [x for x in events if x["event"] == "formal_run_started"]
    require(len(preflight) == len(started) == 1 and started[0]["resume_of"] == str(prior.SOURCE)
            and events[-1]["event"] == "preflight_or_run_stopped", "RoCo 断点身份不符。")
    for name, expected in preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
    failures = [x for x in events if x["event"] == "model_failure_classified"]
    require([(x["stage"], x["category"]) for x in failures] == [("eoh_E1_05", "unknown"), ("eoh_E1_07", "unknown")],
            "服务错误记录不符。")
    responses = {x["candidate_id"]: x for x in events if x["event"] == "proposal_response"}
    require(set(responses) == {"c0094", "c0096"}, "已返回提案身份不符。")
    require(not any(x.get("candidate_id") in responses for x in events if x["event"] == "evaluation_budget_reserved"),
            "提案已有评价启动记录，不能重复评价。")
    pending = {c.candidate_id: (c, spec) for c, spec in zip(state["pending"], state["pending_specs"])}
    candidates = []
    for ident in ("c0094", "c0096"):
        candidate, (stage, parents, prompt) = pending[ident]
        row = responses[ident]
        require(candidate.stage == stage == row["stage"] and list(candidate.parents) == parents
                and row["prompt"] == prompt and row["generation"] == 2, f"提案来源或提示不符：{ident}")
        parsed = controller.parse_proposal(row["response"])
        check_modification_scope(parsed["code"])
        candidate.thought, candidate.code = parsed["thought"], parsed["code"]
        candidate.generation_order = row["generation_order"]
        candidates.append(candidate)
    require(sorted(x.generation_order for x in candidates) == [91, 92], "生成顺序不符。")
    cpu = [x["accounting"]["cpu_seconds"] for x in events if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == 3 and len(state["cpu_samples"]) == 86
            and len([x for x in events if x["event"] == "model_attempt"]) == 4, "累计开销不符。")
    return controller, {"config": config, "candidates": candidates,
                        "cpu_samples": state["cpu_samples"] + cpu,
                        "prior_hashes": preflight[0]["code_sha256"]}


@contextmanager
def evaluation_guard(config, log):
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
            raise ValueError("已有 P1 正式研发运行，不能并发评价。")
        if result not in (0, 0x80):
            raise OSError(ctypes.get_last_error(), "无法取得 P1 研发互斥锁。")
        acquired = True
        matrix.verify_training_config(config)
        started = set()
        for path in (ROOT / "outputs/p1_roco").glob("*/config.json"):
            prior_config = json.loads(path.read_text(encoding="utf-8"))
            if (prior_config.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                    or matrix.run_identity(prior_config) != matrix.run_identity(config)):
                continue
            event_path = path.parent / "events.jsonl"
            if event_path.is_file() and any(x["event"] == "formal_run_started" for x in read_jsonl(event_path)):
                started.add(path.parent.resolve())
        require(started == {prior.previous.SOURCE.resolve(), prior.SOURCE.resolve(), SOURCE.resolve()},
                "RoCo 首组存在未知已启动运行。")
        for path in RECOVERY.glob("*/evaluation_events.jsonl"):
            require(not any(x["event"] == "saved_evaluation_started" for x in read_jsonl(path)),
                    "保存响应的首次评价已有启动记录，不能重复执行。")
        log.event("saved_evaluation_started", source_events=str(SOURCE / "events.jsonl"),
                  source_events_sha256=digest(SOURCE / "events.jsonl"), candidates=["c0094", "c0096"],
                  prior_model_calls=103, prior_evaluations=89, model_call_cap=420)
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def evaluate_one(controller, candidate):
    backend = controller.backend
    backend.prepare_proposal(candidate)
    try:
        controller.event("reused_model_proposal", generation=2, candidate_id=candidate.candidate_id,
                         source_events=str(SOURCE / "events.jsonl"))
        result = backend.evaluate(candidate)
        controller.event("evaluation", candidate_id=candidate.candidate_id, result=result)
        if result.get("status") == "infrastructure_error":
            raise StopRun(f"评价基础设施不可信：{result.get('error')}")
        return result
    finally:
        backend.release_proposal(candidate)


def simulate():
    from tests.test_p0_controller import FixtureBackend, MemoryLog
    backend, log = FixtureBackend(), MemoryLog()
    controller, state = reconstruct(backend, log)
    with evaluation_guard(state["config"], MemoryLog()):
        pass
    for candidate in state["candidates"]:
        evaluate_one(controller, candidate)
    require(not backend.calls and len(backend.evaluations) == 2
            and [x["candidate_id"] for x in log.items if x["event"] == "evaluation"] == ["c0094", "c0096"],
            "首次评价流程或模型调用计数不符。")
    print(json.dumps({"status": "passed", "candidates": ["c0094", "c0096"],
                      "new_model_calls": 0, "new_evaluations": 2}, ensure_ascii=False))


def run():
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    matrix.verify_training_software(matrix.load_matrix()[0])
    class CheckBackend:
        cfg = config
    reconstruct(CheckBackend(), None)
    resources = memory_and_priority()
    require(resources["available_memory_bytes"] >= config["resources"]["min_available_system_memory_bytes"],
            "可用内存不足。")
    run_dir = RECOVERY / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8] + "_evaluation")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("xb") as out:
        out.write((SOURCE / "config.json").read_bytes())
    log = JsonLog(run_dir / "evaluation_events.jsonl")
    runtime = ClassifiedRoCoRuntime(run_dir, run_dir / "config.json", log, cli=["no-model-call"])
    controller, state = reconstruct(runtime, log)
    for path in [*(Path(x) for x in state["prior_hashes"]), SOURCE / "events.jsonl",
                 SOURCE / "config.json", Path(__file__).resolve()]:
        if str(path) not in runtime.protected_hashes:
            runtime.protected.append(path)
            runtime.protected_hashes[str(path)] = digest(path)
    budget = runtime.budget
    budget.model_normal = 103
    budget.eval_starts = budget.eval_normal = 89
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, code_sha256=runtime.protected_hashes,
              prior_model_calls=103, prior_evaluations=89, evaluation_cpu_seconds_used=budget.cpu_finished)
    print(f"保存响应的首次评价启动；记录：{run_dir}", flush=True)
    with evaluation_guard(config, log):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(evaluate_one, controller, candidate) for candidate in state["candidates"]]
            results = [future.result() for future in futures]
    summary = {"status": "completed", "source_events": str(SOURCE / "events.jsonl"),
               "source_events_sha256": digest(SOURCE / "events.jsonl"), "new_model_calls": 0,
               "prior_model_calls": 103, "prior_evaluations": 89, "new_evaluations": 2,
               "results": [{"candidate_id": c.candidate_id, "status": r["status"],
                            "average": r.get("average"), "error": r.get("error")}
                           for c, r in zip(state["candidates"], results)]}
    with (run_dir / "summary.json").open("x", encoding="utf-8", newline="\n") as out:
        json.dump(summary, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    log.event("saved_evaluation_finished", status="completed", summary=str(run_dir / "summary.json"))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


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
        return run()
    class CheckBackend:
        cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    _, state = reconstruct(CheckBackend(), None)
    print(json.dumps({"status": "passed", "candidates": [x.candidate_id for x in state["candidates"]],
                      "prior_model_calls": 103, "prior_evaluations": len(state["cpu_samples"]),
                      "new_model_calls": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"保存响应评价停止：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
