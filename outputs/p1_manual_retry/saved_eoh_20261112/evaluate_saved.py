"""只评价两次中断后已返回、尚未评价的 EoH 提案，不调用模型。"""

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
sys.path.insert(0, str(ROOT))

from scripts import p1_formal_matrix as matrix
from scripts.p0_controller import Candidate, StopRun, check_modification_scope
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from scripts.p1_classified_runtime import ClassifiedRoCoRuntime, ClassifiedTaskRuntime


CASES = {
    "proposed": {
        "source": ROOT / "outputs/p1_control/20261112_230214_43fec153_resume",
        "base": ROOT / "outputs/p1_manual_retry/fusion_2_generation3_20261112_recovery/prepare_generation6.py",
        "candidates": ("c0323", "c0325", "c0326"),
        "failed": ("eoh_E1_02", "unknown"),
        "prior_calls": 357,
        "prior_evaluations": 318,
        "new_calls": 4,
        "new_evaluations": 0,
        "generation": 6,
        "cap": 422,
    },
    "roco": {
        "source": ROOT / "outputs/p1_roco/20261112_230339_86dc16b5_resume",
        "base": ROOT / "outputs/p1_manual_retry/roco_2_quota_recovery_20261112/prepare_resume.py",
        "candidates": ("c0089", "c0090", "c0092"),
        "failed": ("eoh_E1_03", "quota"),
        "prior_calls": 93,
        "prior_evaluations": 83,
        "new_calls": 6,
        "new_evaluations": 2,
        "generation": 2,
        "cap": 420,
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_events(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_base(case_name):
    spec = importlib.util.spec_from_file_location("saved_eoh_base_" + case_name, CASES[case_name]["base"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reconstruct(case_name, backend, log):
    case = CASES[case_name]
    base = load_base(case_name)
    controller, earlier = base.reconstruct(backend, log)
    source = case["source"]
    config = earlier["config"]
    require((source / "config.json").read_bytes() == (base.INTERRUPTED / "config.json").read_bytes()
            if case_name == "proposed" else
            (source / "config.json").read_bytes() == (base.SOURCE / "config.json").read_bytes(),
            "本次断点配置不一致。")
    events = read_events(source / "events.jsonl")
    require([x["event_index"] for x in events] == list(range(1, len(events) + 1)), "事件序号不连续。")
    starts = [x for x in events if x["event"] == "formal_run_started"]
    preflights = [x for x in events if x["event"] == "preflight"]
    parent = base.INTERRUPTED if case_name == "proposed" else base.SOURCE
    require(len(starts) == len(preflights) == 1 and starts[0]["resume_of"] == str(parent)
            and starts[0]["interrupted_events_sha256"] == digest(parent / "events.jsonl")
            and starts[0]["total_model_call_cap"] == case["cap"]
            and events[-1]["event"] == "preflight_or_run_stopped", "断点身份或上限不符。")
    for name, expected in preflights[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
    failures = [(x["stage"], x["category"]) for x in events if x["event"] == "model_failure_classified"]
    attempts = {x["stage"]: x for x in events if x["event"] == "model_attempt"}
    responses = {x["candidate_id"]: x for x in events if x["event"] == "proposal_response"}
    registrations = {x["candidate_id"]: x for x in events if x["event"] == "candidate_registered"}
    require(failures == [case["failed"]] and len(attempts) == case["new_calls"]
            and len([x for x in events if x["event"] == "evaluation_budget_reserved"]) == case["new_evaluations"],
            "失败类型、调用数或评价数不符。")
    candidates = []
    for ident in case["candidates"]:
        reg, row = registrations[ident], responses[ident]
        require(reg["candidate_id"] == f"c{reg['sequence']:04d}" and reg["stage"] == row["stage"]
                and row["generation"] == case["generation"]
                and attempts[row["stage"]]["prompt"] == row["prompt"]
                and not any(x.get("candidate_id") == ident for x in events
                            if x["event"] == "evaluation_budget_reserved"), f"候选来源或评价状态不符：{ident}")
        parsed = controller.parse_proposal(row["response"])
        check_modification_scope(parsed["code"])
        candidates.append(Candidate(ident, reg["sequence"], reg["stage"], tuple(reg["parents"]),
                                    thought=parsed["thought"], code=parsed["code"],
                                    generation_order=row["generation_order"]))
    cpu = [x["accounting"]["cpu_seconds"] for x in events if x["event"] == "evaluation_cpu_final"]
    require(len(earlier["cpu_samples"]) == case["prior_evaluations"]
            and len(cpu) == case["new_evaluations"], "累计评价 CPU 记录不符。")
    return controller, {"base": base, "config": config, "candidates": candidates,
                        "cpu_samples": earlier["cpu_samples"] + cpu,
                        "prior_hashes": preflights[0]["code_sha256"]}


def assert_not_evaluated(config, candidate_ids):
    identity = matrix.run_identity(config)
    candidates = set(candidate_ids)
    roots = [ROOT / "outputs/p1_control", ROOT / "outputs/p1_roco", ROOT / "outputs/p1_manual_retry"]
    for root in roots:
        for pattern in ("*/config.json", "*/*/config.json"):
            for path in root.glob(pattern):
                other = json.loads(path.read_text(encoding="utf-8"))
                if (other.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                        or matrix.run_identity(other) != identity):
                    continue
                for event_file in (path.parent / "events.jsonl", path.parent / "evaluation_events.jsonl"):
                    if event_file.is_file():
                        used = {x.get("candidate_id") for x in read_events(event_file)
                                if x["event"] in ("evaluation_budget_reserved", "evaluation")}
                        require(not candidates & used, f"候选已有评价记录：{event_file}")


@contextmanager
def evaluation_guard(case_name, state, log):
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
        config, base, source = state["config"], state["base"], CASES[case_name]["source"]
        matrix.verify_training_config(config)
        expected = ({base.base.SOURCE, base.generation3.previous.INTERRUPTED,
                     base.generation3.INTERRUPTED, base.INTERRUPTED, source}
                    if case_name == "proposed" else {base.previous.SOURCE, base.SOURCE, source})
        started = set()
        run_root = ROOT / ("outputs/p1_control" if case_name == "proposed" else "outputs/p1_roco")
        for path in run_root.glob("*/config.json"):
            other = json.loads(path.read_text(encoding="utf-8"))
            if (other.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                    or matrix.run_identity(other) != matrix.run_identity(config)):
                continue
            event_file = path.parent / "events.jsonl"
            if event_file.is_file() and any(x["event"] == "formal_run_started" for x in read_events(event_file)):
                started.add(path.parent.resolve())
        require(started == {p.resolve() for p in expected}, "该研发存在未知的已启动运行。")
        ids = [x.candidate_id for x in state["candidates"]]
        assert_not_evaluated(config, ids)
        log.event("saved_evaluation_started", source_events=str(source / "events.jsonl"),
                  source_events_sha256=digest(source / "events.jsonl"), candidates=ids,
                  prior_model_calls=CASES[case_name]["prior_calls"] + CASES[case_name]["new_calls"],
                  prior_evaluations=len(state["cpu_samples"]), model_call_cap=CASES[case_name]["cap"])
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def evaluate_one(controller, candidate, case_name):
    backend = controller.backend
    backend.prepare_proposal(candidate)
    try:
        controller.event("reused_model_proposal", generation=CASES[case_name]["generation"],
                         candidate_id=candidate.candidate_id,
                         source_events=str(CASES[case_name]["source"] / "events.jsonl"))
        result = backend.evaluate(candidate)
        controller.event("evaluation", candidate_id=candidate.candidate_id, result=result)
        if result.get("status") == "infrastructure_error":
            raise StopRun(f"评价基础设施不可信：{result.get('error')}")
        return result
    finally:
        backend.release_proposal(candidate)


class NoModelCalls:
    def model(self, stage, prompt):
        raise AssertionError("首次评价不得调用模型。")


class EvaluationOnlyTaskRuntime(NoModelCalls, ClassifiedTaskRuntime):
    pass


class EvaluationOnlyRoCoRuntime(NoModelCalls, ClassifiedRoCoRuntime):
    pass


def simulate(case_name):
    from tests.test_p0_controller import FixtureBackend, MemoryLog
    backend, log = FixtureBackend(), MemoryLog()
    controller, state = reconstruct(case_name, backend, log)
    with evaluation_guard(case_name, state, MemoryLog()):
        pass
    for candidate in state["candidates"]:
        evaluate_one(controller, candidate, case_name)
    require(not backend.calls and len(backend.evaluations) == 3
            and [x["candidate_id"] for x in log.items if x["event"] == "evaluation"]
            == list(CASES[case_name]["candidates"]), "首次评价或模型调用计数不符。")
    print(json.dumps({"status": "passed", "method": case_name, "new_model_calls": 0,
                      "new_evaluations": 3}, ensure_ascii=False))


def run(case_name):
    case = CASES[case_name]
    config = json.loads((case["source"] / "config.json").read_text(encoding="utf-8"))
    matrix.verify_training_software(matrix.load_matrix()[0])
    class CheckBackend:
        cfg = config
    reconstruct(case_name, CheckBackend(), None)
    resources = memory_and_priority()
    require(resources["available_memory_bytes"] >= config["resources"]["min_available_system_memory_bytes"],
            "可用内存不足。")
    run_dir = RECOVERY / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + case_name + "_"
                          + uuid.uuid4().hex[:8] + "_evaluation")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("xb") as out:
        out.write((case["source"] / "config.json").read_bytes())
    log = JsonLog(run_dir / "evaluation_events.jsonl")
    runtime_class = EvaluationOnlyTaskRuntime if case_name == "proposed" else EvaluationOnlyRoCoRuntime
    runtime = runtime_class(run_dir, run_dir / "config.json", log, cli=["no-model-call"])
    controller, state = reconstruct(case_name, runtime, log)
    for path in [*(Path(x) for x in state["prior_hashes"]), case["source"] / "events.jsonl",
                 case["source"] / "config.json", Path(__file__).resolve()]:
        if str(path) not in runtime.protected_hashes:
            runtime.protected.append(path)
            runtime.protected_hashes[str(path)] = digest(path)
    budget = runtime.budget
    if case_name == "proposed":
        budget.cfg = state["base"].extended_budget_config(config)
    budget.model_normal = case["prior_calls"] + case["new_calls"] - (1 if case_name == "proposed" else 0)
    budget.eval_starts = budget.eval_normal = len(state["cpu_samples"])
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, code_sha256=runtime.protected_hashes,
              prior_model_calls=case["prior_calls"] + case["new_calls"],
              prior_evaluations=budget.eval_starts, evaluation_cpu_seconds_used=budget.cpu_finished)
    print(f"{case_name} 保存响应的首次评价启动；记录：{run_dir}", flush=True)
    with evaluation_guard(case_name, state, log):
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(evaluate_one, controller, candidate, case_name)
                       for candidate in state["candidates"]]
            results = [future.result() for future in futures]
    summary = {"status": "completed", "source_events": str(case["source"] / "events.jsonl"),
               "source_events_sha256": digest(case["source"] / "events.jsonl"), "new_model_calls": 0,
               "prior_model_calls": case["prior_calls"] + case["new_calls"],
               "prior_evaluations": len(state["cpu_samples"]), "new_evaluations": len(results),
               "results": [{"candidate_id": c.candidate_id, "status": r["status"],
                            "average": r.get("average"), "error": r.get("error")}
                           for c, r in zip(state["candidates"], results)]}
    with (run_dir / "summary.json").open("x", encoding="utf-8", newline="\n") as out:
        json.dump(summary, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    log.event("saved_evaluation_finished", status="completed", summary=str(run_dir / "summary.json"))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=CASES, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--simulate", action="store_true")
    actions.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.simulate:
        simulate(args.method)
    elif args.run:
        run(args.method)
    else:
        class CheckBackend:
            cfg = json.loads((CASES[args.method]["source"] / "config.json").read_text(encoding="utf-8"))
        _, state = reconstruct(args.method, CheckBackend(), None)
        print(json.dumps({"status": "passed", "method": args.method,
                          "candidates": [x.candidate_id for x in state["candidates"]],
                          "prior_evaluations": len(state["cpu_samples"]), "new_model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"保存响应评价停止：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
