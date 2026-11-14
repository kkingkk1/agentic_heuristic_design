"""只首次评价 RoCo 第 2 代限流后保存的 c0097，不发起模型调用。"""

import argparse
from contextlib import contextmanager
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import sys


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
SOURCE = ROOT / "outputs/p1_roco/20261112_233727_937536c2_resume"
sys.path.insert(0, str(ROOT))

from scripts import p1_formal_matrix as matrix
from scripts.p0_controller import StopRun, check_modification_scope
from scripts.p0_runtime import digest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load("saved_eoh_evaluation_common", RECOVERY / "evaluate_saved.py")
base = load("saved_eoh_roco_previous", RECOVERY / "resume_roco_generation2.py")
require = common.require
read_events = common.read_events
common.CASES["roco"] = {"source": SOURCE, "candidates": ("c0097",), "generation": 2,
                         "prior_calls": 99, "new_calls": 6, "prior_evaluations": 88,
                         "new_evaluations": 4, "cap": 420}


def reconstruct(case_name, backend, log):
    require(case_name == "roco", "评价对象不符。")
    controller, earlier = base.reconstruct(backend, log)
    config = earlier["config"]
    require((SOURCE / "config.json").read_bytes() == (base.SOURCE / "config.json").read_bytes(), "断点配置不一致。")
    events = read_events(SOURCE / "events.jsonl")
    require([x["event_index"] for x in events] == list(range(1, len(events) + 1)), "事件序号不连续。")
    preflights = [x for x in events if x["event"] == "preflight"]
    starts = [x for x in events if x["event"] == "formal_run_started"]
    require(len(preflights) == len(starts) == 1 and starts[0]["resume_of"] == str(base.SOURCE)
            and starts[0]["interrupted_events_sha256"] == digest(base.SOURCE / "events.jsonl")
            and starts[0]["saved_evaluation_sha256"] == digest(base.SAVED / "evaluation_events.jsonl")
            and starts[0]["total_model_call_cap"] == 420
            and events[-1]["event"] == "preflight_or_run_stopped", "续跑断点身份不符。")
    for name, expected in preflights[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
    failures = [x for x in events if x["event"] == "model_failure_classified"]
    attempts = {x["stage"]: x for x in events if x["event"] == "model_attempt"}
    responses = {x["candidate_id"]: x for x in events if x["event"] == "proposal_response"}
    evaluations = [x for x in events if x["event"] == "evaluation_budget_reserved"]
    require(len(failures) == 1 and failures[0]["stage"] == "eoh_E1_10" and failures[0]["category"] == "quota"
            and len(attempts) == 6 and len(evaluations) == 4
            and set(responses) == {"c0093", "c0094", "c0095", "c0096", "c0097"}
            and not any(x.get("candidate_id") == "c0097" for x in evaluations), "限流或保存提案不符。")
    candidate = next(x for x in earlier["pending"] if x.candidate_id == "c0097")
    row = responses["c0097"]
    specifications = {x.candidate_id: spec for x, spec in zip(earlier["pending"], earlier["pending_specs"])}
    stage, parents, prompt = specifications["c0097"]
    require(candidate.stage == stage == row["stage"] and list(candidate.parents) == parents
            and row["generation"] == 2 and row["prompt"] == attempts[stage]["prompt"] == prompt,
            "保存候选的父代或提示不符。")
    parsed = controller.parse_proposal(row["response"])
    check_modification_scope(parsed["code"])
    candidate.thought, candidate.code = parsed["thought"], parsed["code"]
    candidate.generation_order = row["generation_order"]
    cpu = [x["accounting"]["cpu_seconds"] for x in events if x["event"] == "evaluation_cpu_final"]
    require(len(earlier["cpu_samples"]) == 88 and len(cpu) == 4, "累计评价 CPU 不符。")
    hashes = dict(preflights[0]["code_sha256"])
    hashes[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    return controller, {"base": base, "config": config, "candidates": [candidate],
                        "cpu_samples": earlier["cpu_samples"] + cpu, "prior_hashes": hashes}


@contextmanager
def evaluation_guard(case_name, state, log):
    require(case_name == "roco" and os.name == "nt", "评价对象或平台不符。")
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
        config = state["config"]
        matrix.verify_training_config(config)
        expected = {base.base.previous.SOURCE, base.base.SOURCE, base.SOURCE, SOURCE}
        started = set()
        for path in (ROOT / "outputs/p1_roco").glob("*/config.json"):
            other = json.loads(path.read_text(encoding="utf-8"))
            if (other.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                    or matrix.run_identity(other) != matrix.run_identity(config)):
                continue
            event_file = path.parent / "events.jsonl"
            if event_file.is_file() and any(x["event"] == "formal_run_started" for x in read_events(event_file)):
                started.add(path.parent.resolve())
        require(started == {x.resolve() for x in expected}, "RoCo 研发存在未知的已启动运行。")
        common.assert_not_evaluated(config, ["c0097"])
        log.event("saved_evaluation_started", source_events=str(SOURCE / "events.jsonl"),
                  source_events_sha256=digest(SOURCE / "events.jsonl"), candidates=["c0097"],
                  prior_model_calls=105, prior_evaluations=92, model_call_cap=420)
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


common.reconstruct = reconstruct
common.evaluation_guard = evaluation_guard


def simulate():
    from tests.test_p0_controller import FixtureBackend, MemoryLog
    backend, log = FixtureBackend(), MemoryLog()
    controller, state = reconstruct("roco", backend, log)
    with evaluation_guard("roco", state, MemoryLog()):
        pass
    common.evaluate_one(controller, state["candidates"][0], "roco")
    require(not backend.calls and len(backend.evaluations) == 1
            and [x["candidate_id"] for x in log.items if x["event"] == "evaluation"] == ["c0097"],
            "首次评价或模型调用计数不符。")
    print(json.dumps({"status": "passed", "candidate": "c0097", "new_model_calls": 0,
                      "new_evaluations": 1}, ensure_ascii=False))


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
        common.run("roco")
    else:
        class CheckBackend:
            cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
        _, state = reconstruct("roco", CheckBackend(), None)
        print(json.dumps({"status": "passed", "candidate": "c0097", "prior_model_calls": 105,
                          "prior_evaluations": len(state["cpu_samples"]), "new_model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"RoCo 保存响应评价停止：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
