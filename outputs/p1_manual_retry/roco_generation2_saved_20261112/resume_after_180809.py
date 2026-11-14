"""复用 RoCo 第 2 代已完成的首次评价，继续尚未启动的原定槽位。"""

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


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
SOURCE = ROOT / "outputs/p1_roco/20261112_180809_e040f878_resume"
SAVED = RECOVERY / "20261112_181024_8854108b_evaluation"
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate, p1_formal_matrix as matrix
from scripts.p0_controller import StopRun, check_modification_scope
from scripts.p0_runtime import JsonLog, digest, memory_and_priority

spec = importlib.util.spec_from_file_location("roco_saved_after_180809", RECOVERY / "evaluate_180809_saved.py")
prior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prior
spec.loader.exec_module(prior)
require, read_jsonl = prior.require, prior.read_jsonl


def reconstruct(backend, log):
    controller, state = prior.prior.reconstruct(backend, log)
    require((SOURCE / "config.json").read_bytes() == (prior.prior.SOURCE / "config.json").read_bytes()
            == (SAVED / "config.json").read_bytes(), "恢复配置不一致。")
    events = read_jsonl(SOURCE / "events.jsonl")
    preflight = [x for x in events if x["event"] == "preflight"]
    starts = [x for x in events if x["event"] == "formal_run_started"]
    require(len(preflight) == len(starts) == 1 and starts[0]["resume_of"] == str(prior.prior.SOURCE)
            and events[-1]["event"] == "preflight_or_run_stopped", "断点身份不符。")
    for name, expected in preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
    failures = [x for x in events if x["event"] == "model_failure_classified"]
    require({(x["stage"], x["category"]) for x in failures} == {("eoh_E2_04", "unknown"), ("eoh_E2_07", "unknown")}
            and len([x for x in events if x["event"] == "model_attempt"]) == 4, "服务失败或调用数不符。")
    skipped = [x["candidate_id"] for x in events if x["event"] == "candidate_skipped"]
    require(skipped == [f"c{x:04d}" for x in range(106, 129)], "未启动槽位不符。")
    responses = {x["candidate_id"]: x for x in events if x["event"] == "proposal_response"}
    require(set(responses) == {"c0103", "c0104"}
            and sorted(x["generation_order"] for x in responses.values()) == [97, 98], "已返回提案不符。")
    saved_events = read_jsonl(SAVED / "evaluation_events.jsonl")
    summary = json.loads((SAVED / "summary.json").read_text(encoding="utf-8"))
    require(summary["status"] == "completed" and summary["source_events_sha256"] == digest(SOURCE / "events.jsonl")
            and summary["new_model_calls"] == 0 and summary["new_evaluations"] == 2,
            "保存响应的首次评价身份不符。")
    saved_preflight = [x for x in saved_events if x["event"] == "preflight"]
    require(len(saved_preflight) == 1, "首次评价启动检查不唯一。")
    for name, expected in saved_preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"首次评价的受保护文件变化：{name}")
    pending = {c.candidate_id: (c, s) for c, s in zip(state["pending"], state["pending_specs"])}
    for ident, row in responses.items():
        candidate, (stage, parents, prompt) = pending[ident]
        require(candidate.stage == stage == row["stage"] and list(candidate.parents) == parents
                and row["prompt"] == prompt and row["generation"] == 2, f"提案来源不符：{ident}")
        parsed = controller.parse_proposal(row["response"])
        check_modification_scope(parsed["code"])
        candidate.thought, candidate.code = parsed["thought"], parsed["code"]
        candidate.generation_order = row["generation_order"]
    for ident in ("c0102", "c0105"):
        pending[ident][0].error = "模型服务返回 Bad Request，未产生提案；不自动重试。"
    evaluated = [(x, SOURCE) for x in events if x["event"] == "evaluation"]
    evaluated += [(x, SAVED) for x in saved_events if x["event"] == "evaluation"]
    require({x["candidate_id"] for x, _ in evaluated} == {"c0097", "c0103", "c0104"}, "首次评价不完整。")
    candidates = {c.candidate_id: c for c in controller.candidates}
    for row, origin in evaluated:
        ident, result = row["candidate_id"], row["result"]
        candidate = candidates[ident]
        directory = origin / "candidates" / ident
        require(json.loads((directory / "result.json").read_text(encoding="utf-8")) == result
                and (directory / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8") == candidate.code + "\n",
                f"候选或评价结果不一致：{ident}")
        if result["status"] != "passed":
            candidate.error = str(result.get("error") or result["status"])
            continue
        checked = p0_tsp_aco_evaluate.read_complete(directory / "stdout.txt")
        require(checked["average"] == result["average"] and checked["fitness"] == result["fitness"]
                and checked["objectives"] == result["objectives"], f"评价输出不一致：{ident}")
        candidate.objective, candidate.fitness = float(result["average"]), float(result["fitness"])
        candidate.completed_order, candidate.evaluation_origin = result["completed_monotonic_ns"], ident
    controller._generated = 98
    require(max(c.completed_order for c in controller.candidates if c.valid) < time.monotonic_ns(),
            "当前单调时钟早于历史评价。")
    cpu = [x["accounting"]["cpu_seconds"] for x in events if x["event"] == "evaluation_cpu_final"]
    cpu += [x["accounting"]["cpu_seconds"] for x in saved_events if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == 3 and len(state["cpu_samples"]) == 94, "累计评价 CPU 不符。")
    return controller, {"config": state["config"], "population": state["population"],
                        "pending": [pending[x][0] for x in skipped],
                        "pending_specs": [pending[x][1] for x in skipped],
                        "cpu_samples": state["cpu_samples"] + cpu,
                        "prior_hashes": preflight[0]["code_sha256"], "saved_hashes": saved_preflight[0]["code_sha256"]}


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
        for path in (ROOT / "outputs/p1_roco").glob("*/config.json"):
            if path.parent.resolve() == run_dir.resolve():
                continue
            other = json.loads(path.read_text(encoding="utf-8"))
            if (other.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                    or matrix.run_identity(other) != matrix.run_identity(config)):
                continue
            event_path = path.parent / "events.jsonl"
            if event_path.is_file() and any(x["event"] == "formal_run_started" for x in read_jsonl(event_path)):
                started.add(path.parent.resolve())
        require(started == {prior.prior.prior.prior.previous.SOURCE.resolve(), prior.prior.prior.prior.SOURCE.resolve(),
                            prior.prior.prior.SOURCE.resolve(), prior.prior.SOURCE.resolve(), SOURCE.resolve()},
                "RoCo 首组存在未知已启动运行。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=config["formal_replicate"], selection_seed=task["selection_seed"],
                  resume_of=str(SOURCE), interrupted_events_sha256=digest(SOURCE / "events.jsonl"),
                  saved_evaluation_sha256=digest(SAVED / "evaluation_events.jsonl"), total_model_call_cap=420,
                  note="保留已失败请求，复用首次评价，只执行尚未启动的原槽位。")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def run_pending(controller, state):
    require(len(state["pending"]) == len(state["pending_specs"]) == 23, "未启动槽位数量不符。")
    controller.generation = 2
    entries = []
    for candidate, (stage, parents, prompt) in zip(state["pending"], state["pending_specs"]):
        require(candidate.stage == stage and list(candidate.parents) == parents, "候选或父代不符。")
        controller.event("reused_candidate_registration", generation=2, candidate_id=candidate.candidate_id,
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
    require(len(backend.calls) == len(backend.evaluations) == 23
            and all(stage not in ("eoh_E2_04", "eoh_E2_07") for stage, _ in backend.calls),
            "未启动槽位或失败请求处理不符。")
    selected = controller.run_remaining(state["population"])
    require(len(selected) == 10 and controller.generation == 6 and len(backend.calls) == 308
            and len(backend.evaluations) == 273
            and [x["generation"] for x in log.items if x["event"] == "selection"] == [2, 3, 4, 5, 6],
            "后续代数、预算槽位或筛选不符。")
    print(json.dumps({"status": "passed", "prior_model_calls": 112, "prior_evaluations": 97,
                      "unstarted_proposals": 23, "later_model_calls": 308,
                      "later_evaluation_slots": 273, "model_call_cap": 420,
                      "formal_mutex_shared": True}, ensure_ascii=False))


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
    run_dir = ROOT / "outputs/p1_roco" / (datetime.now().strftime("%Y%m%d_%H%M%S")
                                            + "_" + uuid.uuid4().hex[:8] + "_resume")
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("xb") as out:
        out.write((SOURCE / "config.json").read_bytes())
    log = JsonLog(run_dir / "events.jsonl")
    runtime = prior.prior.prior.prior.RecoveryRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    controller, state = reconstruct(runtime, log)
    paths = [*(Path(x) for x in state["prior_hashes"]), *(Path(x) for x in state["saved_hashes"]),
             SOURCE / "events.jsonl", SOURCE / "config.json", SAVED / "evaluation_events.jsonl",
             SAVED / "summary.json", RECOVERY / "evaluate_180809_saved.py", Path(__file__).resolve()]
    for ident in ("c0103", "c0104"):
        paths.extend(SAVED / "candidates" / ident / item for item in ("result.json", "stdout.txt", "problems/tsp_aco/gpt.py"))
    paths.extend(SOURCE / "candidates/c0097" / item for item in ("result.json", "problems/tsp_aco/gpt.py"))
    for path in paths:
        if str(path) not in runtime.protected_hashes:
            runtime.protected.append(path)
            runtime.protected_hashes[str(path)] = digest(path)
    budget = runtime.budget
    budget.model_normal = 112
    budget.eval_starts = budget.eval_normal = 97
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
              code_sha256=runtime.protected_hashes, selection_seed=config["task"]["selection_seed"],
              generations=6, continuation=True, normal_model_slots_used=112,
              physical_model_cli_invocations_used=112, evaluation_starts_used=97,
              evaluation_cpu_seconds_used=budget.cpu_finished)
    print(f"RoCo 第 2 代剩余槽位启动前检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(config, run_dir, log):
            log.event("resume_checkpoint", interrupted_run=str(SOURCE), generation=2,
                      failed_candidate_ids=["c0102", "c0105"], saved_evaluation_run=str(SAVED),
                      unstarted_candidates=[x.candidate_id for x in state["pending"]],
                      model_calls_already_used=112, evaluation_starts_already_used=97, total_model_call_cap=420)
            run_pending(controller, state)
            selected = controller.run_remaining(state["population"])
            require(len(selected) == 10 and controller.generation == 6, "RoCo 未完成六代。")
        print(f"RoCo 六代完成；记录：{run_dir}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"RoCo 第 2 代续跑停止：{exc}；记录：{run_dir}", file=sys.stderr, flush=True)
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
    print(json.dumps({"status": "passed", "candidates": len(controller.candidates), "model_calls_used": 112,
                      "evaluation_starts": len(state["cpu_samples"]), "unstarted_candidates": len(state["pending"]),
                      "model_call_cap": 420}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"RoCo 剩余槽位检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
