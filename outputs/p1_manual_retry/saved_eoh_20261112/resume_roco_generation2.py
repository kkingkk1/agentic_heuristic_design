"""从 RoCo 第 2 次研发第 2 代未启动槽位续跑，保留限流失败及首次评价。"""

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
SOURCE = ROOT / "outputs/p1_roco/20261112_230339_86dc16b5_resume"
SAVED = RECOVERY / "20261112_231034_roco_5d06e3ec_evaluation"
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate, p1_formal_matrix as matrix
from scripts.p0_controller import Candidate, StopRun, check_modification_scope, timing_from_result
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from scripts.p1_classified_runtime import ClassifiedRoCoRuntime

spec = importlib.util.spec_from_file_location(
    "roco_second_generation2_base",
    ROOT / "outputs/p1_manual_retry/roco_2_quota_recovery_20261112/prepare_resume.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
require = base.require
read_events = base.read_jsonl


class ResumeController(base.ContinueController):
    def __init__(self, *args, **kwargs):
        self._replay_eoh = None
        self._replay_specs = None
        super().__init__(*args, **kwargs)

    def batch(self, specs):
        if self._future_population is None and self._replay_eoh is not None:
            require(len(specs) == len(self._replay_specs) == 40, "第 2 代 EoH 重放槽位不符。")
            for actual, expected in zip(specs, self._replay_specs):
                stage, parents, prompt = actual
                require((stage, [x.candidate_id for x in parents], prompt) == expected,
                        f"第 2 代 EoH 提示或父代不符：{stage}")
            candidates, self._replay_eoh = self._replay_eoh, None
            self.event("reused_eoh_proposals", generation=2, source_events=str(SOURCE / "events.jsonl"),
                       saved_evaluation_events=str(SAVED / "evaluation_events.jsonl"))
            return candidates
        return super().batch(specs)


class ResumeRuntime(ClassifiedRoCoRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resume_first_generation = True

    def before_generation(self):
        if self.resume_first_generation:
            self.resume_first_generation = False
            self.log.event("reused_generation_budget_check", generation=2, source_events=str(SOURCE / "events.jsonl"))
            return
        return super().before_generation()


def apply_result(controller, candidate, response, result, directory):
    parsed = controller.parse_proposal(response["response"])
    check_modification_scope(parsed["code"])
    candidate.thought, candidate.code = parsed["thought"], parsed["code"]
    candidate.generation_order = response["generation_order"]
    require(json.loads((directory / "result.json").read_text(encoding="utf-8")) == result
            and (directory / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8") == candidate.code + "\n",
            f"候选或评价记录不一致：{candidate.candidate_id}")
    if result["status"] != "passed":
        candidate.error = str(result.get("error") or result["status"])
        return
    checked = p0_tsp_aco_evaluate.read_complete(directory / "stdout.txt")
    require(checked["average"] == result["average"] and checked["fitness"] == result["fitness"]
            and checked["objectives"] == result["objectives"], f"评价输出不一致：{candidate.candidate_id}")
    candidate.objective = candidate.raw_objective = float(result["average"])
    candidate.fitness = float(result["fitness"])
    candidate.timing_seconds, candidate.timing_scope = timing_from_result(result)
    candidate.completed_order = result["completed_monotonic_ns"]
    candidate.evaluation_origin = candidate.candidate_id


def reconstruct(backend, log):
    with patch.object(base, "ContinueController", ResumeController):
        controller, earlier = base.reconstruct(backend, log)
    config = earlier["config"]
    require((SOURCE / "config.json").read_bytes() == (base.SOURCE / "config.json").read_bytes()
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
            and starts[0]["resume_of"] == str(base.SOURCE)
            and starts[0]["interrupted_events_sha256"] == digest(base.SOURCE / "events.jsonl")
            and starts[0]["saved_evaluation_sha256"] == digest(base.SAVED / "evaluation_events.jsonl")
            and starts[0]["total_model_call_cap"] == 420
            and events[-1]["event"] == "preflight_or_run_stopped", "第 2 代断点身份不符。")
    require(summary["status"] == "completed" and summary["new_model_calls"] == 0
            and summary["new_evaluations"] == 3 and summary["prior_model_calls"] == 99
            and summary["source_events_sha256"] == digest(SOURCE / "events.jsonl"), "首次评价身份不符。")
    for row in (preflight[0], saved_preflight[0]):
        for name, expected in row["code_sha256"].items():
            require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
    failures = [x for x in events if x["event"] == "model_failure_classified"]
    require(len(failures) == 1 and failures[0]["stage"] == "eoh_E1_03" and failures[0]["category"] == "quota"
            and len([x for x in events if x["event"] == "model_attempt"]) == 6,
            "限流失败或模型调用数不符。")
    responses = {x["candidate_id"]: x for x in events if x["event"] == "proposal_response"}
    source_evaluations = {x["candidate_id"]: x["result"] for x in events if x["event"] == "evaluation"}
    saved_evaluations = {x["candidate_id"]: x["result"] for x in saved if x["event"] == "evaluation"}
    require(set(source_evaluations) == {"c0087", "c0088"}
            and set(saved_evaluations) == {"c0089", "c0090", "c0092"}
            and set(responses) == set(source_evaluations) | set(saved_evaluations), "候选或首次评价不符。")
    candidates = {x.candidate_id: x for x in controller.candidates}
    for ident, result in source_evaluations.items():
        apply_result(controller, candidates[ident], responses[ident], result, SOURCE / "candidates" / ident)
    require(sorted(responses[x]["generation_order"] for x in source_evaluations) == [84, 85], "第 1 代生成顺序不符。")
    controller._generated = 85
    pool = [*earlier["eoh"], *earlier["last"], earlier["elite_fusion"], *earlier["mutations"]]
    selected = controller.select(pool, controller.n)
    selections = [x for x in events if x["event"] == "selection"]
    require(len(selections) == 1 and selections[0]["generation"] == 1
            and selections[0]["pool"] == [x.candidate_id for x in pool if x is not None]
            and selections[0]["selected"] == [{"candidate_id": x.candidate_id, "evaluation_origin": x.evaluation_origin}
                                              for x in selected], "第 1 代筛选重建不一致。")
    history = [x for x in events if x["event"] == "history_appended"]
    role_keys = {label: role for role, label in base.ROLE_LABELS.items()}
    require(len(history) == 3 and all(x["role"] in role_keys for x in history)
            and {role_keys[x["role"]] for x in history} == set(earlier["lessons"]), "第 1 代历史不完整。")
    for row in history:
        role = role_keys[row["role"]]
        require(row["generation"] == 1 and row["reflection"] == earlier["lessons"][role], "角色历史不符。")
        controller.history[role].append(row["reflection"])
    registrations = [x for x in events if x["event"] == "candidate_registered"]
    skipped = [x["candidate_id"] for x in events if x["event"] == "candidate_skipped"]
    require(len(registrations) == 40 and skipped == [f"c{i:04d}" for i in range(93, 129)],
            "第 2 代未启动槽位不符。")
    for row in registrations:
        ident, seq = row["candidate_id"], row["sequence"]
        require(ident == f"c{seq:04d}" and seq == len(controller.candidates) + 1, "候选编号不连续。")
        candidate = Candidate(ident, seq, row["stage"], tuple(row["parents"]))
        if ident == "c0091":
            candidate.error = "模型返回 HTTP 429；保留失败，不自动重试。"
        controller.candidates.append(candidate)
        candidates[ident] = candidate
    for ident, result in saved_evaluations.items():
        apply_result(controller, candidates[ident], responses[ident], result, SAVED / "candidates" / ident)
    require(sorted(responses[x]["generation_order"] for x in saved_evaluations) == [86, 87, 88], "第 2 代生成顺序不符。")
    controller._generated = 88
    require(max(c.completed_order for c in controller.candidates if c.valid) < time.monotonic_ns(),
            "当前单调时钟早于历史评价。")
    attempts = {x["stage"]: x for x in events if x["event"] == "model_attempt" and x["stage"].startswith("eoh_")}
    require(len(attempts) == 4, "第 2 代模型调用数不符。")
    directions = {"E1": "Create a heuristic as different as possible from these parents.",
                  "E2": "Identify their shared idea and create a new structure based on it.",
                  "M1": "Modify this heuristic's structure to improve performance.",
                  "M2": "Adjust parameters while retaining this heuristic's overall logic."}
    random_state = controller.random.getstate()
    specs = []
    for row in registrations:
        op = row["stage"].split("_")[1]
        parents = controller.eoh_parents(selected, 5 if op in ("E1", "E2") else 1)
        require([x.candidate_id for x in parents] == row["parents"], f"随机父代不符：{row['candidate_id']}")
        context = "\n".join(f"Algorithm description: {x.thought}\nCode:\n{x.code}\n" for x in parents)
        prompt = f"{controller.task}\n{context}{directions[op]}\n{controller.output}"
        if row["stage"] in attempts:
            require(attempts[row["stage"]]["prompt"] == prompt, f"原提示不符：{row['stage']}")
        specs.append((row["stage"], row["parents"], prompt))
    controller.random.setstate(random_state)
    controller._replay_eoh = [candidates[x["candidate_id"]] for x in registrations]
    controller._replay_specs = specs
    cpu = [x["accounting"]["cpu_seconds"] for x in (*events, *saved) if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == 5 and len(earlier["cpu_samples"]) == 83, "累计评价 CPU 不符。")
    return controller, {"config": config, "population": selected,
                        "pending": [candidates[x] for x in skipped], "pending_specs": specs[4:],
                        "cpu_samples": earlier["cpu_samples"] + cpu,
                        "prior_hashes": preflight[0]["code_sha256"],
                        "saved_hashes": saved_preflight[0]["code_sha256"]}


def run_pending(controller, state):
    require(len(state["pending"]) == len(state["pending_specs"]) == 36, "待执行槽位数量不符。")
    controller.generation = 2
    entries = []
    for candidate, (stage, parents, prompt) in zip(state["pending"], state["pending_specs"]):
        require(candidate.stage == stage and list(candidate.parents) == parents, "待执行候选或父代不符。")
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
    controller.generation = 1


@contextmanager
def recovery_guard(config, run_dir, log, resume_basis="offline_simulation"):
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
            event_file = path.parent / "events.jsonl"
            if event_file.is_file() and any(x["event"] == "formal_run_started" for x in read_events(event_file)):
                started.add(path.parent.resolve())
        require(started == {base.previous.SOURCE.resolve(), base.SOURCE.resolve(), SOURCE.resolve()},
                "RoCo 第 2 次研发存在未知的已启动运行。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=config["formal_replicate"], selection_seed=task["selection_seed"],
                  resume_of=str(SOURCE), interrupted_events_sha256=digest(SOURCE / "events.jsonl"),
                  saved_evaluation_sha256=digest(SAVED / "evaluation_events.jsonl"), total_model_call_cap=420,
                  resume_basis=resume_basis,
                  note="保留 429 失败；复用首次评价，仅执行未启动的原槽位。")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def simulate():
    from tests.test_p0_controller import FixtureBackend, MemoryLog
    backend, log = FixtureBackend(), MemoryLog()
    controller, state = reconstruct(backend, log)
    with recovery_guard(state["config"], RECOVERY / "unused_roco_guard_fixture", MemoryLog()):
        code = ("from pathlib import Path\nfrom scripts import p1_formal_matrix as m\ntry:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED", "续跑互斥锁不一致。")
    run_pending(controller, state)
    require(len(backend.calls) == 36 and len(backend.evaluations) == 36
            and all(stage != "eoh_E1_03" for stage, _ in backend.calls), "未启动槽位或限流失败保留不符。")
    selected = controller.run_future(state["population"])
    require(len(selected) == 10 and controller.generation == 6 and len(backend.calls) == 321
            and len(backend.evaluations) == 286
            and [x["generation"] for x in log.items if x["event"] == "selection"] == [2, 3, 4, 5, 6],
            "后续代数、预算槽位或筛选不符。")
    print(json.dumps({"status": "passed", "prior_model_calls": 99, "prior_evaluations": 88,
                      "unstarted_eoh": 36, "remaining_model_calls": 321,
                      "remaining_evaluation_slots": 286, "model_call_cap": 420,
                      "formal_mutex_shared": True}, ensure_ascii=False))


def resume(resume_basis):
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
    runtime = ResumeRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    controller, state = reconstruct(runtime, log)
    for path in [*(Path(x) for x in state["prior_hashes"]), *(Path(x) for x in state["saved_hashes"]),
                 SOURCE / "events.jsonl", SOURCE / "config.json", SAVED / "evaluation_events.jsonl",
                 SAVED / "summary.json", Path(__file__).resolve()]:
        if str(path) not in runtime.protected_hashes:
            runtime.protected.append(path)
            runtime.protected_hashes[str(path)] = digest(path)
    budget = runtime.budget
    budget.model_normal = 99
    budget.eval_starts = budget.eval_normal = 88
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
              code_sha256=runtime.protected_hashes, selection_seed=config["task"]["selection_seed"],
              generations=6, continuation=True, normal_model_slots_used=99,
              physical_model_cli_invocations_used=99, evaluation_starts_used=88, resume_basis=resume_basis,
              evaluation_cpu_seconds_used=budget.cpu_finished)
    print(f"RoCo 第 2 次研发第 2 代恢复检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(config, run_dir, log, resume_basis=resume_basis):
            log.event("resume_checkpoint", interrupted_run=str(SOURCE), generation=2,
                      failed_candidate_id="c0091", saved_evaluation_run=str(SAVED),
                      unstarted_candidates=[x.candidate_id for x in state["pending"]],
                      model_calls_already_used=99, evaluation_starts_already_used=88, total_model_call_cap=420)
            run_pending(controller, state)
            selected = controller.run_future(state["population"])
            require(len(selected) == 10 and controller.generation == 6, "RoCo 第 2 次研发未完成六代。")
        print(f"RoCo 第 2 次研发六代完成；记录：{run_dir}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"RoCo 第 2 次研发续跑停止：{exc}；记录：{run_dir}", file=sys.stderr, flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--simulate", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--service-restored", action="store_true", help="服务已恢复时续跑")
    parser.add_argument("--user-requested-resume", action="store_true", help="用户明确要求手动续跑；再次限流时停止")
    args = parser.parse_args()
    if args.simulate:
        simulate()
        return 0
    if args.run:
        require(args.service_restored != args.user_requested_resume,
                "续跑须明确记录服务恢复或用户手动指示其中一项。")
        return resume("user_requested" if args.user_requested_resume else "service_restored")
    class CheckBackend:
        cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    controller, state = reconstruct(CheckBackend(), None)
    print(json.dumps({"status": "passed", "model_calls_used": 99,
                      "evaluation_starts": len(state["cpu_samples"]), "unstarted_eoh": len(state["pending"]),
                      "total_model_call_cap": 420}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"RoCo 第 2 代恢复检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
