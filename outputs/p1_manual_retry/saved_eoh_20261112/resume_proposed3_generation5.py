"""重放本方法第 3 次研发第 3、4 代，并续跑第 5 代未启动槽位。"""

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
SOURCE = ROOT / "outputs/p1_control/20261113_171057_018c3785_resume"
PREVIOUS_SCRIPT = RECOVERY / "resume_proposed3_generation3_chain.py"
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate, p1_formal_matrix as matrix
from scripts import p1_classified_runtime
from scripts.p0_controller import Controller, StopRun, check_modification_scope, timing_from_result
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from tests.test_p0_controller import FixtureBackend, MemoryLog


spec = importlib.util.spec_from_file_location("proposed3_generation3_chain_recovery", PREVIOUS_SCRIPT)
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
require, read_jsonl = previous.require, previous.read_jsonl


class ReplayBoundary(BaseException):
    pass


class ReplayController(previous.previous.ReplayController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_forward = True
        self.current_source_rows = []
        self.fourth_population = None
        self.fourth_random_state = None
        self.fifth_candidates = None
        self.fifth_specs = None
        self.final_phase = None
        self.final_population = None
        self.used_roles = set()

    def event(self, kind, **fields):
        if not self.replay_forward and self.final_phase == 1 and kind == "initial_population":
            return Controller.event(self, "resume_population", source_generation=4, **fields)
        return super().event(kind, **fields)

    def select(self, candidates, needed):
        selected = super().select(candidates, needed)
        if self.replay_forward and not self.replay_generation2 and self.generation == 4 and needed == self.n:
            self.fourth_population = selected
            self.fourth_random_state = self.random.getstate()
        return selected

    def request(self, stage, prompt, parser):
        if not self.replay_forward:
            return Controller.request(self, stage, prompt, parser)
        if not self.replay_generation2 and self.generation in (3, 4):
            rows = [x for x in self.current_source_rows if x["event"] == "model_response"
                    and x.get("generation") == self.generation and x["stage"] == stage]
            key = (self.generation, stage)
            require(len(rows) == 1 and key not in self.used_roles and rows[0]["prompt"] == prompt,
                    f"历史角色请求不符：{key}")
            self.used_roles.add(key)
            self.event("reused_role_response", generation=self.generation, stage=stage,
                       source_events=str(SOURCE / "events.jsonl"))
            return parser(rows[0]["response"])
        return super().request(stage, prompt, parser)

    def batch(self, specs):
        if not self.replay_forward:
            if self.final_phase == 0:
                require(len(specs) == self.initial_count and all(stage.startswith("init_") and not parents
                        for stage, parents, _ in specs), "续跑种群位置不符。")
                self.final_phase = 1
                return self.final_population
            if self.final_phase == 1:
                actual = [(stage, [x.candidate_id for x in parents], prompt) for stage, parents, prompt in specs]
                require(self.generation == 5 and actual == self.fifth_specs, "第 5 代重放父代或提示不符。")
                self.final_phase = 2
                self.event("reused_eoh_proposals", generation=5, source_events=str(SOURCE / "events.jsonl"))
                return self.fifth_candidates
            return Controller.batch(self, specs)
        if self.replay_generation2 or self.future_phase in (0, 1):
            return super().batch(specs)
        require(self.generation in (3, 4, 5), "重放到未知代次。")
        entries = [(self.register(stage, tuple(p.candidate_id for p in parents)), prompt)
                   for stage, parents, prompt in specs]
        registrations = {x["candidate_id"]: x for x in self.current_source_rows
                         if x["event"] == "candidate_registered" and x.get("generation") == self.generation}
        for candidate, _ in entries:
            row = registrations.get(candidate.candidate_id)
            require(row is not None and row["stage"] == candidate.stage and row["parents"] == list(candidate.parents),
                    f"历史候选注册不符：{candidate.candidate_id}")
        if self.generation == 5:
            selection = next(x for x in self.current_source_rows if x["event"] == "selection" and x.get("generation") == 4)
            attempts = {x["stage"]: x for x in self.current_source_rows if x["event"] == "model_attempt"
                        and x["event_index"] > selection["event_index"]}
            require(len(entries) == len(registrations) == 40 and len(attempts) == 33
                    and self.fourth_population is not None, "第 5 代断点槽位不完整。")
            for candidate, prompt in entries:
                if candidate.stage in attempts:
                    require(attempts[candidate.stage]["prompt"] == prompt, f"第 5 代提示不符：{candidate.stage}")
            self.fifth_candidates = [candidate for candidate, _ in entries]
            self.fifth_specs = [(candidate.stage, list(candidate.parents), prompt) for candidate, prompt in entries]
            raise ReplayBoundary()
        responses = {x["candidate_id"]: x for x in self.current_source_rows if x["event"] == "proposal_response"}
        results = {x["candidate_id"]: x["result"] for x in self.current_source_rows if x["event"] == "evaluation"}
        orders = []
        for candidate, prompt in entries:
            response, result = responses.get(candidate.candidate_id), results.get(candidate.candidate_id)
            require(response is not None and response["prompt"] == prompt and result is not None,
                    f"历史提案、提示或评价缺失：{candidate.candidate_id}")
            apply_result(self, candidate, response, result)
            orders.append(response["generation_order"])
        require(sorted(orders) == list(range(self._generated + 1, self._generated + len(orders) + 1)),
                "生成顺序不连续。")
        self._generated += len(orders)
        return [candidate for candidate, _ in entries]

    def run_from_fifth(self, population):
        require(len(population) == 10 and self.fourth_random_state is not None, "第 4 代种群不完整。")
        self.replay_forward = False
        self.final_phase = 0
        self.final_population = population
        self._generation_offset = 4
        self.generations = 2
        self.random.setstate(self.fourth_random_state)
        return Controller.run(self)


class RecoveryRuntime(previous.previous.RecoveryRuntime):
    def before_generation(self):
        if self.resume_first_generation:
            self.resume_first_generation = False
            self.log.event("reused_generation_budget_check", generation=5, source_events=str(SOURCE / "events.jsonl"))
            return
        return super().before_generation()


def apply_result(controller, candidate, response, result):
    parsed = controller.parse_proposal(response["response"])
    check_modification_scope(parsed["code"])
    candidate.thought, candidate.code = parsed["thought"], parsed["code"]
    candidate.generation_order = response["generation_order"]
    directory = SOURCE / "candidates" / candidate.candidate_id
    require(json.loads((directory / "result.json").read_text(encoding="utf-8")) == result
            and (directory / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8") == candidate.code + "\n",
            f"候选或评价记录不符：{candidate.candidate_id}")
    if result["status"] != "passed":
        require(result["status"] == "candidate_error" or
                (candidate.candidate_id == "c0293" and result["status"] == "infrastructure_error"
                 and result.get("error") == "其他并发任务停止运行。"), "存在未预期的评价失败。")
        candidate.error = str(result.get("error") or result["status"])
        return
    checked = p0_tsp_aco_evaluate.read_complete(directory / "stdout.txt")
    require(all(checked[key] == result[key] for key in ("average", "fitness", "objectives")),
            f"评价输出不符：{candidate.candidate_id}")
    candidate.objective = candidate.raw_objective = float(result["average"])
    candidate.fitness = float(result["fitness"])
    candidate.timing_seconds, candidate.timing_scope = timing_from_result(result)
    candidate.completed_order = result["completed_monotonic_ns"]
    candidate.evaluation_origin = candidate.candidate_id


def reconstruct(backend, log):
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))

    class OfflineBackend:
        cfg = config

        def before_generation(self):
            pass

    segments = previous.find_segments(config)
    indexes = [i for i, (directory, _) in enumerate(segments) if directory.resolve() == SOURCE.resolve()]
    require(len(indexes) == 1 and indexes[0] > 0 and indexes[0] == len(segments) - 1, "无法唯一定位第 5 代断点。")
    index = indexes[0]
    replay_log = MemoryLog()
    with patch.object(previous, "find_segments", return_value=segments[:index]), \
            patch.object(previous.previous, "ReplayController", ReplayController):
        controller, state = previous.reconstruct(OfflineBackend(), replay_log)
    directory, rows = segments[index]
    require((directory / "config.json").read_bytes() == (previous.previous.SOURCE / "config.json").read_bytes()
            and [x["event_index"] for x in rows] == list(range(1, len(rows) + 1)), "断点配置或事件编号不符。")
    started = [x for x in rows if x["event"] == "formal_run_started"]
    preflight = [x for x in rows if x["event"] == "preflight"]
    require(len(started) == len(preflight) == 1 and started[0]["resume_of"] == str(state["latest"])
            and started[0]["interrupted_events_sha256"] == digest(state["latest"] / "events.jsonl")
            and started[0]["total_model_call_cap"] == 420 and started[0]["effective_model_concurrency"] == 1
            and preflight[0]["physical_model_cli_invocations_used"] == state["model_calls"] == 184
            and preflight[0]["evaluation_starts_used"] == state["evaluation_starts"] == 165
            and rows[-1]["event"] == "preflight_or_run_stopped", "断点链、预算或停止状态不符。")
    for name, expected in preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
    attempts = [x for x in rows if x["event"] == "model_attempt"]
    failures = [x for x in rows if x["event"] == "model_failure_classified"]
    responses = {x["candidate_id"]: x for x in rows if x["event"] == "proposal_response"}
    results = {x["candidate_id"]: x["result"] for x in rows if x["event"] == "evaluation"}
    eval_ids = [x["candidate_id"] for x in rows if x["event"] == "evaluation_budget_reserved"]
    require(len(attempts) == 138 and len(responses) == len(results) == len(eval_ids) == 123
            and len(set(eval_ids)) == 123 and set(eval_ids) == set(responses) == set(results)
            and len(failures) == 1 and failures[0]["stage"] == "eoh_M2_03" and failures[0]["category"] == "network",
            "请求、评价或失败状态不符。")
    registrations = [x["candidate_id"] for x in rows if x["event"] == "reused_candidate_registration"]
    require(registrations == [x.candidate_id for x in state["pending"]], "第 3 代待恢复候选不符。")
    by_stage = {x["stage"]: x for x in attempts[:len(state["pending"]) ]}
    orders = []
    for candidate, (stage, parents, prompt) in zip(state["pending"], state["pending_specs"]):
        response = responses.get(candidate.candidate_id)
        require(candidate.stage == stage and list(candidate.parents) == parents and stage in by_stage
                and by_stage[stage]["prompt"] == prompt and response is not None and response["prompt"] == prompt,
                f"第 3 代已返回候选不符：{candidate.candidate_id}")
        apply_result(controller, candidate, response, results[candidate.candidate_id])
        orders.append(response["generation_order"])
    require(sorted(orders) == list(range(controller._generated + 1, controller._generated + len(orders) + 1)),
            "第 3 代恢复提案顺序不连续。")
    controller._generated += len(orders)
    controller.current_source_rows = rows
    try:
        controller.run_from_third(state["population"])
    except ReplayBoundary:
        pass
    else:
        raise ValueError("重放没有到达第 5 代断点。")
    for generation in (3, 4):
        original = [x for x in rows if x["event"] == "selection" and x.get("generation") == generation]
        replayed = [x for x in replay_log.items if x["event"] == "selection" and x.get("generation") == generation]
        require(len(original) == len(replayed) == 1 and original[0]["pool"] == replayed[0]["pool"]
                and original[0]["selected"] == replayed[0]["selected"], f"第 {generation} 代筛选重放不一致。")
        source_transitions = [(x["role"], x["round"], x["previous"], x["proposal"], x["current"]) for x in rows
                              if x["event"] == "branch_transition" and x.get("generation") == generation]
        replay_transitions = [(x["role"], x["round"], x["previous"], x["proposal"], x["current"]) for x in replay_log.items
                              if x["event"] == "branch_transition" and x.get("generation") == generation]
        require(source_transitions == replay_transitions, f"第 {generation} 代分支状态不一致。")
    require(len(controller.used_roles) == 14, "第 3、4 代角色响应不完整。")
    history = [x for x in rows if x["event"] == "history_appended" and x.get("generation") == 4]
    require(len(history) == 3 and all(controller.history[x["role"]][-1] == x["reflection"] for x in history),
            "第 4 代角色历史不一致。")
    fifth_responses = {x["candidate_id"]: x for x in rows if x["event"] == "proposal_response" and x.get("generation") == 5}
    require(list(fifth_responses) == [f"c{i:04d}" for i in range(262, 294)]
            and sorted(x["generation_order"] for x in fifth_responses.values())
            == list(range(controller._generated + 1, controller._generated + 33)), "第 5 代已返回提案不符。")
    fifth = {x.candidate_id: x for x in controller.fifth_candidates}
    for ident, response in fifth_responses.items():
        apply_result(controller, fifth[ident], response, results[ident])
    controller._generated += len(fifth_responses)
    fifth["c0294"].error = "模型连接失败；保留失败请求，不自动重试。"
    pending = controller.fifth_candidates[33:]
    require(len(pending) == 7 and [x.candidate_id for x in pending] == [f"c{i:04d}" for i in range(295, 302)],
            "第 5 代未启动槽位不符。")
    cpu = [x["accounting"]["cpu_seconds"] for x in rows if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == 123, "评价 CPU 记录不完整。")
    controller.backend, controller.log = backend, log
    state.update(population=controller.fourth_population, pending=pending, pending_specs=controller.fifth_specs[33:],
                 cpu_samples=state["cpu_samples"] + cpu, prior_hashes=preflight[0]["code_sha256"],
                 segments=segments, latest=directory, model_calls=322, evaluation_starts=288)
    return controller, state


def run_pending(controller, state):
    require(len(state["pending"]) == len(state["pending_specs"]) == 7, "未启动槽位数量不符。")
    controller._generation_offset = 0
    controller.generation = 5
    entries = []
    for candidate, (stage, parents, prompt) in zip(state["pending"], state["pending_specs"]):
        require(candidate.stage == stage and list(candidate.parents) == parents, "待执行槽位不符。")
        controller.event("reused_candidate_registration", generation=5, candidate_id=candidate.candidate_id,
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


def lineage():
    paths, seen, current = set(), set(), SOURCE.resolve()
    while True:
        require(current not in seen and current.is_relative_to((ROOT / "outputs/p1_control").resolve()), "断点链成环或越界。")
        seen.add(current)
        paths.add(current)
        rows = read_jsonl(current / "events.jsonl")
        starts = [x for x in rows if x["event"] == "formal_run_started"]
        require(len(starts) == 1, "断点启动记录不唯一。")
        if not starts[0].get("resume_of"):
            return paths
        parent = Path(starts[0]["resume_of"]).resolve()
        require(starts[0].get("interrupted_events_sha256") == digest(parent / "events.jsonl"), "父断点哈希不符。")
        current = parent


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
        for path in (ROOT / "outputs/p1_control").glob("*/config.json"):
            if path.parent.resolve() == run_dir.resolve():
                continue
            other = json.loads(path.read_text(encoding="utf-8"))
            if (other.get("matrix_contract_sha256") == config["matrix_contract_sha256"]
                    and matrix.run_identity(other) == matrix.run_identity(config)):
                event_path = path.parent / "events.jsonl"
                if event_path.is_file() and any(x["event"] == "formal_run_started" for x in read_jsonl(event_path)):
                    started.add(path.parent.resolve())
        require(started == lineage(), "同一研发存在未知已启动运行。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=3, selection_seed=task["selection_seed"], resume_of=str(SOURCE),
                  interrupted_events_sha256=digest(SOURCE / "events.jsonl"), total_model_call_cap=420,
                  frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4,
                  note="保留请求失败和中断评价，仅执行第 5 代未启动槽位。")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def simulate():
    backend, log = FixtureBackend(), MemoryLog()
    backend.cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    controller, state = reconstruct(backend, log)
    with recovery_guard(state, RECOVERY / "unused_proposed3_generation5_fixture", MemoryLog()):
        code = ("from pathlib import Path\nfrom scripts import p1_formal_matrix as m\ntry:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED", "研发互斥锁不一致。")
    run_pending(controller, state)
    require(len(backend.calls) == len(backend.evaluations) == 7
            and all(stage != "eoh_M2_03" for stage, _ in backend.calls), "未启动槽位模拟不符。")
    selected = controller.run_from_fifth(state["population"])
    require(len(selected) == 10 and controller.generation == 6
            and state["model_calls"] + len(backend.calls) <= 420
            and state["evaluation_starts"] + len(backend.evaluations) <= 378
            and [x["generation"] for x in log.items if x["event"] == "selection"] == [5, 6],
            "后续代次或预算模拟不符。")
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
            "限流后启动了其他模型请求。")
    print(json.dumps({"status": "passed", "prior_model_calls": state["model_calls"],
                      "prior_evaluations": state["evaluation_starts"], "unstarted_eoh": len(state["pending"]),
                      "remaining_model_calls": len(backend.calls), "remaining_evaluations": len(backend.evaluations),
                      "formal_mutex_shared": True, "model_concurrency": 1, "stop_after_429": True}, ensure_ascii=False))


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
    protected = [path for directory, _ in state["segments"] for path in (directory / "events.jsonl", directory / "config.json")]
    for path in [*(Path(x) for x in state["prior_hashes"]), *protected, PREVIOUS_SCRIPT, Path(__file__).resolve()]:
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
              generations=6, continuation=True, normal_model_slots_used=322, physical_model_cli_invocations_used=322,
              evaluation_starts_used=288, evaluation_cpu_seconds_used=budget.cpu_finished,
              frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4)
    print(f"本方法第 3 次研发第 5 代断点检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(state, run_dir, log):
            log.event("resume_checkpoint", interrupted_run=str(SOURCE), generation=5, failed_candidate_id="c0294",
                      interrupted_evaluation="c0293", unstarted_candidates=[x.candidate_id for x in state["pending"]],
                      model_calls_already_used=322, evaluation_starts_already_used=288, total_model_call_cap=420)
            run_pending(controller, state)
            selected = controller.run_from_fifth(state["population"])
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
        print(json.dumps({"status": "passed", "model_calls_used": state["model_calls"],
                          "evaluation_starts": state["evaluation_starts"], "unstarted_eoh": len(state["pending"])},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"本方法第 3 次研发第 5 代恢复检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
