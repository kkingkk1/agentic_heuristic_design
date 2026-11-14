"""重放 RoCo 末代已完成部分，保留第一轮失败并继续后续流程。"""

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import uuid
from unittest.mock import patch


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
SOURCE = ROOT / "outputs/p1_roco/20261113_223251_4e4a7f6f_resume"
PREVIOUS_SCRIPT = RECOVERY / "resume_roco3_generation5.py"
sys.path.insert(0, str(ROOT))

from baselines.roco.paper_controller import RoCoController
from scripts import p0_model_probe as probe, p1_formal_matrix as matrix
from scripts import p1_classified_runtime
from scripts.p0_controller import Controller, StopRun
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from tests.test_p0_controller import FixtureBackend, MemoryLog


spec = importlib.util.spec_from_file_location("roco3_generation5_recovery", PREVIOUS_SCRIPT)
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
require, read_events = previous.require, previous.read_events


class ReplayBoundary(BaseException):
    pass


def apply_source(controller, candidate, response, result, invalid):
    with patch.object(previous, "SOURCE", SOURCE):
        previous.apply_source(controller, candidate, response, result, invalid)


class ReplayController(previous.ReplayController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_final = False
        self.final_rows = []
        self.final_roles_used = set()
        self.fifth_population = None
        self.fifth_random_state = None
        self.sixth_eoh = None
        self.sixth_eoh_specs = None
        self.cached_roles = {}
        self.cached_batches = {}
        self.runtime_six_phase = None
        self.runtime_population = None

    def event(self, kind, **fields):
        if self.runtime_six_phase == 1 and kind == "initial_population":
            return Controller.event(self, "resume_population", source_generation=5, **fields)
        if self.runtime_six_phase == 2 and self.generation == 6:
            if kind == "elite_pair" and self.cached_roles:
                kind = "reused_elite_pair"
            elif kind == "branch_transition" and fields.get("round") == 0 and self.cached_roles:
                kind = "reused_branch_transition"
        return super().event(kind, **fields)

    def select(self, candidates, needed):
        selected = super().select(candidates, needed)
        if self.replay_final and self.generation == 5 and needed == self.n:
            self.fifth_population = selected
            self.fifth_random_state = self.random.getstate()
        return selected

    def request(self, stage, prompt, parser):
        if self.runtime_six_phase == 2 and self.generation == 6 and stage in self.cached_roles:
            row = self.cached_roles.pop(stage)
            require(row["prompt"] == prompt, f"末代历史角色提示不符：{stage}")
            self.event("reused_role_response", generation=6, stage=stage, source_events=str(SOURCE / "events.jsonl"))
            return parser(row["response"])
        if self.replay_final and self.live_fifth_phase == 2 and self.generation in (5, 6):
            rows = [x for x in self.final_rows if x["event"] == "model_response"
                    and x.get("generation") == self.generation and x["stage"] == stage]
            key = (self.generation, stage)
            require(len(rows) == 1 and key not in self.final_roles_used and rows[0]["prompt"] == prompt,
                    f"历史角色请求不符：{key}")
            self.final_roles_used.add(key)
            self.event("reused_role_response", generation=self.generation, stage=stage,
                       source_events=str(SOURCE / "events.jsonl"))
            return parser(rows[0]["response"])
        return super().request(stage, prompt, parser)

    def batch(self, specs):
        if self.runtime_six_phase is not None:
            if self.runtime_six_phase == 0:
                require(len(specs) == self.initial_count and all(stage.startswith("init_") and not parents
                        for stage, parents, _ in specs), "续跑种群位置不符。")
                self.runtime_six_phase = 1
                return self.runtime_population
            if self.runtime_six_phase == 1:
                actual = [(stage, [x.candidate_id for x in parents], prompt) for stage, parents, prompt in specs]
                require(self.generation == 6 and actual == self.sixth_eoh_specs, "末代 EoH 父代或提示不符。")
                self.runtime_six_phase = 2
                self.event("reused_eoh_proposals", generation=6, source_events=str(SOURCE / "events.jsonl"))
                return self.sixth_eoh
            key = tuple(stage for stage, _, _ in specs)
            if self.generation == 6 and key in self.cached_batches:
                expected, candidates = self.cached_batches.pop(key)
                actual = [(stage, [p.candidate_id for p in parents], prompt) for stage, parents, prompt in specs]
                require(actual == expected, f"末代协作提案提示或父代不符：{key}")
                for candidate in candidates:
                    self.event("reused_candidate_registration", generation=6, candidate_id=candidate.candidate_id,
                               stage=candidate.stage, parents=list(candidate.parents), source_events=str(SOURCE / "events.jsonl"))
                return candidates
            return Controller.batch(self, specs)
        if not self.replay_final or self.live_fifth_phase in (0, 1):
            return super().batch(specs)
        require(self.generation in (5, 6), "重放到未知代次。")
        entries = [(self.register(stage, tuple(p.candidate_id for p in parents)), prompt)
                   for stage, parents, prompt in specs]
        registrations = {x["candidate_id"]: x for x in self.final_rows
                         if x["event"] == "candidate_registered" and x.get("generation") == self.generation}
        for candidate, _ in entries:
            row = registrations.get(candidate.candidate_id)
            require(row is not None and row["stage"] == candidate.stage and row["parents"] == list(candidate.parents),
                    f"历史候选注册不符：{candidate.candidate_id}")
        responses = {x["candidate_id"]: x for x in self.final_rows if x["event"] == "proposal_response"}
        results = {x["candidate_id"]: x["result"] for x in self.final_rows if x["event"] == "evaluation"}
        invalid = {x["candidate_id"]: x["error"] for x in self.final_rows if x["event"] == "candidate_invalid"}
        selection = next(x for x in self.final_rows if x["event"] == "selection" and x.get("generation") == 5)
        final_attempts = {x["stage"]: x for x in self.final_rows if x["event"] == "model_attempt"
                          and x["event_index"] > selection["event_index"]}
        orders = []
        for candidate, prompt in entries:
            response = responses.get(candidate.candidate_id)
            if self.generation == 6:
                require(candidate.stage in final_attempts and final_attempts[candidate.stage]["prompt"] == prompt,
                        f"末代原始提示不符：{candidate.candidate_id}")
            if candidate.candidate_id == "c0362":
                require(self.generation == 6 and candidate.stage == "round_1_exploiter" and response is None,
                        "失败分支身份不符。")
                candidate.error = "模型连接失败；保留失败请求，不自动重试。"
                continue
            require(response is not None and response["prompt"] == prompt, f"历史提案不符：{candidate.candidate_id}")
            apply_source(self, candidate, response, results.get(candidate.candidate_id), invalid.get(candidate.candidate_id))
            orders.append(response["generation_order"])
        require(sorted(orders) == list(range(self._generated + 1, self._generated + len(orders) + 1)),
                "生成顺序不连续。")
        self._generated += len(orders)
        candidates = [candidate for candidate, _ in entries]
        item_specs = [(candidate.stage, list(candidate.parents), prompt) for candidate, prompt in entries]
        if self.generation == 6:
            key = tuple(x.stage for x in candidates)
            if len(candidates) == 40:
                require(self.fifth_population is not None and self.sixth_eoh is None, "末代种群不完整。")
                self.sixth_eoh, self.sixth_eoh_specs = candidates, item_specs
            else:
                self.cached_batches[key] = (item_specs, candidates)
            if key == ("round_1_explorer", "round_1_exploiter"):
                require([c.candidate_id for c in candidates] == ["c0361", "c0362"]
                        and results["c0361"]["status"] == "infrastructure_error"
                        and results["c0361"].get("error") == "其他并发任务停止运行。", "末代失败记录不符。")
                raise ReplayBoundary()
        return candidates

    def run_from_sixth(self, population):
        require(len(population) == 10 and self.fifth_random_state is not None, "第 5 代种群不完整。")
        self.replay_final = False
        self.live_fifth_phase = None
        self.runtime_six_phase = 0
        self.runtime_population = population
        self._generation_offset = 5
        self.generations = 1
        self.random.setstate(self.fifth_random_state)
        return RoCoController.run(self)


class RecoveryRuntime(previous.RecoveryRuntime):
    def before_generation(self):
        if self.resume_first_generation:
            self.resume_first_generation = False
            self.log.event("reused_generation_budget_check", generation=6, source_events=str(SOURCE / "events.jsonl"))
            return
        return super().before_generation()


def reconstruct(backend, log):
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))

    class OfflineBackend:
        cfg = config

        def before_generation(self):
            pass

    segments = previous.previous.find_segments(config)
    indexes = [i for i, (directory, _) in enumerate(segments) if directory.resolve() == previous.SOURCE.resolve()]
    source_indexes = [i for i, (directory, _) in enumerate(segments) if directory.resolve() == SOURCE.resolve()]
    require(len(indexes) == len(source_indexes) == 1 and source_indexes[0] == indexes[0] + 1
            and source_indexes[0] == len(segments) - 1, "无法唯一定位末代断点。")
    replay_log = MemoryLog()
    with patch.object(previous.previous, "find_segments", return_value=segments[:indexes[0] + 1]), \
            patch.object(previous, "ReplayController", ReplayController):
        controller, state = previous.reconstruct(OfflineBackend(), replay_log)
    directory, rows = segments[source_indexes[0]]
    require((directory / "config.json").read_bytes() == (previous.SOURCE / "config.json").read_bytes()
            and [x["event_index"] for x in rows] == list(range(1, len(rows) + 1)), "断点配置或事件编号不符。")
    started = [x for x in rows if x["event"] == "formal_run_started"]
    preflight = [x for x in rows if x["event"] == "preflight"]
    require(len(started) == len(preflight) == 1 and started[0]["resume_of"] == str(state["latest"])
            and started[0]["interrupted_events_sha256"] == digest(state["latest"] / "events.jsonl")
            and started[0]["total_model_call_cap"] == 420 and started[0]["effective_model_concurrency"] == 1
            and preflight[0]["physical_model_cli_invocations_used"] == state["model_calls"] == 295
            and preflight[0]["evaluation_starts_used"] == state["evaluation_starts"] == 256
            and rows[-1]["event"] == "preflight_or_run_stopped", "断点链、预算或停止状态不符。")
    for name, expected in preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
    attempts = [x for x in rows if x["event"] == "model_attempt"]
    failures = [x for x in rows if x["event"] == "model_failure_classified"]
    responses = {x["candidate_id"]: x for x in rows if x["event"] == "proposal_response"}
    results = {x["candidate_id"]: x["result"] for x in rows if x["event"] == "evaluation"}
    invalid = {x["candidate_id"]: x["error"] for x in rows if x["event"] == "candidate_invalid"}
    eval_ids = [x["candidate_id"] for x in rows if x["event"] == "evaluation_budget_reserved"]
    require(len(attempts) == 104 and len(responses) == 94 and len(results) == len(eval_ids) == 93
            and len(set(eval_ids)) == 93 and set(eval_ids) == set(results)
            and set(responses) == set(results) | set(invalid) and not set(results) & set(invalid)
            and len(invalid) == 1 and len(failures) == 1 and failures[0]["stage"] == "round_1_exploiter"
            and failures[0]["category"] == "network", "请求、评价或失败状态不符。")
    process = [x for x in rows if x["event"] == "model_process" and x["stage"] == "round_1_exploiter"]
    require(process and "Connection failed" in process[-1]["result"].get("stdout", ""), "模型连接失败证据不符。")
    registrations = [x["candidate_id"] for x in rows if x["event"] == "reused_candidate_registration"]
    require(registrations == [x.candidate_id for x in state["pending"]], "第 5 代待恢复候选不符。")
    by_stage = {x["stage"]: x for x in attempts[:len(state["pending"])]}
    orders = []
    for candidate, (stage, parents, prompt) in zip(state["pending"], state["pending_specs"]):
        response = responses.get(candidate.candidate_id)
        require(candidate.stage == stage and list(candidate.parents) == parents and stage in by_stage
                and by_stage[stage]["prompt"] == prompt and response is not None and response["prompt"] == prompt,
                f"第 5 代已返回候选不符：{candidate.candidate_id}")
        apply_source(controller, candidate, response, results.get(candidate.candidate_id), invalid.get(candidate.candidate_id))
        orders.append(response["generation_order"])
    require(sorted(orders) == list(range(controller._generated + 1, controller._generated + len(orders) + 1)),
            "第 5 代恢复提案顺序不连续。")
    controller._generated += len(orders)
    controller.final_rows = rows
    controller.replay_final = True
    replay_start = len(replay_log.items)
    try:
        controller.run_from_fifth(state["population"])
    except ReplayBoundary:
        pass
    else:
        raise ValueError("重放没有到达末代第一轮断点。")
    replayed = replay_log.items[replay_start:]
    original_selection = [x for x in rows if x["event"] == "selection" and x.get("generation") == 5]
    replay_selection = [x for x in replayed if x["event"] == "selection" and x.get("generation") == 5]
    require(len(original_selection) == len(replay_selection) == 1
            and original_selection[0]["pool"] == replay_selection[0]["pool"]
            and original_selection[0]["selected"] == replay_selection[0]["selected"], "第 5 代筛选重放不一致。")
    for generation, kinds in ((5, ("elite_pair", "branch_transition", "history_appended", "slot_skipped")),
                              (6, ("elite_pair", "branch_transition"))):
        for kind in kinds:
            clean = lambda x: {k: v for k, v in x.items() if k not in ("event_index", "at_utc")}
            left = [clean(x) for x in rows if x["event"] == kind and x.get("generation") == generation]
            right = [clean(x) for x in replayed if x["event"] == kind and x.get("generation") == generation]
            require(left == right, f"第 {generation} 代 {kind} 重放不一致。")
    require(len(controller.final_roles_used) == 9 and controller.sixth_eoh is not None,
            "第 5 代与末代历史响应不完整。")
    controller.cached_roles = {x["stage"]: x for x in rows if x["event"] == "model_response" and x.get("generation") == 6}
    require(set(controller.cached_roles) == {"critic_initial", "critic_1"}
            and set(controller.cached_batches) == {("start_explorer", "start_exploiter"),
                                                    ("round_1_explorer", "round_1_exploiter")},
            "末代协作断点不完整。")
    cpu = [x["accounting"]["cpu_seconds"] for x in rows if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == len(eval_ids), "评价 CPU 记录不完整。")
    controller.backend, controller.log = backend, log
    state.update(population=controller.fifth_population, cpu_samples=state["cpu_samples"] + cpu,
                 prior_hashes=preflight[0]["code_sha256"], segments=segments, latest=directory,
                 model_calls=399, evaluation_starts=349)
    return controller, state


class GuardLog:
    def __init__(self, log):
        self.log = log

    def event(self, kind, **fields):
        if kind == "formal_run_started":
            fields["note"] = "保留末代第一轮失败，沿用两支上一有效状态并跳过本轮融合。"
        return self.log.event(kind, **fields)


def simulate():
    backend, log = FixtureBackend(), MemoryLog()
    backend.cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    controller, state = reconstruct(backend, log)
    with previous.previous.recovery_guard(state, RECOVERY / "unused_roco3_gen6_round1_fixture", GuardLog(MemoryLog())):
        code = ("from pathlib import Path\nfrom scripts import p1_formal_matrix as m\ntry:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED", "研发互斥锁不一致。")
    selected = controller.run_from_sixth(state["population"])
    transitions = [x for x in log.items if x["event"] == "branch_transition" and x.get("generation") == 6
                   and x.get("round") == 1]
    require(len(transitions) == 2 and [(x["proposal"], x["current"]) for x in transitions]
            == [("c0361", "c0359"), ("c0362", "c0360")], "无效分支没有沿用上一有效状态。")
    skipped = [x for x in log.items if x["event"] == "slot_skipped" and x.get("generation") == 6]
    require(any(x["stage"] == "fusion_1" for x in skipped) and len(selected) == 10 and controller.generation == 6
            and state["model_calls"] + len(backend.calls) <= 420
            and state["evaluation_starts"] + len(backend.evaluations) <= 378
            and [x["generation"] for x in log.items if x["event"] == "selection"] == [6],
            "跳过融合、后续代次或预算模拟不符。")
    limited_log = MemoryLog()
    runtime = RecoveryRuntime(SOURCE, SOURCE / "config.json", limited_log, cli=["offline-fixture"])
    limited_controller, limited_state = reconstruct(runtime, limited_log)
    with patch.object(probe, "run_command", return_value={"exit_code": 1, "stdout": "", "stderr": ""}), \
            patch.object(probe, "extract_message", side_effect=probe.ProbeError("offline 429")), \
            patch.object(p1_classified_runtime, "classify_failure",
                         return_value={"category": "quota", "source": "fixture", "plugin_warnings": []}):
        try:
            limited_controller.run_from_sixth(limited_state["population"])
        except StopRun:
            pass
        else:
            raise ValueError("限流夹具没有停止。")
    attempts = [x for x in limited_log.items if x["event"] == "model_attempt"]
    require(len(attempts) == 1 and attempts[0]["stage"] == "critic_2", "限流后启动了其他模型请求。")
    print(json.dumps({"status": "passed", "prior_model_calls": state["model_calls"],
                      "prior_evaluations": state["evaluation_starts"], "remaining_model_calls": len(backend.calls),
                      "remaining_evaluations": len(backend.evaluations), "failed_candidates": ["c0361", "c0362"],
                      "skipped_fusion": "fusion_1", "formal_mutex_shared": True,
                      "model_concurrency": 1, "stop_after_429": True}, ensure_ascii=False))


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
    run_dir = ROOT / "outputs/p1_roco" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8] + "_resume")
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
              generations=6, continuation=True, normal_model_slots_used=399, physical_model_cli_invocations_used=399,
              evaluation_starts_used=349, evaluation_cpu_seconds_used=budget.cpu_finished,
              frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4)
    print(f"RoCo 第 3 次研发末代协作断点检查通过；记录：{run_dir}", flush=True)
    try:
        with previous.previous.recovery_guard(state, run_dir, GuardLog(log)):
            log.event("resume_checkpoint", interrupted_run=str(SOURCE), generation=6,
                      unavailable_candidates=["c0361", "c0362"], skipped_fusion="fusion_1",
                      model_calls_already_used=399, evaluation_starts_already_used=349, total_model_call_cap=420)
            selected = controller.run_from_sixth(state["population"])
            require(len(selected) == 10 and controller.generation == 6, "六代未完成。")
        print(f"RoCo 第 3 次研发六代完成；记录：{run_dir}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"RoCo 第 3 次研发续跑停止：{exc}；记录：{run_dir}", file=sys.stderr, flush=True)
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
                          "evaluation_starts": state["evaluation_starts"], "unavailable_candidates": ["c0361", "c0362"]},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"RoCo 第 3 次研发末代协作恢复检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
