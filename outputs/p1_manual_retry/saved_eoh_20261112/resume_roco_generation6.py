"""重放 RoCo 第 5 代筛选并续跑第 6 代未启动的 EoH 槽位。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
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
SOURCE = ROOT / "outputs/p1_roco/20261113_133039_0866a3bf_resume"
sys.path.insert(0, str(ROOT))

from baselines.roco.paper_controller import RoCoController
from scripts import p0_model_probe as probe, p1_formal_matrix as matrix
from scripts.p0_controller import StopRun
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from tests.test_p0_controller import FixtureBackend, MemoryLog


spec = importlib.util.spec_from_file_location("roco_generation5_mutation", RECOVERY / "resume_roco_generation5_mutation.py")
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
chain = previous.chain
require, read_events = chain.require, chain.read_events
verify_segment = previous.verify_segment


class ReplayBoundary(BaseException):
    """在第 6 代实际执行前退出重放。"""


class ReplayController(previous.ReplayController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_to_sixth = False
        self.sixth_running = False
        self.sixth_source_rows = []
        self.fifth_population = None
        self.fifth_random_state = None
        self.sixth_candidates = None
        self.sixth_specs = None
        self.sixth_source = None

    def event(self, kind, **fields):
        if self.sixth_running and kind == "initial_population" and self._future_initial_event:
            self._future_initial_event = False
            return RoCoController.event(self, "resume_population", source_generation=5, **fields)
        return super().event(kind, **fields)

    def select(self, candidates, needed):
        selected = super().select(candidates, needed)
        if self.replay_to_sixth and self.generation == 5 and needed == self.n:
            self.fifth_population = selected
            self.fifth_random_state = self.random.getstate()
        return selected

    def batch(self, specs):
        if self.sixth_running and self._future_population is None and self._replay_eoh is not None:
            require(len(specs) == len(self._replay_specs) == 40, "第 6 代 EoH 槽位数不符。")
            for actual, expected in zip(specs, self._replay_specs):
                stage, parents, prompt = actual
                require((stage, [x.candidate_id for x in parents], prompt) == expected,
                        f"第 6 代 EoH 父代或提示变化：{stage}")
            candidates, self._replay_eoh = self._replay_eoh, None
            self.event("reused_eoh_proposals", generation=6,
                       source_events=str(self.sixth_source / "events.jsonl"))
            return candidates
        if (not self.replay_to_sixth or self.generation != 6
                or self._future_population is not None or self._replay_eoh is not None):
            return super().batch(specs)
        entries = [(self.register(stage, tuple(p.candidate_id for p in parents)), prompt)
                   for stage, parents, prompt in specs]
        registrations = {x["candidate_id"]: x for x in self.sixth_source_rows
                         if x["event"] == "candidate_registered" and x.get("generation") == 6}
        for candidate, _ in entries:
            row = registrations.get(candidate.candidate_id)
            require(row is not None and row["stage"] == candidate.stage
                    and row["parents"] == list(candidate.parents), f"第 6 代候选注册不符：{candidate.candidate_id}")
        require(len(entries) == len(registrations) == 40 and self.fifth_population is not None,
                "第 6 代候选或第 5 代种群不完整。")
        self.sixth_candidates = [c for c, _ in entries]
        self.sixth_specs = [(c.stage, list(c.parents), prompt) for c, prompt in entries]
        raise ReplayBoundary()

    def run_from_sixth(self, population):
        require(len(population) == 10, "第 5 代种群不完整。")
        self._generation_offset = 5
        self._future_population = population
        self.generations = 1
        return RoCoController.run(self)


class ResumeRuntime(previous.previous.ResumeRuntime):
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

    segments = chain.find_segments(config)
    source_indexes = [i for i, (directory, _) in enumerate(segments) if directory.resolve() == SOURCE.resolve()]
    require(len(source_indexes) == 1 and source_indexes[0] > 0, "无法定位第 6 代断点。")
    index = source_indexes[0]
    with patch.object(chain, "find_segments", return_value=segments[:index]), \
            patch.object(previous, "ReplayController", ReplayController):
        controller, state = previous.reconstruct(OfflineBackend(), log)
    directory, source = segments[index]
    hashes = verify_segment(directory, source, state["latest"], state["model_calls"], state["evaluation_starts"])
    require(source[-1]["event"] == "preflight_or_run_stopped", "第 6 代原片段尚未停止。")
    controller.sixth_source_rows = source
    responses = {x["candidate_id"]: x for x in source if x["event"] == "proposal_response"
                 and x.get("generation") == 5}
    results = {x["candidate_id"]: x["result"] for x in source if x["event"] == "evaluation"}
    orders = []
    for candidate in state["pending"]:
        row, result = responses.get(candidate.candidate_id), results.get(candidate.candidate_id)
        require(row is not None and result is not None, f"第 5 代变异评价缺失：{candidate.candidate_id}")
        chain.base.apply_result(controller, candidate, row, result, SOURCE / "candidates" / candidate.candidate_id)
        orders.append(row["generation_order"])
    require(sorted(orders) == list(range(controller._generated + 1, controller._generated + len(orders) + 1)),
            "第 5 代变异生成顺序不连续。")
    controller._generated += len(orders)
    controller.replay_to_sixth = True
    replay_start = len(log.items)
    try:
        controller.run_from_fifth(state["population"])
    except ReplayBoundary:
        pass
    else:
        raise ValueError("重放没有到达第 6 代断点。")
    original_selections = [x for x in source if x["event"] == "selection" and x.get("generation") == 5]
    replay_selections = [x for x in log.items[replay_start:] if x["event"] == "selection" and x.get("generation") == 5]
    require(len(original_selections) == len(replay_selections) == 1
            and original_selections[0]["pool"] == replay_selections[0]["pool"]
            and original_selections[0]["selected"] == replay_selections[0]["selected"], "第 5 代筛选重放不一致。")
    for event in ("history_appended", "reused_role_response", "reused_mutation_proposals"):
        original = [{k: v for k, v in x.items() if k not in ("event_index", "at_utc")}
                    for x in source if x["event"] == event and x.get("generation") == 5]
        replayed = [x for x in log.items[replay_start:] if x["event"] == event and x.get("generation") == 5]
        require(original == replayed, f"第 5 代 {event} 重放不一致。")

    first = next(i for i, x in enumerate(source)
                 if x["event"] == "candidate_registered" and x.get("generation") == 6)
    sixth_rows = source[first:]
    attempts = [x for x in sixth_rows if x["event"] == "model_attempt"]
    failures = [x for x in sixth_rows if x["event"] == "model_failure_classified"]
    require([x["stage"] for x in attempts] == ["eoh_E1_01", "eoh_E1_02"]
            and len(failures) == 1 and failures[0]["stage"] == "eoh_E1_02"
            and failures[0]["category"] == "network", "第 6 代失败位置不符。")
    by_stage = {c.stage: c for c in controller.sixth_candidates}
    spec_by_id = {c.candidate_id: spec for c, spec in zip(controller.sixth_candidates, controller.sixth_specs)}
    for attempt in attempts:
        candidate = by_stage[attempt["stage"]]
        require(attempt["prompt"] == spec_by_id[candidate.candidate_id][2], "第 6 代原提示不符。")
    sixth_responses = [x for x in sixth_rows if x["event"] == "proposal_response"]
    require(len(sixth_responses) == 1 and sixth_responses[0]["candidate_id"] == by_stage["eoh_E1_01"].candidate_id,
            "第 6 代成功响应不符。")
    candidate = by_stage["eoh_E1_01"]
    result = results[candidate.candidate_id]
    require(result["status"] == "infrastructure_error", "被中断评价的状态不符。")
    chain.base.apply_result(controller, candidate, sixth_responses[0], result, SOURCE / "candidates" / candidate.candidate_id)
    require(sixth_responses[0]["generation_order"] == controller._generated + 1, "生成顺序不符。")
    controller._generated += 1
    by_stage["eoh_E1_02"].error = "模型请求连接失败；保留失败，不自动重试。"
    attempted_ids = {by_stage[x["stage"]].candidate_id for x in attempts}
    pending = chain.unstarted_candidates(controller.sixth_candidates, attempted_ids, sixth_rows)
    require(len(pending) == 38, "第 6 代未启动槽位数不符。")
    model_calls = state["model_calls"] + len([x for x in source if x["event"] == "model_attempt"])
    eval_starts = state["evaluation_starts"] + len([x for x in source if x["event"] == "evaluation_budget_reserved"])
    cpu_samples = state["cpu_samples"] + [x["accounting"]["cpu_seconds"] for x in source
                                         if x["event"] == "evaluation_cpu_final"]
    require(model_calls == 357 and eval_starts == len(cpu_samples) == 304, "原断点预算不符。")
    state["prior_hashes"].update(hashes)
    state["protected"].extend((directory / "events.jsonl", directory / "config.json"))
    prior = directory

    for directory, rows in segments[index + 1:]:
        hashes = verify_segment(directory, rows, prior, model_calls, eval_starts)
        require(rows[-1]["event"] == "preflight_or_run_stopped", "后续片段尚未停止。")
        registrations = [x["candidate_id"] for x in rows if x["event"] == "reused_candidate_registration"]
        require(registrations == [x.candidate_id for x in pending], "后续待执行槽位不符。")
        pending_by_stage = {x.stage: x for x in pending}
        attempts = [x for x in rows if x["event"] == "model_attempt"]
        require(len({x["stage"] for x in attempts}) == len(attempts)
                and all(x["stage"] in pending_by_stage for x in attempts), "后续片段含重复或非 EoH 调用。")
        require(len([x for x in rows if x["event"] == "model_budget_reserved"]) == len(attempts),
                "后续片段模型预算预留不符。")
        for attempt in attempts:
            candidate = pending_by_stage[attempt["stage"]]
            require(attempt["prompt"] == spec_by_id[candidate.candidate_id][2], "后续片段原提示不符。")
        responses = {x["candidate_id"]: x for x in rows if x["event"] == "proposal_response"}
        results = {x["candidate_id"]: x["result"] for x in rows if x["event"] == "evaluation"}
        eval_ids = [x["candidate_id"] for x in rows if x["event"] == "evaluation_budget_reserved"]
        require(len(eval_ids) == len(set(eval_ids)) and set(eval_ids) == set(results), "后续评价不完整或重复。")
        saved_results = {}
        saved_count = 0
        for saved_dir, saved_rows, saved_hashes in chain.find_saved(directory):
            state["prior_hashes"].update(saved_hashes)
            items = {x["candidate_id"]: x["result"] for x in saved_rows if x["event"] == "evaluation"}
            saved_ids = [x["candidate_id"] for x in saved_rows if x["event"] == "evaluation_budget_reserved"]
            require(len(saved_ids) == len(set(saved_ids)) and set(saved_ids) == set(items)
                    and not set(items) & set(saved_results), "保存响应的评价不完整或重复。")
            for ident, result in items.items():
                saved_results[ident] = (result, saved_dir / "candidates" / ident)
            saved_count += len(saved_ids)
            cpu_samples.extend(x["accounting"]["cpu_seconds"] for x in saved_rows if x["event"] == "evaluation_cpu_final")
            state["protected"].extend((saved_dir / "evaluation_events.jsonl", saved_dir / "summary.json", saved_dir / "config.json"))
        invalid = {x["candidate_id"]: x["error"] for x in rows if x["event"] == "candidate_invalid"}
        require(not set(results) & set(saved_results) and not (set(results) | set(saved_results)) & set(invalid)
                and set(responses) == set(results) | set(saved_results) | set(invalid), "响应与首次评价不一致。")
        orders = sorted(x["generation_order"] for x in responses.values())
        require(orders == list(range(controller._generated + 1, controller._generated + len(orders) + 1)),
                "后续片段生成顺序不连续。")
        controller._generated += len(orders)
        candidates = {x.candidate_id: x for x in pending}
        for ident, response in responses.items():
            require(ident in candidates and response["generation"] == 6, "后续响应候选不符。")
            if ident in invalid:
                candidates[ident].generation_order = response["generation_order"]
                candidates[ident].error = invalid[ident]
            else:
                result, candidate_dir = ((results[ident], directory / "candidates" / ident)
                                         if ident in results else saved_results[ident])
                chain.base.apply_result(controller, candidates[ident], response, result, candidate_dir)
        failures = [x for x in rows if x["event"] == "model_failure_classified"]
        for failure in failures:
            require(failure["stage"] in pending_by_stage, "后续失败槽位不符。")
            pending_by_stage[failure["stage"]].error = f"模型请求失败（{failure['category']}）；不自动重试。"
        attempted_ids = {pending_by_stage[x["stage"]].candidate_id for x in attempts}
        require(attempted_ids == set(responses) | {pending_by_stage[x["stage"]].candidate_id for x in failures},
                "后续调用没有对应响应或失败。")
        pending = chain.unstarted_candidates(pending, attempted_ids, rows)
        cpu = [x["accounting"]["cpu_seconds"] for x in rows if x["event"] == "evaluation_cpu_final"]
        require(len(cpu) == len(eval_ids), "后续评价 CPU 记录不完整。")
        cpu_samples.extend(cpu)
        model_calls += len(attempts)
        eval_starts += len(eval_ids) + saved_count
        require(model_calls <= 420 and eval_starts <= 378 and len(cpu_samples) == eval_starts, "累计预算不符。")
        state["prior_hashes"].update(hashes)
        state["protected"].extend((directory / "events.jsonl", directory / "config.json"))
        prior = directory

    require(pending and controller.fifth_random_state is not None, "第 6 代没有可恢复槽位或随机状态。")
    controller.random.setstate(controller.fifth_random_state)
    controller.replay_to_sixth = False
    controller.sixth_running = True
    controller.backend = backend
    controller.sixth_source = prior
    controller._replay_eoh = controller.sixth_candidates
    controller._replay_specs = controller.sixth_specs
    state.update(population=controller.fifth_population, pending=pending,
                 pending_specs=[spec_by_id[c.candidate_id] for c in pending],
                 model_calls=model_calls, evaluation_starts=eval_starts, cpu_samples=cpu_samples,
                 segments=segments, latest=prior)
    return controller, state


def run_pending(controller, state):
    controller._generation_offset = 0
    controller.generation = 6
    entries = []
    for candidate, (stage, parents, prompt) in zip(state["pending"], state["pending_specs"]):
        require(candidate.stage == stage and list(candidate.parents) == parents, "待执行候选或父代不符。")
        controller.event("reused_candidate_registration", generation=6, candidate_id=candidate.candidate_id,
                         stage=stage, parents=parents, source_events=str(state["latest"] / "events.jsonl"))
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


def simulate():
    backend, log = FixtureBackend(), MemoryLog()
    backend.cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    controller, state = reconstruct(backend, log)
    with patch.object(chain, "reconstruct", side_effect=reconstruct), patch.object(chain, "run_pending", side_effect=run_pending):
        chain.verify_serial_runtime()
    runtime_log = MemoryLog()
    runtime = ResumeRuntime(SOURCE, SOURCE / "config.json", runtime_log, cli=["offline-fixture"])
    runtime.before_generation()
    require([x.get("generation") for x in runtime_log.items if x["event"] == "reused_generation_budget_check"] == [6],
            "恢复代预算检查不符。")
    with chain.recovery_guard(state, RECOVERY / "unused_roco_gen6_fixture", MemoryLog()):
        code = ("from pathlib import Path\nfrom scripts import p1_formal_matrix as m\ntry:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED", "续跑互斥锁不一致。")
    count = len(state["pending"])
    run_pending(controller, state)
    require(len(backend.calls) == count and len(backend.evaluations) == count, "未启动槽位重放不符。")
    require(not {"eoh_E1_01", "eoh_E1_02"} & {stage for stage, _ in backend.calls}
            and not {"eoh_E1_01", "eoh_E1_02"} & set(backend.evaluations), "失败或中断槽位被重复执行。")
    selected = controller.run_from_sixth(state["population"])
    selections = [x["generation"] for x in log.items if x["event"] == "selection"]
    require(len(selected) == 10 and controller.generation == 6 and selections == [2, 3, 4, 5, 6],
            "重放后的代数或筛选不符。")
    require(state["model_calls"] + len(backend.calls) <= 420
            and state["evaluation_starts"] + len(backend.evaluations) <= 378, "模拟超出预算。")
    print(json.dumps({"status": "passed", "prior_model_calls": state["model_calls"],
                      "prior_evaluations": state["evaluation_starts"], "unstarted_eoh": count,
                      "remaining_model_calls": len(backend.calls), "remaining_evaluations": len(backend.evaluations),
                      "formal_mutex_shared": True}, ensure_ascii=False))


def resume():
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    matrix.verify_training_software(matrix.load_matrix()[0])
    check_backend = FixtureBackend()
    check_backend.cfg = config
    reconstruct(check_backend, MemoryLog())
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
    runtime = ResumeRuntime(run_dir, run_dir / "config.json", log, cli=cli)
    chain.configure_model_concurrency(runtime)
    controller, state = reconstruct(runtime, MemoryLog())
    controller.log = log
    for path in [*(Path(x) for x in state["prior_hashes"]), *state["protected"], Path(__file__).resolve()]:
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
              generations=6, continuation=True, normal_model_slots_used=state["model_calls"],
              physical_model_cli_invocations_used=state["model_calls"], evaluation_starts_used=state["evaluation_starts"],
              evaluation_cpu_seconds_used=budget.cpu_finished, resume_basis="user_requested",
              frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4)
    print(f"RoCo 第 6 代断点检查通过；记录：{run_dir}", flush=True)
    try:
        with chain.recovery_guard(state, run_dir, log, resume_basis="user_requested"):
            log.event("resume_checkpoint", interrupted_run=str(state["latest"]), generation=6,
                      unstarted_candidates=[x.candidate_id for x in state["pending"]],
                      model_calls_already_used=state["model_calls"], evaluation_starts_already_used=state["evaluation_starts"],
                      total_model_call_cap=420)
            run_pending(controller, state)
            selected = controller.run_from_sixth(state["population"])
            require(len(selected) == 10 and controller.generation == 6, "六代研发未完成。")
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
    parser.add_argument("--user-requested-resume", action="store_true")
    args = parser.parse_args()
    if args.simulate:
        simulate()
        return 0
    if args.run:
        require(args.user_requested_resume, "正式续跑须记录用户指示。")
        return resume()
    backend, log = FixtureBackend(), MemoryLog()
    backend.cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    _, state = reconstruct(backend, log)
    print(json.dumps({"status": "passed", "model_calls_used": state["model_calls"],
                      "evaluation_starts": state["evaluation_starts"], "unstarted_eoh": len(state["pending"]),
                      "latest": str(state["latest"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
