"""复用 RoCo 第 5 代协作结果，继续未启动的精英变异并完成研发。"""

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
SOURCE = ROOT / "outputs/p1_roco/20261113_124748_ca7ee547_resume"
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe, p1_formal_matrix as matrix
from scripts.p0_controller import StopRun
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from tests.test_p0_controller import FixtureBackend, MemoryLog


spec = importlib.util.spec_from_file_location("roco_generation5_recovery", RECOVERY / "resume_roco_generation5.py")
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
chain = previous.chain
require, read_events = chain.require, chain.read_events
verify_segment = previous.previous.previous.verify_segment


class ReplayBoundary(BaseException):
    """在第 5 代精英变异执行前退出重放。"""


class ReplayController(previous.ReplayController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_collaboration = False
        self.resume_collaboration = False
        self.mutation_source = None
        self.mutation_source_rows = []
        self.role_responses = set()
        self.cached_batches = {}
        self.mutation_candidates = None
        self.mutation_specs = None

    def event(self, kind, **fields):
        if (self.resume_collaboration and self.generation == 5
                and kind in ("elite_pair", "branch_transition", "slot_skipped")):
            return
        return super().event(kind, **fields)

    def request(self, stage, prompt, parser):
        if not (self.replay_collaboration or self.resume_collaboration) or self.generation != 5:
            return super().request(stage, prompt, parser)
        rows = [x for x in self.mutation_source_rows if x["event"] == "model_response"
                and x.get("generation") == 5 and x.get("stage") == stage]
        require(len(rows) == 1 and rows[0]["prompt"] == prompt, f"第 5 代角色请求不符：{stage}")
        if self.replay_collaboration:
            require(stage not in self.role_responses, f"角色响应重复：{stage}")
            self.role_responses.add(stage)
            self.event("model_response", stage=stage, generation=5, prompt=prompt, response=rows[0]["response"])
        else:
            self.event("reused_role_response", generation=5, stage=stage,
                       source_events=str(SOURCE / "events.jsonl"))
        return parser(rows[0]["response"])

    def batch(self, specs):
        if self._future_population is not None or self._replay_eoh is not None:
            return super().batch(specs)
        if not (self.replay_collaboration or self.resume_collaboration) or self.generation != 5:
            return super().batch(specs)
        key = tuple(stage for stage, _, _ in specs)
        actual_specs = [(stage, [p.candidate_id for p in parents], prompt) for stage, parents, prompt in specs]
        mutation = bool(key) and all(stage.startswith("mutation_") for stage in key)
        if self.resume_collaboration:
            if mutation:
                require(actual_specs == self.mutation_specs, "第 5 代变异父代或提示变化。")
                self.event("reused_mutation_proposals", generation=5,
                           source_events=str(self.mutation_source / "events.jsonl"))
                return self.mutation_candidates
            cached = self.cached_batches.get(key)
            require(cached is not None and actual_specs == cached[0], "第 5 代协作父代或提示变化。")
            return cached[1]

        entries = [(self.register(stage, tuple(p.candidate_id for p in parents)), prompt)
                   for stage, parents, prompt in specs]
        registrations = {x["candidate_id"]: x for x in self.mutation_source_rows
                         if x["event"] == "candidate_registered" and x.get("generation") == 5}
        for candidate, _ in entries:
            row = registrations.get(candidate.candidate_id)
            require(row is not None and row["stage"] == candidate.stage
                    and row["parents"] == list(candidate.parents), f"候选注册不符：{candidate.candidate_id}")
        if mutation:
            require(len(entries) == 6, "精英变异槽位数量不符。")
            self.mutation_candidates = [c for c, _ in entries]
            self.mutation_specs = actual_specs
            raise ReplayBoundary()

        responses = {x["candidate_id"]: x for x in self.mutation_source_rows
                     if x["event"] == "proposal_response" and x.get("generation") == 5}
        results = {x["candidate_id"]: x["result"] for x in self.mutation_source_rows if x["event"] == "evaluation"}
        orders = []
        for candidate, prompt in entries:
            row, result = responses.get(candidate.candidate_id), results.get(candidate.candidate_id)
            require(row is not None and result is not None and row["prompt"] == prompt,
                    f"第 5 代协作提案、提示或评价缺失：{candidate.candidate_id}")
            chain.base.apply_result(self, candidate, row, result, SOURCE / "candidates" / candidate.candidate_id)
            orders.append(row["generation_order"])
        require(sorted(orders) == list(range(self._generated + 1, self._generated + len(orders) + 1)),
                "第 5 代协作提案生成顺序不连续。")
        self._generated += len(orders)
        require(key not in self.cached_batches, "协作阶段重复。")
        candidates = [candidate for candidate, _ in entries]
        self.cached_batches[key] = (actual_specs, candidates)
        return candidates


def reconstruct(backend, log):
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    class OfflineBackend:
        cfg = config

        def before_generation(self):
            pass

    segments = chain.find_segments(config)
    source_indexes = [i for i, (directory, _) in enumerate(segments) if directory.resolve() == SOURCE.resolve()]
    require(len(source_indexes) == 1 and source_indexes[0] > 0, "无法定位第 5 代变异断点。")
    index = source_indexes[0]
    with patch.object(chain, "find_segments", return_value=segments[:index]), \
            patch.object(previous, "ReplayController", ReplayController):
        controller, state = previous.reconstruct(OfflineBackend(), log)
    directory, source = segments[index]
    hashes = verify_segment(directory, source, state["latest"], state["model_calls"], state["evaluation_starts"])
    require(source[-1]["event"] == "preflight_or_run_stopped", "变异断点片段尚未停止。")
    controller.mutation_source_rows = source
    responses = {x["candidate_id"]: x for x in source if x["event"] == "proposal_response"
                 and x.get("generation") == 5}
    results = {x["candidate_id"]: x["result"] for x in source if x["event"] == "evaluation"}
    orders = []
    for candidate in state["pending"]:
        row, result = responses.get(candidate.candidate_id), results.get(candidate.candidate_id)
        require(row is not None and result is not None, f"第 5 代 EoH 评价缺失：{candidate.candidate_id}")
        chain.base.apply_result(controller, candidate, row, result, SOURCE / "candidates" / candidate.candidate_id)
        orders.append(row["generation_order"])
    require(sorted(orders) == list(range(controller._generated + 1, controller._generated + len(orders) + 1)),
            "第 5 代 EoH 生成顺序不连续。")
    controller._generated += len(orders)
    controller.replay_collaboration = True
    replay_start = len(log.items)
    try:
        controller.run_from_fifth(state["population"])
    except ReplayBoundary:
        pass
    else:
        raise ValueError("重放没有到达第 5 代变异断点。")
    require(len(controller.role_responses) == 7, "第 5 代角色响应不完整。")
    for event in ("candidate_registered", "elite_pair", "branch_transition", "slot_skipped"):
        original = [{k: v for k, v in x.items() if k not in ("event_index", "at_utc")}
                    for x in source if x["event"] == event and x.get("generation") == 5]
        replayed = [x for x in log.items[replay_start:] if x["event"] == event and x.get("generation") == 5]
        require(original == replayed, f"第 5 代 {event} 重放不一致。")

    first = next(i for i, x in enumerate(source) if x["event"] == "candidate_registered"
                 and x.get("generation") == 5 and str(x.get("stage", "")).startswith("mutation_"))
    mutation_rows = source[first:]
    attempts = [x for x in mutation_rows if x["event"] == "model_attempt"]
    failures = [x for x in mutation_rows if x["event"] == "model_failure_classified"]
    require([x["stage"] for x in attempts] == ["mutation_A_explorer", "mutation_A_exploiter", "mutation_A_integrator"]
            and len(failures) == 1 and failures[0]["stage"] == "mutation_A_integrator"
            and failures[0]["category"] == "network", "变异失败位置不符。")
    by_stage = {c.stage: c for c in controller.mutation_candidates}
    spec_by_id = {c.candidate_id: spec for c, spec in zip(controller.mutation_candidates, controller.mutation_specs)}
    for attempt in attempts:
        candidate = by_stage[attempt["stage"]]
        require(attempt["prompt"] == spec_by_id[candidate.candidate_id][2], "变异原提示不符。")
    mutation_responses = [x for x in mutation_rows if x["event"] == "proposal_response"]
    require(len(mutation_responses) == 2, "变异响应数量不符。")
    orders = []
    for row in mutation_responses:
        candidate = by_stage[row["stage"]]
        require(row["candidate_id"] == candidate.candidate_id and candidate.candidate_id in results,
                "变异响应或首次评价不符。")
        result = results[candidate.candidate_id]
        if candidate.stage == "mutation_A_exploiter":
            require(result["status"] == "infrastructure_error", "被中断评价的状态不符。")
        chain.base.apply_result(controller, candidate, row, result, SOURCE / "candidates" / candidate.candidate_id)
        orders.append(row["generation_order"])
    require(sorted(orders) == list(range(controller._generated + 1, controller._generated + len(orders) + 1)),
            "变异生成顺序不连续。")
    controller._generated += len(orders)
    by_stage["mutation_A_integrator"].error = "模型请求连接失败；保留失败，不自动重试。"
    attempted_ids = {by_stage[x["stage"]].candidate_id for x in attempts}
    pending = chain.unstarted_candidates(controller.mutation_candidates, attempted_ids, mutation_rows)
    require(len(pending) == 3, "未启动变异槽位数不符。")
    model_calls = state["model_calls"] + len([x for x in source if x["event"] == "model_attempt"])
    eval_starts = state["evaluation_starts"] + len([x for x in source if x["event"] == "evaluation_budget_reserved"])
    cpu_samples = state["cpu_samples"] + [x["accounting"]["cpu_seconds"] for x in source
                                         if x["event"] == "evaluation_cpu_final"]
    require(model_calls == 352 and eval_starts == len(cpu_samples) == 300, "原断点预算不符。")
    state["prior_hashes"].update(hashes)
    state["protected"].extend((directory / "events.jsonl", directory / "config.json"))
    prior = directory

    for directory, rows in segments[index + 1:]:
        hashes = verify_segment(directory, rows, prior, model_calls, eval_starts)
        require(rows[-1]["event"] == "preflight_or_run_stopped", "后续变异片段尚未停止。")
        registrations = [x["candidate_id"] for x in rows if x["event"] == "reused_candidate_registration"]
        require(registrations == [x.candidate_id for x in pending], "后续待执行变异不符。")
        pending_by_stage = {x.stage: x for x in pending}
        attempts = [x for x in rows if x["event"] == "model_attempt"]
        require(len({x["stage"] for x in attempts}) == len(attempts)
                and all(x["stage"] in pending_by_stage for x in attempts), "后续片段含重复或非变异调用。")
        require(len([x for x in rows if x["event"] == "model_budget_reserved"]) == len(attempts),
                "后续变异模型预算预留不符。")
        for attempt in attempts:
            candidate = pending_by_stage[attempt["stage"]]
            require(attempt["prompt"] == spec_by_id[candidate.candidate_id][2], "后续变异提示不符。")
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
                "后续变异生成顺序不连续。")
        controller._generated += len(orders)
        candidates = {x.candidate_id: x for x in pending}
        for ident, response in responses.items():
            require(ident in candidates and response["generation"] == 5, "后续变异候选不符。")
            if ident in invalid:
                candidates[ident].generation_order = response["generation_order"]
                candidates[ident].error = invalid[ident]
            else:
                result, candidate_dir = ((results[ident], directory / "candidates" / ident)
                                         if ident in results else saved_results[ident])
                chain.base.apply_result(controller, candidates[ident], response, result, candidate_dir)
        failures = [x for x in rows if x["event"] == "model_failure_classified"]
        for failure in failures:
            require(failure["stage"] in pending_by_stage, "后续失败变异不符。")
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

    require(pending and controller.fourth_random_state is not None, "没有可恢复变异或随机状态。")
    controller.random.setstate(controller.fourth_random_state)
    controller.replay_collaboration = False
    controller.resume_collaboration = True
    controller.backend = backend
    controller.mutation_source = prior
    controller._replay_eoh = controller.fifth_candidates
    controller._replay_specs = controller.fifth_specs
    state.update(pending=pending, pending_specs=[spec_by_id[c.candidate_id] for c in pending],
                 model_calls=model_calls, evaluation_starts=eval_starts, cpu_samples=cpu_samples,
                 segments=segments, latest=prior)
    return controller, state


def run_pending(controller, state):
    controller._generation_offset = 0
    controller.generation = 5
    entries = []
    for candidate, (stage, parents, prompt) in zip(state["pending"], state["pending_specs"]):
        require(candidate.stage == stage and list(candidate.parents) == parents, "待执行变异或父代不符。")
        controller.event("reused_candidate_registration", generation=5, candidate_id=candidate.candidate_id,
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
    controller.generation = 4


def simulate():
    backend, log = FixtureBackend(), MemoryLog()
    backend.cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    controller, state = reconstruct(backend, log)
    with patch.object(chain, "reconstruct", side_effect=reconstruct), patch.object(chain, "run_pending", side_effect=run_pending):
        chain.verify_serial_runtime()
    with chain.recovery_guard(state, RECOVERY / "unused_roco_gen5_mutation_fixture", MemoryLog()):
        code = ("from pathlib import Path\nfrom scripts import p1_formal_matrix as m\ntry:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED", "续跑互斥锁不一致。")
    count = len(state["pending"])
    run_pending(controller, state)
    require(len(backend.calls) == count and len(backend.evaluations) == count, "未启动变异重放不符。")
    require(not {"mutation_A_explorer", "mutation_A_exploiter", "mutation_A_integrator"}
            & {stage for stage, _ in backend.calls}, "已启动变异被重复执行。")
    selected = controller.run_from_fifth(state["population"])
    selections = [x["generation"] for x in log.items if x["event"] == "selection"]
    require(len(selected) == 10 and controller.generation == 6 and selections == [2, 3, 4, 5, 6],
            "重放后的代数或筛选不符。")
    require(state["model_calls"] + len(backend.calls) <= 420
            and state["evaluation_starts"] + len(backend.evaluations) <= 378, "模拟超出预算。")
    print(json.dumps({"status": "passed", "prior_model_calls": state["model_calls"],
                      "prior_evaluations": state["evaluation_starts"], "unstarted_mutations": count,
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
    runtime = previous.ResumeRuntime(run_dir, run_dir / "config.json", log, cli=cli)
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
    print(f"RoCo 第 5 代变异断点检查通过；记录：{run_dir}", flush=True)
    try:
        with chain.recovery_guard(state, run_dir, log, resume_basis="user_requested"):
            log.event("resume_checkpoint", interrupted_run=str(state["latest"]), generation=5,
                      unstarted_candidates=[x.candidate_id for x in state["pending"]],
                      model_calls_already_used=state["model_calls"], evaluation_starts_already_used=state["evaluation_starts"],
                      total_model_call_cap=420)
            run_pending(controller, state)
            selected = controller.run_from_fifth(state["population"])
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
                      "evaluation_starts": state["evaluation_starts"], "unstarted_mutations": len(state["pending"]),
                      "latest": str(state["latest"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
