"""重放 RoCo 第 3 次研发第 1 代，并保留末轮融合请求失败。"""

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid
from unittest.mock import patch


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
SOURCE = ROOT / "outputs/p1_roco/20261113_192144_7c3416c7"
sys.path.insert(0, str(ROOT))

from baselines.roco.paper_controller import RoCoController, ROLE_LABELS
from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate, p1_formal_matrix as matrix
from scripts import p1_classified_runtime
from scripts.p0_controller import StopRun, check_modification_scope
from scripts.p0_runtime import JsonLog, digest, memory_and_priority
from scripts.p1_classified_runtime import ClassifiedRoCoRuntime
from scripts.p1_task_contract import load_task_contract
from tests.test_p0_controller import FixtureBackend, MemoryLog


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_events(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class ReplayController(RoCoController):
    def __init__(self, *args, source_rows, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_rows = source_rows
        self.replay_mode = True
        self.used_roles = set()

    def event(self, kind, **fields):
        if self.replay_mode:
            if kind == "initial_population":
                rows = [x for x in self.source_rows if x["event"] == kind]
                require(len(rows) == 1 and rows[0]["candidate_ids"] == fields["candidate_ids"], "初始种群不一致。")
                kind = "reused_initial_population"
            elif kind == "elite_pair":
                rows = [x for x in self.source_rows if x["event"] == kind and x.get("generation") == 1]
                require(len(rows) == 1 and (rows[0]["a"], rows[0]["b"]) == (fields["a"], fields["b"]),
                        "第 1 代双精英不一致。")
                kind = "reused_elite_pair"
            elif kind == "branch_transition":
                role = ROLE_LABELS.get(fields["role"], fields["role"])
                rows = [x for x in self.source_rows if x["event"] == kind and x.get("generation") == 1
                        and x["role"] == role and x["round"] == fields["round"]]
                require(len(rows) == 1 and all(rows[0][key] == fields[key]
                        for key in ("previous", "proposal", "current", "from_objective", "proposal_objective", "to_objective")),
                        "分支状态不一致。")
                kind = "reused_branch_transition"
            elif kind in ("candidate_registered", "role_mapping"):
                kind = "reused_" + kind
        return super().event(kind, **fields)

    def request(self, stage, prompt, parser):
        if not self.replay_mode:
            return super().request(stage, prompt, parser)
        rows = [x for x in self.source_rows if x["event"] == "model_response"
                and x.get("generation") == 1 and x["stage"] == stage]
        require(len(rows) == 1 and stage not in self.used_roles and rows[0]["prompt"] == prompt,
                f"历史 Critic 请求不符：{stage}")
        self.used_roles.add(stage)
        self.event("reused_role_response", generation=1, stage=stage, source_events=str(SOURCE / "events.jsonl"))
        return parser(rows[0]["response"])

    def batch(self, specs):
        if not self.replay_mode:
            return super().batch(specs)
        entries = [(self.register(stage, tuple(p.candidate_id for p in parents)), prompt)
                   for stage, parents, prompt in specs]
        registrations = {x["candidate_id"]: x for x in self.source_rows if x["event"] == "candidate_registered"}
        responses = {x["candidate_id"]: x for x in self.source_rows if x["event"] == "proposal_response"}
        results = {x["candidate_id"]: x["result"] for x in self.source_rows if x["event"] == "evaluation"}
        orders = []
        for candidate, prompt in entries:
            row = registrations.get(candidate.candidate_id)
            require(row is not None and row["stage"] == candidate.stage and row["parents"] == list(candidate.parents),
                    f"历史候选注册不符：{candidate.candidate_id}")
            response = responses.get(candidate.candidate_id)
            if response is None:
                failed = [x for x in self.source_rows if x["event"] == "model_attempt" and x["stage"] == "fusion_final"]
                require(candidate.candidate_id == "c0082" and candidate.stage == "fusion_final" and len(entries) == 1
                        and len(failed) == 1 and failed[0]["prompt"] == prompt, "失败融合提示不符。")
                candidate.error = "模型连接失败；保留失败请求，不自动重试。"
                self.replay_mode = False
                continue
            require(response["prompt"] == prompt and candidate.candidate_id in results,
                    f"历史提案或评价缺失：{candidate.candidate_id}")
            apply_result(self, candidate, response, results[candidate.candidate_id])
            orders.append(response["generation_order"])
        require(sorted(orders) == list(range(self._generated + 1, self._generated + len(orders) + 1)),
                "历史生成顺序不连续。")
        self._generated += len(orders)
        return [candidate for candidate, _ in entries]


class RecoveryRuntime(ClassifiedRoCoRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        resources = self.cfg["resources"]
        require(resources["max_concurrent_model_calls"] == resources["max_concurrent_evaluations"] == 4,
                "冻结并发合同变化。")
        self.model_slots = threading.BoundedSemaphore(1)
        self.resume_first_generation = True

    def before_generation(self):
        if self.resume_first_generation:
            self.resume_first_generation = False
            self.log.event("reused_generation_budget_check", generation=1, source_events=str(SOURCE / "events.jsonl"))
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
        require(result["status"] == "candidate_error", "存在未预期的评价基础设施故障。")
        candidate.error = str(result.get("error") or result["status"])
        return
    checked = p0_tsp_aco_evaluate.read_complete(directory / "stdout.txt")
    require(all(checked[key] == result[key] for key in ("average", "fitness", "objectives")),
            f"评价输出不符：{candidate.candidate_id}")
    candidate.objective = float(result["average"])
    candidate.fitness = float(result["fitness"])
    candidate.completed_order = result["completed_monotonic_ns"]
    candidate.evaluation_origin = candidate.candidate_id


def load_state():
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    matrix.verify_training_config(config)
    rows = read_events(SOURCE / "events.jsonl")
    require([x["event_index"] for x in rows] == list(range(1, len(rows) + 1)), "事件编号不连续。")
    starts = [x for x in rows if x["event"] == "formal_run_started"]
    preflight = [x for x in rows if x["event"] == "preflight"]
    require(len(starts) == len(preflight) == 1 and starts[0]["method"] == "roco_paper_adapted"
            and starts[0]["formal_replicate"] == config["formal_replicate"] == 3
            and starts[0]["matrix_contract_sha256"] == config["matrix_contract_sha256"]
            and starts[0]["selection_seed"] == config["task"]["selection_seed"]
            and rows[-1]["event"] == "preflight_or_run_stopped", "研发身份或停止状态不符。")
    for name, expected in preflight[0]["code_sha256"].items():
        require(Path(name).is_file() and digest(Path(name)) == expected, f"受保护文件变化：{name}")
    failures = [x for x in rows if x["event"] == "model_failure_classified"]
    attempts = [x for x in rows if x["event"] == "model_attempt"]
    eval_ids = [x["candidate_id"] for x in rows if x["event"] == "evaluation_budget_reserved"]
    results = [x for x in rows if x["event"] == "evaluation"]
    responses = [x for x in rows if x["event"] == "proposal_response"]
    registrations = [x for x in rows if x["event"] == "candidate_registered"]
    require(len(attempts) == 86 and len(eval_ids) == len(results) == len(responses) == 81
            and len(set(eval_ids)) == 81 and set(eval_ids) == {x["candidate_id"] for x in results}
            == {x["candidate_id"] for x in responses} and len(registrations) == 82
            and len(failures) == 1 and failures[0]["stage"] == "fusion_final" and failures[0]["category"] == "network"
            and attempts[-1]["stage"] == "fusion_final", "请求、评价或失败状态不符。")
    cpu = [x["accounting"]["cpu_seconds"] for x in rows if x["event"] == "evaluation_cpu_final"]
    require(len(cpu) == 81, "评价 CPU 记录不完整。")
    return {"config": config, "rows": rows, "cpu_samples": cpu, "prior_hashes": preflight[0]["code_sha256"]}


def make_controller(backend, log, state):
    task = state["config"]["task"]
    return ReplayController(backend, log, source_rows=state["rows"], population_size=task["population_size"],
                            initial_count=task["initial_proposals"], rounds=task["collaboration_rounds"],
                            seed=task["selection_seed"], generations=task["generations"],
                            task_contract=load_task_contract("tsp_aco"))


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
        raise OSError(ctypes.get_last_error(), "无法创建研发互斥锁。")
    acquired = False
    try:
        result = kernel.WaitForSingleObject(handle, 0)
        require(result in (0, 0x80), "已有正式研发运行或无法取得互斥锁。")
        acquired = True
        matrix.verify_training_config(config)
        started = set()
        for path in (ROOT / "outputs/p1_roco").glob("*/config.json"):
            if path.parent.resolve() == run_dir.resolve():
                continue
            other = json.loads(path.read_text(encoding="utf-8"))
            if (other.get("matrix_contract_sha256") == config["matrix_contract_sha256"]
                    and matrix.run_identity(other) == matrix.run_identity(config)):
                event_path = path.parent / "events.jsonl"
                if event_path.is_file() and any(x["event"] == "formal_run_started" for x in read_events(event_path)):
                    started.add(path.parent.resolve())
        require(started == {SOURCE.resolve()}, "同一研发存在未知已启动运行。")
        task = config["task"]
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=task["name"], prompt_condition=task["prompt_condition"], method=config["method"]["id"],
                  formal_replicate=3, selection_seed=task["selection_seed"], resume_of=str(SOURCE),
                  interrupted_events_sha256=digest(SOURCE / "events.jsonl"), total_model_call_cap=420,
                  frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4,
                  note="末轮融合请求失败不重试；保留失败并继续反思和变异。")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def simulate():
    state = load_state()
    backend, log = FixtureBackend(), MemoryLog()
    backend.cfg = state["config"]
    controller = make_controller(backend, log, state)
    with recovery_guard(state["config"], RECOVERY / "unused_roco3_fusion_fixture", MemoryLog()):
        code = ("from pathlib import Path\nfrom scripts import p1_formal_matrix as m\ntry:\n"
                " with m.formal_run_guard({}, Path('unused'), None): pass\n"
                "except ValueError as e:\n print('LOCKED' if '已有 P1' in str(e) else 'OTHER')\n")
        child = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", timeout=30)
        require(child.returncode == 0 and child.stdout.strip() == "LOCKED", "研发互斥锁不一致。")
    selected = controller.run()
    require(len(selected) == 10 and controller.generation == 6 and len(backend.calls) == 334
            and len(backend.evaluations) == 296 and backend.calls[0][0] == "reflection_explorer"
            and [x["generation"] for x in log.items if x["event"] == "selection"] == list(range(1, 7)),
            "反思续跑、代次或预算模拟不符。")
    limited_log = MemoryLog()
    runtime = RecoveryRuntime(SOURCE, SOURCE / "config.json", limited_log, cli=["offline-fixture"])
    limited_controller = make_controller(runtime, limited_log, state)
    with patch.object(probe, "run_command", return_value={"exit_code": 1, "stdout": "", "stderr": ""}), \
            patch.object(probe, "extract_message", side_effect=probe.ProbeError("offline 429")), \
            patch.object(p1_classified_runtime, "classify_failure",
                         return_value={"category": "quota", "source": "fixture", "plugin_warnings": []}):
        try:
            limited_controller.run()
        except StopRun:
            pass
        else:
            raise ValueError("限流夹具没有停止。")
    attempts = [x for x in limited_log.items if x["event"] == "model_attempt"]
    require(len(attempts) == 1 and attempts[0]["stage"] == "reflection_explorer", "限流后启动其他模型请求。")
    print(json.dumps({"status": "passed", "prior_model_calls": 86, "prior_evaluations": 81,
                      "remaining_model_calls": 334, "remaining_evaluations": 296, "failed_candidate": "c0082",
                      "formal_mutex_shared": True, "model_concurrency": 1, "stop_after_429": True}, ensure_ascii=False))


def resume():
    state = load_state()
    config = state["config"]
    matrix.verify_training_software(matrix.load_matrix()[0])
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
    for path in [*(Path(x) for x in state["prior_hashes"]), SOURCE / "events.jsonl", SOURCE / "config.json", Path(__file__).resolve()]:
        if str(path) not in runtime.protected_hashes:
            runtime.protected.append(path)
            runtime.protected_hashes[str(path)] = digest(path)
    budget = runtime.budget
    budget.model_normal = 86
    budget.eval_starts = budget.eval_normal = 81
    budget.cpu_finished = sum(state["cpu_samples"])
    budget.eval_cpu_samples = state["cpu_samples"][:]
    log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
              code_sha256=runtime.protected_hashes, selection_seed=config["task"]["selection_seed"],
              generations=6, continuation=True, normal_model_slots_used=86, physical_model_cli_invocations_used=86,
              evaluation_starts_used=81, evaluation_cpu_seconds_used=budget.cpu_finished,
              frozen_model_concurrency=4, effective_model_concurrency=1, evaluation_concurrency=4)
    print(f"RoCo 第 3 次研发融合断点检查通过；记录：{run_dir}", flush=True)
    try:
        with recovery_guard(config, run_dir, log):
            log.event("resume_checkpoint", interrupted_run=str(SOURCE), generation=1, failed_candidate_id="c0082",
                      next_stage="reflection_explorer", model_calls_already_used=86, evaluation_starts_already_used=81,
                      total_model_call_cap=420)
            controller = make_controller(runtime, log, state)
            selected = controller.run()
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
        state = load_state()
        print(json.dumps({"status": "passed", "model_calls_used": 86, "evaluation_starts": len(state["cpu_samples"]),
                          "failed_fusion": "c0082"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, StopRun) as exc:
        print(f"RoCo 第 3 次研发融合恢复检查失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
