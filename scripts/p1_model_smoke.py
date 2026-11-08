"""六种方法的有界单候选检查；只使用训练集和一个合成测试实例。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import threading
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.eoh.paper_controller import EoHController, make_prompt  # noqa: E402
from baselines.eoh.runtime import EoHRuntime  # noqa: E402
from baselines.hsevo.controller import HSEvoController  # noqa: E402
from baselines.hsevo.runtime import HSEvoRuntime  # noqa: E402
from baselines.mcts_ahd.controller import MCTSAHDController  # noqa: E402
from baselines.mcts_ahd.runtime import MCTSAHDRuntime  # noqa: E402
from baselines.reevo.controller import ReEvoController  # noqa: E402
from baselines.reevo.runtime import ReEvoRuntime  # noqa: E402
from baselines.roco.paper_controller import RoCoController  # noqa: E402
from baselines.roco.runtime import RoCoRuntime  # noqa: E402
from scripts import p0_budget_quota, p0_model_probe as probe, p1_calibrate_test_cost, p1_token_usage  # noqa: E402
from scripts.p0_controller import Candidate, Controller, StopRun  # noqa: E402
from scripts.p0_runtime import JsonLog, Runtime, digest, memory_and_priority, quota_gate  # noqa: E402
from scripts.p1_runtime import TaskRuntime  # noqa: E402
from scripts.p1_task_contract import load_task_contract  # noqa: E402


METHODS = (
    ("proposed", "proposed_solution_quality_and_speed_agents", TaskRuntime, 1, 1),
    ("eoh", "eoh_paper_adapted", EoHRuntime, 1, 1),
    ("reevo", "reevo_2f70042_adapted", ReEvoRuntime, 1, 1),
    ("hsevo", "hsevo_fb569d6_adapted", HSEvoRuntime, 1, 1),
    ("mcts", "mcts_ahd_ee9c4f4_adapted", MCTSAHDRuntime, 2, 1),
    ("roco", "roco_paper_adapted", RoCoRuntime, 1, 1),
)
MAX_CALLS = 7
SYNTHETIC_TIMEOUT_SECONDS = 60


def plan(method=None):
    methods = tuple(spec for spec in METHODS if method is None or spec[0] == method)
    if not methods:
        raise ValueError("未知工程检查方法。")
    max_calls = sum(spec[3] for spec in methods)
    max_evaluations = sum(spec[4] for spec in methods)
    return {"task": "tsp_aco", "prompt_condition": "white_box", "model": probe.MODEL,
            "access": "Codex CLI / current ChatGPT login", "max_model_cli_calls": max_calls,
            "max_evaluation_starts": max_evaluations, "evaluation_wall_timeout_seconds_each": 60,
            "max_evaluation_process_tree_cpu_seconds": max_evaluations * max(os.cpu_count() or 1, 28) * 62,
            "max_concurrent_methods": 4, "infrastructure_retries": 0, "reset_credits": 0,
            "synthetic_test_instances_per_candidate": 1, "synthetic_test_size": 50,
            "synthetic_test_iterations": 500, "synthetic_timeout_seconds_each": SYNTHETIC_TIMEOUT_SECONDS,
            "uses_formal_test_data": False,
            "operations": [{"method": name, "max_model_calls": calls, "max_evaluations": evaluations}
                           for name, _identifier, _runtime, calls, evaluations in methods],
            "note": "每种方法只检查一个候选；不构成完整代验收或性能比较。"}


class SmokeBackend:
    """只在本脚本显式获准的工程检查中使用登录模型。"""

    def __init__(self, runtime, counters, *, max_calls=MAX_CALLS, allow_unquantified_quota=True):
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("工程检查的模型调用上限必须是正整数。")
        if type(allow_unquantified_quota) is not bool:
            raise ValueError("不可量化余量的处理方式必须是布尔值。")
        self.runtime, self.counters = runtime, counters
        self.contract = runtime.contract
        self.max_calls = max_calls
        self.allow_unquantified_quota = allow_unquantified_quota

    def model(self, stage, prompt):
        with self.counters["lock"]:
            if self.counters["stop"].is_set():
                raise StopRun("其他检查遇到额度或基础设施阻塞。")
            if self.counters["calls"] >= self.max_calls:
                raise StopRun(f"真实模型检查达到本次 {self.max_calls} 次调用上限。")
            quota = p0_budget_quota.read_limits()
            quota_gate(quota, self.max_calls - self.counters["calls"],
                       allow_unquantified=self.allow_unquantified_quota)
            self.runtime.log.event("smoke_quota", stage=stage, quota=quota)
            self.counters["calls"] += 1
        # 工程配置仍禁止普通 TaskRuntime 请求模型；此脚本只为已授权的单候选检查调用基类。
        return Runtime.model(self.runtime, stage, prompt)

    def prepare_proposal(self, candidate):
        return self.runtime.prepare_proposal(candidate)

    def release_proposal(self, candidate):
        return self.runtime.release_proposal(candidate)

    def evaluate(self, candidate):
        return self.runtime.evaluate(candidate)


def make_config(method_id, calls, evaluations):
    source = json.loads((ROOT / "cfg/p0_tsp_aco.json").read_text(encoding="utf-8"))
    task = dict(source["task"])
    task.update({"prompt_condition": "white_box", "objective_direction": "min"})
    budget = dict(source["budget"])
    budget.update({"normal_model_calls": calls, "additional_infrastructure_retry_calls": 0,
                   "max_initialization_and_generation_model_cli_invocations": calls,
                   "normal_code_generation_slots": evaluations, "additional_infrastructure_evaluation_starts": 0,
                   "max_initialization_and_generation_evaluation_starts": evaluations,
                   "future_initialization_and_generation_eval_process_tree_cpu_seconds":
                       evaluations * max(os.cpu_count() or 1, 28) * 62})
    return {"status": "engineering", "method": {"id": method_id}, "task": task,
            "model": source["model"], "resources": source["resources"], "budget": budget,
            "scope": "经单次授权的 P1 登录渠道单候选检查；不构成正式训练。"}


def run_one(name, runtime, log, counters, *, max_calls=MAX_CALLS):
    backend = SmokeBackend(runtime, counters, max_calls=max_calls)
    contract = runtime.contract
    if name == "proposed":
        controller = Controller(backend, log, task_contract=contract)
        prompt = (f"{contract.task_prompt()}\nPropose a new heuristic without an existing heuristic.\n"
                  f"{contract.output_prompt()}")
        candidate = controller.batch([("init_01", (), prompt)])[0]
    elif name == "eoh":
        controller = EoHController(backend, log, contract)
        candidate = controller.batch([("init_01", (), make_prompt(contract, "I1", ()))])[0]
    elif name == "reevo":
        controller = ReEvoController(backend, log, contract)
        candidate = controller.batch([("init_01", (), controller.initial_prompt())])[0]
    elif name == "hsevo":
        controller = HSEvoController(backend, log, contract)
        candidate = controller.batch([("init_01", (), controller.initial_prompt(0))])[0]
    elif name == "mcts":
        controller = MCTSAHDController(backend, log, contract)
        candidate = controller.create_candidate("i1", ())
    elif name == "roco":
        controller = RoCoController(backend, log, task_contract=contract)
        source = ROOT / "outputs/p0_control/20261103_211613_fc69d43a/candidates/c0001"
        record = json.loads((source / "result.json").read_text(encoding="utf-8"))
        if record.get("status") != "passed" or not math.isfinite(record["average"]):
            raise StopRun("RoCo 工程父代缺少有效的 P0 评价记录。")
        parent = Candidate("p0_c0001", 0, "fixture", (), thought="P0 evaluated engineering candidate",
                           code=(source / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8"),
                           objective=record["average"], fitness=record["fitness"])
        critic = {"reflection": "Improve global exploration.", "advice": "Explore alternative edge scoring."}
        role = controller.design_roles[0]
        candidate = controller.batch([(f"start_{role}", (parent,), controller.branch_prompt(role, parent, critic))])[0]
    else:
        raise ValueError("未知工程检查方法。")
    if candidate is None:
        raise StopRun("没有生成候选。")
    return {"candidate_id": candidate.candidate_id, "valid": candidate.valid, "error": candidate.error,
            "objective": candidate.objective, "normal_model_calls": runtime.budget.model_normal,
            "evaluation_starts": runtime.budget.eval_starts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="执行已获授权的有界检查；默认只打印方案")
    parser.add_argument("--accept-unquantified-quota", action="store_true", help="用户已接受本次订阅余量无法换算为调用次数")
    parser.add_argument("--method", choices=[spec[0] for spec in METHODS], help="仅检查所选方法")
    args = parser.parse_args()
    methods = tuple(spec for spec in METHODS if args.method is None or spec[0] == args.method)
    run_plan = plan(args.method)
    if not args.run:
        print(json.dumps(run_plan, ensure_ascii=False, indent=2))
        return 0
    if not args.accept_unquantified_quota:
        parser.error("本次订阅剩余次数无法量化；须先取得用户对该具体检查方案的授权。")
    resources = memory_and_priority()
    if resources["available_memory_bytes"] < 8589934592:
        raise StopRun("可用内存低于 8 GiB，停止检查。")
    cli = probe.cli_command()
    login = probe.run_command(cli + ["login", "status"])
    if login.get("exit_code") != 0 or "Logged in using ChatGPT" not in (login.get("stdout", "") + login.get("stderr", "")):
        raise StopRun("无法确认 Codex CLI 的 ChatGPT 登录。")
    quota = p0_budget_quota.read_limits()
    quota_gate(quota, run_plan["max_model_cli_calls"], allow_unquantified=True)
    root = ROOT / "outputs/p1_model_smoke" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    with (root / "preflight.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"plan": run_plan, "resources": resources, "quota": quota}, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    counters = {"calls": 0, "lock": threading.Lock(), "stop": threading.Event()}
    print(f"登录模型检查开始；记录：{root}", flush=True)

    def work(spec):
        name, method_id, runtime_type, calls, evaluations = spec
        if counters["stop"].is_set():
            return {"method": name, "valid": False, "error": "其他检查遇到额度或基础设施阻塞。"}
        directory = root / name
        directory.mkdir()
        config = make_config(method_id, calls, evaluations)
        config_path = directory / "config.json"
        with config_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        log = JsonLog(directory / "events.jsonl")
        try:
            runtime = runtime_type(directory, config_path, log, cli=cli)
            runtime.protected.append(Path(__file__).resolve())
            runtime.protected_hashes[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
            log.event("smoke_integrity_preflight", code_sha256=runtime.protected_hashes)
            result = run_one(name, runtime, log, counters, max_calls=run_plan["max_model_cli_calls"])
            if result["valid"]:
                candidate_root = directory / "candidates" / result["candidate_id"]
                synthetic = p1_calibrate_test_cost.calibrate("tsp_aco", 50, directory / "synthetic",
                                                           SYNTHETIC_TIMEOUT_SECONDS, candidate_root=candidate_root)
                result["synthetic_status"] = synthetic["status"]
                result["synthetic_record"] = str(directory / "synthetic/tsp_aco/50/result.json")
                if synthetic["status"] != "passed":
                    result["error"] = synthetic.get("error", "合成实例检查未通过。")
                log.event("smoke_synthetic", candidate_id=result["candidate_id"], status=synthetic["status"],
                          record=result["synthetic_record"])
            else:
                result["synthetic_status"] = "not_run"
            runtime.assert_integrity()
            log.event("smoke_integrity_final", code_sha256=runtime.protected_hashes)
            result["method"] = name
            log.event("smoke_finish", **result)
            print(f"{name}: 训练 {'通过' if result['valid'] else '未通过'}，"
                  f"合成测试 {result.get('synthetic_status', 'not_run')}", flush=True)
            return result
        except (Exception, KeyboardInterrupt) as exc:
            message = f"{type(exc).__name__}: {exc}"
            log.event("smoke_finish", method=name, valid=False, error=message)
            if isinstance(exc, KeyboardInterrupt) or any(term in message for term in ("额度", "限流", "基础设施", "预算")):
                counters["stop"].set()
            print(f"{name}: 检查停止，{message}", flush=True)
            return {"method": name, "valid": False, "error": message}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(work, spec) for spec in methods]
        results = [future.result() for future in as_completed(futures)]
    order = {spec[0]: i for i, spec in enumerate(methods)}
    results.sort(key=lambda result: order[result["method"]])
    with (root / "summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"calls": counters["calls"], "results": results, "formal_test_data_used": False},
                  stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    logs = [root / name / "events.jsonl" for name, *_ in methods if (root / name / "events.jsonl").is_file()]
    usage = p1_token_usage.summarize(p1_token_usage.read_events(path) for path in logs)
    usage["source_logs"] = [str(path) for path in logs]
    with (root / "token_usage.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(usage, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"登录模型检查结束；记录：{root}")
    return 0 if (len(results) == len(methods)
                 and all(result.get("valid") and result.get("synthetic_status") == "passed" for result in results)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
