"""本方法 P1 多任务运行入口；仅接受已冻结的完整运行配置。"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import p0_budget_quota, p0_model_probe as probe  # noqa: E402
from scripts.p0_controller import Controller, StopRun  # noqa: E402
from scripts.p0_runtime import JsonLog, generation_requirements, memory_and_priority, quota_gate  # noqa: E402
from scripts.p1_formal_matrix import formal_run_guard, verify_training_config  # noqa: E402
from scripts.p1_runtime import TaskRuntime  # noqa: E402


def validate_config(config, *, method_id="proposed_solution_quality_and_speed_agents"):
    if (not isinstance(config, dict) or config.get("status") != "frozen"
            or not isinstance(config.get("method"), dict) or config["method"].get("id") != method_id):
        raise ValueError(f"正式入口只接受 status=frozen 且方法标识为 {method_id} 的完整配置。")
    task, budget = config.get("task"), config.get("budget")
    if not isinstance(task, dict) or not isinstance(budget, dict):
        raise ValueError("缺少任务或预算配置。")
    generations, seed = task.get("generations"), task.get("selection_seed")
    initial, population = task.get("initial_proposals"), task.get("population_size")
    if (type(generations) is not int or generations < 1 or type(seed) is not int or seed < 0
            or type(initial) is not int or type(population) is not int
            or (initial, population) not in ((30, 10), (6, 2))
            or task.get("collaboration_rounds") != 3
            or task.get("reference_seed_in_prompt_or_population") is not False):
        raise ValueError("初始化、代数、选择种子或无种子合同无效。")
    generation_calls, generation_evaluations = generation_requirements(population)
    model_calls = initial + generation_calls * generations
    code_slots = initial + generation_evaluations * generations
    model_retries, eval_retries = (budget.get("additional_infrastructure_retry_calls"),
                                   budget.get("additional_infrastructure_evaluation_starts"))
    cpu = budget.get("future_initialization_and_generation_eval_process_tree_cpu_seconds")
    if (type(model_retries) is not int or model_retries < 0
            or type(eval_retries) is not int or eval_retries < 0
            or budget.get("normal_model_calls") != model_calls
            or budget.get("normal_code_generation_slots") != code_slots
            or budget.get("max_initialization_and_generation_model_cli_invocations") != model_calls + model_retries
            or budget.get("max_initialization_and_generation_evaluation_starts") != code_slots + eval_retries
            or isinstance(cpu, bool) or not isinstance(cpu, (int, float)) or not math.isfinite(cpu) or cpu <= 0):
        raise ValueError("模型调用、评价次数或进程树 CPU 预算与代数不一致。")
    model, resources = config.get("model"), config.get("resources")
    if (not isinstance(model, dict) or model.get("name") != probe.MODEL
            or model.get("reasoning_effort_override") is not False
            or not isinstance(resources, dict) or resources.get("process_priority") != "Normal"
            or any(type(resources.get(key)) is not int or resources[key] <= 0
                   for key in ("max_concurrent_evaluations", "max_concurrent_model_calls",
                               "min_available_system_memory_bytes"))):
        raise ValueError("模型或运行资源配置无效。")
    return generations, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="已冻结的 P1 单运行配置 JSON")
    parser.add_argument("--run", action="store_true", help="通过门禁后执行；默认只做启动前检查")
    parser.add_argument("--allow-unquantified-quota", action="store_true",
                        help="仅在用户明确接受时，允许本次运行的订阅余量不可计数")
    args = parser.parse_args()
    if args.allow_unquantified_quota and not args.run:
        parser.error("--allow-unquantified-quota 只适用于明确启动的本次运行。")
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        generations, seed = validate_config(config)
        verify_training_config(config)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"配置无效：{exc}", file=sys.stderr)
        return 1

    run_dir = ROOT / "outputs" / "p1_control" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    log = JsonLog(run_dir / "events.jsonl")
    config_path = run_dir / "config.json"
    with config_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        runtime = TaskRuntime(run_dir, config_path, log,
                              allow_unquantified_quota=args.allow_unquantified_quota)
        resources = memory_and_priority()
        if resources["available_memory_bytes"] < config["resources"]["min_available_system_memory_bytes"]:
            raise StopRun("可用内存低于冻结阈值。")
        cli = probe.cli_command()
        version = probe.run_command(cli + ["--version"])
        login = probe.run_command(cli + ["login", "status"])
        if (version.get("exit_code") != 0 or login.get("exit_code") != 0
                or "Logged in using ChatGPT" not in (login.get("stdout", "") + login.get("stderr", ""))):
            raise StopRun("无法确认 Codex CLI 版本或 ChatGPT 登录。")
        quota = p0_budget_quota.read_limits()
        log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(), quota=quota,
                  code_sha256=runtime.protected_hashes, selection_seed=seed, generations=generations,
                  unquantified_quota_accepted_for_this_run=args.allow_unquantified_quota,
                  context_note="模型调用使用新会话；服务端内部截断不可见。")
        required = config["budget"]["max_initialization_and_generation_model_cli_invocations"]
        mode = quota_gate(quota, required, allow_unquantified=args.allow_unquantified_quota)
        log.event("initial_quota_gate", mode=mode, required_model_calls=required)
        if not args.run:
            log.event("finish", status="preflight_passed")
            print(f"启动前检查通过；记录：{run_dir}")
            return 0
        runtime.cli = cli
        controller = Controller(runtime, log, population_size=config["task"]["population_size"],
                                initial_count=config["task"]["initial_proposals"], rounds=3,
                                seed=seed, generations=generations, task_contract=runtime.contract)
        with formal_run_guard(config, run_dir, log):
            controller.run()
        print(f"训练完成；记录：{run_dir}")
        return 0
    except (StopRun, probe.ProbeError, p0_budget_quota.QuotaProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"运行停止：{exc}；记录：{run_dir}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
