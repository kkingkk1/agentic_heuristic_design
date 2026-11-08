"""HSEvo 固定来源适配入口；仅接受已冻结的单运行配置。"""

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

from baselines.hsevo.controller import HSEvoController  # noqa: E402
from baselines.hsevo.runtime import HSEvoRuntime  # noqa: E402
from scripts import p0_budget_quota, p0_model_probe as probe  # noqa: E402
from scripts.p0_controller import StopRun  # noqa: E402
from scripts.p0_runtime import JsonLog, memory_and_priority, quota_gate  # noqa: E402
from scripts.p1_formal_matrix import formal_run_guard, verify_training_config  # noqa: E402


def validate_config(config):
    if (not isinstance(config, dict) or config.get("status") != "frozen"
            or not isinstance(config.get("method"), dict)
            or config["method"].get("id") != "hsevo_fb569d6_adapted"):
        raise ValueError("只接受已冻结、标明 hsevo_fb569d6_adapted 的运行配置。")
    task, budget = config.get("task"), config.get("budget")
    if not isinstance(task, dict) or not isinstance(budget, dict):
        raise ValueError("缺少任务或预算配置。")
    generations, seed = task.get("generations"), task.get("selection_seed")
    if (type(generations) is not int or generations < 1 or type(seed) is not int or seed < 0
            or task.get("population_size") != 10 or task.get("initial_proposals") != 30
            or task.get("mutation_count") != 5 or task.get("reference_seed_in_prompt_or_population") is not True
            or task.get("reference_seed_source") != "reevo_2f70042_task_adapted"
            or task.get("hm_size") != 5 or task.get("hs_iterations") != 5
            or task.get("hmcr") != 0.7 or task.get("par") != 0.5 or task.get("bandwidth") != 0.2):
        raise ValueError("HSEvo 初始化、种子、代数或和声搜索参数无效。")
    model_calls, evaluation_slots = 30 + 18 * generations, 31 + 25 * generations
    model_retries = budget.get("additional_infrastructure_retry_calls")
    eval_retries = budget.get("additional_infrastructure_evaluation_starts")
    cpu = budget.get("future_initialization_and_generation_eval_process_tree_cpu_seconds")
    if (type(model_retries) is not int or model_retries < 0 or type(eval_retries) is not int or eval_retries < 0
            or budget.get("normal_model_calls") != model_calls
            or budget.get("normal_code_generation_slots") != evaluation_slots
            or budget.get("max_initialization_and_generation_model_cli_invocations") != model_calls + model_retries
            or budget.get("max_initialization_and_generation_evaluation_starts") != evaluation_slots + eval_retries
            or isinstance(cpu, bool) or not isinstance(cpu, (int, float)) or not math.isfinite(cpu) or cpu <= 0):
        raise ValueError("HSEvo 模型调用、含种子和和声搜索的评价次数或 CPU 预算与代数不一致。")
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
    parser.add_argument("--config", required=True, help="已冻结的 HSEvo 单运行配置 JSON")
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

    run_dir = ROOT / "outputs" / "p1_hsevo" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    log = JsonLog(run_dir / "events.jsonl")
    config_path = run_dir / "config.json"
    with config_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        runtime = HSEvoRuntime(run_dir, config_path, log, allow_unquantified_quota=args.allow_unquantified_quota)
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
                  method="hsevo_fb569d6_adapted", unquantified_quota_accepted_for_this_run=args.allow_unquantified_quota)
        required = config["budget"]["max_initialization_and_generation_model_cli_invocations"]
        mode = quota_gate(quota, required, allow_unquantified=args.allow_unquantified_quota)
        log.event("initial_quota_gate", mode=mode, required_model_calls=required)
        if not args.run:
            log.event("finish", status="preflight_passed")
            print(f"启动前检查通过；记录：{run_dir}")
            return 0
        runtime.cli = cli
        controller = HSEvoController(runtime, log, runtime.contract, population_size=10, initial_count=30,
                                     mutation_count=5, generations=generations, seed=seed)
        with formal_run_guard(config, run_dir, log):
            controller.run()
        print(f"HSEvo 训练完成；记录：{run_dir}")
        return 0
    except (StopRun, probe.ProbeError, p0_budget_quota.QuotaProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"运行停止：{exc}；记录：{run_dir}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
