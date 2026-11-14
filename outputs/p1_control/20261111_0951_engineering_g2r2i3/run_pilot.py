"""本次 TSP-ACO 小规模工程观察：3 个初始提案、2 代、每代 2 轮。"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import p0_budget_quota, p0_model_probe as probe  # noqa: E402
from scripts.p0_controller import Controller, StopRun  # noqa: E402
from scripts.p0_runtime import JsonLog, digest, generation_requirements, memory_and_priority, quota_gate  # noqa: E402
from scripts.p1_formal_matrix import verify_training_software  # noqa: E402
from scripts.p1_runtime import TaskRuntime  # noqa: E402


def build_config():
    resource = json.loads((ROOT / "cfg/p0_tsp_aco.json").read_text(encoding="utf-8"))
    matrix = json.loads((ROOT / "cfg/p1_formal_matrix.json").read_text(encoding="utf-8"))
    initial, population, rounds, generations = 3, 2, 2, 2
    per_generation_calls, per_generation_evaluations = generation_requirements(population, rounds=rounds)
    calls = initial + generations * per_generation_calls
    evaluations = initial + generations * per_generation_evaluations
    if (per_generation_calls, per_generation_evaluations, calls, evaluations) != (29, 23, 61, 49):
        raise ValueError("工程运行槽位与指定规模不一致。")
    base_task = resource["task"]
    task = {key: base_task[key] for key in (
        "name", "function", "dataset", "dataset_sha256", "dataset_generator_seed",
        "instance_count", "problem_size", "evaluation_seeds", "ants", "iterations_per_instance",
        "whole_training_wall_timeout_seconds",
    )}
    task.update({
        "prompt_condition": "white_box", "objective_direction": "min",
        "reference_seed_in_prompt_or_population": False, "evaluation_mode": "final_quality",
        "selection_seed": 2026111101, "initial_proposals": initial, "population_size": population,
        "collaboration_rounds": rounds, "generations": generations,
    })
    model = copy.deepcopy(resource["model"])
    model["retry_policy"] = "本次不自动重试或修复候选；失败及开销照实记录。"
    return {
        "status": "frozen",
        "scope": "仅本次本方法工程观察；2 轮是临时设置，不属于 P1 正式对照。",
        "method": {"id": "proposed_solution_quality_and_speed_agents"},
        "task": task, "model": model, "resources": copy.deepcopy(resource["resources"]),
        "budget": {
            "normal_model_calls": calls, "additional_infrastructure_retry_calls": 0,
            "max_initialization_and_generation_model_cli_invocations": calls,
            "normal_code_generation_slots": evaluations, "additional_infrastructure_evaluation_starts": 0,
            "max_initialization_and_generation_evaluation_starts": evaluations,
            "future_initialization_and_generation_eval_process_tree_cpu_seconds":
                matrix["training_process_tree_cpu_seconds"],
            "token_limits": matrix["token_limits"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--allow-unquantified-quota", action="store_true")
    args = parser.parse_args()
    config = build_config()
    config_path = RUN_DIR / "config.json"
    with config_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    log = JsonLog(RUN_DIR / "events.jsonl")
    started = time.monotonic()
    try:
        matrix = json.loads((ROOT / "cfg/p1_formal_matrix.json").read_text(encoding="utf-8"))
        verify_training_software(matrix)
        runtime = TaskRuntime(RUN_DIR, config_path, log, allow_unquantified_quota=args.allow_unquantified_quota)
        driver = Path(__file__).resolve()
        runtime.protected.append(driver)
        runtime.protected_hashes[str(driver)] = digest(driver)
        resources = memory_and_priority()
        if resources["available_memory_bytes"] < config["resources"]["min_available_system_memory_bytes"]:
            raise StopRun("可用内存低于资源合同阈值。")
        cli = probe.cli_command()
        version = probe.run_command(cli + ["--version"])
        login = probe.run_command(cli + ["login", "status"])
        if (version.get("exit_code") != 0 or login.get("exit_code") != 0
                or "Logged in using ChatGPT" not in (login.get("stdout", "") + login.get("stderr", ""))):
            raise StopRun("无法确认 Codex CLI 版本或 ChatGPT 登录。")
        quota = p0_budget_quota.read_limits()
        log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
                  quota=quota, code_sha256=runtime.protected_hashes,
                  scope=config["scope"], selection_seed=config["task"]["selection_seed"],
                  initial_proposals=3, population_size=2, generations=2, collaboration_rounds=2,
                  unquantified_quota_accepted_for_this_run=args.allow_unquantified_quota,
                  context_note="每次模型调用新建会话；服务端内部截断不可见。")
        mode = quota_gate(quota, config["budget"]["normal_model_calls"],
                          allow_unquantified=args.allow_unquantified_quota)
        log.event("initial_quota_gate", mode=mode, required_model_calls=config["budget"]["normal_model_calls"])
        runtime.cli = cli
        task = config["task"]
        controller = Controller(runtime, log, population_size=task["population_size"],
                                initial_count=task["initial_proposals"], rounds=task["collaboration_rounds"],
                                seed=task["selection_seed"], generations=task["generations"],
                                task_contract=runtime.contract)
        selected = controller.run()
        log.event("pilot_completed", population=[
            {"candidate_id": candidate.candidate_id, "evaluation_origin": candidate.evaluation_origin,
             "stage": candidate.stage, "fitness": candidate.fitness,
             "average_objective": candidate.raw_objective, "average_solve_wall_seconds": candidate.timing_seconds}
            for candidate in selected],
            model_calls=runtime.budget.model_normal, evaluation_starts=runtime.budget.eval_starts,
            process_tree_cpu_seconds=runtime.budget.used_cpu(), elapsed_seconds=time.monotonic() - started)
        print(f"工程运行完成：{RUN_DIR}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, p0_budget_quota.QuotaProbeError, OSError, ValueError, KeyError) as exc:
        log.event("pilot_stopped", error=f"{type(exc).__name__}: {exc}",
                  elapsed_seconds=time.monotonic() - started)
        print(f"工程运行停止：{exc}；记录：{RUN_DIR}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        log.event("pilot_stopped", error=f"{type(exc).__name__}: {exc}",
                  elapsed_seconds=time.monotonic() - started)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
