"""用户授权的收益时间比反馈工程试跑：3 个初始提案、两代各两轮。"""

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
import traceback


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(ROOT))

from scripts import p0_model_probe as probe  # noqa: E402
from scripts.p0_controller import Controller, StopRun  # noqa: E402
from scripts.p0_runtime import JsonLog, digest, generation_requirements, memory_and_priority  # noqa: E402
from scripts.p1_formal_matrix import verify_training_software  # noqa: E402
from scripts.p1_runtime import TaskRuntime  # noqa: E402


class PilotController(Controller):
    def long_reflection_prompt(self, role, reflections, changes):
        return (super().long_reflection_prompt(role, reflections, changes)
                + "\nFor this run, aim for at most 40 words in <ans>; "
                  "the hard limit remains strictly fewer than 50 words.")


def build_config():
    resource = json.loads((ROOT / "cfg/p0_tsp_aco.json").read_text(encoding="utf-8"))
    matrix = json.loads((ROOT / "cfg/p1_formal_matrix.json").read_text(encoding="utf-8"))
    initial, population, rounds, generations = 3, 2, 2, 2
    per_generation = generation_requirements(population, rounds=rounds)
    calls = initial + generations * per_generation[0]
    evaluations = initial + generations * per_generation[1]
    if (per_generation, calls, evaluations) != ((29, 23), 61, 49):
        raise ValueError("试跑模型调用或评价槽位与所选规模不符。")
    source = resource["task"]
    task = {name: source[name] for name in (
        "name", "function", "dataset", "dataset_sha256", "dataset_generator_seed",
        "instance_count", "problem_size", "evaluation_seeds", "ants", "iterations_per_instance",
        "whole_training_wall_timeout_seconds",
    )}
    task.update({"prompt_condition": "white_box", "objective_direction": "min",
                 "reference_seed_in_prompt_or_population": False, "evaluation_mode": "final_quality",
                 "selection_seed": 2026111103, "initial_proposals": initial, "population_size": population,
                 "collaboration_rounds": rounds, "generations": generations})
    model = copy.deepcopy(resource["model"])
    model["retry_policy"] = "本次不自动重试或修复候选；失败及开销照实记录。"
    return {
        "status": "frozen",
        "scope": "用户授权的收益时间比反馈两代工程试跑；不属于 P1 正式对照。",
        "trial_adjustment": "长期反思提示目标不超过 40 词；解析硬上限仍为少于 50 词。",
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
    }, matrix


def check_environment(config, matrix):
    verify_training_software(matrix)
    resources = memory_and_priority()
    if resources["available_memory_bytes"] < config["resources"]["min_available_system_memory_bytes"]:
        raise StopRun("可用内存低于本次资源合同阈值。")
    cli = probe.cli_command()
    version = probe.run_command(cli + ["--version"])
    login = probe.run_command(cli + ["login", "status"])
    if (version.get("exit_code") != 0 or login.get("exit_code") != 0
            or "Logged in using ChatGPT" not in (login.get("stdout", "") + login.get("stderr", ""))):
        raise StopRun("无法确认 Codex CLI 版本或 ChatGPT 登录。")
    return resources, cli, version.get("stdout", "").strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config, matrix = build_config()
    if args.check:
        resources, _, version = check_environment(config, matrix)
        print(f"试跑预检通过：61 次模型调用、49 次评价槽位；可用内存 {resources['available_memory_bytes']}；{version}")
        return 0

    config_path = RUN_DIR / "config.json"
    with config_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    log = JsonLog(RUN_DIR / "events.jsonl")
    started = time.monotonic()
    try:
        resources, cli, version = check_environment(config, matrix)
        runtime = TaskRuntime(RUN_DIR, config_path, log)
        driver = Path(__file__).resolve()
        runtime.protected.append(driver)
        runtime.protected_hashes[str(driver)] = digest(driver)
        for source in runtime.protected:
            source = source.resolve()
            if source.is_relative_to(ROOT) and not source.is_relative_to(RUN_DIR) and source.suffix in (
                    ".py", ".json", ".yaml", ".txt"):
                snapshot = RUN_DIR / "source_snapshots" / source.relative_to(ROOT)
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as src, snapshot.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                if digest(snapshot) != runtime.protected_hashes[str(source)]:
                    raise StopRun(f"源码快照不一致：{source}")
        runtime.assert_integrity()
        log.event("preflight", resources=resources, cli_version=version, code_sha256=runtime.protected_hashes,
                  scope=config["scope"], selection_seed=config["task"]["selection_seed"],
                  initial_proposals=3, population_size=2, generations=2, collaboration_rounds=2,
                  reflection_target_words=40, context_note="每次模型调用新建会话；服务端内部截断不可见。")
        runtime.cli = cli
        task = config["task"]
        controller = PilotController(runtime, log, population_size=task["population_size"],
                                     initial_count=task["initial_proposals"], rounds=task["collaboration_rounds"],
                                     seed=task["selection_seed"], generations=task["generations"],
                                     task_contract=runtime.contract)
        selected = controller.run()
        runtime.assert_integrity()
        log.event("pilot_completed", population=[
            {"candidate_id": candidate.candidate_id, "evaluation_origin": candidate.evaluation_origin,
             "stage": candidate.stage, "fitness": candidate.fitness,
             "average_objective": candidate.raw_objective, "average_solve_wall_seconds": candidate.timing_seconds}
            for candidate in selected], model_calls=runtime.budget.model_normal,
            evaluation_starts=runtime.budget.eval_starts, process_tree_cpu_seconds=runtime.budget.used_cpu(),
            elapsed_seconds=time.monotonic() - started)
        print(f"工程试跑完成：{RUN_DIR}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("pilot_stopped", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - started)
        print(f"工程试跑停止：{exc}；记录：{RUN_DIR}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        log.event("pilot_stopped", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - started)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
