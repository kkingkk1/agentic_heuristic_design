"""与本方法两代工程试跑使用相同任务和预算的 RoCo 适配试跑。"""

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
REFERENCE = ROOT / "outputs/p1_control/20261111_172813_speed_feedback_i3_g2_r2"

from baselines.roco.paper_controller import RoCoController  # noqa: E402
from baselines.roco.runtime import RoCoRuntime  # noqa: E402
from scripts import p0_model_probe as probe  # noqa: E402
from scripts.p0_controller import StopRun  # noqa: E402
from scripts.p0_runtime import JsonLog, digest, generation_requirements, memory_and_priority  # noqa: E402
from scripts.p1_formal_matrix import verify_training_software  # noqa: E402


class PilotRoCoController(RoCoController):
    def long_reflection_prompt(self, role, reflections, changes):
        return (super().long_reflection_prompt(role, reflections, changes)
                + "\nFor this run, aim for at most 40 words in <ans>; "
                  "the hard limit remains strictly fewer than 50 words.")


def build_config():
    source = json.loads((REFERENCE / "config.json").read_text(encoding="utf-8"))
    if source.get("method", {}).get("id") != "proposed_solution_quality_and_speed_agents":
        raise ValueError("参照试跑的方法身份不符。")
    config = copy.deepcopy(source)
    config["scope"] = "与本方法两代试跑同预算的 RoCo 工程试跑；不属于 P1 正式对照。"
    config["method"] = {"id": "roco_paper_adapted"}
    task, budget = config["task"], config["budget"]
    if (task["name"], task["prompt_condition"], task["problem_size"], task["evaluation_mode"],
            task["initial_proposals"], task["population_size"], task["generations"],
            task["collaboration_rounds"]) != ("tsp_aco", "white_box", 50, "final_quality", 3, 2, 2, 2):
        raise ValueError("参照试跑的任务或规模不符。")
    per_generation = generation_requirements(task["population_size"], rounds=task["collaboration_rounds"])
    calls = task["initial_proposals"] + task["generations"] * per_generation[0]
    evaluations = task["initial_proposals"] + task["generations"] * per_generation[1]
    if ((calls, evaluations) != (61, 49)
            or budget["normal_model_calls"] != calls
            or budget["max_initialization_and_generation_model_cli_invocations"] != calls
            or budget["normal_code_generation_slots"] != evaluations
            or budget["max_initialization_and_generation_evaluation_starts"] != evaluations
            or budget["additional_infrastructure_retry_calls"] != 0
            or budget["additional_infrastructure_evaluation_starts"] != 0):
        raise ValueError("参照试跑的调用、评价或重试预算不符。")
    return config


def reference_hashes():
    events = (REFERENCE / "events.jsonl").read_text(encoding="utf-8").splitlines()
    preflight = next(json.loads(line) for line in events if json.loads(line).get("event") == "preflight")
    for name, expected in preflight["code_sha256"].items():
        source = Path(name)
        if not source.is_file() or digest(source) != expected:
            raise StopRun(f"参照试跑的受保护输入已变化：{source}")
    return len(preflight["code_sha256"])


def check_environment(config):
    matrix = json.loads((ROOT / "cfg/p1_formal_matrix.json").read_text(encoding="utf-8"))
    verify_training_software(matrix)
    checked = reference_hashes()
    resources = memory_and_priority()
    if resources["available_memory_bytes"] < config["resources"]["min_available_system_memory_bytes"]:
        raise StopRun("可用内存低于本次资源合同阈值。")
    cli = probe.cli_command()
    version = probe.run_command(cli + ["--version"])
    login = probe.run_command(cli + ["login", "status"])
    if (version.get("exit_code") != 0 or login.get("exit_code") != 0
            or "Logged in using ChatGPT" not in (login.get("stdout", "") + login.get("stderr", ""))):
        raise StopRun("无法确认 Codex CLI 版本或 ChatGPT 登录。")
    return resources, cli, version.get("stdout", "").strip(), checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = build_config()
    if args.check:
        resources, _, version, checked = check_environment(config)
        print(f"RoCo 试跑预检通过：61 次模型调用、49 次评价槽位；"
              f"{checked} 项参照输入一致；可用内存 {resources['available_memory_bytes']}；{version}")
        return 0

    config_path = RUN_DIR / "config.json"
    with config_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    log = JsonLog(RUN_DIR / "events.jsonl")
    started = time.monotonic()
    try:
        resources, cli, version, checked = check_environment(config)
        runtime = RoCoRuntime(RUN_DIR, config_path, log)
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
                  scope=config["scope"], reference_config_sha256=digest(REFERENCE / "config.json"),
                  reference_events_sha256=digest(REFERENCE / "events.jsonl"), reference_inputs_verified=checked,
                  selection_seed=config["task"]["selection_seed"], initial_proposals=3, population_size=2,
                  generations=2, collaboration_rounds=2, reflection_target_words=40,
                  context_note="每次模型调用新建会话；服务端内部截断不可见。")
        runtime.cli = cli
        task = config["task"]
        controller = PilotRoCoController(runtime, log, population_size=task["population_size"],
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
        print(f"RoCo 工程试跑完成：{RUN_DIR}", flush=True)
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("pilot_stopped", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - started)
        print(f"RoCo 工程试跑停止：{exc}；记录：{RUN_DIR}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        log.event("pilot_stopped", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - started)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
