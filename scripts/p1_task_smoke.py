"""逐条件检查登录模型生成和完整训练评价；不执行正式研发或测试。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import threading
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import p0_budget_quota, p0_model_probe as probe, p1_model_smoke as smoke  # noqa: E402
from scripts.p0_controller import Controller, StopRun  # noqa: E402
from scripts.p0_runtime import JsonLog, digest, memory_and_priority, quota_gate  # noqa: E402
from scripts.p1_runtime import TRAINING, TRAINING_PARAMETERS, TaskRuntime  # noqa: E402
from scripts.p1_task_contract import load_task_contract  # noqa: E402


CONDITIONS = tuple((name, black_box)
                   for name in ("tsp_aco", "op_aco", "cvrp_aco", "mkp_aco", "bpp_offline_aco", "tsp_gls")
                   for black_box in ((False,) if name == "tsp_gls" else (False, True)))
MAX_CALLS = len(CONDITIONS)
NUMBA_PATH = r"D:\Tools\agentic-heuristic-p1-numba"


def plan():
    return {
        "model": probe.MODEL,
        "access": "Codex CLI / current ChatGPT login",
        "conditions": [{"task": name, "prompt_condition": "black_box" if black else "white_box"}
                       for name, black in CONDITIONS],
        "max_model_cli_calls": MAX_CALLS,
        "max_training_evaluations": MAX_CALLS,
        "max_concurrent_conditions": 4,
        "infrastructure_retries": 0,
        "uses_formal_test_data": False,
        "note": "每个条件只检查一次无种子提案及其完整训练评价；不构成正式研发或性能比较。",
    }


def make_config(name, black_box):
    contract = load_task_contract(name, black_box=black_box)
    evaluator, dataset, dataset_hash = TRAINING[name]
    count, ants, iterations, generator_seed = TRAINING_PARAMETERS[name]
    config = smoke.make_config("proposed_qe", 1, 1)
    config["task"].update({
        "name": name,
        "prompt_condition": "black_box" if black_box else "white_box",
        "problem_size": contract.train_size,
        "objective_direction": contract.direction,
        "function": contract.signature.removeprefix("def ").removesuffix(":"),
        "instance_count": count,
        "ants": ants,
        "iterations_per_instance": iterations,
        "dataset_generator_seed": generator_seed,
        "evaluation_seeds": [20261103 + i for i in range(count)],
        "dataset": dataset.relative_to(ROOT).as_posix(),
        "dataset_sha256": dataset_hash,
        "whole_training_wall_timeout_seconds": evaluator.TIMEOUT_SECONDS,
        "reference_seed_in_prompt_or_population": False,
    })
    if name == "tsp_gls":
        config["task"]["perturbation_moves"] = 30
        config["resources"]["pythonpath"].append(NUMBA_PATH)
    config["budget"]["future_initialization_and_generation_eval_process_tree_cpu_seconds"] = (
        max(os.cpu_count() or 1, 28) * (evaluator.TIMEOUT_SECONDS + 2)
    )
    config["scope"] = "P1 任务和提示的单候选工程检查；不构成正式训练。"
    return config


def run_one(backend, log):
    contract = backend.contract
    controller = Controller(backend, log, task_contract=contract)
    prompt = (f"{contract.task_prompt()}\nPropose a new heuristic without an existing heuristic.\n"
              f"{contract.output_prompt()}")
    candidate = controller.batch([("init_01", (), prompt)])[0]
    return {"candidate_id": candidate.candidate_id, "valid": candidate.valid,
            "error": candidate.error, "objective": candidate.objective}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="执行有界检查；默认只打印方案")
    parser.add_argument("--allow-unquantified-quota", action="store_true",
                        help="本次小预算检查接受订阅余量无法换算为调用次数")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps(plan(), ensure_ascii=False, indent=2))
        return 0
    try:
        resources = memory_and_priority()
        if resources["available_memory_bytes"] < 8589934592:
            raise StopRun("可用内存低于 8 GiB。")
        cli = probe.cli_command()
        login = probe.run_command(cli + ["login", "status"])
        if (login.get("exit_code") != 0
                or "Logged in using ChatGPT" not in (login.get("stdout", "") + login.get("stderr", ""))):
            raise StopRun("无法确认 Codex CLI 的 ChatGPT 登录。")
        quota = p0_budget_quota.read_limits()
        mode = quota_gate(quota, MAX_CALLS, allow_unquantified=args.allow_unquantified_quota)
    except (StopRun, probe.ProbeError, p0_budget_quota.QuotaProbeError, OSError) as exc:
        print(f"启动前检查失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    root = ROOT / "outputs/p1_task_smoke" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    with (root / "preflight.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"plan": plan(), "resources": resources, "quota": quota, "quota_gate": mode},
                  stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    counters = {"calls": 0, "lock": threading.Lock(), "stop": threading.Event()}
    print(f"任务接入检查开始；记录：{root}", flush=True)

    def work(condition):
        name, black_box = condition
        label = name + ("_black_box" if black_box else "_white_box")
        if counters["stop"].is_set():
            return {"condition": label, "valid": False, "error": "其他条件遇到基础设施阻塞。"}
        directory = root / label
        directory.mkdir()
        log = JsonLog(directory / "events.jsonl")
        try:
            config_path = directory / "config.json"
            with config_path.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(make_config(name, black_box), stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            runtime = TaskRuntime(directory, config_path, log, cli=cli)
            for path in (Path(__file__).resolve(), Path(smoke.__file__).resolve()):
                runtime.protected.append(path)
                runtime.protected_hashes[str(path)] = digest(path)
            log.event("smoke_integrity_preflight", code_sha256=runtime.protected_hashes)
            backend = smoke.SmokeBackend(runtime, counters, max_calls=MAX_CALLS,
                                         allow_unquantified_quota=args.allow_unquantified_quota)
            result = run_one(backend, log)
            runtime.assert_integrity()
            log.event("smoke_integrity_final", code_sha256=runtime.protected_hashes)
            result.update({"condition": label, "model_calls": runtime.budget.model_normal,
                           "evaluation_starts": runtime.budget.eval_starts})
        except (Exception, KeyboardInterrupt) as exc:
            counters["stop"].set()
            result = {"condition": label, "valid": False, "error": f"{type(exc).__name__}: {exc}"}
        log.event("smoke_finish", **result)
        print(f"{label}: {'通过' if result['valid'] else '未通过'}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(work, condition) for condition in CONDITIONS]
        results = [future.result() for future in as_completed(futures)]
    order = {name + ("_black_box" if black else "_white_box"): i
             for i, (name, black) in enumerate(CONDITIONS)}
    results.sort(key=lambda result: order[result["condition"]])
    with (root / "summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"model_calls": counters["calls"], "results": results}, stream,
                  ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"任务接入检查结束；记录：{root}", flush=True)
    return 0 if len(results) == MAX_CALLS and all(result.get("valid") for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
