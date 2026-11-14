"""从已完成的正式研发运行冻结最终候选，供后续测试使用。"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.eoh.paper_controller import parse_proposal as parse_eoh  # noqa: E402
from baselines.mcts_ahd.controller import parse_proposal as parse_mcts  # noqa: E402
from baselines.reevo.controller import parse_generated, seed_code, source_files, text  # noqa: E402
from scripts import p1_run_eoh, p1_run_generation, p1_run_hsevo, p1_run_mcts_ahd, p1_run_reevo, p1_run_roco  # noqa: E402
from scripts.p0_controller import check_modification_scope  # noqa: E402
from scripts.p0_runtime import digest  # noqa: E402
from scripts.p1_formal_matrix import verify_training_config  # noqa: E402
from scripts.p1_runtime import TRAINING  # noqa: E402
from scripts.p1_task_contract import load_task_contract  # noqa: E402


METHODS = {
    "proposed_solution_quality_and_speed_agents": ("p1_control", p1_run_generation.validate_config),
    "roco_paper_adapted": ("p1_roco", p1_run_roco.validate_config),
    "eoh_paper_adapted": ("p1_eoh", p1_run_eoh.validate_config),
    "reevo_2f70042_adapted": ("p1_reevo", p1_run_reevo.validate_config),
    "hsevo_fb569d6_adapted": ("p1_hsevo", p1_run_hsevo.validate_config),
    "mcts_ahd_ee9c4f4_adapted": ("p1_mcts_ahd", p1_run_mcts_ahd.validate_config),
}


def final_candidate(events, method):
    if (not events or events[-1].get("event") != "finish" or events[-1].get("status") != "completed"
            or any(event.get("event") == "preflight_or_run_stopped" for event in events)):
        raise ValueError("研发运行未完整结束，不能冻结候选。")
    if method in ("proposed_solution_quality_and_speed_agents", "roco_paper_adapted", "eoh_paper_adapted"):
        selections = [event for event in events if event.get("event") == "selection"]
        selected = selections[-1].get("selected") if selections else None
        candidate = selected[0].get("candidate_id") if isinstance(selected, list) and selected else None
    elif method in ("reevo_2f70042_adapted", "hsevo_fb569d6_adapted"):
        candidate = events[-1].get("elite")
    elif method == "mcts_ahd_ee9c4f4_adapted":
        candidate = events[-1].get("best")
    else:
        raise ValueError("未知方法。")
    if not isinstance(candidate, str) or re.fullmatch(r"c[0-9]{4,}", candidate) is None:
        raise ValueError("无法从最终选择确认候选 ID。")
    return candidate


def original_code(events, method, candidate_id, contract):
    registrations = [event for event in events if event.get("event") == "candidate_registered"
                     and event.get("candidate_id") == candidate_id]
    if len(registrations) != 1:
        raise ValueError("候选注册记录缺失或重复。")
    if registrations[0].get("stage") == "seed" and method in ("reevo_2f70042_adapted", "hsevo_fb569d6_adapted"):
        return seed_code(contract, text(source_files(contract)["seed_func"]))
    harmony = [event for event in events if event.get("event") == "harmony_candidate"
               and event.get("candidate_id") == candidate_id]
    if harmony and method == "hsevo_fb569d6_adapted":
        if len(harmony) != 1 or not isinstance(harmony[0].get("code"), str):
            raise ValueError("和声候选代码记录不完整。")
        return contract.parse_proposal(f"<thought>harmony</thought><code>{harmony[0]['code']}</code>")["code"]
    responses = [event for event in events if event.get("event") == "proposal_response"
                 and event.get("candidate_id") == candidate_id]
    if len(responses) != 1:
        raise ValueError("候选原始模型响应缺失或重复。")
    response = responses[0].get("response")
    if method in ("reevo_2f70042_adapted", "hsevo_fb569d6_adapted"):
        return parse_generated(contract, response)["code"]
    if method == "eoh_paper_adapted":
        return parse_eoh(contract, response)["code"]
    if method == "mcts_ahd_ee9c4f4_adapted":
        return parse_mcts(contract, response)["code"]
    return contract.parse_proposal(response)["code"]


def freeze(run_dir):
    run_dir = Path(run_dir).resolve()
    config_path, events_path = run_dir / "config.json", run_dir / "events.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    method = config.get("method", {}).get("id")
    if method not in METHODS:
        raise ValueError("研发方法标识无效。")
    output_name, validate = METHODS[method]
    if run_dir.parent != (ROOT / "outputs" / output_name).resolve():
        raise ValueError("研发运行目录不在对应方法的正式输出目录。")
    validate(config)
    verify_training_config(config)
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    preflight = [event for event in events if event.get("event") == "preflight"]
    started = [event for event in events if event.get("event") == "formal_run_started"]
    if len(preflight) != 1 or not isinstance(preflight[0].get("code_sha256"), dict):
        raise ValueError("正式运行的启动检查或代码哈希记录缺失。")
    if (len(started) != 1 or started[0].get("matrix_contract_sha256") != config["matrix_contract_sha256"]
            or started[0].get("task") != config["task"]["name"]
            or started[0].get("prompt_condition") != config["task"]["prompt_condition"]
            or started[0].get("method") != method or started[0].get("formal_replicate") != config["formal_replicate"]
            or started[0].get("selection_seed") != config["task"]["selection_seed"]):
        raise ValueError("正式研发的矩阵、重复索引或启动记录不一致。")
    for filename, expected in preflight[0]["code_sha256"].items():
        if digest(Path(filename)) != expected:
            raise ValueError(f"受保护文件已变化：{filename}")
    candidate_id = final_candidate(events, method)
    task = config["task"]
    contract = load_task_contract(task["name"], black_box=task["prompt_condition"] == "black_box")
    candidate_root = run_dir / "candidates" / candidate_id
    candidate_path = candidate_root / "problems" / task["name"] / "gpt.py"
    code = original_code(events, method, candidate_id, contract)
    if candidate_path.read_text(encoding="utf-8") != code + "\n":
        raise ValueError("当前候选代码与原始提案不一致。")
    check_modification_scope(code)
    record = json.loads((candidate_root / "result.json").read_text(encoding="utf-8"))
    evaluations = [event.get("result") for event in events if event.get("event") == "evaluation"
                   and event.get("candidate_id") == candidate_id]
    if (record.get("status") != "passed" or len(evaluations) != 1 or evaluations[0] != record
            or record.get("candidate_id") != candidate_id):
        raise ValueError("候选评价记录缺失、不一致或未通过。")
    read_options = ({"stagnation_rounds": task["stagnation_rounds"]}
                    if task.get("evaluation_mode") == "quality_per_time" else {})
    checked = TRAINING[task["name"]][0].read_complete(candidate_root / "stdout.txt", **read_options)
    if (not math.isclose(checked["average"], record["average"], rel_tol=1e-12, abs_tol=1e-12)
            or checked["fitness"] != record["fitness"] or checked["objectives"] != record["objectives"]):
        raise ValueError("训练评价独立重读与结果记录不一致。")
    for field in ("solve_wall_seconds", "average_solve_wall_seconds", "heuristic_wall_seconds",
                  "average_heuristic_wall_seconds", "stopping_reasons"):
        if checked.get(field) != record.get(field):
            raise ValueError(f"训练计时或停止记录不一致：{field}。")
    destination = ROOT / "outputs/p1_frozen_candidates" / method / run_dir.name
    destination.mkdir(parents=True, exist_ok=False)
    frozen = destination / "candidate.py"
    with frozen.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(code + "\n")
    manifest = {"status": "frozen_after_training", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "method": method, "task": task["name"], "prompt_condition": task["prompt_condition"],
                "matrix_contract_sha256": config["matrix_contract_sha256"],
                "formal_replicate": config["formal_replicate"], "selection_seed": task["selection_seed"],
                "source_run": str(run_dir), "source_candidate_id": candidate_id,
                "source_config_sha256": digest(config_path), "source_events_sha256": digest(events_path),
                "source_result_sha256": digest(candidate_root / "result.json"),
                "source_stdout_sha256": digest(candidate_root / "stdout.txt"),
                "candidate_sha256": digest(frozen), "train_objective": checked["average"],
                "train_fitness": checked["fitness"], "note": "训练完成后冻结；尚未执行测试。"}
    with (destination / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="已完成的正式研发运行目录")
    args = parser.parse_args()
    try:
        destination = freeze(args.run_dir)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"候选未冻结：{exc}", file=sys.stderr)
        return 1
    print(f"候选已冻结：{destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
