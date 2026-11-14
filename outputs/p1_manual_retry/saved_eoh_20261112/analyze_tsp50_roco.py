"""核验 TSP50 六次研发及测试来源，生成运行级描述性比较。"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


RECOVERY = Path(__file__).resolve().parent
ROOT = RECOVERY.parents[2]
EXPERIMENT = "tsp50-roco-20261114"
ANALYSIS = ROOT / "outputs/p1_tsp50_roco_scale_20261112_135601_c2bfdab0/analysis"
EVIDENCE = ROOT / "vault/Results/Evidence" / EXPERIMENT
REPORT = ROOT / "vault/Results/Reports" / f"{EXPERIMENT}.md"
CONFIG = ROOT / "outputs/p1_tsp50_roco_scale_20261112_135601_c2bfdab0/test.json"
MATRIX_HASH = "a169698d296bd79094f11488741fca47f1ff67574e3c90501c6ae7789fde3d71"
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial import distance_matrix
from scripts import p1_test_evaluate as test
from scripts.p0_runtime import digest
from scripts.p1_runtime import DATA_CONTRACT
from scripts.p1_token_usage import summarize as summarize_tokens


RUNS = [
    ("proposed_solution_quality_and_speed_agents", 1, "20261113_143912_4468d76e_resume",
     "outputs/p1_control/20261113_151011_1e444253_resume", "20261114_001043_141c347d", 421, 370),
    ("proposed_solution_quality_and_speed_agents", 2, "20261112_211734_d7c5ee74_resume",
     "outputs/p1_control/20261112_211734_d7c5ee74_resume", "20261114_002016_c2e731dc", 420, 372),
    ("proposed_solution_quality_and_speed_agents", 3, "20261113_182734_19e8e233_resume",
     "outputs/p1_control/20261113_182734_19e8e233_resume", "20261114_002938_b4d6ed89", 419, 371),
    ("roco_paper_adapted", 1, "20261112_183109_ae37c237_resume",
     "outputs/p1_roco/20261112_183109_ae37c237_resume", "20261114_101348_23cee5d4_resume", 420, 368),
    ("roco_paper_adapted", 2, "20261113_134209_41bdbcb7_resume",
     "outputs/p1_roco/20261113_134209_41bdbcb7_resume", "20261114_100728_2fe0d630", 420, 358),
    ("roco_paper_adapted", 3, "20261113_223251_4e4a7f6f_resume",
     "outputs/p1_roco/20261113_235648_34d12148_resume", "20261114_101742_0bd83178", 417, 362),
]
EXTRAS = {
    ("proposed_solution_quality_and_speed_agents", 1): [
        "outputs/p1_manual_retry/fusion_2_20261112_143551_66d3a6ad/events.jsonl",
        "outputs/p1_manual_retry/fusion_2_20261112_143551_66d3a6ad/evaluation_events.jsonl",
        "outputs/p1_manual_retry/round_3_quality_20261112_145256_3c8fe086/events.jsonl",
        "outputs/p1_manual_retry/round_3_quality_20261112_145256_3c8fe086/evaluation_events.jsonl",
        "outputs/p1_manual_retry/saved_eoh_20261112/20261112_230934_proposed_9dcc64a8_evaluation/evaluation_events.jsonl",
    ],
    ("roco_paper_adapted", 1): [
        "outputs/p1_manual_retry/roco_generation2_saved_20261112/20261112_175444_2229cb9a_evaluation/evaluation_events.jsonl",
        "outputs/p1_manual_retry/roco_generation2_saved_20261112/20261112_181024_8854108b_evaluation/evaluation_events.jsonl",
    ],
    ("roco_paper_adapted", 2): [
        "outputs/p1_manual_retry/roco_2_quota_saved_20261112/20261112_221401_cedfe544_evaluation/evaluation_events.jsonl",
        "outputs/p1_manual_retry/saved_eoh_20261112/20261112_231034_roco_5d06e3ec_evaluation/evaluation_events.jsonl",
        "outputs/p1_manual_retry/saved_eoh_20261112/20261112_234107_roco_605caa9c_evaluation/evaluation_events.jsonl",
    ],
}
FIELDS = ["method", "replicate", "selection_seed", "candidate_id", "test_instances", "route_length_mean",
          "average_solve_seconds", "average_heuristic_seconds", "test_active_wall_seconds", "test_mode",
          "model_calls", "training_evaluations", "training_passed", "training_candidate_errors",
          "training_infrastructure_errors", "training_eval_cpu_seconds", "test_cpu_seconds",
          "observed_input_tokens", "observed_output_tokens", "calls_without_usage"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def events(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_events(path):
    return [json.loads(x[len(test.PREFIX):]) for x in path.read_text(encoding="utf-8").splitlines()
            if x.startswith(test.PREFIX)]


def training_logs(terminal):
    directory, seen, paths = terminal.resolve(), set(), []
    while True:
        require(directory not in seen and directory.is_relative_to((ROOT / "outputs").resolve()), "研发断点链成环或越界。")
        seen.add(directory)
        path = directory / "events.jsonl"
        records = events(path)
        started = [x for x in records if x.get("event") == "formal_run_started"]
        require(len(started) == 1, "研发启动记录不唯一。")
        paths.append(path)
        if not started[0].get("resume_of"):
            return list(reversed(paths))
        parent = Path(started[0]["resume_of"]).resolve()
        parent_hash = started[0].get("interrupted_events_sha256") or started[0].get("original_events_sha256")
        require(parent_hash == digest(parent / "events.jsonl"), "研发断点哈希不符。")
        directory = parent


def audit():
    config = read_json(CONFIG)
    contract = read_json(DATA_CONTRACT)
    require(config["matrix_contract_sha256"] == MATRIX_HASH, "测试矩阵身份不符。")
    entry = next(x for x in contract["tasks"]["tsp_aco"]["test"] if x["size"] == 50)
    dataset = ROOT / entry["path"]
    require(digest(dataset) == entry["sha256"], "测试数据哈希不符。")
    data = np.load(dataset, allow_pickle=False)
    require(data.shape == (64, 50, 2) and np.isfinite(data).all(), "测试数据无效。")
    sources = {CONFIG.resolve(), DATA_CONTRACT.resolve(), dataset.resolve(), Path(__file__).resolve()}
    rows, environments = [], set()
    valid_test_paths = set()
    for method, replicate, frozen_name, terminal_name, test_name, calls_expected, eval_expected in RUNS:
        frozen = ROOT / "outputs/p1_frozen_candidates" / method / frozen_name
        manifest_path, candidate_path = frozen / "manifest.json", frozen / "candidate.py"
        manifest = read_json(manifest_path)
        require(manifest["method"] == method and manifest["formal_replicate"] == replicate
                and manifest["matrix_contract_sha256"] == MATRIX_HASH and digest(candidate_path) == manifest["candidate_sha256"],
                f"冻结候选身份或哈希不符：{method}/{replicate}")
        source_run = Path(manifest["source_run"])
        candidate_root = source_run / "candidates" / manifest["source_candidate_id"]
        for field, path in (("source_config_sha256", source_run / "config.json"),
                            ("source_events_sha256", source_run / "events.jsonl"),
                            ("source_result_sha256", candidate_root / "result.json"),
                            ("source_stdout_sha256", candidate_root / "stdout.txt")):
            require(digest(path) == manifest[field], f"候选来源哈希不符：{path}")
            sources.add(path.resolve())
        if manifest.get("training_completion_run"):
            complete = Path(manifest["training_completion_run"]) / "events.jsonl"
            require(digest(complete) == manifest["training_completion_events_sha256"], "研发终点哈希不符。")
            sources.add(complete.resolve())
        if manifest.get("source_registration_events"):
            path = Path(manifest["source_registration_events"])
            require(digest(path) == manifest["source_registration_events_sha256"], "注册来源哈希不符。")
            sources.add(path.resolve())
        if manifest.get("selection_evaluation_origin_run"):
            directory = Path(manifest["selection_evaluation_origin_run"])
            origin = directory / "candidates" / manifest["selection_evaluation_origin"]
            for field, path in (("selection_evaluation_origin_events_sha256", directory / "events.jsonl"),
                                ("selection_evaluation_origin_result_sha256", origin / "result.json"),
                                ("selection_evaluation_origin_stdout_sha256", origin / "stdout.txt")):
                require(digest(path) == manifest[field], "去重评价来源哈希不符。")
                sources.add(path.resolve())
            require((origin / "problems/tsp_aco/gpt.py").read_text(encoding="utf-8")
                    == candidate_path.read_text(encoding="utf-8"), "去重候选代码不一致。")
        sources.update((manifest_path.resolve(), candidate_path.resolve()))
        run_paths = training_logs(ROOT / terminal_name)
        extra_paths = [ROOT / x for x in EXTRAS.get((method, replicate), [])]
        logs = [events(p) for p in [*run_paths, *extra_paths]]
        all_events = [x for log in logs for x in log]
        calls = sum(x.get("event") == "model_attempt" for x in all_events)
        evals = [x for x in all_events if x.get("event") == "evaluation"]
        ids = [x["candidate_id"] for x in evals]
        statuses = Counter(x["result"]["status"] for x in evals)
        require(calls == calls_expected and len(evals) == eval_expected and len(ids) == len(set(ids))
                and set(statuses) <= {"passed", "candidate_error", "infrastructure_error"}, "研发成本或评价身份不符。")
        cpu_events = [x for x in all_events if x.get("event") == "evaluation_cpu_final"]
        require(len(cpu_events) == len(evals) and {x["candidate_id"] for x in cpu_events} == set(ids)
                and all(isinstance(x["accounting"]["cpu_seconds"], (int, float))
                        and x["accounting"]["cpu_seconds"] >= 0 for x in cpu_events), "研发评价 CPU 记录不符。")
        training_cpu = math.fsum(x["accounting"]["cpu_seconds"] for x in cpu_events)
        tokens = summarize_tokens(logs)
        require(tokens["model_attempt_events"] == calls, "模型调用计数不符。")
        if method == "proposed_solution_quality_and_speed_agents":
            registrations = [x for x in all_events if x.get("event") == "candidate_registered"]
            by_id = {x["candidate_id"]: x for x in registrations}
            require(len(by_id) == len(registrations), "候选注册重复。")
            selected_id = manifest["source_candidate_id"]
            require(selected_id in by_id and by_id[selected_id]["generation"] == 6, "最终候选来源不符。")
            ancestors, queue = set(), [selected_id]
            while queue:
                ident = queue.pop()
                if ident in ancestors:
                    continue
                require(ident in by_id, f"父代登记缺失：{ident}")
                ancestors.add(ident)
                queue.extend(by_id[ident].get("parents", []))
            speed = {i for i in ancestors if by_id[i].get("source_role") == "speed_agent"}
            if replicate == 1:
                require(by_id[selected_id]["stage"] == "eoh_M2_06" and by_id[selected_id]["parents"] == ["c0302"]
                        and not speed, "第 1 次最终候选来源不符。")
            elif replicate == 2:
                chain = ["c0335", "c0289", "c0243", "c0181", "c0118", "c0082", "c0080"]
                require(by_id[selected_id]["stage"] == "eoh_E2_05"
                        and all(b in by_id[a]["parents"] for a, b in zip(chain, chain[1:]))
                        and by_id["c0082"]["stage"] == "fusion_final" and "c0080" in speed,
                        "第 2 次速度分支来源不符。")
            else:
                require(by_id[selected_id]["stage"] == "eoh_E2_04" and not speed
                        and {"c0319", "c0085"} <= ancestors
                        and all(by_id[i]["source_role"] == "integrator" for i in ("c0319", "c0085")),
                        "第 3 次最终候选来源不符。")
        sources.update(p.resolve() for p in [*run_paths, *extra_paths])
        test_root = ROOT / "outputs/p1_test" / method / test_name
        result_path = test_root / "result.json"
        result = read_json(result_path)
        require(result["status"] == "passed" and result["method"] == method
                and result["formal_replicate"] == replicate and result["candidate_id"] == manifest["source_candidate_id"]
                and result["candidate_sha256"] == manifest["candidate_sha256"]
                and result["matrix_contract_sha256"] == MATRIX_HASH, "正式测试身份或状态不符。")
        for path, expected in result["protected_sha256"].items():
            require(Path(path).is_file() and digest(Path(path)) == expected, f"测试受保护文件变化：{path}")
        if result.get("resumed_from"):
            original = Path(result["resumed_from"])
            require(replicate == 1 and method == "roco_paper_adapted"
                    and digest(original / "stdout.txt") == result["source_stdout_sha256"]
                    and digest(original / "result.json") == result["source_result_sha256"]
                    and digest(test_root / "stdout.txt") == result["resume_stdout_sha256"]
                    and digest(test_root / "combined_stdout.txt") == result["combined_stdout_sha256"]
                    and read_json(original / "result.json")["status"] == "timeout", "分段测试来源不符。")
            combined = test_events(test_root / "combined_stdout.txt")
            before = test_events(original / "stdout.txt")
            after = test_events(test_root / "stdout.txt")
            require(len(before) == 15 and len(after) == 50 and combined[:-1] == before + after
                    and result["retained_instance_count"] == 14 and result["resumed_instance_indexes"] == [14, 63]
                    and result["elapsed_seconds"] <= 900, "分段实例或运行预算不符。")
            stdout_path = test_root / "combined_stdout.txt"
            sources.update((original / "stdout.txt", original / "result.json", test_root / "stdout.txt"))
            mode = "resumed_after_sleep"
        else:
            stdout_path, mode = test_root / "stdout.txt", "continuous"
        output = test.read_complete(stdout_path, entry, manifest, "min", dataset, contract["tasks"]["tsp_aco"]["train"]["ants"])
        for field in ("objectives", "average", "fitness", "solve_wall_seconds", "average_solve_wall_seconds",
                      "heuristic_wall_seconds", "average_heuristic_wall_seconds", "start"):
            require(output[field] == result[field], f"测试结果重读不一致：{method}/{replicate}/{field}")
        data_events = test_events(stdout_path)
        require(len(data_events) == 66, "测试事件数量不符。")
        for i, item in enumerate(data_events[1:-1]):
            tour = item.get("tour")
            require(isinstance(tour, list) and len(tour) == 50 and all(type(x) is int for x in tour)
                    and sorted(tour) == list(range(50)), f"实例 {i} 路径不可行。")
            distances = distance_matrix(data[i], data[i])
            target = float(distances[tour, np.roll(tour, 1)].sum())
            require(math.isclose(item["objective"], target, rel_tol=1e-12, abs_tol=1e-12), f"实例 {i} 目标不符。")
        start = result["start"]
        environments.add((start["numpy"], start["torch"], start["torch_threads"], start["n_ants"],
                          start["n_iterations"], start["seed_base"]))
        valid_test_paths.add(result_path.resolve())
        sources.update((result_path.resolve(), stdout_path.resolve()))
        rows.append({"method": method, "replicate": replicate, "selection_seed": manifest["selection_seed"],
                     "candidate_id": manifest["source_candidate_id"], "test_instances": 64,
                     "route_length_mean": result["average"], "average_solve_seconds": result["average_solve_wall_seconds"],
                     "average_heuristic_seconds": result["average_heuristic_wall_seconds"],
                     "test_active_wall_seconds": result["elapsed_seconds"], "test_mode": mode,
                     "model_calls": calls, "training_evaluations": len(evals), "training_passed": statuses["passed"],
                     "training_candidate_errors": statuses["candidate_error"],
                     "training_infrastructure_errors": statuses["infrastructure_error"],
                     "training_eval_cpu_seconds": training_cpu,
                     "test_cpu_seconds": result["process_tree"]["cpu_seconds"] + result.get("previous_process_tree_cpu_seconds", 0),
                     "observed_input_tokens": tokens["observed_tokens"]["input_tokens"],
                     "observed_output_tokens": tokens["observed_tokens"]["output_tokens"],
                     "calls_without_usage": tokens["attempts_without_usage"]})
    require(len(environments) == 1 and len(rows) == 6, "测试软件或矩阵不一致。")
    all_test_paths = set()
    for path in (ROOT / "outputs/p1_test").glob("**/result.json"):
        result = read_json(path)
        if result.get("matrix_contract_sha256") == MATRIX_HASH and result.get("task") == "tsp_aco" and result.get("size") == 50:
            all_test_paths.add(path.resolve())
    failed = (ROOT / "outputs/p1_test/roco_paper_adapted/20261114_003857_23054d62/result.json").resolve()
    require(all_test_paths == valid_test_paths | {failed}, "存在未知测试、遗漏或重复结果。")
    require({(x["method"], x["replicate"]) for x in rows} == {(m, i) for m in
            ("proposed_solution_quality_and_speed_agents", "roco_paper_adapted") for i in (1, 2, 3)}, "比较矩阵不完整。")
    return rows, sorted(sources, key=str), environments.pop()


def render(rows):
    proposed = [x for x in rows if x["method"] == "proposed_solution_quality_and_speed_agents"]
    roco = [x for x in rows if x["method"] == "roco_paper_adapted"]
    avg = lambda items, key: statistics.mean(x[key] for x in items)
    sd = lambda items, key: statistics.stdev(x[key] for x in items)
    p, r = avg(proposed, "route_length_mean"), avg(roco, "route_length_mean")
    ps, rs = sd(proposed, "route_length_mean"), sd(roco, "route_length_mean")
    pt, rt = avg(proposed, "average_solve_seconds"), avg(roco, "average_solve_seconds")
    delta, relative = p - r, (p / r - 1) * 100
    resumed = read_json(ROOT / "outputs/p1_test/roco_paper_adapted/20261114_101348_23cee5d4_resume/result.json")
    prefix_time = statistics.mean(resumed["solve_wall_seconds"][:14])
    suffix_time = statistics.mean(resumed["solve_wall_seconds"][14:])
    each = lambda items, key: "、".join(f"{x[key]:.6f}" for x in items)
    costs = lambda items, key: sum(x[key] for x in items)
    limits = ("仅 TSP50 白盒任务，每法三次研发；不能推广到其他任务，也不能作显著性声明。"
              "本方法首组授权上限为 422 次，其余为 420 次；研发中有网络错误、无效提案和断点恢复。"
              "两种方法的 Critic 输入与反馈不同，不能把差异仅归因于角色分工。"
              "模型为 GPT-5.6 Luna、CLI 默认温度，数据和模型设置与论文不同。"
              "本方法测试主要在休眠前，RoCo 主要在唤醒后；RoCo 首组跨休眠分段，墙钟耗时可能受时段与负载影响，不能据此判定方法速度。")
    analysis = f'''# TSP50 对照分析

## 问题与口径

比较本方法和 RoCo 适配在 TSP50 白盒任务上的最终路线长度。每法三次独立研发，每次冻结一个候选并测试相同 64 个实例。路线长度越低越好；独立统计单位是研发运行，标准差为样本标准差。

## 质量

- 本方法逐次均值：{each(proposed, "route_length_mean")}；总体为 {p:.6f} ± {ps:.6f}。
- RoCo 逐次均值：{each(roco, "route_length_mean")}；总体为 {r:.6f} ± {rs:.6f}。
- 本方法减 RoCo 为 {delta:+.6f}，相对 RoCo 为 {relative:+.3f}%。本批次观察到 RoCo 的平均路线更短；这不是显著性或普遍优越性结论。

## 耗时与开销

- 本方法三次平均单实例求解时间：{each(proposed, "average_solve_seconds")} 秒，运行均值 {pt:.3f} 秒。
- RoCo 三次平均单实例求解时间：{each(roco, "average_solve_seconds")} 秒，运行均值 {rt:.3f} 秒。测试时段和休眠中断构成明显混杂，不能据此宣称速度优势。
- RoCo 首组休眠前 14 个实例平均 {prefix_time:.3f} 秒，唤醒后其余 50 个实例平均 {suffix_time:.3f} 秒。两段实例不同，不能仅凭这一差异确定原因。
- 本方法共 {costs(proposed, "model_calls")} 次模型调用、{costs(proposed, "training_evaluations")} 次训练评价；RoCo 共 {costs(roco, "model_calls")} 次调用、{costs(roco, "training_evaluations")} 次评价。失败及其开销保留。
- 训练评价子进程 CPU 时间合计：本方法 {costs(proposed, "training_eval_cpu_seconds"):.2f} 秒，RoCo {costs(roco, "training_eval_cpu_seconds"):.2f} 秒；不含模型调用进程。
- 六个冻结候选均通过 64 个测试实例的路径与目标核验，共 384 条有效结果。研发阶段的候选错误与基础设施中断分别列于数据表。

## 本方法候选来源

- 第 1 次最终候选 `c0358` 来自第 6 代 EoH `M2`，父代为 `c0302`；两者代码及评价相同。沿登记的父代关系未发现速度智能体候选。
- 第 2 次最终候选 `c0335` 来自第 6 代 EoH `E2`。一条已登记的父代链为 `c0335 → c0289 → c0243 → c0181 → c0118 → c0082 → c0080`；其中 `c0082` 是末轮融合，`c0080` 是第 3 轮速度分支提案。
- 第 3 次最终候选 `c0333` 来自第 6 代 EoH `E2`。其父代链含 Integrator 变异 `c0319` 和 `c0085`，未发现速度智能体候选。

父代链只能说明登记的候选来源。反思等提示信息可能跨角色传递；这些记录不能单独证明速度角色对最终结果的贡献。

## 限制

{limits}

逐运行记录与来源哈希见 [Evidence](../../../vault/Results/Evidence/{EXPERIMENT}/evidence.md)。
'''
    stats = f'''# TSP50 统计附录

- 独立单位：研发运行；每法 n=3。64 个固定测试实例用于计算每次运行的平均路线长度，不作为 64 次独立研发。
- 本方法路线均值 {p:.9f}，样本标准差 {ps:.9f}；RoCo 均值 {r:.9f}，样本标准差 {rs:.9f}。
- 本方法减 RoCo 为 {delta:+.9f}，以 RoCo 为分母的相对差异为 {relative:+.6f}%。
- 不计算显著性检验或置信区间。项目比较合同要求额外推断在观察结果前预登记；当前只有三个独立重复，不支持事后补做推断。
- 模型用量仅为日志中可回读的 token；{costs(proposed, "calls_without_usage")} 次本方法请求及 {costs(roco, "calls_without_usage")} 次 RoCo 请求缺少用量，未知值不按零计入总消耗。
- CPU 时间为评价子进程树实测，不包括模型调用；断点测试的 CPU 时间包括中断前已实际消耗部分。
- RoCo 首组断点前保留 14 个实例，断点后完成 50 个实例；已完成实例未重复。原失败记录保留，分段结果与原始两段逐项一致。
- 六次测试均使用相同数据、种子、30 只蚂蚁、500 次迭代和相同软件版本。完整求解时间包含准备和检查；函数时间单列。
- 未建立同合同的本方法修改前版本作为配对基线；本次只回答用户要求的外部 RoCo 比较，不声明代码修改带来的提升。

{limits}
'''
    evidence = f'''# TSP50 本方法与 RoCo 对照证据

## 研究问题与方法

在固定 TSP50 白盒 ACO 任务和本批训练合同下，比较本方法与 RoCo 适配。每法三次研发、每次六代；每次冻结一个候选，在同一组 64 个测试实例上各求解一次。

## 结果与边界

本方法平均路线长度为 {p:.6f} ± {ps:.6f}，RoCo 为 {r:.6f} ± {rs:.6f}；较低为好。本方法相对 RoCo 高 {relative:.3f}%。这是本批三个研发重复的描述性结果。

六个候选均完成 64 个有效测试实例。[RoCo 第 1 次测试](../../../../outputs/p1_test/roco_paper_adapted/20261114_003857_23054d62/result.json)因系统休眠中断；前 14 个完整实例保留，后 50 个从断点继续。原测试休眠前后的活动时间保守计入 124 秒，续跑最多 776 秒；休眠时长不计入可用运行预算。[恢复结果](../../../../outputs/p1_test/roco_paper_adapted/20261114_101348_23cee5d4_resume/result.json)保存两段来源、合并结果及哈希，不将两段当成独立重复。

{limits}

## 文件

- [运行级数据](data.csv)和[来源清单](manifest.json)记录六次研发及测试来源和哈希。
- [详细分析](../../../../outputs/p1_tsp50_roco_scale_20261112_135601_c2bfdab0/analysis/analysis-report.md)和[统计附录](../../../../outputs/p1_tsp50_roco_scale_20261112_135601_c2bfdab0/analysis/stats-appendix.md)。
- [研发运行记录](../../../../outputs/p1_tsp50_roco_scale_20261112_135601_c2bfdab0/analysis/run-history.md)保留各次失败与断点恢复的历史说明。
- [结果报告](../../Reports/{EXPERIMENT}.md)用于决策；协议与阶段状态见 [P2 阶段计划](../../../Plan/mechanisms/P2-正式外部对照.md)。
'''
    report = f'''---
type: results-report
status: active
evidence_package: Results/Evidence/{EXPERIMENT}
---

# TSP50 本方法与 RoCo 对照

## 研究问题

在本批固定任务和研发预算下，本方法与 RoCo 适配设计的启发式在 TSP50 上表现如何？

## 发现

- 本方法三次研发的平均路线长度为 **{p:.6f} ± {ps:.6f}**；RoCo 为 **{r:.6f} ± {rs:.6f}**。较低更好，本方法在本批中高 {relative:.3f}%。
- 六个冻结候选均通过 64 个测试实例的路径与目标检查；每法独立重复数为 3，结果仅作描述性比较。
- 完整求解时间的运行均值分别为 {pt:.3f} 秒和 {rt:.3f} 秒，但测试跨休眠和不同时段，不能据此判断哪种方法更快。
- 两种方法分别使用 {costs(proposed, "model_calls")} 次和 {costs(roco, "model_calls")} 次模型调用；训练评价 CPU 时间分别为 {costs(proposed, "training_eval_cpu_seconds"):.0f} 秒和 {costs(roco, "training_eval_cpu_seconds"):.0f} 秒，均不含模型调用进程。失败请求的部分 token 用量未知。

## 限制

{limits}

## 决定

本批没有观察到本方法的路线质量优于 RoCo；暂不作统计显著性、速度优势或单独角色机制的结论。

## 下一步

先在受控且连续的同一负载条件下设计重复计时，并在追加研发重复前冻结预算与分析规则。

## 证据

[完整证据包](../Evidence/{EXPERIMENT}/evidence.md)。
'''
    return analysis, stats, evidence, report


def write(rows, sources, environment):
    require(not ANALYSIS.exists() and not EVIDENCE.exists() and not REPORT.exists(), "目标已存在，不能覆盖。")
    analysis, stats, evidence, report = render(rows)
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip())
    commands = []
    for method, _, frozen_name, _, _, _, _ in RUNS:
        commands.append(["python", "-B", "-u", str(ROOT / "scripts/p1_test_evaluate.py"), "--candidate-dir",
                         str(ROOT / "outputs/p1_frozen_candidates" / method / frozen_name), "--config", str(CONFIG)])
    commands.append(["python", "-B", "-u", str(RECOVERY / "resume_roco1_tsp50_test.py"), "--run"])
    commands.append(["python", "-B", str(Path(__file__).resolve()), "--write"])
    ANALYSIS.mkdir(parents=True, exist_ok=False)
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    for path, content in ((ANALYSIS / "analysis-report.md", analysis), (ANALYSIS / "stats-appendix.md", stats),
                          (EVIDENCE / "evidence.md", evidence), (EVIDENCE / "data.csv", table.getvalue()), (REPORT, report)):
        with path.open("x", encoding="utf-8", newline="\n") as out:
            out.write(content)
    manifest = {"experiment_id": EXPERIMENT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_head": head, "git_worktree_dirty": dirty, "matrix_contract_sha256": MATRIX_HASH,
                "unit": "independent_research_run", "methods": 2, "replicates_per_method": 3,
                "test_instances_per_candidate": 64, "numpy_torch_threads_ants_iterations_seed_base": list(environment),
                "commands": commands,
                "source_files": [{"path": str(path), "sha256": digest(path)} for path in sources],
                "canonical_table": {"path": str(EVIDENCE / "data.csv"), "sha256": digest(EVIDENCE / "data.csv")},
                "notes": ["RoCo 首组测试因系统睡眠中断；14 个已完成实例保留，其余 50 个续跑。",
                          "失败请求的 token 用量未知；不作显著性或速度优势声明。"]}
    with (EVIDENCE / "manifest.json").open("x", encoding="utf-8", newline="\n") as out:
        json.dump(manifest, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps({"status": "written", "evidence": str(EVIDENCE), "analysis": str(ANALYSIS),
                      "report": str(REPORT), "sources": len(sources)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows, sources, environment = audit()
    if args.write:
        write(rows, sources, environment)
    else:
        print(json.dumps({"status": "passed", "runs": len(rows), "verified_test_tours": 384,
                          "sources": len(sources), "environment": environment}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"TSP50 比较核验失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
