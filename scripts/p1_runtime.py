"""将 P0 的登录模型和进程树预算监督用于已接入的 ACO 任务。"""

import json
import re
from functools import partial
from pathlib import Path
import threading

from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate as tsp_evaluator
from scripts import p1_bpp_aco_evaluate, p1_cvrp_aco_evaluate, p1_mkp_aco_evaluate, p1_op_aco_evaluate, p1_tsp_gls_evaluate
from scripts.p0_controller import StopRun, check_modification_scope, code_hash
from scripts.p0_runtime import Budget, JsonLog, ROOT, Runtime, digest
from scripts.p1_task_contract import load_task_contract


TRAINING = {
    "tsp_aco": (tsp_evaluator, ROOT / "problems/tsp_aco/dataset/train50_dataset.npy",
                "49cfd10c6d21efd39736e5baf57510b4ce6945c3a057cdbe8dcad398c728599a"),
    "op_aco": (p1_op_aco_evaluate, p1_op_aco_evaluate.DATASET, p1_op_aco_evaluate.DATASET_SHA256),
    "cvrp_aco": (p1_cvrp_aco_evaluate, p1_cvrp_aco_evaluate.DATASET, p1_cvrp_aco_evaluate.DATASET_SHA256),
    "mkp_aco": (p1_mkp_aco_evaluate, p1_mkp_aco_evaluate.DATASET, p1_mkp_aco_evaluate.DATASET_SHA256),
    "bpp_offline_aco": (p1_bpp_aco_evaluate, p1_bpp_aco_evaluate.DATASET, p1_bpp_aco_evaluate.DATASET_SHA256),
    "tsp_gls": (p1_tsp_gls_evaluate, p1_tsp_gls_evaluate.DATASET, p1_tsp_gls_evaluate.DATASET_SHA256),
}
TRAINING_PARAMETERS = {
    # instance_count, ants, iterations, dataset_generator_seed
    "tsp_aco": (5, 30, 200, 1234),
    "op_aco": (5, 30, 100, 1234),
    "cvrp_aco": (10, 20, 100, 1234),
    "mkp_aco": (5, 10, 50, 1234),
    "bpp_offline_aco": (5, 20, 100, 5),
    "tsp_gls": (10, None, 1200, 5),
}
DATA_CONTRACT = ROOT / "cfg/p1_data_contract.json"


def _evaluation_evidence(root, result):
    """提取逐实例记录；失败时同时保存尚未归类的输出。"""
    stdout_path, stderr_path = root / "stdout.txt", root / "stderr.txt"
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else None
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else None
    instances, malformed, event_types, start = [], [], [], None
    if stdout is not None:
        for line in stdout.splitlines():
            prefix = next((item for item in ("P0_JSON ", "P1_JSON ") if line.startswith(item)), None)
            if prefix is None:
                continue
            try:
                item = json.loads(line[len(prefix):])
                if not isinstance(item, dict):
                    raise ValueError("评价事件不是对象。")
            except (ValueError, TypeError):
                malformed.append(line)
                continue
            event_types.append(item.get("event"))
            if item.get("event") == "start":
                keys = ("dataset", "seed_base", "time_scope", "heuristic_time_scope")
                start = {key: item[key] for key in keys if key in item}
            elif item.get("event") == "instance":
                keys = ("index", "seed", "objective", "iterations", "solve_wall_seconds",
                        "heuristic_wall_seconds", "stop_reason")
                instances.append({key: item[key] for key in keys if key in item})
    if result and result.get("status") == "passed":
        objectives = result.get("objectives")
        if (stdout is None or malformed or not isinstance(objectives, list)
                or event_types != ["start"] + ["instance"] * len(objectives) + ["complete"]
                or [item.get("index") for item in instances] != list(range(len(objectives)))
                or [item.get("objective") for item in instances] != objectives):
            raise ValueError(f"候选逐实例输出与通过的评价不一致：{root.name}")
        for key in ("solve_wall_seconds", "heuristic_wall_seconds"):
            if key in result and [item.get(key) for item in instances] != result[key]:
                raise ValueError(f"候选逐实例计时与评价不一致：{root.name} {key}")
        result_start = result.get("start", {})
        if (not isinstance(result_start, dict) or start is None
                or any(start.get(key) != result_start[key]
                       for key in ("dataset", "seed_base", "time_scope", "heuristic_time_scope")
                       if key in result_start)
                or (type(start.get("seed_base")) is int and
                    [item.get("seed") for item in instances] !=
                    [start["seed_base"] + index for index in range(len(instances))])
                or ("stopping_reasons" in result and
                    [item.get("stop_reason") for item in instances] != result["stopping_reasons"])):
            raise ValueError(f"候选实例种子、计时范围或停止原因不一致：{root.name}")
    evidence = {"instances": instances}
    if start is not None:
        evidence["evaluation_start"] = start
    if not result or result.get("status") != "passed":
        if stdout is not None:
            evidence["failure_stdout"] = stdout
        if stderr:
            evidence["failure_stderr"] = stderr
        if malformed:
            evidence["malformed_events"] = malformed
    return evidence


def _generation_record(events, run_dir, generation, status, origins, task_name):
    """从一个代次边界内的事件与候选输出建立可核对的记录。"""
    if task_name not in TRAINING:
        raise ValueError("代次归并的任务名称无效。")
    candidates, stages, processes, trace = {}, {}, {}, []
    for event in events:
        if event.get("event") != "candidate_registered":
            continue
        identifier, stage = event.get("candidate_id"), event.get("stage")
        if (not isinstance(identifier, str) or not re.fullmatch(r"c[0-9]{4,}", identifier)
                or identifier in candidates or not isinstance(stage, str) or stage in stages
                or event.get("generation") != generation):
            raise ValueError("候选注册的 ID、阶段或代次无效。")
        keys = ("candidate_id", "sequence", "stage", "source_role", "operator", "generation", "round", "parents")
        candidates[identifier] = {key: event.get(key) for key in keys}
        stages[stage] = identifier

    selection = None
    responses = {event.get("stage") for event in events if event.get("event") == "model_response"}
    proposal_responses = {event.get("stage") for event in events if event.get("event") == "proposal_response"}
    for event in events:
        kind, identifier, stage = event.get("event"), event.get("candidate_id"), event.get("stage")
        if kind == "model_process" and stage in stages:
            proposal = event.get("result", {}).get("proposal") if isinstance(event.get("result"), dict) else None
            if isinstance(proposal, dict):
                processes.setdefault(stage, []).append(proposal)
        if kind == "proposal_response":
            if identifier not in candidates or stage != candidates[identifier]["stage"]:
                raise ValueError("提案响应与候选注册不一致。")
            row = candidates[identifier]
            if "generation_order" in row:
                raise ValueError("候选提案响应重复。")
            row["generation_order"] = event.get("generation_order")
            if "response" in event:
                row["raw_response"] = event["response"]
            else:
                matching = [item for item in processes.get(stage, [])
                            if isinstance(item.get("code"), str)
                            and code_hash(item["code"]) == event.get("code_sha256")
                            and item.get("thought") == event.get("thought")]
                if len(matching) != 1:
                    raise ValueError("结构化提案与候选响应缺失或不一致。")
                row["code_sha256"] = event["code_sha256"]
                row["_proposal"] = matching[0]
        elif kind == "evaluation":
            if identifier not in candidates or "evaluation" in candidates[identifier]:
                raise ValueError("候选评价与注册不一致或重复。")
            candidates[identifier]["evaluation"] = event.get("result")
        elif kind in ("candidate_invalid", "candidate_skipped", "candidate_infrastructure_error"):
            if identifier not in candidates:
                raise ValueError("候选失败记录缺少注册。")
            candidates[identifier].setdefault("issues", []).append(event)
        elif kind == "initial_population":
            if generation != 0 or selection is not None:
                raise ValueError("初始化种群记录无效或重复。")
            selection = {"event_index": event["event_index"], "candidate_ids": event.get("candidate_ids")}
        elif kind == "selection":
            if event.get("generation") != generation or selection is not None:
                raise ValueError("代末筛选记录无效或重复。")
            selection = {"event_index": event["event_index"],
                         "pool": event.get("pool"), "selected": event.get("selected")}

        if kind not in ("candidate_registered", "proposal_response", "evaluation", "candidate_invalid",
                        "candidate_skipped", "candidate_infrastructure_error", "initial_population", "selection",
                        "preflight", "formal_run_started"):
            entry = dict(event)
            if kind == "model_process" and isinstance(entry.get("result"), dict):
                result = dict(entry["result"])
                proposal = result.pop("proposal", None)
                if isinstance(proposal, dict) and isinstance(proposal.get("code"), str):
                    result["proposal_code_sha256"] = code_hash(proposal["code"])
                    # 响应尚未写入时，保留已收到的提案，避免中断丢失信息。
                    if stage not in proposal_responses:
                        result["proposal"] = proposal
                if stage in responses:
                    result.pop("message", None)
                entry["result"] = result
            trace.append(entry)

    if status == "completed" and selection is None:
        raise ValueError("已完成代次缺少筛选结果。")
    selected = None
    selection_origins = {}
    if selection is not None:
        if generation == 0:
            selected = selection.get("candidate_ids")
        else:
            entries = selection.get("selected")
            if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
                raise ValueError("代末入选记录无效。")
            selected = [item.get("candidate_id") for item in entries]
            selection_origins = {item.get("candidate_id"): item.get("evaluation_origin") for item in entries}
        if (not isinstance(selected, list) or len(set(selected)) != len(selected)
                or any(identifier not in candidates for identifier in selected)):
            raise ValueError("入选候选与本代注册不一致。")

    for identifier, row in candidates.items():
        root = run_dir / "candidates" / identifier
        result_path = root / "result.json"
        recorded = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
        result = row.get("evaluation")
        if result is not None and recorded is not None and result != recorded:
            raise ValueError(f"候选评价事件与文件不一致：{identifier}")
        if result is None and recorded is not None:
            result = recorded
            row["evaluation_event_missing"] = True
        if result is not None and not isinstance(result, dict):
            raise ValueError(f"候选评价不是对象：{identifier}")
        if result is not None and result.get("candidate_id") != identifier:
            raise ValueError(f"候选评价的 ID 不一致：{identifier}")
        if result and result.get("status") == "passed" and recorded is None:
            raise ValueError(f"通过的候选缺少评价文件：{identifier}")
        if result is not None:
            omitted = {"command", "resources"}
            if result.get("status") == "passed":
                # 逐实例值及计时范围由下面的实例记录保存，无需在评价摘要中重复。
                omitted.update(("objectives", "solve_wall_seconds", "heuristic_wall_seconds", "stopping_reasons", "start"))
            row["evaluation"] = {key: value for key, value in result.items() if key not in omitted}
        row.update(_evaluation_evidence(root, result))
        row["selected"] = None if selected is None else identifier in selected
        issues = {issue["event"] for issue in row.get("issues", [])}
        if "candidate_invalid" in issues:
            row["status"], row["valid"] = "invalid_proposal", False
        elif result and result.get("status") in ("candidate_error", "timeout"):
            row["status"], row["valid"] = result["status"], False
        elif "candidate_skipped" in issues:
            row["status"], row["valid"] = "skipped", None
        elif issues or (result and result.get("status") == "infrastructure_error"):
            row["status"], row["valid"] = "infrastructure_error", None
        elif result and result.get("status") == "passed":
            row["status"], row["valid"] = "evaluation_passed", True if selection is not None else None
        else:
            row["status"], row["valid"] = "incomplete", None
        proposal = row.pop("_proposal", None)
        if proposal is None and "generation_order" not in row and processes.get(row["stage"]):
            # 进程结束但控制器尚未记录提案时，保留原文及未完成状态。
            proposal = processes[row["stage"]][-1]
            if isinstance(proposal.get("code"), str):
                row["code_sha256"] = code_hash(proposal["code"])
        passed = bool(result and result.get("status") == "passed" and not row.get("issues"))
        if passed:
            source = root / "problems" / task_name / "gpt.py"
            if (proposal is None or not isinstance(proposal.get("code"), str) or not source.is_file()
                    or source.read_text(encoding="utf-8") != proposal["code"] + "\n"):
                raise ValueError(f"通过评价的候选代码与提案不一致：{identifier}")
        # 入选与协作相关候选保留完整提案；未完成状态保守保留原文。
        keep_code = (row["selected"] or row["valid"] is not True or
                     row.get("operator") in ("startup", "continuation", "round_fusion", "elite_fusion"))
        if proposal is not None and keep_code:
            row["thought"], row["code"] = proposal.get("thought"), proposal.get("code")
        if passed and row.get("code_sha256"):
            completed = result.get("completed_monotonic_ns")
            key = row["code_sha256"]
            if type(completed) is not int:
                origins[key] = None
            elif key not in origins:
                origins[key] = (completed, row["sequence"], identifier)
            elif origins[key] is not None:
                origins[key] = min(origins[key], (completed, row["sequence"], identifier))

    if selected is not None:
        for identifier in selected:
            row = candidates[identifier]
            if row["valid"] is not True or "code" not in row:
                raise ValueError(f"入选候选没有完整有效的提案及评价：{identifier}")
            origin = origins.get(row.get("code_sha256"))
            if generation and origin is not None and selection_origins.get(identifier) != origin[2]:
                raise ValueError(f"入选候选的去重评价来源不一致：{identifier}")
            row["evaluation_origin"] = (selection_origins.get(identifier) if generation else
                                        (origin[2] if origin is not None else None))
    return {"schema_version": 1, "generation": generation, "status": status,
            "source_event_range": [events[0]["event_index"], events[-1]["event_index"]],
            "selection": selection, "candidates": list(candidates.values()), "trace": trace}


class TrainingLog(JsonLog):
    """直接记录可追溯的提案和反馈，避免重复保存完整提示和成功进程输出。"""

    def __init__(self, path, contract):
        super().__init__(path)
        self.contract = contract
        self.pending_prompts = {}
        self.prompt_lock = threading.Lock()
        self.generation_offset = 0
        self.next_generation = 0
        self.evaluation_origins = {}
        self.consolidation_failed = False

    def consolidate(self, status):
        """只创建当前代的记录，不覆盖或移除运行现场。"""
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.generation_offset)
                data = stream.read()
                next_offset = stream.tell()
            if not data or not data.endswith(b"\n"):
                raise ValueError("代次事件缺失或末行不完整。")
            events = [json.loads(line) for line in data.decode("utf-8").splitlines()]
            record = _generation_record(events, self.path.parent, self.next_generation, status,
                                        self.evaluation_origins, self.contract.name)
            target_dir = self.path.parent / "generation_records"
            target_dir.mkdir(exist_ok=True)
            target = target_dir / f"generation_{self.next_generation:04d}.json"
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            self.generation_offset = next_offset
            self.next_generation += 1
        except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
            self.consolidation_failed = True
            raise StopRun(f"代次归并失败，原始记录保留：{type(exc).__name__}: {exc}") from exc

    def event(self, kind, **fields):
        if kind in ("model_attempt", "model_response", "proposal_response"):
            prompt = fields.pop("prompt", None)
            if kind == "model_attempt":
                if isinstance(prompt, str):
                    fields["prompt_sha256"] = code_hash(prompt)
                with self.prompt_lock:
                    self.pending_prompts[fields.get("stage")] = prompt
        if kind == "model_process":
            result = fields.get("result")
            if isinstance(result, dict):
                try:
                    message, usage, event_types, warnings = probe.extract_message(result)
                except (probe.ProbeError, TypeError, ValueError, AttributeError, KeyError):
                    with self.prompt_lock:
                        fields["failed_prompt"] = self.pending_prompts.get(fields.get("stage"))
                else:
                    compact = {key: value for key, value in result.items() if key != "stdout"}
                    compact.update(usage=usage, event_types=event_types, warnings=warnings)
                    try:
                        compact["proposal"] = self.contract.parse_proposal(message)
                    except (probe.ProbeError, TypeError, ValueError, SyntaxError):
                        compact["message"] = message
                    fields["result"] = compact
                    if warnings:
                        with self.prompt_lock:
                            fields["failed_prompt"] = self.pending_prompts.get(fields.get("stage"))
        elif kind == "proposal_response":
            response = fields.get("response")
            try:
                proposal = self.contract.parse_proposal(response)
            except (probe.ProbeError, TypeError, ValueError, SyntaxError):
                pass  # 无法解析的提案保留原响应和后续错误记录。
            else:
                fields.pop("response", None)
                fields.update(thought=proposal["thought"], code_sha256=code_hash(proposal["code"]))
            with self.prompt_lock:
                self.pending_prompts.pop(fields.get("stage"), None)
        elif kind == "invalid_role_response":
            with self.prompt_lock:
                fields["failed_prompt"] = self.pending_prompts.get(fields.get("stage"))
        super().event(kind, **fields)
        if not getattr(self, "consolidation_failed", True):
            # 这些事件在批次工作线程全部结束后由控制器写入。
            if kind == "initial_population" or (kind == "history_appended" and fields.get("role") in ("integrator", "Integrator")):
                self.consolidate("completed")
            elif kind == "finish" and fields.get("status") == "stopped":
                self.consolidate("stopped")


class TaskRuntime(Runtime):
    """评价使用各任务原有入口；工程配置只允许调用评价，不允许请求模型。"""

    def __init__(self, run_dir, config_path, log, *, cli=None, allow_unquantified_quota=False):
        self.run_dir = Path(run_dir).resolve()
        config_path = Path(config_path).resolve()
        if config_path != self.run_dir / "config.json":
            raise ValueError("运行配置必须保存在独立运行目录的 config.json。")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("status") not in ("frozen", "engineering"):
            raise ValueError("运行配置状态必须为 frozen 或 engineering。")
        task = config.get("task")
        if not isinstance(task, dict):
            raise ValueError("运行配置缺少任务合同。")
        name, condition = task.get("name"), task.get("prompt_condition")
        if (name not in TRAINING or condition not in ("white_box", "black_box")
                or (name == "tsp_gls" and condition != "white_box")):
            raise ValueError("任务名称或白盒／黑盒条件无效。")
        mode = task.get("evaluation_mode", "final_quality")
        stagnation_rounds = task.get("stagnation_rounds")
        if (mode not in ("final_quality", "quality_per_time")
                or (mode == "final_quality" and stagnation_rounds is not None)
                or (mode == "quality_per_time" and (name != "tsp_aco"
                    or type(stagnation_rounds) is not int or stagnation_rounds < 1))):
            raise ValueError("评价设置或停滞轮数无效；速度设置目前只支持 TSP-ACO。")
        contract = load_task_contract(name, black_box=condition == "black_box")
        module, dataset, expected_hash = TRAINING[name]
        expected_timeout = module.TIMEOUT_SECONDS
        instance_count, ants, iterations, generator_seed = TRAINING_PARAMETERS[name]
        signature = contract.signature.removeprefix("def ").removesuffix(":")
        data_contract = json.loads(DATA_CONTRACT.read_text(encoding="utf-8"))
        frozen_train = data_contract.get("tasks", {}).get(name, {}).get("train", {})
        if (data_contract.get("status") != "frozen" or data_contract.get("contract_id") != "p1_data_20261104_v1"
                or frozen_train.get("path") != dataset.relative_to(ROOT).as_posix()
                or frozen_train.get("sha256") != expected_hash or frozen_train.get("instances") != instance_count
                or frozen_train.get("size") != contract.train_size or frozen_train.get("ants") != ants
                or frozen_train.get("iterations") != iterations or frozen_train.get("generator_seed") != generator_seed
                or frozen_train.get("evaluation_seeds") != [20261103 + i for i in range(instance_count)]):
            raise ValueError("训练数据与冻结的 P1 数据合同不一致。")
        if (task.get("problem_size") != contract.train_size
                or task.get("objective_direction") != contract.direction
                or task.get("function") != signature
                or task.get("instance_count") != instance_count
                or task.get("ants") != ants
                or task.get("iterations_per_instance") != iterations
                or task.get("dataset_generator_seed") != generator_seed
                or task.get("evaluation_seeds") != [20261103 + i for i in range(instance_count)]
                or (name == "tsp_gls" and task.get("perturbation_moves") != 30)
                or task.get("whole_training_wall_timeout_seconds") != expected_timeout
                or task.get("dataset_sha256") != expected_hash
                or Path(task.get("dataset", "")).as_posix() != dataset.relative_to(ROOT).as_posix()
                or config.get("model", {}).get("name") != probe.MODEL):
            raise ValueError("运行配置与任务、数据、时限或模型合同不一致。")
        if not dataset.is_file() or digest(dataset) != expected_hash:
            raise StopRun("训练数据的 SHA-256 与任务合同不一致。")

        self.cfg, self.log, self.contract = config, log, contract
        # 旧参数仅用于兼容调用方，不再参与启动或代次检查。
        self.cli = cli
        self.problem_name, self.problem_size = name, contract.train_size
        task_dir = ROOT / "problems" / name
        self.eval_script = task_dir / ("eval_black_box.py" if contract.black_box else "eval.py")
        self.read_complete = (partial(module.read_complete, stagnation_rounds=stagnation_rounds)
                              if mode == "quality_per_time" else module.read_complete)
        self.additional_candidate_faults = ("候选签名", "候选必须定义", "启发式转换后含非有限值")
        self.budget = Budget(config, log)
        self.model_slots = threading.BoundedSemaphore(config["resources"]["max_concurrent_model_calls"])
        self.eval_slots = threading.BoundedSemaphore(config["resources"]["max_concurrent_evaluations"])
        self.cancelled = threading.Event()
        stem = name + ("_black_box" if contract.black_box else "")
        prompt_dir = ROOT / "prompts" / stem
        solver = task_dir / ("gls.py" if name == "tsp_gls" else "aco.py")
        self.protected = [config_path, DATA_CONTRACT, dataset, solver, task_dir / "eval.py",
                          ROOT / "cfg" / "problem" / f"{stem}.yaml",
                          prompt_dir / "func_desc.txt", prompt_dir / "func_signature.txt",
                          Path(module.__file__).resolve(),
                          *(ROOT / "scripts" / filename for filename in
                            ("p0_controller.py", "p0_runtime.py", "p0_windows_job.py", "p0_model_probe.py",
                             "p0_budget_model_probe.py", "p0_budget_quota.py", "p1_task_contract.py",
                             "p1_runtime.py", "p1_run_generation.py"))]
        if name == "tsp_gls":
            self.protected.append(task_dir / "lkh_reference.py")
        else:
            self.protected.append(task_dir / "eval_black_box.py")
        if config.get("matrix_contract_sha256"):
            from scripts.p1_formal_matrix import MATRIX_PATH, implementation_paths, test_cells
            self.protected.extend((MATRIX_PATH, *implementation_paths()))
            for frozen_task, _, size in test_cells(data_contract):
                entry = next(row for row in data_contract["tasks"][frozen_task]["test"] if row["size"] == size)
                self.protected.append(ROOT / entry["path"])
            self.protected = list(dict.fromkeys(self.protected))
        self.protected_hashes = {str(path): digest(path) for path in self.protected}

    def model(self, stage, prompt):
        if self.cfg["status"] != "frozen":
            raise StopRun("工程配置禁止请求模型；正式运行须先冻结完整合同。")
        return super().model(stage, prompt)

    def evaluate(self, candidate):
        check_modification_scope(candidate.code)
        return super().evaluate(candidate)
