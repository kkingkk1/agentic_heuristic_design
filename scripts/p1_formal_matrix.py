"""展开并核验 P1 正式矩阵；默认只检查，不启动模型或测试。"""

import argparse
from contextlib import contextmanager
import copy
import ctypes
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.p0_runtime import digest  # noqa: E402
from scripts.p1_runtime import DATA_CONTRACT, TRAINING, TRAINING_PARAMETERS  # noqa: E402
from scripts.p1_task_contract import load_task_contract  # noqa: E402


MATRIX_PATH = ROOT / "cfg/p1_formal_matrix.json"
RESOURCE_PATH = ROOT / "cfg/p0_tsp_aco.json"
TASKS = ("tsp_aco", "op_aco", "cvrp_aco", "mkp_aco", "bpp_offline_aco", "tsp_gls")
METHODS = {
    "proposed_solution_quality_and_speed_agents": (1, 39, 32),
    "roco_paper_adapted": (1, 39, 32),
}
PILOT_CONDITIONS = [("tsp_aco", "white_box")]
PILOT_SIZES = {"tsp_aco": (50,)}
RUNNERS = {
    "proposed_solution_quality_and_speed_agents": "scripts/p1_run_generation.py",
    "roco_paper_adapted": "scripts/p1_run_roco.py",
    "eoh_paper_adapted": "scripts/p1_run_eoh.py",
    "reevo_2f70042_adapted": "scripts/p1_run_reevo.py",
    "hsevo_fb569d6_adapted": "scripts/p1_run_hsevo.py",
    "mcts_ahd_ee9c4f4_adapted": "scripts/p1_run_mcts_ahd.py",
}
OUTPUT_DIRS = {
    "proposed_solution_quality_and_speed_agents": "p1_control", "roco_paper_adapted": "p1_roco", "eoh_paper_adapted": "p1_eoh",
    "reevo_2f70042_adapted": "p1_reevo", "hsevo_fb569d6_adapted": "p1_hsevo",
    "mcts_ahd_ee9c4f4_adapted": "p1_mcts_ahd",
}
NUMBA_PATH = r"D:\Tools\agentic-heuristic-p1-numba"


def conditions():
    return list(PILOT_CONDITIONS)


def implementation_paths():
    """收集会改变任务、方法、参考或矩阵合同的源码和提示。"""
    scripts = ("p0_controller.py", "p0_runtime.py", "p0_windows_job.py", "p0_model_probe.py",
               "p0_budget_model_probe.py", "p0_budget_quota.py", "p1_runtime.py", "p1_task_contract.py",
               "p1_formal_matrix.py", "p1_freeze_candidate.py", "p1_test_evaluate.py",
               "p1_token_usage.py",
               "p1_reference_test_evaluate.py", "p1_deepaco_test_evaluate.py", "p1_deepaco_worker.py",
               "p1_deepaco_tsp.py", *(Path(RUNNERS[method]).name for method in METHODS))
    paths = {ROOT / "scripts" / name for name in scripts}
    for task in {task for task, _ in conditions()}:
        evaluator, _, _ = TRAINING[task]
        paths.add(Path(evaluator.__file__).resolve())
        directory = ROOT / "problems" / task
        paths.add(directory / "eval.py")
        paths.add(directory / ("gls.py" if task == "tsp_gls" else "aco.py"))
        paths.add(directory / ("lkh_reference.py" if task == "tsp_gls" else "eval_black_box.py"))
    for task, condition in conditions():
        stem = task + ("_black_box" if condition == "black_box" else "")
        paths.update((ROOT / "cfg/problem" / f"{stem}.yaml",
                      ROOT / "prompts" / stem / "func_desc.txt",
                      ROOT / "prompts" / stem / "func_signature.txt"))
    for directory in ("roco", "aco_reference"):
        for path in (ROOT / "baselines" / directory).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".txt") and "__pycache__" not in path.parts:
                paths.add(path)
    paths.add(ROOT / "baselines/deepaco/reference.py")
    return sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix())


def implementation_digest():
    result = hashlib.sha256()
    paths = implementation_paths()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        result.update(len(relative).to_bytes(4, "big"))
        result.update(relative)
        result.update(hashlib.sha256(path.read_bytes()).digest())
    return result.hexdigest(), len(paths)


def verify_training_software(matrix):
    resource = json.loads(RESOURCE_PATH.read_text(encoding="utf-8"))["resources"]
    if Path(sys.executable).resolve() != Path(resource["python_executable"]).resolve():
        raise ValueError("P1 研发必须使用资源合同中的 Python。")
    import numpy as np
    import scipy
    import torch
    import yaml
    available = {}
    for distribution in metadata.distributions(path=resource["pythonpath"]):
        name = distribution.metadata.get("Name", "").lower().replace("_", "-")
        available.setdefault(name, distribution.version)
    actual = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
              "torch": str(torch.__version__), "torch_cuda": torch.version.cuda or "none",
              "pyyaml": yaml.__version__}
    if actual != matrix.get("training_software_versions"):
        raise ValueError("P1 研发的软件版本与冻结合同不一致。")
    for name in ("numpy", "scipy", "torch", "pyyaml"):
        if available.get(name) != actual[name]:
            raise ValueError(f"P1 父进程与评价子进程的软件版本不一致：{name}")


def load_matrix(*, allow_pending=False):
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    expected_status = ("frozen", "pending") if allow_pending else ("frozen",)
    if (not isinstance(matrix, dict) or matrix.get("status") not in expected_status
            or matrix.get("contract_id") != "p1_tsp50_small_20261107"
            or matrix.get("data_contract_sha256") != digest(DATA_CONTRACT)
            or matrix.get("resource_config_sha256") != digest(RESOURCE_PATH)):
        raise ValueError("P1 矩阵或所依赖的数据、资源合同未冻结或已变化。")
    data = json.loads(DATA_CONTRACT.read_text(encoding="utf-8"))
    if data.get("status") != "frozen" or set(data.get("tasks", {})) != set(TASKS):
        raise ValueError("P1 数据合同的任务集合无效。")
    seeds = matrix.get("selection_seeds")
    fixed_integers = {"initial_proposals": 6, "population_size": 2, "collaboration_rounds": 3,
                      "research_repeats": 2, "solver_repeats_per_instance": 1,
                      "normal_and_retry_evaluation_limit": 32}
    if (matrix.get("conditions") != [{"task": "tsp_aco", "prompt_condition": "white_box", "test_sizes": [50]}]
            or matrix.get("method_ids") != list(METHODS)
            or any(type(matrix.get(key)) is not int or matrix[key] != value
                   for key, value in fixed_integers.items())
            or not isinstance(seeds, list) or len(seeds) != 2 or len(set(seeds)) != 2
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("P1 重复数、选择种子或评价上限无效。")
    if (type(matrix.get("model_retry_limit")) is not int or matrix["model_retry_limit"] != 0
            or type(matrix.get("evaluation_retry_reservation")) is not int
            or matrix["evaluation_retry_reservation"] != 0
            or type(matrix.get("training_process_tree_cpu_seconds")) is not int
            or matrix["training_process_tree_cpu_seconds"] <= 0
            or type(matrix.get("model_cli_limit_per_research")) is not int
            or matrix["model_cli_limit_per_research"] != 39
            or type(matrix.get("max_concurrent_research_runs")) is not int
            or matrix["max_concurrent_research_runs"] != 1):
        raise ValueError("P1 研发预算无效。")
    token = matrix.get("token_limits", {})
    if (not isinstance(token, dict) or token.get("input_including_cached", 0) is not None
            or token.get("output", 0) is not None
            or matrix.get("comparison", {}).get("primary_family_size") != 1
            or type(matrix.get("test_process_tree_cpu_multiplier")) is not int
            or matrix["test_process_tree_cpu_multiplier"] < 1):
        raise ValueError("P1 token、比较家族或测试 CPU 合同无效。")
    actual_hash, _ = implementation_digest()
    if not allow_pending and matrix.get("implementation_sha256") != actual_hash:
        raise ValueError("P1 矩阵冻结后的实现或提示发生变化。")
    rows = [row for row in data["tasks"]["tsp_aco"]["test"] if row.get("size") == 50]
    if (len(rows) != 1 or rows[0].get("instances") != 64
            or matrix.get("test_wall_seconds") != {"tsp_aco": {"50": 900}}
            or type(matrix["test_wall_seconds"]["tsp_aco"]["50"]) is not int):
        raise ValueError("P1 测试规模或预算无效：tsp_aco")
    return matrix, data


def run_cells(matrix):
    return [(task, condition, method, replicate)
            for task, condition in conditions()
            for method in METHODS
            for replicate in range(1, matrix["research_repeats"] + 1)]


def test_cells(data):
    return [(task, condition, row["size"])
            for task, condition in conditions()
            for row in data["tasks"][task]["test"] if row["size"] in PILOT_SIZES[task]]


def _matrix_hash():
    return digest(MATRIX_PATH)


def speed_stagnation(matrix, task, mode):
    if mode == "final_quality":
        return None
    if mode == "quality_per_time" and task == "tsp_aco":
        raise ValueError("速度设置尚未单独冻结停滞轮数与预算，目前只允许工程验证。")
    raise ValueError("评价设置或任务无效。")


def training_config(matrix, task, condition, method, replicate, *, evaluation_mode="final_quality"):
    if (task, condition) not in conditions() or method not in METHODS:
        raise ValueError("任务、提示条件或方法不在 P1 矩阵中。")
    if type(replicate) is not int or not 1 <= replicate <= matrix["research_repeats"]:
        raise ValueError("研发重复索引不在 P1 矩阵中。")
    source = json.loads(RESOURCE_PATH.read_text(encoding="utf-8"))
    contract = load_task_contract(task, black_box=condition == "black_box")
    evaluator, dataset, dataset_hash = TRAINING[task]
    count, ants, iterations, generator_seed = TRAINING_PARAMETERS[task]
    generations, calls, evaluations = METHODS[method]
    resources = copy.deepcopy(source["resources"])
    if task == "tsp_gls" and NUMBA_PATH not in resources["pythonpath"]:
        resources["pythonpath"].append(NUMBA_PATH)
    task_config = {
        "name": task, "prompt_condition": condition, "problem_size": contract.train_size,
        "objective_direction": contract.direction,
        "function": contract.signature.removeprefix("def ").removesuffix(":"),
        "instance_count": count, "ants": ants, "iterations_per_instance": iterations,
        "dataset_generator_seed": generator_seed,
        "evaluation_seeds": [20261103 + index for index in range(count)],
        "dataset": dataset.relative_to(ROOT).as_posix(), "dataset_sha256": dataset_hash,
        "whole_training_wall_timeout_seconds": evaluator.TIMEOUT_SECONDS,
        "selection_seed": matrix["selection_seeds"][replicate - 1], "population_size": matrix["population_size"],
        "reference_seed_in_prompt_or_population": method in ("reevo_2f70042_adapted", "hsevo_fb569d6_adapted"),
    }
    speed_stagnation(matrix, task, evaluation_mode)
    if task == "tsp_gls":
        task_config["perturbation_moves"] = 30
    if method == "mcts_ahd_ee9c4f4_adapted":
        task_config.update(initial_proposals=10 if condition == "black_box" else 4,
                           max_evaluations=matrix["normal_and_retry_evaluation_limit"])
    else:
        task_config.update(generations=generations, initial_proposals=matrix["initial_proposals"])
    if method in ("proposed_solution_quality_and_speed_agents", "roco_paper_adapted"):
        task_config["collaboration_rounds"] = matrix["collaboration_rounds"]
    if method in ("reevo_2f70042_adapted", "hsevo_fb569d6_adapted"):
        task_config["mutation_count"] = 5
    if method == "hsevo_fb569d6_adapted":
        task_config.update(reference_seed_source="reevo_2f70042_task_adapted", hm_size=5,
                           hs_iterations=5, hmcr=0.7, par=0.5, bandwidth=0.2)
    model_retries = matrix["model_retry_limit"]
    eval_retries = min(matrix["evaluation_retry_reservation"],
                       matrix["normal_and_retry_evaluation_limit"] - evaluations)
    if eval_retries < 0 or calls + model_retries > matrix["model_cli_limit_per_research"]:
        raise ValueError("方法的结构槽位超过统一预算。")
    budget = {
        "normal_model_calls": calls, "additional_infrastructure_retry_calls": model_retries,
        "max_initialization_and_generation_model_cli_invocations": calls + model_retries,
        "normal_code_generation_slots": evaluations, "additional_infrastructure_evaluation_starts": eval_retries,
        "max_initialization_and_generation_evaluation_starts": evaluations + eval_retries,
        "future_initialization_and_generation_eval_process_tree_cpu_seconds":
            matrix["training_process_tree_cpu_seconds"],
    }
    return {"status": "frozen", "matrix_contract_sha256": _matrix_hash(),
            "formal_replicate": replicate, "method": {"id": method}, "task": task_config,
            "model": copy.deepcopy(source["model"]), "resources": resources, "budget": budget}


def test_config(matrix, task, condition, size, *, evaluation_mode="final_quality"):
    if ((task, condition) not in conditions() or type(size) is not int
            or size not in PILOT_SIZES[task]):
        raise ValueError("测试任务、提示条件或规模无效。")
    timeout = matrix.get("test_wall_seconds", {}).get(task, {}).get(str(size))
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("测试规模没有冻结的预算。")
    config = {"status": "frozen", "matrix_contract_sha256": _matrix_hash(), "task": task,
              "prompt_condition": condition, "size": size, "data_contract_sha256": digest(DATA_CONTRACT),
              "solver_repeats_per_instance": matrix["solver_repeats_per_instance"],
              "whole_dataset_wall_timeout_seconds": timeout,
              "process_tree_cpu_seconds": timeout * matrix["test_process_tree_cpu_multiplier"]}
    speed_stagnation(matrix, task, evaluation_mode)
    return config


def verify_training_config(config):
    matrix, _ = load_matrix()
    verify_training_software(matrix)
    if not isinstance(config, dict):
        raise ValueError("P1 研发配置不是对象。")
    task, method = config.get("task", {}), config.get("method", {})
    if not isinstance(task, dict) or not isinstance(method, dict):
        raise ValueError("P1 研发任务或方法无效。")
    expected = training_config(matrix, task.get("name"), task.get("prompt_condition"),
                               method.get("id"), config.get("formal_replicate"),
                               evaluation_mode=task.get("evaluation_mode", "final_quality"))
    if config != expected:
        raise ValueError("研发配置与冻结的 P1 矩阵、重复索引或预算不一致。")
    return matrix


def run_identity(config):
    return (config["task"]["name"], config["task"]["prompt_condition"],
            config["method"]["id"], config["formal_replicate"],
            config["task"].get("evaluation_mode", "final_quality"))


def previously_started(config, current_dir):
    root = ROOT / "outputs" / OUTPUT_DIRS[config["method"]["id"]]
    for path in root.glob("*/config.json"):
        if path.parent.resolve() == Path(current_dir).resolve():
            continue
        previous = json.loads(path.read_text(encoding="utf-8"))
        if (previous.get("matrix_contract_sha256") != config["matrix_contract_sha256"]
                or run_identity(previous) != run_identity(config)):
            continue
        events_path = path.parent / "events.jsonl"
        with events_path.open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("event") == "formal_run_started":
                    return path.parent
                if event.get("event") in ("finish", "preflight_or_run_stopped"):
                    break
    return None


@contextmanager
def formal_run_guard(config, run_dir, log):
    """按矩阵串行启动研发，避免同一重复索引被静默重跑。"""
    if os.name != "nt":
        raise ValueError("P1 正式研发的进程互斥要求 Windows。")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.ReleaseMutex.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    handle = kernel.CreateMutexW(None, False, r"Local\agentic_heuristic_p1_formal")
    if not handle:
        raise OSError(ctypes.get_last_error(), "无法创建 P1 研发互斥锁。")
    acquired = False
    try:
        result = kernel.WaitForSingleObject(handle, 0)
        if result == 0x102:
            raise ValueError("已有 P1 正式研发运行；本矩阵要求串行启动。")
        if result not in (0, 0x80):
            raise OSError(ctypes.get_last_error(), "无法取得 P1 研发互斥锁。")
        acquired = True
        verify_training_config(config)
        previous = previously_started(config, run_dir)
        if previous is not None:
            raise ValueError(f"该 P1 研发重复已启动，原记录：{previous}")
        log.event("formal_run_started", matrix_contract_sha256=config["matrix_contract_sha256"],
                  task=config["task"]["name"], prompt_condition=config["task"]["prompt_condition"],
                  method=config["method"]["id"], formal_replicate=config["formal_replicate"],
                  selection_seed=config["task"]["selection_seed"])
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def verify_test_config(config):
    matrix, _ = load_matrix()
    if not isinstance(config, dict):
        raise ValueError("P1 测试配置不是对象。")
    expected = test_config(matrix, config.get("task"), config.get("prompt_condition"), config.get("size"),
                           evaluation_mode=config.get("evaluation_mode", "final_quality"))
    if config != expected:
        raise ValueError("测试配置与冻结的 P1 矩阵或预算不一致。")
    return matrix


def reference_cells(data):
    return [(kind, task, size) for task, _, size in test_cells(data) for kind in ("expert", "deepaco")]


def reference_config(matrix, data, kind, task, size):
    if (kind, task, size) not in reference_cells(data) or type(size) is not int:
        raise ValueError("参考方法、任务或规模不在 P1 矩阵中。")
    resource = json.loads(RESOURCE_PATH.read_text(encoding="utf-8"))["resources"]
    entry = next(row for row in data["tasks"][task]["test"] if row["size"] == size)
    common = {"status": "frozen", "matrix_contract_sha256": _matrix_hash(), "task": task, "size": size,
              "data_contract_sha256": digest(DATA_CONTRACT), "resource_config_sha256": digest(RESOURCE_PATH),
              "solver_repeats_per_instance": 1}
    if task == "tsp_gls":
        from problems.tsp_gls.lkh_reference import load_reference
        common["lkh_reference_sha256"] = load_reference(size, entry["sha256"])["sha256"]
    if kind == "expert":
        from scripts import p1_reference_test_evaluate as reference
        timeout = matrix["test_wall_seconds"][task][str(size)]
        common.update(method=reference.method_for(task), source_commit=reference.SOURCE_COMMIT,
                      compute_dtype="float64", implementation_sha256=reference.implementation_snapshot(task),
                      software_versions=reference.software_versions(task, resource),
                      whole_dataset_wall_timeout_seconds=timeout,
                      process_tree_cpu_seconds=timeout * matrix["test_process_tree_cpu_multiplier"])
    elif kind == "deepaco":
        from baselines.deepaco.reference import CHECKPOINT_BLOBS
        from scripts import p1_deepaco_test_evaluate as reference
        name, method, _ = reference.TASKS[task]
        checkpoint = matrix["deepaco_checkpoint"][task][str(size)]
        _, snapshot = reference.reference_implementation_snapshot(task, checkpoint)
        timeout = matrix["test_wall_seconds"][task][str(size)]
        common.update(method=method, source_commit=reference.SOURCE_COMMIT, checkpoint=checkpoint,
                      checkpoint_git_blob=CHECKPOINT_BLOBS[f"pretrained/{name}/{checkpoint}.pt"],
                      checkpoint_scale_mode="matched" if int(checkpoint.removeprefix(name)) == size else "cross_scale",
                      k_sparse=matrix["deepaco_k_sparse"] if name in ("tsp", "op") else None,
                      implementation_sha256=snapshot, software_versions=reference.software_versions(name),
                      whole_dataset_wall_timeout_seconds=timeout,
                      process_tree_cpu_seconds=timeout * matrix["test_process_tree_cpu_multiplier"])
    elif kind == "neuopt":
        from scripts import p1_neuopt_test_evaluate as reference
        policy = matrix["neuopt"]
        if not reference.torch.cuda.is_available() or reference.torch.cuda.get_device_name(0) != policy["gpu_name"]:
            raise ValueError("NeuOpt GPU 与冻结硬件不一致。")
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                    check=True, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("无法检查 NeuOpt GPU 驱动版本。") from exc
        devices = [[part.strip() for part in line.split(",", 1)]
                   for line in result.stdout.splitlines() if line.strip()]
        if [policy["gpu_name"], policy["driver_version_at_freeze"]] not in devices:
            raise ValueError("NeuOpt GPU 驱动版本与冻结合同不一致。")
        timeout = policy["test_wall_seconds"][str(size)]
        common.update(method=reference.METHOD, source_commit=reference.SOURCE_COMMIT, compute_dtype="float64",
                      checkpoint=f"tsp{size}.pt", device=policy["device"], gpu_name=policy["gpu_name"],
                      gpu_driver=policy["driver_version_at_freeze"],
                      search_steps=policy["steps"][str(size)],
                      implementation_sha256=reference.implementation_snapshot(), software_versions=reference.software_versions(),
                      whole_dataset_wall_timeout_seconds=timeout,
                      process_tree_cpu_seconds=timeout * policy["test_process_tree_cpu_multiplier"])
    else:
        from scripts import p1_gnngls_test_evaluate as reference
        policy = matrix["gnngls"]
        checkpoint = policy["checkpoint_and_scaler_size"][str(size)]
        mode = "matched" if checkpoint == size else "cross_scale"
        timeout = policy["test_wall_seconds"][str(size)]
        common.update(method=reference.METHOD, source_commit=reference.SOURCE_COMMIT, compute_dtype="float64",
                      device=policy["device"], checkpoint_size=checkpoint, scaler_size=checkpoint,
                      checkpoint_scale_mode=mode, scaler_scale_mode=mode,
                      search_time_limit_seconds=policy["search_time_limit_seconds"][str(size)],
                      perturbation_moves=policy["perturbation_moves"],
                      implementation_sha256=reference.implementation_snapshot(),
                      software_versions=reference.software_versions(resource),
                      whole_dataset_wall_timeout_seconds=timeout,
                      process_tree_cpu_seconds=timeout * policy["test_process_tree_cpu_multiplier"])
        found = {}
        for distribution in metadata.distributions(path=[str(reference.ISOLATED)]):
            name = distribution.metadata.get("Name", "").lower().replace("_", "-")
            found.setdefault(name, distribution.version)
        if any(found.get(name) != version for name, version in policy["runtime_library_versions"].items()):
            raise ValueError("GNNGLS 隔离环境的运行库版本与冻结合同不一致。")
    versions = matrix["training_software_versions"]
    if kind == "gnngls":
        expected_versions = matrix["gnngls"]["software_versions"]
    else:
        keys = ["python", "numpy", "torch"]
        if kind == "expert" or kind == "deepaco" and task in ("tsp_aco", "cvrp_aco"):
            keys.append("scipy")
        if kind == "expert" and task in ("tsp_gls", "bpp_offline_aco") or kind == "deepaco" and task == "bpp_offline_aco":
            keys.extend(("numba", "llvmlite"))
        if kind == "neuopt":
            keys.append("torch_cuda")
        expected_versions = {key: versions[key] for key in keys}
        if kind == "deepaco":
            expected_versions.update(matrix["deepaco_extra_versions"])
    if common["software_versions"] != expected_versions:
        raise ValueError("P1 参考方法软件版本与冻结合同不一致。")
    return common


def verify_reference_config(config, kind):
    matrix, data = load_matrix()
    verify_training_software(matrix)
    if not isinstance(config, dict):
        raise ValueError("P1 参考配置不是对象。")
    expected = reference_config(matrix, data, kind, config.get("task"), config.get("size"))
    if config != expected:
        raise ValueError("参考配置与冻结的 P1 矩阵、来源或预算不一致。")
    return matrix


def summary(matrix, data):
    cells, tests = run_cells(matrix), test_cells(data)
    model_calls = sum(METHODS[method][1] for _, _, method, _ in cells)
    evaluations = sum(METHODS[method][2] for _, _, method, _ in cells)
    evaluation_retries = sum(min(matrix["evaluation_retry_reservation"],
                                 matrix["normal_and_retry_evaluation_limit"] - METHODS[method][2])
                             for _, _, method, _ in cells)
    candidate_runs = len(tests) * len(METHODS) * matrix["research_repeats"]
    candidate_wall = sum(matrix["test_wall_seconds"][task][str(size)] for task, _, size in tests) * len(METHODS) * matrix["research_repeats"]
    reference_wall = reference_cpu = 0
    for kind, task, size in reference_cells(data):
        if kind in ("expert", "deepaco"):
            timeout = matrix["test_wall_seconds"][task][str(size)]
            multiplier = matrix["test_process_tree_cpu_multiplier"]
        else:
            policy = matrix[kind]
            timeout = policy["test_wall_seconds"][str(size)]
            multiplier = policy["test_process_tree_cpu_multiplier"]
        reference_wall += timeout
        reference_cpu += timeout * multiplier
    return {"conditions": len(conditions()), "methods": len(METHODS), "research_runs": len(cells),
            "test_condition_scales": len(tests), "candidate_test_runs": candidate_runs,
            "candidate_test_instances": candidate_runs * 64,
            "normal_model_calls": model_calls, "max_model_calls_including_retries":
                model_calls + len(cells) * matrix["model_retry_limit"],
            "normal_evaluation_slots": evaluations,
            "max_evaluation_starts_including_retries": evaluations + evaluation_retries,
            "max_training_process_tree_cpu_seconds": len(cells) * matrix["training_process_tree_cpu_seconds"],
            "candidate_test_max_wall_seconds_sum": candidate_wall,
            "candidate_test_max_cpu_seconds_sum": candidate_wall * matrix["test_process_tree_cpu_multiplier"],
            "reference_test_runs": len(reference_cells(data)),
            "reference_test_instances": len(reference_cells(data)) * 64,
            "reference_test_max_wall_seconds_sum": reference_wall,
            "reference_test_max_cpu_seconds_sum": reference_cpu,
            "primary_comparisons": len(tests) * (len(METHODS) - 1),
            "model_token_total_limit": None}


def verify_expansion(matrix, data):
    """复用现有入口校验器，离线核对完整展开结果。"""
    from scripts.p1_freeze_candidate import METHODS as validators
    from scripts import p1_deepaco_test_evaluate, p1_gnngls_test_evaluate, p1_neuopt_test_evaluate
    from scripts import p1_reference_test_evaluate, p1_test_evaluate

    verify_training_software(matrix)
    for task in {task for task, _ in conditions()}:
        item = data["tasks"][task]
        for row in (item["train"], *(row for row in item["test"] if row["size"] in PILOT_SIZES[task])):
            path = (ROOT / row["path"]).resolve()
            if not path.is_relative_to((ROOT / "problems").resolve()) or digest(path) != row["sha256"]:
                raise ValueError(f"P1 实例文件位置或哈希不符：{row['path']}")
    for task, condition, method, replicate in run_cells(matrix):
        validators[method][1](training_config(matrix, task, condition, method, replicate))
    for task, condition, size in test_cells(data):
        config = test_config(matrix, task, condition, size)
        candidate = {"status": "frozen_after_training", "task": task, "prompt_condition": condition}
        p1_test_evaluate.load_test_contract(config, candidate, data)
    for kind, task, size in reference_cells(data):
        config = reference_config(matrix, data, kind, task, size)
        if kind == "expert":
            p1_reference_test_evaluate.load_reference_contract(config, data, digest(DATA_CONTRACT), digest(RESOURCE_PATH))
        elif kind == "deepaco":
            p1_deepaco_test_evaluate.load_reference_contract(config, data, digest(DATA_CONTRACT))
        elif kind == "neuopt":
            p1_neuopt_test_evaluate.load_reference_contract(config, data, digest(DATA_CONTRACT), digest(RESOURCE_PATH))
        else:
            p1_gnngls_test_evaluate.load_reference_contract(config, data, digest(DATA_CONTRACT), digest(RESOURCE_PATH))
    result = summary(matrix, data)
    if result["primary_comparisons"] != matrix["comparison"]["primary_family_size"]:
        raise ValueError("主要比较家族与矩阵展开数量不一致。")
    return {"status": "passed", "formal_experiment_started": False, "model_calls": 0, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("check", help="校验矩阵并打印汇总，不发起实验")
    train = sub.add_parser("train-config", help="生成一个研发运行的配置")
    train.add_argument("--task", required=True, choices=TASKS)
    train.add_argument("--condition", required=True, choices=("white_box", "black_box"))
    train.add_argument("--method", required=True, choices=tuple(METHODS))
    train.add_argument("--replicate", required=True, type=int)
    train.add_argument("--evaluation-mode", choices=("final_quality", "quality_per_time"), default="final_quality")
    test = sub.add_parser("test-config", help="生成一个任务条件和规模的测试配置")
    test.add_argument("--task", required=True, choices=TASKS)
    test.add_argument("--condition", required=True, choices=("white_box", "black_box"))
    test.add_argument("--size", required=True, type=int)
    test.add_argument("--evaluation-mode", choices=("final_quality", "quality_per_time"), default="final_quality")
    reference = sub.add_parser("reference-config", help="生成一个求解参考的测试配置")
    reference.add_argument("--kind", required=True, choices=("expert", "deepaco", "neuopt", "gnngls"))
    reference.add_argument("--task", required=True, choices=TASKS)
    reference.add_argument("--size", required=True, type=int)
    for item in (train, test, reference):
        item.add_argument("--output", type=Path, help="仅新建配置文件；省略时输出 JSON")
    args = parser.parse_args()
    try:
        matrix, data = load_matrix()
        if args.command in (None, "check"):
            result = verify_expansion(matrix, data)
        elif args.command == "train-config":
            result = training_config(matrix, args.task, args.condition, args.method, args.replicate,
                                     evaluation_mode=args.evaluation_mode)
        elif args.command == "test-config":
            result = test_config(matrix, args.task, args.condition, args.size, evaluation_mode=args.evaluation_mode)
        else:
            result = reference_config(matrix, data, args.kind, args.task, args.size)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if getattr(args, "output", None):
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            print(f"配置已新建：{args.output}")
        else:
            print(text, end="")
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"P1 矩阵无效：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
