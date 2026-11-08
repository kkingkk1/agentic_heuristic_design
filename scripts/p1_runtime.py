"""将 P0 的登录模型和进程树预算监督用于已接入的 ACO 任务。"""

import json
from pathlib import Path
import threading

from scripts import p0_model_probe as probe, p0_tsp_aco_evaluate as tsp_evaluator
from scripts import p1_bpp_aco_evaluate, p1_cvrp_aco_evaluate, p1_mkp_aco_evaluate, p1_op_aco_evaluate, p1_tsp_gls_evaluate
from scripts.p0_controller import StopRun, check_modification_scope
from scripts.p0_runtime import Budget, ROOT, Runtime, digest
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
        self.cli, self.allow_unquantified_quota = cli, allow_unquantified_quota
        self.problem_name, self.problem_size = name, contract.train_size
        task_dir = ROOT / "problems" / name
        self.eval_script = task_dir / ("eval_black_box.py" if contract.black_box else "eval.py")
        self.read_complete = module.read_complete
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
