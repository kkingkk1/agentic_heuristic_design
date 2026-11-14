"""P0 的实际模型和评价接入；预算失败时停止，不改变冻结配置。"""

from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from scripts import p0_model_probe as probe
from scripts import p0_tsp_aco_evaluate as evaluator
from scripts.p0_budget_model_probe import RETRY_CONFIG
from scripts.p0_controller import StopRun
from scripts.p0_windows_job import WindowsJob


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "cfg" / "p0_tsp_aco.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generation_requirements(population_size: int, *, rounds: int = 3) -> tuple[int, int]:
    """按实际协作轮数计算一代的正常模型调用和候选评价槽位。"""
    if type(population_size) is not int or population_size < 2:
        raise ValueError("种群至少需要两个不同候选。")
    if type(rounds) is not int or rounds < 1:
        raise ValueError("协作轮数必须为正整数。")
    return 4 * population_size + 13 + 4 * rounds, 4 * population_size + 9 + 3 * rounds


def quota_gate(quota, required, *, allow_unquantified=False):
    """确认订阅允许普通调用；旧参数仅为兼容既有入口。"""
    if quota.get("ordinary_usage_allowed") is not True:
        raise StopRun("订阅服务未确认当前允许普通调用。")
    return "ordinary_usage_allowed"


class JsonLog:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.sequence = 0
        with path.open("x", encoding="utf-8"):
            pass

    def event(self, kind, **fields):
        with self.lock, self.path.open("a", encoding="utf-8", newline="\n") as stream:
            self.sequence += 1
            item = {"event_index": self.sequence, "event": kind, "at_utc": utc_now(), **fields}
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()


class Budget:
    """并发调用在启动前原子计数；评价 CPU 包含作业内已退出的后代。"""

    def __init__(self, config, log):
        self.cfg, self.log = config, log
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.model_normal = self.model_retries = self.eval_starts = 0
        self.eval_normal = self.eval_retries = 0
        self.cpu_finished = 0.0
        self.eval_cpu_samples = []
        self.jobs = {}
        self.proposals = {}
        self.cpu_limit = config["budget"]["future_initialization_and_generation_eval_process_tree_cpu_seconds"]
        self.cores = max(os.cpu_count() or 1, 28)
        self.timeout = config["task"]["whole_training_wall_timeout_seconds"]

    def model_start(self, retry=False):
        with self.lock:
            budget = self.cfg["budget"]
            if retry:
                if self.model_retries >= budget["additional_infrastructure_retry_calls"]:
                    raise StopRun("模型基础设施重试预算耗尽。")
                self.model_retries += 1
            else:
                if self.model_normal >= budget["normal_model_calls"]:
                    raise StopRun("正常模型调用预算耗尽。")
                self.model_normal += 1
            if self.model_normal + self.model_retries > budget["max_initialization_and_generation_model_cli_invocations"]:
                raise StopRun("模型 CLI 总调用预算耗尽。")
            self.log.event("model_budget_reserved", normal=self.model_normal, retries=self.model_retries)

    def used_cpu(self):
        with self.lock:
            return self.cpu_finished + sum(job.accounting()["cpu_seconds"] for job in self.jobs.values())

    def reserve_proposal(self, identifier, cancelled):
        """在模型请求前为每个并发提案预留一次完整评价。"""
        with self.condition:
            if identifier in self.proposals:
                raise StopRun(f"提案预算已预留：{identifier}")
            while True:
                if cancelled.is_set():
                    raise StopRun("运行已停止，取消等待预算。")
                unstarted = sum(not started for started in self.proposals.values())
                budget = self.cfg["budget"]
                if (self.eval_normal + unstarted >= budget["normal_code_generation_slots"]
                        or self.eval_starts + unstarted >= budget["max_initialization_and_generation_evaluation_starts"]):
                    raise StopRun("没有可为下一提案预留的评价槽位。")
                each = self.cores * (self.timeout + 2)
                used = self.used_cpu()
                if used + (len(self.proposals) + 1) * each <= self.cpu_limit:
                    self.proposals[identifier] = False
                    self.log.event("proposal_budget_reserved", candidate_id=identifier,
                                   outstanding=len(self.proposals), per_proposal_cpu_seconds=each,
                                   used_cpu_seconds=used)
                    return
                if not self.proposals:
                    raise StopRun("CPU 余量不足以为下一提案预留完整评价。")
                self.condition.wait(timeout=0.1)

    def release_proposal(self, identifier):
        with self.condition:
            if identifier not in self.proposals:
                raise StopRun(f"提案预算未预留：{identifier}")
            del self.proposals[identifier]
            self.log.event("proposal_budget_released", candidate_id=identifier,
                           outstanding=len(self.proposals))
            self.condition.notify_all()

    def begin_evaluation(self, identifier, job, *, retry=False):
        with self.lock:
            if identifier not in self.proposals or self.proposals[identifier]:
                raise StopRun("评价启动前缺少该提案的预算预留。")
            budget = self.cfg["budget"]
            if self.eval_starts >= budget["max_initialization_and_generation_evaluation_starts"]:
                raise StopRun("评价启动预算耗尽。")
            if retry and self.eval_retries >= budget["additional_infrastructure_evaluation_starts"]:
                raise StopRun("评价基础设施重验预算耗尽。")
            if not retry and self.eval_normal >= budget["normal_code_generation_slots"]:
                raise StopRun("正常评价槽位预算耗尽。")
            used = self.used_cpu()
            # 同一主机的所有作业共享逻辑 CPU。预留完整 60 秒、全部逻辑 CPU 和两秒监督余量。
            reservation = self.cores * (self.timeout + 2)
            if used + reservation > self.cpu_limit:
                raise StopRun(f"评价进程树 CPU 余量不足：已用 {used:.3f}，需预留 {reservation} 秒。")
            self.eval_starts += 1
            self.proposals[identifier] = True
            if retry:
                self.eval_retries += 1
            else:
                self.eval_normal += 1
            self.jobs[identifier] = job
            self.log.event("evaluation_budget_reserved", candidate_id=identifier, starts=self.eval_starts,
                           normal=self.eval_normal, retries=self.eval_retries, used_cpu_seconds=used,
                           shared_future_cpu_reservation_seconds=reservation)

    def check_cpu(self):
        with self.lock:
            used = self.used_cpu()
            if used >= self.cpu_limit - self.cores * 2:
                for job in self.jobs.values():
                    job.terminate()
                raise StopRun(f"评价进程树 CPU 接近上限，已终止作业：{used:.3f} 秒。")
            return used

    def finish_evaluation(self, identifier):
        with self.lock:
            job = self.jobs.pop(identifier)
            accounting = job.accounting()
            self.cpu_finished += accounting["cpu_seconds"]
            self.eval_cpu_samples.append(accounting["cpu_seconds"])
            self.log.event("evaluation_cpu_final", candidate_id=identifier, accounting=accounting,
                           total_cpu_seconds=self.cpu_finished)
            return accounting

    def before_generation(self):
        with self.lock:
            budget = self.cfg["budget"]
            model_slots, evaluation_slots = generation_requirements(
                self.cfg["task"]["population_size"], rounds=self.cfg["task"].get("collaboration_rounds", 3))
            if budget["normal_model_calls"] - self.model_normal < model_slots:
                raise StopRun("剩余正常模型调用不足以覆盖一代。")
            remaining_retries = budget["additional_infrastructure_evaluation_starts"] - self.eval_retries
            needed_evaluations = evaluation_slots + remaining_retries
            if (budget["normal_code_generation_slots"] - self.eval_normal < evaluation_slots
                    or budget["max_initialization_and_generation_evaluation_starts"] - self.eval_starts < needed_evaluations):
                raise StopRun("剩余评价启动次数不足以覆盖一代及基础设施重验预留。")
            remaining_cpu = self.cpu_limit - self.used_cpu()
            if not self.eval_cpu_samples:
                raise StopRun("没有本运行的评价进程树 CPU 样本，无法估计完整代开销。")
            empirical_estimate = max(self.eval_cpu_samples) * needed_evaluations
            self.log.event("generation_cpu_budget_check", remaining_cpu_seconds=remaining_cpu,
                           evaluation_starts_to_cover=needed_evaluations,
                           largest_initialization_sample_seconds=max(self.eval_cpu_samples),
                           empirical_estimate_seconds=empirical_estimate,
                           note="以本运行最大实测值估算，并非后续候选耗时的上界。")
            if remaining_cpu < max(self.cores * (self.timeout + 2), empirical_estimate):
                raise StopRun("评价 CPU 预算不足以覆盖下一次完整评价或本代经验估算。")


def memory_and_priority():
    if os.name != "nt":
        raise StopRun("实际运行要求 Windows。")

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_uint64), ("avail_phys", ctypes.c_uint64),
                    ("total_page", ctypes.c_uint64), ("avail_page", ctypes.c_uint64),
                    ("total_virtual", ctypes.c_uint64), ("avail_virtual", ctypes.c_uint64),
                    ("avail_extended", ctypes.c_uint64)]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel.GlobalMemoryStatusEx.restype = ctypes.c_int
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetPriorityClass.argtypes = [ctypes.c_void_p]
    kernel.GetPriorityClass.restype = ctypes.c_uint32
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise StopRun("无法检查可用内存。")
    priority = kernel.GetPriorityClass(kernel.GetCurrentProcess())
    if priority != 0x20:
        raise StopRun(f"控制进程不是 Normal 优先级：{priority}。")
    return {"available_memory_bytes": status.avail_phys, "priority": "Normal"}


class Runtime:
    def __init__(self, run_dir, config, log, *, cli=None, allow_unquantified_quota=False):
        self.run_dir, self.cfg, self.log = Path(run_dir), config, log
        self.cli = cli
        self.allow_unquantified_quota = allow_unquantified_quota
        self.problem_name, self.problem_size = "tsp_aco", 50
        self.eval_script, self.read_complete = evaluator.EVAL, evaluator.read_complete
        self.additional_candidate_faults = ()
        self.budget = Budget(config, log)
        self.model_slots = threading.BoundedSemaphore(config["resources"]["max_concurrent_model_calls"])
        self.eval_slots = threading.BoundedSemaphore(config["resources"]["max_concurrent_evaluations"])
        self.cancelled = threading.Event()
        self.protected = [ROOT / "problems/tsp_aco/eval.py", ROOT / "problems/tsp_aco/aco.py",
                          ROOT / "problems/tsp_aco/dataset/train50_dataset.npy", CONFIG,
                          ROOT / "cfg/problem/tsp_aco.yaml", ROOT / "cfg/p0_tsp_aco_pending.json",
                          *(ROOT / "scripts" / name for name in
                            ("p0_controller.py", "p0_runtime.py", "p0_windows_job.py",
                             "p0_run_generation.py", "p0_tsp_aco_evaluate.py", "p0_model_probe.py",
                             "p0_budget_model_probe.py", "p0_budget_quota.py"))]
        self.protected_hashes = {str(path): digest(path) for path in self.protected}
        dataset = self.protected[2]
        if digest(dataset) != config["task"]["dataset_sha256"]:
            raise StopRun("训练数据的 SHA-256 与冻结配置不一致。")

    def assert_integrity(self):
        for path in self.protected:
            if not path.is_file() or digest(path) != self.protected_hashes[str(path)]:
                self.cancelled.set()
                raise StopRun(f"冻结文件已改变，评价结果不可信：{path}")

    def before_generation(self):
        self.budget.before_generation()
        from scripts import p0_budget_quota
        try:
            quota = p0_budget_quota.read_limits()
        except p0_budget_quota.QuotaProbeError as exc:
            raise StopRun(f"无法回读订阅额度：{exc}") from exc
        self.log.event("generation_quota", quota=quota)
        model_slots, _ = generation_requirements(
            self.cfg["task"]["population_size"], rounds=self.cfg["task"].get("collaboration_rounds", 3))
        required = model_slots + (self.cfg["budget"]["additional_infrastructure_retry_calls"] - self.budget.model_retries)
        mode = quota_gate(quota, required, allow_unquantified=self.allow_unquantified_quota)
        self.log.event("generation_quota_gate", mode=mode, required_model_calls=required)

    def prepare_proposal(self, candidate):
        self.budget.reserve_proposal(candidate.candidate_id, self.cancelled)

    def release_proposal(self, candidate):
        self.budget.release_proposal(candidate.candidate_id)

    def model(self, stage, prompt):
        with self.model_slots:
            retry = False
            while True:
                if self.cancelled.is_set():
                    raise StopRun("运行已停止。")
                self.budget.model_start(retry=retry)
                if self.cli is None:
                    self.cli = probe.cli_command()
                command = self.cli + ["exec", "--ignore-user-config", "--skip-git-repo-check",
                                      "--json", "--sandbox", "read-only", "--model", probe.MODEL]
                command += probe.HTTPS_CONFIG + RETRY_CONFIG + ["--cd", str(self.run_dir), "-"]
                self.log.event("model_attempt", stage=stage, retry=retry, prompt=prompt, command=command,
                               underlying_transport_attempts=None)
                result = probe.run_command(command, prompt, cwd=self.run_dir)
                self.log.event("model_process", stage=stage, retry=retry, result=result,
                               missing_failed_request_usage=True)
                try:
                    message, usage, event_types, warnings = probe.extract_message(result)
                    self.log.event("model_usage", stage=stage, usage=usage, event_types=event_types,
                                   warnings=warnings)
                    if warnings:
                        raise StopRun("CLI 出现内部重试或传输回退，调用次数不能按冻结预算核实。")
                    return message
                except probe.ProbeError as exc:
                    raw = " ".join(str(result.get(key, "")) for key in ("stdout", "stderr", "error"))
                    if re.search(r"rate.limit|quota|usage limit|429", raw, re.I):
                        self.cancelled.set()
                        raise StopRun("模型额度或限流失败；未自动重置。") from exc
                    if re.search(r"connection (?:failed|refused|reset)|request timed out|error sending request", raw, re.I):
                        if self.cfg["budget"]["additional_infrastructure_retry_calls"] == 0:
                            self.log.event("infrastructure_retry_unavailable", stage=stage, reason=str(exc))
                            self.cancelled.set()
                            raise StopRun("模型连接失败；冻结合同不允许自动重试。") from exc
                        self.log.event("infrastructure_retry", stage=stage, reason=str(exc))
                        retry = True
                        continue
                    self.cancelled.set()
                    raise StopRun(f"模型调用失败或事件不可信：{exc}") from exc

    def evaluate(self, candidate):
        with self.eval_slots:
            if self.cancelled.is_set():
                raise StopRun("运行已停止。")
            self.assert_integrity()
            resources = memory_and_priority()
            if resources["available_memory_bytes"] < self.cfg["resources"]["min_available_system_memory_bytes"]:
                raise StopRun("可用内存低于冻结阈值。")
            root = self.run_dir / "candidates" / candidate.candidate_id
            module_dir = root / "problems" / self.problem_name
            module_dir.mkdir(parents=True, exist_ok=False)
            candidate_file = module_dir / "gpt.py"
            candidate_file.write_text(candidate.code + "\n", encoding="utf-8")
            candidate_digest = digest(candidate_file)
            out_path, err_path = root / "stdout.txt", root / "stderr.txt"
            env = os.environ.copy()
            env.update(self.cfg["resources"]["thread_environment"])
            env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPATH": os.pathsep.join(self.cfg["resources"]["pythonpath"])})
            command = [self.cfg["resources"]["python_executable"], "-B", "-u",
                       str(self.eval_script), str(self.problem_size), str(root), "train"]
            if self.cfg["task"].get("evaluation_mode") == "quality_per_time":
                command.extend(["--stagnation-rounds", str(self.cfg["task"]["stagnation_rounds"])])
            start = time.monotonic()
            result = {"status": "infrastructure_error", "command": command, "resources": resources,
                      "started_at_utc": utc_now(), "candidate_id": candidate.candidate_id}
            with WindowsJob() as job:
                reserved = False
                try:
                    self.budget.begin_evaluation(candidate.candidate_id, job)
                    reserved = True
                    with out_path.open("x", encoding="utf-8") as out, err_path.open("x", encoding="utf-8") as err:
                        process = job.start(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                            stdout=out, stderr=err)
                        result["pid"] = process.pid
                        timeout = self.cfg["task"]["whole_training_wall_timeout_seconds"]
                        while True:
                            usage = job.accounting()
                            if usage["active_processes"] == 0:
                                break
                            if self.cancelled.is_set():
                                job.terminate()
                                result["status"] = "infrastructure_error"
                                result["error"] = "其他并发任务停止运行。"
                                break
                            self.budget.check_cpu()
                            if time.monotonic() - start >= timeout:
                                job.terminate()
                                result["status"] = "timeout"
                                break
                            time.sleep(0.05)
                        deadline = time.monotonic() + 10
                        while job.accounting()["active_processes"]:
                            if time.monotonic() >= deadline:
                                raise StopRun("进程树终止后仍有进程存活。")
                            time.sleep(0.05)
                        process.wait(timeout=5)
                        result["exit_code"] = process.returncode
                    self.assert_integrity()
                    if not candidate_file.is_file() or digest(candidate_file) != candidate_digest:
                        result["candidate_modified"] = True
                        if result["status"] != "timeout":
                            result["status"] = "candidate_error"
                        result["error"] = "评价期间候选代码发生修改；修改后的代码需要新 ID。"
                    if result["status"] != "timeout" and "error" not in result:
                        if result["exit_code"] == 0:
                            try:
                                result.update(self.read_complete(out_path))
                                result["status"] = "passed"
                                result["completed_monotonic_ns"] = time.monotonic_ns()
                            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                                result["error"] = f"评价输出不可信：{exc}"
                        else:
                            stderr = err_path.read_text(encoding="utf-8", errors="replace")
                            candidate_faults = ((str(module_dir / "gpt.py"), "启发式必须返回", "启发式矩阵必须是",
                                                 "启发式矩阵转换后含非有限值", "蚂蚁路径不是城市排列", "蚂蚁路径代价无效")
                                                + self.additional_candidate_faults)
                            if any(marker in stderr for marker in candidate_faults):
                                result["status"] = "candidate_error"
                                result["error"] = "候选运行或输出非法；详见 stderr.txt。"
                            else:
                                result["error"] = "评价进程失败且不能归因于候选；详见 stderr.txt。"
                except (OSError, StopRun) as exc:
                    self.cancelled.set()
                    result["error"] = f"{type(exc).__name__}: {exc}"
                    try:
                        job.terminate()
                        deadline = time.monotonic() + 10
                        while job.accounting()["active_processes"]:
                            if time.monotonic() >= deadline:
                                raise OSError("作业终止后仍有进程存活；CPU 记录可能不完整。")
                            time.sleep(0.05)
                    except OSError as term_exc:
                        result["termination_error"] = str(term_exc)
                        result["cpu_accounting_incomplete"] = True
                finally:
                    if reserved:
                        result["process_tree_cpu"] = self.budget.finish_evaluation(candidate.candidate_id)
            result["elapsed_seconds"] = time.monotonic() - start
            with (root / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            return result
