"""论文适配 EoH 的代次预算门禁；模型与任务评价沿用 P1 后端。"""

from pathlib import Path

from scripts.p0_controller import StopRun
from scripts.p0_runtime import ROOT, digest
from scripts.p1_runtime import TaskRuntime


class EoHRuntime(TaskRuntime):
    def __init__(self, run_dir, config_path, log, *, cli=None, allow_unquantified_quota=False):
        super().__init__(run_dir, config_path, log, cli=cli, allow_unquantified_quota=allow_unquantified_quota)
        if self.cfg.get("method", {}).get("id") != "eoh_paper_adapted":
            raise ValueError("运行配置不是已标识的 EoH 论文适配版本。")
        extra = [Path(__file__).resolve(), ROOT / "baselines/eoh/paper_controller.py",
                 ROOT / "baselines/eoh/LICENSE.txt", ROOT / "scripts/p1_run_eoh.py"]
        self.protected.extend(extra)
        self.protected_hashes.update({str(path): digest(path) for path in extra})

    def before_generation(self):
        budget = self.budget
        with budget.lock:
            settings = self.cfg["budget"]
            required = 5 * self.cfg["task"]["population_size"]
            remaining_eval_retries = settings["additional_infrastructure_evaluation_starts"] - budget.eval_retries
            needed_evaluations = required + remaining_eval_retries
            if settings["normal_model_calls"] - budget.model_normal < required:
                raise StopRun("EoH 剩余模型调用不足以覆盖一代。")
            if (settings["normal_code_generation_slots"] - budget.eval_normal < required
                    or settings["max_initialization_and_generation_evaluation_starts"] - budget.eval_starts < needed_evaluations):
                raise StopRun("EoH 剩余评价启动次数不足以覆盖一代及重试预留。")
            if not budget.eval_cpu_samples:
                raise StopRun("没有本运行的评价 CPU 样本，无法估计 EoH 完整代开销。")
            remaining_cpu = budget.cpu_limit - budget.used_cpu()
            empirical = max(budget.eval_cpu_samples) * needed_evaluations
            self.log.event("generation_cpu_budget_check", remaining_cpu_seconds=remaining_cpu,
                           evaluation_starts_to_cover=needed_evaluations,
                           largest_initialization_sample_seconds=max(budget.eval_cpu_samples),
                           empirical_estimate_seconds=empirical,
                           note="实测估算不是后续候选耗时上界。")
            if remaining_cpu < max(budget.cores * (budget.timeout + 2), empirical):
                raise StopRun("评价 CPU 预算不足以覆盖 EoH 下一代经验估算。")
