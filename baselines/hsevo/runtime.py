"""HSEvo 适配版的模型调用、和声评价和 CPU 预算门禁。"""

from pathlib import Path

from baselines.hsevo.controller import source_files
from scripts import p0_budget_quota
from scripts.p0_controller import StopRun
from scripts.p0_runtime import ROOT, digest, quota_gate
from scripts.p1_runtime import TaskRuntime


class HSEvoRuntime(TaskRuntime):
    def __init__(self, run_dir, config_path, log, *, cli=None, allow_unquantified_quota=False):
        super().__init__(run_dir, config_path, log, cli=cli, allow_unquantified_quota=allow_unquantified_quota)
        if self.cfg.get("method", {}).get("id") != "hsevo_fb569d6_adapted":
            raise ValueError("运行配置不是已标识的 HSEvo 来源适配版本。")
        extra = [Path(__file__).resolve(), ROOT / "baselines/hsevo/controller.py",
                 ROOT / "baselines/hsevo/LICENSE.txt", ROOT / "baselines/reevo/controller.py",
                 ROOT / "baselines/reevo/LICENSE.txt", ROOT / "scripts/p1_run_hsevo.py",
                 *source_files(self.contract).values()]
        self.protected.extend(extra)
        self.protected_hashes.update({str(path): digest(path) for path in extra})

    def before_generation(self):
        budget = self.budget
        with budget.lock:
            settings, task = self.cfg["budget"], self.cfg["task"]
            population, mutations = task["population_size"], task["mutation_count"]
            required_calls = population + mutations + 3
            required_evaluations = population + mutations + task["hm_size"] + task["hs_iterations"]
            remaining_retries = settings["additional_infrastructure_evaluation_starts"] - budget.eval_retries
            needed_evaluations = required_evaluations + remaining_retries
            if settings["normal_model_calls"] - budget.model_normal < required_calls:
                raise StopRun("HSEvo 剩余模型调用不足以覆盖反思、交叉、变异与和声提参。")
            if (settings["normal_code_generation_slots"] - budget.eval_normal < required_evaluations
                    or settings["max_initialization_and_generation_evaluation_starts"] - budget.eval_starts < needed_evaluations):
                raise StopRun("HSEvo 剩余评价启动次数不足以覆盖本代和声搜索及重试预留。")
            if not budget.eval_cpu_samples:
                raise StopRun("没有本运行的评价 CPU 样本，无法估计 HSEvo 本代开销。")
            remaining_cpu = budget.cpu_limit - budget.used_cpu()
            empirical = max(budget.eval_cpu_samples) * needed_evaluations
            self.log.event("generation_cpu_budget_check", remaining_cpu_seconds=remaining_cpu,
                           evaluation_starts_to_cover=needed_evaluations,
                           largest_initialization_sample_seconds=max(budget.eval_cpu_samples),
                           empirical_estimate_seconds=empirical,
                           note="实测估算不是后续候选耗时上界。")
            if remaining_cpu < max(budget.cores * (budget.timeout + 2), empirical):
                raise StopRun("评价 CPU 预算不足以覆盖 HSEvo 本代经验估算。")
        try:
            quota = p0_budget_quota.read_limits()
        except p0_budget_quota.QuotaProbeError as exc:
            raise StopRun(f"无法回读订阅额度：{exc}") from exc
        self.log.event("generation_quota", quota=quota)
        required_calls += settings["additional_infrastructure_retry_calls"] - budget.model_retries
        mode = quota_gate(quota, required_calls, allow_unquantified=self.allow_unquantified_quota)
        self.log.event("generation_quota_gate", mode=mode, required_model_calls=required_calls)
