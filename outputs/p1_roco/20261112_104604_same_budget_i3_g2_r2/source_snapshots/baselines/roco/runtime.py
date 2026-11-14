"""论文角色适配的 RoCo 运行时与来源保护。"""

from pathlib import Path

from scripts.p0_runtime import ROOT, digest
from scripts.p1_runtime import TaskRuntime


class RoCoRuntime(TaskRuntime):
    def __init__(self, run_dir, config_path, log, *, cli=None, allow_unquantified_quota=False):
        super().__init__(run_dir, config_path, log, cli=cli, allow_unquantified_quota=allow_unquantified_quota)
        if self.cfg.get("method", {}).get("id") != "roco_paper_adapted":
            raise ValueError("运行配置不是已标识的 RoCo 论文适配版本。")
        extra = [Path(__file__).resolve(), ROOT / "baselines/roco/paper_controller.py",
                 ROOT / "scripts/p1_run_roco.py",
                 ROOT / "vault/References/RoCo/2025__Xu__RoCo_Role-Based_LLMs_Collaboration_for_Automatic_Heuristic_Design.pdf"]
        self.protected.extend(extra)
        self.protected_hashes.update({str(path): digest(path) for path in extra})
