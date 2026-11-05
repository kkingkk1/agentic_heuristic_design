"""P0 运行入口；完整代须先通过只读预算门禁。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import p0_budget_quota, p0_model_probe  # noqa: E402
from scripts.p0_controller import Controller, StopRun  # noqa: E402
from scripts.p0_runtime import CONFIG, JsonLog, Runtime, digest, memory_and_priority, quota_gate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="通过门禁后执行初始化及一代；默认只做启动前检查")
    parser.add_argument("--allow-unquantified-quota", action="store_true",
                        help="仅本次运行接受调用余量不可计数；须由用户明确确认")
    args = parser.parse_args()
    if args.allow_unquantified_quota and not args.run:
        parser.error("--allow-unquantified-quota 只适用于明确启动的本次运行。")
    run_dir = ROOT / "outputs" / "p0_control" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    log = JsonLog(run_dir / "events.jsonl")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with (run_dir / "config.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        resources = memory_and_priority()
        if resources["available_memory_bytes"] < config["resources"]["min_available_system_memory_bytes"]:
            raise StopRun("可用内存低于冻结阈值。")
        cli = p0_model_probe.cli_command()
        version = p0_model_probe.run_command(cli + ["--version"])
        login = p0_model_probe.run_command(cli + ["login", "status"])
        if version.get("exit_code") != 0 or login.get("exit_code") != 0 or "Logged in using ChatGPT" not in (login.get("stdout", "") + login.get("stderr", "")):
            raise StopRun("无法确认 Codex CLI 版本或 ChatGPT 登录。")
        quota = p0_budget_quota.read_limits()
        code_files = [ROOT / "scripts" / name for name in
                      ("p0_controller.py", "p0_runtime.py", "p0_windows_job.py", "p0_run_generation.py",
                       "p0_model_probe.py", "p0_budget_model_probe.py", "p0_budget_quota.py",
                       "p0_tsp_aco_evaluate.py")]
        code_files += [ROOT / "problems/tsp_aco/eval.py", ROOT / "problems/tsp_aco/aco.py"]
        log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
                  quota=quota, code_sha256={str(path): digest(path) for path in code_files},
                  selection_seed=20261103,
                  unquantified_quota_accepted_for_this_run=args.allow_unquantified_quota,
                  context_note="每次模型调用为新会话；本运行无此前角色历史。服务端内部截断不可见，长历史使用前另行核验。")
        required = config["budget"]["max_initialization_and_generation_model_cli_invocations"]
        mode = quota_gate(quota, required, allow_unquantified=args.allow_unquantified_quota)
        log.event("initial_quota_gate", mode=mode, required_model_calls=required)
        if not args.run:
            log.event("finish", status="preflight_passed")
            print(f"启动前检查通过；记录：{run_dir}")
            return 0
        runtime = Runtime(run_dir, config, log, cli=cli,
                          allow_unquantified_quota=args.allow_unquantified_quota)
        controller = Controller(runtime, log, population_size=config["task"]["population_size"],
                                initial_count=config["task"]["initial_proposals"],
                                rounds=config["task"]["collaboration_rounds"])
        controller.run()
        print(f"初始化和一代完成；记录：{run_dir}")
        return 0
    except (StopRun, p0_model_probe.ProbeError, p0_budget_quota.QuotaProbeError, OSError, ValueError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"运行停止：{exc}；记录：{run_dir}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
