"""RoCo 论文角色适配入口；仅接受已冻结的单运行配置。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.roco.paper_controller import RoCoController  # noqa: E402
from baselines.roco.runtime import RoCoRuntime  # noqa: E402
from scripts import p0_model_probe as probe, p1_run_generation  # noqa: E402
from scripts.p0_controller import StopRun  # noqa: E402
from scripts.p0_runtime import JsonLog, memory_and_priority  # noqa: E402
from scripts.p1_formal_matrix import formal_run_guard, verify_training_config  # noqa: E402


def validate_config(config):
    return p1_run_generation.validate_config(config, method_id="roco_paper_adapted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="已冻结的 RoCo 单运行配置 JSON")
    parser.add_argument("--run", action="store_true", help="通过门禁后执行；默认只做启动前检查")
    parser.add_argument("--allow-unquantified-quota", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        generations, seed = validate_config(config)
        verify_training_config(config)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"配置无效：{exc}", file=sys.stderr)
        return 1
    run_dir = ROOT / "outputs" / "p1_roco" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    log = JsonLog(run_dir / "events.jsonl")
    config_path = run_dir / "config.json"
    with config_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        runtime = RoCoRuntime(run_dir, config_path, log, allow_unquantified_quota=args.allow_unquantified_quota)
        resources = memory_and_priority()
        if resources["available_memory_bytes"] < config["resources"]["min_available_system_memory_bytes"]:
            raise StopRun("可用内存低于冻结阈值。")
        cli = probe.cli_command()
        version = probe.run_command(cli + ["--version"])
        login = probe.run_command(cli + ["login", "status"])
        if (version.get("exit_code") != 0 or login.get("exit_code") != 0
                or "Logged in using ChatGPT" not in (login.get("stdout", "") + login.get("stderr", ""))):
            raise StopRun("无法确认 Codex CLI 版本或 ChatGPT 登录。")
        log.event("preflight", resources=resources, cli_version=version.get("stdout", "").strip(),
                  code_sha256=runtime.protected_hashes, selection_seed=seed, generations=generations,
                  method="roco_paper_adapted")
        if not args.run:
            log.event("finish", status="preflight_passed")
            print(f"启动前检查通过；记录：{run_dir}")
            return 0
        runtime.cli = cli
        controller = RoCoController(runtime, log, population_size=config["task"]["population_size"],
                                    initial_count=config["task"]["initial_proposals"], rounds=3,
                                    seed=seed, generations=generations, task_contract=runtime.contract)
        with formal_run_guard(config, run_dir, log):
            controller.run()
        print(f"RoCo 训练完成；记录：{run_dir}")
        return 0
    except (StopRun, probe.ProbeError, OSError, ValueError, KeyError) as exc:
        log.event("preflight_or_run_stopped", error=f"{type(exc).__name__}: {exc}")
        print(f"运行停止：{exc}；记录：{run_dir}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
