"""并发调用四次 P0 无种子模型探针，保留每次提示与原始事件。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import p0_model_probe as probe  # noqa: E402

OUTPUT_ROOT = ROOT / "outputs" / "p0_budget"
WORKERS = 4
RETRY_CONFIG = [
    "--disable", "unbounded_connection_retries",
    "-c", "model_providers.p0_chatgpt_https.request_max_retries=0",
    "-c", "model_providers.p0_chatgpt_https.stream_max_retries=0",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="outputs/p0_budget 中已有且未运行过本探针的目录")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(OUTPUT_ROOT.resolve()) or not run_dir.is_dir():
        parser.error("run_dir 须为 outputs/p0_budget 内已有目录。")
    summary_path = run_dir / "parallel_model_summary.json"
    worker_dirs = [run_dir / f"model_probe_{index}" for index in range(1, WORKERS + 1)]
    if summary_path.exists() or any(path.exists() for path in worker_dirs):
        parser.error("探针输出已存在；请使用新的运行目录。")

    cli = probe.cli_command()
    proposal, _ = probe.prompts()
    # 仅作用于当前探针进程；不更改任务1脚本或用户级配置。
    probe.HTTPS_CONFIG = [*probe.HTTPS_CONFIG, *RETRY_CONFIG]
    for worker_dir in worker_dirs:
        worker_dir.mkdir()
    start_utc = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()

    def worker(index: int, worker_dir: Path) -> dict[str, Any]:
        records = worker_dir / "calls.jsonl"
        try:
            response = probe.call(cli, records, worker_dir, f"parallel_{index}", proposal)
            parsed = probe.parse_proposal(response)
            return {"index": index, "status": "passed", "thought_characters": len(parsed["thought"]),
                    "code_characters": len(parsed["code"]), "records": str(records)}
        except Exception as exc:
            # 任一槽位失败不重试；同时保留其他已启动调用的返回和原始记录。
            return {"index": index, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "records": str(records)}

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(worker, index, directory) for index, directory in enumerate(worker_dirs, 1)]
        results = [future.result() for future in futures]
    summary = {
        "started_at_utc": start_utc,
        "elapsed_seconds": time.monotonic() - start,
        "model": probe.MODEL,
        "concurrency": WORKERS,
        "cli_retry_overrides": RETRY_CONFIG,
        "purpose": "预算与并发探针；生成结果不进入初始化种群。",
        "results": results,
    }
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"并发探针完成；记录：{summary_path}", flush=True)
    return 0 if all(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
