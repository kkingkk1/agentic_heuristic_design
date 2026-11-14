"""只读记录当前 Codex 订阅额度窗口，不估算剩余请求数。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.p0_model_probe import HTTPS_CONFIG, MODEL, ProbeError, cli_command  # noqa: E402

OUTPUT_ROOT = ROOT / "outputs" / "p0_budget"
REQUEST_TIMEOUT_SECONDS = 40


class QuotaProbeError(Exception):
    """订阅额度回读失败，不能据此放行运行。"""


def _window(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise QuotaProbeError("额度窗口格式无效。")
    used = value.get("usedPercent")
    if isinstance(used, bool) or not isinstance(used, (int, float)) or not 0 <= used <= 100:
        raise QuotaProbeError("额度窗口缺少有效使用百分比。")
    return {
        "used_percent": used,
        "duration_minutes": value.get("windowDurationMins"),
        "resets_at_unix": value.get("resetsAt"),
    }


def sanitize_limits(result: Any) -> dict[str, Any]:
    """保留额度字段，不保存账号 ID、凭证或重置额度 ID。"""
    if not isinstance(result, dict) or not isinstance(result.get("rateLimits"), dict):
        raise QuotaProbeError("额度回读缺少 rateLimits。")
    by_id = result.get("rateLimitsByLimitId")
    if by_id is None:
        rate = result["rateLimits"]
        by_id = {rate.get("limitId") or "default": rate}
    if not isinstance(by_id, dict) or not by_id:
        raise QuotaProbeError("额度回读缺少可用窗口。")
    buckets = []
    for bucket_id, item in by_id.items():
        if not isinstance(item, dict):
            raise QuotaProbeError("额度桶格式无效。")
        credits = item.get("credits")
        buckets.append({
            "limit_id": bucket_id,
            "primary": _window(item.get("primary")),
            "secondary": _window(item.get("secondary")),
            "spend_control_reached": item.get("spendControlReached"),
            "rate_limit_reached_type": item.get("rateLimitReachedType"),
            "has_credits": credits.get("hasCredits") if isinstance(credits, dict) else None,
        })
    if not any(bucket["primary"] is not None or bucket["secondary"] is not None for bucket in buckets):
        raise QuotaProbeError("额度回读没有任何窗口。")
    resets = result.get("rateLimitResetCredits")
    return {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "codex app-server account/rateLimits/read",
        "requested_model": MODEL,
        "model_specific_bucket_confirmed": False,
        "ordinary_usage_allowed": result.get("ordinaryUsageAllowed"),
        "buckets": buckets,
        "available_reset_credit_count": resets.get("availableCount") if isinstance(resets, dict) else None,
        "remaining_model_calls": None,
        "note": "窗口百分比不能直接换算成剩余模型调用次数；本次只读回读不使用重置额度。",
    }


def read_limits() -> dict[str, Any]:
    try:
        command = cli_command() + ["app-server"] + HTTPS_CONFIG
    except ProbeError as exc:
        raise QuotaProbeError("找不到现有 Codex CLI。") from exc
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except OSError as exc:
        raise QuotaProbeError(f"无法启动 Codex app-server：{type(exc).__name__}") from exc

    lines: queue.Queue[str | None] = queue.Queue()

    def collect() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=collect, daemon=True).start()

    def send(value: dict[str, Any]) -> None:
        assert process.stdin is not None
        try:
            process.stdin.write(json.dumps(value) + "\n")
            process.stdin.flush()
        except OSError as exc:
            raise QuotaProbeError("Codex app-server 输入连接失败。") from exc

    def receive(request_id: int) -> dict[str, Any]:
        deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise QuotaProbeError("Codex app-server 回读超时。")
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty as exc:
                raise QuotaProbeError("Codex app-server 回读超时。") from exc
            if line is None:
                raise QuotaProbeError("Codex app-server 在返回额度前退出。")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QuotaProbeError("Codex app-server 返回无效 JSON。") from exc
            if not isinstance(event, dict) or event.get("id") != request_id:
                continue
            if "error" in event:
                error = event["error"]
                code = error.get("code") if isinstance(error, dict) else None
                raise QuotaProbeError(f"Codex app-server 返回错误，代码：{code}。")
            if not isinstance(event.get("result"), dict):
                raise QuotaProbeError("Codex app-server 响应缺少 result。")
            return event["result"]

    try:
        send({"method": "initialize", "id": 1, "params": {
            "clientInfo": {"name": "p0_budget_readonly", "title": "P0 budget read-only", "version": "0.1.0"},
        }})
        receive(1)
        send({"method": "initialized", "params": {}})
        send({"method": "account/rateLimits/read", "id": 2})
        return sanitize_limits(receive(2))
    finally:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="outputs/p0_budget 中尚不存在的快照文件")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or not output.parent.is_dir() or output.exists():
        parser.error("输出须位于已有 outputs/p0_budget 子目录，且文件不存在。")
    try:
        record = {"status": "passed", **read_limits()}
    except (QuotaProbeError, OSError) as exc:
        record = {"status": "failed", "observed_at_utc": datetime.now(timezone.utc).isoformat(), "error": str(exc)}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"额度回读：{record['status']}；记录：{output}")
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
