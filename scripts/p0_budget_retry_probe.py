"""用未监听的本机端口检查 Codex 的有限重试配置；不请求真实模型服务。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "p0_budget"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or not output.parent.is_dir() or output.exists():
        parser.error("输出须位于已有 outputs/p0_budget 子目录，且文件不存在。")
    command_path = shutil.which("codex.cmd")
    if not command_path:
        parser.error("找不到现有 codex.cmd。")
    native = (Path(command_path).parent / "node_modules" / "@openai" / "codex" / "node_modules" /
              "@openai" / "codex-win32-x64" / "vendor" / "x86_64-pc-windows-msvc" / "bin" / "codex.exe")
    if not native.is_file():
        parser.error("找不到现有 Codex 本机可执行文件。")

    # 绑定但不监听：保留端口并使连接失败，不接受或读取任何请求数据。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as closed_port:
        closed_port.bind(("127.0.0.1", 0))
        port = closed_port.getsockname()[1]
        command = [
            str(native), "exec", "--ignore-user-config", "--skip-git-repo-check", "--ephemeral",
            "--json", "--sandbox", "read-only", "--model", "gpt-5.6-luna",
            "--disable", "unbounded_connection_retries",
            "-c", 'model_provider="p0_refusal_fixture"',
            "-c", 'model_providers.p0_refusal_fixture.name="P0 refusal fixture"',
            "-c", f'model_providers.p0_refusal_fixture.base_url="http://127.0.0.1:{port}"',
            "-c", 'model_providers.p0_refusal_fixture.env_key="P0_RETRY_FIXTURE_KEY"',
            "-c", "model_providers.p0_refusal_fixture.requires_openai_auth=false",
            "-c", "model_providers.p0_refusal_fixture.supports_websockets=false",
            "-c", 'model_providers.p0_refusal_fixture.wire_api="responses"',
            "-c", "model_providers.p0_refusal_fixture.request_max_retries=0",
            "-c", "model_providers.p0_refusal_fixture.stream_max_retries=0",
            "--cd", str(output.parent), "-",
        ]
        env = os.environ.copy()
        env["P0_RETRY_FIXTURE_KEY"] = "not-a-real-token"
        start = time.monotonic()
        record = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
                  "fixture": "本机端口已绑定但不监听，使用虚构密钥，不访问真实模型端点。"}
        try:
            result = subprocess.run(
                command, input="Return only OK.", capture_output=True, text=True, encoding="utf-8",
                errors="replace", env=env, timeout=30, check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            record.update({"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
            retry_events = [event for event in events if event.get("type") == "error" and
                            "Reconnecting" in event.get("message", "")]
            record["status"] = ("passed" if result.returncode != 0 and
                                any(event.get("type") == "turn.failed" for event in events) and
                                not retry_events and "waiting to retry" not in result.stderr else "failed")
            record["visible_reconnect_events"] = len(retry_events)
        except subprocess.TimeoutExpired:
            record.update({"status": "timeout", "error": "本机拒绝连接测试超过 30 秒，已停止测试进程。"})
        except (OSError, json.JSONDecodeError) as exc:
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        record["elapsed_seconds"] = time.monotonic() - start
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(f"离线拒绝连接检查：{record['status']}；记录：{output}")
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
