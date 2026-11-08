"""P0 任务1：用已登录的 Codex CLI 探测生成和角色输出协议。"""

from __future__ import annotations

import ast
import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-5.6-luna"
# 仅本进程参数：沿用 ChatGPT 后端和登录，直接使用此前回退成功的 HTTPS。
HTTPS_CONFIG = [
    "-c", 'model_provider="p0_chatgpt_https"',
    "-c", 'model_providers.p0_chatgpt_https.name="OpenAI"',
    "-c", 'model_providers.p0_chatgpt_https.base_url="https://chatgpt.com/backend-api/codex"',
    "-c", "model_providers.p0_chatgpt_https.requires_openai_auth=true",
    "-c", "model_providers.p0_chatgpt_https.supports_websockets=false",
    "-c", 'model_providers.p0_chatgpt_https.wire_api="responses"',
]
TASK = (
    "Design the edge heuristic module for a fixed ant-colony TSP solver. "
    "The objective is to minimize tour length. The module takes a NumPy "
    "distance matrix and returns an edge-score matrix of the same shape. "
    "Its interface is def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray."
)


class ProbeError(Exception):
    """调用或输出协议未通过。"""


def cli_command() -> list[str]:
    """调用当前 npm Codex 包的入口，绕开 Windows 批处理的 shell 转义。"""
    command = shutil.which("codex.cmd")
    node = shutil.which("node.exe")
    if not command or not node:
        raise ProbeError("找不到现有 codex.cmd 或 node.exe。")
    entry = Path(command).parent / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
    if not entry.is_file():
        raise ProbeError(f"找不到当前 Codex 包入口：{entry}")
    return [node, str(entry)]


def run_command(command: list[str], input_text: str | None = None, cwd: Path | None = None) -> dict[str, Any]:
    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        result = subprocess.run(
            command,
            input=None if input_text is None else input_text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            check=False,
        )
        try:
            stdout = result.stdout.decode("utf-8")
            stderr = result.stderr.decode("utf-8")
        except UnicodeDecodeError as exc:
            return {
                "command": command,
                "started_at_utc": started_at,
                "elapsed_seconds": round(time.monotonic() - start, 3),
                "exit_code": result.returncode,
                "error": f"UTF-8 解码失败：{exc}",
                "stdout_base64": base64.b64encode(result.stdout).decode("ascii"),
                "stderr_base64": base64.b64encode(result.stderr).decode("ascii"),
            }
        return {
            "command": command,
            "started_at_utc": started_at,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "exit_code": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    except OSError as exc:
        return {
            "command": command,
            "started_at_utc": started_at,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "exit_code": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def audit_context(cli: list[str], run_dir: Path, prompt: str) -> dict[str, Any]:
    """记录 CLI 本地可见的上下文结构，不复制用户级指令正文。"""
    diagnostic = run_command(
        cli + ["debug", "prompt-input", "-c", f'model="{MODEL}"', prompt], cwd=run_dir
    )
    audit: dict[str, Any] = {
        "event": "context_audit",
        "scope": "debug prompt-input 使用用户配置，不能传 exec 的 --ignore-user-config；仅作同目录对照，并非本次服务端完整输入。",
        "debug_exit_code": diagnostic.get("exit_code"),
    }
    try:
        if diagnostic.get("exit_code") != 0:
            raise ValueError(diagnostic.get("error") or diagnostic.get("stderr") or "debug prompt-input 失败")
        messages = json.loads(diagnostic["stdout"])
        parts = []
        for message in messages:
            for item in message.get("content", []):
                text = item.get("text", "")
                parts.append({
                    "role": message.get("role"), "type": item.get("type"),
                    "characters": len(text), "contains_prompt": prompt in text,
                    "mentions_agents_md": "AGENTS.md" in text,
                    "mentions_skills": "skills" in text.lower(),
                })
        audit["prompt_input_parts"] = parts
        audit["explicit_prompt_present_in_comparison"] = any(part["contains_prompt"] for part in parts)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        audit["debug_error"] = str(exc)

    catalog = run_command(cli + ["debug", "models", "--bundled"])
    try:
        if catalog.get("exit_code") != 0:
            raise ValueError(catalog.get("error") or catalog.get("stderr") or "bundled 模型目录不可用")
        models = json.loads(catalog["stdout"])["models"]
        matches = [model for model in models if model.get("slug") == MODEL]
        if len(matches) != 1:
            raise ValueError("bundled 模型目录没有唯一匹配项")
        keys = ("slug", "display_name", "context_window", "max_context_window", "effective_context_window_percent", "truncation_policy")
        audit["bundled_model_catalog"] = {key: matches[0].get(key) for key in keys}
        audit["catalog_scope"] = "本机 bundled 目录字段；不代表服务端返回的权重快照，也不推断截断策略字段的作用对象。"
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        audit["catalog_error"] = str(exc)
    return audit


def tagged(text: str, names: tuple[str, ...]) -> list[str]:
    pattern = r"\s*" + r"\s*".join(
        rf"<{name}>\s*(.*?)\s*</{name}>" for name in names
    ) + r"\s*"
    match = re.fullmatch(pattern, text, flags=re.DOTALL)
    if not match or any(not value.strip() for value in match.groups()):
        raise ProbeError(f"输出必须依次且仅包含非空标签：{', '.join(names)}。")
    return [value.strip() for value in match.groups()]


def english_words(text: str) -> int:
    return len(re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", text, flags=re.UNICODE))


def parse_proposal(text: str) -> dict[str, str]:
    thought, code = tagged(text, ("thought", "code"))
    try:
        module = ast.parse(code)
    except SyntaxError as exc:
        raise ProbeError(f"候选代码语法错误：{exc}") from exc
    functions = [
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "heuristics_v2"
    ]
    if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef):
        raise ProbeError("候选须定义且仅定义一个同步 heuristics_v2 函数。")
    function = functions[0]
    args = function.args
    if (
        args.posonlyargs or len(args.args) != 1 or args.args[0].arg != "distance_matrix"
        or args.vararg or args.kwonlyargs or args.kwarg or args.defaults
        or args.args[0].annotation is None or function.returns is None
        or ast.unparse(args.args[0].annotation) != "np.ndarray"
        or ast.unparse(function.returns) != "np.ndarray"
    ):
        raise ProbeError("候选函数签名不符合 heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray。")
    if not any(
        isinstance(node, ast.Import) and any(alias.name == "numpy" and alias.asname == "np" for alias in node.names)
        for node in module.body
    ):
        raise ProbeError("完整模块须包含 import numpy as np。")
    def returns_value(node: ast.AST) -> bool:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return False
        if isinstance(node, ast.Return):
            return node.value is not None
        return any(returns_value(child) for child in ast.iter_child_nodes(node))

    if not any(returns_value(statement) for statement in function.body):
        raise ProbeError("候选函数缺少返回值。")
    try:
        compile(module, "<model-proposal>", "exec")
    except SyntaxError as exc:
        raise ProbeError(f"候选代码无法编译：{exc}") from exc
    return {"thought": thought, "code": code}


def parse_critic(text: str) -> dict[str, str]:
    reflection, advice = tagged(text, ("ref", "ans"))
    # Prompt 5 正文要求设计提示少于 20 词；其格式上限分别为 30 与 40 词。
    if english_words(reflection) >= 20 or english_words(advice) > 40:
        raise ProbeError("初始 Critic 输出超过论文 Prompt 5 的词数限制。")
    return {"reflection": reflection, "advice": advice}


def parse_long_term(text: str) -> str:
    (reflection,) = tagged(text, ("ans",))
    if english_words(reflection) >= 50:
        raise ProbeError("长期反思须少于 50 个词。")
    return reflection


def parse_merge(text: str) -> str:
    result = text.strip()
    if not result or "```" in result or re.search(r"</?(?:ref|ans|code|thought)>", result):
        raise ProbeError("合并长期反思须为非空纯文本，不附加标签或代码。")
    return result


def extract_message(result: dict[str, Any]) -> tuple[str, Any, list[str], list[str]]:
    if result.get("exit_code") != 0:
        raise ProbeError(f"Codex 进程失败，退出码：{result.get('exit_code')}。")
    if result.get("error"):
        raise ProbeError(str(result["error"]))
    events = []
    try:
        for line in result["stdout"].splitlines():
            if line.strip():
                events.append(json.loads(line))
    except (json.JSONDecodeError, KeyError) as exc:
        raise ProbeError(f"Codex JSONL 无法解析：{exc}") from exc
    if not events or any(event.get("type") == "turn.failed" for event in events):
        raise ProbeError("Codex 返回失败事件或没有事件。")
    # CLI 可能先报告 WebSocket 超时，随后在同一调用中转为 HTTPS 并成功完成。
    # 只容许实测的非终止重试；其他错误依旧停止。
    warnings = []
    for event in events:
        if event.get("type") == "error":
            message = event.get("message", "")
            recovered_network = "Reconnecting... waiting for network (Connection failed: error sending request)"
            if not (re.fullmatch(r"Reconnecting\.\.\. \d+/\d+ \(request timed out\)", message)
                    or message == recovered_network):
                raise ProbeError(f"Codex 返回错误事件：{message}")
            warnings.append(message)
    completed = [event for event in events if event.get("type") == "turn.completed"]
    if len(completed) != 1:
        raise ProbeError("Codex 未返回唯一 turn.completed 事件。")
    items = [event.get("item", {}) for event in events if event.get("type") == "item.completed"]
    unexpected = [
        event.get("item", {}).get("type")
        for event in events
        if event.get("type") in {"item.started", "item.updated"}
        and event.get("item", {}).get("type") not in {"agent_message", "reasoning"}
    ]
    for item in items:
        if item.get("type") == "error" and item.get("message") == "Falling back from WebSockets to HTTPS transport. request timed out":
            warnings.append(item["message"])
        elif item.get("type") not in {"agent_message", "reasoning"}:
            unexpected.append(item.get("type"))
    if unexpected:
        raise ProbeError(f"角色调用包含非预期工具或动作：{unexpected}")
    messages = [item.get("text") for item in items if item.get("type") == "agent_message"]
    if not messages or not isinstance(messages[-1], str) or not messages[-1].strip():
        raise ProbeError("Codex 未提供非空最终消息。")
    event_types = [event.get("type", "") for event in events]
    return messages[-1], completed[0].get("usage"), event_types, warnings


def call(cli: list[str], records: Path, run_dir: Path, stage: str, prompt: str) -> str:
    command = cli + [
        "exec", "--ignore-user-config", "--skip-git-repo-check",
        "--json", "--sandbox", "read-only", "--model", MODEL,
    ] + HTTPS_CONFIG + ["--cd", str(run_dir), "-"]
    append_record(records, {"event": "prompt", "stage": stage, "prompt": prompt})
    print(f"调用 {stage}，模型 {MODEL}。", flush=True)
    result = run_command(command, prompt)
    append_record(records, {"event": "process", "stage": stage, **result})
    message, usage, event_types, warnings = extract_message(result)
    append_record(records, {
        "event": "response", "stage": stage, "message": message,
        "usage": usage, "event_types": event_types, "warnings": warnings,
    })
    if any("Falling back from WebSockets to HTTPS transport" in warning for warning in warnings):
        raise ProbeError("HTTPS 配置未生效：本次仍发生 WebSocket 回退。")
    return message


def prompts() -> tuple[str, str]:
    proposal = f"""{TASK}
Propose one new heuristic without being given an existing heuristic.
Return exactly the following two non-empty tags, in this order, with no Markdown fences or other text:
<thought>A concise description of the heuristic rule.</thought>
<code>Complete executable Python module, including import numpy as np and the full function.</code>
Do not use tools or read files; respond only with the requested output.
"""
    critic = f"""{TASK}
The following codes and scores are synthetic fixtures for testing this prompt, not measured results.
Better code:
import numpy as np
def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    return 1.0 / (distance_matrix + 1.0)
Better objective score: 100.0
Worse code:
import numpy as np
def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    return np.ones_like(distance_matrix)
Worse objective score: 120.0
Note: a lower objective score is better.
Act as a reflector and critical evaluator in the domain of optimization heuristics. Reflect on why the better algorithm may outperform the other and give hints for designing better heuristics in fewer than 20 words. Then analyze the worse algorithm's limitations and provide specific, actionable improvements.
Return exactly:
<ref>Reflection on current code's evolution direction (max 30 words)</ref>
<ans>Specific critique and actionable suggestions (max 40 words)</ans>
Do not use tools or read files; respond only with the requested output.
"""
    return proposal, critic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-proposal", type=Path, help="复用已完成但曾被解析器误判的 proposal 记录")
    options = parser.parse_args()
    if options.resume_proposal:
        records = options.resume_proposal.resolve()
        allowed = (ROOT / "outputs" / "p0_model_call").resolve()
        if records.name != "calls.jsonl" or not records.is_relative_to(allowed) or not records.is_file():
            parser.error("--resume-proposal 必须指向本项目 outputs/p0_model_call 下已有的 calls.jsonl。")
        run_dir = records.parent
    else:
        run_dir = ROOT / "outputs" / "p0_model_call" / (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        records = run_dir / "calls.jsonl"
        append_record(records, {
            "event": "start", "model_requested": MODEL,
            "transport": "项目调用通过临时 p0_chatgpt_https 配置直接使用 ChatGPT HTTPS 后端。",
            "context": "每次调用均为新的持久化会话；只显式传入该次 prompt，不恢复其他会话。",
            "truncation": "调用脚本不截断提示；CLI/服务端内部上下文与截断未由该接口完整暴露。",
            "fixtures": "Critic 和长期反思中的代码与分数为人工构造的协议测试数据，不是评价结果。",
        })
    try:
        cli = cli_command()
        version = run_command(cli + ["--version"])
        login = run_command(cli + ["login", "status"])
        append_record(records, {"event": "resume_preflight" if options.resume_proposal else "preflight", "version": version, "login_status": login})
        if version.get("exit_code") != 0:
            raise ProbeError("无法获取 Codex CLI 版本。")
        if login.get("exit_code") != 0 or "Logged in using ChatGPT" not in (login.get("stdout", "") + login.get("stderr", "")):
            raise ProbeError("Codex CLI 未确认通过 ChatGPT 登录。")

        proposal_prompt, critic_prompt = prompts()
        append_record(records, audit_context(cli, run_dir, proposal_prompt))
        if options.resume_proposal:
            try:
                previous = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
            except (OSError, json.JSONDecodeError) as exc:
                raise ProbeError(f"无法读取原调用记录：{exc}") from exc
            old_prompts = [entry for entry in previous if entry.get("event") == "prompt"]
            old_processes = [entry for entry in previous if entry.get("event") == "process"]
            finishes = [entry for entry in previous if entry.get("event") == "finish"]
            old_preflight = [entry for entry in previous if entry.get("event") == "preflight"]
            if (
                len(old_prompts) != 1 or old_prompts[0].get("stage") != "proposal"
                or old_prompts[0].get("prompt") != proposal_prompt
                or len(old_processes) != 1 or old_processes[0].get("stage") != "proposal"
                or not finishes or finishes[-1].get("error") != "Codex 返回错误事件或没有事件。"
                or not old_preflight or old_preflight[-1]["version"].get("stdout") != version.get("stdout")
            ):
                raise ProbeError("原记录不是本次可恢复的初始解析器误判，或提示／CLI 版本已变。")
            result = old_processes[0]
            message, usage, event_types, warnings = extract_message(result)
            append_record(records, {"event": "resume", "reason": "复用已完成的 proposal；先前解析器误判已恢复的传输重试。", "version": version, "login_status": login})
            append_record(records, {"event": "response", "stage": "proposal", "message": message, "usage": usage, "event_types": event_types, "warnings": warnings})
        else:
            message = call(cli, records, run_dir, "proposal", proposal_prompt)
        proposal = parse_proposal(message)
        append_record(records, {"event": "parsed", "stage": "proposal", **proposal})

        critic = parse_critic(call(cli, records, run_dir, "initial_critic", critic_prompt))
        append_record(records, {"event": "parsed", "stage": "initial_critic", **critic})

        reflections = {}
        for role, description in (
            ("Solution Quality Agent", "focus on improving the objective value"),
            ("Speed Agent", "focus on reducing computational cost"),
        ):
            prompt = f"""Below is your prior short-term reflection on designing heuristics for {TASK}
{critic['reflection']}
Based on the reflection and synthetic objective scores changing from 120.0 to 100.0 for role {role}, {description}, write constructive hints for designing better heuristics in fewer than 50 words. Return exactly <ans>...</ans>.
Do not use tools or read files; respond only with the requested output.
"""
            stage = f"long_term_{role}"
            reflections[role] = parse_long_term(call(cli, records, run_dir, stage, prompt))
            append_record(records, {"event": "parsed", "stage": stage, "reflection": reflections[role]})

        merge_prompt = f"""Merge these two long-term reflections into one concise experience text. Preserve the useful insights; return plain text only.
Solution Quality Agent: {reflections['Solution Quality Agent']}
Speed Agent: {reflections['Speed Agent']}
Do not use tools or read files; respond only with the requested output.
"""
        merged = parse_merge(call(cli, records, run_dir, "long_term_merge", merge_prompt))
        append_record(records, {"event": "parsed", "stage": "long_term_merge", "reflection": merged})
        append_record(records, {"event": "finish", "status": "passed"})
        print(f"协议探针通过；记录：{records}", flush=True)
        return 0
    except ProbeError as exc:
        append_record(records, {"event": "finish", "status": "failed", "error": str(exc)})
        print(f"协议探针停止：{exc}\n记录：{records}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
