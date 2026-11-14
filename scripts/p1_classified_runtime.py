"""P1 的模型失败分类：模型终止事件与后台目录警告分别处理。"""

import json
from pathlib import Path
import re

from baselines.roco.runtime import RoCoRuntime
from scripts import p0_model_probe as probe
from scripts.p0_budget_model_probe import RETRY_CONFIG
from scripts.p0_controller import StopRun
from scripts.p0_runtime import digest
from scripts.p1_runtime import TaskRuntime


ROOT = Path(__file__).resolve().parents[1]
POLICY = re.compile(r"potentially violating our usage policy|policy[_ -]?violation|safety[_ -]?check", re.I)
QUOTA = re.compile(r"rate.limit|quota|usage limit|429", re.I)
NETWORK = re.compile(r"connection (?:failed|refused|reset)|request timed out|error sending request", re.I)


def classify_failure(result):
    """优先使用模型终止事件；后台插件目录警告不参与请求失败分类。"""
    stdout = result.get("stdout") if isinstance(result.get("stdout"), str) else ""
    stderr = result.get("stderr") if isinstance(result.get("stderr"), str) else ""
    plugin_warnings = []
    other_stderr = []
    for line in stderr.splitlines():
        if ("codex_core_plugins::manager" in line.lower()
                and ("plugin catalog" in line.lower() or "/backend-api/ps/plugins/list" in line.lower())):
            plugin_warnings.append(line)
        else:
            other_stderr.append(line)
    terminal, errors, unstructured = [], [], []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            unstructured.append(line)
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "turn.failed":
            detail = event.get("error")
            if isinstance(detail, dict):
                message = " ".join(str(detail.get(key, "")) for key in ("code", "message"))
            else:
                message = str(detail or "")
            if message.strip():
                terminal.append(message)
        elif event.get("type") == "error" and isinstance(event.get("message"), str):
            errors.append(event["message"])
    if terminal:
        message, source = terminal[-1], "turn.failed"
    elif errors:
        message, source = errors[-1], "error_event"
    else:
        message = " ".join([str(result.get("error") or ""), *unstructured, *other_stderr])
        source = "process_output"
    if POLICY.search(message):
        category = "policy"
    elif QUOTA.search(message):
        category = "quota"
    elif NETWORK.search(message):
        category = "network"
    else:
        category = "unknown"
    return {"category": category, "source": source, "plugin_warnings": plugin_warnings}


class ClassifiedModelMixin:
    """保持原模型、预算和提示，仅修正失败来源识别。"""

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
                classification = classify_failure(result)
                for warning in classification["plugin_warnings"]:
                    self.log.event("background_warning", stage=stage, component="plugin_catalog", message=warning)
                try:
                    message, usage, event_types, warnings = probe.extract_message(result)
                    self.log.event("model_usage", stage=stage, usage=usage, event_types=event_types,
                                   warnings=warnings)
                    if warnings:
                        raise StopRun("CLI 出现内部重试或传输回退，调用次数不能按冻结预算核实。")
                    return message
                except probe.ProbeError as exc:
                    category = classification["category"]
                    self.log.event("model_failure_classified", stage=stage, category=category,
                                   evidence_source=classification["source"])
                    if category == "policy":
                        self.cancelled.set()
                        raise StopRun("模型请求被安全审核拒绝；已停止且未自动重试。") from exc
                    if category == "quota":
                        self.cancelled.set()
                        raise StopRun("模型额度或限流失败；未自动重置。") from exc
                    if category == "network":
                        if self.cfg["budget"]["additional_infrastructure_retry_calls"] == 0:
                            self.log.event("infrastructure_retry_unavailable", stage=stage, reason=str(exc))
                            self.cancelled.set()
                            raise StopRun("模型连接失败；冻结合同不允许自动重试。") from exc
                        self.log.event("infrastructure_retry", stage=stage, reason=str(exc))
                        retry = True
                        continue
                    self.cancelled.set()
                    raise StopRun(f"模型调用失败或事件不可信：{exc}") from exc


def protect_classified_sources(runtime):
    for path in (Path(__file__).resolve(), ROOT / "scripts/p1_classified_runner.py"):
        runtime.protected.append(path)
        runtime.protected_hashes[str(path)] = digest(path)


class ClassifiedTaskRuntime(ClassifiedModelMixin, TaskRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        protect_classified_sources(self)


class ClassifiedRoCoRuntime(ClassifiedModelMixin, RoCoRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        protect_classified_sources(self)
