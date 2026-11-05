"""离线汇总运行事件中的模型 token；不读取账号，也不请求模型。"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")


def summarize(logs):
    """逐日志、逐阶段核对请求与用量；未知值不作为零用量。"""
    counts = {kind: Counter() for kind in ("model_attempt", "model_process", "model_usage")}
    totals = {field: 0 for field in FIELDS}
    missing = {field: 0 for field in FIELDS}
    invalid = {field: 0 for field in FIELDS}
    inconsistent = warning_events = missing_warning_metadata = 0
    for source_index, events in enumerate(logs):
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("事件必须是 JSON 对象。")
            kind = event.get("event")
            if not isinstance(kind, str):
                raise ValueError("事件缺少有效的 event 类型。")
            if kind not in counts:
                continue
            stage = event.get("stage")
            if not isinstance(stage, str) or not stage:
                raise ValueError(f"{kind} 事件缺少有效的 stage。")
            counts[kind][(source_index, stage)] += 1
            if kind != "model_usage":
                continue
            usage = event.get("usage")
            if not isinstance(usage, dict):
                usage = {}
            valid_values = {}
            for field in FIELDS:
                value = usage.get(field)
                if field not in usage or value is None:
                    missing[field] += 1
                elif type(value) is not int or value < 0:
                    invalid[field] += 1
                else:
                    totals[field] += value
                    valid_values[field] = value
            if (("input_tokens" in valid_values and "cached_input_tokens" in valid_values
                 and valid_values["cached_input_tokens"] > valid_values["input_tokens"])
                    or ("output_tokens" in valid_values and "reasoning_output_tokens" in valid_values
                        and valid_values["reasoning_output_tokens"] > valid_values["output_tokens"])):
                inconsistent += 1
            warnings = event.get("warnings")
            if not isinstance(warnings, list):
                missing_warning_metadata += 1
            elif warnings:
                warning_events += 1

    attempts, processes, usages = (counts[kind] for kind in counts)
    stages = attempts.keys() | processes.keys() | usages.keys()
    attempts_without_usage = sum(max(attempts[key] - usages[key], 0) for key in stages)
    usages_without_process = sum(max(usages[key] - processes[key], 0) for key in stages)
    process_mismatches = sum(abs(attempts[key] - processes[key]) for key in stages)
    incomplete_fields = any(missing.values()) or any(invalid.values())
    complete = (sum(attempts.values()) > 0 and not any((attempts_without_usage, usages_without_process,
                process_mismatches, incomplete_fields, inconsistent, warning_events, missing_warning_metadata)))
    return {
        "model_attempt_events": sum(attempts.values()),
        "model_process_events": sum(processes.values()),
        "model_usage_events": sum(usages.values()),
        "attempts_without_usage": attempts_without_usage,
        "usage_events_without_process": usages_without_process,
        "attempt_process_count_mismatches": process_mismatches,
        "stages_with_repeated_attempts": sum(value > 1 for value in attempts.values()),
        "observed_tokens": totals,
        "usage_events_missing_field": missing,
        "usage_events_invalid_field": invalid,
        "usage_events_with_inconsistent_subsets": inconsistent,
        "usage_events_with_warnings": warning_events,
        "usage_events_missing_warning_metadata": missing_warning_metadata,
        "local_records_complete": complete,
        "note": ("仅统计本地已回读用量；缓存输入属于输入，reasoning 输出属于输出。"
                 "缺失或失败请求不按零计算；服务端不可见开销不包含在内。"
                 "按日志和阶段核对事件数，重复阶段不能逐次证明对应关系。"),
    }


def read_events(path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是有效 JSON。") from exc


def main():
    # Windows 的管道默认编码可能不是 UTF-8；JSON 输出固定为 UTF-8。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="一个或多个运行的 events.jsonl 路径")
    args = parser.parse_args()
    try:
        paths = [path.resolve(strict=True) for path in args.logs]
        if len(set(paths)) != len(paths):
            raise ValueError("同一日志不能重复统计。")
        summary = summarize(read_events(path) for path in paths)
    except (OSError, ValueError) as exc:
        print(f"统计失败：{exc}", file=sys.stderr)
        return 1
    summary["source_logs"] = [str(path) for path in paths]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
