"""生成并独立重放不调用模型、不启动评价进程的 P0 模拟整代记录。"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scripts.p0_controller import Controller  # noqa: E402
from scripts.p0_runtime import JsonLog  # noqa: E402
from test_p0_controller import FixtureBackend  # noqa: E402


def main():
    root = ROOT / "outputs/p0_control_checks" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    log = JsonLog(root / "events.jsonl")
    report = {"status": "failed", "fixture": "全部模型响应与评价分数均为人工构造，不代表真实运行。"}
    try:
        backend = FixtureBackend()
        Controller(backend, log).run()
        rows = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
        assert [row["event_index"] for row in rows] == list(range(1, len(rows) + 1))
        registrations = [row for row in rows if row["event"] == "candidate_registered"]
        evaluations = {row["candidate_id"]: row["result"] for row in rows if row["event"] == "evaluation"}
        stages = {row["stage"]: row for row in registrations}
        assert len(registrations) == 88 and len(evaluations) == 88
        assert len(backend.calls) == 95 and len(backend.evaluations) == 88
        counts = Counter(row["operator"] for row in registrations)
        assert counts == {"initialization": 30, "E1": 10, "E2": 10, "M1": 10, "M2": 10,
                          "startup": 2, "continuation": 6, "round_fusion": 3,
                          "elite_fusion": 1, "elite_mutation": 6}
        expected_stages = [f"eoh_{op}_{index:02d}" for op in ("E1", "E2", "M1", "M2")
                           for index in range(1, 11)]
        expected_stages += ["round_3_Q", "round_3_E", "fusion_3", "fusion_final"]
        expected_stages += [f"mutation_{elite}_{role}" for elite in ("A", "B") for role in ("Q", "E", "I")]
        expected_pool = [stages[stage]["candidate_id"] for stage in expected_stages]
        selection = next(row for row in rows if row["event"] == "selection")
        assert selection["pool"] == expected_pool
        generation = {row["candidate_id"]: row["generation_order"] for row in rows
                      if row["event"] == "proposal_response"}
        ranked = sorted(expected_pool, key=lambda identifier: (evaluations[identifier]["average"], generation[identifier]))
        assert [item["candidate_id"] for item in selection["selected"]] == ranked[:10]
        selection_index = selection["event_index"]
        history = [row for row in rows if row["event"] == "history_appended"]
        assert len(history) == 3 and all(row["event_index"] > selection_index for row in history)
        assert rows[-1]["event"] == "finish" and rows[-1]["status"] == "completed"
        report.update({"status": "passed", "model_calls": len(backend.calls),
                       "evaluation_slots": len(evaluations), "candidate_slots": len(registrations),
                       "operator_counts": dict(counts), "pool_size": len(expected_pool),
                       "selected": ranked[:10], "event_count": len(rows)})
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    with (root / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"任务4模拟记录重放：{report['status']}；记录：{root}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
