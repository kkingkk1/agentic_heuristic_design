"""读取已核验的 TSP-GLS 测试参考，并定义参考停止阈值。"""

from hashlib import sha256
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "outputs/p1_tsp_gls/20261105_lkh_test_reference_01/result.json"
REFERENCE_SHA256 = "e9d89b4d49cee3f7bacf88d185f3e4ee8152f24defc1bf7aa072ac4be352bcc4"
ABS_TOLERANCE = 1e-12
REL_TOLERANCE = 1e-12


def reference_digest():
    return sha256(REFERENCE_PATH.read_bytes()).hexdigest()


def stop_threshold(reference):
    if isinstance(reference, bool) or not isinstance(reference, (int, float)) or not math.isfinite(reference) or reference <= 0:
        raise ValueError("LKH 参考长度必须是有限正数。")
    return float(reference) + max(ABS_TOLERANCE, REL_TOLERANCE * abs(float(reference)))


def reached_reference(objective, reference):
    if isinstance(objective, bool) or not isinstance(objective, (int, float)) or not math.isfinite(objective) or objective <= 0:
        raise ValueError("GLS 长度必须是有限正数。")
    return float(objective) <= stop_threshold(reference)


def load_reference(size, dataset_sha256):
    if reference_digest() != REFERENCE_SHA256:
        raise ValueError("LKH 参考文件哈希不符。")
    result = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    if (result.get("status") != "passed" or result.get("optimality_proven") is not False
            or result.get("protocol", {}).get("scale") != 1_000_000
            or result.get("protocol", {}).get("runs") != 10):
        raise ValueError("LKH 参考来源或协议无效。")
    group = result.get("datasets", {}).get(str(size), {})
    rows = group.get("instances", [])
    if (group.get("dataset_sha256") != dataset_sha256 or len(rows) != 64
            or [row.get("index") for row in rows] != list(range(64))):
        raise ValueError("LKH 参考与测试数据或实例顺序不一致。")
    values = [row.get("objective") for row in rows]
    for value in values:
        stop_threshold(value)
    mean = group.get("mean_objective")
    if (isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean)
            or not math.isclose(mean, math.fsum(values) / 64, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("LKH 参考均值无效。")
    return {"objectives": values, "mean": float(mean), "sha256": REFERENCE_SHA256}
