"""独立重算逐实例路线，核对数据身份与 ReEvo 笔记公布的均值。"""

from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
REFERENCE = RUN / "result.json"
CONTRACT = ROOT / "cfg" / "p1_data_contract.json"
PUBLISHED_MEANS = {20: 3.8362853943492015, 50: 5.68457994395107,
                   100: 7.778580370400294, 200: 10.71194600194464}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    registered = {item["size"]: item for item in contract["tasks"]["tsp_gls"]["test"]}
    if (reference["status"] != "passed" or reference["optimality_proven"] is not False
            or reference["protocol"]["data_contract_sha256"] != digest(CONTRACT)
            or reference["protocol"]["scale"] != 1_000_000 or reference["protocol"]["runs"] != 10
            or set(reference["datasets"]) != {str(size) for size in PUBLISHED_MEANS}):
        raise ValueError("结果状态、协议或数据合同不符。")
    results = {}
    max_error = 0.0
    for size, published in PUBLISHED_MEANS.items():
        result = reference["datasets"][str(size)]
        item = registered[size]
        if result["dataset"] != item["path"] or result["dataset_sha256"] != item["sha256"]:
            raise ValueError(f"TSP{size} 的数据登记不符。")
        path = ROOT / item["path"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"TSP{size} 的数据哈希不符。")
        points = np.load(path, allow_pickle=False)
        records = result["instances"]
        if (points.dtype != np.float64 or points.shape != (64, size, 2) or len(records) != 64
                or [row["index"] for row in records] != list(range(64))):
            raise ValueError(f"TSP{size} 的数据类型或实例数量不符。")
        objectives = []
        for row, coordinates in zip(records, points):
            tour, objective = row["tour"], row["objective"]
            if (len(tour) != size + 1 or tour[0] != tour[-1]
                    or any(type(node) is not int for node in tour)
                    or sorted(tour[:-1]) != list(range(size)) or not math.isfinite(objective)):
                raise ValueError(f"TSP{size} 实例 {row['index']} 的路线无效。")
            recomputed = math.fsum(math.hypot(float(coordinates[a, 0]) - float(coordinates[b, 0]),
                                              float(coordinates[a, 1]) - float(coordinates[b, 1]))
                                   for a, b in zip(tour, tour[1:]))
            difference = abs(recomputed - objective)
            max_error = max(max_error, difference)
            if difference > 1e-12:
                raise ValueError(f"TSP{size} 实例 {row['index']} 的路长不符。")
            objectives.append(objective)
        mean = float(np.mean(objectives))
        if mean != result["mean_objective"]:
            raise ValueError(f"TSP{size} 的平均值不符。")
        results[str(size)] = {"instances": len(records), "mean_objective": mean,
                              "published_reevo_mean": published,
                              "difference_from_published": mean - published}
    report = {"status": "passed", "reference_sha256": digest(REFERENCE),
              "data_contract_sha256": digest(CONTRACT), "auditor_sha256": digest(Path(__file__)),
              "route_count": 256, "max_absolute_route_error": max_error, "sizes": results,
              "optimality_proven": False,
              "note": "逐边以 math.hypot 和 math.fsum 重算；均值一致不证明历史笔记使用了相同文件。"}
    with (RUN / "independent_audit.json").open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(f"audit passed: 256 routes, max error {max_error:.3g}")


if __name__ == "__main__":
    main()
