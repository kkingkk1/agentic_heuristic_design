"""用修改前的 float64 求解器重放同一训练夹具，并比较全部路线。"""

from hashlib import sha256
import json
from pathlib import Path
import sys
import types

import numpy as np
from scipy.spatial import distance_matrix


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
SOLVER = ROOT / "problems/tsp_gls/gls.py"
DATASET = ROOT / "problems/tsp_gls/dataset/train200_dataset.npy"
OLD_SOLVER_SHA256 = "bce35a2cc2186906df2752824aec9de810b610e561312e4c1cfd9f14fa71da7d"
DATASET_SHA256 = "f8da974c2a9e935b86c05991f0ed722d01bd1e3a4a18ddde65c381b224086235"
FIXTURE_SHA256 = "dff978f28f787af1d30f90b0e7dd70f11e74b620385064de683434ac56cb5c6e"


def digest(data):
    return sha256(data).hexdigest()


def main():
    marker = "# P1 测试：".encode("utf-8")
    original = SOLVER.read_bytes().split(marker, 1)[0].rstrip(b"\n") + b"\n"
    if (digest(original) != OLD_SOLVER_SHA256 or digest(DATASET.read_bytes()) != DATASET_SHA256
            or digest((RUN / "problems/tsp_gls/gpt.py").read_bytes()) != FIXTURE_SHA256):
        raise ValueError("求解器、训练数据或夹具身份不符。")
    result = json.loads((RUN / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "passed":
        raise ValueError("修改后的训练夹具未通过。")
    prefix = "P1_JSON "
    events = [json.loads(line[len(prefix):]) for line in (RUN / "stdout.txt").read_text(encoding="utf-8").splitlines()
              if line.startswith(prefix)]
    rows = events[1:-1]
    points = np.load(DATASET, allow_pickle=False)
    if len(rows) != 10 or points.shape != (10, 200, 2):
        raise ValueError("训练实例或输出数量不符。")
    old_module = types.ModuleType("p1_pre_stop_gls")
    old_module.__file__ = str(SOLVER)
    sys.modules[old_module.__name__] = old_module
    without_cache = original.replace(b"usecache = True", b"usecache = False", 1)
    exec(compile(without_cache, str(SOLVER), "exec"), old_module.__dict__)
    for index, (coordinates, row) in enumerate(zip(points, rows)):
        distances = distance_matrix(coordinates, coordinates) + np.eye(200) * 1e-5
        tour = old_module.guided_local_search(distances, distances, 30, 1200)
        objective = float(distances[tour, np.roll(tour, 1)].sum())
        if (row["index"] != index or row["iterations"] != 1200
                or tour.tolist() != row["tour"] or objective != row["objective"]):
            raise ValueError(f"训练实例 {index} 的路线或长度不同。")
    report = {"status": "passed", "training_instances": len(rows), "routes_identical": True,
              "objectives_identical": True, "iterations_per_instance": 1200,
              "pre_change_solver_sha256": OLD_SOLVER_SHA256, "dataset_sha256": DATASET_SHA256,
              "fixture_sha256": FIXTURE_SHA256,
              "replay_result_sha256": digest((RUN / "result.json").read_bytes()),
              "note": "从当前文件的未修改部分还原修改前求解器；仅关闭 Numba 缓存以便内存重放。"}
    with (RUN / "same_contract_audit.json").open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print("same-contract audit passed: 10 identical routes and objectives")


if __name__ == "__main__":
    main()
