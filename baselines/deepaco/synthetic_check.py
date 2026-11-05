"""用真实 PyG 和合成实例检查 DeepACO 权重及网络；不读取正式数据。"""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import PurePosixPath
import time

import numpy as np
import torch
import torch_geometric

from baselines.deepaco.reference import CHECKPOINT_BLOBS, DeepACOReference


def main():
    seed = 7451
    rng = np.random.RandomState(seed)
    begin = time.monotonic()
    models = {}
    for name in sorted(CHECKPOINT_BLOBS):
        path = PurePosixPath(name)
        task, checkpoint = path.parent.name, path.stem
        # 构造器核对来源 Git blob，以 weights_only=True 加载并严格匹配网络参数。
        models[(task, checkpoint)] = DeepACOReference(task, checkpoint)

    cvrp_coordinates = np.vstack((np.array([[0.5, 0.5]]), rng.random_sample((20, 2))))
    cvrp_demands = np.concatenate((np.zeros(1), rng.randint(1, 10, size=20)))
    cases = (
        ("tsp", "tsp20", {"coordinates": rng.random_sample((20, 2)), "k_sparse": 19}, 20),
        ("op", "op100", {"coordinates": rng.random_sample((100, 2)), "k_sparse": 20}, 100),
        ("cvrp", "cvrp20", {"coordinates": cvrp_coordinates, "demands": cvrp_demands}, 21),
        ("mkp", "mkp300", {"prize": rng.random_sample(300),
                            "weight": rng.random_sample((300, 5)) / 300}, 300),
        ("bpp", "bpp120", {"demands": rng.randint(20, 101, size=120)}, 121),
    )
    results = []
    for task, checkpoint, inputs, size in cases:
        model = models[(task, checkpoint)]
        case_begin = time.monotonic()
        first = model.predict(**inputs)
        second = model.predict(**inputs)
        if (first.shape != (size, size) or not np.isfinite(first).all()
                or (first <= 0).any() or not np.array_equal(first, second)):
            raise ValueError(f"DeepACO {task}/{checkpoint} 合成推理结果无效或不可重复。")
        results.append({"task": task, "checkpoint": checkpoint, "matrix_shape": list(first.shape),
                        "compute_dtype": str(model.dtype), "output_dtype": str(first.dtype),
                        "minimum": float(first.min()), "maximum": float(first.max()),
                        "sha256": hashlib.sha256(first.tobytes()).hexdigest(),
                        "repeated_output_equal": True,
                        "elapsed_seconds": time.monotonic() - case_begin})
    print(json.dumps({"status": "passed", "at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
                      "device": "cpu", "formal_test_data_used": False,
                      "torch": torch.__version__, "numpy": np.__version__,
                      "torch_geometric": torch_geometric.__version__,
                      "pyparsing": importlib.metadata.version("pyparsing"),
                      "xxhash": importlib.metadata.version("xxhash"),
                      "torch_num_threads": torch.get_num_threads(),
                      "weights_strictly_loaded": sorted(CHECKPOINT_BLOBS), "cases": results,
                      "elapsed_seconds": time.monotonic() - begin}, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
