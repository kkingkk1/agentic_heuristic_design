"""首批剩余新研发的单并发入口；不改写已冻结的配置或运行时源码。"""

import argparse
from pathlib import Path
import sys
import threading


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts import p1_run_generation, p1_run_roco
from scripts.p0_runtime import digest
from scripts.p1_classified_runtime import ClassifiedRoCoRuntime, ClassifiedTaskRuntime


class SerialModelCalls:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        resources = self.cfg["resources"]
        if resources["max_concurrent_model_calls"] != 4 or resources["max_concurrent_evaluations"] != 4:
            raise ValueError("冻结的并发合同不是预期的模型 4、评价 4。")
        self.model_slots = threading.BoundedSemaphore(1)
        source = Path(__file__).resolve()
        self.protected.append(source)
        self.protected_hashes[str(source)] = digest(source)
        self.log.event("effective_concurrency", frozen_model_concurrency=4, effective_model_concurrency=1,
                       evaluation_concurrency=4, reason="user_requested")


class SerialTaskRuntime(SerialModelCalls, ClassifiedTaskRuntime):
    pass


class SerialRoCoRuntime(SerialModelCalls, ClassifiedRoCoRuntime):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--method", required=True, choices=("proposed", "roco"))
    args, remaining = parser.parse_known_args()
    module = p1_run_generation if args.method == "proposed" else p1_run_roco
    field = "TaskRuntime" if args.method == "proposed" else "RoCoRuntime"
    replacement = SerialTaskRuntime if args.method == "proposed" else SerialRoCoRuntime
    previous, argv = getattr(module, field), sys.argv
    try:
        setattr(module, field, replacement)
        sys.argv = [argv[0], *remaining]
        return module.main()
    finally:
        setattr(module, field, previous)
        sys.argv = argv


if __name__ == "__main__":
    raise SystemExit(main())
