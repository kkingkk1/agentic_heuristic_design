"""保留冻结源码，为 P1 两种方法启用独立的模型错误分类入口。"""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import p1_run_generation, p1_run_roco
from scripts.p1_classified_runtime import ClassifiedRoCoRuntime, ClassifiedTaskRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--method", required=True, choices=("proposed", "roco"))
    args, remaining = parser.parse_known_args()
    module = p1_run_generation if args.method == "proposed" else p1_run_roco
    field = "TaskRuntime" if args.method == "proposed" else "RoCoRuntime"
    replacement = ClassifiedTaskRuntime if args.method == "proposed" else ClassifiedRoCoRuntime
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
