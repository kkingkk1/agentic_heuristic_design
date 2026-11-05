"""将已接入的 DeepACO 参考送入任务评价；预算监督由外层负责。"""

import argparse
import importlib.util
from pathlib import Path
import sys
import threading


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.deepaco.reference import DeepACOReference  # noqa: E402


_TASK_IMPORT_LOCK = threading.RLock()


def load_task(task):
    name = {"tsp": "tsp_aco", "op": "op_aco", "bpp": "bpp_offline_aco",
            "cvrp": "cvrp_aco", "mkp": "mkp_aco"}[task]
    directory = ROOT / "problems" / name
    spec = importlib.util.spec_from_file_location("p1_deepaco_task_aco", directory / "aco.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    with _TASK_IMPORT_LOCK:
        absent = object()
        previous = sys.modules.get("aco", absent)
        try:
            sys.modules["aco"] = source
            spec = importlib.util.spec_from_file_location("p1_deepaco_task_eval", directory / "eval.py")
            evaluator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluator)
        finally:
            if previous is absent:
                sys.modules.pop("aco", None)
            else:
                sys.modules["aco"] = previous
    return evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("tsp", "op", "bpp", "cvrp", "mkp"))
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--phase", required=True, choices=("train", "test"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--k-sparse", type=int)
    args = parser.parse_args()
    if (args.task in ("tsp", "op") and (args.k_sparse is None or not 1 <= args.k_sparse < args.size)
            or args.task not in ("tsp", "op") and args.k_sparse is not None):
        parser.error("TSP 和 OP 必须指定有效的 --k-sparse；其他任务不接受该参数。")
    evaluator = load_task(args.task)
    model = DeepACOReference(args.task, args.checkpoint)
    sys.argv = [str(Path(evaluator.__file__)), str(args.size), str(ROOT), args.phase]
    if args.task == "tsp":
        from scripts.p1_deepaco_tsp import run
        run(evaluator, model, size=args.size, phase=args.phase, k_sparse=args.k_sparse)
    elif args.task == "op":
        evaluator.main(deepaco_model=model, k_sparse=args.k_sparse)
    else:
        evaluator.main(deepaco_model=model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
