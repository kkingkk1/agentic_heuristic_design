"""核验并单独冻结 P1 训练集和测试集；不评价任何候选。"""

import argparse
import ast
from hashlib import sha1, sha256
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.p1_runtime import TRAINING, TRAINING_PARAMETERS  # noqa: E402
from scripts.p1_generate_test_datasets import SOURCE_BLOBS as ACO_SOURCE_BLOBS  # noqa: E402
from problems.tsp_gls.gen_p1_data import SOURCE_BLOBS as GLS_SOURCE_BLOBS  # noqa: E402


CONTRACT = ROOT / "cfg/p1_data_contract.json"
TEST_SIZES = {"tsp_aco": (50, 100, 200), "op_aco": (50, 100, 200, 500),
              "cvrp_aco": (50, 100, 200, 500), "mkp_aco": (100, 200, 300, 500, 700),
              "bpp_offline_aco": (500, 700, 1000), "tsp_gls": (20, 50, 100, 200)}
SOURCE_COMMIT = "4ad8cfc55056f2d2da94689366ac346eaa620505"


def file_identity(path):
    content = path.read_bytes()
    return {"sha256": sha256(content).hexdigest(),
            "git_blob": sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest(),
            "bytes": len(content)}


def arrays_for(path, task, size, count):
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        try:
            arrays = {key: data[key].copy() for key in data.files}
        finally:
            data.close()
    else:
        arrays = {"values": data}
    expected = {
        "tsp_aco": {"values": (count, size, 2)},
        "op_aco": {"coordinates": (count, size, 2)},
        "cvrp_aco": {"values": (count, size + 1, 3)},
        "mkp_aco": {"prizes": (count, size), "weights": (count, size, 5)},
        "bpp_offline_aco": {"demands": (count, size)},
        "tsp_gls": {"values": (count, size, 2)},
    }[task]
    if set(arrays) != set(expected) or any(arrays[key].shape != shape for key, shape in expected.items()):
        raise ValueError(f"{path} 的数组字段或形状不符。")
    if any(array.dtype.kind not in "fiu" or not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError(f"{path} 含无效数值。")
    if task in ("tsp_aco", "op_aco", "tsp_gls"):
        values = arrays["coordinates"] if task == "op_aco" else arrays["values"]
        if values.dtype.kind != "f" or (values < 0).any() or (values >= 1).any():
            raise ValueError(f"{path} 的坐标范围不符。")
    if task == "cvrp_aco":
        values = arrays["values"]
        if (values.dtype.kind != "f" or not (values[:, 0, 0] == 0).all() or not (values[:, 0, 1:] == 0.5).all()
                or (values[:, 1:, 0] < 1).any() or (values[:, 1:, 0] > 9).any()
                or not np.equal(values[:, 1:, 0], np.floor(values[:, 1:, 0])).all()
                or (values[:, 1:, 1:] < 0).any() or (values[:, 1:, 1:] >= 1).any()):
            raise ValueError(f"{path} 的仓库或需求无效。")
    if task == "mkp_aco" and (arrays["prizes"].dtype.kind != "f" or arrays["weights"].dtype.kind != "f"
                              or (arrays["prizes"] < 0).any() or (arrays["weights"] < 0).any()):
        raise ValueError(f"{path} 的价值或权重无效。")
    if task == "bpp_offline_aco" and (arrays["demands"].dtype.kind not in "iu"
                                      or (arrays["demands"] < 20).any() or (arrays["demands"] > 100).any()):
        raise ValueError(f"{path} 的物品大小无效。")
    return arrays


def instance_hashes(arrays, count):
    hashes = []
    for index in range(count):
        hasher = sha256()
        for key in sorted(arrays):
            value = np.ascontiguousarray(arrays[key][index])
            hasher.update(key.encode("utf-8") + b"\0" + str(value.dtype).encode() + b"\0")
            hasher.update(str(value.shape).encode() + b"\0" + value.tobytes())
        hashes.append(hasher.hexdigest())
    return hashes


def test_constants(name):
    path = ROOT / "problems" / name / "eval.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"TEST_DATASET_SHA256", "TEST_ITERATIONS", "TEST_PARAMETERS", "TEST_MAX_LEN", "TEST_SEED_BASE"}
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in wanted:
                values[key] = ast.literal_eval(node.value)
    return values


def build_contract():
    tasks = {}
    for name, (supervisor, train_path, train_expected) in TRAINING.items():
        train_count, ants, train_iterations, train_seed = TRAINING_PARAMETERS[name]
        constants = test_constants(name)
        if constants.get("TEST_SEED_BASE") != 2026110400:
            raise ValueError(f"{name} 的测试随机种子基础值与合同不符。")
        train_size = 200 if name == "tsp_gls" else 100 if name == "mkp_aco" else 500 if name == "bpp_offline_aco" else 50
        identity = file_identity(train_path)
        if identity["sha256"] != train_expected:
            raise ValueError(f"{train_path} 的训练集哈希不符。")
        train_arrays = arrays_for(train_path, name, train_size, train_count)
        train_hashes = instance_hashes(train_arrays, train_count)
        train_instances = set(train_hashes)
        if len(train_instances) != train_count:
            raise ValueError(f"{name} 的训练集包含重复实例。")
        train = {"path": train_path.relative_to(ROOT).as_posix(), **identity, "instances": train_count,
                 "size": train_size, "generator_seed": train_seed,
                 "evaluation_seeds": [20261103 + i for i in range(train_count)],
                 "iterations": train_iterations, "ants": ants}
        tests = []
        for size in TEST_SIZES[name]:
            extension = "npz" if name in ("op_aco", "mkp_aco", "bpp_offline_aco") else "npy"
            path = train_path.parent / f"test{size}_p1_dataset.{extension}"
            identity = file_identity(path)
            if identity["sha256"] != constants["TEST_DATASET_SHA256"][size]:
                raise ValueError(f"{path} 的测试集哈希不符。")
            arrays = arrays_for(path, name, size, 64)
            test_hashes = instance_hashes(arrays, 64)
            if len(set(test_hashes)) != 64:
                raise ValueError(f"{name} 规模 {size} 的测试集包含重复实例。")
            if size == train_size and train_instances.intersection(test_hashes):
                raise ValueError(f"{name} 规模 {size} 的训练和测试包含相同实例。")
            source_blob = ACO_SOURCE_BLOBS.get((name, size))
            if name == "tsp_gls":
                source_blob = GLS_SOURCE_BLOBS[path.name]
            if source_blob is not None and identity["git_blob"] != source_blob:
                raise ValueError(f"{path} 与登记来源的 Git blob 不符。")
            test = {"path": path.relative_to(ROOT).as_posix(), **identity, "size": size, "instances": 64,
                    "evaluation_seed_base": constants["TEST_SEED_BASE"] + size * 100,
                    "iterations": constants["TEST_PARAMETERS"][size][1] if name == "tsp_gls" else constants["TEST_ITERATIONS"],
                    "source_git_blob_verified": source_blob is not None,
                    "origin_note": ("与已登记来源 Git blob 相同，不证明是论文原始实例。" if source_blob is not None else
                                    "按已登记生成器延续随机序列生成；未证明为论文原始实例。")}
            if name == "tsp_gls":
                test["perturbation_moves"] = constants["TEST_PARAMETERS"][size][0]
            if name == "op_aco":
                test["max_path_length"] = constants["TEST_MAX_LEN"][size]
            tests.append(test)
        tasks[name] = {"train": train, "test": tests, "training_supervisor": Path(supervisor.__file__).relative_to(ROOT).as_posix()}
    return {"status": "frozen", "contract_id": "p1_data_20261104_v1", "scope": "只冻结所列实例文件和实例求解种子；不冻结方法、重复数或总预算。",
            "source_candidate_commit": SOURCE_COMMIT, "notes": ["训练和同规模测试实例无完全重复。",
            "来源一致性不证明这些数据是 RoCo 论文的原始实例。", "测试实例未用于候选评价。"], "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="写入尚不存在的数据合同；默认只核验")
    args = parser.parse_args()
    try:
        contract = build_contract()
        if args.freeze:
            with CONTRACT.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(contract, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        print(f"数据核验通过：{len(contract['tasks'])} 个任务，{sum(len(task['test']) for task in contract['tasks'].values())} 个测试集。")
        if args.freeze:
            print(f"数据合同：{CONTRACT}")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"数据未冻结：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
