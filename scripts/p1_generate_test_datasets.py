"""重建 ACO 测试候选数据；拒绝覆盖文件，也不评价候选。"""

from hashlib import sha1, sha256
from io import BytesIO
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_BLOBS = {
    ("op_aco", 50): "82c7c8b14b2e3e1bfd1e2316f8155634e13c7901",
    ("op_aco", 100): "bc37824a0da2cbd57092a7872d14b71a90476435",
    ("op_aco", 200): "77f8e25bd4d8dd7fcad3897555cc588bbd4208b5",
    ("cvrp_aco", 50): "b4d4f5b2353eb5ec5d7e2607b876bab49ca6f1cd",
    ("cvrp_aco", 100): "097f81f684874dd19bd04cd32c666e198ce5ca58",
    ("mkp_aco", 100): "bb9268c51d13e50a7806e73f119d92493d3d77bb",
    ("mkp_aco", 200): "56215dc98cc9df06bf841d92a59c023d6b570e53",
    ("mkp_aco", 300): "eda26297310c65e515cfd65a2d4de8bb22760285",
    ("mkp_aco", 500): "baca251205157ae2d9f420a4a4749567a55ec1a5",
    ("bpp_offline_aco", 500): "d632f5fe0f97f84934dfcb389c36782506cbf414",
    ("bpp_offline_aco", 1000): "cf9be4cbb97b230790153329cf0e61d9fe83a785",
}


def archive(**arrays):
    output = BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def npy(array):
    output = BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def build_datasets():
    data = {}

    rng = np.random.RandomState(4567)
    for size in (50, 100, 200, 500):
        data["op_aco", size] = archive(coordinates=rng.rand(64, size, 2))

    rng = np.random.RandomState(1234)

    def cvrp_instance(size):
        locations = rng.rand(size, 2)
        demands = rng.randint(1, 10, size, dtype=np.int32)
        locations = np.concatenate((np.array([[0.5, 0.5]]), locations))
        demands = np.concatenate((np.zeros(1), demands))
        return np.concatenate((demands.reshape(-1, 1), locations), axis=1)

    for _ in range(10):
        cvrp_instance(50)
    for size in (20, 50, 100):
        for _ in range(64):
            cvrp_instance(size)
    for size in (20, 50, 100, 200, 500):
        instances = np.array([cvrp_instance(size) for _ in range(64)])
        if size != 20:
            data["cvrp_aco", size] = npy(instances)

    rng = np.random.RandomState(4567)
    for size in (100, 200, 300, 500, 700):
        prizes, weights = [], []
        for _ in range(64):
            prize = rng.rand(size)
            weight = rng.rand(size, 5)
            capacity = rng.uniform(low=weight.max(0), high=weight.sum(0))
            prizes.append(prize)
            weights.append(weight / capacity.reshape(1, 5))
        data["mkp_aco", size] = archive(prizes=np.stack(prizes), weights=np.stack(weights))

    rng = np.random.RandomState(4)
    for size in (120, 500, 1000, 700):
        demands = rng.randint(20, 101, size=(64, size), dtype=np.int32)
        if size != 120:
            data["bpp_offline_aco", size] = archive(demands=demands)
    return data


def git_blob(data):
    return sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def dataset_path(task, size):
    suffix = "npy" if task == "cvrp_aco" else "npz"
    return ROOT / "problems" / task / "dataset" / f"test{size}_p1_dataset.{suffix}"


def main():
    data = build_datasets()
    for key, expected in SOURCE_BLOBS.items():
        if git_blob(data[key]) != expected:
            raise ValueError(f"{key} 与固定来源的 Git blob 不一致。")
    paths = {key: dataset_path(*key) for key in data}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"不覆盖已有文件：{existing}")
    for key, content in data.items():
        path = paths[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(content)
        print(f"{path} {len(content)} {sha256(content).hexdigest()} {git_blob(content)}")


if __name__ == "__main__":
    main()
