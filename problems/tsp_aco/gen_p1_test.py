"""按已登记的上游随机序列生成 P1 TSP-ACO 测试实例，不覆盖已有文件。"""

from hashlib import sha256
from pathlib import Path

import numpy as np


DATASET_DIR = Path(__file__).resolve().parent / "dataset"
TRAIN = DATASET_DIR / "train50_dataset.npy"
SIZES = (50, 100, 200)


def datasets():
    """复现上游 50/100 测试数据，再延续其随机流生成项目的 200 数据。"""
    rng = np.random.RandomState(1234)
    train = rng.rand(5, 50, 2)
    for size in (20, 50, 100):
        rng.rand(64, size, 2)  # 上游验证集占用的随机数，不写入项目。
    generated = {}
    for size in (20, 50, 100):
        values = rng.rand(64, size, 2)
        if size in (50, 100):
            generated[size] = values
    generated[200] = rng.rand(64, 200, 2)
    return train, generated


def main():
    train, generated = datasets()
    if not np.array_equal(np.load(TRAIN, allow_pickle=False), train):
        raise ValueError("本地训练集与上游生成序列不一致；停止生成测试集。")
    targets = {size: DATASET_DIR / f"test{size}_p1_dataset.npy" for size in SIZES}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"测试集已存在，不覆盖：{existing}")
    for size, path in targets.items():
        with path.open("xb") as output:
            np.save(output, generated[size], allow_pickle=False)
        print(f"{path.name} {path.stat().st_size} {sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
