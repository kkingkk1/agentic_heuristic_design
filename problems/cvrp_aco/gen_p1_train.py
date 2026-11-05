"""生成与固定来源逐字节一致的 CVRP 训练集，不覆盖已有文件。"""

from hashlib import sha1, sha256
from io import BytesIO
from pathlib import Path

import numpy as np


DATASET = Path(__file__).resolve().parent / "dataset" / "train50_dataset.npy"
SOURCE_GIT_BLOB = "1b9f2252f5509a2748dc3ff2c7ee5e6a83a0f221"


def build_dataset():
    rng = np.random.RandomState(1234)
    instances = []
    for _ in range(10):
        locations = rng.rand(50, 2)
        demand = rng.randint(1, 10, 50)
        locations = np.concatenate((np.array([[0.5, 0.5]]), locations), axis=0)
        demand = np.concatenate((np.zeros(1), demand))
        instances.append(np.concatenate((demand.reshape(-1, 1), locations), axis=1))
    buffer = BytesIO()
    np.save(buffer, np.array(instances), allow_pickle=False)
    return buffer.getvalue()


def main():
    data = build_dataset()
    git_blob = sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
    if git_blob != SOURCE_GIT_BLOB:
        raise ValueError("生成文件与固定来源的 Git blob 不一致。")
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    with DATASET.open("xb") as output:
        output.write(data)
    print(f"{DATASET} {len(data)} {sha256(data).hexdigest()} {git_blob}")


if __name__ == "__main__":
    main()
