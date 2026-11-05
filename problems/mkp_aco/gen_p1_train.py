"""生成与固定来源逐字节一致的 MKP 训练集，不覆盖已有文件。"""

from hashlib import sha1, sha256
from io import BytesIO
from pathlib import Path

import numpy as np


DATASET = Path(__file__).resolve().parent / "dataset" / "train100_dataset.npz"
SOURCE_GIT_BLOB = "d107465e7802149c58ccd00fac71bcfc00ec12ca"


def build_dataset():
    rng = np.random.RandomState(1234)
    prizes, weights = [], []
    for _ in range(5):
        prize = rng.rand(100)
        weight = rng.rand(100, 5)
        capacity = rng.uniform(low=weight.max(0), high=weight.sum(0))
        prizes.append(prize)
        weights.append(weight / capacity.reshape(1, *capacity.shape))
    buffer = BytesIO()
    np.savez(buffer, prizes=np.stack(prizes), weights=np.stack(weights))
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
