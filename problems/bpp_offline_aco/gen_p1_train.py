"""生成与固定来源逐字节一致的离线 BPP 训练集，不覆盖已有文件。"""

from hashlib import sha1, sha256
from io import BytesIO
from pathlib import Path

import numpy as np


DATASET = Path(__file__).resolve().parent / "dataset" / "train500_dataset.npz"
SOURCE_GIT_BLOB = "6184f952df4cf3a5e8c8da1f9d9052c8b2043735"


def build_dataset():
    rng = np.random.RandomState(5)
    buffer = BytesIO()
    np.savez(buffer, demands=rng.randint(20, 101, size=(5, 500)))
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
