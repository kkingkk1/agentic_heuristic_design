"""生成与固定来源逐字节一致的 OP 训练坐标，不覆盖已有文件。"""

from hashlib import sha1, sha256
from io import BytesIO
from pathlib import Path

import numpy as np


DATASET = Path(__file__).resolve().parent / "dataset" / "train50_dataset.npz"
SOURCE_GIT_BLOB = "437462740e19bc27fffa20b56739d1730adf1d2b"


def build_dataset():
    rng = np.random.RandomState(1234)
    buffer = BytesIO()
    np.savez(buffer, coordinates=rng.rand(5, 50, 2))
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
