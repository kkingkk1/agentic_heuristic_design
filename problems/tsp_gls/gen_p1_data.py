"""生成来源一致的 GLS 训练集与候选测试集，拒绝覆盖。"""

from hashlib import sha1, sha256
from io import BytesIO
from pathlib import Path

import numpy as np


DATASET_DIR = Path(__file__).resolve().parent / "dataset"
SOURCE_BLOBS = {
    "train200_dataset.npy": "8fef7b54548a15fd942baaaad7d005312d1ef67d",
    "test20_p1_dataset.npy": "c6d5013dafad155a6255908bae583e1b032a9a35",
    "test50_p1_dataset.npy": "e2f1bde00675c4320bd029da17d9322a600c6b36",
    "test100_p1_dataset.npy": "580d4a4811148a02f60b4b0b6b266bc52598dd02",
    "test200_p1_dataset.npy": "cb393810a8f567704a8f32516f615c55ec7647c0",
}


def array_bytes(values):
    buffer = BytesIO()
    np.save(buffer, values, allow_pickle=False)
    return buffer.getvalue()


def build_datasets():
    data = {"train200_dataset.npy": array_bytes(np.random.RandomState(5).random_sample((10, 200, 2)))}
    rng = np.random.RandomState(4)
    for size in (20, 50, 100, 200):
        data[f"test{size}_p1_dataset.npy"] = array_bytes(rng.random_sample((64, size, 2)))
    return data


def git_blob(data):
    return sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def main():
    data = build_datasets()
    for name, expected in SOURCE_BLOBS.items():
        if git_blob(data[name]) != expected:
            raise ValueError(f"{name} 与固定来源数据不一致。")
    paths = {name: DATASET_DIR / name for name in data}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"不覆盖已有文件：{existing}")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in data.items():
        with paths[name].open("xb") as output:
            output.write(content)
        print(f"{paths[name]} {len(content)} {sha256(content).hexdigest()} {git_blob(content)}")


if __name__ == "__main__":
    main()
