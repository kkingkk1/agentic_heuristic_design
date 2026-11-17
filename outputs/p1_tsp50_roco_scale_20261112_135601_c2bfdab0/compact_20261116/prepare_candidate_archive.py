"""读取、核验和恢复 TSP50 候选归档；本程序不删除文件。"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile


STAGING = Path(__file__).resolve().parent
ROOT = STAGING.parents[2]
ARCHIVE = STAGING / "candidate_archive_v2.zip"
ARCHIVE_SHA256 = "98acd23e9aaefbe35081cdcd105361320ffbc6d8d54d84da67ee5ee2bcbcce8f"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def archive_only_check():
    data = ARCHIVE.read_bytes()
    require(sha256(data) == ARCHIVE_SHA256, "归档哈希与已核验版本不一致。")
    files, candidates, seen = 0, 0, set()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        require(archive.testzip() is None, "归档 CRC 校验失败。")
        index = json.loads(archive.read("index.json"))
        members = sorted({entry["member"] for entry in index.values()})
        require(len(members) == 58 and len(index) == 9604, "归档索引规模不一致。")
        for member in members:
            for number, line in enumerate(archive.read(member).splitlines()):
                row = json.loads(line)
                require(all(row.get(k) is not None for k in
                            ("thought", "generation", "source_role", "operator", "parents")), "候选来源信息缺失。")
                candidates += 1
                for item in row["files"]:
                    name, raw = item["path"], item["content_utf8"].encode("utf-8")
                    require(name not in seen and index[name]["member"] == member
                            and index[name]["candidate_index"] == number, "归档索引错误。")
                    require(len(raw) == item["size_bytes"] == index[name]["size_bytes"]
                            and sha256(raw) == item["sha256"] == index[name]["sha256"], "归档内容校验失败。")
                    seen.add(name)
                    files += 1
        require(seen == set(index) and candidates == 2401 and files == 9604, "归档内容不完整。")
    return candidates, files


def archived_file(archive, index, name):
    entry = index[name]
    row = json.loads(archive.read(entry["member"]).splitlines()[entry["candidate_index"]])
    matches = [x for x in row["files"] if x["path"] == name]
    require(len(matches) == 1, "归档候选文件不唯一。")
    item = matches[0]
    data = item["content_utf8"].encode("utf-8")
    require(len(data) == entry["size_bytes"] and sha256(data) == entry["sha256"], "归档文件校验失败。")
    return data


def restore_candidate(relative):
    # 仅允许一个已登记的原候选目录；已存在且不同的文件绝不覆盖。
    name = relative.replace("\\", "/")
    require(not name.startswith("/") and not re.match(r"^[A-Za-z]:", name), "恢复路径必须为项目相对路径。")
    name = name.rstrip("/")
    parts = name.split("/")
    require(len(parts) == 5 and parts[0] == "outputs" and parts[1] in ("p1_control", "p1_roco")
            and parts[3] == "candidates" and re.fullmatch(r"c\d+", parts[4])
            and all(part not in ("", ".", "..") and ":" not in part for part in parts), "候选目录路径无效。")
    directory = ROOT.joinpath(*parts)
    allowed_base = (ROOT / "outputs" / parts[1]).resolve()
    require(directory.resolve().is_relative_to(allowed_base), "恢复路径越界。")
    with zipfile.ZipFile(ARCHIVE) as archive:
        require(sha256(ARCHIVE.read_bytes()) == ARCHIVE_SHA256, "归档哈希不符。")
        index = json.loads(archive.read("index.json"))
        names = sorted(path for path in index if path.startswith(name + "/"))
        require(len(names) == 4, "归档中候选文件数量异常。")
        payloads = [(ROOT / path, archived_file(archive, index, path)) for path in names]
        for path, data in payloads:
            require(path.resolve().is_relative_to(allowed_base) and path.parent.is_dir(), "恢复目标越界或目录不存在。")
            if path.exists():
                require(path.is_file() and path.read_bytes() == data, f"目标已存在且内容不同：{path}")
        restored = 0
        for path, data in payloads:
            if path.exists():
                continue
            with path.open("xb") as out:
                out.write(data)
            require(path.read_bytes() == data, f"恢复后回读不一致：{path}")
            restored += 1
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("archive-check", "inspect", "restore-candidate"))
    parser.add_argument("path", nargs="?")
    args = parser.parse_args()
    if args.action == "archive-check":
        candidates, files = archive_only_check()
        print(json.dumps({"status": "verified", "candidate_count": candidates, "files": files,
                          "archive_sha256": ARCHIVE_SHA256}, ensure_ascii=False))
        return
    if args.action == "restore-candidate":
        require(args.path, "restore-candidate 需要原始候选目录的项目相对路径。")
        restored = restore_candidate(args.path)
        print(json.dumps({"status": "restored", "new_files": restored}, ensure_ascii=False))
        return
    require(args.path, "inspect 需要原始文件的项目相对路径。")
    data = ARCHIVE.read_bytes()
    require(sha256(data) == ARCHIVE_SHA256, "归档哈希不符。")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        index = json.loads(archive.read("index.json"))
        data = archived_file(archive, index, args.path.replace("\\", "/"))
    sys.stdout.buffer.write(data)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        print(f"候选归档读取失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
