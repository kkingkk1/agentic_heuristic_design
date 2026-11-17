"""按文件无损压缩 TSP50 事件日志，保留原路径；本程序不删除文件。"""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


STAGING = Path(__file__).resolve().parent
ROOT = STAGING.parents[2]
JOURNAL = STAGING / "event_compression_journal.jsonl"
COMPRESSED = 0x800
FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def physical_size(path):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel.GetCompressedFileSizeW
    function.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
    function.restype = ctypes.c_ulong
    high = ctypes.c_ulong(0)
    ctypes.set_last_error(0)
    low = function(str(path), ctypes.byref(high))
    error = ctypes.get_last_error()
    require(not (low == 0xFFFFFFFF and error), f"无法读取压缩大小：{path}，错误 {error}")
    return (high.value << 32) | low


def cloud_tag(path):
    result = subprocess.run(["fsutil", "reparsepoint", "query", str(path)], capture_output=True,
                            creationflags=FLAGS, check=False)
    require(result.returncode == 0 and b"0x9000601a" in result.stdout.lower(),
            f"无法确认 CLOUD_6 标记：{path}")


def compact(path, compress):
    result = subprocess.run(["compact", "/C" if compress else "/U", "/Q", str(path)],
                            capture_output=True, creationflags=FLAGS, check=False)
    require(result.returncode == 0, f"文件压缩属性更改失败：{path}，退出码 {result.returncode}")


def snapshot(path):
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest(path), "python_attributes": stat.st_file_attributes,
            "compressed": bool(stat.st_file_attributes & COMPRESSED), "physical_bytes": physical_size(path)}


def write(log, record):
    log.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    log.flush()
    os.fsync(log.fileno())


def main():
    require(not JOURNAL.exists(), "压缩记录已存在，不得覆盖或重复执行。")
    raw = subprocess.run(["git", "ls-files", "-z", "--", "outputs/p1_control", "outputs/p1_roco"],
                         cwd=ROOT, capture_output=True, check=True).stdout
    paths = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        relative = Path(os.fsdecode(item))
        if len(relative.parts) == 4 and relative.name == "events.jsonl":
            path = (ROOT / relative).resolve()
            require(path.is_relative_to(ROOT.resolve()) and path.is_file(), f"事件文件路径无效：{path}")
            paths.append(path)
    require(len(paths) == 77 and len({str(p).casefold() for p in paths}) == 77, "事件文件集合不一致。")
    before = []
    for path in sorted(paths, key=str):
        cloud_tag(path)
        before.append(snapshot(path))
    changed = []
    with JOURNAL.open("x", encoding="utf-8", newline="\n") as log:
        write(log, {"event": "before", "files": before})
        try:
            for old in before:
                path = Path(old["path"])
                if not old["compressed"]:
                    write(log, {"event": "starting", "path": str(path)})
                    changed.append(old)
                    compact(path, True)
                cloud_tag(path)
                new = snapshot(path)
                require(new["compressed"] and new["size_bytes"] == old["size_bytes"]
                        and new["sha256"] == old["sha256"] and new["mtime_ns"] == old["mtime_ns"],
                        f"压缩后内容或时间不一致：{path}")
                write(log, {"event": "verified", **new})
            total_before = sum(item["physical_bytes"] for item in before)
            total_after = sum(physical_size(Path(item["path"])) for item in before)
            write(log, {"event": "complete", "files": len(before), "changed": len(changed),
                        "physical_bytes_before": total_before, "physical_bytes_after": total_after})
            print(json.dumps({"status": "compressed", "files": len(before), "changed": len(changed),
                              "physical_bytes_before": total_before, "physical_bytes_after": total_after}, ensure_ascii=False))
        except BaseException as exc:
            failed = []
            for old in reversed(changed):
                path = Path(old["path"])
                try:
                    if path.stat().st_file_attributes & COMPRESSED:
                        compact(path, False)
                    now = snapshot(path)
                    require(now["sha256"] == old["sha256"] and now["compressed"] == old["compressed"],
                            f"压缩属性回退未通过：{path}")
                except (OSError, ValueError) as rollback_error:
                    failed.append({"path": str(path), "error": str(rollback_error)})
            write(log, {"event": "rollback", "reason": str(exc), "rollback_failures": failed})
            raise


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"事件日志压缩失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
