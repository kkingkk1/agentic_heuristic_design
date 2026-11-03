"""复制方案三材料，导出完整可见消息，并生成逐文件核验清单；不删除源文件。"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import re
import shutil
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
SOURCE = Path(r"E:\OneDrive\project\static_determine\roco_paperbanana")
DEST = Path(r"E:\OneDrive\project\static_determine\agentic_heuristic_design")
WORK = DEST / "outputs/migration_20261021"
THREAD = "01a1574e-adf4-7030-aea3-61bda75d9b62"
PARTS = [Path("outputs/satellite_multiagent_briefing") / n for n in
         ("scheme_three_20261021", "scheme_three_slim_20261021")]


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory():
    assert SOURCE.resolve().parent == DEST.resolve().parent
    files, groups = [], []
    for rel in PARTS:
        folder = SOURCE / rel
        assert folder.resolve().is_relative_to(SOURCE.resolve())
        entries = sorted(folder.rglob("*"))
        assert not any(os.lstat(p).st_file_attributes & 1024 for p in [folder, *entries]), "重解析点"
        selected = [p for p in entries if p.is_file()]
        groups.append({"path": str(folder), "type": "directory", "file_count": len(selected),
                       "bytes": sum(p.stat().st_size for p in selected), "reparse_points": False})
        for src in selected:
            files.append({"source": str(src), "destination": str(DEST / src.relative_to(SOURCE)),
                          "bytes": src.stat().st_size, "sha256": digest(src)})
    assert len({x["source"].lower() for x in files}) == len(files)
    return {"source_root": str(SOURCE), "destination_root": str(DEST),
            "groups": groups, "files": files, "file_count": len(files),
            "total_bytes": sum(x["bytes"] for x in files),
            "source_removal": "awaiting_exact_manifest_confirmation",
            "onedrive_process": "running_at_preflight", "cloud_sync_completion": "not_verified"}


def copy_materials():
    data = inventory()
    for item in data["files"]:
        src, dst = Path(item["source"]), Path(item["destination"])
        assert not dst.exists(), dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        assert dst.stat().st_size == item["bytes"] and digest(dst) == item["sha256"]
    (WORK / "迁移清单.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"copied_files": data["file_count"], "verified_bytes": data["total_bytes"],
                      "source_files_removed": 0}, ensure_ascii=False))


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(x.get("text", "") for x in content if isinstance(x, dict) and isinstance(x.get("text"), str))
    return ""


def messages():
    paths = sorted(Path(r"C:\Users\a_wei\.codex\sessions").rglob(f"*{THREAD}*"))
    rows, seen, origins = [], set(), []
    for p in paths:
        lines = p.read_text(encoding="utf-8").splitlines()
        valid = []
        for line in lines:
            try:
                valid.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not valid or valid[0].get("payload", {}).get("id") != THREAD:
            continue
        origins.append({"path": str(p), "bytes_at_read": p.stat().st_size,
                        "first_timestamp": valid[0].get("timestamp"), "last_timestamp": valid[-1].get("timestamp")})
        for row in valid:
            payload = row.get("payload", {})
            item = payload.get("item", {})
            if row.get("type") != "event_msg" or payload.get("type") != "item_completed":
                continue
            if item.get("type") not in {"UserMessage", "AgentMessage"}:
                continue
            identity = item.get("id")
            if not identity or identity in seen:
                continue
            seen.add(identity)
            rows.append({"id": identity, "thread_id": THREAD, "timestamp": row.get("timestamp"),
                         "role": "user" if item["type"] == "UserMessage" else "assistant",
                         "phase": item.get("phase"), "content": item.get("content", [])})
    rows.sort(key=lambda x: x["timestamp"] or "")
    assert rows and content_text(rows[0]["content"]).strip() == "roco的输入是啥"
    assert any("说错了，是方案3" in content_text(x["content"]) for x in rows)
    return rows, origins


def export_history():
    rows, origins = messages()
    hist = DEST / "vault/History"
    hist.mkdir(parents=True, exist_ok=True)
    jsonl = hist / "查找_ROCO_输入_可见消息.jsonl"
    jsonl.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    md = ["# 查找 ROCO 输入：完整可见会话历史", "", f"会话 ID：`{THREAD}`。", "",
          f"导出时间：{now}。", "", f"包含 {len(rows)} 条用户与助手可见消息，按原顺序保存。", "",
          "这是本次导出的快照；应用内原会话与侧栏项目归属保持原状。", "",
          "逐条原始可见消息字段见 [JSONL](查找_ROCO_输入_可见消息.jsonl)。", "",
          "模型内部推理、系统与开发者指令、工具内部数据不属于可见会话，未导出。", "",
          "历史方案可能已被后续决定替代；最终设计以当前框架文档为准。", ""]
    asset_paths = set()
    for i, row in enumerate(rows, 1):
        text = content_text(row["content"])
        label = "用户" if row["role"] == "user" else "助手"
        md += [f"## {i}. {label}", "", f"时间：{row['timestamp']}；消息 ID：`{row['id']}`。", "", text.rstrip(), ""]
        asset_paths.update(re.findall(r"[A-Za-z]:[\\/][^\r\n\"<>]*?codex-clipboard-[^\s\"<>/\\]+\.png", text))
        for c in row["content"] if isinstance(row["content"], list) else []:
            if isinstance(c, dict):
                for k in ("path", "image_path"):
                    if isinstance(c.get(k), str):
                        asset_paths.add(c[k])
    assets, normalized_assets = [], set()
    for raw in sorted(asset_paths):
        src = Path(raw)
        normalized = str(src.resolve()).casefold()
        if normalized in normalized_assets:
            continue
        normalized_assets.add(normalized)
        found = src.exists() and src.is_file()
        info = {"source": raw, "available": found}
        if found:
            dst = hist / "attachments" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            assert digest(src) == digest(dst)
            info["destination"] = str(dst)
        assets.append(info)
    if assets:
        md += ["## 附件保存情况", ""]
        for item in assets:
            if item["available"]:
                md.append(f"- [{Path(item['destination']).name}](attachments/{Path(item['destination']).name})。")
            else:
                md.append(f"- 原附件当前不可用：`{item['source']}`。原消息引用保留。")
    (hist / "查找_ROCO_输入_完整会话.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    architecture = [content_text(x["content"]) for x in rows if x["role"] == "assistant"
                    and "这版架构统一为" in content_text(x["content"])
                    and "不另建全量历史候选" in content_text(x["content"])]
    assert len(architecture) == 1
    knowledge = DEST / "vault/Knowledge/通用启发式自动设计框架.md"
    knowledge.parent.mkdir(parents=True, exist_ok=True)
    if not knowledge.exists():
        knowledge.write_text("# 通用启发式自动设计框架\n\n"
                             "更新日期：2026-10-21。状态：用户已确认的方案三，尚未实现或验证性能。\n\n"
                             "以下迁入本会话最后确认的完整架构说明。早期方案仅作为历史，不覆盖本设计。\n\n"
                             + architecture[0] + "\n", encoding="utf-8")
    meta = {"thread_id": THREAD, "title": "查找 ROCO 输入", "source_project": str(SOURCE),
            "exported_at": now, "message_count": len(rows),
            "user_messages": sum(x["role"] == "user" for x in rows),
            "assistant_messages": sum(x["role"] == "assistant" for x in rows),
            "sources": origins, "assets": assets, "app_thread_moved": False,
            "app_thread_move": "deferred_by_user"}
    (WORK / "会话导出核验.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("message_count", "user_messages", "assistant_messages", "app_thread_moved")}, ensure_ascii=False))


def verify():
    data = json.loads((WORK / "迁移清单.json").read_text(encoding="utf-8"))
    for item in data["files"]:
        if data["source_removal"] == "complete":
            assert not Path(item["source"]).exists()
        else:
            assert Path(item["source"]).exists()
        assert digest(Path(item["destination"])) == item["sha256"]
    exported = [json.loads(x) for x in (DEST / "vault/History/查找_ROCO_输入_可见消息.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({x["id"] for x in exported}) == len(exported)
    assert all(x["role"] in {"user", "assistant"} for x in exported)
    assert all(any(c.get("text") for c in x["content"] if isinstance(c, dict)) for x in exported)
    print(json.dumps({"copied_files_verified": len(data["files"]), "messages_verified": len(exported),
                      "source_removal": data["source_removal"]}, ensure_ascii=False))


if __name__ == "__main__":
    {"copy": copy_materials, "history": export_history, "verify": verify}[sys.argv[1]]()
