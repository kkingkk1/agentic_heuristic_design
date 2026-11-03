"""仅在用户批准最终清单后执行源端清理；须由 Git Bash 启动。"""
from pathlib import Path
import importlib.util
import json
import os
import stat
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
WORK = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("materials", WORK / "migrate_materials.py")
materials = importlib.util.module_from_spec(spec)
spec.loader.exec_module(materials)
spec2 = importlib.util.spec_from_file_location("cleanup", WORK / "prepare_source_cleanup.py")
cleanup = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(cleanup)


def run():
    assert sys.argv[1:] == ["--approved-final-manifest"], "必须先取得用户针对最终清单的确认"
    assert os.environ.get("MSYSTEM"), "必须由Git Bash启动"
    manifest = WORK / "迁移清单.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["source_removal"] == "awaiting_exact_manifest_confirmation"
    roots = [Path(g["path"]).resolve() for g in data["groups"]]
    expected_roots = [(materials.SOURCE / p).resolve() for p in materials.PARTS]
    assert roots == expected_roots and len(set(roots)) == 2
    expected = {Path(f["source"]).resolve() for f in data["files"]}
    actual = {p.resolve() for folder in roots for p in folder.rglob("*") if p.is_file()}
    assert expected == actual and len(expected) == 22
    for folder in roots:
        assert folder.is_relative_to(materials.SOURCE.resolve())
        for path in [folder, *folder.rglob("*")]:
            assert not os.lstat(path).st_file_attributes & 1024, path
    for entry in data["files"]:
        src, dst = Path(entry["source"]), Path(entry["destination"])
        assert src.resolve() in expected
        assert dst.resolve().is_relative_to(materials.DEST.resolve())
        assert src.stat().st_size == entry["bytes"]
        assert materials.digest(src) == materials.digest(dst) == entry["sha256"]
    ps = cleanup.extract.presentations()
    found = [p for p in ps if p.Name == cleanup.extract.SNAPSHOT.name]
    assert len(found) == 1 and found[0].Slides.Count == 17
    live = found[0]
    source = cleanup.Presentation(cleanup.extract.SNAPSHOT)
    for i in (14, 15):
        slide = live.Slides.Item(i)
        prior = source.slides[i-1]
        assert slide.SlideID == prior.slide_id and slide.Shapes.Count == len(prior.shapes)
        for current, old in zip(slide.Shapes, prior.shapes):
            assert current.Name == old.name
            if old.has_text_frame:
                assert current.TextFrame.TextRange.Text.replace("\r", "\n") == old.text
    before = WORK / "源端清理前最新合集.pptx"
    assert not before.exists()
    live.SaveCopyAs(str(before), 24)
    cleanup.edit(live)
    live.Save()
    assert live.Saved and live.Slides.Count == 15
    live.SaveCopyAs(str(cleanup.FINAL), 24)
    a, b = cleanup.extract.drawing.package(before), cleanup.extract.drawing.package(cleanup.FINAL)
    x, y = cleanup.extract.drawing.slide_parts(a), cleanup.extract.drawing.slide_parts(b)
    ids = list(x)[:13]
    assert all(a[x[sid]] == b[y[sid]] for sid in ids), "其他页面发生变化，停止文件删除"
    # 所有删除均在同一Python进程中逐项执行，不扩展目标。
    removed = []
    for entry in data["files"]:
        src = Path(entry["source"])
        extended = "\\\\?\\" + str(src.resolve())
        os.unlink(extended)
        assert not src.exists()
        removed.append(str(src))
    for folder in roots:
        assert not list(folder.iterdir())
        os.rmdir("\\\\?\\" + str(folder))
        assert not folder.exists()
    for entry in data["files"]:
        assert materials.digest(Path(entry["destination"])) == entry["sha256"]
    data["source_removal"] = "complete"
    data["removed_source_files"] = len(removed)
    data["source_presentation"] = {"removed_original_pages": [14, 15], "current_page_count": 15,
                                   "latest_backup": str(before), "other_pages_1_to_13_preserved": True}
    data["acl_changes"] = []
    data["attribute_changes"] = []
    manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"removed_files": len(removed), "removed_bytes": data["total_bytes"],
                      "source_presentation_pages": 15, "destination_copy_verified": True,
                      "acl_changes": False, "attribute_changes": False,
                      "onedrive_cloud_sync_completion": "not_verified"}, ensure_ascii=False))


def finish_empty_directories():
    """恢复已完成文件删除后的空目录清理，不扩展文件删除范围。"""
    assert os.environ.get("MSYSTEM"), "必须由Git Bash启动"
    manifest = WORK / "迁移清单.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["source_removal"] in {"awaiting_exact_manifest_confirmation", "files_removed_empty_directories_pending"}
    roots = [Path(g["path"]).resolve() for g in data["groups"]]
    assert roots == [(materials.SOURCE / p).resolve() for p in materials.PARTS]
    assert len(data["files"]) == 22
    for item in data["files"]:
        assert not Path(item["source"]).exists()
        assert materials.digest(Path(item["destination"])) == item["sha256"]
    attribute_changes, removed_dirs = [], []
    for root in roots:
        assert root.is_relative_to(materials.SOURCE.resolve())
        if not root.exists():
            continue
        children = list(root.rglob("*"))
        assert not any(p.is_file() for p in children), "发现清单外文件，停止清理"
        assert not any(os.lstat(p).st_file_attributes & 1024 for p in [root, *children])
        dirs = sorted([p for p in children if p.is_dir()], key=lambda p: len(p.parts), reverse=True) + [root]
        for folder in dirs:
            assert not list(folder.iterdir())
            old_attrs = os.lstat(folder).st_file_attributes
            changed = bool(old_attrs & stat.FILE_ATTRIBUTE_READONLY)
            try:
                if changed:
                    os.chmod(folder, stat.S_IWRITE)
                    assert os.lstat(folder).st_file_attributes == old_attrs & ~stat.FILE_ATTRIBUTE_READONLY
                os.rmdir("\\\\?\\" + str(folder.resolve()))
                assert not folder.exists()
            except Exception:
                if changed and folder.exists():
                    os.chmod(folder, stat.S_IREAD)
                raise
            if changed:
                attribute_changes.append({"path": str(folder), "original_attributes": old_attrs,
                                          "change": "clear_readonly_before_removing_empty_directory"})
            removed_dirs.append(str(folder))
    assert all(not p.exists() for p in roots)
    data["source_removal"] = "complete"
    data["removed_source_files"] = 22
    data["removed_empty_directories"] = removed_dirs
    data["source_presentation"] = {"removed_original_pages": [14, 15], "current_page_count": 15,
                                   "latest_backup": str(WORK / "源端清理前最新合集.pptx"),
                                   "other_pages_1_to_13_preserved": True}
    data["acl_changes"] = []
    data["attribute_changes"] = attribute_changes
    data["acl_review"] = "只读核查存在Everyone Deny DeleteSubdirectoriesAndFiles；未修改ACL"
    manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"removed_files": 22, "removed_bytes": data["total_bytes"],
                      "removed_empty_directories": len(removed_dirs), "acl_changes": False,
                      "readonly_cleared_for_removed_directories": len(attribute_changes),
                      "destination_copy_verified": True}, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["--finish-approved-empty-directories"]:
        finish_empty_directories()
    else:
        run()
