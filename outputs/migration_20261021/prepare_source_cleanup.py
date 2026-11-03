"""生成原合集迁出方案三后的预览；不操作活动演示。"""
from pathlib import Path
import importlib.util
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
from pptx import Presentation

ROOT = Path(r"E:\OneDrive\project\static_determine\agentic_heuristic_design")
WORK = ROOT / "outputs/migration_20261021"
spec = importlib.util.spec_from_file_location("extract", WORK / "extract_presentation.py")
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)
FINAL = WORK / "源合集迁出方案三核验稿.pptx"


def edit(p):
    assert p.Slides.Count == 17
    assert "方案三" in p.Slides.Item(14).Shapes.Item("页面标题").TextFrame.TextRange.Text
    assert "方案三" in p.Slides.Item(15).Shapes.Item("页面标题").TextFrame.TextRange.Text
    p.Slides.Item(15).Delete()
    p.Slides.Item(14).Delete()
    p.Slides.Item(14).Shapes.Item("页码").TextFrame.TextRange.Text = "14"
    s = p.Slides.Item(15)
    s.Shapes.Item("页码").TextFrame.TextRange.Text = "15"
    table = s.Shapes.Item("可编辑比较表").Table
    matches = [r for r in range(2, table.Rows.Count + 1)
               if "方案三" in table.Cell(r, 2).Shape.TextFrame.TextRange.Text]
    assert len(matches) == 1
    table.Rows.Item(matches[0]).Delete()
    for i in range(1, table.Rows.Count + 1):
        table.Rows.Item(i).Height = 54 if i == 1 else 56
    s.Shapes.Item("TextBox 4").TextFrame.TextRange.Text = "方案一、二均为研究设想，尚未验证创新性或性能提升。"
    extract.drawing.notes(s, "本合集迁出方案三后保留15页：2–3页PaperBanana，4–9页RoCo，10–11页方案一，12–13页方案二，14页参考文献，15页来源附录。方案三迁至同级agentic_heuristic_design项目。原有论文、来源与边界保持各自归属。")


if __name__ == "__main__":
    if not FINAL.exists():
        ps = extract.presentations()
        p = ps.Open(str(extract.SNAPSHOT), False, True, False)
        try:
            edit(p)
            p.SaveAs(str(FINAL), 24)
            p.Slides.Item(15).Export(str(WORK / "源合集迁出预览.png"), "PNG", 1600, 900)
        finally:
            p.Close()
    a, b = Presentation(extract.SNAPSHOT), Presentation(FINAL)
    assert len(b.slides) == 15
    assert [b.slides[i].slide_id for i in range(13)] == [a.slides[i].slide_id for i in range(13)]
    p0, p1 = extract.drawing.package(extract.SNAPSHOT), extract.drawing.package(FINAL)
    x, y = extract.drawing.slide_parts(p0), extract.drawing.slide_parts(p1)
    assert all(p0[x[a.slides[i].slide_id]] == p1[y[b.slides[i].slide_id]] for i in range(13))
    print("source_cleanup_preview_ready_no_live_change")
