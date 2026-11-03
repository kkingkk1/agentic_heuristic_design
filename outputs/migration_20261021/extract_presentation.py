"""从活动合集取只读快照，在新项目生成方案三独立演示；不修改原合集。"""
from pathlib import Path
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import pythoncom
import win32com.client as win32
from pptx import Presentation

ROOT = Path(r"E:\OneDrive\project\static_determine\agentic_heuristic_design")
WORK = ROOT / "outputs/migration_20261021"
WRITING = ROOT / "vault/Writing"
SNAPSHOT = WRITING / "面向卫星任务调度的多智能体协作研究.pptx"
FINAL = WRITING / "方案三_通用启发式自动设计.pptx"
spec = importlib.util.spec_from_file_location("drawing", ROOT / "outputs/satellite_multiagent_briefing/scheme_three_20261021/add_scheme_three.py")
drawing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drawing)


def presentations():
    app = win32.GetActiveObject("PowerPoint.Application")
    return win32.Dispatch(app._oleobj_.Invoke(app._oleobj_.GetIDsOfNames("Presentations"), 0,
                                             pythoncom.DISPATCH_PROPERTYGET, True))


def create():
    assert not SNAPSHOT.exists() and not FINAL.exists()
    WRITING.mkdir(parents=True, exist_ok=True)
    ps = presentations()
    found = [p for p in ps if p.Name == SNAPSHOT.name]
    assert len(found) == 1
    live = found[0]
    assert live.Slides.Count == 17
    before = bool(live.Saved)
    keep = [live.Slides.Item(i).SlideID for i in (14, 15, 16)]
    assert "方案三" in live.Slides.Item(14).Shapes.Item("页面标题").TextFrame.TextRange.Text
    assert "方案三" in live.Slides.Item(15).Shapes.Item("页面标题").TextFrame.TextRange.Text
    live.SaveCopyAs(str(SNAPSHOT), 24)
    assert bool(live.Saved) == before
    p = ps.Open(str(SNAPSHOT), False, True, False)
    try:
        for n in range(p.Slides.Count, 0, -1):
            if p.Slides.Item(n).SlideID not in keep:
                p.Slides.Item(n).Delete()
        assert p.Slides.Count == 3
        for i in range(1, 4):
            s = p.Slides.Item(i)
            s.Shapes.Item("页码").TextFrame.TextRange.Text = str(i)
            body = [x for x in s.NotesPage.Shapes if x.Type == 14 and x.PlaceholderFormat.Type == 2][0]
            value = body.TextFrame.TextRange.Text.replace("[2]", "[1]")
            value = value.replace("第14页", "第1页").replace("第15页", "第2页")
            body.TextFrame.TextRange.Text = value
        s = p.Slides.Item(3)
        citation = [x for x in s.Shapes if x.HasTextFrame and "RoCo: Role-Based" in x.TextFrame.TextRange.Text]
        assert len(citation) == 1
        cite_name = citation[0].Name
        for n in range(s.Shapes.Count, 0, -1):
            if s.Shapes.Item(n).Name not in {"页面标题", "页码", cite_name}:
                s.Shapes.Item(n).Delete()
        s.Shapes.Item("页面标题").TextFrame.TextRange.Text = "参考文献与方案边界"
        c = s.Shapes.Item(cite_name)
        c.Left, c.Top, c.Width, c.Height = 55, 150, 850, 110
        c.TextFrame.TextRange.Text = "[1] Xu, J.; Wei, F.; Chen, W. RoCo: Role-Based LLMs Collaboration for\rAutomatic Heuristic Design. arXiv:2512.03762v2.\rhttps://arxiv.org/abs/2512.03762v2"
        c.TextFrame.TextRange.Font.Size = 21
        drawing.text(s, "来源与边界", "机制来源：§3.4–3.5、Algorithm 1、Prompts 8–10。\r本方案扩展：完整启发式冷启动；质量／效率双方向自主协作。\r尚未实现，尚无性能提升结论。", 55, 295, 850, 120, 22,
                     drawing.INK, False, 1)
        drawing.notes(s, "本独立演示由原合集第14–15页迁入，正文内容与备注保留。RoCo原论文是机制来源，完整启发式冷启动、质量／效率角色替换与本项目筛选规则属于拟议扩展。原合集快照保存在同目录供溯源，不作为当前方案。")
        p.SaveAs(str(FINAL), 24)
        overflow = []
        for i in range(1, 4):
            s = p.Slides.Item(i)
            for sh in s.Shapes:
                if sh.HasTextFrame and sh.TextFrame.HasText:
                    f = sh.TextFrame2
                    if f.TextRange.BoundHeight > sh.Height - f.MarginTop - f.MarginBottom + 1.5:
                        overflow.append((i, sh.Name))
            s.Export(str(WORK / f"独立演示第{i}页.png"), "PNG", 1600, 900)
        assert not overflow, overflow
    finally:
        p.Close()
    check()
    print(json.dumps({"slides": 3, "source_slides": 17, "source_saved_flag_preserved": True,
                      "source_modified_by_this_script": False}, ensure_ascii=False))


def check():
    old, new = Presentation(SNAPSHOT), Presentation(FINAL)
    assert len(old.slides) == 17 and len(new.slides) == 3
    for oi, ni in ((13, 0), (14, 1)):
        prior = {sh.name: sh for sh in old.slides[oi].shapes}
        current = {sh.name: sh for sh in new.slides[ni].shapes}
        assert prior.keys() == current.keys()
        for name in prior:
            a, b = prior[name], current[name]
            assert (a.left, a.top, a.width, a.height) == (b.left, b.top, b.width, b.height)
            if name != "页码" and a.has_text_frame:
                assert a.text == b.text
        note = new.slides[ni].notes_slide.notes_text_frame.text
        assert "三" in note and "精英变异" in note
    assert len([x for x in new.slides[0].shapes if x.has_table][0].table.rows) == 5
    print("standalone_content_and_geometry_verified")


if __name__ == "__main__":
    {"create": create, "check": check}[sys.argv[1]]()
