"""仅精简方案三第15页，保留完整机制说明与其他页面。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
spec = importlib.util.spec_from_file_location("scheme_three", Path(__file__).resolve().parent.parent / "scheme_three_20261021" / "add_scheme_three.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from pptx import Presentation
import pythoncom
import win32com.client as win32

WORK = Path(__file__).resolve().parent
BACKUP = WORK / "修改前备份.pptx"
FINAL = WORK / "第15页精简核验稿.pptx"
PNG = WORK / "第15页.png"
SLIDE_ID = 293


def current():
    app = win32.GetActiveObject("PowerPoint.Application")
    # 活动实例的动态属性包装可能失效，按实际查询到的属性ID取集合。
    dispid = app._oleobj_.GetIDsOfNames("Presentations")
    presentations = win32.Dispatch(app._oleobj_.Invoke(dispid, 0, pythoncom.DISPATCH_PROPERTYGET, True))
    found = [p for p in presentations if p.Name == base.NAME]
    assert len(found) == 1
    return presentations, found[0]


def edit(p):
    assert p.Slides.Count == 17
    s = p.Slides.FindBySlideID(SLIDE_ID)
    assert s.SlideIndex == 15
    body = [x for x in s.NotesPage.Shapes if x.Type == 14 and x.PlaceholderFormat.Type == 2][0]
    old_notes = body.TextFrame.TextRange.Text
    for i in range(s.Shapes.Count, 0, -1):
        if s.Shapes.Item(i).Name not in {"页面标题", "页码"}:
            s.Shapes.Item(i).Delete()
    t, b, a = base.text, base.box, base.arrow
    t(s, "冷启动概览", "问题 → 从零生成与验证 → 初始种群", 55, 108, 850, 40, 23, align=1)
    b(s, "当前种群", "当前种群\r选择精英对", 55, 220, 180, 70, base.GRAY, base.CREAM)
    b(s, "自主双分支", "解质量智能体\r效率智能体", 300, 220, 215, 70, size=24)
    b(s, "Integrator", "Integrator", 590, 220, 155, 70, size=24)
    b(s, "程序评价", "执行评估", 795, 220, 110, 70, base.GRAY, base.CREAM, 20)
    b(s, "Critic", "Critic", 590, 350, 155, 60, size=24)
    b(s, "反思记忆与变异", "反思与记忆\r精英变异", 300, 350, 215, 60, base.TEAL, base.CREAM, 22)
    a(s, "精英对交给双分支", [(235, 255), (300, 255)])
    a(s, "双分支交给整合", [(515, 255), (590, 255)])
    a(s, "融合候选执行", [(745, 255), (795, 255)])
    a(s, "原生候选执行", [(515, 236), (550, 236), (550, 185), (850, 185), (850, 220)])
    t(s, "候选评价说明", "候选均需评价", 610, 154, 200, 27, 18)
    a(s, "分支成绩给整合", [(795, 277), (745, 277)], base.TEAL, True)
    t(s, "成绩标签", "成绩", 745, 293, 65, 26, 18, base.TEAL)
    a(s, "结果给Critic", [(850, 290), (850, 380), (745, 380)])
    a(s, "反思记忆更新", [(590, 380), (515, 380)], base.TEAL)
    a(s, "反馈双分支", [(407.5, 350), (407.5, 290)], base.TEAL, True)
    t(s, "反馈标签", "反馈", 420, 305, 80, 28, 20, base.TEAL)
    a(s, "评价筛选下一代", [(300, 380), (145, 380), (145, 290)], base.TEAL)
    t(s, "代际筛选标签", "评价筛选", 151, 341, 140, 28, 20, base.TEAL)
    # 主图只展示阶段关系；逐轮／逐代时序保留于备注。
    old_notes = old_notes.replace(
        "图中蓝色为候选与实测证据的传递，绿色虚线为反馈与反思。程序节点用米白底和灰边。图下方为每代 T 轮协作结束后执行的反思、变异和筛选，不是每轮都生成长期记忆。",
        "精简图说明：蓝色为候选与实测证据流；绿色为反馈、反思记忆与代际演化。上方一行概括冷启动，两个分支从问题直接生成并验证完整程序。六个框保留四个智能体、种群和程序评价，并把反思、记忆、精英变异合并显示；合并框不是第五个智能体。反馈每轮可用；本代长期反思、角色历史更新和三角色精英变异在代末执行。左下方‘评价筛选’表示所有变异候选先经同一执行工具评价，再与父代和其他有效候选竞争下一代。图上省略职责说明，见第14页；具体时序见下文。")
    old_notes = old_notes.replace("“分支成绩”回传线", "“成绩”回传线")
    base.notes(s, old_notes)


def export(p):
    s = p.Slides.FindBySlideID(SLIDE_ID)
    bad = []
    for sh in s.Shapes:
        if sh.HasTextFrame and sh.TextFrame.HasText:
            f = sh.TextFrame2
            if f.TextRange.BoundHeight > sh.Height - f.MarginTop - f.MarginBottom + 1.5:
                bad.append(sh.Name)
    assert not bad, bad
    s.Export(str(PNG), "PNG", 1600, 900)


def check(enforce_other_slides=True):
    a, b = base.package(BACKUP), base.package(FINAL)
    x, y = base.slide_parts(a), base.slide_parts(b)
    assert list(x) == list(y) and len(y) == 17
    other_changes = [i + 1 for i, sid in enumerate(x) if sid != SLIDE_ID and a[x[sid]] != b[y[sid]]]
    if enforce_other_slides:
        assert not other_changes, other_changes
    for k in a:
        if k.startswith(("ppt/media/", "ppt/theme/")):
            assert a[k] == b[k], k
        if k.startswith(("ppt/slideMasters/", "ppt/slideLayouts/")):
            norm = lambda v: re.sub(rb"10/2[01]/2026", b"DATE_FIELD", v)
            assert norm(a[k]) == norm(b[k]), k
    p0, p1 = Presentation(BACKUP).slides[14], Presentation(FINAL).slides[14]
    text0 = "\n".join(s.text for s in p0.shapes if s.has_text_frame)
    text1 = "\n".join(s.text for s in p1.shapes if s.has_text_frame)
    assert all(t in text1 for t in ("解质量智能体", "效率智能体", "Critic", "Integrator", "精英变异", "反思与记忆", "初始种群"))
    assert not re.search("TODO|XXXX|lorem|ipsum", text1, re.I)
    assert len(text1) < len(text0) * .65
    note = p1.notes_slide.notes_text_frame.text
    assert all(t in note for t in ("三角色精英变异", "本代长期反思", "质量、效率、整合三个方向", "首代历史记忆为空", "完整启发式"))
    old_note = p0.notes_slide.notes_text_frame.text
    assert old_note[old_note.index("冷启动："):].replace("“分支成绩”", "“成绩”") == note[note.index("冷启动："):]
    return {"slides": 17, "changed_page": 15, "characters_before": len(text0), "characters_after": len(text1),
            "shapes_before": len(p0.shapes), "shapes_after": len(p1.shapes), "other_slide_xml_unchanged": not other_changes,
            "other_changed_pages_during_edit": other_changes,
            "media_preserved": True, "master_layout_only_possible_change": "PowerPoint自动日期字段",
            "changed_parts": [k for k in a.keys() & b.keys() if a[k] != b[k]]}


def stage():
    presentations, _ = current()
    assert not FINAL.exists()
    p = presentations.Open(str(BACKUP), False, False, False)
    try:
        edit(p)
        p.SaveAs(str(FINAL), 24)
        export(p)
    finally:
        p.Close()
    print(json.dumps(check(), ensure_ascii=False))


def apply():
    check()
    _, p = current()
    assert p.Saved and p.Slides.Count == 17
    s = p.Slides.FindBySlideID(SLIDE_ID)
    old = Presentation(BACKUP).slides[14]
    assert s.Shapes.Count == len(old.shapes)
    for sh, prior in zip(s.Shapes, old.shapes):
        assert sh.Name == prior.name
        if prior.has_text_frame:
            assert sh.TextFrame.TextRange.Text.replace("\r", "\n") == prior.text
    digest = hashlib.sha256(PNG.read_bytes()).hexdigest()
    edit(p)
    p.Save()
    assert p.Saved
    p.SaveCopyAs(str(FINAL), 24)
    export(p)
    assert hashlib.sha256(PNG.read_bytes()).hexdigest() == digest
    # 共享活动演示的其他编辑必须保留，并明确报告，不能回退到早期副本。
    print(json.dumps({**check(False), "live_saved": True, "final_render_matches_reviewed": True}, ensure_ascii=False))


def refine():
    presentations, _ = current()
    p = presentations.Open(str(FINAL), False, False, False)
    try:
        p.Slides.FindBySlideID(SLIDE_ID).Shapes.Item("程序评价_正文").TextFrame.TextRange.Text = "执行评估"
        p.Save()
        export(p)
    finally:
        p.Close()
    print(json.dumps(check(), ensure_ascii=False))


if __name__ == "__main__":
    {"stage": stage, "refine": refine, "check": lambda: print(json.dumps(check(), ensure_ascii=False)),
     "audit": lambda: print(json.dumps(check(False), ensure_ascii=False)), "apply": apply}[sys.argv[1]]()
