"""用 PowerPoint 原生对象追加方案三，先核验副本，再应用活动演示。"""
from pathlib import Path
import hashlib
import json
import posixpath
import re
import sys
import zipfile

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
from lxml import etree
from pptx import Presentation
import win32com.client as win32

WORK = Path(__file__).resolve().parent
BACKUP = WORK / "应用前备份.pptx"
TEST = WORK / "方案三最终核验稿.pptx"
LIVE = WORK.parents[2] / "vault" / "Writing" / "面向卫星任务调度的多智能体协作研究.pptx"
NAME = LIVE.name
BLUE, TEAL, INK, GRAY = "06527C", "00A990", "243744", "667780"
PALE, CREAM = "E5F3F7", "FBF9ED"
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}

ROWS = [
    ("角色", "角色分工", "输入 → 输出"),
    ("解质量智能体", "自主决定改哪里、怎么改\r提升解质量并总结经验",
     "问题／父代、实测结果与角色记忆\r→ 完整代码、改动说明与长期反思"),
    ("效率智能体", "自主改进算法与实现\r降低耗时并总结经验",
     "问题／父代、运行剖析与角色记忆\r→ 完整代码、效率改进与长期反思"),
    ("Critic", "基于实测结果比较候选\r输出建议与短期反思",
     "候选代码、实测对照与协作记录\r→ 修改建议、短期反思"),
    ("Integrator", "融合有效修改、处理冲突\r合并经验并引导精英变异",
     "双分支代码、实测结果与角色记忆\r→ 融合代码、整合经验与变异代码"),
]

COMMON_NOTES = """方案三：本次讨论提出的通用启发式自动设计框架，尚未实现或验证效果。
本方案借鉴 RoCo 的种群协作、短期反思、长期反思、历史记忆与精英变异。将 Explorer／Exploiter 替换为解质量／效率两个自主设计方向，以及从零生成完整启发式，属于本方案的扩展，不能称为 RoCo 原论文的现成能力。
来源：[2] Xu, Wei, Chen. RoCo: Role-Based LLMs Collaboration for Automatic Heuristic Design, arXiv:2512.03762v2。§3.1–3.5，PDF pp.3–4；Algorithm 1，p.10；Prompts 8–10，p.16。
只有四个 LLM 角色：解质量智能体、效率智能体、Critic、Integrator。不设 Planner、Retriever，也不设独立的候选档案。
程序控制器仅负责种群规模、父代选择、调用顺序、预算、运行与生存筛选；不决定启发式的结构、待改模块或修改方法。
解质量与效率智能体自行理解问题、提出完整启发式、决定改哪里和怎么改；Critic 的建议可供采用、改写或拒绝。四个角色可使用同一基础模型，不表示训练四个独立模型。
"""

ROLE_NOTES = COMMON_NOTES + """
输入合同：问题目标与优化方向、约束、输入语义、合法输出、开发实例、运行环境，以及设计预算和单次求解预算。用户不提供求解框架或初始启发式代码。必须具备独立、可信且固定的评价环境；若尚缺，先依据问题合同补齐并核验评价逻辑，再固定它。评价器只判定程序输出，不为求解器预设搜索模块。

解质量智能体——冷启动输入：问题合同、开发实例特征、程序接口与预算；输出：完整启发式的设计描述和可执行代码。协作输入：相同精英父代、当前与另一分支候选、实测成绩、Critic 建议和短期反思；输出：新的解质量候选和修改说明。代末输入：本分支逐轮反思与修改前后实测变化；输出：本代解质量长期反思。变异输入：所选精英原代码、本代解质量长期反思、先前各代的解质量历史记忆；输出：新的完整变异程序。

效率智能体——冷启动同样直接生成完整程序，初始阶段不要求已有剖析。协作输入：父代、候选代码、实例规模、运行耗时、剖析证据、解质量变化和 Critic 反馈；输出：效率改进候选与修改依据。智能体可改算法、数据结构、计算复用或实现方式，不预先限定改动模块。代末总结自己的长期反思，按效率角色记忆指导精英变异。效率指生成求解器的执行效率；声称降低渐近复杂度时需额外给出分析，运行更快本身不能证明复杂度阶数降低。

Critic——输入：候选描述、完整代码或差异、相同实例与预算下的可行性、目标值、耗时、错误信息和本轮记录。输出一：可操作的修改建议；输出二：短期反思，说明哪些修改与改善或退化相伴及其证据边界。客观分数由程序计算。Critic 不替代两个设计智能体指定下一步必须修改的模块。实际执行失败的候选进入修复和反馈，未通过者不进入有效种群。

Integrator——融合输入：两个分支的父代关系、候选描述、完整代码、修改点、实测结果及 Critic 反馈；输出：解决接口与设计冲突后的完整融合代码，并解释保留或放弃哪些修改。代末输入：Q／E 的本代长期反思；输出：整合长期反思。精英变异输入：一份精英原代码、本代整合长期反思、历代整合记忆；输出：整合方向的变异程序。双代码融合与单精英代码的记忆引导变异是两种不同操作。

表格对各阶段的输入输出做合并速查。“问题／父代”表示冷启动只有问题，演化阶段才有父代；初始角色记忆为空。Q、E、I 三个方向均可做精英变异，并非只有 Integrator 变异。
"""

FLOW_NOTES = COMMON_NOTES + """
图中蓝色为候选与实测证据的传递，绿色虚线为反馈与反思。程序节点用米白底和灰边。图下方为每代 T 轮协作结束后执行的反思、变异和筛选，不是每轮都生成长期记忆。

冷启动：Q／E 各自从问题合同直接生成多份完整启发式。程序编译或加载、执行、验证可行性并计算解质量与耗时。允许失败后修复，只有验证通过的程序构成初始种群。每个个体包含简短设计描述、完整代码和实测评价。首代历史记忆为空。

每代开始：程序从当前种群选择精英父代对，两个设计智能体读取相同父代与证据，自主继承、组合或重构。沿用 RoCo 的多轮协作，但双分支改为质量与效率方向。
一轮协作：对当前候选先执行、评分并比较；Critic 输出建议和短期反思。Q／E 自行决定下一版修改，生成新代码；两条原生分支分别执行与评价，再供 Integrator 参考实测结果融合；融合代码也独立执行与评价。图上两条候选路径均指向程序执行模块：一条保留原生分支，另一条经过 Integrator。“分支成绩”回传线表示 Integrator 读取两个分支已经取得的实测报告，再判断哪些修改有效。图上总执行模块表示每一次代码变更后复用的同一固定工具，而非仅在整合后运行一次。下一轮以更新后的候选和客观证据继续协作，直至 T 轮或本代预算耗尽。多轮后可再融合两分支历轮较优版本，融合产物仍需运行。

本代长期反思：Q／E 分别把本分支多轮短期反思与修改前后质量、耗时及差值总结为角色经验；Integrator 合并两份经验。长期反思不是完整源代码或原始日志的复制。
历史记忆：按质量、效率、整合三个方向积累先前各代的长期反思。当前代的长期反思与先前历史作为不同输入提供给变异，避免在同次调用里重复塞入当前经验。完成本代使用后再把当前长期反思追加到对应角色历史，供后代使用。未预设外部检索或 Retriever；历史记忆的容量和压缩政策留待实现时按上下文预算确定。

精英变异：沿用本代开始时选出的精英原程序作为基底；每个基底分别由 Q／E／I 结合本代角色长期反思和先前角色历史改写，形成三个方向的完整新程序。原代码是基底，反思是修改依据，不把两者混为“历史档案”。所有变异程序均独立执行与评价。

生存筛选：父代种群、各轮有效 Q／E 候选、各轮与最终融合候选、有效精英变异候选统一竞争下一代。保留 N 个可执行程序并维持候选多样性。本方案明确保留父代并纳入各轮有效产物；这属于拟议筛选规则，不把它说成 RoCo 原文算法唯一无歧义的规则。具体质量与耗时折中政策需在实验前固定，不能由智能体修改客观目标、可行域或评价规则。

只维护三类必要状态：当前种群；本代候选、执行报告及短期反思缓冲；本代长期反思和先前各代角色历史记忆。不另建保存全部历史代码、全部失败与谱系的候选档案核心模块。

评价与输出：解质量在相同单次求解预算下比较；效率结合可比工作量的耗时、规模扩展表现或达到相同质量所需时间评价，并报告质量损失。预算包含预处理，问题目标与约束固定。使用开发／验证实例设计和选型；最终冻结程序及参数，再在未见测试实例上评估。输出为完整、可复用的启发式程序和验证证据；部署到新实例时直接运行冻结代码，默认不再调用 LLM。框架不限于卫星调度，不承诺对所有优化问题都能直接取得有效启发式。
"""


def rgb(value):
    return int(value[:2], 16) + 256 * int(value[2:4], 16) + 65536 * int(value[4:], 16)


def connect():
    app = win32.GetActiveObject("PowerPoint.Application")
    found = [app.Presentations.Item(i) for i in range(1, app.Presentations.Count + 1)
             if app.Presentations.Item(i).Name == NAME]
    assert len(found) == 1
    return app, found[0]


def notes(slide, value):
    body = [s for s in slide.NotesPage.Shapes if s.Type == 14 and s.PlaceholderFormat.Type == 2]
    assert len(body) == 1
    body[0].TextFrame.TextRange.Text = value


def text(slide, name, value, x, y, width, height, size=20, color=BLUE, bold=False, align=2):
    sh = slide.Shapes.AddTextbox(1, x, y, width, height)
    sh.Name = name
    f = sh.TextFrame
    f.AutoSize = 0
    f.WordWrap = -1
    f.MarginLeft = f.MarginRight = 2
    f.MarginTop = f.MarginBottom = 0
    f.VerticalAnchor = 3
    tr = f.TextRange
    tr.Text = value
    tr.Font.Name = tr.Font.NameFarEast = "Microsoft YaHei"
    tr.Font.Size = size
    tr.Font.Bold = -1 if bold else 0
    tr.Font.Color.RGB = rgb(color)
    pf = tr.ParagraphFormat
    pf.Alignment = align
    pf.SpaceBefore = pf.SpaceAfter = 0
    pf.LineRuleWithin = -1
    pf.SpaceWithin = 1
    sh.Shadow.Visible = 0
    return sh


def box(slide, name, value, x, y, width, height, color=BLUE, fill=PALE, size=20, bold=False):
    sh = slide.Shapes.AddShape(1, x, y, width, height)
    sh.Name = name
    sh.Fill.Solid()
    sh.Fill.ForeColor.RGB = rgb(fill)
    sh.Line.ForeColor.RGB = rgb(color)
    sh.Line.Weight = 1.2
    sh.Shadow.Visible = 0
    tx = text(slide, name + "_正文", value, x + 5, y + 3, width - 10, height - 6,
              size, color, bold)
    return sh, tx


def arrow(slide, name, points, color=BLUE, dash=False):
    for i, (a, b) in enumerate(zip(points, points[1:])):
        sh = slide.Shapes.AddConnector(1, a[0], a[1], b[0], b[1])
        sh.Name = name + f"_{i}"
        sh.Line.ForeColor.RGB = rgb(color)
        sh.Line.Weight = 1.4
        sh.Line.DashStyle = 4 if dash else 1
        sh.Line.EndArrowheadStyle = 3 if i == len(points) - 2 else 1
        sh.Shadow.Visible = 0


def edit(p):
    assert p.Slides.Count == 15
    role = p.Slides.Item(12).Duplicate().Item(1)
    role.MoveTo(14)
    role.Shapes.Item("页面标题").TextFrame.TextRange.Text = "方案三：通用启发式自动设计（角色分工）"
    role.Shapes.Item("页码").TextFrame.TextRange.Text = "14"
    role.Shapes.Item("框架定位").TextFrame.TextRange.Text = "只有问题，无方法框架和初始代码；由四个智能体共同设计完整启发式"
    role.Shapes.Item("本页要点").TextFrame.TextRange.Text = "首轮从问题生成代码；后续自主改进，程序模块负责运行与客观评分"
    shape = role.Shapes.Item("五智能体分工与输入输出")
    shape.Name = "方案三_四智能体分工与输入输出"
    table = shape.Table
    assert table.Rows.Count == 6
    table.Rows.Item(6).Delete()
    for r, row in enumerate(ROWS, 1):
        for c, value in enumerate(row, 1):
            f = table.Cell(r, c).Shape.TextFrame
            f.MarginTop = f.MarginBottom = 2.5
            f.MarginLeft = f.MarginRight = 6
            f.TextRange.Text = value
            tr = f.TextRange
            tr.Font.Name = tr.Font.NameFarEast = "Microsoft YaHei"
            tr.Font.Size = 22 if r == 1 else 20
            tr.Font.Bold = -1 if r == 1 else 0
            tr.Font.Color.RGB = rgb("FFFFFF" if r == 1 or c == 1 else INK)
            tr.ParagraphFormat.SpaceBefore = tr.ParagraphFormat.SpaceAfter = 0
            tr.ParagraphFormat.LineRuleWithin = -1
            tr.ParagraphFormat.SpaceWithin = 1
        table.Rows.Item(r).Height = 44 if r == 1 else 68
    notes(role, ROLE_NOTES)

    flow = p.Slides.Item(13).Duplicate().Item(1)
    flow.MoveTo(15)
    for i in range(flow.Shapes.Count, 0, -1):
        if flow.Shapes.Item(i).Name not in {"页面标题", "页码"}:
            flow.Shapes.Item(i).Delete()
    flow.Shapes.Item("页面标题").TextFrame.TextRange.Text = "方案三：通用启发式自动设计（种群演化）"
    flow.Shapes.Item("页码").TextFrame.TextRange.Text = "15"
    # 同色表示同类职责；程序模块与 LLM 角色采用不同底色。
    box(flow, "问题输入", "问题定义＋开发实例\r＋运行要求", 55, 103, 280, 54)
    box(flow, "从零生成", "双智能体从零设计\r生成完整启发式代码", 390, 103, 240, 54)
    box(flow, "初始验证", "程序验证与评价\r建立初始种群", 690, 103, 215, 54, GRAY, CREAM)
    arrow(flow, "问题到从零生成", [(335, 130), (390, 130)])
    arrow(flow, "首轮代码到验证", [(630, 130), (690, 130)])
    # 外层回路表示下一代成为新的当前种群。
    arrow(flow, "初始种群进入演化", [(797, 157), (797, 174), (145, 174), (145, 225)])
    arrow(flow, "下一代回到种群", [(905, 448), (925, 448), (925, 174), (145, 174), (145, 225)])
    box(flow, "当前种群", "当前种群\r程序选择精英对", 55, 225, 180, 90, GRAY, CREAM)
    box(flow, "自主双分支", "解质量智能体\r效率智能体\r自主决定改哪里、怎么改", 280, 225, 245, 90)
    box(flow, "Integrator", "Integrator\r融合有效修改\r处理设计冲突", 560, 225, 170, 90)
    box(flow, "执行评估", "程序执行\r与评估\r质量／耗时", 780, 225, 125, 90, GRAY, CREAM)
    arrow(flow, "精英对交给双分支", [(235, 270), (280, 270)])
    arrow(flow, "双分支交给整合", [(525, 270), (560, 270)])
    arrow(flow, "融合代码执行", [(730, 270), (780, 270)])
    text(flow, "融合路径标签", "融合", 731, 240, 48, 26, 18)
    arrow(flow, "原生候选独立执行", [(525, 242), (542, 242), (542, 203), (842, 203), (842, 225)])
    text(flow, "原生路径标签", "双分支也分别运行", 590, 176, 180, 26, 18)
    box(flow, "Critic", "Critic\r建议＋短期反思", 560, 345, 170, 56)
    arrow(flow, "实测结果给Critic", [(842, 315), (842, 373), (730, 373)])
    text(flow, "实测结果标签", "实测结果", 790, 337, 108, 27, 20)
    arrow(flow, "分支实测返回整合", [(780, 298), (730, 298)], TEAL, True)
    text(flow, "分支成绩标签", "分支成绩", 671, 317, 105, 23, 18, TEAL)
    arrow(flow, "Critic反馈双分支", [(560, 373), (407, 373), (407, 315)], TEAL, True)
    text(flow, "多轮反馈标签", "下一轮反馈", 416, 338, 140, 28, 20, TEAL)
    # 每代 T 轮结束：反思、记忆引导变异，再统一筛选。
    arrow(flow, "本代结束总结", [(645, 401), (645, 413), (170, 413), (170, 419)], TEAL, True)
    text(flow, "代末触发标签", "T 轮后", 745, 388, 100, 25, 18, TEAL)
    box(flow, "本代反思", "双分支总结并整合\r形成本代长期反思", 55, 419, 230, 54, TEAL, CREAM)
    box(flow, "记忆引导精英变异", "本代反思＋历代记忆\r精英代码 → 三向变异", 340, 419, 250, 54, TEAL, CREAM)
    box(flow, "统一筛选", "程序评价并筛选下一代\r旧种群＋本代有效候选", 650, 419, 255, 54, GRAY, CREAM)
    arrow(flow, "反思用于变异", [(285, 449), (340, 449)], TEAL)
    arrow(flow, "变异用于筛选", [(590, 449), (650, 449)], TEAL)
    text(flow, "本页要点", "输出：经验证的完整启发式程序；新实例直接运行冻结代码",
         52.6, 480, 853.9, 32, 21, TEAL, True, 1)
    notes(flow, FLOW_NOTES)

    p.Slides.Item(16).Shapes.Item("页码").TextFrame.TextRange.Text = "16"
    appendix = p.Slides.Item(17)
    appendix.Shapes.Item("页码").TextFrame.TextRange.Text = "17"
    t = appendix.Shapes.Item("可编辑比较表").Table
    assert t.Rows.Count == 6
    t.Rows.Add()
    for c, value in enumerate(("14–15", "方案三：本次讨论的拟议系统；借鉴 [2] §3.4–3.5\r完整启发式冷启动与质量／效率分工为扩展，尚无实验结果"), 1):
        f = t.Cell(7, c).Shape.TextFrame
        f.TextRange.Text = value
        f.TextRange.Font.Name = f.TextRange.Font.NameFarEast = "Microsoft YaHei"
        f.TextRange.Font.Size = 20
    for r in range(1, 8):
        for c in range(1, 3):
            f = t.Cell(r, c).Shape.TextFrame
            f.MarginTop = f.MarginBottom = 1
            f.TextRange.ParagraphFormat.SpaceBefore = f.TextRange.ParagraphFormat.SpaceAfter = 0
            f.TextRange.ParagraphFormat.LineRuleWithin = -1
            f.TextRange.ParagraphFormat.SpaceWithin = 1
        t.Rows.Item(r).Height = 42 if r == 1 else 50
    appendix.Shapes.Item("TextBox 4").TextFrame.TextRange.Text = "方案一、二、三均为研究设想，尚未验证创新性或性能提升。"
    source_notes = "逐页来源按当前17页演示更新。2–3页介绍PaperBanana；4–9页介绍RoCo；10–11页为方案一，12–13页为方案二，14–15页为方案三。来源[3]仅用于视觉参考。方案三借鉴[2]§3.4–3.5、Algorithm 1与Prompts 8–10；从零设计完整启发式及质量／效率分工为本次讨论的扩展。科学事实、匿名代码行为和本项目设想分别标明，不将新框架称为原论文方法或已验证结果。"
    notes(appendix, source_notes)
    assert p.Slides.Count == 17


def bounds(p):
    bad = []
    for n in (14, 15, 16, 17):
        for sh in p.Slides.Item(n).Shapes:
            cells = [sh]
            if sh.HasTable:
                t = sh.Table
                cells = [t.Cell(r, c).Shape for r in range(1, t.Rows.Count + 1)
                         for c in range(1, t.Columns.Count + 1)]
            for cell in cells:
                if cell.HasTextFrame and cell.TextFrame.HasText:
                    f = cell.TextFrame2
                    inner = cell.Height - f.MarginTop - f.MarginBottom
                    if f.TextRange.BoundHeight > inner + 1.5:
                        bad.append({"slide": n, "shape": sh.Name, "text": cell.TextFrame.TextRange.Text,
                                    "height": round(f.TextRange.BoundHeight, 2), "available": round(inner, 2)})
    return bad


def export(p):
    for n in (14, 15, 16, 17):
        p.Slides.Item(n).Export(str(WORK / f"第{n}页.png"), "PNG", 1600, 900)


def package(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        return {k: z.read(k) for k in z.namelist()}


def slide_parts(pkg):
    rel = etree.fromstring(pkg["ppt/_rels/presentation.xml.rels"])
    mapping = {r.get("Id"): posixpath.normpath(posixpath.join("ppt", r.get("Target"))) for r in rel}
    ids = etree.fromstring(pkg["ppt/presentation.xml"]).find("p:sldIdLst", NS)
    return {int(x.get("id")): mapping[x.get("{" + NS["r"] + "}id")] for x in ids}


def check():
    a, b = package(BACKUP), package(TEST)
    x, y = slide_parts(a), slide_parts(b)
    assert len(x) == 15 and len(y) == 17
    assert list(x)[:13] == list(y)[:13]
    for sid in list(x)[:13]:
        assert a[x[sid]] == b[y[sid]], ("原有正文变化", sid)
    for key in a:
        if key.startswith(("ppt/media/", "ppt/theme/")):
            assert a[key] == b[key], key
        if key.startswith(("ppt/slideMasters/", "ppt/slideLayouts/")):
            norm = lambda v: re.sub(rb"10/2[01]/2026", b"DATE", v)
            assert norm(a[key]) == norm(b[key]), key
    texts = ["\n".join(etree.fromstring(b[part]).xpath("//a:t/text()", namespaces=NS))
             for part in list(y.values())[13:15]]
    assert all(role in texts[0] for role in ("解质量智能体", "效率智能体", "Critic", "Integrator"))
    assert all(t in texts[1] for t in ("初始种群", "自主决定改哪里", "长期反思", "历代记忆", "三向变异", "下一代"))
    assert not re.search("TODO|XXXX|lorem|ipsum|Retriever|Planner|候选档案", "\n".join(texts), re.I)
    p = Presentation(TEST)
    notes_text = "\n".join(p.slides[n].notes_slide.notes_text_frame.text for n in (13, 14))
    assert all(t in notes_text for t in ("首代历史记忆为空", "Q／E／I", "当前种群", "本代开始时选出的", "完整启发式", "三类必要状态"))
    broken = []
    for name, value in b.items():
        if name.endswith(".rels"):
            folder = posixpath.dirname(posixpath.dirname(name))
            for rel in etree.fromstring(value):
                if rel.get("TargetMode") == "External":
                    continue
                target = rel.get("Target")
                resolved = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(folder, target))
                if resolved not in b:
                    broken.append((name, target))
    assert not broken, broken
    return {"slides": 17, "scheme_three": [14, 15], "original_pages_1_to_13_preserved": True,
            "media_preserved": True, "relationships_valid": True,
            "changed_existing_parts": [k for k in a.keys() & b.keys() if a[k] != b[k]],
            "added_parts": sorted(b.keys() - a.keys())}


def stage():
    app, _ = connect()
    assert not TEST.exists()
    p = app.Presentations.Open(str(BACKUP), False, False, False)
    try:
        edit(p)
        p.SaveAs(str(TEST), 24)
        export(p)
        bad = bounds(p)
        print(json.dumps({"overflow": bad}, ensure_ascii=False))
        assert not bad
    finally:
        p.Close()
    print(json.dumps(check(), ensure_ascii=False))


def apply():
    check()
    _, p = connect()
    assert p.Saved and p.Slides.Count == 15
    old = Presentation(BACKUP)
    assert [p.Slides.Item(i).SlideID for i in range(1, 16)] == [s.slide_id for s in old.slides]
    # 逐形状核对，避免覆盖制作期间用户在活动演示中的修改。
    for i, old_slide in enumerate(old.slides, 1):
        s = p.Slides.Item(i)
        assert s.Shapes.Count == len(old_slide.shapes)
        for current, prior in zip(s.Shapes, old_slide.shapes):
            assert current.Name == prior.name
            if prior.has_text_frame:
                assert current.TextFrame.TextRange.Text.replace("\r", "\n") == prior.text
    hashes = {n: hashlib.sha256((WORK / f"第{n}页.png").read_bytes()).hexdigest() for n in (14, 15, 16, 17)}
    edit(p)
    assert not bounds(p)
    p.Save()
    assert p.Saved
    p.SaveCopyAs(str(TEST), 24)
    export(p)
    assert hashes == {n: hashlib.sha256((WORK / f"第{n}页.png").read_bytes()).hexdigest() for n in hashes}
    print(json.dumps({**check(), "live_saved": True, "final_render_matches_reviewed": True}, ensure_ascii=False))


def refresh():
    app, _ = connect()
    p = app.Presentations.Open(str(TEST), False, False, False)
    try:
        s = p.Slides.Item(15)
        for name in ("本代反思", "记忆引导精英变异", "统一筛选"):
            s.Shapes.Item(name).Top = 419
            s.Shapes.Item(name).Height = 54
            s.Shapes.Item(name + "_正文").Top = 422
            s.Shapes.Item(name + "_正文").Height = 48
        s.Shapes.Item("本代结束总结_2").Height = 6
        for sh in s.Shapes:
            if sh.Type == 1 and sh.Fill.Type == 3:
                old_color = sh.Fill.ForeColor.RGB
                sh.Fill.Solid()
                sh.Fill.ForeColor.RGB = old_color
        s.Shapes.Item("本代反思_正文").TextFrame.TextRange.Text = "双分支总结并整合\r形成本代长期反思"
        s.Shapes.Item("代末触发标签").Left = 745
        s.Shapes.Item("代末触发标签").Top = 388
        s.Shapes.Item("实测结果标签").Left = 790
        if not any(sh.Name == "分支实测返回整合_0" for sh in s.Shapes):
            arrow(s, "分支实测返回整合", [(780, 298), (730, 298)], TEAL, True)
            text(s, "分支成绩标签", "分支成绩", 671, 317, 105, 23, 18, TEAL)
        notes(s, FLOW_NOTES)
        notes(p.Slides.Item(17), "逐页来源按当前17页演示更新。2–3页介绍PaperBanana；4–9页介绍RoCo；10–11页为方案一，12–13页为方案二，14–15页为方案三。来源[3]仅用于视觉参考。方案三借鉴[2]§3.4–3.5、Algorithm 1与Prompts 8–10；从零设计完整启发式及质量／效率分工为本次讨论的扩展。科学事实、匿名代码行为和本项目设想分别标明，不将新框架称为原论文方法或已验证结果。")
        assert not bounds(p), bounds(p)
        p.Save()
        export(p)
    finally:
        p.Close()
    print(json.dumps(check(), ensure_ascii=False))


if __name__ == "__main__":
    {"stage": stage, "refresh": refresh, "check": lambda: print(json.dumps(check(), ensure_ascii=False)), "apply": apply}[sys.argv[1]]()
