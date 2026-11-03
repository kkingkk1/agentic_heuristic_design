# RoCo 多智能体框架与原理：PPT 验收记录

日期：2026-10-07。

## 交付与范围

已完成用户确认的 12 页中文、16:9 演示文稿。
内容集中于多智能体框架、角色分工、协作时序、文本反思、跨代记忆与精英变异。
每页包含讲解备注与原文定位。
流程图使用 PowerPoint 原生形状和连接线，消融表使用原生表格。
文件未使用原文图像截图作为整页内容。

本轮创建或修改的文件：

- [演示文稿](E:/OneDrive/my_library/10_Academic_Literature/Preprints/2025__Xu__RoCo_Role-Based_LLMs_Collaboration_for_Automatic_Heuristic_Design/04-RoCo-多智能体框架与原理.pptx)。
- [本验收记录](E:/OneDrive/my_library/10_Academic_Literature/Preprints/2025__Xu__RoCo_Role-Based_LLMs_Collaboration_for_Automatic_Heuristic_Design/04-RoCo-多智能体框架与原理-验收.md)。
- [验证用渲染 PDF](E:/OneDrive/project/static_determine/roco_paperbanana/outputs/roco-ppt/04-RoCo-多智能体框架与原理.pdf)。这是制作中间产物，不作为资料库正式 PDF 归档或 Zotero 同步对象。
- [制作与核验脚本](E:/OneDrive/project/static_determine/roco_paperbanana/outputs/roco-ppt/make_roco_ppt.py)。

未修改源论文、已有 MinerU 缓存、项目 Java 代码或问题合同。
未执行安装、Git 提交、推送或 Zotero 写操作。

## 来源核验

- 论文：Jiawei Xu, Fengfeng Wei, Weineng Chen. *RoCo: Role-Based LLMs Collaboration for Automatic Heuristic Design*。
- 版本：正文首页明确标注 arXiv:2512.03762v2，4 Dec 2025，共 22 页。
- [原文入口](https://arxiv.org/abs/2512.03762v2)。
- Zotero Web API 实时解析的父条目：`J7FI486W`。
- 正文附件：`2KFHK2IA`，`linked_file`，`application/pdf`。
- 本地正文大小：4,843,131 字节。
- SHA-256：`c6b8ed097fb26c96191bf0894ddb5e4ca873ec335d62faffcf44873376a1aa6b`。
- 已有 MinerU 缓存的附件键、大小、修改时间和哈希与当前正文一致，因此复用缓存。
- 框架图以原文 Figure 1 为依据重建。
- 协作顺序直接核对 PDF p.10 的 Algorithm 1。
- 消融数字直接核对 PDF p.6 的 Table 3，并查看该页渲染图。

## 逐页证据地图

1. **导读**：论文题名、作者、版本，PDF p.1。
2. **分工动机**：创新与局部改进的互补，§1，PDF pp.1–2。
3. **总体框架**：EoH 候选与 RoCo 候选合并、Top-N 筛选，§3.3、Figure 1、Algorithm 1，PDF pp.3–4、10。
4. **四角色**：Explorer、Exploiter、Critic、Integrator，§3.4；模型和采样设置见 §4.1，PDF pp.4–5。
5. **启发式表示**：描述、代码、分数，§3.2；求解框架中的函数接口见附录，PDF pp.3、12–13。
6. **单轮协作**：评议、双分支改进、融合，Algorithm 1 lines 9–17；提示见 Appendix E，PDF pp.10、14–16。
7. **多轮与精英融合**：分支最佳候选和最终融合，Algorithm 1 lines 3–23，PDF p.10。
8. **反思与记忆**：短期反思、目标值变化、长期反思和历史记忆，§3.5 Eqs.15–16、Prompt 9，PDF pp.4、16。
9. **精英变异**：两精英各三种角色变异，§3.5 Eqs.17–18、Algorithm 1、Prompt 10，PDF pp.4、10、16。
10. **消融证据**：Table 3 的 TSP 黑盒结果，PDF p.6。
11. **总结与边界**：§3、Table 3、Appendix D，PDF pp.3–4、6、13。
12. **参考来源**：完整论文引文、核对入口和返回第 11 页的内部链接。

## 内容边界

- Critic 提供文字反馈，候选分数来自程序执行与问题评价。
- 四角色可共享模型后端；论文实验使用 GPT-4o-mini。
- 文本记忆更新不等同于模型参数训练。
- 在 ACO 和 GLS 中，生成的函数常是求解框架内的启发式组件。
- 消融结果限定在原文的 TSP 设置，不扩展为所有任务的保证。
- 第 10 页仅报告 Table 3 中六行黑盒结果，均值与标准差逐项核对。
- 去掉 Explorer 的标准差小于完整 RoCo，未声称完整框架在每一项都具有最低方差。
- 正文的调用预算与 Appendix D 的评价预算口径不同，讲解备注和总结页已注明。
- Algorithm 1、§3.3 和提示模板有比较对象、候选收集范围及记忆输入的简写差异，相关备注保留说明。
- 反思和提示的文本标签存在不一致，未照抄错误标签作为确定算法分支。
- 未证明 RoCo 对卫星成像调度的效果；项目问题合同保持原状。

## 制作与核验

使用本机已有 Python 3.13.12、python-pptx、MarkItDown、PyMuPDF、Pillow、lxml 和 LibreOffice。
字体为 Microsoft YaHei；本机字体文件存在，最终渲染中文完整。
制作使用 `python-pptx`，这是新建文稿，不涉及既有 PPT 的包保真。
任务以 Normal 优先级执行；未改变全局 Office 设置或工具链。

工作目录：`E:\OneDrive\project\static_determine\roco_paperbanana`。

实际命令：

```powershell
python 'C:\Users\a_wei\.codex\skills\zotero-pdf-link\scripts\resolve_zotero_linked_pdf.py' J7FI486W
python 'E:\OneDrive\project\static_determine\roco_paperbanana\outputs\roco-ppt\make_roco_ppt.py' build
python 'E:\OneDrive\project\static_determine\roco_paperbanana\outputs\roco-ppt\make_roco_ppt.py' render
python 'E:\OneDrive\project\static_determine\roco_paperbanana\outputs\roco-ppt\make_roco_ppt.py' inspect
```

渲染命令由脚本串行调用，设置 180 秒超时：

```powershell
& 'C:\Users\a_wei\Tools\LibreOffice\SourceDir\LibreOffice\program\soffice.com' --headless --convert-to pdf --outdir 'E:\OneDrive\project\static_determine\roco_paperbanana\outputs\roco-ppt' 'E:\OneDrive\my_library\10_Academic_Literature\Preprints\2025__Xu__RoCo_Role-Based_LLMs_Collaboration_for_Automatic_Heuristic_Design\04-RoCo-多智能体框架与原理.pptx'
```

检查结果：

- PPTX 包完整、可读取：通过。
- 幻灯片数为 12：通过。
- 12 页讲解备注均存在：通过。
- 角色名称、关键数字和来源脚注：通过。
- MarkItDown 最终文件文本检查及占位符扫描：通过。
- 全部形状在页面范围内：通过。
- 渲染 PDF 为 12 页：通过。
- 每页每个文本框与表格单元格的完整文本均可在对应 PDF 页面回读：通过。
- 全部 12 页视觉检查：通过。
- 第 3、7、11 页修改后的再次视觉检查：通过。
- 源论文哈希保持不变：通过。

## 视觉复核与修正

独立只读 Visual Verifier 首轮检查全部 12 页。
首轮发现第 7 页六个框将“候选”拆词换行，严重程度为 low。
已把标签明确分为“第 n 轮”与“候选”两行。
第 3 页补全 EoH 全称。
第 11 页缩小节点间距，使右边距至少 0.5 英寸。
重建、重渲染后复核第 3、7、11 页。
独立复核终态为 **pass**，未发现剩余遮挡、越界、箭头误接或脚注碰撞。

## 已记录的非阻塞问题

- 首次读取原文 PDF 文本时，Windows GBK 输出遇到 `UnicodeEncodeError`。改为显式 UTF-8 输出后读取成功，后续制作脚本固定 stdout 为 UTF-8。
- LibreOffice 转换期间出现 `Could not find platform independent libraries <prefix>`，退出码为 0，且生成的 12 页 PDF 已通过文字与视觉检查。
- MuPDF 预览出现 `No common ancestor in structure tree` 的标签结构警告。页面渲染和文字回读成功；预览从最后一行获取图像数据，避免混入警告文本。
- 最终视觉核验基于本机 LibreOffice 导出的 PDF。未验证其他计算机 PowerPoint 的字体替代或该中间 PDF 的无障碍标签树。

最终状态：**已完成**。
PPT 可直接打开演示；Q&A 可停留在第 11 页原理总结。
