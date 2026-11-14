# 来源与迁移范围

更新日期：2026-11-09。

## 项目来源

源项目：`E:\OneDrive\project\static_determine\roco_paperbanana`。
新项目：`E:\OneDrive\project\static_determine\agentic_heuristic_design`。
主机：`DESKTOP-UKPM2VE`。
迁移时原项目不是 Git 仓库，没有 branch、HEAD 或 commit 可记录；当次迁移未自动初始化 Git、提交或建立远端。这是迁移时的记录，当前项目状态以本项目仓库为准。

用户首先说“方案2”，随后明确更正为“方案3”。
本项目只承接四智能体通用启发式种群演化路线。
方案一、方案二与卫星问题模板继续归原项目。

## 已接收材料

- 迁移时确认的架构与四角色输入输出已收录到[框架文档](通用启发式自动设计框架.md)；后续决定以该文档为准。
- 原合集第14–15页，提取为[独立方案演示](../Writing/方案三_通用启发式自动设计.pptx)，补充来源与边界页。
- 原[完整合集快照](../Writing/面向卫星任务调度的多智能体协作研究.pptx)，供历史制作记录的引用与溯源。
- 两个`scheme_three`制作目录，共22个文件、40,136,616字节。逐文件复制后校验SHA-256一致。
- 原会话的完整可见消息、机器可读JSONL及三份附件副本，保存到[会话历史](../History/查找_ROCO_输入_完整会话.md)。
- RoCo 原文、正式笔记、演示及完整 MinerU 解析，直接保存到 `vault/References/RoCo/`，共 51 个文件、8,716,119 字节。复制后逐文件核验大小与 SHA-256 一致，源文件保留。

完整文件清单和源端清理范围见[迁移清单](../../outputs/migration_20261021/迁移清单.json)。

## 会话身份与范围

会话ID：`01a1574e-adf4-7030-aea3-61bda75d9b62`。
应用标题：`查找 ROCO 输入`。
导出合并该会话的三个连续记录文件，按消息ID去重，保留可见消息原文与顺序。
不以摘要替代对话，也不导出系统指令、开发者指令或模型内部推理。
这是导出时点的快照，之后的追加消息需再次导出才能包含。

用户已选择“先保存完整会话记录，稍后再改侧栏归属”。
因此在导出时，该应用内会话及其工作目录仍属于原项目。

## 原文依据与本方案扩展

个体表示按 EoH §3.3 组织为 Thought、Code、Fitness，思路与代码共同演化及先描述再编码依据 §3.1、§3.4。
反思、角色记忆与精英变异借鉴 RoCo §3.5、Algorithm 1及Prompts 8–10。
无种子初始化依据 EoH §3.4（PDF 第 4 页）与附录 B.1（第 14 页），30 个原始提案和种群规模 10 依据 RoCo 附录 D。RoCo 未详述初始化生成方式；将上述生成方式与数量设置结合，并按统一评价、错误记录、有效候选去重后筛选，属于本项目接入约定。已有参考种子仅用于工程检查，不进入首版初始化提示或种群；历史讨论曾将有种子初始化作为有效率过低时单独比较的备选，不是当前已启动任务。
待设计模块的开放范围和解质量／速度角色分工属于本项目决定。
不将这些扩展说成RoCo已经验证的能力。

本项目直接使用以下项目内资料：

- [EoH 原文 PDF](../References/EoH/2024__Liu__Evolution_of_Heuristics.pdf)。
- [RoCo 原文 PDF](../References/RoCo/2025__Xu__RoCo_Role-Based_LLMs_Collaboration_for_Automatic_Heuristic_Design.pdf)。
- [全文总结](../References/RoCo/01-full-paper-summary.md)与[可迁移知识](../References/RoCo/03-transferable-knowledge.md)。
- [MinerU 全文](../References/RoCo/mineru/full.md)，同目录保留全部结构数据与图片。
- [RoCo 框架演示](../References/RoCo/05-RoCo-多智能体框架与原理-参考范例修正版.pptx)。

迁入目录保留来源文件的原内容与结构。当前设计和阅读入口不再依赖外部目录。
历史会话、旧制作记录和缓存元数据中的原路径用于追溯，未改写为当前执行入口。
资料库原文件及已有 Zotero 关联未修改，也未为项目副本新建 Zotero 条目。

来源元数据有一项既有不一致：`metadata.json` 与 `mineru/cache-metadata.json` 记录的 PDF 为 4,843,131 字节，哈希以 `c6b8ed09` 开头；实际源 PDF 为 4,849,727 字节，哈希与该记录不同。
本次保留这些历史元数据，不重跑解析，也不宣称当前 PDF 与旧解析版本已核验一致。
实际复制哈希和该差异已记录在迁移清单的 `reference_materials` 字段。

### 2026-10-22 新增 EoH 个体表示依据

EoH 原文通过 Zotero Web API 回读核实，父条目为 `CJAXMLEU`，独立资料目录中的正文附件为 `XNJPP9DQ`。原文题名为 *Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model*，ICML 2024，PMLR 235，出版入口见 [PMLR](https://proceedings.mlr.press/v235/liu24bs.html)。

项目 PDF 从资料库该论文目录的 `00-source.pdf` 复制，大小为 651,933 字节，SHA-256 为 `de3051b2c7bc4e334748f5d888f003a2c6caf62e727847adda62a31503664b0e`。Zotero 中另一个平铺正文附件与它逐字节相同，本次不清理任何源文件或附件。原文 §3.3 和相应 MinerU 段落已交叉核对；本次只采用个体表示和先思路后代码的生成顺序，适应度及四角色协作继续采用本项目合同。

## 未迁入内容

- 卫星问题层Java源码、pom.xml、测试、target与共享Scene。
- 方案一、方案二的独立制作目录。
- 原项目六角色联合路线的计划与问题模板验收记录。
- Zotero 整库与其他会话。迁移时只复制 RoCo 论文和解析资料；EoH 原文于后续单独加入，见上文。
- Codex应用的原始会话状态和项目登记配置。

历史制作副本可能包含其他方案的旧合集，仅用于追溯。
迁入的旧PPT编辑脚本仍假设当时的演示结构，不作为本项目当前实现入口执行。

## 源端清理状态

新项目的复制、导出与独立演示已建立。
用户已确认最终清单，源端两个制作目录已清理，共移除22个文件、40,136,616字节及3个空目录。
原合集的方案三第14–15页已移除，并同步来源附录；清理后当时为15页。
清理前最新合集保存在[迁出前快照](../../outputs/migration_20261021/源端清理前最新合集.pptx)。
未修改ACL；仅在清理受阻后清除3个已批准空目录的只读位，再删除目录，原属性记录在迁移清单中。
OneDrive进程在检查时运行，云端同步完成状态未验证。
源端删除可能传播到OneDrive云端；新项目中的已校验副本、RoCo资料库原件和应用内原会话均保留。
