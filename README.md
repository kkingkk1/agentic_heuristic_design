# 四智能体通用启发式自动设计

在给定求解框架中自动设计启发式模块，沿用 RoCo 主流程，将两个设计角色替换为解质量智能体和速度智能体。

P0 已完成 ACO＋TSP 白盒单任务一代工程验收。P1 已完成任务和方法接入，并冻结 TSP50 首批矩阵。P2 已完成本方法与 RoCo 的三次独立研发及测试，结果见 [TSP50 对照报告](vault/Results/Reports/tsp50-roco-20261114.md)。该结果只适用于本批单任务描述性比较；项目尚无跨任务性能结论。运行采用 Windows 本机 Python 与已有 Codex CLI，模型为当前订阅的 GPT-5.6 Luna。

## 当前入口

- [P0 单任务一代接入与验收](vault/Plan/mechanisms/P0-单任务一代接入与验收.md)：模型真实调用、已有评价脚本、完整一代与失败检查。
- [唯一实施主计划](vault/Plan/mechanisms/多智能体协作的通用启发式自动设计.md)：研究目标、方法逻辑与阶段路线。
- [已确认框架](vault/Knowledge/通用启发式自动设计框架.md)：方法事实与评价定义。
- [P1 任务与对照接入](vault/Plan/mechanisms/P1-任务与正式对照接入.md)、[P2 正式外部对照](vault/Plan/mechanisms/P2-正式外部对照.md)、[P3 同设置对照与内部消融](vault/Plan/mechanisms/P3-同设置对照与内部消融.md)。
- [研究进度入口](vault/01-Research-Plan-and-Progress.md)、[知识索引](vault/02-Index.md)、[项目规则](AGENTS.md)。

## 资料

- [RoCo 原文](vault/References/RoCo/2025__Xu__RoCo_Role-Based_LLMs_Collaboration_for_Automatic_Heuristic_Design.pdf)、[EoH 原文](vault/References/EoH/2024__Liu__Evolution_of_Heuristics.pdf)。
- [来源与历史记录](vault/Knowledge/Source-Provenance.md)、[完整可见会话](vault/History/查找_ROCO_输入_完整会话.md)。
- [方案三演示](vault/Writing/方案三_通用启发式自动设计.pptx)为较早设计快照，当前方法以框架和实施稿为准。
