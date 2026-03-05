# 天枢 TianShu — 分层知识库

## 给AI的使用说明

这个目录是天枢项目的持久化知识库。每次新session：

1. **必读**: `/home/ubuntu/stage2_abc/GROUND_TRUTH.md` (核心哲学，不变)
2. **必读**: `/home/ubuntu/stage2_abc/BOOTSTRAP.md` (当前状态/指针，每次session结束更新)
3. **按需**: 本目录下的专题文件，根据任务需要 grep/read

## 文件索引

| 文件 | 内容 | 何时需要 |
|------|------|----------|
| `arch_merge_engine.md` | merge_engine_v3 完整架构和API | 修改引擎、理解数据流 |
| `arch_data_assets.md` | database2 数据资产清单 | 加载数据、理解归一化 |
| `arch_code_map.md` | 代码文件地图（活/死/角色） | 找代码、判断文件用途 |
| `findings_statistics.md` | 所有统计发现和数值结论 | 做统计分析、引用前次结果 |
| `findings_dead_ends.md` | 已证明的死路和原因 | 避免重复踩坑 |
| `decisions_log.md` | 用户确认的关键决策时间线 | 理解为什么这样设计 |
| `session_history.md` | 各session概要（发现/决策/产出） | 理解项目演进脉络 |

## 更新规则

- AI不应自行修改 GROUND_TRUTH.md（需用户确认）
- AI应在每次session结束时更新 BOOTSTRAP.md
- 专题文件由AI在发现新知识时追加更新
- 所有文件使用中文描述 + 英文代码/数据
