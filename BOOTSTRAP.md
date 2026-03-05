# 天枢 TianShu — BOOTSTRAP (给下一个AI session的启动指南)
# 最后更新: 2026-02-28

---

## 0. 如何使用

1. 读本文件 (当前状态 + 知识索引)
2. 读 `GROUND_TRUTH.md` (不变的核心哲学)
3. 按需读 `knowledge/` 下的专题文件 (不用全读, 根据任务选择)
4. **不要信任py文件的注释/版本号** — 以代码实际行为为准

---

## 1. 项目一句话

基于zigzag归并的多级波段池, 通过模长对称预测下一段走势, 构建外汇交易策略。

## 2. 当前阶段

**理论验证阶段** — 1-2周内给出go/no-go结论。

核心未验证假设: **mod_C ≈ mod_A 在正确的AB约束条件下是精确的预测公式**。
- 对称性本身已证实 (z=97.8, 显著高于随机)
- 但预测精度严重依赖如何定义"好的AB" — 约束条件尚未确定
- 用户将通过可视化案例指导建立正确约束

## 3. 可信代码

| 文件 | 角色 | 状态 |
|------|------|------|
| `merge_engine_v3.py` | 核心引擎 | ✅ 商用基础架构 |
| `dynamic_engine.py` | 增量引擎 | ✅ 功能完整 |
| `fsd_engine.py` | 轨迹追踪 | ✅ 架构OK, 未充分验证 |
| `emerge_v32.py` | 多TF可视化 | 🚧 开发中 |
| `visualize_v3.py` | 静态可视化 | ✅ 旧版可用 |

## 4. 数据源

- **原始**: `/home/ubuntu/DataBase/base_kline/` (48品种, TSV, 有时间戳)
- **归一化**: `/home/ubuntu/database2/{TF}/` (106品种, CSV, close[0]=1.0, **无时间戳**)
- **预建zig**: `/home/ubuntu/database2/zig_all_v21/cl128/M15/` (106品种)
- 详见 `knowledge/arch_data_assets.md`

## 5. 当前待做

1. **emerge v3.2 可视化**: 四窗口(M5/M15/M30/H1), 用normalized数据, 无时间戳, 支持缩放/平移
2. **AB约束条件**: 用户将在可视化上指出具体案例, AI反向推导约束公式
3. **mod_C预测验证**: 在正确约束下重跑统计, 看精度是否显著提升

## 6. 用户沟通要点

- 中文沟通, 代码英文
- **brutal honesty**, 不美化结果
- **时间戳无意义** — 只看相对位置/几何关系
- **线段就是线段, 不分级别**
- **模长对称优先于幅度对称**
- 做到不行了再问

## 7. 知识库索引

| 文件 | 内容 |
|------|------|
| `knowledge/arch_merge_engine.md` | 引擎API完整参考 |
| `knowledge/arch_data_assets.md` | 数据资产清单 |
| `knowledge/findings_statistics.md` | 统计发现和数值 |
| `knowledge/README.md` | 知识库使用说明 |
