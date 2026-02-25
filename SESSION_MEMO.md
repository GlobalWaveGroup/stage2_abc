# 天枢 TianShu — Session Memo (给下一个AI session)
# 写于: 2026-02-25 (v3 session)
# 写给: 下一次对话的AI，请完整阅读后再动手

---

## 0. 如何使用本文件

1. 先读 `GROUND_TRUTH.md` — 用户确认的权威逻辑链（核心哲学、概念定义、5步框架）
2. 再读本文件 — 当前工作状态、已踩的坑、待解决的问题
3. **不要信任本目录下任何.py文件的注释/docstring版本号** — 代码多次被覆盖，注释可能是旧的
4. **以代码实际行为为准，不以注释为准**
5. **当前可信代码: merge_engine_v3.py + visualize_v3.py** — v2文件保留但已被v3取代

---

## 1. 项目目标

给定外汇K线数据，构建完整的波段池（Segment Pool）：
- 输入：raw OHLC K线
- 处理：基础ZG → 逐级归并(初始池) → **pool_fusion(消融级别界限)** → 完整波段池
- 输出：所有尺度的波段共存于一个池中，线段就是线段，不分级别

**v3 里程碑**: 波段池被用户确认为"可作为商用底层基础架构"。
当前处于**第1步：波段生成**，静态池已基本完成，下一步是增强功能（见第8节）。

---

## 2. 环境

- **本地工作目录**: `/home/ubuntu/stage2_abc/` （已初始化git，5+ commits）
- **远程m5**: `ssh m5` (47.111.129.50:6006), 80核/503G RAM/8×V100, SSH不稳定
- **原始数据**: `/home/ubuntu/DataBase/base_kline/` — 48对×5TF, TSV格式, 25年
- **验证数据**: EURUSD_H1.csv, 当前用末尾200根 (2024-12-18 ~ 2024-12-31)
- **用户沟通语言**: 中文，代码/数据用英文

---

## 3. 核心代码文件 (可信赖的)

### 3.1 merge_engine_v3.py (~888行) — 归并引擎

关键函数:
- `load_kline()` — TSV加载
- `calculate_base_zg(high, low, rb=0.5)` — KZig风格基础zigzag
- `_check_amp_merge(p1,p2,p3,p4)` — 幅度归并条件
- `amplitude_merge_one_pass(pivots)` — 幅度归并一轮 (滑窗+贪心)
- `classify_three_segments(pivots, i)` — 横向归并4类分类
- `lateral_merge_one_pass(pivots)` — 横向归并一轮 (滑窗+贪心)
- `full_merge_engine(pivots)` — 逐级交替迭代至不动点
- `compute_pivot_importance(results)` — 多维拐点重要性(7维)
- `build_segment_pool(results, pivot_info)` — 构建初始去重波段池
- **`pool_fusion(pool, pivot_info)`** — 🔑 **v3核心**: 池内三波归并，消融级别
- `prune_redundant(pool)` — 冗余删除（**暂不执行**）

### 3.2 visualize_v3.py (~524行) — HTML可视化生成器

- 快照层：base(金色) / amp(红色渐变) / lat(青色虚线)
- Extra层：滑窗收集的额外线段（绿色/橙色虚线）
- **Fusion层**：池内三波归并产出的新线段（**紫色系，虚线**）
- Fusion Top N 滑块：控制显示最重要的前N条
- Min imp 滑块：按重要性过滤
- 鼠标悬停fusion线段显示详情（来源、via路径、幅度、重要性）
- Top20重要拐点标记 (H1-H10, L1-L10)
- Step+/- 逐步回放
- 输出: merge_v3.html (429KB)

### 3.3 GROUND_TRUTH.md — 权威逻辑链文档

### 3.4 k-zg归并all-2.3.1m.mq5 — MT5原始幅度归并指标源码

### 3.5 旧版文件 (保留但已被v3取代)
- merge_engine_v2.py, visualize_v2.py, merge_v2.html — v2版本
- merge_engine.py — v1版本

---

## 4. v3 架构

```
Phase 1: 逐级归并（和v2相同）
    基础ZG(109pv) → 幅度+横向交替 → 不动点(3pv)
    产出: 初始池 301条 + 快照9个 + extra_segments 229条

Phase 2: pool_fusion() — 消融级别界限 🔑
    核心原则: 线段就是线段，不分级别/来源
    算法: 
      1. 用端点建邻接索引
      2. 搜索所有首尾相连的三段组合
      3. 无条件产出新线段(首→尾)
      4. 新线段参与下一轮搜索
      5. 循环至不动点
    结果: 301 → 2970条 (+2669 fusion), 3轮不动点, 0.166秒
```

### 验证通过的关键连接:
| 连接 | 路径(via) | 状态 |
|------|-----------|------|
| H1(bar3)→L8(bar72) | — | OK |
| H1(bar3)→L5(bar98) | — | OK |
| L4(bar39)→H2(bar166) | via(52,98) | OK |
| H7(bar142)→L2(bar195) | — | OK |

---

## 5. 用户已确认的关键决策

1. **基础ZG用KZig(rb=0.5)而非标准zigzag(2,1,1)** — 可接受
2. **幅度归并和横向归并双向互生** — 交替迭代
3. **横向归并4类分类**: 收敛→连, 扩张→连, b最长有交错→连, b最长无交错→不连
4. **滑窗收集+贪心推进**: 不遗漏中间结构
5. **冗余删除暂不执行**
6. **验证窗口200根K线**
7. **🔑 消融级别界限**: 线段就是线段，三条首尾相连→无条件产出
8. **🔑 v3是商用底层基础架构**: 可用于数量统计、特征统计、向量统计等方向
9. **300K旧HTML只是因为2000根K线导致数据量大，不是逻辑更好** — 已澄清

---

## 6. 已知问题

### 6.1 [已解决] 高层级连接缺失
H1→L8, L4→H2, H7→L2 等连接在v2中缺失。
**原因**: 逐级归并的贪心吃掉了重要中间拐点（如L4在A5消失）。
**解决**: pool_fusion()在池内无条件三波归并，消融级别界限。

### 6.2 [低] 基础ZG vs MT5 zigzag(2,1,1)细微差异
用户说"可接受"但未逐点对比。

### 6.3 [低] 横向归并分类的Fallback逻辑
`classify_three_segments`的fallback几何含义不够清晰。

---

## 7. 不可信赖的文件

以下文件来自之前多个AI session，**逻辑未经用户验证，不要使用**:
- abc_v5_ablation.py, strategy_engine.py, comparability_quality.py
- full_system.py, integrated_strategy*.py, multilevel*.py, pyramid_strategy.py
- scoring_system.py, fuzzy_score.py
- merge_engine.py (v1), merge_engine_v2.py (已被v3取代)

---

## 8. 下一步功能方向（用户明确要求，按编号）

### 8.1 点的重要性 — 增加时间衰减维度
- **新维度 D8: recency（时间临近性）** — 离最后一根K线越近，权重越大
- 设计: 可用指数衰减 exp(-lambda * (total_bars - bar_idx)) 或线性衰减
- 需加入 `compute_pivot_importance()` 的维度和权重中

### 8.2 点的重要性 — 可调百分比截断 + 赋值
- 当前: Top10峰+Top10谷，按排名序号标记
- **需要改为**: 可调百分比（缺省前50%），并给出一个重要性赋值（不是序号）
- 比如: 如果有54个峰，前50%=27个峰，每个有一个归一化的重要性得分
- 不是按排名1,2,3...而是按实际importance值

### 8.3 线的重要性 — 端点重要性的乘积
- **当前**: importance = min(两端点重要性) — 短板决定
- **修改为**: importance = 两端点重要性的**乘积**
- 乘积比min更合理：两端都重要时乘积远大于只有一端重要的

### 8.4 对称结构识别（5种维度对称）
- 在波段池中寻找**对称结构**
- **5种维度对称（AI初步理解，待用户确认）**:
  1. **幅度对称**: |seg_A| ≈ |seg_C| （经典ABC等幅）
  2. **时间对称**: time_A ≈ time_C （等时间发展）
  3. **斜率对称**: slope_A ≈ -slope_C （镜像倾斜）
  4. **内部结构对称**: complexity_A ≈ complexity_C （相似的内部波段数）
  5. **幅度-时间复合对称**: (amp_A/time_A) ≈ (amp_C/time_C) 即速率对称
- **每个对称结构的结束位置 = 重要转折点** — 这是对称预测的核心价值
- 实现方向: 对池中每对(seg_A, seg_B, seg_C)三波，计算5维对称度向量

### 8.5 动态K线生命周期 — 从静态到动态
- 把静态的波段池过程，用**动态K线逐步生成**来表达
- 每条线段有**生命周期**: 产生(哪根K线开始)→发展→结束
- 在K线推进过程中，多条线段同时存活，各自在发展中
- **关键**: 当多条线段同时达到对称均衡状态 → **转折概率大增**
- 这个"多线段对称均衡共振"信号 → 用于**机器学习**的特征
- 实现方向: 
  - 为每条线段标记 birth_bar 和 death_bar（或当前是否存活）
  - 在每根K线上，计算当前存活线段的对称均衡度
  - 多线段均衡共振 → 转折概率 → ML特征向量

---

## 9. 与用户沟通的注意事项

- **中文沟通**，代码/数据可英文
- **不要美化结果**，用户要求brutal honesty和统计严格性
- **每步必须验证** — 4层验证框架
- **不要主动推进到下一步** — 当前步未经用户确认绝不进下一步
- **文件修改前git commit**
- 用户的表达方式有时引用道家哲学，但背后是严格的数学/工程思维
- **v3被视为商用基础架构** — 后续修改要慎重，保持向后兼容

---

## 10. 快速验证命令

```bash
# 运行v3引擎 (200根K线)
python3 /home/ubuntu/stage2_abc/merge_engine_v3.py

# 生成v3可视化
python3 /home/ubuntu/stage2_abc/visualize_v3.py
# 输出: /home/ubuntu/stage2_abc/merge_v3.html

# 查看git状态
cd /home/ubuntu/stage2_abc && git log --oneline
```
