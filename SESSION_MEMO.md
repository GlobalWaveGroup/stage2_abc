# 天枢 TianShu — Session Memo (给下一个AI session)
# 写于: 2026-02-25
# 写给: 下一次对话的AI，请完整阅读后再动手

---

## 0. 如何使用本文件

1. 先读 `GROUND_TRUTH.md` — 用户确认的权威逻辑链（核心哲学、概念定义、5步框架）
2. 再读本文件 — 当前工作状态、已踩的坑、待解决的问题
3. **不要信任本目录下任何.py文件的注释/docstring版本号** — 代码多次被覆盖，注释可能是旧的
4. **以代码实际行为为准，不以注释为准**

---

## 1. 项目目标

给定外汇K线数据，构建完整的波段池（Segment Pool）：
- 输入：raw OHLC K线
- 处理：基础ZG（zigzag）→ 幅度归并 + 横向归并 → 交替迭代至不动点
- 输出：所有尺度的波段共存于一个池中，每条波段有来源标记和重要性评分

波段池是后续步骤（可比性度量、R(X,Y)→P(Z|R)映射、策略向量）的基础。
当前处于**第1步：波段生成**，尚未进入第2步。

---

## 2. 环境

- **本地工作目录**: `/home/ubuntu/stage2_abc/` （已初始化git）
- **远程m5**: `ssh m5` (47.111.129.50:6006), 80核/503G RAM/8×V100, SSH不稳定
- **原始数据**: `/home/ubuntu/DataBase/base_kline/` — 48对×5TF, TSV格式, 25年
- **验证数据**: EURUSD_H1.csv, 当前用末尾200根 (2024-12-18 ~ 2024-12-31)
- **用户沟通语言**: 中文，代码/数据用英文

---

## 3. 核心代码文件 (可信赖的)

### 3.1 merge_engine_v2.py (625行) — 归并引擎

关键函数:
- `load_kline()` — TSV加载
- `calculate_base_zg(high, low, rb=0.5)` — KZig风格基础zigzag，**不是标准zigzag(2,1,1)**，每个拐点约2根K线
- `_check_amp_merge(p1,p2,p3,p4)` — 幅度归并条件：P1和P4为4点中的极值对
- `amplitude_merge_one_pass(pivots)` — 幅度归并一轮，返回(新拐点, changed, 滑窗发现的所有对)
- `classify_three_segments(pivots, i)` — 横向归并4类分类
- `_check_lat_merge(pivots, i)` — 横向归并条件检查
- `lateral_merge_one_pass(pivots)` — 横向归并一轮，返回(新拐点, changed, 滑窗发现的所有对)
- `full_merge_engine(pivots)` — **主引擎**，交替迭代至不动点
- `compute_pivot_importance(results)` — 拐点重要性(快照出现次数 + 端点次数)
- `build_segment_pool(results, pivot_info)` — 构建去重波段池（含extra_segments）
- `prune_redundant(pool)` — 冗余删除（**用户说暂不执行**，逻辑存在但不应调用）

数据结构:
- 拐点: `(bar_index, price, direction)`, direction=1峰/-1谷
- 快照: `(type, label, pivots_list)`, type='base'|'amp'|'lat'
- extra_segments: `[(p_start, p_end, source_label)]` — 滑窗收集的被贪心跳过的线段

### 3.2 visualize_v2.py (335行) — HTML可视化生成器

- 读取引擎输出，生成交互式Canvas图表
- 快照层：base(金色细线) / amp(红色渐变实线) / lat(青色虚线)
- Extra层：滑窗收集的额外线段（绿色/橙色虚线，半透明）
- 控件：Show All / Amp Only / Lat Only / Extra开关 / Step+/- 逐步回放
- 输出: merge_v2.html

### 3.3 GROUND_TRUTH.md (326行) — 权威逻辑链文档

用户亲自确认的核心设计。**修改前必须与用户确认**。
Section 1-7 是多个session积累的稳定内容。
Section 8 (归并引擎设计) 在本session中被多次修改，**以代码实际行为为准**。

### 3.4 k-zg归并all-2.3.1m.mq5 — MT5原始指标源码 (UTF-16LE, 378行)

用户MT5端的幅度归并指标。本session已读取并据此实现Python版本。

---

## 4. 当前引擎架构 (v2.1)

```
full_merge_engine(pivots):
    current = 基础ZG拐点
    记录快照: L0
    
    while(有变化):
        # 尝试幅度归并一轮
        if 幅度有变化:
            滑窗收集: 所有满足amp条件的拐点对 → extra_segments
            横向滑窗收集: 当前序列上满足lat条件的拐点对 → extra_segments
            贪心推进: 产生下一级拐点序列
            记录快照: A1, A2, ...
            continue
        
        # 幅度无机会 → 横向归并一轮
        if 横向有变化:
            滑窗收集: 满足lat条件的拐点对 → extra_segments
            贪心推进: 产生下一级拐点序列
            记录快照: T1, T2, ...
            continue  # 回馈给幅度归并
        
        break  # 不动点
```

### 200根K线的实际运行结果:

```
L0(109) → A1(63) → A2(41) → A3(25) → A4(15) → A5(9) → A6(7) → T1(5) → T2(3)
9轮迭代, 波段池301条
```

---

## 5. 用户已确认的关键决策

1. **基础ZG用KZig(rb=0.5)而非标准zigzag(2,1,1)** — 可接受作为原子度量工具
2. **幅度归并和横向归并双向互生** — 不是单向依附，横向结果必须回馈幅度
3. **横向归并4类分类**: 收敛→连, 扩张→连, b最长有交错→连, b最长无交错→不连
4. **所有中间级别的线段都有意义** — 不应因贪心跳跃而丢失
5. **滑窗收集+贪心推进**: 线段收集不遗漏，拐点序列正常递进
6. **冗余删除暂不执行** — 在横向/纵向归并完全交互完成之前，冗余是"过程中必要的"
7. **验证窗口用200根K线** — 计算量和输出可控

---

## 6. 已知问题和未解决的争议

### 6.1 [严重] 300K HTML版本已被覆盖

之前某个session生成的merge_v2.html达到300K+，包含旧引擎（每个幅度级别独立做横向归并穷尽）的全部可视化数据。那个版本的横向归并产出了124条线段（200根K线场景下），用户评价"接近想要的效果"。

当前版本43K，使用新引擎（交替迭代+滑窗收集），横向归并主链只产出2条线段，但通过extra_segments收集了229条额外线段，波段池301条（vs旧版248条）。

**问题**：新引擎的可视化中extra_segments的展示方式（半透明虚线）可能不如旧引擎（每个级别有独立的横向zigzag连线）直观。用户正在看图验证，结论未出。

**教训**：在覆盖文件前应该git commit或备份。已初始化git。

### 6.2 [中等] 幅度归并的贪心遗漏问题

贪心从左到右扫描，找到4拐点极值对就跳3步。但被跳过的位置可能也有满足条件的三波。
当前通过"滑窗收集"解决——滑窗逐个检查所有位置，满足的都记录进extra_segments。
**但这些extra线段没有快照**，不参与后续的归并迭代，只进入波段池。

用户说"考虑到你的上下文"，可能暗示对这个方案不完全满意。需要下次确认。

### 6.3 [中等] 横向归并分类的Fallback逻辑

`classify_three_segments`中，当三段不严格匹配类型1/2/3时有个fallback检查交错。这个fallback的几何含义不够清晰，可能导致一些不该连的被连了。需要用具体案例验证。

### 6.4 [低] 基础ZG vs MT5 zigzag(2,1,1)

Python的KZig和MT5的zigzag(2,1,1)可能有细微差异。用户说"可接受"但未做过逐点对比验证。如果后续移植到MT5，需要确认一致性。

### 6.5 [低] 冗余删除逻辑

`prune_redundant()`用的是"80%时间重叠 + 60%幅度相似"的硬阈值。用户说暂不执行。这个函数代码保留但main()中不调用。

---

## 7. 不可信赖的文件 (前序session遗留)

以下文件来自之前多个AI session，**逻辑未经用户验证，不要使用**:

- abc_v5_ablation.py, strategy_engine.py, comparability_quality.py — 旧版策略/评分代码
- abc_collector*.py, abc_v2/v3/v4*.py — 旧版数据收集器
- full_system.py, integrated_strategy*.py — 旧版完整系统
- multilevel*.py, pyramid_strategy.py — 旧版多层策略
- scoring_system.py, fuzzy_score.py — 旧版评分
- merge_engine.py (无v2后缀) — v1引擎，已被v2替代
- ABC_TianShu_EA.mq5, ABC_TianShu_TV.pine — 旧版MT5/TV指标，未验证

---

## 8. 下一步工作 (按优先级)

### 8.1 [等待] 用户验证当前可视化

用户正在看 merge_v2.html。需要他确认：
- 幅度归并的快照层是否正确反映了趋势结构
- extra_segments（横向滑窗收集的线段）是否覆盖了他看到的横向整理结构
- 是否还有遗漏的结构类型

### 8.2 [可能需要] 重审旧引擎的横向归并方式

旧引擎在每个幅度级别上独立做横向归并穷尽，产出了更丰富的可视化。如果用户对新引擎的extra_segments展示不满意，可能需要**保留两种模式**：
- 主链：交替迭代至不动点（用于序列推进）
- 辅助：每个中间级别独立做横向归并穷尽（只用于线段收集，不影响主链）

### 8.3 [待做] MT5指标移植

Python验证通过后，需要移植到MQL5在MT5中可视化验证。

### 8.4 [待做] 第2步：可比性度量

波段池确认完成后，进入(X,Y)配对和R(X,Y)关系向量的设计。

---

## 9. 与用户沟通的注意事项

- **中文沟通**，代码/数据可英文
- **不要美化结果**，用户要求"brutal honesty"和统计严格性
- **每步必须验证** — 4层验证框架（概念→单案例→小批量→统计）
- **不要主动推进到下一步** — 当前步未经用户确认绝不进下一步
- **任何AI总结都不可信** — 包括本文件。以用户确认为准。
- **文件修改前git commit** — 已有过300K文件被覆盖的教训
- 用户的表达方式有时引用道家哲学，但背后是严格的数学/工程思维

---

## 10. 快速验证命令

```bash
# 运行引擎 (200根K线)
python3 /home/ubuntu/stage2_abc/merge_engine_v2.py

# 生成可视化
python3 /home/ubuntu/stage2_abc/visualize_v2.py
# 输出: /home/ubuntu/stage2_abc/merge_v2.html

# 查看git状态
cd /home/ubuntu/stage2_abc && git log --oneline

# 调整K线数量: 修改两个文件中的 limit=200
# merge_engine_v2.py 第545行
# visualize_v2.py 第329行(main函数)
```
