# 天枢 TianShu — 研发日志
# 全过程记录：每次commit的事实、数据、AI评估、用户反馈、最终结论
# 始于: 2026-02-28

---

## 格式说明

每个条目包含：
- **Commit**: hash + 标题
- **做了什么**: 事实描述
- **关键数据**: 统计/测试结果
- **AI评估**: 方向判断、风险、建议
- **用户反馈**: （多轮交互记录）
- **结论**: 达成一致的最终判断

---

## 历史补录 (2026-02-25 ~ 2026-02-28)

以下为本次session之前的commit摘要，无完整交互记录。

### `13eb1c4` — snapshot before memo: v2.1 engine
- 初始快照，交替迭代归并引擎 + 滑窗

### `b32dcc3` — add SESSION_MEMO.md
- 创建session交接文档

### `e6bd38f` — multi-dimension pivot importance
- 多维拐点重要性评分，Top20标记

### `505ec95` — 波段池三波融合: pool_fusion()
- **里程碑**: pool_fusion() 无条件产出，301→2970线段
- 用户确认 "可作为商用底层基础架构"

### `f54031d` ~ `ea2b2b7` — v3.2: 对称结构识别
- 5维对称向量（幅度/时间/模长/斜率/复杂度）
- D8 recency 维度加入重要性
- 几何形态分类（V底/V顶/上升/下降）

### `695d163` — v3.3: 统一对称谱 (compute_symmetry_spectrum)
- 镜像对 + 中心对扫描，多维ratio向量

### `1bcf763` — symmetry_reality_check
- **负面结论**: 对称性并非客观规律，是几何必然性
- 但后来认定此测试方法有缺陷（聚合特征 vs 独立AB对）

### `e03fcf5` — 动态引擎v1
- DynamicZG + DynamicMerger + DynamicPool
- 逐K线推进无前瞻

### `513f1f9` ~ `7bc5b61` — 全量预计算 + 聚类分析
- 51维统计特征，faiss k-means
- **负面结论**: 对Z预测力极弱（接近随机）
- 后确认：测试方法论错误（聚合slice特征，非独立AB对）

### `cb0a196` ~ `bfff962` — predict_symmetric_image()
- 路径一核心算法首次实现
- Mirror + Center 两种对称预测

### `465a526` ~ `b318778` — v4.2 可视化 + 编码优化
- 动态4面板预测可视化
- 增量编码 22MB→3.2MB

### `8b2ef50` ~ `8be9d2a` — FSD 引擎
- 统一状态-策略框架 (fsd_engine.py)
- 单面板可视化 + 轨迹/偏差/边界

### `32751c3` — FSD v2: 滑窗 pool_fusion + prune_redundant
- **关键**: 2970→300段/窗口，10x压缩，94%端点保留
- 148s→33s/2000bars

### `ff6d3f2` ~ `97339a6` — v3.1 可视化
- 完整v3 pipeline 滑窗200bar/stride10
- A1-A6骨架加粗，L0淡化

### `770a42d` — v32: 实时归并可视化
- FastAPI + WebSocket 架构
- 可调窗口 10-1000，逐K步进，键盘控制

### `a353aa3` — v32: 模长守恒弧线预测
- **核心创新**: C≈A 从直线变为弧线
- R = sqrt(norm_amp² + norm_time²) 守恒
- 弧线 + R包络 可视化

---

## #001 — 2026-02-28

### Commit: `da22a05`
**v32: add triangle (converging/diverging) and modulus-only prediction types**

### 做了什么

在 `predict_symmetric_image()` 中新增两种预测类型：

1. **Triangle（三角收敛/发散）**
   - 收敛: B/A幅度比 0.15 < ratio < 0.85 → 振幅递减
   - 发散: B/A幅度比 1.18 < ratio < 6.0 → 振幅递增
   - 预测: amp_C = amp_B × ratio（等比缩放）
   - 可视化: 金色(收敛)/紫色(发散) + 边界线填充区域

2. **ModOnly（模长守恒但幅度不对称）**
   - 条件: mod_sym > 0.80 且 amp_sym < 0.85
   - 与三角可并存（同一AB对可同时产生triangle + modonly预测）
   - 可视化: 青色虚线弧线

3. **前端更新**
   - 4种类型4套颜色方案
   - 三角边界线延伸到预测区域
   - Info bar显示各类型计数 M/C/T/MO

### 关键数据

**预测类型分布（EURUSD H1, pos=500）：**

| 窗口 | Mirror | Center | Triangle | ModOnly | 总计 |
|------|--------|--------|----------|---------|------|
| 25   | 16     | 14     | 9        | 1       | 40   |
| 50   | 20     | 20     | 0        | 0       | 40   |
| 100  | 2      | 2      | 36       | 0       | 40   |
| 200  | 2      | 0      | 38       | 0       | 40   |

**之前的统计覆盖率分析（200bar窗口, 2582 ABC三元组）：**

| 类别 | 数量 | 占比 |
|------|------|------|
| 经典(amp+mod对称) | 420 | 16.3% |
| ModOnly(mod对称,amp不) | 279 | 10.8% |
| 收敛三角 | 613 | 23.7% |
| 发散三角 | 668 | 25.9% |
| 总可用信号 | 1759 | 68.1% |

**几何正确性验证**: 通过。发散类型确认 B_amp > A_amp，边界线端点正确。

**性能**: Win=25 计算 9.5ms, Win=100 计算 78ms, Win=200 计算 481ms。

### AI评估

**方向判断**: 正确方向。16.3% → 68.1% 覆盖率的提升是实质性的。

**但有几个值得注意的问题:**

1. **三角在大窗口中过度主导** — Win=100时36/40都是三角，把经典对称预测几乎完全淹没了。这可能需要按类型分配配额（比如每种类型最多15条），否则经典对称的高质量信号被三角噪声覆盖。

2. **三角预测的"等比缩放"假设需要验证** — `amp_C = amp_B × ratio` 是一个强假设。收敛三角理论上合理（振幅递减），但等比递减只是最简单的模型。实际市场中三角收敛的缩放比率是否真的保持恒定？这需要统计验证。

3. **ModOnly太少** — 全量只有1-2条，因为条件`mod_sym > 0.80 且 amp_sym < 0.85`与三角区间（amp_ratio在0.15-0.85或1.18-6.0）高度重叠。74%的modonly候选其实同时也是三角。可能需要重新思考modonly的独立价值。

4. **三角边界线的几何**可能有问题 — 当前用A段和B段的端点画边界，但实际三角形态的边界应该连接同方向的极值点（高点连高点，低点连低点），不是简单的A_start→B_start。需要在可视化中验证。

**建议下一步:**
- 按类型配额限制，避免三角淹没经典
- 对三角缩放假设做回测统计验证
- 用户视觉确认三角边界线是否正确

### 用户反馈

用户通过apple2 (10.10.0.11) 浏览器访问 m1 (10.10.0.4:8765) 验证可视化。
WireGuard VPN组网10节点全部互通。
反馈：能看到画面，但高DPI屏幕下显示区域偏小 → 触发 #002 DPI修复。

### 结论

三角+ModOnly预测类型实现完成，功能正确。
AI提出的4个待验证问题（类型配额/缩放假设/ModOnly价值/边界线几何）暂挂，等用户视觉确认后再逐一处理。

---

## #002 — 2026-02-28

### Commit: (待提交)
**v32: fix high-DPI canvas rendering for Retina displays**

### 做了什么

apple2是Mac，屏幕 devicePixelRatio=2（Retina）。Canvas未处理DPI导致：
- 物理像素和CSS像素1:1 → 在2x屏上只占1/4面积，或者模糊拉伸

修复：
- Canvas物理尺寸 = CSS尺寸 × devicePixelRatio
- `ctx.scale(dpr, dpr)` 让所有绘制代码用CSS像素坐标
- 添加 `<meta viewport>` 标签

### 关键数据

无需统计验证，纯UI修复。

### AI评估

标准的高DPI适配，无风险。

### 用户反馈

"OK，完全可以" — 确认修复有效。

### 结论

DPI修复完成，apple2可正常查看v32可视化。
**重要发现**: m1↔apple2 通过WireGuard VPN互通，apple2可直接浏览器访问 `http://10.10.0.4:8765` 实时查看。这建立了 **开发(m1) → 验证(apple2)** 的闭环。

---

## #003 — v32: 完整移植v3全部9项缺失功能

### Commit

`63e9519` — v32: port all 9 missing features from v3 (symmetry, extra segs, arrowheads, target lines, progress, deviation, 5-dim hover, sliders)

### 做了什么

用户明确指出v32相比v3"缺斤少两，东拼西凑"，要求以v3为标准补全所有功能。
逐项对比v3(963行)代码，移植全部9项缺失功能到v32。

**Server端 (visualize_v32_server.py):**
- 新增 `find_symmetric_structures` 导入和调用
- 新增 `show_sym`, `show_extra` toggle开关
- 新增 `min_imp`, `peak_n`, `valley_n` 滑块参数
- Extra segments: 从 `full_merge_engine` 结果中提取并按label分组
- Symmetry: 调用 `find_symmetric_structures(pruned, pi, top_n=200)`，偏移坐标
- Predictions: 新增 `prog` (actual_progress), `cb`/`cp` (mirror center bar/price)
- Pivots: 改为分离peaks/valleys结构，服务端按slider截取

**Frontend端 (visualize_v32.html):**
1. **Symmetry structures**: A-B-C三波着色 (A=方向色, B=灰虚线, C=同A虚线), 轴心菱形标记, score标签
2. **Extra segments**: 按label分组, amp/lat颜色区分, 虚线渲染
3. **Arrowheads**: 所有预测C'线段末端箭头 (弧线末端 + 三角末端 + fallback直线末端)
4. **Target price horizontal line**: 淡色虚线水平标记目标价
5. **Progress indicator**: 绿色实线显示C段已展开部分
6. **Deviation annotation**: 当前K线vs预期路径的偏差 (±N pips)
7. **Mousemove 5-dim vector**: 悬停对称结构时显示 amp/time/mod/slope/cplx 5维向量
8. **Min importance slider**: 全局最低重要性门槛控制
9. **Peak/Valley sliders**: 分别控制显示的峰/谷数量

新增3个helper函数: `drawArrowhead()`, `drawTargetLine()`, `drawProgress()`, `drawDeviation()`

### 关键数据

WebSocket测试 (window=100, end_bar=200, 全功能开启):
- 计算时间: 175ms
- Extra: 8组, 83段
- Pool: 163段 (min_imp=0)
- Symmetry: 200结构
- Predictions: 658条 (含prog字段, mirror有cb/cp, triangle有st/ar)
- Pivots: 10pk/10vl (总24pk/24vl)

### AI评估

这次修改的本质是**基于v3参考实现的功能移植**，而非重新发明。策略正确：
- 从v3逐行对照提取每个功能的渲染逻辑
- 适配v32的WebSocket架构（数据从server获取，而非v3的静态嵌入）
- 保留v32已有的4种预测类型 + 弧线渲染（v3只有mirror/center直线）

风险: 前端代码量从883行增至~1000行，但结构清晰（helper函数分离），可维护性可接受。

### 用户反馈

（待用户在apple2上验证）

### 结论

（待用户确认后补充）

---
