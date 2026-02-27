# 天枢 TianShu — Session Memo (给下一个AI session)
# 写于: 2026-02-27 (FSD v5 session)
# 写给: 下一次对话的AI，请完整阅读后再动手

---

## 0. 如何使用本文件

1. 先读 `GROUND_TRUTH.md` — 用户确认的权威逻辑链（核心哲学、概念定义、5步框架）
2. 再读本文件 — 当前工作状态、已踩的坑、待解决的问题
3. **不要信任本目录下任何.py文件的注释/docstring版本号** — 代码多次被覆盖，注释可能是旧的
4. **以代码实际行为为准，不以注释为准**
5. **当前可信代码: merge_engine_v3.py + dynamic_engine.py + fsd_engine.py + visualize_v5.py** — v3是静态base, FSD是统一框架, v5是当前可视化

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

### 3.1 merge_engine_v3.py (~1757行) — 归并引擎 (v3.2 + 对称预测)

关键函数:
- `load_kline()` — TSV加载
- `calculate_base_zg(high, low, rb=0.5)` — KZig风格基础zigzag
- `_check_amp_merge(p1,p2,p3,p4)` — 幅度归并条件
- `amplitude_merge_one_pass(pivots)` — 幅度归并一轮 (滑窗+贪心)
- `classify_three_segments(pivots, i)` — 横向归并4类分类
- `lateral_merge_one_pass(pivots)` — 横向归并一轮 (滑窗+贪心)
- `full_merge_engine(pivots)` — 逐级交替迭代至不动点
- `compute_pivot_importance(results, total_bars)` — 多维拐点重要性(**8维**, 含D8 recency)
- `build_segment_pool(results, pivot_info)` — 构建初始去重波段池 (重要性=**乘积**)
- **`pool_fusion(pool, pivot_info)`** — 🔑 **v3核心**: 池内三波归并，消融级别
- **`find_symmetric_structures(pool, pivot_info, top_n, max_pool_size)`** — 5维对称向量识别
- **`predict_symmetric_image(pool, pivot_info, current_bar)`** — 🔑 **路径一核心**: 对称映像预测
- `prune_redundant(pool)` — 冗余删除（**暂不执行**）

### 3.2 dynamic_engine.py (~1048行) — 动态引擎
- `DynamicZG` — 逐K线zigzag, confirmed/tentative拐点
- `DynamicMerger` — 增量归并 (amp+lat)
- `DynamicPool` — 增量pool_fusion (O(n^2), 仅用于小数据)
- 对v4.2: **只用 DynamicZG + DynamicMerger** (跳过 DynamicPool)

### 3.3 visualize_v4.py (~540行) — 动态三窗口可视化 (v4.2)
- 2000 bars, 从bar 50开始预测, 窗口150K+50K空白
- 三窗口: Amp / Lat / All
- 增量编码: CONF(confirmed pivots) + TENT_CHG + PRED_CHG
- 交互: 方向键 / 空格播放 / slider
- 输出: merge_v4.html (3.2MB)

### 3.4 visualize_v3.py (~945行) — 静态可视化 (v3.2, 200 bars)

- 静态池可视化, 200根K线窗口
- 输出: merge_v3.html (585KB)

### 3.5 GROUND_TRUTH.md — 权威逻辑链文档

### 3.6 旧版/死路文件 (保留但不要使用)
- merge_engine_v2.py, visualize_v2.py — v2版本
- merge_engine.py — v1版本
- build_vector_store.py, vector_query.py, eval_fast.py — 向量搜索(无信号)
- full_precompute.py, cluster_analysis.py — 聚类分析(无信号)

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

## 8. v3.2 已完成的功能（2026-02-26）

### 8.1 ✅ 点的重要性 — D8 recency 维度
- 新增 D8: recency（时间临近性），指数衰减 exp(-lambda * (max_bar - bar))
- lambda = 3.0 / max_bar, 约5%权重在最远端、95%在最近端
- 权重重分配: 8维总和1.0, D1=0.17, D2=0.12, D3=0.18, D4=0.08, D5=0.13, D6=0.08, D7=0.09, D8=0.15

### 8.2 ✅ 点的重要性 — 百分比截断
- main()输出改为百分比截断，缺省前50%（27峰+27谷）
- 每个点显示实际importance值，不是排名序号

### 8.3 ✅ 线的重要性 — 端点重要性的乘积
- build_segment_pool() 和 pool_fusion() 中均改为 imp1 * imp2
- 两端都重要时乘积远大于只一端重要的情况

### 8.4 ✅ 对称结构识别（5维对称向量，已实现）
- `find_symmetric_structures()`: 扫描池中三波(A,B,C)组合
- 5维对称度: amp_sym, time_sym, mod_sym, slope_sym, complexity_sym
- 综合得分 = 对称度 × 端点重要性
- 搜索优化: max_pool_size=800 (取重要性Top800线段), 1.1秒
- 结果: 200个对称结构, 177个高对称度(>0.8)
- Top1: score=0.624, sym=0.957, bar7→74→98→166 (inv_V_top)
- 可视化: A段(彩色实线) + B段(灰色虚线) + C段(彩色虚线), 对称轴标记, Sym Top滑块, 悬停详情

### 8.5 ✅ 几何形态分类录入 GROUND_TRUTH.md
- 第十一节: 镜像轴对称(V/双顶/头肩), 中心对称(abc/12345), 三角形(收敛/扩张)
- 所有形态本质上可用3段或5段表达

## 9. v3.3 对称谱 (2026-02-26)

### 9.1 ✅ compute_symmetry_spectrum() — 统一对称谱框架
- 搜索两类对称对:
  - **Mirror**: 共享端点, 方向相反 (V/双顶/头肩)
  - **Center**: 通过中心段连接, 同向 (abc/12345)
- 输出完整ratio向量: amp_ratio, time_ratio, mod_ratio, slope_ratio, complexity_diff
- 结果: 28,862 spectra (mirror:5,029, center:23,833)

### 9.2 ✅ 对称现实检验
- 真实数据 vs 随机行走 vs 打乱收益率 → **对称计数无差异** (23-73%百分位)
- 结论: 对称计数是几何必然性, 不是edge。对称**模式**(ratio向量)才可能是特征

### 9.3 ✅ H1 vs M15 分布对比
- amp_ratio, time_ratio, mod_ratio分布跨TF一致 → 比例是无量纲的

## 10. 动态引擎 v1 (2026-02-26)

### 10.1 ✅ DynamicZG — 逐K线zigzag
- 前向推进, 无backward pass (那是前瞻)
- 每个拐点有 birth_bar (被哪根K线确认)
- 最后一个拐点是"临时的"(tentative), 方向反转时前一个变为confirmed
- 验证: 109个确认拐点, 101/109与静态版完全匹配, 8个偏移1-2 bars(预期行为)

### 10.2 ✅ DynamicMerger — 增量归并
- 新确认拐点加入工作序列后检查尾部4拐点
- 幅度归并优先, 横向补充, 连锁触发
- 每条新线段记录 birth_bar

### 10.3 ✅ DynamicPool — 增量pool_fusion
- 新线段加入后增量搜索三波组合 (只搜涉及新线段的)
- 验证: 2970线段与静态版一致 (2550共有, 420因拐点偏移不同)

### 10.4 性能发现
- **核心引擎(ZG+归并+fusion)**: 200 bars = 0.19s ✓
- **瓶颈**: pool_fusion随池增大而O(degree^3), 2000 bars超时
- **解决方案**: 全量预计算改用**滑动窗口**, 每次200 bars静态计算, 性能可控

### 10.5 动态引擎 vs 滑动窗口 的定位
- **动态引擎**: 适合实时场景, 200-500 bars内性能优秀
- **滑动窗口**: 适合历史回测/特征提取, 在固定大小池上计算, O(1)性能
- 两者不冲突, 是不同场景的工具

## 11. 全量预计算 (2026-02-26)

### 11.1 架构
- 滑动窗口: window=200, stride=50
- 每个窗口: 静态引擎 → pool_fusion → 对称谱 → 51维特征向量
- Z outcome: 4个前瞻距离(10/20/50/100 bars) × 5个指标 = 20维
- 特征维度: 基础统计(4) + ratio分布(25) + Top5谱(20) + 池状态(2) = 51维

### 11.2 性能
- 小测试: 2000 bars → 37窗口, 33s (~0.9 w/s) ✓
- 全量 EURUSD H1: 155K bars → 3101窗口, **预计~50min**
- **正在运行** (后台进程, PID记录在precompute_h1.log)

### 11.3 文件
- `dynamic_engine.py` — DynamicZG + DynamicMerger + DynamicPool + DynamicEngine
- `full_precompute.py` — 全量预计算脚本 (滑动窗口 + 特征提取 + Z outcome)
- `cluster_analysis.py` — faiss聚类 + Z分布检验 + shuffle显著性测试
- `precompute_test/precomputed.pkl` — 小规模测试结果
- `precompute_h1/precomputed.pkl` — 全量H1结果 (计算中)

## 12. 关键负面结果 (2026-02-26~27)

### 12.1 ❌ 51维聚类特征 — 无信号
- 滑动窗口(200bar, stride 50) → 对称谱 → 51维统计向量 → k-means聚类
- shuffle test: MI_real ≤ MI_shuffle → **特征对Z outcome无预测力**
- 结论: 对称谱的统计分布特征不是edge

### 12.2 ❌ 60M原始15D向量 k-NN — 无信号
- faiss IVF 向量存储 → k-NN (k=50/100/200) → 4个前瞻距离
- 所有准确率 ≈ 50% (pure random)
- 结论: 对称spectrum的统计模式匹配彻底失败

### 12.3 用户关键洞察 — 从统计到几何
- **统计路线已死**: 无论如何聚合/聚类/k-NN, 对称谱的统计模式没有预测力
- **路径一: 几何推演**: 每个对称结构是独立的case-by-case几何推理
  - 已知A + 对称中心 → 几何推演C'的目标
  - 这不是模式匹配, 是纯几何映射

## 13. 路径一: 对称映像预测 (2026-02-27) — 当前工作

### 13.1 ✅ predict_symmetric_image() 
- merge_engine_v3.py ~line 1187, ~290行
- **Mirror预测**: A到折返点P → C'从P反向, amp≈A, time≈A
- **Center预测**: A→B→ → C'从B_end, 同向A, amp≈A, time≈A
- 评分: importance × log_amp × recency × activity
  - log_amp: log(1 + rel_amp*10) / log(11) — 防大线段主导
  - recency: 1/(1 + dist/0.15*current_bar) — 时效性衰减

### 13.2 ✅ visualize_v4.py v4.2 — 三窗口动态可视化
- **架构**: DynamicZG + DynamicMerger 逐K线推进 (跳过pool_fusion, O(n^2)太慢)
- **数据**: 2000 K-lines, EURUSD H1 (2024-09-04 ~ 2024-12-31)
- **三窗口**: 
  - Amp: base+amp段 → 幅度归并级别预测
  - Lat: lat段 → 时间归并级别预测
  - All: 全部段 → 综合预测
- **交互**: 方向键单K线滚动, 空格播放, 速度滑块, bar slider
- **性能**: 2000 bars 计算 22s, HTML 3.2MB
- **增量编码**: 
  - Confirmed pivots: append-only list (20KB)
  - Tentative: incremental dict (40KB)
  - Predictions: incremental dict (3.1MB)
  - JS端 binary search 重建每帧拐点序列

### 13.3 关键数据发现
- base(964) + amp(236) + lat(245) = 1445 segments for 2000 bars
- 预测分布: Amp 5-9个, Lat 2-10个, All 13-15个 (每帧, max_preds=15)
- 预测主要来自大跨度线段 (span 500+), 小线段预测很快过期

### 13.4 证明死路的文件 (保留但不要使用)
- `build_vector_store.py` — 60M向量存储构建器
- `vector_query.py` — faiss k-NN评估
- `eval_fast.py` — 采样快速评估
- `full_precompute.py` — 51维聚合特征
- `cluster_analysis.py` — k-means + shuffle test
- `vecstore_h1/store.pkl` — 3.6GB向量存储

## 14. FSD统一框架 (2026-02-27) — 当前核心

### 14.1 设计哲学
- **FSD = Full Self-Driving**: 感知→预测(多轨迹)→规划→执行→反馈
- **三个用途共享一个引擎**:
  1. 人眼逐K线检查 (前端可视化)
  2. 后台大规模回测 (批量计算)
  3. ML训练 (上帝视角标注 → 学习 state→action)
- **策略是矩阵不是标量**: 方向、TP、SL、仓位各自是向量/矩阵
- **每根K线必须与所有预测比较**: 正向离差(小概率大机会) vs 负向离差(常态,核心算边界)

### 14.2 ✅ fsd_engine.py (~660行)
- `Trajectory`: 预测轨迹生命周期 (发出→追踪→过期/命中/淘汰)
  - 每根K线 update: progress, deviation, MFE, MAE
  - 淘汰阈值: sqrt(pred_time/10) 缩放, dev_limit=min(1.5*scale,5.0)
- `StateSnapshot`: 一根K线时刻的完整状态快照
  - ZG结构 + 所有活跃轨迹 + 共识 + best_deviation
  - `to_vector()` → 63维固定向量 (15 base + 8×6 traj)
- `FSDEngine.step(h,l,o,c) → StateSnapshot`
  - DynamicZG + DynamicMerger → 新拐点 → predict_symmetric_image → 新轨迹
  - 2000 bars ~30s, 保持 6-9 活跃轨迹
- `OracleLabeler.label_batch()`: 上帝视角标注
  - 方向(10/20/50), MFE双向, 最优TP方向, 最优R:R, 最优轨迹

### 14.3 ✅ visualize_v5.py (~520行) + fsd_v5.html (6.2MB)
- **单窗口全景**: K线 + ZG + 所有活跃轨迹 + 预测线 + 边界包络 + 离差标注
- **底部面板**: 每条轨迹的 type/dir/deviation/progress/MFE/MAE/score/TP 实时更新
- **状态栏**: 共识方向, 活跃轨迹数(多/空), best deviation, Oracle信息
- **Checkbox开关**: Oracle(上帝方向), Boundary(模长边界), Deviation(离差标注)
- **交互**: 方向键单K, 空格播放, 速度滑块, bar slider
- **颜色编码**: Mirror=金色, Center=青色; 多=绿, 空=红; 离差<0.3=绿, <1=橙, >1=红
- **增量编码优化**: PVT只存visible窗口拐点, 17.5MB→6.2MB

### 14.4 关键负面结果(预测评估, 已证明)
- **方向准确率整体: 52.7%** (几乎等于随机)
- **Center预测有信号** (z=15.32 reach100), Mirror无信号
- **小幅(<30 pips)**: 66%方向准, 但可能是trivial均值回归
- **大幅(>80 pips)**: 无信号
- **目标价被系统性回避** (z=-17.14), 比随机更差
- **根因**: 85.7%预测来自span 500+线段, horizon仅50 bars → 时间尺度错配
- **静态TP/SL策略**: z-scores从-40到-216 (massive systematic loss)
- **反向也失败**: 反向也是负z-score
- **用户洞察**: 问题不是预测本身, 是机械化静态评估。需要动态追踪 — 这就是FSD的动机

### 14.5 Oracle标注结果
- 最优轨迹准确度: mean 0.625 (culling后)
- 活跃轨迹数: mean ~7-9, max 30

## 15. 下一步方向（未完成）

### 15.1 ⬜ 视觉验证 (CRITICAL BLOCKER)
- 打开 fsd_v5.html 在浏览器中逐K线滚动
- 用户强调: **没有可视化验证，后面一切都是空中楼阁**
- 检查: 轨迹生成/消亡是否合理, 离差演化是否直观, 边界包络是否有意义
- 找到具体的case, 讨论"在这个时刻应该怎么做"

### 15.2 ⬜ 策略矩阵设计
- 方向: 多/空/观望 (不是0/1, 而是置信度连续值)
- TP: 基于最优轨迹的目标价 (动态调整)
- SL: 基于模长边界/MAE分布
- 仓位: 基于离差和共识强度

### 15.3 ⬜ ML Pipeline
- 输入: 63维状态向量 (StateSnapshot.to_vector())
- 标签: Oracle标注 (最优方向, 最优R:R, 最优轨迹)
- 目标: 学习 state→strategy 映射

### 15.4 ⬜ 多时间框架
- 在M15 / H4 上运行同样的引擎, 检查结果一致性

---

## 16. 与用户沟通的注意事项

- **中文沟通**，代码/数据可英文
- **不要美化结果**，用户要求brutal honesty和统计严格性
- **每步必须验证** — 4层验证框架
- **文件修改前git commit**
- 用户的表达方式有时引用道家哲学，但背后是严格的数学/工程思维
- **v3被视为商用底层基础架构** — 后续修改要慎重，保持向后兼容
- **用户强调动态 = 无前瞻**: 不只是不看未来价格, 而是归并过程本身也是逐步的
- **对称是多维连续特征向量**, 不是二元的。每个segment pair有"对称谱"
- **预测是个案几何推演**, 不是统计模式匹配

---

## 17. 快速验证命令

```bash
# 🔑 运行FSD v5可视化 (2000 bars, ~30s, 输出6.2MB HTML)
python3 -u /home/ubuntu/stage2_abc/visualize_v5.py

# 运行FSD引擎独立测试 (2000 bars, ~30s + oracle标注)
python3 -u /home/ubuntu/stage2_abc/fsd_engine.py

# 运行v4动态可视化 (2000 bars, ~22s, 输出3.2MB HTML)
python3 -u /home/ubuntu/stage2_abc/visualize_v4.py

# 运行v3引擎 (200根K线, 静态)
python3 /home/ubuntu/stage2_abc/merge_engine_v3.py

# 生成v3可视化 (静态)
python3 /home/ubuntu/stage2_abc/visualize_v3.py

# 动态引擎验证
python3 /home/ubuntu/stage2_abc/dynamic_engine.py

# 查看git状态
cd /home/ubuntu/stage2_abc && git log --oneline
```
