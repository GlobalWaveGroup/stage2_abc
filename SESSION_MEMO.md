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

### 3.1 merge_engine_v3.py (~1140行) — 归并引擎 (v3.2)

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
- **`find_symmetric_structures(pool, pivot_info, top_n, max_pool_size)`** — 🔑 **v3.2核心**: 5维对称向量识别
- `prune_redundant(pool)` — 冗余删除（**暂不执行**）

### 3.2 visualize_v3.py (~730行) — HTML可视化生成器 (v3.2)

- 快照层：base(金色) / amp(红色渐变) / lat(青色虚线)
- Extra层：滑窗收集的额外线段（绿色/橙色虚线）
- **Fusion层**：池内三波归并产出的新线段（**紫色系，虚线**）
- **Symmetry层**：对称结构三波(A绿/灰B虚线/C绿虚线) + 对称轴标记 + score标签
- Fusion Top N 滑块、Sym Top 滑块、Min imp 滑块
- 鼠标悬停: fusion线段详情、对称结构5维向量详情
- 可调Peak/Valley显示数量滑块 + 重要性值标注
- Step+/- 逐步回放
- 输出: merge_v3.html (509KB)

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

## 12. 下一步方向（未完成）

### 12.1 全量聚类分析
- 等全量预计算完成后运行 cluster_analysis.py
- 核心问题: 对称谱特征聚类后, Z分布是否集中?
- 如果MI显著 > shuffle → 对称谱有预测力 → 进入ML阶段
- 如果MI不显著 → 对称谱alone不够, 需要更多特征(动态birth_bar分布等)

### 12.2 Cross-TF验证
- H1建索引 → M15查询, 或反向
- 验证对称谱特征是否跨TF泛化

### 12.3 动态引擎增强 (如果聚类有信号)
- birth_bar分布作为额外特征维度
- 多线段共振信号 (多个对称谱同时birth → 转折共振)
- T/F/S线集成

### 12.4 时-空转换动态常数
- 模长对称中的核心未解问题
- 当前用经验归一化, 需理论推导

---

## 13. 与用户沟通的注意事项

- **中文沟通**，代码/数据可英文
- **不要美化结果**，用户要求brutal honesty和统计严格性
- **每步必须验证** — 4层验证框架
- **文件修改前git commit**
- 用户的表达方式有时引用道家哲学，但背后是严格的数学/工程思维
- **v3被视为商用基础架构** — 后续修改要慎重，保持向后兼容
- **用户强调动态 = 无前瞻**: 不只是不看未来价格, 而是归并过程本身也是逐步的

---

## 14. 快速验证命令

```bash
# 运行v3引擎 (200根K线)
python3 /home/ubuntu/stage2_abc/merge_engine_v3.py

# 生成v3可视化
python3 /home/ubuntu/stage2_abc/visualize_v3.py

# 动态引擎验证
python3 /home/ubuntu/stage2_abc/dynamic_engine.py

# 全量预计算 (H1, ~50min)
python3 -u /home/ubuntu/stage2_abc/full_precompute.py full

# 聚类分析 (需要预计算数据)
python3 /home/ubuntu/stage2_abc/cluster_analysis.py

# 查看git状态
cd /home/ubuntu/stage2_abc && git log --oneline
```
