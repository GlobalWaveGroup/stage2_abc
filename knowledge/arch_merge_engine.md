# merge_engine_v3.py 完整架构

路径: `/home/ubuntu/stage2_abc/merge_engine_v3.py` (~1755行)
状态: **商用底层基础架构** (用户确认)

## 核心Pipeline

```
load_kline(path, limit) → DataFrame
    ↓
calculate_base_zg(high, low, rb=0.5) → [(bar, price, dir), ...]
    ↓
full_merge_engine(pivots, max_iterations=200) → results dict
    ↓
compute_pivot_importance(results, total_bars) → pivot_info dict
    ↓
build_segment_pool(results, pivot_info) → pool list
    ↓
pool_fusion(pool, pivot_info, max_rounds=100) → (full_pool, new_segs, log)
    ↓
find_symmetric_structures(pool, pivot_info, top_n=100, max_pool_size=800) → sym list
    ↓
predict_symmetric_image(pool, pivot_info, current_bar, max_pool_size=500) → pred list
```

## 函数签名和返回值

### load_kline(filepath, limit=None) → DataFrame
- 读TSV (tab分隔, skiprows=1)
- 列: date,time → 合成datetime | open,high,low,close 保留
- limit: 取最后N行

### calculate_base_zg(high, low, rb=0.5) → list[(bar,price,dir)]
- KZig风格基础zigzag, rb=反弹灵敏度
- dir: 1=峰, -1=谷
- 保证交替: 峰谷交替出现

### full_merge_engine(pivots) → dict
- 幅度+横向交替迭代至不动点
- 返回:
  - all_snapshots: [(type, label, pivots_list), ...] — type='base'/'amp'/'lat', label='L0'/'A1'/'T1'...
  - extra_segments: [(p_start, p_end, label), ...] — 被贪心跳过但滑窗收集的线段
  - final_pivots: 最终收敛的拐点序列
  - total_iterations, total_amp_levels, total_lat_batches

### compute_pivot_importance(results, total_bars) → dict[bar → info]
- 8维重要性: d1线条数(.17) d2存活(.12) d3幅度(.18) d4时间(.08) d5极端(.13) d6孤立(.08) d7主导(.09) d8临近(.15)
- 每个拐点: bar, price, dir, d1-d8各维度, importance(加权总分0~1)

### build_segment_pool(results, pivot_info) → list[seg_dict]
- 从快照+extra构建去重线段池
- 每条线段: bar_start/end, price_start/end, dir_start/end, span, amplitude, source, source_label, imp_start, imp_end, importance=imp_start*imp_end
- 按importance降序排列

### pool_fusion(pool, pivot_info) → (full_pool, all_new, fusion_log)
- 池内三波归并(无条件), 迭代至不动点
- 新线段有额外字段: fusion_via=(p2_bar, p3_bar), fusion_amps=(amp_A, amp_B, amp_C)
- 典型: 301 → 2970条, 3轮不动点, 0.166秒 (200 bars)

### find_symmetric_structures(pool, pivot_info, top_n, max_pool_size) → list[dict]
- 搜索三波A,B,C对称结构
- 5维对称度: amp(.25) time(.20) mod(.25) slope(.15) complexity(.15)
- score = sym_score × endpoint_imp
- 返回: p1-p4坐标, 5维向量, 方向, 类型(V_bottom/inv_V_top/descending/ascending)

### predict_symmetric_image(pool, pivot_info, current_bar) → list[dict]
- 4种预测: mirror(轴对称), center(中心对称/abc), triangle(三角形), modonly(模长守恒)
- 评分: importance × log_amp × recency × activity
- 返回: A段/B段坐标, 预测目标价/bar, 进度, score

## 归并规则

### 幅度归并
- 四拐点(P1,P2,P3,P4): P1和P4分别是极值对 → 合并中间两点
- 贪心推进+滑窗收集

### 横向归并(幅度无变化后)
- 三段(a,b,c)分4类:
  1. 收敛 |a|>|b|>|c| → 连
  2. 扩张 |a|<|b|<|c| → 连
  3. b最长有交错 → 连
  4. b最长无交错 → 不连

### Pool Fusion
- 任何三条首尾相连的线段 → 无条件产出新线段
- 不需要幅度/横向条件
- 消融级别界限: 线段就是线段，不分级别
