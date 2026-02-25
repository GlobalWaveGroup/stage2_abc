#!/usr/bin/env python3
"""
完整归并引擎 v3.0 — 消融级别界限

架构演进:
  v2: 逐级拐点序列归并（幅度+横向交替）→ 初始波段池(301条)
  v3: v2产出初始池 → pool_fusion()无条件三波归并 → 完整波段池(2970条)
      线段就是线段，不分级别、不分来源。
      任何三条首尾相连的线段 → 无条件产出新线段。
      循环至不动点。

商用基础架构:
  - 静态完整波段池（所有级别线段平等共存）
  - 多维拐点重要性评分（7+维度）
  - 线段重要性 = 端点重要性乘积
  - 为后续提供：数量统计、特征统计、向量统计、对称结构识别、动态K线生命周期
"""

import numpy as np
import pandas as pd
import sys
import time

# =============================================================================
# 1. 数据加载
# =============================================================================

def load_kline(filepath, limit=None):
    df = pd.read_csv(filepath, sep='\t',
                     names=['date','time','open','high','low','close','tickvol','vol','spread'],
                     skiprows=1)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df[['datetime','open','high','low','close']].reset_index(drop=True)
    if limit:
        df = df.tail(limit).reset_index(drop=True)
    return df

# =============================================================================
# 2. 基础ZG计算 (KZig风格)
# =============================================================================

def calculate_base_zg(high, low, rb=0.5):
    n = len(high)
    if n < 3:
        return []

    zg = [None] * n
    last_state = 1
    last_pos = 0
    p_pos = 0
    last_range = 9999.0

    for i in range(1, n):
        is_up = False
        is_down = False

        if ((high[i] > high[i-1] and low[i] >= low[i-1]) or
            (last_state == -1 and high[i] - low[last_pos] > last_range * rb and
             not (high[i] < high[i-1] and low[i] < low[i-1])) or
            (last_state == -1 and i - last_pos > 1 and high[i] > low[last_pos] and
             not (high[i] < high[i-1] and low[i] < low[i-1]))):
            is_up = True

        if ((high[i] <= high[i-1] and low[i] < low[i-1]) or
            (last_state == 1 and high[last_pos] - low[i] > last_range * rb and
             not (high[i] > high[i-1] and low[i] > low[i-1])) or
            (last_state == 1 and i - last_pos > 1 and low[i] < high[last_pos] and
             not (high[i] > high[i-1] and low[i] > low[i-1]))):
            is_down = True

        if is_up:
            if last_state == 1:
                if high[i] < high[last_pos]:
                    continue
                zg[last_pos] = None
            else:
                p_pos = last_pos
            zg[i] = high[i]
            last_pos = i
            last_state = 1
            last_range = high[i] - low[p_pos]
        elif is_down:
            if last_state == -1:
                if low[i] > low[last_pos]:
                    continue
                zg[last_pos] = None
            else:
                p_pos = last_pos
            zg[i] = low[i]
            last_pos = i
            last_state = -1
            last_range = high[p_pos] - low[i]

    pre_price = zg[last_pos]
    pre_pos = last_pos
    fix_state = last_state

    for i in range(last_pos - 1, -1, -1):
        if zg[i] is not None:
            if fix_state == -1:
                fix_state = 1
                if zg[i] < pre_price:
                    zg[i] = None
                    zg[pre_pos] = pre_price
                else:
                    pre_price = zg[i]
                    pre_pos = i
            elif fix_state == 1:
                fix_state = -1
                if zg[i] > pre_price:
                    zg[i] = None
                    zg[pre_pos] = pre_price
                else:
                    pre_price = zg[i]
                    pre_pos = i
        else:
            if fix_state == 1 and low[i] < pre_price:
                pre_price = low[i]
                pre_pos = i
            elif fix_state == -1 and high[i] > pre_price:
                pre_price = high[i]
                pre_pos = i

    pivots = []
    for i in range(n):
        if zg[i] is not None:
            if abs(zg[i] - high[i]) < 1e-10:
                pivots.append((i, zg[i], 1))
            elif abs(zg[i] - low[i]) < 1e-10:
                pivots.append((i, zg[i], -1))

    cleaned = []
    for p in pivots:
        if not cleaned or cleaned[-1][2] != p[2]:
            cleaned.append(p)
        else:
            if p[2] == 1 and p[1] > cleaned[-1][1]:
                cleaned[-1] = p
            elif p[2] == -1 and p[1] < cleaned[-1][1]:
                cleaned[-1] = p

    return cleaned

# =============================================================================
# 3. 单轮幅度归并 — 输入一个级别的拐点，输出下一级的拐点
# =============================================================================

def _check_amp_merge(p1, p2, p3, p4):
    """检查4拐点是否满足幅度归并条件：P1和P4分别为极值"""
    prices = [p1[1], p2[1], p3[1], p4[1]]
    max_idx = prices.index(max(prices))
    min_idx = prices.index(min(prices))
    return (max_idx == 0 and min_idx == 3) or (max_idx == 3 and min_idx == 0)


def amplitude_merge_one_pass(pivots):
    """
    对拐点序列做一轮幅度归并。
    
    双重职责：
    1. 滑窗收集：逐个检查所有相邻4拐点组，所有满足条件的都记录新线段
    2. 贪心推进：产生下一级拐点序列
    
    返回: (新拐点列表, 是否有变化, 滑窗发现的所有新线段)
    """
    if len(pivots) < 4:
        return pivots, False, []

    # === 滑窗收集：所有满足条件的三波 ===
    all_found = []
    for i in range(len(pivots) - 3):
        p1, p2, p3, p4 = pivots[i], pivots[i+1], pivots[i+2], pivots[i+3]
        if _check_amp_merge(p1, p2, p3, p4):
            all_found.append((p1, p4))  # 记录首尾拐点对

    # === 贪心推进 ===
    result = []
    changed = False
    i = 0
    while i < len(pivots):
        if i + 3 < len(pivots):
            p1, p2, p3, p4 = pivots[i], pivots[i+1], pivots[i+2], pivots[i+3]
            if _check_amp_merge(p1, p2, p3, p4):
                result.append(p1)
                i += 3
                changed = True
                continue
        result.append(pivots[i])
        i += 1

    return result, changed, all_found

# =============================================================================
# 4. 横向归并分类与单轮执行
# =============================================================================

def classify_three_segments(pivots, i):
    """分类相邻三段的几何结构"""
    if i + 3 >= len(pivots):
        return None, {}

    p1_idx, p1_price, p1_dir = pivots[i]
    p2_idx, p2_price, p2_dir = pivots[i+1]
    p3_idx, p3_price, p3_dir = pivots[i+2]
    p4_idx, p4_price, p4_dir = pivots[i+3]

    amp_a = abs(p2_price - p1_price)
    amp_b = abs(p3_price - p2_price)
    amp_c = abs(p4_price - p3_price)

    if amp_a > amp_b and amp_b > amp_c:
        return 'converging', {}
    if amp_a < amp_b and amp_b < amp_c:
        return 'expanding', {}

    if amp_a <= amp_b and amp_b >= amp_c:
        time_span = p4_idx - p1_idx
        if time_span == 0:
            return 'no_crossover', {}
        price_diff = p4_price - p1_price
        slope = price_diff / time_span
        if p1_dir == 1:
            return ('crossover' if slope <= 0 else 'no_crossover'), {}
        else:
            return ('crossover' if slope >= 0 else 'no_crossover'), {}

    # Fallback: check crossover for remaining cases
    time_span = p4_idx - p1_idx
    if time_span == 0:
        return 'no_crossover', {}
    price_diff = p4_price - p1_price
    slope = price_diff / time_span
    if p1_dir == 1:
        return ('crossover' if slope <= 0 else 'no_crossover'), {}
    else:
        return ('crossover' if slope >= 0 else 'no_crossover'), {}


def _check_lat_merge(pivots, i):
    """检查位置i开始的4拐点是否满足横向归并条件"""
    seg_type, _ = classify_three_segments(pivots, i)
    return seg_type in ('converging', 'expanding', 'crossover')


def lateral_merge_one_pass(pivots):
    """
    单轮横向归并。
    
    双重职责：
    1. 滑窗收集：所有满足条件的横向三波
    2. 贪心推进：产生下一级拐点序列
    
    返回: (新拐点列表, 是否有变化, 滑窗发现的所有新线段)
    """
    if len(pivots) < 4:
        return pivots, False, []

    # === 滑窗收集 ===
    all_found = []
    for i in range(len(pivots) - 3):
        if _check_lat_merge(pivots, i):
            all_found.append((pivots[i], pivots[i+3]))

    # === 贪心推进 ===
    result = []
    changed = False
    i = 0
    while i < len(pivots):
        if i + 3 < len(pivots):
            if _check_lat_merge(pivots, i):
                result.append(pivots[i])
                i += 3
                changed = True
                continue
        result.append(pivots[i])
        i += 1

    return result, changed, all_found

# =============================================================================
# 5. 完整归并引擎 v3 — 所有级别共存 + 池内融合
# =============================================================================

def full_merge_engine(pivots, max_iterations=200):
    """
    完整归并引擎 — 交替迭代至不动点。
    
    道生一，一生二，二生三，三生万物。
    任何三波结构，只要满足归并条件（幅度增加或时间增加），就必须归并。
    
    核心原则：
    1. 幅度优先：有幅度归并机会就先做
    2. 幅度无机会 → 横向归并一轮 → 回馈 → 幅度再检查
    3. 滑窗收集：每轮扫描中，所有满足条件的三波都产出线段进波段池，
       不因贪心跳跃而遗漏中间级别的结构
    4. 交替至不动点
    """
    log = []
    all_snapshots = []       # (type, label, pivots) — 每个状态快照
    extra_segments = []      # 滑窗收集的被贪心跳过的额外线段 [(p_start, p_end, source_label)]
    
    current = pivots
    all_snapshots.append(('base', 'L0', list(current)))
    
    global_iter = 0
    amp_count = 0
    lat_count = 0
    
    while global_iter < max_iterations:
        global_iter += 1
        
        # === 幅度归并一轮（滑窗收集 + 贪心推进）===
        new_pivots, amp_changed, amp_found = amplitude_merge_one_pass(current)
        if amp_changed:
            amp_count += 1
            label = f'A{amp_count}'
            
            # 滑窗额外发现的幅度线段
            greedy_set = set()
            for j in range(len(new_pivots) - 1):
                greedy_set.add((new_pivots[j][0], new_pivots[j+1][0]))
            for p_start, p_end in amp_found:
                key = (p_start[0], p_end[0])
                if key not in greedy_set:
                    extra_segments.append((p_start, p_end, label))
            
            # 在当前（归并前）序列上也做横向归并滑窗收集
            # 不改变主链推进，只收集中间级别的横向线段
            _, _, lat_found_mid = lateral_merge_one_pass(current)
            n_lat_mid = 0
            for p_start, p_end in lat_found_mid:
                extra_segments.append((p_start, p_end, f'{label}_mid'))
                n_lat_mid += 1
            
            log.append(f"I{global_iter} {label}: {len(current)}->{len(new_pivots)} "
                       f"(amp滑窗{len(amp_found)}, lat中间{n_lat_mid})")
            all_snapshots.append(('amp', label, list(new_pivots)))
            current = new_pivots
            continue
        
        # === 幅度无变化 → 横向归并一轮（滑窗收集 + 贪心推进）===
        new_pivots, lat_changed, lat_found = lateral_merge_one_pass(current)
        if lat_changed:
            lat_count += 1
            label = f'T{lat_count}'
            log.append(f"I{global_iter} {label}: {len(current)}->{len(new_pivots)} (滑窗发现{len(lat_found)}对)")
            all_snapshots.append(('lat', label, list(new_pivots)))
            
            greedy_set = set()
            for j in range(len(new_pivots) - 1):
                greedy_set.add((new_pivots[j][0], new_pivots[j+1][0]))
            for p_start, p_end in lat_found:
                key = (p_start[0], p_end[0])
                if key not in greedy_set:
                    extra_segments.append((p_start, p_end, label))
            
            current = new_pivots
            continue
        
        # === 不动点 ===
        log.append(f"不动点: {global_iter}轮, {len(current)}拐点, A={amp_count}, T={lat_count}, extra={len(extra_segments)}")
        break
    
    return {
        'all_snapshots': all_snapshots,
        'extra_segments': extra_segments,
        'log': log,
        'final_pivots': current,
        'total_iterations': global_iter,
        'total_amp_levels': amp_count,
        'total_lat_batches': lat_count,
    }

# =============================================================================
# 6. 点重要性 → 线段重要性 → 波段池
# =============================================================================

def compute_pivot_importance(results, high=None, low=None, total_bars=None):
    """
    多维度拐点重要性评分。
    
    维度：
    D1. line_count — 经过该点的线段数量（快照+extra中作为端点的次数）
    D2. snapshot_survival — 出现在多少个快照中（存活轮数，越久越重要）
    D3. amplitude — 该点参与的最大幅度线段的幅度（价格影响力）
    D4. time_span — 该点参与的最长线段的时间跨度（时间影响力）
    D5. extremity — 该点价格在同方向(峰/谷)中的极端程度（接近全局极值越重要）
    D6. isolation — 与相邻拐点的时间距离（越孤立越是结构性转折）
    D7. price_range_dominance — 该点到最近反向拐点的价格差占全局价格范围的比例
    D8. recency — 离最后一根K线越近越重要（指数衰减）
    
    各维度归一化到0~1后加权求和。
    """
    from collections import Counter
    
    snapshots = results['all_snapshots']
    extra_segments = results.get('extra_segments', [])
    base = snapshots[0][2]  # 基础ZG拐点序列
    
    if len(base) < 2:
        return {}
    
    # 全局统计
    all_prices = [p[1] for p in base]
    global_high = max(all_prices)
    global_low = min(all_prices)
    global_range = global_high - global_low
    if global_range == 0:
        global_range = 1e-10
    
    peak_prices = [p[1] for p in base if p[2] == 1]
    valley_prices = [p[1] for p in base if p[2] == -1]
    
    # --- D1: line_count (经过该点的线段数量) ---
    line_count = Counter()
    for snap_type, label, pvts in snapshots:
        for j in range(len(pvts) - 1):
            line_count[pvts[j][0]] += 1
            line_count[pvts[j+1][0]] += 1
    for p_start, p_end, label in extra_segments:
        line_count[p_start[0]] += 1
        line_count[p_end[0]] += 1
    
    # --- D2: snapshot_survival ---
    snapshot_count = Counter()
    for snap_type, label, pvts in snapshots:
        seen = set(p[0] for p in pvts)
        for bar in seen:
            snapshot_count[bar] += 1
    
    # --- D3: amplitude (参与的最大幅度线段) ---
    max_amplitude = {}
    for snap_type, label, pvts in snapshots:
        for j in range(len(pvts) - 1):
            amp = abs(pvts[j+1][1] - pvts[j][1])
            for bar in [pvts[j][0], pvts[j+1][0]]:
                max_amplitude[bar] = max(max_amplitude.get(bar, 0), amp)
    for p_start, p_end, label in extra_segments:
        amp = abs(p_end[1] - p_start[1])
        for bar in [p_start[0], p_end[0]]:
            max_amplitude[bar] = max(max_amplitude.get(bar, 0), amp)
    
    # --- D4: time_span (参与的最长线段时间跨度) ---
    max_span = {}
    for snap_type, label, pvts in snapshots:
        for j in range(len(pvts) - 1):
            span = pvts[j+1][0] - pvts[j][0]
            for bar in [pvts[j][0], pvts[j+1][0]]:
                max_span[bar] = max(max_span.get(bar, 0), span)
    for p_start, p_end, label in extra_segments:
        span = p_end[0] - p_start[0]
        for bar in [p_start[0], p_end[0]]:
            max_span[bar] = max(max_span.get(bar, 0), span)
    
    # --- D6: isolation (与相邻拐点的平均时间距离) ---
    isolation = {}
    for idx, p in enumerate(base):
        bar = p[0]
        dists = []
        if idx > 0:
            dists.append(bar - base[idx-1][0])
        if idx < len(base) - 1:
            dists.append(base[idx+1][0] - bar)
        isolation[bar] = sum(dists) / len(dists) if dists else 0
    
    # --- D8: recency (离最后一根K线越近越重要, 指数衰减) ---
    max_bar = max(p[0] for p in base) if base else 0
    if total_bars and total_bars > max_bar:
        max_bar = total_bars - 1  # 使用实际K线数量
    recency = {}
    decay_lambda = 3.0 / max(max_bar, 1)  # ~5%权重在最远端, ~95%在最近端
    for p in base:
        bar = p[0]
        recency[bar] = np.exp(-decay_lambda * (max_bar - bar))
    
    # --- 归一化辅助 ---
    def _norm(d, keys):
        vals = [d.get(k, 0) for k in keys]
        mx = max(vals) if vals else 1
        if mx == 0:
            mx = 1
        return {k: d.get(k, 0) / mx for k in keys}
    
    bars = [p[0] for p in base]
    n_lc = _norm(line_count, bars)
    n_sc = _norm(snapshot_count, bars)
    n_amp = _norm(max_amplitude, bars)
    n_span = _norm(max_span, bars)
    n_iso = _norm(isolation, bars)
    n_rec = _norm(recency, bars)
    
    # --- 构建拐点信息 ---
    pivot_info = {}
    for p in base:
        bar, price, direction = p
        
        # D5: extremity — 峰在所有峰中的排名, 谷在所有谷中的排名
        if direction == 1 and peak_prices:
            extremity = (price - min(peak_prices)) / (max(peak_prices) - min(peak_prices)) if max(peak_prices) != min(peak_prices) else 0.5
        elif direction == -1 and valley_prices:
            extremity = (max(valley_prices) - price) / (max(valley_prices) - min(valley_prices)) if max(valley_prices) != min(valley_prices) else 0.5
        else:
            extremity = 0.5
        
        # D7: price_range_dominance — 与最近反向拐点的价差占全局range
        idx_in_base = bars.index(bar)
        max_local_range = 0
        for offset in [-1, 1]:
            ni = idx_in_base + offset
            if 0 <= ni < len(base):
                max_local_range = max(max_local_range, abs(price - base[ni][1]))
        dominance = max_local_range / global_range
        
        # 加权综合 (权重可调, 8维, 总和=1.0)
        w = {
            'line_count': 0.17,
            'survival': 0.12,
            'amplitude': 0.18,
            'time_span': 0.08,
            'extremity': 0.13,
            'isolation': 0.08,
            'dominance': 0.09,
            'recency': 0.15,   # D8: 时间临近性
        }
        
        importance = (
            w['line_count'] * n_lc.get(bar, 0) +
            w['survival'] * n_sc.get(bar, 0) +
            w['amplitude'] * n_amp.get(bar, 0) +
            w['time_span'] * n_span.get(bar, 0) +
            w['extremity'] * extremity +
            w['isolation'] * n_iso.get(bar, 0) +
            w['dominance'] * dominance +
            w['recency'] * n_rec.get(bar, 0)
        )
        
        pivot_info[bar] = {
            'bar': bar, 'price': price, 'dir': direction,
            'd1_line_count': line_count.get(bar, 0),
            'd2_survival': snapshot_count.get(bar, 0),
            'd3_amplitude': round(max_amplitude.get(bar, 0), 5),
            'd4_time_span': max_span.get(bar, 0),
            'd5_extremity': round(extremity, 3),
            'd6_isolation': round(isolation.get(bar, 0), 1),
            'd7_dominance': round(dominance, 3),
            'd8_recency': round(recency.get(bar, 0), 4),
            'importance': round(importance, 4),
        }
    
    return pivot_info


def build_segment_pool(results, pivot_info):
    """
    构建去重波段池。
    
    来源：
    1. 所有快照中相邻拐点构成的线段
    2. 滑窗收集的额外线段（被贪心跳过但满足归并条件的三波）
    
    自然去重（同一对拐点只保留一次）。
    线段重要性 = 两端点重要性的乘积（联合评估）。
    """
    snapshots = results['all_snapshots']
    extra_segments = results.get('extra_segments', [])
    seg_dict = {}
    
    def _add_seg(p1, p2, source, label):
        key = (p1[0], p2[0])
        if key not in seg_dict:
            info1 = pivot_info.get(p1[0], {})
            info2 = pivot_info.get(p2[0], {})
            imp1 = info1.get('importance', 0)
            imp2 = info2.get('importance', 0)
            seg_dict[key] = {
                'bar_start': p1[0], 'price_start': p1[1], 'dir_start': p1[2],
                'bar_end': p2[0], 'price_end': p2[1], 'dir_end': p2[2],
                'span': p2[0] - p1[0],
                'amplitude': abs(p2[1] - p1[1]),
                'source': source,
                'source_label': label,
                'imp_start': imp1, 'imp_end': imp2,
                'importance': imp1 * imp2,  # 乘积: 联合评估，两端都重要时远大于只一端重要
            }
    
    # 快照中的线段
    for snap_type, label, pvts in snapshots:
        for j in range(len(pvts) - 1):
            _add_seg(pvts[j], pvts[j+1], snap_type, label)
    
    # 滑窗额外线段
    for p_start, p_end, label in extra_segments:
        src = 'amp_extra' if label.startswith('A') else 'lat_extra'
        _add_seg(p_start, p_end, src, label)
    
    pool = sorted(seg_dict.values(), key=lambda s: -s['importance'])
    return pool


def pool_fusion(pool, pivot_info, max_rounds=100):
    """
    波段池内三波归并 — 消融级别界限。
    
    核心原则：
    - 线段就是线段，不分级别、不分来源
    - 任何三条首尾相连的线段构成三波 → 无条件产出新线段
    - 新线段再参与下一轮
    - 循环直到不动点（无新线段产出）
    
    算法：
    1. 用端点建邻接索引：end_index[bar] = 所有以bar为终点的线段
                          start_index[bar] = 所有以bar为起点的线段
    2. 对每个枢轴点bar，找所有 (seg_in → bar → seg_out) 组合
       即 seg_in.end == bar == seg_out.start，且 seg_in 和 seg_out 方向相反
    3. 对每对 (seg_in, seg_out)，在 start_index[seg_out.end] 中找第三段 seg3
       使得 seg_out.end == seg3.start
    4. 产出新线段: seg_in.start → seg3.end
    5. 去重后加入池，重建索引，下一轮
    
    返回: (扩展后的池, 新增线段列表, 日志)
    """
    from collections import defaultdict
    
    # 用 (bar_start, bar_end) 作为唯一键
    seg_set = {}
    for s in pool:
        key = (s['bar_start'], s['bar_end'])
        if key not in seg_set:
            seg_set[key] = s
    
    fusion_log = []
    all_new = []  # 所有新产出的线段
    
    for round_num in range(1, max_rounds + 1):
        # 建索引
        end_at = defaultdict(list)    # bar → [seg that ends here]
        start_at = defaultdict(list)  # bar → [seg that starts here]
        for key, s in seg_set.items():
            end_at[s['bar_end']].append(s)
            start_at[s['bar_start']].append(s)
        
        # 搜索所有三波组合
        new_in_round = []
        
        # 对每个可能的中间枢轴点 (三波的第2和第3个端点)
        # 三波: seg_A(p1→p2) + seg_B(p2→p3) + seg_C(p3→p4)
        # 需要: seg_A.end==p2, seg_B.start==p2, seg_B.end==p3, seg_C.start==p3
        # 产出: p1→p4
        
        for p2 in list(end_at.keys()):
            segs_ending_at_p2 = end_at[p2]
            segs_starting_at_p2 = start_at.get(p2, [])
            
            if not segs_starting_at_p2:
                continue
            
            for seg_A in segs_ending_at_p2:
                for seg_B in segs_starting_at_p2:
                    # seg_A: p1→p2, seg_B: p2→p3
                    p3 = seg_B['bar_end']
                    segs_starting_at_p3 = start_at.get(p3, [])
                    
                    if not segs_starting_at_p3:
                        continue
                    
                    for seg_C in segs_starting_at_p3:
                        # seg_C: p3→p4
                        p1 = seg_A['bar_start']
                        p4 = seg_C['bar_end']
                        
                        # 基本有效性: p1 < p2 < p3 < p4 (时间递增)
                        if not (p1 < p2 and p2 < p3 and p3 < p4):
                            continue
                        
                        # 去重: 如果已存在则跳过
                        new_key = (p1, p4)
                        if new_key in seg_set:
                            continue
                        
                        # 产出新线段
                        info1 = pivot_info.get(p1, {})
                        info4 = pivot_info.get(p4, {})
                        imp1 = info1.get('importance', 0)
                        imp4 = info4.get('importance', 0)
                        
                        new_seg = {
                            'bar_start': p1,
                            'price_start': seg_A['price_start'],
                            'dir_start': seg_A['dir_start'],
                            'bar_end': p4,
                            'price_end': seg_C['price_end'],
                            'dir_end': seg_C['dir_end'],
                            'span': p4 - p1,
                            'amplitude': abs(seg_C['price_end'] - seg_A['price_start']),
                            'source': 'fusion',
                            'source_label': f'F{round_num}',
                            'imp_start': imp1,
                            'imp_end': imp4,
                            'importance': imp1 * imp4,  # 乘积: 联合评估
                            # 元数据：三波组成信息（留给冗余删除用）
                            'fusion_via': (seg_A['bar_end'], seg_B['bar_end']),
                            'fusion_amps': (seg_A['amplitude'], seg_B['amplitude'], seg_C['amplitude']),
                        }
                        
                        seg_set[new_key] = new_seg
                        new_in_round.append(new_seg)
        
        if not new_in_round:
            fusion_log.append(f"F{round_num}: 不动点, 池={len(seg_set)}")
            break
        
        all_new.extend(new_in_round)
        fusion_log.append(f"F{round_num}: +{len(new_in_round)} 新线段, 池={len(seg_set)}")
    
    # 转回列表, 按重要性排序
    full_pool = sorted(seg_set.values(), key=lambda s: -s['importance'])
    
    return full_pool, all_new, fusion_log


# =============================================================================
# 6b. 对称结构识别 — 5维对称向量
# =============================================================================

def find_symmetric_structures(pool, pivot_info, df=None, top_n=100, max_pool_size=800):
    """
    在波段池中扫描所有三波(A,B,C)组合，计算5维对称向量。
    
    三波结构: seg_A → seg_B → seg_C, 首尾相连
    - seg_A.end == seg_B.start
    - seg_B.end == seg_C.start
    - A和C方向相同（都是上升或都是下降），B方向相反（中间段）
    
    5维对称度:
    1. amp_sym  — 幅度对称: 1 - |amp_A - amp_C| / max(amp_A, amp_C)
    2. time_sym — 时间对称: 1 - |time_A - time_C| / max(time_A, time_C)
    3. mod_sym  — 模长对称: 1 - |mod_A - mod_C| / max(mod_A, mod_C)
       mod = sqrt(norm_amp² + norm_time²), 归一化后计算
       注: 时-空转换常数暂用经验值, 这是核心待解问题
    4. slope_sym — 斜率对称: 1 - |slope_A + slope_C| / max(|slope_A|, |slope_C|)
       A和C方向相同 → slope_A和slope_C符号相同 → 镜像是 slope_A ≈ slope_C
       (对于中心对称结构, A的下降对应C的下降)
    5. complexity_sym — 内部结构对称 (当前简化版: 用子波段数近似)
    
    综合对称度 = 5维的加权平均
    
    max_pool_size: 参与搜索的最大线段数（按重要性截取），控制搜索空间
    
    返回: [{score, sym_score, endpoint_imp, vec, p1-p4, ...}, ...] 按综合对称度降序
    """
    from collections import defaultdict
    import math
    
    # 限制搜索空间: 取重要性最高的max_pool_size条线段
    search_pool = pool
    if len(pool) > max_pool_size:
        search_pool = sorted(pool, key=lambda s: -s['importance'])[:max_pool_size]
    
    # 建索引: end_bar → [seg], start_bar → [seg]
    end_at = defaultdict(list)
    start_at = defaultdict(list)
    for s in search_pool:
        end_at[s['bar_end']].append(s)
        start_at[s['bar_start']].append(s)
    
    # 统计全局参数用于归一化 (使用完整池的统计, 不受截取影响)
    all_amps = [s['amplitude'] for s in pool if s['amplitude'] > 0]
    all_spans = [s['span'] for s in pool if s['span'] > 0]
    if not all_amps or not all_spans:
        return []
    
    global_amp = max(all_amps)
    global_span = max(all_spans)
    
    # 时-空转换经验常数: 使幅度和时间在模长计算中权重相当
    # 以全局幅度range / 全局时间range 作为缩放因子
    space_time_ratio = global_amp / max(global_span, 1)
    
    # 子波段数统计（用于complexity_sym）: 用排序+二分实现O(logN)查询
    base_bars_sorted = sorted(p['bar'] for p in pivot_info.values())
    import bisect
    
    def _count_sub_segments(bar_start, bar_end):
        """计算时间区间内的base层拐点数（近似内部结构复杂度）, O(logN)"""
        left = bisect.bisect_right(base_bars_sorted, bar_start)
        right = bisect.bisect_left(base_bars_sorted, bar_end)
        return right - left
    
    def _compute_modulus(amp, span):
        """计算归一化模长"""
        norm_amp = amp / max(global_amp, 1e-10)
        norm_time = span / max(global_span, 1)
        return math.sqrt(norm_amp**2 + norm_time**2)
    
    def _sym_ratio(a, b):
        """对称度: 1 - |a-b|/max(a,b), 值域[0,1], 完全相等=1"""
        mx = max(abs(a), abs(b))
        if mx < 1e-10:
            return 1.0  # 两者都近零，视为完全对称
        return 1.0 - abs(a - b) / mx
    
    # 扫描所有三波组合
    structures = []
    seen = set()
    
    for p2 in list(end_at.keys()):
        # seg_A ends at p2
        for seg_A in end_at[p2]:
            # seg_B starts at p2
            for seg_B in start_at.get(p2, []):
                p3 = seg_B['bar_end']
                
                # seg_C starts at p3
                for seg_C in start_at.get(p3, []):
                    p1 = seg_A['bar_start']
                    p4 = seg_C['bar_end']
                    
                    # 时间递增
                    if not (p1 < p2 and p2 < p3 and p3 < p4):
                        continue
                    
                    # 去重 (用三波的4个端点作为唯一键)
                    key = (p1, p2, p3, p4)
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    # 方向检查: A和C方向相同, B方向相反
                    # 上升段: price_end > price_start, 下降段: price_end < price_start
                    dir_A = 1 if seg_A['price_end'] > seg_A['price_start'] else -1
                    dir_B = 1 if seg_B['price_end'] > seg_B['price_start'] else -1
                    dir_C = 1 if seg_C['price_end'] > seg_C['price_start'] else -1
                    
                    if dir_A != dir_C or dir_A == dir_B:
                        continue  # A和C必须同向, B必须反向
                    
                    # === 计算5维对称度 ===
                    amp_A = seg_A['amplitude']
                    amp_C = seg_C['amplitude']
                    time_A = seg_A['span']
                    time_C = seg_C['span']
                    
                    if amp_A < 1e-10 or amp_C < 1e-10 or time_A < 1 or time_C < 1:
                        continue
                    
                    # D1: 幅度对称
                    amp_sym = _sym_ratio(amp_A, amp_C)
                    
                    # D2: 时间对称
                    time_sym = _sym_ratio(time_A, time_C)
                    
                    # D3: 模长对称
                    mod_A = _compute_modulus(amp_A, time_A)
                    mod_C = _compute_modulus(amp_C, time_C)
                    mod_sym = _sym_ratio(mod_A, mod_C)
                    
                    # D4: 斜率对称
                    slope_A = amp_A / time_A * dir_A
                    slope_C = amp_C / time_C * dir_C
                    # A和C同向, 所以slope符号相同, 镜像对称 = slope_A ≈ slope_C
                    slope_sym = _sym_ratio(abs(slope_A), abs(slope_C))
                    
                    # D5: 内部结构对称 (子波段数)
                    sub_A = _count_sub_segments(p1, p2)
                    sub_C = _count_sub_segments(p3, p4)
                    complexity_sym = _sym_ratio(sub_A, sub_C)
                    
                    # 综合对称度 (加权)
                    w_sym = {
                        'amplitude': 0.25,
                        'time': 0.20,
                        'modulus': 0.25,     # 核心维度
                        'slope': 0.15,
                        'complexity': 0.15,
                    }
                    
                    sym_score = (
                        w_sym['amplitude'] * amp_sym +
                        w_sym['time'] * time_sym +
                        w_sym['modulus'] * mod_sym +
                        w_sym['slope'] * slope_sym +
                        w_sym['complexity'] * complexity_sym
                    )
                    
                    # 结构类型判定
                    amp_B = seg_B['amplitude']
                    if dir_A == -1:
                        # A下降, B上升, C下降 → V底型 (先跌-反弹-再跌)
                        struct_type = 'V_bottom' if amp_C < amp_A else 'descending'
                    else:
                        # A上升, B下降, C上升 → 倒V顶型 (先涨-回调-再涨)
                        struct_type = 'inv_V_top' if amp_C < amp_A else 'ascending'
                    
                    # 端点重要性加权的综合得分
                    imp_A_start = pivot_info.get(p1, {}).get('importance', 0)
                    imp_A_end = pivot_info.get(p2, {}).get('importance', 0)
                    imp_C_end = pivot_info.get(p4, {}).get('importance', 0)
                    endpoint_imp = (imp_A_start + imp_A_end + imp_C_end) / 3.0
                    
                    # 最终得分 = 对称度 × 端点重要性
                    final_score = sym_score * endpoint_imp
                    
                    structures.append({
                        'score': round(final_score, 6),
                        'sym_score': round(sym_score, 4),
                        'endpoint_imp': round(endpoint_imp, 4),
                        'vec': {
                            'amp': round(amp_sym, 4),
                            'time': round(time_sym, 4),
                            'mod': round(mod_sym, 4),
                            'slope': round(slope_sym, 4),
                            'complexity': round(complexity_sym, 4),
                        },
                        'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4,
                        'price_p1': seg_A['price_start'],
                        'price_p2': seg_A['price_end'],
                        'price_p3': seg_B['price_end'],
                        'price_p4': seg_C['price_end'],
                        'amp_A': round(amp_A, 5), 'amp_B': round(amp_B, 5), 'amp_C': round(amp_C, 5),
                        'time_A': time_A, 'time_B': seg_B['span'], 'time_C': time_C,
                        'dir': dir_A,  # A和C的方向
                        'type': struct_type,
                    })
    
    # 按最终得分排序
    structures.sort(key=lambda s: -s['score'])
    
    return structures[:top_n]


def prune_redundant(pool, keep_ratio=0.5):
    """
    冗余删除。
    
    策略：按重要性从高到低遍历。对每条线段，检查它是否与已保留的某条
    高重要性线段"高度重叠"。如果是，标记为冗余。
    
    "高度重叠" = 两条线段的时间区间重叠率 > 阈值，且幅度差异小。
    
    返回: (保留的线段列表, 被删除的线段列表)
    """
    kept = []
    pruned = []
    
    for seg in pool:
        is_redundant = False
        s1, e1 = seg['bar_start'], seg['bar_end']
        
        for k in kept:
            s2, e2 = k['bar_start'], k['bar_end']
            
            # 时间重叠
            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)
            if overlap_end <= overlap_start:
                continue  # 无重叠
            
            overlap_len = overlap_end - overlap_start
            shorter_span = min(e1 - s1, e2 - s2)
            if shorter_span == 0:
                continue
            
            overlap_ratio = overlap_len / shorter_span
            
            # 幅度相似度
            if k['amplitude'] > 0:
                amp_ratio = min(seg['amplitude'], k['amplitude']) / max(seg['amplitude'], k['amplitude'])
            else:
                amp_ratio = 1.0 if seg['amplitude'] == 0 else 0.0
            
            # 如果时间重叠>80% 且 幅度相似>60%，认为冗余
            if overlap_ratio > 0.8 and amp_ratio > 0.6:
                is_redundant = True
                break
        
        if is_redundant:
            pruned.append(seg)
        else:
            kept.append(seg)
    
    return kept, pruned


# =============================================================================
# 7. 主程序
# =============================================================================

def main():
    print("=" * 70)
    print("完整归并引擎 v3.0 — 消融级别界限")
    print("=" * 70)

    filepath = "/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv"
    limit = 200  # 验证阶段用小窗口

    print(f"\n加载: {filepath} (最后{limit}根)")
    df = load_kline(filepath, limit=limit)
    print(f"范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"K线: {len(df)}")

    high = df['high'].values
    low = df['low'].values

    t0 = time.time()
    base_pivots = calculate_base_zg(high, low)
    t1 = time.time()
    print(f"\n基础ZG: {len(base_pivots)} 拐点 ({t1-t0:.3f}s, {len(df)/max(len(base_pivots),1):.2f} K/pivot)")

    t0 = time.time()
    results = full_merge_engine(base_pivots)
    t1 = time.time()
    print(f"\n归并完成! {t1-t0:.3f}s")
    print(f"  迭代次数: {results['total_iterations']}")
    print(f"  幅度归并累计: {results['total_amp_levels']}级")
    print(f"  横向归并累计: {results['total_lat_batches']}批")
    print(f"  最终拐点: {len(results['final_pivots'])}")
    print(f"  快照总数: {len(results['all_snapshots'])}")

    # 快照列表
    print(f"\n快照序列:")
    for snap_type, label, pvts in results['all_snapshots']:
        tag = 'amp' if snap_type == 'amp' else ('lat' if snap_type == 'lat' else 'base')
        print(f"  [{tag:4s}] {label:5s}: {len(pvts):4d} 拐点")

    print("\n归并日志:")
    for entry in results['log']:
        print(f"  {entry}")

    # ===== 点重要性 (多维度) =====
    print("\n" + "=" * 70)
    print("多维度拐点重要性分析")
    print("=" * 70)
    
    pivot_info = compute_pivot_importance(results, total_bars=len(df))
    
    # Top 10 峰 + Top 10 谷
    peaks = sorted([v for v in pivot_info.values() if v['dir'] == 1], key=lambda x: -x['importance'])
    valleys = sorted([v for v in pivot_info.values() if v['dir'] == -1], key=lambda x: -x['importance'])
    
    # 百分比截断 — 缺省显示前50%
    pct = 0.5
    n_show_peaks = max(1, int(len(peaks) * pct))
    n_show_valleys = max(1, int(len(valleys) * pct))
    
    print(f"\nTop {n_show_peaks} 重要峰 (共{len(peaks)}峰, 前{int(pct*100)}%):")
    print(f"  {'bar':>4s} {'price':>9s} | {'imp':>6s} | {'lines':>5s} {'surv':>4s} {'amp':>8s} {'span':>4s} {'extm':>5s} {'isol':>5s} {'dom':>5s} {'rec':>6s} | datetime")
    for p in peaks[:n_show_peaks]:
        dt = df.iloc[p['bar']]['datetime']
        print(f"  {p['bar']:4d} {p['price']:.5f} | {p['importance']:.4f} | "
              f"{p['d1_line_count']:5d} {p['d2_survival']:4d} {p['d3_amplitude']:.5f} {p['d4_time_span']:4d} "
              f"{p['d5_extremity']:.3f} {p['d6_isolation']:5.1f} {p['d7_dominance']:.3f} {p['d8_recency']:.4f} | {dt}")
    
    print(f"\nTop {n_show_valleys} 重要谷 (共{len(valleys)}谷, 前{int(pct*100)}%):")
    print(f"  {'bar':>4s} {'price':>9s} | {'imp':>6s} | {'lines':>5s} {'surv':>4s} {'amp':>8s} {'span':>4s} {'extm':>5s} {'isol':>5s} {'dom':>5s} {'rec':>6s} | datetime")
    for p in valleys[:n_show_valleys]:
        dt = df.iloc[p['bar']]['datetime']
        print(f"  {p['bar']:4d} {p['price']:.5f} | {p['importance']:.4f} | "
              f"{p['d1_line_count']:5d} {p['d2_survival']:4d} {p['d3_amplitude']:.5f} {p['d4_time_span']:4d} "
              f"{p['d5_extremity']:.3f} {p['d6_isolation']:5.1f} {p['d7_dominance']:.3f} {p['d8_recency']:.4f} | {dt}")

    # ===== 波段池 =====
    print("\n" + "=" * 70)
    print("波段池构建 (初始)")
    print("=" * 70)
    
    pool = build_segment_pool(results, pivot_info)
    amp_segs = sum(1 for s in pool if s['source'] == 'amp')
    lat_segs = sum(1 for s in pool if s['source'] == 'lat')
    base_segs = sum(1 for s in pool if s['source'] == 'base')
    print(f"\n初始波段池: {len(pool)} 条 (base:{base_segs}, amp:{amp_segs}, lat:{lat_segs})")

    # ===== 波段池内三波归并 (消融级别) =====
    print("\n" + "=" * 70)
    print("波段池三波归并 — 消融级别界限")
    print("=" * 70)
    
    t0 = time.time()
    full_pool, new_segs, fusion_log = pool_fusion(pool, pivot_info)
    t1 = time.time()
    
    fusion_segs = sum(1 for s in full_pool if s['source'] == 'fusion')
    print(f"\n归并后波段池: {len(full_pool)} 条 (新增fusion:{fusion_segs}) {t1-t0:.3f}s")
    print(f"\n归并日志:")
    for entry in fusion_log:
        print(f"  {entry}")
    
    # 验证关键连接
    print(f"\n关键连接验证:")
    checks = [
        (3, 72, 'H1->L8'), (3, 98, 'H1->L5'),
        (39, 166, 'L4->H2'), (142, 195, 'H7->L2'),
    ]
    pool_keys = {(s['bar_start'], s['bar_end']) for s in full_pool}
    for b1, b2, desc in checks:
        status = 'OK' if (b1, b2) in pool_keys else 'MISSING'
        print(f"  {status:7s} | {desc}")
    
    # Top线段
    print(f"\nTop 20 重要线段:")
    for s in full_pool[:20]:
        d1 = 'H' if s['dir_start'] == 1 else 'L'
        d2 = 'H' if s['dir_end'] == 1 else 'L'
        src = s['source_label']
        via = ''
        if 'fusion_via' in s:
            via = f' via({s["fusion_via"][0]},{s["fusion_via"][1]})'
        print(f"  [{src:6s}] bar{s['bar_start']:4d}{d1}({s['price_start']:.5f}) -> "
              f"bar{s['bar_end']:4d}{d2}({s['price_end']:.5f}) | "
              f"span={s['span']:4d} amp={s['amplitude']:.5f} | imp={s['importance']:.3f}{via}")
    
    # ===== 对称结构识别 =====
    print("\n" + "=" * 70)
    print("对称结构识别 — 5维对称向量")
    print("=" * 70)
    
    t0 = time.time()
    sym_structures = find_symmetric_structures(full_pool, pivot_info, df=df, top_n=200)
    t1 = time.time()
    
    print(f"\n发现 {len(sym_structures)} 个对称结构 ({t1-t0:.3f}s)")
    
    if sym_structures:
        # 统计
        types = {}
        for s in sym_structures:
            t = s['type']
            types[t] = types.get(t, 0) + 1
        print(f"类型分布: {types}")
        
        # 高对称度统计 (sym_score > 0.8)
        high_sym = [s for s in sym_structures if s['sym_score'] > 0.8]
        print(f"高对称度(>0.8): {len(high_sym)}个")
        
        print(f"\nTop 30 对称结构:")
        print(f"  {'score':>7s} {'sym':>5s} {'imp':>5s} | {'amp':>5s} {'time':>5s} {'mod':>5s} {'slp':>5s} {'cplx':>5s} | "
              f"{'type':>12s} | p1→p2→p3→p4 | A/B/C幅度 | A/B/C时间")
        for s in sym_structures[:30]:
            v = s['vec']
            d = '↑' if s['dir'] == 1 else '↓'
            print(f"  {s['score']:.5f} {s['sym_score']:.3f} {s['endpoint_imp']:.3f} | "
                  f"{v['amp']:.3f} {v['time']:.3f} {v['mod']:.3f} {v['slope']:.3f} {v['complexity']:.3f} | "
                  f"{s['type']:>12s}{d} | {s['p1']:3d}→{s['p2']:3d}→{s['p3']:3d}→{s['p4']:3d} | "
                  f"{s['amp_A']:.4f}/{s['amp_B']:.4f}/{s['amp_C']:.4f} | "
                  f"{s['time_A']:3d}/{s['time_B']:3d}/{s['time_C']:3d}")

if __name__ == '__main__':
    main()
