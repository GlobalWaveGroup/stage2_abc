#!/usr/bin/env python3
"""
完整归并引擎 v2.0
核心修正：所有级别的线段共存，不删除低级别信息。
每轮归并在上一级的拐点序列上产生新一级的拐点序列。
最终输出 = 所有级别的拐点序列集合 = 波段池。

架构：
  Level 0: 基础ZG拐点
  Level 1: 对Level 0做一轮幅度归并
  Level 2: 对Level 1做一轮幅度归并
  ...
  Level N: 幅度归并无变化 → 对Level N做横向归并 → Level N+1
  Level N+1: 对Level N+1做幅度归并 → Level N+2
  ...
  直到不动点

分层交替迭代：
  while(有变化) {
      while(幅度归并有变化) { 产生新level; }
      横向归并一轮 → 产生新level;
  }
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
# 5. 完整归并引擎 v2 — 所有级别共存
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

def compute_pivot_importance(results, high=None, low=None):
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
        
        # 加权综合 (权重可调)
        w = {
            'line_count': 0.20,
            'survival': 0.15,
            'amplitude': 0.20,
            'time_span': 0.10,
            'extremity': 0.15,
            'isolation': 0.10,
            'dominance': 0.10,
        }
        
        importance = (
            w['line_count'] * n_lc.get(bar, 0) +
            w['survival'] * n_sc.get(bar, 0) +
            w['amplitude'] * n_amp.get(bar, 0) +
            w['time_span'] * n_span.get(bar, 0) +
            w['extremity'] * extremity +
            w['isolation'] * n_iso.get(bar, 0) +
            w['dominance'] * dominance
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
    线段重要性 = min(两端点重要性)，短板决定。
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
                'importance': min(imp1, imp2),
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
    print("完整归并引擎 v2.1 — 交替迭代至不动点")
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
    
    pivot_info = compute_pivot_importance(results)
    
    # Top 10 峰 + Top 10 谷
    peaks = sorted([v for v in pivot_info.values() if v['dir'] == 1], key=lambda x: -x['importance'])
    valleys = sorted([v for v in pivot_info.values() if v['dir'] == -1], key=lambda x: -x['importance'])
    
    print(f"\nTop 10 重要峰 (共{len(peaks)}峰):")
    print(f"  {'bar':>4s} {'price':>9s} | {'imp':>6s} | {'lines':>5s} {'surv':>4s} {'amp':>8s} {'span':>4s} {'extm':>5s} {'isol':>5s} {'dom':>5s} | datetime")
    for p in peaks[:10]:
        dt = df.iloc[p['bar']]['datetime']
        print(f"  {p['bar']:4d} {p['price']:.5f} | {p['importance']:.4f} | "
              f"{p['d1_line_count']:5d} {p['d2_survival']:4d} {p['d3_amplitude']:.5f} {p['d4_time_span']:4d} "
              f"{p['d5_extremity']:.3f} {p['d6_isolation']:5.1f} {p['d7_dominance']:.3f} | {dt}")
    
    print(f"\nTop 10 重要谷 (共{len(valleys)}谷):")
    print(f"  {'bar':>4s} {'price':>9s} | {'imp':>6s} | {'lines':>5s} {'surv':>4s} {'amp':>8s} {'span':>4s} {'extm':>5s} {'isol':>5s} {'dom':>5s} | datetime")
    for p in valleys[:10]:
        dt = df.iloc[p['bar']]['datetime']
        print(f"  {p['bar']:4d} {p['price']:.5f} | {p['importance']:.4f} | "
              f"{p['d1_line_count']:5d} {p['d2_survival']:4d} {p['d3_amplitude']:.5f} {p['d4_time_span']:4d} "
              f"{p['d5_extremity']:.3f} {p['d6_isolation']:5.1f} {p['d7_dominance']:.3f} | {dt}")

    # ===== 波段池 =====
    print("\n" + "=" * 70)
    print("波段池构建")
    print("=" * 70)
    
    pool = build_segment_pool(results, pivot_info)
    amp_segs = sum(1 for s in pool if s['source'] == 'amp')
    lat_segs = sum(1 for s in pool if s['source'] == 'lat')
    base_segs = sum(1 for s in pool if s['source'] == 'base')
    print(f"\n去重波段池: {len(pool)} 条 (base:{base_segs}, amp:{amp_segs}, lat:{lat_segs})")
    
    # Top线段
    print(f"\nTop 10 重要线段:")
    for s in pool[:10]:
        d1 = 'H' if s['dir_start'] == 1 else 'L'
        d2 = 'H' if s['dir_end'] == 1 else 'L'
        print(f"  [{s['source_label']:5s}] bar{s['bar_start']:4d}{d1}({s['price_start']:.5f}) -> "
              f"bar{s['bar_end']:4d}{d2}({s['price_end']:.5f}) | "
              f"span={s['span']:4d} amp={s['amplitude']:.5f} | imp={s['importance']:.3f}")

if __name__ == '__main__':
    main()
