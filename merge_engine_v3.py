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


# =============================================================================
# 6c. 统一对称谱 — 多维特征向量 (v3.3)
# =============================================================================

def compute_symmetry_spectrum(pool, pivot_info, max_pool_size=800):
    """
    统一对称谱计算 — 覆盖所有对称模式。
    
    核心思想:
    - pool_fusion已经把所有尺度的线段打平了
    - 一条线段可能是base单跳,也可能是多段归并后的整体向量
    - 所以只需搜索线段之间的关系,不需要枚举"2段/3段/4段/5段"
    
    搜索两类对称对:
    
    A. 镜像对(Mirror): 共享一个端点, 方向相反
       即 seg_A→[P]←seg_B, 在点P处折返
       覆盖: V底/V顶, 双顶/双底(当线段是多段折叠后), 头肩(当线段覆盖更大尺度)
       
    B. 中心对(Center): 通过一段中间线段连接, 方向相同
       即 seg_A→[center]→seg_C, A和C同向
       覆盖: abc调整, 12345推动(当线段是多段折叠后), 模长对称
    
    对每个对称对输出完整特征向量(对称谱), 不压缩为标量:
    - amp_ratio, time_ratio, mod_ratio, slope_ratio (各维度的A/C比值)
    - 偏离方向和程度
    - 中心段信息
    - 端点重要性
    
    返回: list of spectrum dicts, 按综合重要性排序
    """
    from collections import defaultdict
    import math, bisect
    
    # 限制搜索空间
    search_pool = pool
    if len(pool) > max_pool_size:
        search_pool = sorted(pool, key=lambda s: -s['importance'])[:max_pool_size]
    
    # 建索引
    end_at = defaultdict(list)    # bar → [seg ending here]
    start_at = defaultdict(list)  # bar → [seg starting here]
    for s in search_pool:
        end_at[s['bar_end']].append(s)
        start_at[s['bar_start']].append(s)
    
    # 全局统计(用完整池)
    all_amps = [s['amplitude'] for s in pool if s['amplitude'] > 0]
    all_spans = [s['span'] for s in pool if s['span'] > 0]
    global_amp = max(all_amps) if all_amps else 1
    global_span = max(all_spans) if all_spans else 1
    
    # 子波段数(复杂度)
    base_bars_sorted = sorted(p['bar'] for p in pivot_info.values())
    
    def _count_sub(bar_start, bar_end):
        left = bisect.bisect_right(base_bars_sorted, bar_start)
        right = bisect.bisect_left(base_bars_sorted, bar_end)
        return right - left
    
    def _seg_direction(s):
        """线段方向: +1上升, -1下降"""
        return 1 if s['price_end'] > s['price_start'] else -1
    
    def _compute_mod(amp, span):
        """归一化模长"""
        na = amp / max(global_amp, 1e-10)
        nt = span / max(global_span, 1)
        return math.sqrt(na**2 + nt**2)
    
    def _safe_ratio(a, b):
        """安全除法, 返回ratio和log_ratio"""
        if b < 1e-10 and a < 1e-10:
            return 1.0, 0.0
        if b < 1e-10:
            return 999.0, math.log(999)
        r = a / b
        return r, math.log(max(r, 1e-10))
    
    def _make_spectrum(seg_L, seg_R, sym_type, center_seg=None, pivot_bar=None):
        """构造一个对称谱向量"""
        amp_L = seg_L['amplitude']
        amp_R = seg_R['amplitude']
        time_L = seg_L['span']
        time_R = seg_R['span']
        mod_L = _compute_mod(amp_L, time_L)
        mod_R = _compute_mod(amp_R, time_R)
        slope_L = amp_L / max(time_L, 1)
        slope_R = amp_R / max(time_R, 1)
        sub_L = _count_sub(seg_L['bar_start'], seg_L['bar_end'])
        sub_R = _count_sub(seg_R['bar_start'], seg_R['bar_end'])
        
        amp_ratio, amp_log = _safe_ratio(amp_L, amp_R)
        time_ratio, time_log = _safe_ratio(time_L, time_R)
        mod_ratio, mod_log = _safe_ratio(mod_L, mod_R)
        slope_ratio, slope_log = _safe_ratio(slope_L, slope_R)
        
        # 端点重要性
        imp_bars = [seg_L['bar_start'], seg_L['bar_end'], seg_R['bar_start'], seg_R['bar_end']]
        imps = [pivot_info.get(b, {}).get('importance', 0) for b in imp_bars]
        mean_imp = sum(imps) / len(imps)
        
        # 中心段信息
        center_amp = 0
        center_span = 0
        center_bar_start = 0
        center_bar_end = 0
        if center_seg:
            center_amp = center_seg['amplitude']
            center_span = center_seg['span']
            center_bar_start = center_seg['bar_start']
            center_bar_end = center_seg['bar_end']
        
        # 对称度(接近1的程度) — 作为排序用的标量, 但完整向量才是真正的特征
        sym_closeness = 1.0 / (1.0 + abs(amp_log) + abs(time_log) + abs(mod_log) + abs(slope_log))
        
        return {
            'type': sym_type,      # 'mirror' or 'center'
            'dir_L': _seg_direction(seg_L),
            'dir_R': _seg_direction(seg_R),
            
            # 左臂
            'L_start': seg_L['bar_start'], 'L_end': seg_L['bar_end'],
            'L_price_start': seg_L['price_start'], 'L_price_end': seg_L['price_end'],
            'L_amp': round(amp_L, 5), 'L_time': time_L,
            'L_mod': round(mod_L, 6), 'L_slope': round(slope_L, 7),
            'L_complexity': sub_L,
            
            # 右臂
            'R_start': seg_R['bar_start'], 'R_end': seg_R['bar_end'],
            'R_price_start': seg_R['price_start'], 'R_price_end': seg_R['price_end'],
            'R_amp': round(amp_R, 5), 'R_time': time_R,
            'R_mod': round(mod_R, 6), 'R_slope': round(slope_R, 7),
            'R_complexity': sub_R,
            
            # 对称谱向量 — 各维度ratio (核心特征)
            'amp_ratio': round(amp_ratio, 4),
            'time_ratio': round(time_ratio, 4),
            'mod_ratio': round(mod_ratio, 4),
            'slope_ratio': round(slope_ratio, 4),
            
            # 对称谱向量 — log空间 (利于连续映射)
            'amp_log': round(amp_log, 4),
            'time_log': round(time_log, 4),
            'mod_log': round(mod_log, 4),
            'slope_log': round(slope_log, 4),
            
            # 复杂度差异
            'complexity_ratio': round(_safe_ratio(max(sub_L,1), max(sub_R,1))[0], 4),
            'complexity_diff': sub_L - sub_R,
            
            # 中心段
            'center_bar_start': center_bar_start,
            'center_bar_end': center_bar_end,
            'center_amp': round(center_amp, 5),
            'center_span': center_span,
            
            # 镜像点(仅mirror类型)
            'pivot_bar': pivot_bar if pivot_bar else 0,
            
            # 端点重要性
            'mean_imp': round(mean_imp, 4),
            'max_imp': round(max(imps), 4),
            
            # 标量汇总(仅用于排序, 不作为特征)
            'sym_closeness': round(sym_closeness, 4),
            'score': round(sym_closeness * mean_imp, 6),
        }
    
    results = []
    seen = set()
    
    # === A. 镜像对: 共享一个端点P, 方向相反 ===
    # seg_L ends at P, seg_R starts at P
    all_bars = set(list(end_at.keys()) + list(start_at.keys()))
    for p in all_bars:
        segs_ending = end_at.get(p, [])
        segs_starting = start_at.get(p, [])
        if not segs_ending or not segs_starting:
            continue
        
        for seg_L in segs_ending:
            dir_L = _seg_direction(seg_L)
            for seg_R in segs_starting:
                dir_R = _seg_direction(seg_R)
                
                # 镜像: 方向相反 (一个上升到P, 另一个从P下降, 或反之)
                if dir_L == dir_R:
                    continue
                
                # 时间递增
                if seg_L['bar_start'] >= p or seg_R['bar_end'] <= p:
                    continue
                
                # 去重
                key = ('M', seg_L['bar_start'], p, seg_R['bar_end'])
                if key in seen:
                    continue
                seen.add(key)
                
                # 最小有效性
                if seg_L['amplitude'] < 1e-10 or seg_R['amplitude'] < 1e-10:
                    continue
                if seg_L['span'] < 1 or seg_R['span'] < 1:
                    continue
                
                spec = _make_spectrum(seg_L, seg_R, 'mirror', pivot_bar=p)
                results.append(spec)
    
    # === B. 中心对: seg_L→center→seg_R, L和R同向 ===
    for p2 in list(end_at.keys()):
        for seg_L in end_at[p2]:
            dir_L = _seg_direction(seg_L)
            for center in start_at.get(p2, []):
                p3 = center['bar_end']
                dir_center = _seg_direction(center)
                
                # 中心方向应该和L/R相反
                if dir_center == dir_L:
                    continue
                
                for seg_R in start_at.get(p3, []):
                    dir_R = _seg_direction(seg_R)
                    
                    # R和L同向
                    if dir_R != dir_L:
                        continue
                    
                    p1 = seg_L['bar_start']
                    p4 = seg_R['bar_end']
                    
                    # 时间递增
                    if not (p1 < p2 and p2 < p3 and p3 < p4):
                        continue
                    
                    # 去重
                    key = ('C', p1, p2, p3, p4)
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    # 最小有效性
                    if seg_L['amplitude'] < 1e-10 or seg_R['amplitude'] < 1e-10:
                        continue
                    if seg_L['span'] < 1 or seg_R['span'] < 1:
                        continue
                    
                    spec = _make_spectrum(seg_L, seg_R, 'center', center_seg=center)
                    results.append(spec)
    
    # 按score排序
    results.sort(key=lambda s: -s['score'])
    
    return results


def predict_symmetric_image(pool, pivot_info, current_bar, max_pool_size=500):
    """
    对称映像预测 — 路径一核心算法。
    
    给定当前时刻的波段池，对每个可能的对称结构（左臂A + 中心/轴），
    推演右臂C'的预测目标。
    
    四种预测类型：
    
    A. Mirror预测（轴对称）：
       已知左臂A到达折返点P → 预测C'从P出发，方向相反于A
       C'_amp ≈ A_amp, C'_time ≈ A_time
       → 目标价格 = P_price ± A_amp (方向相反)
       → 目标时间 = P_bar + A_time
    
    B. Center预测（中心对称 / C≈A经典）：
       已知左臂A和中间段B → 预测C'从B_end出发，方向同A
       C'_amp ≈ A_amp, C'_time ≈ A_time
       → 目标价格 = B_end_price + A_amp * A_direction
       → 目标时间 = B_end_bar + A_time
    
    C. Triangle预测（三角收敛/发散）：
       已知A和B，B/A比率(r)决定C的缩放
       收敛 (r < 0.85): amp_C = amp_B * r (等比递减)
       发散 (r > 1.18): amp_C = amp_B * r (等比递增)
       → 目标价格 = B_end_price + pred_dir * amp_C
       → 目标时间 = B_end_bar + time_A * r (时间也按比率缩放)
    
    D. ModOnly预测（模长守恒但非经典对称）：
       mod_C ≈ mod_A 但 amp_C ≠ amp_A
       直接输出弧线参数，弧线本身表达所有可能的amp/time组合
       仅在非经典、非三角时产生（避免重复）
    
    筛选条件：
    - 预测目标必须在current_bar之后（才有预测价值）
    - 或者预测目标正在展开中（部分完成）
    
    返回: list of prediction dicts, 按重要性排序
    """
    from collections import defaultdict
    import math
    
    # 限制搜索空间
    search_pool = pool
    if len(pool) > max_pool_size:
        search_pool = sorted(pool, key=lambda s: -s['importance'])[:max_pool_size]
    
    # 建索引
    end_at = defaultdict(list)
    start_at = defaultdict(list)
    for s in search_pool:
        end_at[s['bar_end']].append(s)
        start_at[s['bar_start']].append(s)
    
    # 全局统计
    all_amps = [s['amplitude'] for s in pool if s['amplitude'] > 0]
    global_amp = max(all_amps) if all_amps else 1
    
    def _seg_dir(s):
        return 1 if s['price_end'] > s['price_start'] else -1
    
    predictions = []
    seen = set()
    
    # === A. Mirror预测 ===
    # seg_A到达P(fold point), 预测C'从P出发反向
    all_bars = set(list(end_at.keys()) + list(start_at.keys()))
    for p in all_bars:
        segs_ending = end_at.get(p, [])
        segs_starting = start_at.get(p, [])
        
        for seg_A in segs_ending:
            dir_A = _seg_dir(seg_A)
            amp_A = seg_A['amplitude']
            time_A = seg_A['span']
            p1 = seg_A['bar_start']
            
            if amp_A < 1e-10 or time_A < 1:
                continue
            if p1 >= p:
                continue
            
            # 预测C': 从P出发，方向相反
            pred_dir = -dir_A
            p_price = seg_A['price_end']  # 折返点价格
            
            # C≈A 假设
            pred_target_price = p_price + pred_dir * amp_A
            pred_target_bar = p + time_A
            
            # 只保留有预测价值的（目标在未来，或部分在未来）
            if pred_target_bar <= seg_A['bar_start']:
                continue
            
            # 去重
            key = ('M', p1, p, pred_dir)
            if key in seen:
                continue
            seen.add(key)
            
            # 是否已有实际的C段在展开？
            actual_C = None
            actual_progress = 0.0
            for seg_R in segs_starting:
                dir_R = _seg_dir(seg_R)
                if dir_R == pred_dir and seg_R['bar_end'] > p:
                    # 这是一个实际展开的C段
                    if actual_C is None or seg_R['bar_end'] > actual_C['bar_end']:
                        actual_C = seg_R
            
            if actual_C:
                actual_progress = min(1.0, actual_C['amplitude'] / max(amp_A, 1e-10))
            
            # 端点重要性
            imp_A_start = pivot_info.get(p1, {}).get('importance', 0)
            imp_P = pivot_info.get(p, {}).get('importance', 0)
            pred_imp = (imp_A_start + imp_P) / 2
            
            # 相对幅度(占全局的比例)
            rel_amp = amp_A / max(global_amp, 1e-10)
            
            predictions.append({
                'type': 'mirror',
                'sym_type': 'axial',
                
                # 左臂A
                'A_start': p1,
                'A_end': p,
                'A_price_start': seg_A['price_start'],
                'A_price_end': seg_A['price_end'],
                'A_amp': amp_A,
                'A_time': time_A,
                'A_dir': dir_A,
                
                # 对称中心
                'center_bar': p,
                'center_price': p_price,
                'center_type': 'point',
                
                # 预测C'
                'pred_dir': pred_dir,
                'pred_start_bar': p,
                'pred_start_price': p_price,
                'pred_target_price': pred_target_price,
                'pred_target_bar': pred_target_bar,
                'pred_amp': amp_A,
                'pred_time': time_A,
                
                # 实际C展开情况
                'actual_C': actual_C,
                'actual_progress': actual_progress,
                
                # 重要性
                'importance': pred_imp,
                'rel_amp': rel_amp,
                'score': pred_imp * rel_amp,
            })
    
    # === B. Center预测 ===
    # seg_A → center_B → 预测C'从B_end出发，方向同A
    for p2 in list(end_at.keys()):
        for seg_A in end_at[p2]:
            dir_A = _seg_dir(seg_A)
            amp_A = seg_A['amplitude']
            time_A = seg_A['span']
            p1 = seg_A['bar_start']
            
            if amp_A < 1e-10 or time_A < 1:
                continue
            
            for center_B in start_at.get(p2, []):
                dir_B = _seg_dir(center_B)
                p3 = center_B['bar_end']
                
                # center方向应与A相反
                if dir_B == dir_A:
                    continue
                if not (p1 < p2 < p3):
                    continue
                if center_B['amplitude'] < 1e-10:
                    continue
                
                # 预测C': 从p3出发，方向同A
                pred_dir = dir_A
                p3_price = center_B['price_end']
                
                # C≈A 假设
                pred_target_price = p3_price + pred_dir * amp_A
                pred_target_bar = p3 + time_A
                
                if pred_target_bar <= p1:
                    continue
                
                # 去重
                key = ('C', p1, p2, p3, pred_dir)
                if key in seen:
                    continue
                seen.add(key)
                
                # 是否已有实际C段在展开？
                actual_C = None
                actual_progress = 0.0
                for seg_R in start_at.get(p3, []):
                    dir_R = _seg_dir(seg_R)
                    if dir_R == pred_dir and seg_R['bar_end'] > p3:
                        if actual_C is None or seg_R['bar_end'] > actual_C['bar_end']:
                            actual_C = seg_R
                
                if actual_C:
                    actual_progress = min(1.0, actual_C['amplitude'] / max(amp_A, 1e-10))
                
                # 端点重要性
                imp_bars = [p1, p2, p3]
                imps = [pivot_info.get(b, {}).get('importance', 0) for b in imp_bars]
                pred_imp = sum(imps) / len(imps)
                
                rel_amp = amp_A / max(global_amp, 1e-10)
                
                # center段信息
                center_amp = center_B['amplitude']
                center_span = center_B['span']
                
                # B段回撤比例 (B/A)
                retrace_ratio = center_amp / max(amp_A, 1e-10)
                
                predictions.append({
                    'type': 'center',
                    'sym_type': 'rotational',
                    
                    # 左臂A
                    'A_start': p1,
                    'A_end': p2,
                    'A_price_start': seg_A['price_start'],
                    'A_price_end': seg_A['price_end'],
                    'A_amp': amp_A,
                    'A_time': time_A,
                    'A_dir': dir_A,
                    
                    # 中间段B
                    'B_start': p2,
                    'B_end': p3,
                    'B_price_start': center_B['price_start'],
                    'B_price_end': center_B['price_end'],
                    'B_amp': center_amp,
                    'B_time': center_span,
                    'retrace_ratio': round(retrace_ratio, 4),
                    
                    # 对称中心 (B段中点)
                    'center_bar': (p2 + p3) / 2,
                    'center_price': (center_B['price_start'] + center_B['price_end']) / 2,
                    'center_type': 'segment_midpoint',
                    
                    # 预测C'
                    'pred_dir': pred_dir,
                    'pred_start_bar': p3,
                    'pred_start_price': p3_price,
                    'pred_target_price': pred_target_price,
                    'pred_target_bar': pred_target_bar,
                    'pred_amp': amp_A,
                    'pred_time': time_A,
                    
                    # 实际C展开情况
                    'actual_C': actual_C,
                    'actual_progress': actual_progress,
                    
                    # 重要性
                    'importance': pred_imp,
                    'rel_amp': rel_amp,
                    'retrace_ratio': round(retrace_ratio, 4),
                    'score': pred_imp * rel_amp,
                })
    
    # === C. Triangle预测（三角收敛/发散） ===
    # 在center搜索的基础上，对B/A比率不接近1的情况，用等比缩放预测C
    # 收敛: ratio < 0.85 → amp_C = amp_B * ratio (振幅递减)
    # 发散: ratio > 1.18 → amp_C = amp_B * ratio (振幅递增)
    
    # 全局统计(用于模长计算)
    all_spans = [s['span'] for s in pool if s['span'] > 0]
    global_span = max(all_spans) if all_spans else 1
    
    for p2 in list(end_at.keys()):
        for seg_A in end_at[p2]:
            dir_A = _seg_dir(seg_A)
            amp_A = seg_A['amplitude']
            time_A = seg_A['span']
            p1 = seg_A['bar_start']
            
            if amp_A < 1e-10 or time_A < 1:
                continue
            
            for seg_B in start_at.get(p2, []):
                dir_B = _seg_dir(seg_B)
                p3 = seg_B['bar_end']
                amp_B = seg_B['amplitude']
                time_B = seg_B['span']
                
                # B方向应与A相反
                if dir_B == dir_A:
                    continue
                if not (p1 < p2 < p3):
                    continue
                if amp_B < 1e-10 or time_B < 1:
                    continue
                
                # B/A幅度比率
                amp_ratio = amp_B / max(amp_A, 1e-10)
                time_ratio = time_B / max(time_A, 1)
                
                # 判断是否为三角形态 (排除经典对称区间 0.85-1.18)
                # 过滤极端比率: 收敛不低于0.15, 发散不超过6.0
                is_converging = 0.15 < amp_ratio < 0.85
                is_diverging = 1.18 < amp_ratio < 6.0
                
                if not (is_converging or is_diverging):
                    continue  # 经典对称区间 → 已在上面的center预测中处理
                
                # === D. ModOnly预测 (附加) ===
                # 检查此AB对是否满足模长守恒条件
                # ModOnly与三角可以并存: 三角给出缩放直线目标, ModOnly给出弧线目标
                norm_amp_A = amp_A / max(global_amp, 1e-10)
                norm_time_A = time_A / max(global_span, 1)
                mod_A = math.sqrt(norm_amp_A**2 + norm_time_A**2)
                
                norm_amp_B = amp_B / max(global_amp, 1e-10)
                norm_time_B = time_B / max(global_span, 1)
                mod_B = math.sqrt(norm_amp_B**2 + norm_time_B**2)
                
                mod_sym_val = min(mod_A, mod_B) / max(mod_A, mod_B, 1e-10)
                amp_sym_val = 1.0 - abs(amp_A - amp_B) / max(amp_A, amp_B, 1e-10)
                
                if mod_sym_val > 0.80 and amp_sym_val < 0.85:
                    # 模长守恒但幅度不对称 → 附加弧线预测
                    mo_pred_dir = dir_A
                    mo_p3_price = seg_B['price_end']
                    mo_pred_target_price = mo_p3_price + mo_pred_dir * amp_A
                    mo_pred_target_bar = p3 + time_A
                    
                    if mo_pred_target_bar > p1:
                        mo_key = ('MO', p1, p2, p3, mo_pred_dir)
                        if mo_key not in seen:
                            seen.add(mo_key)
                            
                            mo_actual_C = None
                            mo_actual_progress = 0.0
                            for seg_R in start_at.get(p3, []):
                                dr = _seg_dir(seg_R)
                                if dr == mo_pred_dir and seg_R['bar_end'] > p3:
                                    if mo_actual_C is None or seg_R['bar_end'] > mo_actual_C['bar_end']:
                                        mo_actual_C = seg_R
                            if mo_actual_C:
                                mo_actual_progress = min(1.0, mo_actual_C['amplitude'] / max(amp_A, 1e-10))
                            
                            mo_imps = [pivot_info.get(b, {}).get('importance', 0) for b in [p1, p2, p3]]
                            mo_pred_imp = sum(mo_imps) / len(mo_imps)
                            mo_rel_amp = amp_A / max(global_amp, 1e-10)
                            
                            predictions.append({
                                'type': 'modonly',
                                'sym_type': 'modulus_conservation',
                                
                                'A_start': p1, 'A_end': p2,
                                'A_price_start': seg_A['price_start'],
                                'A_price_end': seg_A['price_end'],
                                'A_amp': amp_A, 'A_time': time_A, 'A_dir': dir_A,
                                
                                'B_start': p2, 'B_end': p3,
                                'B_price_start': seg_B['price_start'],
                                'B_price_end': seg_B['price_end'],
                                'B_amp': amp_B, 'B_time': time_B,
                                'retrace_ratio': round(amp_ratio, 4),
                                
                                'mod_A': round(mod_A, 6),
                                'mod_B': round(mod_B, 6),
                                'mod_sym': round(mod_sym_val, 4),
                                'amp_sym': round(amp_sym_val, 4),
                                
                                'pred_dir': mo_pred_dir,
                                'pred_start_bar': p3,
                                'pred_start_price': mo_p3_price,
                                'pred_target_price': mo_pred_target_price,
                                'pred_target_bar': mo_pred_target_bar,
                                'pred_amp': amp_A,
                                'pred_time': time_A,
                                
                                'actual_C': mo_actual_C,
                                'actual_progress': mo_actual_progress,
                                'importance': mo_pred_imp,
                                'rel_amp': mo_rel_amp,
                                'score': mo_pred_imp * mo_rel_amp * 0.7,
                            })
                
                # --- 三角预测 ---
                pred_dir = dir_A  # C方向同A
                p3_price = seg_B['price_end']
                
                # 等比缩放: amp_C = amp_B * ratio
                pred_amp_C = amp_B * amp_ratio
                pred_time_C = max(1, int(time_B * time_ratio))
                
                # 三角类型
                tri_type = 'converging' if is_converging else 'diverging'
                
                pred_target_price = p3_price + pred_dir * pred_amp_C
                pred_target_bar = p3 + pred_time_C
                
                if pred_target_bar <= p1:
                    continue
                
                key = ('T', p1, p2, p3, pred_dir, tri_type)
                if key in seen:
                    continue
                seen.add(key)
                
                # 实际C段
                actual_C = None
                actual_progress = 0.0
                for seg_R in start_at.get(p3, []):
                    dir_R = _seg_dir(seg_R)
                    if dir_R == pred_dir and seg_R['bar_end'] > p3:
                        if actual_C is None or seg_R['bar_end'] > actual_C['bar_end']:
                            actual_C = seg_R
                if actual_C:
                    actual_progress = min(1.0, actual_C['amplitude'] / max(pred_amp_C, 1e-10))
                
                imp_bars = [p1, p2, p3]
                imps = [pivot_info.get(b, {}).get('importance', 0) for b in imp_bars]
                pred_imp = sum(imps) / len(imps)
                rel_amp = pred_amp_C / max(global_amp, 1e-10)
                
                predictions.append({
                    'type': 'triangle',
                    'sym_type': tri_type,  # 'converging' or 'diverging'
                    
                    # 左臂A
                    'A_start': p1, 'A_end': p2,
                    'A_price_start': seg_A['price_start'],
                    'A_price_end': seg_A['price_end'],
                    'A_amp': amp_A, 'A_time': time_A, 'A_dir': dir_A,
                    
                    # 中间段B
                    'B_start': p2, 'B_end': p3,
                    'B_price_start': seg_B['price_start'],
                    'B_price_end': seg_B['price_end'],
                    'B_amp': amp_B, 'B_time': time_B,
                    
                    # 三角参数
                    'amp_ratio': round(amp_ratio, 4),       # B/A幅度比
                    'time_ratio': round(time_ratio, 4),     # B/A时间比
                    'tri_type': tri_type,
                    
                    # 预测C'
                    'pred_dir': pred_dir,
                    'pred_start_bar': p3,
                    'pred_start_price': p3_price,
                    'pred_target_price': pred_target_price,
                    'pred_target_bar': pred_target_bar,
                    'pred_amp': pred_amp_C,
                    'pred_time': pred_time_C,
                    
                    # 三角边界线 (用于可视化)
                    # 上边界: 连接 A段同方向端点
                    # 下边界: 连接 A段反方向端点
                    'boundary_top': [
                        [p1, seg_A['price_start'] if dir_A == 1 else seg_A['price_end']],
                        [p3, seg_B['price_start'] if dir_A == 1 else seg_B['price_end']],
                    ],
                    'boundary_bot': [
                        [p2, seg_A['price_end'] if dir_A == 1 else seg_A['price_start']],
                        [p3, seg_B['price_end'] if dir_A == 1 else seg_B['price_start']],
                    ],
                    
                    'actual_C': actual_C,
                    'actual_progress': actual_progress,
                    'importance': pred_imp,
                    'rel_amp': rel_amp,
                    'score': pred_imp * rel_amp * 0.85,  # 略低于经典对称
                })
    
    # 过滤和评分优化
    import math
    filtered = []
    for p in predictions:
        # 预测目标不能完全在早期历史中
        if p['pred_target_bar'] < current_bar * 0.3:
            continue
        
        # 时效性: 预测起点越接近current_bar越有价值
        time_dist = abs(current_bar - p['pred_start_bar'])
        recency = 1.0 / (1.0 + time_dist / max(current_bar * 0.15, 1))
        
        # 活跃度: 正在展开的预测加分
        activity = 1.0 + p['actual_progress'] * 2.0 if p['actual_progress'] > 0 else 1.0
        if p['actual_progress'] >= 0.95:
            activity *= 0.3
        
        # rel_amp用log平滑: 避免大线段绝对主导
        # log(1 + rel_amp*10) / log(11) 归一化到 [0, 1]
        log_amp = math.log(1 + p['rel_amp'] * 10) / math.log(11)
        
        # 综合score
        p['recency'] = round(recency, 4)
        p['activity'] = round(activity, 3)
        p['score'] = round(p['importance'] * log_amp * recency * activity, 6)
        
        filtered.append(p)
    
    filtered.sort(key=lambda p: -p['score'])
    return filtered


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
    
    # ===== 对称结构识别 (旧版, 保留兼容) =====
    print("\n" + "=" * 70)
    print("对称结构识别 — 5维对称向量 (v3.2旧版)")
    print("=" * 70)
    
    t0 = time.time()
    sym_structures = find_symmetric_structures(full_pool, pivot_info, df=df, top_n=200)
    t1 = time.time()
    
    print(f"\n发现 {len(sym_structures)} 个对称结构 ({t1-t0:.3f}s)")
    
    if sym_structures:
        types = {}
        for s in sym_structures:
            t = s['type']
            types[t] = types.get(t, 0) + 1
        print(f"类型分布: {types}")
        high_sym = [s for s in sym_structures if s['sym_score'] > 0.8]
        print(f"高对称度(>0.8): {len(high_sym)}个")
    
    # ===== 统一对称谱 (v3.3新版) =====
    print("\n" + "=" * 70)
    print("统一对称谱 — 多维特征向量 (v3.3)")
    print("=" * 70)
    
    t0 = time.time()
    spectra = compute_symmetry_spectrum(full_pool, pivot_info, max_pool_size=800)
    t1 = time.time()
    
    n_mirror = sum(1 for s in spectra if s['type'] == 'mirror')
    n_center = sum(1 for s in spectra if s['type'] == 'center')
    print(f"\n对称谱: {len(spectra)} 个 (mirror:{n_mirror}, center:{n_center}) {t1-t0:.3f}s")
    
    # 维度分布统计
    if spectra:
        import numpy as np
        amp_ratios = [s['amp_ratio'] for s in spectra]
        time_ratios = [s['time_ratio'] for s in spectra]
        mod_ratios = [s['mod_ratio'] for s in spectra]
        
        print(f"\namp_ratio 分布: mean={np.mean(amp_ratios):.2f} med={np.median(amp_ratios):.2f} "
              f"std={np.std(amp_ratios):.2f} [0.8~1.2]={sum(1 for r in amp_ratios if 0.8<=r<=1.2)}")
        print(f"time_ratio分布: mean={np.mean(time_ratios):.2f} med={np.median(time_ratios):.2f} "
              f"std={np.std(time_ratios):.2f} [0.8~1.2]={sum(1 for r in time_ratios if 0.8<=r<=1.2)}")
        print(f"mod_ratio 分布: mean={np.mean(mod_ratios):.2f} med={np.median(mod_ratios):.2f} "
              f"std={np.std(mod_ratios):.2f} [0.8~1.2]={sum(1 for r in mod_ratios if 0.8<=r<=1.2)}")
    
    # Top 30
    print(f"\nTop 30 对称谱:")
    print(f"  {'score':>7s} {'sym':>5s} {'imp':>5s} | {'type':>7s} | {'a_rat':>5s} {'t_rat':>5s} {'m_rat':>5s} {'s_rat':>5s} {'c_dif':>4s} | "
          f"L: start→end | R: start→end | center")
    for s in spectra[:30]:
        ctr = ''
        if s['type'] == 'center':
            ctr = f"bar{s['center_bar_start']}→{s['center_bar_end']}"
        elif s['type'] == 'mirror':
            ctr = f"@bar{s['pivot_bar']}"
        dir_sym = '↑' if s['dir_L'] == 1 else '↓'
        print(f"  {s['score']:.5f} {s['sym_closeness']:.3f} {s['mean_imp']:.3f} | "
              f"{s['type']:>7s}{dir_sym} | "
              f"{s['amp_ratio']:5.2f} {s['time_ratio']:5.2f} {s['mod_ratio']:5.2f} {s['slope_ratio']:5.2f} {s['complexity_diff']:+4d} | "
              f"bar{s['L_start']:3d}→{s['L_end']:3d} | bar{s['R_start']:3d}→{s['R_end']:3d} | {ctr}")
    
    # 验证用户给的例子
    print(f"\n--- 用户例子验证 ---")
    
    # 例1: L2(bar7)→H7(bar18) / L4(bar39)→H17(bar46), 以H7(18)→L4(39)为中心
    example1 = [s for s in spectra if s['type']=='center' 
                and s['L_start']==7 and s['L_end']==18 
                and s['R_start']==39 and s['R_end']==46]
    if example1:
        e = example1[0]
        print(f"\n例1 L2H7-L4H17 (中心对称H7→L4):")
        print(f"  amp_ratio={e['amp_ratio']:.3f} time_ratio={e['time_ratio']:.3f} "
              f"mod_ratio={e['mod_ratio']:.3f} slope_ratio={e['slope_ratio']:.3f}")
        print(f"  complexity: L={e['L_complexity']} R={e['R_complexity']} diff={e['complexity_diff']}")
        print(f"  → 幅度近似对称({e['amp_ratio']:.3f}), 时间不对称({e['time_ratio']:.3f}), "
              f"模长{'对称' if 0.8<e['mod_ratio']<1.2 else '不对称'}({e['mod_ratio']:.3f})")
    else:
        print("\n例1 L2H7-L4H17: 未在搜索池中找到 (可能重要性不够高)")
    
    # 例2: L4(39)→H17(46) / L15(47)→H3(52), 以H17(46)→L15(47)为中心  
    example2 = [s for s in spectra if s['type']=='center'
                and s['L_start']==39 and s['L_end']==46
                and s['R_start']==47 and s['R_end']==52]
    if example2:
        e = example2[0]
        print(f"\n例2 L4H17-L15H3 (中心对称H17→L15):")
        print(f"  amp_ratio={e['amp_ratio']:.3f} time_ratio={e['time_ratio']:.3f} "
              f"mod_ratio={e['mod_ratio']:.3f} slope_ratio={e['slope_ratio']:.3f}")
        print(f"  complexity: L={e['L_complexity']} R={e['R_complexity']} diff={e['complexity_diff']}")
        print(f"  → L是3段折叠(L4H27L22H17)的整体, R是1段直达(L15H3)")
    else:
        print("\n例2 L4H17-L15H3: 未在搜索池中找到")

if __name__ == '__main__':
    main()
