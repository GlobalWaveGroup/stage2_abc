#!/usr/bin/env python3
"""
辅助线引擎 — T线 (TrendLine) + F线 (Fibonacci/Horizontal Line)

设计原则:
1. 只从「点」出发 — 不依赖zigzag归并结果，只需要拐点坐标
2. T线和F线互相独立
3. 每条线有 importance score（基于触碰次数、时间跨度、精度）
4. 输出可直接注入 v33 可视化

层级体系:
  点 (pivot) → 最基本信息
  幅度zig线 → 连接点的价格波动
  时间zig线 → 连接点的时间维度
  T线 → 连接同方向拐点的斜线（趋势线）
  F线 → 同价位拐点形成的水平参考（支撑/阻力）
  S线 → 结构间的对称关系
"""

import numpy as np
from collections import defaultdict


# ============================================================
# T线引擎 — 趋势线检测
# ============================================================

def detect_trendlines(pivots, high, low, min_touches=3, max_deviation=0.15,
                      max_lines=50, importance_weights=None):
    """
    从拐点集合中检测趋势线。
    
    算法:
    1. 分别处理 peaks (下降趋势线的触碰点) 和 valleys (上升趋势线的触碰点)
    2. 对每对拐点定义一条候选线
    3. 检查有多少其他同方向拐点落在这条线附近（触碰）
    4. 按 importance 排序，去重保留 top
    
    Args:
        pivots: list of (bar, price, direction)  direction: 1=peak, -1=valley
        high, low: K线数据
        min_touches: 至少触碰几个点才算趋势线
        max_deviation: 触碰判定的最大偏离（相对于线段平均amplitude）
        max_lines: 最多返回几条
        
    Returns:
        list of dict: 每条趋势线 {
            'type': 'support' | 'resistance',
            'points': [(bar, price), ...],  # 触碰点
            'slope': float,  # 斜率 (price/bar)
            'intercept': float,  # 截距
            'bar_start': int, 'bar_end': int,
            'n_touches': int,
            'importance': float,
            'deviation': float,  # 平均偏离
        }
    """
    if len(pivots) < min_touches:
        return []
    
    # 分离 peaks 和 valleys
    peaks = [(b, p) for b, p, d in pivots if d == 1]
    valleys = [(b, p) for b, p, d in pivots if d == -1]
    
    results = []
    
    # 上升趋势线（连接 valleys）= support lines
    support = _find_lines(valleys, 'support', high, low, min_touches, max_deviation)
    results.extend(support)
    
    # 下降趋势线（连接 peaks）= resistance lines
    resistance = _find_lines(peaks, 'resistance', high, low, min_touches, max_deviation)
    results.extend(resistance)
    
    # 按 importance 排序
    results.sort(key=lambda x: -x['importance'])
    
    # 去重: 如果两条线的触碰点高度重叠(>60%)，保留更重要的
    filtered = []
    for line in results:
        pts_set = set(line['point_bars'])
        is_dup = False
        for existing in filtered:
            existing_set = set(existing['point_bars'])
            overlap = len(pts_set & existing_set)
            min_len = min(len(pts_set), len(existing_set))
            if min_len > 0 and overlap / min_len > 0.6:
                is_dup = True
                break
        if not is_dup:
            filtered.append(line)
        if len(filtered) >= max_lines:
            break
    
    return filtered


def _find_lines(points, line_type, high, low, min_touches, max_deviation):
    """
    对一组同方向拐点寻找共线组合。
    
    使用多级容差: 先用严格容差找精确线，再用宽容差找模糊线。
    评分重视精度: precision^2 权重，让3个精确点胜过10个松散点。
    """
    n = len(points)
    if n < min_touches:
        return []
    
    # 计算参考振幅（用于偏差归一化）
    prices = [p for _, p in points]
    if len(prices) < 2:
        return []
    price_range = max(prices) - min(prices)
    if price_range == 0:
        price_range = 0.001
    
    candidates = []
    
    # 优化: 如果点太多，只用重要性高的子集
    if n > 200:
        step = max(1, n // 200)
        pts_sampled = points[::step]
    else:
        pts_sampled = points
    
    # 多级容差: 从严格到宽松
    dev_levels = [
        price_range * 0.01,   # 1% — 极精确
        price_range * 0.03,   # 3% — 精确
        price_range * 0.06,   # 6% — 中等
        price_range * max_deviation,  # 用户指定 — 宽松
    ]
    
    for i in range(len(pts_sampled)):
        for j in range(i + 1, len(pts_sampled)):
            b1, p1 = pts_sampled[i]
            b2, p2 = pts_sampled[j]
            
            if b1 == b2:
                continue
            
            slope = (p2 - p1) / (b2 - b1)
            intercept = p1 - slope * b1
            
            # 对每级容差分别检查
            for dev_threshold in dev_levels:
                touches = []
                deviations = []
                
                for b, p in points:
                    expected = slope * b + intercept
                    dev = abs(p - expected)
                    if dev <= dev_threshold:
                        touches.append((b, p))
                        deviations.append(dev)
                
                if len(touches) >= min_touches:
                    bar_start = min(b for b, _ in touches)
                    bar_end = max(b for b, _ in touches)
                    time_span = bar_end - bar_start
                    if time_span == 0:
                        continue
                    
                    avg_dev = np.mean(deviations) if deviations else 0
                    
                    # 精度评分: deviation越小越好, 使用平方权重
                    # 归一化到 [0, 1], 0=完全偏离, 1=完美拟合
                    precision = 1.0 / (1.0 + (avg_dev / price_range) * 50)
                    
                    # Importance = touches × sqrt(time_span) × precision²
                    # precision² 让精确的3点线远胜松散的10点线
                    importance = len(touches) * np.sqrt(time_span) * precision ** 2
                    
                    candidates.append({
                        'type': line_type,
                        'points': touches,
                        'point_bars': [b for b, _ in touches],
                        'slope': slope,
                        'intercept': intercept,
                        'bar_start': bar_start,
                        'bar_end': bar_end,
                        'time_span': time_span,
                        'n_touches': len(touches),
                        'importance': importance,
                        'deviation': avg_dev,
                        'precision': precision,
                        'dev_level': dev_threshold / price_range,
                    })
    
    return candidates


# ============================================================
# F线引擎 — 水平支撑/阻力线检测
# ============================================================

def detect_horizontal_lines(pivots, high, low, price_tolerance=None, 
                            min_touches=3, max_lines=30):
    """
    从拐点集合中检测水平支撑/阻力线。
    
    算法:
    1. 将所有拐点价格收集
    2. 用价格聚类找到水平聚集区
    3. 对每个聚集区计算触碰次数和importance
    
    Args:
        pivots: list of (bar, price, direction)
        high, low: K线数据
        price_tolerance: 价格容差（同一水平线的偏差范围）。None=自动计算
        min_touches: 至少触碰几次
        max_lines: 最多返回几条
        
    Returns:
        list of dict: 每条水平线 {
            'price': float,  # 水平价位
            'type': 'support' | 'resistance' | 'both',
            'touches': [(bar, price, direction), ...],
            'n_touches': int,
            'n_peaks': int, 'n_valleys': int,
            'bar_start': int, 'bar_end': int,
            'importance': float,
            'tightness': float,  # 触碰点的价格紧密度
        }
    """
    if len(pivots) < min_touches:
        return []
    
    # 自动计算容差: 基于全局价格范围的0.5%
    all_prices = [p for _, p, _ in pivots]
    price_range = max(all_prices) - min(all_prices)
    if price_tolerance is None:
        price_tolerance = price_range * 0.005  # 0.5% of range
    
    # 按价格排序所有拐点
    sorted_pivots = sorted(pivots, key=lambda x: x[1])
    
    # 扫描聚集区: 滑窗法
    lines = []
    used = set()  # 已分配到某条线的拐点索引
    
    for i in range(len(sorted_pivots)):
        if i in used:
            continue
        
        base_price = sorted_pivots[i][1]
        cluster = []
        
        for j in range(i, len(sorted_pivots)):
            if sorted_pivots[j][1] - base_price <= price_tolerance * 2:
                cluster.append((j, sorted_pivots[j]))
            else:
                break
        
        if len(cluster) < min_touches:
            continue
        
        # 计算聚集区的中心价位
        cluster_prices = [p[1][1] for p in cluster]
        center_price = np.median(cluster_prices)
        
        # 重新筛选: 以中心价位为基准，容差范围内的点
        final_cluster = []
        for idx, (b, p, d) in cluster:
            if abs(p - center_price) <= price_tolerance:
                final_cluster.append((b, p, d))
                used.add(idx)
        
        if len(final_cluster) < min_touches:
            continue
        
        n_peaks = sum(1 for _, _, d in final_cluster if d == 1)
        n_valleys = sum(1 for _, _, d in final_cluster if d == -1)
        
        if n_peaks > 0 and n_valleys > 0:
            line_type = 'both'  # 支撑阻力转换
        elif n_peaks >= n_valleys:
            line_type = 'resistance'
        else:
            line_type = 'support'
        
        bars = [b for b, _, _ in final_cluster]
        bar_start = min(bars)
        bar_end = max(bars)
        time_span = bar_end - bar_start
        
        # 价格紧密度
        price_std = np.std([p for _, p, _ in final_cluster])
        tightness = 1.0 / (1.0 + price_std / price_tolerance * 5)
        
        # Importance = touches × time_span × tightness × (S/R flip bonus)
        sr_flip_bonus = 1.5 if line_type == 'both' else 1.0
        importance = len(final_cluster) * np.sqrt(max(1, time_span)) * tightness * sr_flip_bonus
        
        lines.append({
            'price': center_price,
            'type': line_type,
            'touches': final_cluster,
            'touch_bars': bars,
            'n_touches': len(final_cluster),
            'n_peaks': n_peaks,
            'n_valleys': n_valleys,
            'bar_start': bar_start,
            'bar_end': bar_end,
            'time_span': time_span,
            'importance': importance,
            'tightness': tightness,
        })
    
    # 按 importance 排序
    lines.sort(key=lambda x: -x['importance'])
    
    return lines[:max_lines]


# ============================================================
# 统一接口
# ============================================================

def compute_auxiliary_lines(pivot_info, high, low, total_bars=None):
    """
    从 compute_pivot_importance 的输出计算所有辅助线。
    
    Args:
        pivot_info: dict from compute_pivot_importance 
                    {bar: {bar, price, dir, importance, ...}}
        high, low: K线数据
        total_bars: 总K线数（用于时间归一化）
    
    Returns:
        dict: {
            'trendlines': [...],
            'horizontals': [...],
        }
    """
    # 提取拐点列表 (bar, price, direction)
    pivots = [(v['bar'], v['price'], v['dir']) for v in pivot_info.values()]
    pivots.sort(key=lambda x: x[0])
    
    trendlines = detect_trendlines(pivots, high, low)
    horizontals = detect_horizontal_lines(pivots, high, low)
    
    return {
        'trendlines': trendlines,
        'horizontals': horizontals,
    }


def compute_auxiliary_lines_from_pivots(pivot_list, high, low, 
                                         importance_threshold=0.0,
                                         t_min_touches=3, f_min_touches=3,
                                         t_max_lines=20, f_max_lines=20):
    """
    直接从拐点列表计算辅助线（不需要完整的pivot_info）。
    
    Args:
        pivot_list: list of (bar, price, direction)
                    或 list of dict {bar, price, dir, importance}
        high, low: K线数据
        importance_threshold: 只用重要性 > 此阈值的拐点
    
    Returns: same as compute_auxiliary_lines
    """
    # 归一化输入
    pivots = []
    for p in pivot_list:
        if isinstance(p, dict):
            if p.get('importance', 1.0) >= importance_threshold:
                pivots.append((p['bar'], p['price'], p['dir']))
        elif isinstance(p, (list, tuple)):
            pivots.append((int(p[0]), float(p[1]), int(p[2])))
    
    pivots.sort(key=lambda x: x[0])
    
    trendlines = detect_trendlines(pivots, high, low, 
                                    min_touches=t_min_touches,
                                    max_lines=t_max_lines)
    horizontals = detect_horizontal_lines(pivots, high, low,
                                           min_touches=f_min_touches,
                                           max_lines=f_max_lines)
    
    return {
        'trendlines': trendlines,
        'horizontals': horizontals,
    }


# ============================================================
# 测试
# ============================================================

if __name__ == '__main__':
    import sys, os, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from merge_engine_v3 import calculate_base_zg, full_merge_engine, compute_pivot_importance, load_kline
    
    filepath = "/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv"
    limit = 500  # 和标注平台一致
    
    print(f"Loading {filepath} (last {limit} bars)...")
    df = load_kline(filepath, limit=limit)
    high = df['high'].values
    low = df['low'].values
    print(f"Bars: {len(df)}")
    
    # v3 pipeline
    t0 = time.time()
    pivots = calculate_base_zg(high, low)
    results = full_merge_engine(pivots)
    pi = compute_pivot_importance(results, high=high, low=low, total_bars=len(df))
    t1 = time.time()
    print(f"v3 pipeline: {t1-t0:.2f}s, {len(pi)} pivots")
    
    # 提取 top pivots (importance > median)
    imp_values = [v['importance'] for v in pi.values()]
    imp_median = np.median(imp_values)
    imp_p75 = np.percentile(imp_values, 75)
    
    # 用 top 25% 的拐点来检测辅助线
    top_pivots = [(v['bar'], v['price'], v['dir']) 
                  for v in pi.values() if v['importance'] >= imp_p75]
    top_pivots.sort(key=lambda x: x[0])
    print(f"Top 25% pivots: {len(top_pivots)} (importance >= {imp_p75:.4f})")
    
    # T线检测
    t0 = time.time()
    tlines = detect_trendlines(top_pivots, high, low, min_touches=3)
    t1 = time.time()
    print(f"\nT线 (趋势线): {len(tlines)} detected ({t1-t0:.2f}s)")
    for i, t in enumerate(tlines[:10]):
        pts_str = ','.join(f'b{b}' for b in t['point_bars'])
        print(f"  T{i+1}: {t['type']:>10s} | slope={t['slope']:.7f} | touches={t['n_touches']} | "
              f"bars {t['bar_start']}~{t['bar_end']} | imp={t['importance']:.2f} | pts=[{pts_str}]")
    
    # F线检测
    t0 = time.time()
    flines = detect_horizontal_lines(top_pivots, high, low, min_touches=3)
    t1 = time.time()
    print(f"\nF线 (水平线): {len(flines)} detected ({t1-t0:.2f}s)")
    for i, f in enumerate(flines[:10]):
        pts_str = ','.join(f'b{b}' for b in f['touch_bars'])
        print(f"  F{i+1}: {f['type']:>10s} | price={f['price']:.5f} | touches={f['n_touches']} "
              f"(P{f['n_peaks']}/V{f['n_valleys']}) | bars {f['bar_start']}~{f['bar_end']} | "
              f"imp={f['importance']:.2f} | pts=[{pts_str}]")
    
    # 验证: 颈线 H7-H3-H6 应该被检测到
    print(f"\n--- 验证: 你标注的颈线 H7(b6)-H3(b18)-H6(b22) ---")
    h7_price = high[6] if 6 < len(high) else None
    h3_price = high[18] if 18 < len(high) else None
    h6_price = high[22] if 22 < len(high) else None
    if h7_price and h3_price and h6_price:
        target_price = np.mean([h7_price, h3_price, h6_price])
        print(f"  Target price range: {target_price:.5f} (H7={h7_price:.5f}, H3={h3_price:.5f}, H6={h6_price:.5f})")
        for i, f in enumerate(flines):
            if abs(f['price'] - target_price) < 0.001:
                print(f"  MATCH: F{i+1} price={f['price']:.5f} touches={f['n_touches']}")
    
    # 验证: 趋势线 L4-L6-L11
    print(f"\n--- 验证: 你标注的趋势线 L4(b19)-L6(b39)-L11(b62) ---")
    for i, t in enumerate(tlines):
        if t['type'] == 'support':
            bars = set(t['point_bars'])
            if 19 in bars or 39 in bars or 62 in bars:
                matched = bars & {19, 39, 62}
                print(f"  PARTIAL MATCH T{i+1}: matched bars {matched}, all pts={t['point_bars']}")
