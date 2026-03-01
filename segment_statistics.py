#!/usr/bin/env python3
"""
Segment Statistics Engine — P1: 全量zigzag线段统计库

功能:
1. 对 EURUSD H1 全量 155K bars 跑 v3 pipeline
2. 从每个归并层级提取连续 3段/5段 组合
3. 计算归一化特征向量:
   - 单段: amp, time, slope, 归一化amp(相对ATR)
   - 3段: B/A retrace, C/A amp_ratio, C/A time_ratio, C/A mod_ratio, 方向序列
   - 5段: S3占比, S1/S5 ratio, 整体retrace, 各段比例
4. 统计分布(百分位)
5. Outcome记录: 每个组合之后 N 根K线的 zigzag 走势

输出: segment_stats_EURUSD_H1.npz
"""

import numpy as np
import pandas as pd
import time
import sys
import os
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_engine_v3 import (calculate_base_zg, full_merge_engine, 
                               build_segment_pool, compute_pivot_importance, load_kline)


def extract_consecutive_segments(snapshot_pivots, high, low, n_segs=3):
    """
    从一个快照的拐点序列中提取所有连续 n_segs 段组合。
    
    每段: (bar_start, bar_end, price_start, price_end, amp, time_bars, direction)
    每个组合: n_segs 段 + 归一化特征向量
    
    Returns: list of dicts
    """
    pivots = snapshot_pivots
    n = len(pivots)
    if n < n_segs + 1:
        return []
    
    combos = []
    for i in range(n - n_segs):
        segs = []
        valid = True
        for j in range(n_segs):
            p1 = pivots[i + j]
            p2 = pivots[i + j + 1]
            b1 = p1['bar']
            b2 = p2['bar']
            pr1 = p1['price']
            pr2 = p2['price']
            
            amp = pr2 - pr1
            time_bars = b2 - b1
            if time_bars <= 0:
                valid = False
                break
            direction = 1 if amp > 0 else -1
            slope = amp / time_bars if time_bars > 0 else 0
            
            segs.append({
                'b1': b1, 'b2': b2,
                'p1': pr1, 'p2': pr2,
                'amp': amp,
                'abs_amp': abs(amp),
                'time': time_bars,
                'dir': direction,
                'slope': slope,
                'mod': np.sqrt(amp**2 + (time_bars * 0.0001)**2),  # 归一化模长
            })
        
        if not valid:
            continue
        
        # 计算组合特征
        combo = {
            'start_bar': segs[0]['b1'],
            'end_bar': segs[-1]['b2'],
            'n_segs': n_segs,
            'segments': segs,
        }
        
        # 方向序列
        combo['dir_seq'] = ''.join('U' if s['dir'] > 0 else 'D' for s in segs)
        
        # 归一化特征
        if n_segs >= 3:
            A, B, C = segs[0], segs[1], segs[2]
            combo['BA_retrace'] = B['abs_amp'] / A['abs_amp'] if A['abs_amp'] > 0 else 0
            combo['CA_amp_ratio'] = C['abs_amp'] / A['abs_amp'] if A['abs_amp'] > 0 else 0
            combo['CA_time_ratio'] = C['time'] / A['time'] if A['time'] > 0 else 0
            
            # 模长比
            mod_A = np.sqrt(A['abs_amp']**2 + (A['time'] * A['abs_amp'] / 100)**2)
            mod_C = np.sqrt(C['abs_amp']**2 + (C['time'] * C['abs_amp'] / 100)**2)
            combo['CA_mod_ratio'] = mod_C / mod_A if mod_A > 0 else 0
            
            # 总体特征
            total_amp = sum(s['abs_amp'] for s in segs)
            combo['net_amp'] = segs[-1]['p2'] - segs[0]['p1']
            combo['total_amp'] = total_amp
            combo['total_time'] = segs[-1]['b2'] - segs[0]['b1']
            
            # A/C 对称度 (越接近1越对称)
            combo['AC_symmetry'] = min(A['abs_amp'], C['abs_amp']) / max(A['abs_amp'], C['abs_amp']) if max(A['abs_amp'], C['abs_amp']) > 0 else 0
        
        if n_segs >= 5:
            S3 = segs[2]
            combo['S3_dominance'] = S3['abs_amp'] / total_amp if total_amp > 0 else 0
            combo['S1S5_ratio'] = segs[4]['abs_amp'] / segs[0]['abs_amp'] if segs[0]['abs_amp'] > 0 else 0
            combo['S1S3_ratio'] = segs[2]['abs_amp'] / segs[0]['abs_amp'] if segs[0]['abs_amp'] > 0 else 0
            combo['S3S5_ratio'] = segs[4]['abs_amp'] / segs[2]['abs_amp'] if segs[2]['abs_amp'] > 0 else 0
        
        combos.append(combo)
    
    return combos


def classify_3seg(combo):
    """
    分类3段组合的结构类型。
    
    Returns: (category, subcategory) 
    """
    ba = combo['BA_retrace']
    ca = combo['CA_amp_ratio']
    sym = combo['AC_symmetry']
    d = combo['dir_seq']
    
    # 方向模式
    if d in ('UDU', 'DUD'):
        # A和C同向 — 这是调整-恢复型
        if ca < 0.5:
            # C远小于A — 衰竭
            sub = 'exhaustion'
        elif 0.5 <= ca < 0.85:
            # C < A — 收敛
            sub = 'convergent'
        elif 0.85 <= ca <= 1.15:
            # C ≈ A — 对称
            sub = 'symmetric'
        elif 1.15 < ca <= 2.0:
            # C > A — 扩张
            sub = 'expanding'
        else:
            # C >> A — 强趋势恢复
            sub = 'strong_continuation'
        
        # B的深度区分
        if ba < 0.236:
            b_type = 'shallow'
        elif ba < 0.5:
            b_type = 'moderate'
        elif ba < 0.786:
            b_type = 'deep'
        elif ba < 1.0:
            b_type = 'very_deep'
        else:
            b_type = 'over_retrace'
        
        # Fibo型判定
        FIBS = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000]
        fib_dist = min(abs(ba - f) for f in FIBS)
        fib_type = 'fib_precise' if fib_dist < 0.03 else ('fib_near' if fib_dist < 0.06 else 'non_fib')
        
        return ('retracement', sub, b_type, fib_type)
    
    elif d in ('UUD', 'DDU', 'UDD', 'DUU'):
        # 两段同向 + 一段反向 — 趋势型
        return ('trend', 'two_impulse', 'na', 'na')
    
    else:
        return ('other', d, 'na', 'na')


def classify_5seg(combo):
    """分类5段组合"""
    d = combo['dir_seq']
    s3_dom = combo.get('S3_dominance', 0)
    s1s5 = combo.get('S1S5_ratio', 0)
    
    if d in ('UDUDU', 'DUDUD'):
        # 交替型5浪
        if s3_dom > 0.35:
            wave_type = 'S3_dominant'  # 第3浪最强
        elif s3_dom < 0.15:
            wave_type = 'S3_weak'
        else:
            wave_type = 'balanced'
        
        # 1浪 vs 5浪
        if s1s5 > 1.5:
            end_type = 'S5_extension'
        elif s1s5 < 0.67:
            end_type = 'S5_truncated'
        else:
            end_type = 'S5_normal'
        
        return ('impulse_5', wave_type, end_type)
    else:
        return ('complex_5', d, 'na')


def compute_outcome(high, low, end_bar, horizons=[5, 10, 20, 50]):
    """
    计算 end_bar 之后的走势。
    
    Returns: dict with outcome stats for each horizon
    """
    n = len(high)
    outcomes = {}
    
    for h in horizons:
        if end_bar + h >= n:
            continue
        
        future_high = high[end_bar + 1: end_bar + h + 1]
        future_low = low[end_bar + 1: end_bar + h + 1]
        
        if len(future_high) == 0:
            continue
        
        max_up = max(future_high) - high[end_bar]
        max_down = low[end_bar] - min(future_low)
        close_change = (high[end_bar + h] + low[end_bar + h]) / 2 - (high[end_bar] + low[end_bar]) / 2
        
        outcomes[f'h{h}_max_up'] = max_up
        outcomes[f'h{h}_max_down'] = max_down
        outcomes[f'h{h}_close_chg'] = close_change
        outcomes[f'h{h}_direction'] = 1 if close_change > 0 else -1
        # 上下比
        outcomes[f'h{h}_up_down_ratio'] = max_up / max_down if max_down > 0 else 10.0
    
    return outcomes


def run_statistics(filepath, limit=0):
    """
    主函数: 对指定品种跑全量统计。
    """
    print(f'Loading {filepath}...')
    df = load_kline(filepath, limit=limit)
    print(f'Loaded: {len(df)} bars')
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    
    # Step 1: Base zigzag
    t0 = time.time()
    pivots = calculate_base_zg(high, low)
    print(f'Base ZG: {len(pivots)} pivots ({time.time()-t0:.2f}s)')
    
    # Step 2: Full merge
    t0 = time.time()
    results = full_merge_engine(pivots)
    print(f'Merge: {time.time()-t0:.2f}s, {len(results["all_snapshots"])} snapshots')
    
    # Step 3: 从每个快照提取连续组合
    # 将 pivot 列表转换为 [{bar, price}, ...] 格式
    all_3seg = []
    all_5seg = []
    
    t0 = time.time()
    for snap_type, label, pvt_list in results['all_snapshots']:
        # pvt_list 是 (bar, price, direction) 的 tuple 列表
        pivot_dicts = []
        for p in pvt_list:
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                pivot_dicts.append({'bar': int(p[0]), 'price': float(p[1]), 'dir': int(p[2])})
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                pivot_dicts.append({'bar': int(p[0]), 'price': float(p[1])})
            elif isinstance(p, dict):
                pivot_dicts.append(p)
        
        if len(pivot_dicts) < 4:
            continue
        
        # 3段组合
        combos_3 = extract_consecutive_segments(pivot_dicts, high, low, n_segs=3)
        for c in combos_3:
            c['source_level'] = label
            cat = classify_3seg(c)
            c['category'] = cat[0]
            c['subcategory'] = cat[1]
            c['b_type'] = cat[2]
            c['fib_type'] = cat[3]
            c['outcome'] = compute_outcome(high, low, c['end_bar'])
        all_3seg.extend(combos_3)
        
        # 5段组合 (仅对较高级别, 节省计算)
        if label not in ('L0',):  # 跳过L0的5段，太多太碎
            combos_5 = extract_consecutive_segments(pivot_dicts, high, low, n_segs=5)
            for c in combos_5:
                c['source_level'] = label
                cat = classify_5seg(c)
                c['category'] = cat[0]
                c['subcategory'] = cat[1]
                if len(cat) > 2:
                    c['end_type'] = cat[2]
                c['outcome'] = compute_outcome(high, low, c['end_bar'])
            all_5seg.extend(combos_5)
    
    elapsed = time.time() - t0
    print(f'Extracted: {len(all_3seg)} 3-seg combos, {len(all_5seg)} 5-seg combos ({elapsed:.2f}s)')
    
    return {
        'n_bars': len(df),
        'n_base_pivots': len(pivots),
        'n_snapshots': len(results['all_snapshots']),
        'snapshot_labels': [label for _, label, _ in results['all_snapshots']],
        'all_3seg': all_3seg,
        'all_5seg': all_5seg,
        'high': high,
        'low': low,
    }


def print_statistics(stats):
    """打印统计摘要"""
    all_3 = stats['all_3seg']
    all_5 = stats['all_5seg']
    
    print(f'\n{"="*70}')
    print(f'SEGMENT STATISTICS SUMMARY')
    print(f'{"="*70}')
    print(f'Bars: {stats["n_bars"]}, Base pivots: {stats["n_base_pivots"]}')
    print(f'3-seg combos: {len(all_3)}, 5-seg combos: {len(all_5)}')
    
    # === 3段统计 ===
    print(f'\n--- 3-Segment Statistics ---')
    
    # 按类别分布
    from collections import Counter
    cats = Counter((c['category'], c['subcategory']) for c in all_3)
    print(f'\nCategory distribution (top 15):')
    for (cat, sub), count in cats.most_common(15):
        pct = count / len(all_3) * 100
        print(f'  {cat}/{sub}: {count:>7d} ({pct:.1f}%)')
    
    # 按B深度
    b_types = Counter(c.get('b_type', 'na') for c in all_3 if c['category'] == 'retracement')
    print(f'\nB retrace depth (retracement only):')
    for bt, count in b_types.most_common():
        print(f'  {bt}: {count:>7d}')
    
    # Fibo精度
    fib_types = Counter(c.get('fib_type', 'na') for c in all_3 if c['category'] == 'retracement')
    print(f'\nFibo precision (retracement only):')
    for ft, count in fib_types.most_common():
        print(f'  {ft}: {count:>7d}')
    
    # 关键比率分布
    ba_vals = [c['BA_retrace'] for c in all_3 if c['category'] == 'retracement' and 0 < c['BA_retrace'] < 5]
    ca_vals = [c['CA_amp_ratio'] for c in all_3 if c['category'] == 'retracement' and 0 < c['CA_amp_ratio'] < 5]
    
    if ba_vals:
        ba = np.array(ba_vals)
        print(f'\nBA retrace distribution (N={len(ba)}):')
        for p in [5, 10, 25, 50, 75, 90, 95]:
            print(f'  P{p}: {np.percentile(ba, p):.4f}')
    
    if ca_vals:
        ca = np.array(ca_vals)
        print(f'\nCA amp ratio distribution (N={len(ca)}):')
        for p in [5, 10, 25, 50, 75, 90, 95]:
            print(f'  P{p}: {np.percentile(ca, p):.4f}')
    
    # === Outcome 统计 ===
    print(f'\n--- Outcome Statistics by Category ---')
    for (cat, sub), count in cats.most_common(8):
        subset = [c for c in all_3 if c['category'] == cat and c['subcategory'] == sub]
        h20_dirs = [c['outcome'].get('h20_direction', 0) for c in subset if 'h20_direction' in c.get('outcome', {})]
        if len(h20_dirs) > 100:
            up_pct = sum(1 for d in h20_dirs if d > 0) / len(h20_dirs) * 100
            # 最后一段的方向
            last_dirs = [c['segments'][-1]['dir'] for c in subset[:len(h20_dirs)]]
            # 续涨概率 (最后段=U → 后续也涨)
            cont = sum(1 for d, ld in zip(h20_dirs, last_dirs) if d == ld) / len(h20_dirs) * 100
            print(f'  {cat}/{sub} (N={len(h20_dirs)}): h20 up={up_pct:.1f}%, continuation={cont:.1f}%')
    
    # === 5段统计 ===
    if all_5:
        print(f'\n--- 5-Segment Statistics ---')
        cats5 = Counter((c['category'], c['subcategory']) for c in all_5)
        for (cat, sub), count in cats5.most_common(10):
            print(f'  {cat}/{sub}: {count:>7d}')
    
    print(f'\n{"="*70}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', default='EURUSD')
    parser.add_argument('--tf', default='H1')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    
    filepath = f'/home/ubuntu/DataBase/base_kline/{args.pair}_{args.tf}.csv'
    
    stats = run_statistics(filepath, limit=args.limit)
    print_statistics(stats)
    
    if args.save:
        outpath = f'segment_stats_{args.pair}_{args.tf}.pkl'
        # 不保存 high/low 原始数据，太大
        save_data = {k: v for k, v in stats.items() if k not in ('high', 'low')}
        with open(outpath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f'\nSaved to {outpath} ({os.path.getsize(outpath)/(1024*1024):.1f} MB)')
