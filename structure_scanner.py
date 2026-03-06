#!/usr/bin/env python3
"""
Structure Scanner — 自动枚举zigzag结构 + 关联未来走势

Pipeline:
  1. 滑窗扫描: 在每个merge层级上枚举3段/5段连续结构
  2. 几何指纹: dir_seq, retrace_ratios, amp_ratios, time_ratios, symmetry
  3. 未来走势: 结构结束后 H 根K线的 direction/magnitude/drawdown
  4. 聚类分析: 按几何指纹聚类, 统计每个簇的预测力
  5. 输出: 高预测力形态

用法:
  python3 structure_scanner.py               # 默认EURUSD H1, 全量
  python3 structure_scanner.py --bars 5000   # 只用最近5000根
"""

import sys, os, json, time, argparse
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from merge_engine_v3 import (
    calculate_base_zg,
    full_merge_engine,
    compute_pivot_importance,
)


# =============================================================================
# 1. 数据加载
# =============================================================================

def load_klines(pair='EURUSD', tf='H1', max_bars=None):
    """加载K线数据"""
    path = f'/home/ubuntu/DataBase/base_kline/{pair}_{tf}.csv'
    df = pd.read_csv(path, sep='\t')
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
    if max_bars:
        df = df.tail(max_bars).reset_index(drop=True)
    return df


# =============================================================================
# 2. 结构枚举
# =============================================================================

def enumerate_structures(snapshots, df, pivot_info, horizon_bars=[5, 10, 20, 50]):
    """
    在每个snapshot层级上, 枚举3段和5段连续结构.
    
    对每个结构:
      - 计算几何指纹 (direction sequence, amplitude ratios, time ratios, retrace)
      - 关联未来走势 (各horizon的方向/幅度/回撤)
    
    Returns: list of structure dicts
    """
    total_bars = len(df)
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    structures = []
    
    for snap_type, snap_label, pivots in snapshots:
        n_pivots = len(pivots)
        if n_pivots < 4:  # need at least 4 pivots for 3 segments
            continue
        
        # Build segments from consecutive pivots
        segments = []
        for j in range(n_pivots - 1):
            b1, p1, d1 = pivots[j]
            b2, p2, d2 = pivots[j + 1]
            segments.append({
                'b1': int(b1), 'p1': float(p1), 'd1': int(d1),
                'b2': int(b2), 'p2': float(p2), 'd2': int(d2),
                'amp': abs(p2 - p1),
                'span': int(b2 - b1),
                'dir': 1 if p2 > p1 else -1,
            })
        
        # Enumerate 3-segment windows
        for i in range(len(segments) - 2):
            s3 = segments[i:i+3]
            struct = _build_structure(s3, 3, snap_label, df, pivot_info,
                                     highs, lows, closes, total_bars, horizon_bars)
            if struct:
                structures.append(struct)
        
        # Enumerate 5-segment windows
        for i in range(len(segments) - 4):
            s5 = segments[i:i+5]
            struct = _build_structure(s5, 5, snap_label, df, pivot_info,
                                     highs, lows, closes, total_bars, horizon_bars)
            if struct:
                structures.append(struct)
    
    return structures


def _build_structure(segs, n_legs, source_label, df, pivot_info,
                     highs, lows, closes, total_bars, horizon_bars):
    """Build a single structure dict with geometric fingerprint + future outcome."""
    
    start_bar = segs[0]['b1']
    end_bar = segs[-1]['b2']
    
    if end_bar >= total_bars - max(horizon_bars):
        return None  # not enough future data
    
    # --- Geometric fingerprint ---
    dir_seq = ''.join('U' if s['dir'] > 0 else 'D' for s in segs)
    
    amps = [s['amp'] for s in segs]
    spans = [s['span'] for s in segs]
    total_amp = abs(segs[-1]['p2'] - segs[0]['p1'])
    total_span = end_bar - start_bar
    
    # Amplitude ratios (each leg / total)
    amp_ratios = [a / total_amp if total_amp > 0 else 0 for a in amps]
    
    # Time ratios (each leg / total)
    time_ratios = [s / total_span if total_span > 0 else 0 for s in spans]
    
    # Retrace ratios (for 3-seg: how much does B retrace A)
    retrace_ratios = []
    for j in range(1, len(segs)):
        if amps[j-1] > 0:
            retrace_ratios.append(amps[j] / amps[j-1])
        else:
            retrace_ratios.append(0)
    
    # Symmetry: |amp_first - amp_last| / avg
    sym_score = abs(amps[0] - amps[-1]) / (np.mean(amps) + 1e-10)
    
    # Slope ratios (amp/time for each leg)
    slopes = [a / max(s, 1) for a, s in zip(amps, spans)]
    
    # Endpoint importance
    imp_start = pivot_info.get(start_bar, {}).get('importance', 0)
    imp_end = pivot_info.get(end_bar, {}).get('importance', 0)
    
    # Modulus of each leg
    moduli = [np.sqrt(s['span']**2 + (s['amp'] * 100000)**2) for s in segs]
    mod_ratios = [m / max(moduli) if max(moduli) > 0 else 0 for m in moduli]
    
    # --- Future outcome ---
    outcomes = {}
    end_price = closes[end_bar] if end_bar < total_bars else segs[-1]['p2']
    
    for h in horizon_bars:
        fut_end = min(end_bar + h, total_bars - 1)
        if fut_end <= end_bar:
            continue
        
        fut_closes = closes[end_bar:fut_end + 1]
        fut_highs = highs[end_bar:fut_end + 1]
        fut_lows = lows[end_bar:fut_end + 1]
        
        fut_max = np.max(fut_highs)
        fut_min = np.min(fut_lows)
        fut_close = closes[fut_end]
        
        move = fut_close - end_price
        max_up = fut_max - end_price
        max_down = end_price - fut_min
        
        # Direction: +1 up, -1 down, 0 flat (threshold: 5 pips)
        threshold = 0.0005  # 5 pips
        direction = 1 if move > threshold else (-1 if move < -threshold else 0)
        
        outcomes[f'H{h}'] = {
            'move': round(move, 6),
            'move_pips': round(move * 10000, 1),
            'direction': direction,
            'max_up_pips': round(max_up * 10000, 1),
            'max_down_pips': round(max_down * 10000, 1),
            'max_favorable': round(max_up * 10000, 1) if segs[-1]['dir'] > 0 else round(max_down * 10000, 1),
            'max_adverse': round(max_down * 10000, 1) if segs[-1]['dir'] > 0 else round(max_up * 10000, 1),
        }
    
    # --- Merge level category ---
    # L0 = base, A1-A2 = low, A3-A5 = mid, A6+ / T* = high
    def _level_cat(label):
        if label == 'L0': return 'B'
        if label.startswith('A'):
            n = int(label[1:])
            if n <= 2: return 'L'
            if n <= 5: return 'M'
            return 'H'
        if label.startswith('T'): return 'H'
        return 'B'
    level_cat = _level_cat(source_label)
    
    # --- Fingerprint string (for grouping) ---
    # Coarse quantization: 4 bins for retrace (fewer combos)
    def q_retrace(v):
        """Quantize retrace to 4 buckets: shallow/normal/deep/extreme"""
        if v < 0.382: return 'S'   # shallow
        if v < 0.786: return 'N'   # normal (38-78%)
        if v < 1.5:   return 'D'   # deep (78-150%)
        return 'X'                  # extreme (>150%)
    
    # Coarse amp ratio: 3 bins (small/medium/large relative to total)
    def q_amp(v):
        if v < 0.25: return 's'
        if v < 0.6:  return 'm'
        return 'L'
    
    retrace_q = ''.join(q_retrace(r) for r in retrace_ratios)
    amp_q = ''.join(q_amp(r) for r in amp_ratios)
    
    fingerprint = f"{dir_seq}|{retrace_q}|{amp_q}|{level_cat}"
    
    return {
        'source': source_label,
        'level_cat': level_cat,
        'n_legs': n_legs,
        'start_bar': start_bar,
        'end_bar': end_bar,
        'dir_seq': dir_seq,
        'fingerprint': fingerprint,
        'total_amp': round(total_amp, 6),
        'total_span': total_span,
        'amps': [round(a, 6) for a in amps],
        'spans': spans,
        'amp_ratios': [round(r, 4) for r in amp_ratios],
        'time_ratios': [round(r, 4) for r in time_ratios],
        'retrace_ratios': [round(r, 4) for r in retrace_ratios],
        'slopes': [round(s, 8) for s in slopes],
        'mod_ratios': [round(r, 4) for r in mod_ratios],
        'sym_score': round(sym_score, 4),
        'imp_start': round(imp_start, 4),
        'imp_end': round(imp_end, 4),
        'imp_product': round(imp_start * imp_end, 4),
        'outcomes': outcomes,
    }


# =============================================================================
# 3. 聚类分析 + 预测力统计
# =============================================================================

def analyze_predictive_power(structures, min_samples=10):
    """
    按fingerprint分组, 统计每组的预测力.
    
    Returns: list of cluster dicts, sorted by prediction strength
    """
    # Group by fingerprint
    groups = defaultdict(list)
    for s in structures:
        groups[s['fingerprint']].append(s)
    
    clusters = []
    for fp, members in groups.items():
        if len(members) < min_samples:
            continue
        
        # Aggregate outcomes for each horizon
        horizons = set()
        for m in members:
            horizons.update(m['outcomes'].keys())
        
        cluster = {
            'fingerprint': fp,
            'n_samples': len(members),
            'dir_seq': members[0]['dir_seq'],
            'n_legs': members[0]['n_legs'],
            'sources': list(set(m['source'] for m in members)),
            'avg_imp': round(np.mean([m['imp_product'] for m in members]), 4),
            'avg_sym': round(np.mean([m['sym_score'] for m in members]), 4),
            'avg_retrace': [round(np.mean([m['retrace_ratios'][i] for m in members if i < len(m['retrace_ratios'])]), 4)
                           for i in range(max(len(m['retrace_ratios']) for m in members))],
        }
        
        # Per-horizon statistics
        horizon_stats = {}
        for h in sorted(horizons):
            moves = [m['outcomes'][h]['move_pips'] for m in members if h in m['outcomes']]
            dirs = [m['outcomes'][h]['direction'] for m in members if h in m['outcomes']]
            fav = [m['outcomes'][h]['max_favorable'] for m in members if h in m['outcomes']]
            adv = [m['outcomes'][h]['max_adverse'] for m in members if h in m['outcomes']]
            
            if len(moves) < min_samples:
                continue
            
            n_up = sum(1 for d in dirs if d > 0)
            n_down = sum(1 for d in dirs if d < 0)
            n_flat = sum(1 for d in dirs if d == 0)
            n_total = len(dirs)
            
            # Continuation: does price continue in the direction of the last leg?
            last_dir = 1 if members[0]['dir_seq'][-1] == 'U' else -1
            n_continue = sum(1 for d in dirs if d == last_dir)
            n_reverse = sum(1 for d in dirs if d == -last_dir)
            
            continuation_rate = n_continue / n_total if n_total > 0 else 0
            reversal_rate = n_reverse / n_total if n_total > 0 else 0
            
            # Prediction strength = |continuation_rate - 0.5| * 2
            # 0 = random, 1 = perfect prediction
            pred_strength = abs(continuation_rate - 0.5) * 2
            
            # Edge: avg favorable - avg adverse (risk-reward)
            avg_fav = np.mean(fav) if fav else 0
            avg_adv = np.mean(adv) if adv else 0
            edge = avg_fav - avg_adv
            
            horizon_stats[h] = {
                'n': n_total,
                'avg_move': round(np.mean(moves), 2),
                'std_move': round(np.std(moves), 2),
                'median_move': round(np.median(moves), 2),
                'up_rate': round(n_up / n_total, 4),
                'down_rate': round(n_down / n_total, 4),
                'flat_rate': round(n_flat / n_total, 4),
                'continuation_rate': round(continuation_rate, 4),
                'reversal_rate': round(reversal_rate, 4),
                'pred_strength': round(pred_strength, 4),
                'avg_favorable': round(avg_fav, 2),
                'avg_adverse': round(avg_adv, 2),
                'edge_pips': round(edge, 2),
                'win_rate_if_continue': round(n_continue / n_total, 4),
            }
        
        cluster['horizon_stats'] = horizon_stats
        
        # Overall prediction strength (average across horizons)
        if horizon_stats:
            cluster['avg_pred_strength'] = round(
                np.mean([v['pred_strength'] for v in horizon_stats.values()]), 4)
            cluster['best_horizon'] = max(horizon_stats.keys(),
                                          key=lambda k: horizon_stats[k]['pred_strength'])
            cluster['best_pred_strength'] = horizon_stats[cluster['best_horizon']]['pred_strength']
            cluster['best_edge'] = horizon_stats[cluster['best_horizon']]['edge_pips']
        else:
            cluster['avg_pred_strength'] = 0
            cluster['best_horizon'] = None
            cluster['best_pred_strength'] = 0
            cluster['best_edge'] = 0
        
        clusters.append(cluster)
    
    # Sort by best prediction strength
    clusters.sort(key=lambda c: c['best_pred_strength'], reverse=True)
    return clusters


# =============================================================================
# 4. 滑窗运行: 在长序列上分段运行pipeline
# =============================================================================

def run_sliding_pipeline(df, window=500, step=200, horizon_bars=[5, 10, 20, 50]):
    """
    在整个数据上用滑动窗口运行merge_engine, 枚举结构, 收集样本.
    
    window=500: 每次处理500根K线
    step=200: 每次前进200根
    horizon_bars: 观察未来多远
    """
    total = len(df)
    all_structures = []
    seen_keys = set()  # dedup: (source_label, global_start_bar, global_end_bar)
    n_windows = 0
    n_dupes = 0
    
    highs = df['high'].values
    lows = df['low'].values
    
    for start in range(0, total - window - max(horizon_bars), step):
        end = start + window
        
        # Slice: window + future horizon for outcome measurement
        df_win = df.iloc[start:end + max(horizon_bars)].reset_index(drop=True)
        h = df_win['high'].values
        l = df_win['low'].values
        
        # Run merge engine on the window portion only
        base = calculate_base_zg(h[:window], l[:window], rb=0.5)
        if len(base) < 6:
            continue
        
        results = full_merge_engine(base)
        pi = compute_pivot_importance(results, high=h[:window], low=l[:window], total_bars=window)
        
        # Enumerate structures (using extended df for future outcomes)
        structs = enumerate_structures(
            results['all_snapshots'], df_win, pi, horizon_bars
        )
        
        # Offset bar indices to global + dedup
        for s in structs:
            s['start_bar'] += start
            s['end_bar'] += start
            s['window_start'] = start
            
            # Dedup key: same source level + same global bar range
            key = (s['source'], s['start_bar'], s['end_bar'])
            if key in seen_keys:
                n_dupes += 1
                continue
            seen_keys.add(key)
            all_structures.append(s)
        
        n_windows += 1
        
        if n_windows % 50 == 0:
            print(f"  Window {n_windows}: start={start}, structs={len(all_structures)}, dupes_skipped={n_dupes}")
    
    print(f"  Dedup: {n_dupes} duplicates removed, {len(all_structures)} unique structures")
    return all_structures


# =============================================================================
# 5. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Structure Scanner — auto-enumerate zigzag structures')
    parser.add_argument('--pair', default='EURUSD', help='Currency pair')
    parser.add_argument('--tf', default='H1', help='Timeframe')
    parser.add_argument('--bars', type=int, default=None, help='Max bars to use (None=all)')
    parser.add_argument('--window', type=int, default=500, help='Sliding window size')
    parser.add_argument('--step', type=int, default=200, help='Sliding window step')
    parser.add_argument('--min-samples', type=int, default=20, help='Min samples per cluster')
    parser.add_argument('--output', default=None, help='Output JSON path')
    args = parser.parse_args()
    
    print(f"=== Structure Scanner: {args.pair} {args.tf} ===")
    t0 = time.time()
    
    # Load data
    df = load_klines(args.pair, args.tf, args.bars)
    print(f"Loaded {len(df)} bars")
    
    # Run sliding pipeline
    print(f"Running sliding pipeline (window={args.window}, step={args.step})...")
    horizons = [5, 10, 20, 50]
    structures = run_sliding_pipeline(df, args.window, args.step, horizons)
    print(f"Total structures enumerated: {len(structures)}")
    
    # Split by n_legs
    s3 = [s for s in structures if s['n_legs'] == 3]
    s5 = [s for s in structures if s['n_legs'] == 5]
    print(f"  3-leg: {len(s3)}, 5-leg: {len(s5)}")
    
    # Level distribution
    from collections import Counter
    level_dist = Counter(s['level_cat'] for s in structures)
    print(f"  By level: {dict(level_dist)}")
    
    # Analyze predictive power
    print(f"\nAnalyzing predictive power (min_samples={args.min_samples})...")
    clusters_3 = analyze_predictive_power(s3, args.min_samples)
    clusters_5 = analyze_predictive_power(s5, args.min_samples)
    print(f"Clusters with >= {args.min_samples} samples: 3-leg={len(clusters_3)}, 5-leg={len(clusters_5)}")
    
    # Report top clusters
    def report_top(clusters, name, top_n=20):
        print(f"\n{'='*70}")
        print(f"TOP {top_n} {name} structures by prediction strength")
        print(f"{'='*70}")
        for i, c in enumerate(clusters[:top_n]):
            bh = c['best_horizon']
            bhs = c['horizon_stats'].get(bh, {})
            print(f"\n  [{i+1}] {c['fingerprint']}")
            print(f"      samples={c['n_samples']}, sources={c['sources'][:3]}")
            print(f"      avg_retrace={c['avg_retrace']}, sym={c['avg_sym']:.3f}, imp={c['avg_imp']:.4f}")
            print(f"      BEST: {bh} — pred_str={c['best_pred_strength']:.3f}, edge={c['best_edge']:.1f}pips")
            if bhs:
                print(f"        cont={bhs['continuation_rate']:.3f}, rev={bhs['reversal_rate']:.3f}")
                print(f"        avg_move={bhs['avg_move']:.1f}p, favorable={bhs['avg_favorable']:.1f}p, adverse={bhs['avg_adverse']:.1f}p")
            # Show H10 stats too
            h10 = c['horizon_stats'].get('H10', {})
            if h10:
                print(f"      H10: cont={h10['continuation_rate']:.3f}, edge={h10['edge_pips']:.1f}p, move={h10['avg_move']:.1f}p")
    
    report_top(clusters_3, '3-leg')
    report_top(clusters_5, '5-leg')
    
    elapsed = time.time() - t0
    print(f"\n\nCompleted in {elapsed:.1f}s")
    
    # Save results
    output_path = args.output or f'structure_scan_{args.pair}_{args.tf}.json'
    result = {
        'pair': args.pair,
        'tf': args.tf,
        'total_bars': len(df),
        'total_structures': len(structures),
        'n_3leg': len(s3),
        'n_5leg': len(s5),
        'n_clusters_3': len(clusters_3),
        'n_clusters_5': len(clusters_5),
        'horizons': horizons,
        'clusters_3': clusters_3[:50],  # top 50
        'clusters_5': clusters_5[:50],
        'elapsed_seconds': round(elapsed, 1),
    }
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
