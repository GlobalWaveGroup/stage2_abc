#!/usr/bin/env python3
"""
Scanner V2 — 双层滑窗: 外层定importance, 内层找形态

架构:
  外层大滑窗 (10K/20K/50K bars):
    - 跑 base_zg → merge_engine → compute_importance
    - 得到每个拐点的全局importance分数
    
  内层小窗口 (500 bars, 可配置):
    - 在外层窗口内滑动
    - 对每个百分位(10%/30%/50%/100%), 取importance达标的拐点
    - 在这些拐点上构建相邻3段结构, 提取fingerprint
    - 关联未来走势

  聚类:
    - 按 (外层大小, 百分位, fingerprint) 聚类
    - 统计每个簇的未来走势概率

用法:
  python3 scanner_v2.py --bars 20000
  python3 scanner_v2.py --outer 10000,20000 --inner 500 --pct 10,30,50,100
  python3 scanner_v2.py  # 全量, 默认参数
"""

import sys, os, json, time, argparse
from collections import defaultdict, Counter
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
    path = f'/home/ubuntu/DataBase/base_kline/{pair}_{tf}.csv'
    df = pd.read_csv(path, sep='\t')
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
    if max_bars:
        df = df.tail(max_bars).reset_index(drop=True)
    return df


# =============================================================================
# 2. 外层: 对一个大窗口跑importance
# =============================================================================

def compute_importance_map(h, l, n_bars):
    """
    对 n_bars 根K线跑 base_zg → merge → importance.
    返回 dict: { bar_index: importance_info_dict }
    """
    base = calculate_base_zg(h[:n_bars], l[:n_bars], rb=0.5)
    if len(base) < 10:
        return {}
    results = full_merge_engine(base)
    pi = compute_pivot_importance(results, high=h[:n_bars], low=l[:n_bars], total_bars=n_bars)
    return pi


# =============================================================================
# 3. 内层: 在小窗口内, 用importance筛选拐点, 构建结构
# =============================================================================

def scan_inner_window(pi, inner_start, inner_end, pct_thresholds,
                      closes, highs, lows, total_bars, horizon_bars):
    """
    在 [inner_start, inner_end) 范围内:
      - 取落在此范围的拐点
      - 对每个百分位阈值, 筛选importance达标的拐点
      - 构建3段结构

    pi: 外层importance map (bar -> info)
    pct_thresholds: dict { pct: importance_threshold }
    inner_start, inner_end: 内层窗口的bar范围
    
    Returns: list of structure dicts
    """
    max_horizon = max(horizon_bars)

    # 收集此窗口内的拐点, 按bar排序
    window_pivots = [(bar, info) for bar, info in pi.items()
                     if inner_start <= bar < inner_end]
    if len(window_pivots) < 4:
        return []
    window_pivots.sort(key=lambda x: x[0])

    structures = []

    for pct, threshold in pct_thresholds.items():
        # 筛选importance >= threshold的拐点
        filtered = [(bar, info) for bar, info in window_pivots
                    if info['importance'] >= threshold]
        if len(filtered) < 4:
            continue

        # HL交替
        alternating = enforce_alternation(filtered)
        if len(alternating) < 4:
            continue

        # 构建相邻3段结构
        for i in range(len(alternating) - 3):
            pts = alternating[i:i+4]
            struct = build_structure(
                pts, pct, closes, highs, lows, total_bars, horizon_bars
            )
            if struct:
                structures.append(struct)

    return structures


def enforce_alternation(ordered_pivots):
    """
    确保拐点序列严格HL交替。
    连续同方向: 保留importance更高的。
    """
    if len(ordered_pivots) < 2:
        return ordered_pivots

    result = [ordered_pivots[0]]

    for i in range(1, len(ordered_pivots)):
        bar_i, info_i = ordered_pivots[i]
        bar_prev, info_prev = result[-1]

        if info_i['dir'] != info_prev['dir']:
            result.append(ordered_pivots[i])
        else:
            if info_i['importance'] > info_prev['importance']:
                result[-1] = ordered_pivots[i]

    return result


def build_structure(pts, pct, closes, highs, lows, total_bars, horizon_bars):
    """从4个拐点构建3段结构, 计算fingerprint + outcome."""
    max_horizon = max(horizon_bars)

    bars = [p[0] for p in pts]
    prices = [p[1]['price'] for p in pts]
    dirs = [p[1]['dir'] for p in pts]
    imps = [p[1]['importance'] for p in pts]

    start_bar = bars[0]
    end_bar = bars[3]

    if end_bar + max_horizon >= total_bars:
        return None

    # 构建3段
    segs = []
    for j in range(3):
        b1, p1 = bars[j], prices[j]
        b2, p2 = bars[j+1], prices[j+1]
        amp = abs(p2 - p1)
        span = b2 - b1
        if span <= 0:
            return None
        seg_dir = 1 if p2 > p1 else -1
        segs.append({'amp': amp, 'span': span, 'dir': seg_dir})

    # --- Geometric fingerprint ---
    dir_seq = ''.join('U' if s['dir'] > 0 else 'D' for s in segs)

    amps = [s['amp'] for s in segs]
    spans = [s['span'] for s in segs]
    total_amp = abs(prices[3] - prices[0])
    total_span = end_bar - start_bar

    if total_amp < 1e-10 or total_span <= 0:
        return None

    # Retrace ratios
    retrace_ratios = []
    for j in range(1, 3):
        retrace_ratios.append(amps[j] / amps[j-1] if amps[j-1] > 0 else 0)

    # Amp ratios (each leg / total)
    amp_ratios = [a / total_amp for a in amps]

    # Time ratios
    time_ratios = [s / total_span for s in spans]

    # Symmetry
    sym_score = abs(amps[0] - amps[2]) / (np.mean(amps) + 1e-10)

    # --- Quantize fingerprint ---
    def q_retrace(v):
        if v < 0.382: return 'S'
        if v < 0.786: return 'N'
        if v < 1.5:   return 'D'
        return 'X'

    def q_amp(v):
        if v < 0.25: return 's'
        if v < 0.6:  return 'm'
        return 'L'

    retrace_q = ''.join(q_retrace(r) for r in retrace_ratios)
    amp_q = ''.join(q_amp(r) for r in amp_ratios)

    fingerprint = f"{dir_seq}|{retrace_q}|{amp_q}"

    # --- Future outcome ---
    outcomes = {}
    end_price = closes[end_bar]

    for horizon in horizon_bars:
        fut_end = min(end_bar + horizon, total_bars - 1)
        if fut_end <= end_bar:
            continue

        fut_highs = highs[end_bar:fut_end + 1]
        fut_lows = lows[end_bar:fut_end + 1]
        fut_close = closes[fut_end]

        move = fut_close - end_price
        max_up = np.max(fut_highs) - end_price
        max_down = end_price - np.min(fut_lows)

        threshold = 0.0005  # 5 pips
        direction = 1 if move > threshold else (-1 if move < -threshold else 0)

        last_dir = segs[-1]['dir']
        max_favorable = max_up if last_dir > 0 else max_down
        max_adverse = max_down if last_dir > 0 else max_up

        outcomes[f'H{horizon}'] = {
            'move_pips': round(move * 10000, 1),
            'direction': direction,
            'max_favorable': round(max_favorable * 10000, 1),
            'max_adverse': round(max_adverse * 10000, 1),
        }

    # 跨度分档: 3段结构总共跨多少根K线
    def _span_bucket(span):
        if span <= 10: return 'S10'
        if span <= 20: return 'S20'
        if span <= 50: return 'S50'
        if span <= 100: return 'S100'
        if span <= 200: return 'S200'
        return 'S200+'
    span_bkt = _span_bucket(total_span)

    return {
        'pct': pct,
        'fingerprint': fingerprint,
        'dir_seq': dir_seq,
        'start_bar': start_bar,
        'end_bar': end_bar,
        'total_amp': round(total_amp, 6),
        'total_span': total_span,
        'span_bucket': span_bkt,
        'retrace_ratios': [round(r, 4) for r in retrace_ratios],
        'amp_ratios': [round(r, 4) for r in amp_ratios],
        'time_ratios': [round(r, 4) for r in time_ratios],
        'sym_score': round(sym_score, 4),
        'avg_imp': round(np.mean(imps), 4),
        'min_imp': round(min(imps), 4),
        'outcomes': outcomes,
    }


# =============================================================================
# 4. 双层滑窗驱动
# =============================================================================

def run_dual_window(df, outer_sizes=[10000, 20000, 50000],
                    inner_size=500, inner_step=200,
                    percentiles=[10, 20, 30, 50, 75, 100],
                    horizon_bars=[5, 10, 20, 50],
                    outer_step_ratio=0.5):
    """
    双层滑窗:
      外层 (outer_sizes): 每个大小跑一次 merge+importance
      内层 (inner_size, inner_step): 在外层内滑动, 找形态

    Returns: dict { (outer_size, pct): [structures] }
    """
    total = len(df)
    max_horizon = max(horizon_bars)
    h_arr = df['high'].values
    l_arr = df['low'].values
    c_arr = df['close'].values

    all_structures = defaultdict(list)
    seen_keys = set()

    for outer in outer_sizes:
        outer_step = int(outer * outer_step_ratio)
        n_outer = 0
        n_structs_total = 0
        n_dupes_total = 0

        print(f"\n  outer={outer}, outer_step={outer_step}, inner={inner_size}, inner_step={inner_step}")

        for o_start in range(0, total - outer - max_horizon, outer_step):
            o_end = o_start + outer  # 外层窗口的bar范围 [o_start, o_end)

            # --- 外层: 跑importance ---
            pi = compute_importance_map(
                h_arr[o_start:o_end + max_horizon],
                l_arr[o_start:o_end + max_horizon],
                outer  # 只用前outer根算importance
            )
            if not pi:
                continue

            # importance分数分布 → 各百分位的阈值
            all_imps = sorted([info['importance'] for info in pi.values()], reverse=True)
            n_pivots = len(all_imps)
            pct_thresholds = {}
            for pct in percentiles:
                idx = max(0, min(int(n_pivots * pct / 100) - 1, n_pivots - 1))
                pct_thresholds[pct] = all_imps[idx]

            # 把pi的bar index偏移到全局坐标
            # pi的key是相对于外层窗口的 (0 ~ outer-1)
            # 需要 +o_start 变成全局坐标
            pi_global = {}
            for bar_local, info in pi.items():
                pi_global[bar_local + o_start] = info

            # --- 内层: 滑动小窗口找形态 ---
            n_inner = 0
            for i_start in range(o_start, o_end - inner_size, inner_step):
                i_end = i_start + inner_size

                structs = scan_inner_window(
                    pi_global, i_start, i_end, pct_thresholds,
                    c_arr, h_arr, l_arr, total, horizon_bars
                )

                # 去重 (同一个outer_size + pct + 全局start/end bar)
                for s in structs:
                    key = (outer, s['pct'], s['start_bar'], s['end_bar'])
                    if key in seen_keys:
                        n_dupes_total += 1
                        continue
                    seen_keys.add(key)
                    s['outer_size'] = outer
                    all_structures[(outer, s['pct'])].append(s)
                    n_structs_total += 1

                n_inner += 1

            n_outer += 1
            if n_outer % 5 == 0:
                print(f"    outer_win {n_outer}: o_start={o_start}, pivots={len(pi)}, structs={n_structs_total}, dupes={n_dupes_total}")

        print(f"    Done: {n_outer} outer windows, {n_structs_total} structures, {n_dupes_total} dupes skipped")

    return all_structures


# =============================================================================
# 5. 聚类分析 — 按 (pct, span_bucket, fingerprint) 三维聚类
# =============================================================================

SPAN_BUCKETS = ['S10', 'S20', 'S50', 'S100', 'S200', 'S200+']

def analyze_clusters(structures_by_group, min_samples=30):
    """
    把所有outer_size的结构合并, 按 (pct, span_bucket, fingerprint) 聚类.
    外层大小不再作为分组维度(已验证影响不大).
    """
    # 合并所有outer_size的结构, 只按pct分
    by_pct = defaultdict(list)
    for (outer, pct), structs in structures_by_group.items():
        by_pct[pct].extend(structs)

    all_clusters = []

    for pct in sorted(by_pct.keys()):
        structs = by_pct[pct]
        # 按 (span_bucket, fingerprint) 分组
        groups = defaultdict(list)
        for s in structs:
            groups[(s['span_bucket'], s['fingerprint'])].append(s)

        for (sb, fp), members in groups.items():
            if len(members) < min_samples:
                continue

            horizons = set()
            for m in members:
                horizons.update(m['outcomes'].keys())

            cluster = {
                'pct': pct,
                'span_bucket': sb,
                'fingerprint': fp,
                'n_samples': len(members),
                'dir_seq': members[0]['dir_seq'],
                'avg_span': round(np.mean([m['total_span'] for m in members]), 1),
                'avg_amp_pips': round(np.mean([m['total_amp'] for m in members]) * 10000, 1),
                'avg_imp': round(np.mean([m['avg_imp'] for m in members]), 4),
                'avg_sym': round(np.mean([m['sym_score'] for m in members]), 4),
                'avg_retrace': [
                    round(np.mean([m['retrace_ratios'][i] for m in members]), 4)
                    for i in range(2)
                ],
            }

            horizon_stats = {}
            for h_key in sorted(horizons):
                moves = [m['outcomes'][h_key]['move_pips'] for m in members if h_key in m['outcomes']]
                dirs = [m['outcomes'][h_key]['direction'] for m in members if h_key in m['outcomes']]
                fav = [m['outcomes'][h_key]['max_favorable'] for m in members if h_key in m['outcomes']]
                adv = [m['outcomes'][h_key]['max_adverse'] for m in members if h_key in m['outcomes']]

                if len(moves) < min_samples:
                    continue

                n_total = len(dirs)
                last_dir = 1 if members[0]['dir_seq'][-1] == 'U' else -1
                n_continue = sum(1 for d in dirs if d == last_dir)
                n_reverse = sum(1 for d in dirs if d == -last_dir)

                cont_rate = n_continue / n_total
                rev_rate = n_reverse / n_total
                pred_strength = abs(cont_rate - 0.5) * 2

                avg_fav = np.mean(fav)
                avg_adv = np.mean(adv)
                edge = avg_fav - avg_adv

                # 盈亏比 (反转方向做单: favorable/adverse)
                rr_ratio = avg_fav / avg_adv if avg_adv > 0 else 0

                horizon_stats[h_key] = {
                    'n': n_total,
                    'avg_move': round(np.mean(moves), 2),
                    'std_move': round(np.std(moves), 2),
                    'continuation_rate': round(cont_rate, 4),
                    'reversal_rate': round(rev_rate, 4),
                    'pred_strength': round(pred_strength, 4),
                    'avg_favorable': round(avg_fav, 2),
                    'avg_adverse': round(avg_adv, 2),
                    'edge_pips': round(edge, 2),
                    'rr_ratio': round(rr_ratio, 3),
                }

            cluster['horizon_stats'] = horizon_stats

            if horizon_stats:
                cluster['best_horizon'] = max(horizon_stats.keys(),
                    key=lambda k: horizon_stats[k]['pred_strength'])
                cluster['best_pred_strength'] = horizon_stats[cluster['best_horizon']]['pred_strength']
                cluster['best_edge'] = horizon_stats[cluster['best_horizon']]['edge_pips']
            else:
                cluster['best_horizon'] = None
                cluster['best_pred_strength'] = 0
                cluster['best_edge'] = 0

            # 综合评分: 概率 × 盈亏比 × sqrt(频次)
            # 用H10作为基准horizon
            h10 = horizon_stats.get('H10', {})
            if h10:
                rev_rate = h10['reversal_rate']
                rr = h10['rr_ratio']
                freq = cluster['n_samples']
                # 期望值 = rev_rate * avg_favorable - (1-rev_rate) * avg_adverse
                ev_per_trade = rev_rate * h10['avg_favorable'] - (1 - rev_rate) * h10['avg_adverse']
                # 综合分 = 期望值/trade × sqrt(频次) (越高越好)
                composite = ev_per_trade * np.sqrt(freq)
                cluster['ev_per_trade'] = round(ev_per_trade, 2)
                cluster['composite_score'] = round(composite, 1)
            else:
                cluster['ev_per_trade'] = 0
                cluster['composite_score'] = 0

            all_clusters.append(cluster)

    return all_clusters


# =============================================================================
# 6. 报告
# =============================================================================

def report(clusters, top_n=15):
    # --- 按 (pct, span_bucket) 分组的明细报告 ---
    groups = defaultdict(list)
    for c in clusters:
        groups[(c['pct'], c['span_bucket'])].append(c)

    for (pct, sb), cls in sorted(groups.items()):
        cls.sort(key=lambda c: c['composite_score'], reverse=True)  # 最高composite在前
        # 只报告有一定数量cluster的格子
        if len(cls) < 3:
            continue
        print(f"\n{'='*70}")
        print(f"top {pct}% pivots, span {sb} — {len(cls)} clusters")
        print(f"{'='*70}")

        for i, c in enumerate(cls[:top_n]):
            h10 = c['horizon_stats'].get('H10', {})
            h20 = c['horizon_stats'].get('H20', {})
            print(f"\n  [{i+1}] {c['fingerprint']}  (n={c['n_samples']}, avg_span={c['avg_span']:.0f}, amp={c['avg_amp_pips']:.0f}p)")
            print(f"      imp={c['avg_imp']:.3f}, retrace={c['avg_retrace']}, sym={c['avg_sym']:.3f}")
            if h10:
                print(f"      H10: rev={h10['reversal_rate']:.3f}, edge={h10['edge_pips']:+.1f}p, fav={h10['avg_favorable']:.1f}p, adv={h10['avg_adverse']:.1f}p, RR={h10['rr_ratio']:.2f}")
            if h20:
                print(f"      H20: rev={h20['reversal_rate']:.3f}, edge={h20['edge_pips']:+.1f}p, RR={h20['rr_ratio']:.2f}")
            print(f"      EV/trade={c['ev_per_trade']:+.1f}p, composite={c['composite_score']:+.0f}")

    # --- BASELINE: 每个(pct, span_bucket)格子的整体统计 ---
    print(f"\n{'='*70}")
    print(f"BASELINE: 各(百分位, 跨度)格子的整体反转率")
    print(f"{'='*70}")
    print(f"  {'pct':>5s} {'span':>6s} {'cls':>4s} {'samples':>8s} {'rev_H10':>8s} {'edge_H10':>9s} {'rev_H20':>8s} {'edge_H20':>9s} {'avg_amp':>8s}")

    for (pct, sb), cls in sorted(groups.items()):
        h10_data = [(c['horizon_stats']['H10']['reversal_rate'],
                      c['horizon_stats']['H10']['edge_pips'],
                      c['n_samples'])
                     for c in cls if 'H10' in c['horizon_stats']]
        if not h10_data:
            continue
        tn = sum(n for _,_,n in h10_data)
        r10 = sum(r*n for r,_,n in h10_data) / tn
        e10 = sum(e*n for _,e,n in h10_data) / tn

        h20_data = [(c['horizon_stats']['H20']['reversal_rate'],
                      c['horizon_stats']['H20']['edge_pips'],
                      c['n_samples'])
                     for c in cls if 'H20' in c['horizon_stats']]
        r20 = sum(r*n for r,_,n in h20_data) / sum(n for _,_,n in h20_data) if h20_data else 0
        e20 = sum(e*n for _,e,n in h20_data) / sum(n for _,_,n in h20_data) if h20_data else 0

        avg_amp = np.mean([c['avg_amp_pips'] for c in cls])
        print(f"  {pct:>5d} {sb:>6s} {len(cls):>4d} {tn:>8d} {r10:>8.3f} {e10:>+9.1f}p {r20:>8.3f} {e20:>+9.1f}p {avg_amp:>8.0f}p")

    # --- FINGERPRINT区分度 ---
    print(f"\n{'='*70}")
    print(f"FINGERPRINT区分度: 同格子内反转率std (越大=fingerprint越有区分价值)")
    print(f"{'='*70}")
    print(f"  {'pct':>5s} {'span':>6s} {'cls':>4s} {'rev_std':>8s} {'rev_range':>12s} {'edge_std':>9s}")

    for (pct, sb), cls in sorted(groups.items()):
        revs = [c['horizon_stats']['H10']['reversal_rate']
                for c in cls if 'H10' in c['horizon_stats']]
        edges = [c['horizon_stats']['H10']['edge_pips']
                 for c in cls if 'H10' in c['horizon_stats']]
        if len(revs) < 3:
            continue
        print(f"  {pct:>5d} {sb:>6s} {len(revs):>4d} {np.std(revs):>8.3f} {min(revs):.3f}~{max(revs):.3f} {np.std(edges):>9.1f}p")

    # --- TOP综合排名: 概率×盈亏比×频次 最优的fingerprint ---
    print(f"\n{'='*70}")
    print(f"TOP 30 综合排名: EV/trade × sqrt(freq)")
    print(f"  (正EV=反转做单有正期望, 负composite=亏损)")
    print(f"{'='*70}")
    # 只看有H10数据的
    ranked = [c for c in clusters if 'H10' in c['horizon_stats']]
    ranked.sort(key=lambda c: c['composite_score'], reverse=True)
    print(f"  {'rank':>4s} {'pct':>4s} {'span':>5s} {'fingerprint':>18s} {'n':>6s} {'rev%':>6s} {'edge':>7s} {'fav':>6s} {'adv':>6s} {'RR':>5s} {'EV':>7s} {'comp':>7s}")
    for i, c in enumerate(ranked[:30]):
        h10 = c['horizon_stats']['H10']
        print(f"  {i+1:>4d} {c['pct']:>4d} {c['span_bucket']:>5s} {c['fingerprint']:>18s} {c['n_samples']:>6d} {h10['reversal_rate']:>6.1%} {h10['edge_pips']:>+7.1f} {h10['avg_favorable']:>6.1f} {h10['avg_adverse']:>6.1f} {h10['rr_ratio']:>5.2f} {c['ev_per_trade']:>+7.1f} {c['composite_score']:>+7.0f}")


# =============================================================================
# 7. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Scanner V2 — dual-window importance-driven')
    parser.add_argument('--pair', default='EURUSD')
    parser.add_argument('--tf', default='H1')
    parser.add_argument('--bars', type=int, default=None, help='Max bars (None=all)')
    parser.add_argument('--outer', default='10000,20000,50000', help='Outer window sizes')
    parser.add_argument('--inner', type=int, default=500, help='Inner window size')
    parser.add_argument('--inner-step', type=int, default=200, help='Inner window step')
    parser.add_argument('--pct', default='10,20,30,50,75,100', help='Importance percentiles')
    parser.add_argument('--min-samples', type=int, default=30)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    outer_sizes = [int(x) for x in args.outer.split(',')]
    percentiles = [int(x) for x in args.pct.split(',')]

    print(f"=== Scanner V2 (dual-window): {args.pair} {args.tf} ===")
    print(f"  outer: {outer_sizes}")
    print(f"  inner: {args.inner}, step: {args.inner_step}")
    print(f"  percentiles: {percentiles}")
    t0 = time.time()

    df = load_klines(args.pair, args.tf, args.bars)
    print(f"  Loaded {len(df)} bars")

    max_horizon = 50
    outer_sizes = [o for o in outer_sizes if o + max_horizon < len(df)]
    if not outer_sizes:
        print("ERROR: not enough data for any outer size")
        return

    print(f"\nRunning dual-window scan...")
    structures = run_dual_window(
        df, outer_sizes, args.inner, args.inner_step,
        percentiles, [5, 10, 20, 50]
    )

    total_structs = sum(len(v) for v in structures.values())
    print(f"\nTotal structures: {total_structs}")
    for (outer, pct), structs in sorted(structures.items()):
        print(f"  outer={outer}, pct={pct}: {len(structs)} structures")

    print(f"\nAnalyzing clusters (min_samples={args.min_samples})...")
    clusters = analyze_clusters(structures, args.min_samples)
    print(f"Total clusters: {len(clusters)}")

    report(clusters)

    elapsed = time.time() - t0
    print(f"\n\nCompleted in {elapsed:.1f}s")

    output_path = args.output or f'scan_v2_{args.pair}_{args.tf}.json'
    result = {
        'pair': args.pair, 'tf': args.tf,
        'total_bars': len(df),
        'outer_sizes': outer_sizes,
        'inner_size': args.inner, 'inner_step': args.inner_step,
        'percentiles': percentiles,
        'total_structures': total_structs,
        'n_clusters': len(clusters),
        'clusters': sorted(clusters, key=lambda c: c['composite_score'], reverse=True)[:100],
        'elapsed_seconds': round(elapsed, 1),
    }
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
