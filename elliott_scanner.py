#!/usr/bin/env python3
"""
Elliott Wave 12345+abc 模式扫描器

在全量zigzag数据中搜索满足Elliott推动浪+调整浪几何约束的8段结构，
统计后续走势分布。

基于用户标注的标准案例定义约束:
  推动浪(5段): 浪3最长, 浪2回撤<100%, 浪4不重叠
  调整浪(3段): c/a在一定范围内

支持多级别zigzag（L0, A1, A2, ...）上的扫描。
"""

import sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, '/home/ubuntu/stage2_abc')
from merge_engine_v3 import (
    load_kline, calculate_base_zg, full_merge_engine,
    build_segment_pool
)


def check_impulse_5wave(pivots_9, direction='down'):
    """
    检查9个连续pivot是否构成Elliott推动浪(12345) + 调整浪(abc)
    
    pivots_9: list of 9 tuples (bar, price, dir)
    direction: 'down' = 推动下跌, 'up' = 推动上涨
    
    返回: (match, details_dict) or (False, None)
    
    8段结构 (9个pivot):
      推动: P0→P1(浪1) P1→P2(浪2) P2→P3(浪3) P3→P4(浪4) P4→P5(浪5)
      调整: P5→P6(a浪)  P6→P7(b浪)  P7→P8(c浪)
    """
    if len(pivots_9) != 9:
        return False, None
    
    P = pivots_9
    
    # 提取8段的幅度和方向
    segs = []
    for i in range(8):
        b1, p1, d1 = P[i]
        b2, p2, d2 = P[i + 1]
        amp = abs(p2 - p1)
        time_span = abs(b2 - b1)
        seg_dir = 1 if p2 > p1 else -1
        segs.append({
            'b1': b1, 'p1': p1, 'b2': b2, 'p2': p2,
            'amp': amp, 'time': time_span, 'dir': seg_dir
        })
    
    # 方向序列检查: 必须交替
    dirs = [s['dir'] for s in segs]
    for i in range(1, len(dirs)):
        if dirs[i] == dirs[i - 1]:
            return False, None
    
    # 确定推动方向
    if direction == 'down':
        # 浪1,3,5 应该是 DN(-1), 浪2,4 应该是 UP(1)
        if dirs[0] != -1:
            return False, None
    else:
        # 浪1,3,5 应该是 UP(1), 浪2,4 应该是 DN(-1)
        if dirs[0] != 1:
            return False, None
    
    w1, w2, w3, w4, w5 = segs[0], segs[1], segs[2], segs[3], segs[4]
    wa, wb, wc = segs[5], segs[6], segs[7]
    
    # ========== 推动浪铁律 ==========
    
    # 规则1: 浪3不能是最短的推动浪 (1,3,5中)
    amps_135 = [w1['amp'], w3['amp'], w5['amp']]
    if w3['amp'] == min(amps_135):
        return False, None
    
    # 规则2: 浪2回撤不超过浪1的100%
    retrace_2 = w2['amp'] / max(w1['amp'], 1e-10)
    if retrace_2 >= 1.0:
        return False, None
    
    # 规则3: 浪4不进入浪1的价格领域
    if direction == 'down':
        # 浪1是下跌: P0(高)→P1(低), 浪4是上涨: P3(低)→P4(高)
        # 浪4的顶(P4) 不能 >= 浪1的底(P1)
        if P[4][1] >= P[1][1]:
            return False, None
    else:
        # 浪1是上涨: P0(低)→P1(高), 浪4是下跌: P3(高)→P4(低)
        # 浪4的底(P4) 不能 <= 浪1的顶(P1)
        if P[4][1] <= P[1][1]:
            return False, None
    
    # ========== 推动浪偏好约束（宽松版） ==========
    
    # 浪3最好是最长的（强偏好，但不强制）
    w3_is_longest = w3['amp'] == max(amps_135)
    
    # 浪2回撤浪1: 通常 23.6% - 78.6%
    retrace_2_ok = 0.15 <= retrace_2 <= 0.90
    
    # 浪4回撤浪3: 通常 23.6% - 50%
    retrace_4 = w4['amp'] / max(w3['amp'], 1e-10)
    retrace_4_ok = 0.10 <= retrace_4 <= 0.65
    
    # ========== 调整浪约束（宽松版） ==========
    
    # 调整浪方向应与推动浪相反
    if direction == 'down':
        # 调整浪a应该UP, b应该DN, c应该UP
        if wa['dir'] != 1:
            return False, None
    else:
        if wa['dir'] != -1:
            return False, None
    
    # c/a 比值: 通常 0.618 - 1.618
    ca_ratio = wc['amp'] / max(wa['amp'], 1e-10)
    ca_ok = 0.5 <= ca_ratio <= 2.0
    
    # b回撤a: 通常 38.2% - 100%
    retrace_b = wb['amp'] / max(wa['amp'], 1e-10)
    
    # 调整浪总幅度不超过推动浪（否则推动浪判断可能有误）
    total_impulse = abs(P[5][1] - P[0][1])
    total_correction = abs(P[8][1] - P[5][1])
    correction_ratio = total_correction / max(total_impulse, 1e-10)
    
    # 约束: 调整不能超过推动的100%（否则趋势已经反转了）
    if correction_ratio >= 1.0:
        return False, None
    
    # ========== 评分 ==========
    score = 0
    if w3_is_longest:
        score += 2
    if retrace_2_ok:
        score += 1
    if retrace_4_ok:
        score += 1
    if ca_ok:
        score += 1
    if 0.3 <= retrace_b <= 1.0:
        score += 1
    if correction_ratio <= 0.618:
        score += 1
    # 浪3 > 1.5倍浪1 (强趋势)
    if w3['amp'] > 1.5 * w1['amp']:
        score += 1
    
    details = {
        'direction': direction,
        'score': score,
        'w3_is_longest': w3_is_longest,
        'retrace_2': round(retrace_2, 4),
        'retrace_4': round(retrace_4, 4),
        'retrace_b': round(retrace_b, 4),
        'ca_ratio': round(ca_ratio, 4),
        'correction_ratio': round(correction_ratio, 4),
        'w3_w1_ratio': round(w3['amp'] / max(w1['amp'], 1e-10), 4),
        'amps': [round(s['amp'], 6) for s in segs],
        'times': [s['time'] for s in segs],
        'bar_start': P[0][0],
        'bar_end': P[8][0],
        'price_start': P[0][1],
        'price_end': P[8][1],
        'pivots': [(p[0], round(p[1], 5), p[2]) for p in P],
    }
    
    return True, details


def scan_pivots_for_elliott(pivots, min_score=3):
    """
    在pivot序列上滑动窗口扫描8段(9个pivot)的Elliott模式
    
    pivots: list of (bar, price, dir)
    min_score: 最低得分阈值
    
    返回: list of match dicts
    """
    matches = []
    n = len(pivots)
    
    for i in range(n - 8):
        window = pivots[i:i + 9]
        
        # 检查下跌推动
        ok, details = check_impulse_5wave(window, direction='down')
        if ok and details['score'] >= min_score:
            details['pivot_idx_start'] = i
            matches.append(details)
            continue  # 不重复检查同一窗口
        
        # 检查上涨推动
        ok, details = check_impulse_5wave(window, direction='up')
        if ok and details['score'] >= min_score:
            details['pivot_idx_start'] = i
            matches.append(details)
    
    return matches


def compute_outcome(df_highs, df_lows, df_closes, bar_end, horizons=[10, 20, 50, 100]):
    """
    计算模式结束后的后续走势
    
    bar_end: 模式结束的bar index
    horizons: 观察的前瞻窗口 (bars)
    
    返回: dict of outcomes
    """
    n = len(df_closes)
    outcomes = {}
    
    price_at_end = df_closes[min(bar_end, n - 1)]
    
    for h in horizons:
        target = bar_end + h
        if target >= n:
            continue
        
        # 区间内的最高/最低/收盘
        slice_high = df_highs[bar_end + 1:target + 1]
        slice_low = df_lows[bar_end + 1:target + 1]
        
        if len(slice_high) == 0:
            continue
        
        max_price = slice_high.max()
        min_price = slice_low.min()
        close_price = df_closes[target]
        
        outcomes[f'h{h}'] = {
            'close_chg': round((close_price - price_at_end) / price_at_end * 10000, 2),  # pips (×10000)
            'max_up': round((max_price - price_at_end) / price_at_end * 10000, 2),
            'max_dn': round((price_at_end - min_price) / price_at_end * 10000, 2),
            'close_dir': 'UP' if close_price > price_at_end else 'DN',
        }
    
    return outcomes


def run_full_scan(pairs=None, timeframes=None, min_score=3, use_merge=True):
    """
    全量扫描
    
    pairs: 品种列表, None=全部
    timeframes: 时间框架列表, None=['H1']
    min_score: 最低得分
    use_merge: True=在多级别归并上扫描, False=只在base zigzag上扫描
    """
    data_dir = '/home/ubuntu/DataBase/base_kline'
    
    if timeframes is None:
        timeframes = ['H1']
    
    import os
    if pairs is None:
        # 自动发现所有品种
        all_files = os.listdir(data_dir)
        pair_set = set()
        for f in all_files:
            if f.endswith('.csv'):
                parts = f.replace('.csv', '').split('_')
                if len(parts) == 2:
                    pair_set.add(parts[0])
        pairs = sorted(pair_set)
    
    all_matches = []
    all_outcomes = defaultdict(list)
    
    for tf in timeframes:
        for pair in pairs:
            fpath = os.path.join(data_dir, f'{pair}_{tf}.csv')
            if not os.path.exists(fpath):
                continue
            
            try:
                df = load_kline(fpath, limit=200000)
            except Exception as e:
                print(f'  SKIP {pair}_{tf}: {e}')
                continue
            
            if len(df) < 100:
                continue
            
            highs = df['high'].values.astype(float)
            lows = df['low'].values.astype(float)
            closes = df['close'].values.astype(float)
            
            base = calculate_base_zg(highs, lows, rb=0.5)
            
            # 在不同级别上扫描
            scan_levels = [('L0', base)]
            
            if use_merge and len(base) > 20:
                try:
                    results = full_merge_engine(base)
                    for snap_type, label, pvts in results['all_snapshots']:
                        if len(pvts) >= 9:
                            scan_levels.append((label, pvts))
                except Exception:
                    pass
            
            pair_matches = 0
            for level_name, pvts in scan_levels:
                matches = scan_pivots_for_elliott(pvts, min_score=min_score)
                
                for m in matches:
                    m['pair'] = pair
                    m['timeframe'] = tf
                    m['level'] = level_name
                    
                    # 计算后续走势
                    bar_end = m['bar_end']
                    outcome = compute_outcome(highs, lows, closes, bar_end)
                    m['outcome'] = outcome
                    
                    # 收集按方向分类的outcome
                    for hkey, hval in outcome.items():
                        all_outcomes[f"{m['direction']}_{hkey}"].append(hval)
                    
                    all_matches.append(m)
                    pair_matches += 1
            
            if pair_matches > 0:
                print(f'  {pair}_{tf}: {pair_matches} matches (levels: {len(scan_levels)})')
    
    return all_matches, all_outcomes


def print_report(all_matches, all_outcomes):
    """打印统计报告"""
    print('\n' + '=' * 80)
    print(f'ELLIOTT 12345+abc PATTERN SCAN REPORT')
    print('=' * 80)
    
    n = len(all_matches)
    print(f'\nTotal matches: {n}')
    
    if n == 0:
        print('No matches found.')
        return
    
    # 按方向统计
    down_matches = [m for m in all_matches if m['direction'] == 'down']
    up_matches = [m for m in all_matches if m['direction'] == 'up']
    print(f'  Down impulse: {len(down_matches)}')
    print(f'  Up impulse:   {len(up_matches)}')
    
    # 按品种统计
    pair_counts = defaultdict(int)
    for m in all_matches:
        pair_counts[f"{m['pair']}_{m['timeframe']}"] += 1
    print(f'\nBy pair (top 15):')
    for pair, cnt in sorted(pair_counts.items(), key=lambda x: -x[1])[:15]:
        print(f'  {pair}: {cnt}')
    
    # 按级别统计
    level_counts = defaultdict(int)
    for m in all_matches:
        level_counts[m['level']] += 1
    print(f'\nBy merge level:')
    for level, cnt in sorted(level_counts.items(), key=lambda x: -x[1]):
        print(f'  {level}: {cnt}')
    
    # 按得分统计
    score_counts = defaultdict(int)
    for m in all_matches:
        score_counts[m['score']] += 1
    print(f'\nBy score:')
    for sc in sorted(score_counts.keys()):
        print(f'  score={sc}: {score_counts[sc]}')
    
    # ========== 核心: 后续走势统计 ==========
    print('\n' + '=' * 80)
    print('OUTCOME STATISTICS (post-pattern movement)')
    print('=' * 80)
    
    for direction in ['down', 'up']:
        dir_matches = [m for m in all_matches if m['direction'] == direction]
        if not dir_matches:
            continue
        
        print(f'\n--- {direction.upper()} impulse ({len(dir_matches)} cases) ---')
        print(f'After a {direction} impulse 12345 + abc correction:')
        if direction == 'down':
            print(f'  Expected: price continues DOWN (impulse direction)')
        else:
            print(f'  Expected: price continues UP (impulse direction)')
        
        for horizon in [10, 20, 50, 100]:
            hkey = f'h{horizon}'
            cases_with_outcome = [m for m in dir_matches if hkey in m.get('outcome', {})]
            if not cases_with_outcome:
                continue
            
            chgs = [m['outcome'][hkey]['close_chg'] for m in cases_with_outcome]
            max_ups = [m['outcome'][hkey]['max_up'] for m in cases_with_outcome]
            max_dns = [m['outcome'][hkey]['max_dn'] for m in cases_with_outcome]
            
            # 推动方向的延续概率
            if direction == 'down':
                continue_count = sum(1 for c in chgs if c < 0)
            else:
                continue_count = sum(1 for c in chgs if c > 0)
            
            continue_pct = continue_count / len(chgs) * 100
            
            avg_chg = np.mean(chgs)
            median_chg = np.median(chgs)
            std_chg = np.std(chgs)
            avg_max_up = np.mean(max_ups)
            avg_max_dn = np.mean(max_dns)
            
            print(f'\n  H+{horizon} bars ({len(chgs)} cases):')
            print(f'    Direction continuation: {continue_pct:.1f}% ({continue_count}/{len(chgs)})')
            print(f'    Avg change: {avg_chg:+.1f} pips | Median: {median_chg:+.1f} pips | Std: {std_chg:.1f}')
            print(f'    Avg max favorable: {(avg_max_dn if direction=="down" else avg_max_up):.1f} pips')
            print(f'    Avg max adverse:   {(avg_max_up if direction=="down" else avg_max_dn):.1f} pips')
    
    # ========== 按得分分层的后续走势 ==========
    print('\n' + '=' * 80)
    print('OUTCOME BY SCORE (higher score = more textbook)')
    print('=' * 80)
    
    for min_sc in [3, 5, 6, 7]:
        high_score = [m for m in all_matches if m['score'] >= min_sc]
        if not high_score:
            continue
        
        cases_h50 = [m for m in high_score if 'h50' in m.get('outcome', {})]
        if not cases_h50:
            continue
        
        chgs = [m['outcome']['h50']['close_chg'] for m in cases_h50]
        
        continue_count = 0
        for m in cases_h50:
            c = m['outcome']['h50']['close_chg']
            if (m['direction'] == 'down' and c < 0) or (m['direction'] == 'up' and c > 0):
                continue_count += 1
        
        continue_pct = continue_count / len(chgs) * 100
        print(f'\n  Score >= {min_sc}: {len(cases_h50)} cases')
        print(f'    H+50 continuation: {continue_pct:.1f}%')
        print(f'    H+50 avg change: {np.mean(chgs):+.1f} pips | median: {np.median(chgs):+.1f}')
    
    # ========== 最佳案例展示 ==========
    print('\n' + '=' * 80)
    print('TOP SCORING MATCHES (score >= 6)')
    print('=' * 80)
    
    top = sorted([m for m in all_matches if m['score'] >= 6], key=lambda x: -x['score'])
    for i, m in enumerate(top[:20]):
        h50 = m.get('outcome', {}).get('h50', {})
        h50_str = f"h50={h50.get('close_chg', '?'):+.1f}pips" if h50 else 'n/a'
        print(f"  #{i+1} {m['pair']}_{m['timeframe']} {m['level']} score={m['score']} "
              f"{m['direction']} bar={m['bar_start']}-{m['bar_end']} "
              f"w3/w1={m['w3_w1_ratio']:.2f} ret2={m['retrace_2']:.0%} ret4={m['retrace_4']:.0%} "
              f"c/a={m['ca_ratio']:.2f} corr={m['correction_ratio']:.0%} | {h50_str}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', nargs='+', default=None, help='Specific pairs to scan')
    parser.add_argument('--tf', nargs='+', default=['H1'], help='Timeframes')
    parser.add_argument('--score', type=int, default=3, help='Minimum score')
    parser.add_argument('--no-merge', action='store_true', help='Only scan base zigzag')
    parser.add_argument('--quick', action='store_true', help='Quick: only EURUSD')
    args = parser.parse_args()
    
    if args.quick:
        args.pairs = ['EURUSD']
    
    print(f'Elliott 12345+abc Scanner')
    print(f'Pairs: {args.pairs or "ALL"} | TF: {args.tf} | Min score: {args.score} | Merge: {not args.no_merge}')
    print()
    
    t0 = time.time()
    matches, outcomes = run_full_scan(
        pairs=args.pairs,
        timeframes=args.tf,
        min_score=args.score,
        use_merge=not args.no_merge,
    )
    elapsed = time.time() - t0
    
    print(f'\nScan completed in {elapsed:.1f}s')
    print_report(matches, outcomes)
