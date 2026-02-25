"""
Stage2 ABC V3 — Intra-B Entry
===============================
Core change: don't wait for B to complete. Enter DURING B wave
when conditions are met.

Phase 1: Collect a massive table of "what if I entered at this bar during B?"
  - For every confirmed A wave, track B bar-by-bar
  - At each B bar, record features + simulate trade outcome
  - This gives us ground truth: which B-bar conditions lead to profit

Phase 2: Find optimal entry conditions from this table

Uses zigzag(2,1,1) + merge for A wave detection.
Entry points are raw bar-level (no zigzag confirmation needed).
"""

import os
import csv
import numpy as np
from multiprocessing import Pool
from collections import defaultdict
from math import erfc, sqrt

DATA_DIR = "/home/ubuntu/DataBase/base_kline"

# ═══════════════════════════════════════════════════════════════════════
# BATCH ZIGZAG(2,1,1)
# For speed, use batch mode to get all L0 pivots, then merge
# ═══════════════════════════════════════════════════════════════════════

def zigzag_211_batch(highs, lows, dev_points=0.0001):
    """Batch zigzag(2,1,1). Returns list of (bar_idx, price, direction)."""
    n = len(highs)
    if n < 3:
        return []

    pivots = []
    # Find first extreme
    if highs[0] >= highs[1]:
        pivots.append((0, highs[0], 1))
        searching = -1  # search for low
        last_low = lows[0]
        last_low_bar = 0
    else:
        pivots.append((0, lows[0], -1))
        searching = 1  # search for high
        last_high = highs[0]
        last_high_bar = 0

    if searching == 1:
        last_high = highs[0]
        last_high_bar = 0
    else:
        last_low = lows[0]
        last_low_bar = 0

    for i in range(1, n):
        if searching == 1:
            if highs[i] > last_high:
                last_high = highs[i]
                last_high_bar = i
            elif last_high_bar == i - 1:  # backstep=1
                if last_high - pivots[-1][1] > dev_points:
                    pivots.append((last_high_bar, last_high, 1))
                    searching = -1
                    last_low = lows[i]
                    last_low_bar = i
        else:  # searching for low
            if lows[i] < last_low:
                last_low = lows[i]
                last_low_bar = i
            elif last_low_bar == i - 1:
                if pivots[-1][1] - last_low > dev_points:
                    pivots.append((last_low_bar, last_low, -1))
                    searching = 1
                    last_high = highs[i]
                    last_high_bar = i

    return pivots


def merge_pivots_once(pivots):
    """One round of merge. Returns (merged_list, changed)."""
    if len(pivots) < 3:
        return pivots[:], False
    result = []
    i = 0
    changed = False
    while i < len(pivots):
        if i + 2 >= len(pivots):
            result.extend(pivots[i:])
            break
        p1, p2, p3 = pivots[i], pivots[i+1], pivots[i+2]
        can_merge = False
        if p1[2] == 1 and p2[2] == -1 and p3[2] == 1:
            if p3[1] >= p1[1]: can_merge = True
        elif p1[2] == -1 and p2[2] == 1 and p3[2] == -1:
            if p3[1] <= p1[1]: can_merge = True
        if can_merge:
            if p1[2] == 1:
                kept = p3 if p3[1] > p1[1] else p1
            else:
                kept = p3 if p3[1] < p1[1] else p1
            result.append(kept)
            i += 3
            changed = True
        else:
            result.append(p1)
            i += 1
    return result, changed


def merge_to_level(pivots, levels):
    """Merge N times."""
    cur = pivots[:]
    for _ in range(levels):
        cur, changed = merge_pivots_once(cur)
        if not changed:
            break
    return cur


# ═══════════════════════════════════════════════════════════════════════
# INTRA-B FEATURE COLLECTION
# ═══════════════════════════════════════════════════════════════════════

def collect_intra_b_entries(pair, tf, highs, lows, closes, pivots, spread_cost):
    """
    For each confirmed A wave, track B development bar-by-bar.
    At each bar during B, record features and simulate a hypothetical trade.

    A wave = pivots[i] → pivots[i+1]
    B starts at pivots[i+1] and we don't know where it ends yet.
    At each bar j > A_end, compute B-so-far features.
    If conditions seem plausible, simulate entry.

    We sample every N-th bar to keep data manageable.
    """
    n = len(closes)
    entries = []

    for i in range(len(pivots) - 1):
        p0 = pivots[i]      # A start
        p1 = pivots[i + 1]  # A end = B start

        a_bars = p1[0] - p0[0]
        a_amp = abs(p1[1] - p0[1])
        a_dir = 1 if p1[1] > p0[1] else -1
        a_end_bar = p1[0]
        a_end_price = p1[1]

        if a_bars < 2 or a_amp <= 0:
            continue

        a_slope = a_amp / a_bars

        # Track B wave bar by bar
        # B moves AGAINST A direction
        # Stop tracking when:
        #   - B goes too deep (> 1.5 * A_amp)
        #   - Too many bars (> 10 * A_bars)
        #   - End of data

        max_b_bars = min(int(a_bars * 10), 2000)
        max_b_depth = a_amp * 1.5

        b_extreme_price = a_end_price  # tracks how deep B has gone
        b_extreme_bar = a_end_bar

        # Sample: check every bar but only record every few bars
        # to keep dataset manageable
        sample_interval = max(1, a_bars // 10)

        for j in range(a_end_bar + 1, min(a_end_bar + max_b_bars, n - 100)):
            bar_idx = j
            c = closes[bar_idx]
            h = highs[bar_idx]
            l = lows[bar_idx]

            # B depth so far (how much has price retraced against A)
            if a_dir == 1:
                # A went up, B goes down
                b_depth = a_end_price - l  # using low as worst case
                b_current = a_end_price - c
                if l < b_extreme_price:
                    b_extreme_price = l
                    b_extreme_bar = bar_idx
            else:
                # A went down, B goes up
                b_depth = h - a_end_price
                b_current = c - a_end_price
                if h > b_extreme_price:
                    b_extreme_price = h
                    b_extreme_bar = bar_idx

            # B too deep? Stop.
            if b_depth > max_b_depth:
                break

            # B depth ratio (how much of A has been retraced)
            b_depth_ratio = b_depth / a_amp if a_amp > 0 else 0
            b_current_ratio = b_current / a_amp if a_amp > 0 else 0

            # B timing
            b_bars_so_far = bar_idx - a_end_bar
            b_time_ratio = b_bars_so_far / a_bars if a_bars > 0 else 0

            # B slope (current, using extreme)
            b_amp_so_far = abs(b_extreme_price - a_end_price)
            b_bars_to_extreme = max(1, b_extreme_bar - a_end_bar)
            b_slope = b_amp_so_far / b_bars_to_extreme if b_bars_to_extreme > 0 else 0
            slope_ratio = b_slope / a_slope if a_slope > 0 else 999

            # B deceleration: compare slope of last N bars vs first N bars
            lookback = max(3, b_bars_so_far // 3)
            if b_bars_so_far >= lookback * 2:
                if a_dir == 1:
                    early_move = abs(closes[a_end_bar + lookback] - a_end_price)
                    late_move = abs(c - closes[bar_idx - lookback])
                else:
                    early_move = abs(closes[a_end_bar + lookback] - a_end_price)
                    late_move = abs(c - closes[bar_idx - lookback])
                decel_ratio = late_move / early_move if early_move > 0 else 1.0
            else:
                decel_ratio = 1.0  # unknown

            # B minimum depth to be interesting
            if b_depth_ratio < 0.05:
                continue  # B hasn't retraced enough yet

            # Sample to keep data manageable
            if b_bars_so_far % sample_interval != 0 and b_bars_so_far > 3:
                continue

            # ── Simulate entry at this bar ──
            entry_price = c
            # Apply spread
            half_spread = spread_cost / 2
            if a_dir == 1:
                entry_adj = entry_price + half_spread
            else:
                entry_adj = entry_price - half_spread

            # TP/SL: based on A amp, from current entry price
            tp_dist = a_amp * 0.80
            sl_dist = tp_dist * 0.40  # 1:2.0 risk/reward base

            if tp_dist <= 0 or sl_dist <= 0:
                continue
            if spread_cost / tp_dist > 0.15:
                continue

            if a_dir == 1:
                tp_price = entry_adj + tp_dist
                sl_price = entry_adj - sl_dist
            else:
                tp_price = entry_adj - tp_dist
                sl_price = entry_adj + sl_dist

            # Simple simulation (no dynamic for data collection — keep it clean)
            exit_bar = min(bar_idx + max(int(a_bars * 8), 200), n - 1)
            exit_price = closes[exit_bar]
            exit_reason = 'timeout'

            for k in range(bar_idx + 1, exit_bar + 1):
                if a_dir == 1:
                    if lows[k] <= sl_price:
                        exit_price = sl_price; exit_reason = 'sl'; exit_bar = k; break
                    if highs[k] >= tp_price:
                        exit_price = tp_price; exit_reason = 'tp'; exit_bar = k; break
                else:
                    if highs[k] >= sl_price:
                        exit_price = sl_price; exit_reason = 'sl'; exit_bar = k; break
                    if lows[k] <= tp_price:
                        exit_price = tp_price; exit_reason = 'tp'; exit_bar = k; break

            # Apply exit spread
            if a_dir == 1:
                pnl = (exit_price - half_spread) - entry_adj
            else:
                pnl = entry_adj - (exit_price + half_spread)

            pnl_r = pnl / sl_dist if sl_dist > 0 else 0

            year = 0
            # Quick year extraction (avoid loading dates)

            entries.append({
                'pair': pair, 'tf': tf,
                'entry_bar': bar_idx,
                # A features
                'a_dir': a_dir, 'a_amp': a_amp, 'a_bars': a_bars, 'a_slope': a_slope,
                # B-so-far features (the key dimensions)
                'b_depth_ratio': round(b_depth_ratio, 4),
                'b_current_ratio': round(b_current_ratio, 4),
                'b_time_ratio': round(b_time_ratio, 4),
                'b_slope_ratio': round(slope_ratio, 4),
                'b_decel_ratio': round(decel_ratio, 4),
                'b_bars_so_far': b_bars_so_far,
                # Result
                'pnl_r': round(pnl_r, 4),
                'exit_reason': exit_reason,
                'hold_bars': exit_bar - bar_idx,
            })

    return entries


PAIR_INFO = {
    'EURUSD': {'spread': 1.2, 'pip': 0.0001}, 'GBPUSD': {'spread': 1.5, 'pip': 0.0001},
    'USDJPY': {'spread': 1.3, 'pip': 0.01}, 'USDCHF': {'spread': 1.5, 'pip': 0.0001},
    'AUDUSD': {'spread': 1.4, 'pip': 0.0001}, 'NZDUSD': {'spread': 1.8, 'pip': 0.0001},
    'USDCAD': {'spread': 1.6, 'pip': 0.0001}, 'EURGBP': {'spread': 1.8, 'pip': 0.0001},
    'EURJPY': {'spread': 1.8, 'pip': 0.01}, 'GBPJPY': {'spread': 2.5, 'pip': 0.01},
    'EURAUD': {'spread': 2.5, 'pip': 0.0001}, 'EURNZD': {'spread': 3.0, 'pip': 0.0001},
    'EURCAD': {'spread': 2.5, 'pip': 0.0001}, 'EURCHF': {'spread': 2.0, 'pip': 0.0001},
    'GBPAUD': {'spread': 3.0, 'pip': 0.0001}, 'GBPCAD': {'spread': 3.0, 'pip': 0.0001},
    'GBPCHF': {'spread': 2.8, 'pip': 0.0001}, 'GBPNZD': {'spread': 4.0, 'pip': 0.0001},
    'AUDCAD': {'spread': 2.5, 'pip': 0.0001}, 'AUDCHF': {'spread': 2.5, 'pip': 0.0001},
    'AUDJPY': {'spread': 2.0, 'pip': 0.01}, 'AUDNZD': {'spread': 2.5, 'pip': 0.0001},
    'CADJPY': {'spread': 2.0, 'pip': 0.01}, 'CADCHF': {'spread': 2.5, 'pip': 0.0001},
    'CHFJPY': {'spread': 2.5, 'pip': 0.01}, 'NZDJPY': {'spread': 2.5, 'pip': 0.01},
    'NZDCAD': {'spread': 3.0, 'pip': 0.0001}, 'NZDCHF': {'spread': 3.0, 'pip': 0.0001},
    'EURNOK': {'spread': 25.0, 'pip': 0.0001}, 'EURSEK': {'spread': 30.0, 'pip': 0.0001},
    'EURPLN': {'spread': 25.0, 'pip': 0.0001}, 'EURTRY': {'spread': 50.0, 'pip': 0.0001},
    'EURHKD': {'spread': 20.0, 'pip': 0.0001}, 'EURCNH': {'spread': 30.0, 'pip': 0.0001},
    'GBPNOK': {'spread': 40.0, 'pip': 0.0001}, 'USDNOK': {'spread': 20.0, 'pip': 0.0001},
    'USDSEK': {'spread': 25.0, 'pip': 0.0001}, 'USDPLN': {'spread': 25.0, 'pip': 0.0001},
    'USDTRY': {'spread': 50.0, 'pip': 0.0001}, 'USDHUF': {'spread': 20.0, 'pip': 0.01},
    'USDMXN': {'spread': 30.0, 'pip': 0.0001}, 'USDZAR': {'spread': 40.0, 'pip': 0.0001},
    'USDSGD': {'spread': 3.0, 'pip': 0.0001}, 'USDCNH': {'spread': 30.0, 'pip': 0.0001},
    'USDRMB': {'spread': 30.0, 'pip': 0.0001}, 'USDHKD': {'spread': 5.0, 'pip': 0.0001},
    'XAUUSD': {'spread': 3.0, 'pip': 0.01}, 'XAGUSD': {'spread': 3.0, 'pip': 0.001},
}

def get_spread_cost(pair):
    info = PAIR_INFO.get(pair, {'spread': 5.0, 'pip': 0.0001})
    return (info['spread'] + 1.0) * info['pip']

def get_dev(pair):
    info = PAIR_INFO.get(pair, {'pip': 0.0001})
    return info['pip']


def load_ohlcv(pair, tf):
    fpath = os.path.join(DATA_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fpath):
        return None
    dates, highs, lows, closes = [], [], [], []
    with open(fpath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) < 6: continue
            dates.append(f"{row[0]} {row[1]}")
            highs.append(float(row[3]))
            lows.append(float(row[4]))
            closes.append(float(row[5]))
    return dates, np.array(highs), np.array(lows), np.array(closes)


def process_pair_tf(args):
    pair, tf, merge_level = args
    data = load_ohlcv(pair, tf)
    if data is None:
        return []

    dates, highs, lows, closes = data
    dev = get_dev(pair)
    spread = get_spread_cost(pair)

    # L0 zigzag
    l0 = zigzag_211_batch(highs, lows, dev)
    if len(l0) < 5:
        return []

    # Merge to target level for A wave detection
    pivots = merge_to_level(l0, merge_level)
    if len(pivots) < 3:
        return []

    entries = collect_intra_b_entries(pair, tf, highs, lows, closes, pivots, spread)
    print(f"  {pair}_{tf}: {len(pivots)} pivots, {len(entries)} entry samples")
    return entries


def get_all_pairs():
    pairs = set()
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.csv'):
            parts = fname.replace('.csv', '').rsplit('_', 1)
            if len(parts) == 2: pairs.add(parts[0])
    return sorted(pairs)


def analyze(all_entries):
    """Analyze the entry condition space."""
    n = len(all_entries)
    pnls = np.array([e['pnl_r'] for e in all_entries])
    wins = np.sum(pnls > 0)

    print(f"\n{'='*90}")
    print(f"  OVERALL: {n:,} entry samples")
    print(f"  WR={wins/n*100:.1f}%  avgR={np.mean(pnls):.4f}  PF={abs(np.sum(pnls[pnls>0])/np.sum(pnls[pnls<=0])):.2f}")
    print(f"{'='*90}")

    # ── 2D Heatmap: b_depth_ratio × b_time_ratio ──
    print(f"\n  HEATMAP: avg PnL(R) by b_depth_ratio × b_time_ratio")
    print(f"  (This shows WHERE in B-wave development to enter)")

    depth_bins = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 1.50]
    time_bins = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    grid = defaultdict(list)
    for e in all_entries:
        di = len(depth_bins) - 1
        for j in range(len(depth_bins) - 1):
            if e['b_depth_ratio'] < depth_bins[j + 1]:
                di = j; break
        ti = len(time_bins) - 1
        for j in range(len(time_bins) - 1):
            if e['b_time_ratio'] < time_bins[j + 1]:
                ti = j; break
        grid[(di, ti)].append(e['pnl_r'])

    header = f"{'depth\\time':>14s}"
    for j in range(len(time_bins) - 1):
        header += f" {time_bins[j]:.1f}-{time_bins[j+1]:.1f}".rjust(10)
    print(header)
    print("-" * (14 + 10 * (len(time_bins) - 1)))

    for i in range(len(depth_bins) - 1):
        label = f"{depth_bins[i]:.2f}-{depth_bins[i+1]:.2f}"
        row = f"{label:>14s}"
        for j in range(len(time_bins) - 1):
            vals = grid.get((i, j), [])
            if len(vals) < 200:
                row += f"{'---':>10s}"
            else:
                avg = np.mean(vals)
                row += f"{avg:>10.3f}"
        print(row)

    # ── By b_slope_ratio ──
    print(f"\n  BY B SLOPE RATIO (lower = B weaker than A)")
    sr_bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0]
    print(f"{'slope_ratio':>14s} {'n':>8s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s}")
    print("-" * 48)
    for i in range(len(sr_bins) - 1):
        subset = [e for e in all_entries if sr_bins[i] <= e['b_slope_ratio'] < sr_bins[i+1]]
        if len(subset) < 200: continue
        p = np.array([e['pnl_r'] for e in subset])
        w = p[p > 0]; l = p[p <= 0]
        pf = abs(np.sum(w)/np.sum(l)) if np.sum(l) != 0 else float('inf')
        label = f"{sr_bins[i]:.1f}-{sr_bins[i+1]:.1f}"
        print(f"{label:>14s} {len(p):>8,} {np.sum(p>0)/len(p)*100:>6.1f} {np.mean(p):>8.4f} {pf:>7.2f}")

    # ── By b_decel_ratio (B slowing down) ──
    print(f"\n  BY B DECELERATION (lower = B slowing down)")
    dc_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0]
    print(f"{'decel':>14s} {'n':>8s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s}")
    print("-" * 48)
    for i in range(len(dc_bins) - 1):
        subset = [e for e in all_entries if dc_bins[i] <= e['b_decel_ratio'] < dc_bins[i+1]]
        if len(subset) < 200: continue
        p = np.array([e['pnl_r'] for e in subset])
        w = p[p > 0]; l = p[p <= 0]
        pf = abs(np.sum(w)/np.sum(l)) if np.sum(l) != 0 else float('inf')
        label = f"{dc_bins[i]:.1f}-{dc_bins[i+1]:.1f}"
        print(f"{label:>14s} {len(p):>8,} {np.sum(p>0)/len(p)*100:>6.1f} {np.mean(p):>8.4f} {pf:>7.2f}")

    # ── Best conditions ──
    print(f"\n{'='*90}")
    print(f"  COMBINED CONDITION SEARCH")
    print(f"{'='*90}")
    print(f"  Testing: b_slope_ratio < X AND b_depth_ratio in [Y1, Y2] AND b_time_ratio > Z")

    conditions = [
        ('slope<0.3 depth[0.2-0.7] time>1.0',
         lambda e: e['b_slope_ratio'] < 0.3 and 0.2 <= e['b_depth_ratio'] < 0.7 and e['b_time_ratio'] > 1.0),
        ('slope<0.5 depth[0.2-0.7] time>0.8',
         lambda e: e['b_slope_ratio'] < 0.5 and 0.2 <= e['b_depth_ratio'] < 0.7 and e['b_time_ratio'] > 0.8),
        ('slope<0.3 depth[0.3-1.0] time>1.5',
         lambda e: e['b_slope_ratio'] < 0.3 and 0.3 <= e['b_depth_ratio'] < 1.0 and e['b_time_ratio'] > 1.5),
        ('slope<0.5 depth[0.1-0.5] time>0.5',
         lambda e: e['b_slope_ratio'] < 0.5 and 0.1 <= e['b_depth_ratio'] < 0.5 and e['b_time_ratio'] > 0.5),
        ('slope<0.3 decel<0.5 depth[0.2-0.8]',
         lambda e: e['b_slope_ratio'] < 0.3 and e['b_decel_ratio'] < 0.5 and 0.2 <= e['b_depth_ratio'] < 0.8),
        ('slope<0.5 decel<0.6 time>1.0',
         lambda e: e['b_slope_ratio'] < 0.5 and e['b_decel_ratio'] < 0.6 and e['b_time_ratio'] > 1.0),
        ('slope<0.3 decel<0.4 depth[0.15-0.6]',
         lambda e: e['b_slope_ratio'] < 0.3 and e['b_decel_ratio'] < 0.4 and 0.15 <= e['b_depth_ratio'] < 0.6),
        ('ALL (baseline)',
         lambda e: True),
    ]

    print(f"\n{'Condition':>42s} {'n':>8s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s}")
    print("-" * 75)
    for name, filt in conditions:
        subset = [e for e in all_entries if filt(e)]
        if len(subset) < 100: continue
        p = np.array([e['pnl_r'] for e in subset])
        w = p[p > 0]; l = p[p <= 0]
        pf = abs(np.sum(w)/np.sum(l)) if np.sum(l) != 0 else float('inf')
        print(f"{name:>42s} {len(p):>8,} {np.sum(p>0)/len(p)*100:>6.1f} {np.mean(p):>8.4f} {pf:>7.2f}")


def main():
    pairs = get_all_pairs()
    tfs = ['H1']  # Start with H1 only for speed
    merge_level = 3  # Use L3 merged pivots for A wave detection

    print(f"ABC V3 — Intra-B Entry Collection")
    print(f"  Pairs: {len(pairs)}, TFs: {tfs}")
    print(f"  ZigZag: (2,1,1), merge to L{merge_level}")
    print(f"  Entry: during B wave, no confirmation needed")
    print()

    tasks = [(pair, tf, merge_level) for pair in pairs for tf in tfs]
    print(f"Tasks: {len(tasks)}")

    all_entries = []
    with Pool(40) as pool:
        for batch in pool.imap_unordered(process_pair_tf, tasks):
            all_entries.extend(batch)

    print(f"\nTotal entry samples: {len(all_entries):,}")

    if all_entries:
        analyze(all_entries)

        # Save
        out = "/home/ubuntu/stage2_abc/abc_v3_intra_b_samples.csv"
        fields = list(all_entries[0].keys())
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for e in all_entries:
                w.writerow(e)
        print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
