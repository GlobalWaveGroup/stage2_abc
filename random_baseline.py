"""
Random Baseline Test
====================
Use the EXACT same trade simulation logic (SL/TP/dynamic adjustment)
but with RANDOM entry points and RANDOM directions.

This answers: is the positive expectancy from the ABC structure,
or just from the SL/TP mechanics (tight SL + trailing TP)?

We match the ABC strategy on:
  - Same pairs, same timeframes
  - Same number of trades per pair-TF
  - Same A_amp distribution (sampled from actual ABC trades)
  - Same trade simulation engine
  - Only difference: entry bar is random, direction is random

If random baseline also shows PF~4+ and avg_R~0.6, then the
ABC structure provides ZERO edge — the mechanics do all the work.
"""

import os
import sys
import csv
import numpy as np
from multiprocessing import Pool
from collections import defaultdict

DATA_DIR = "/home/ubuntu/DataBase/base_kline"
ABC_FILE = "/home/ubuntu/stage2_abc/abc_all_triples.csv"
OUT_DIR = "/home/ubuntu/stage2_abc/analysis"


def load_ohlcv(pair, tf):
    fpath = os.path.join(DATA_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fpath):
        return None
    opens, highs, lows, closes = [], [], [], []
    with open(fpath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) < 6:
                continue
            opens.append(float(row[2]))
            highs.append(float(row[3]))
            lows.append(float(row[4]))
            closes.append(float(row[5]))
    return np.array(opens), np.array(highs), np.array(lows), np.array(closes)


def simulate_trade(entry_bar, a_dir, a_amp, highs, lows, closes, a_bars):
    """Exact same logic as abc_collector.py simulate_trade."""
    if entry_bar >= len(closes) - 1:
        return None

    entry_price = closes[entry_bar]

    tp_distance = a_amp * 0.80
    sl_distance = tp_distance / 2.5

    if sl_distance <= 0 or tp_distance <= 0:
        return None

    if a_dir == 1:
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
    else:
        tp_price = entry_price - tp_distance
        sl_price = entry_price + sl_distance

    max_favorable = 0.0
    current_tp = tp_price
    current_sl = sl_price
    hit_breakeven = False

    max_bars = max(int(a_bars * 8), 500)
    end_bar = min(entry_bar + max_bars, len(closes) - 1)

    exit_bar = end_bar
    exit_price = closes[end_bar]
    exit_reason = 'timeout'

    for bar in range(entry_bar + 1, end_bar + 1):
        h = highs[bar]
        l = lows[bar]

        if a_dir == 1:
            favorable = h - entry_price
        else:
            favorable = entry_price - l

        if favorable > max_favorable:
            max_favorable = favorable

        progress = max_favorable / tp_distance if tp_distance > 0 else 0
        bars_elapsed = bar - entry_bar

        # Check SL
        if a_dir == 1:
            if l <= current_sl:
                exit_bar = bar
                exit_price = current_sl
                exit_reason = 'sl'
                break
        else:
            if h >= current_sl:
                exit_bar = bar
                exit_price = current_sl
                exit_reason = 'sl'
                break

        # Dynamic adjustments (identical to ABC)
        if progress >= 0.30 and not hit_breakeven:
            current_sl = entry_price
            hit_breakeven = True

        if progress >= 0.60:
            if a_dir == 1:
                lock_sl = entry_price + max_favorable * 0.50
                current_sl = max(current_sl, lock_sl)
            else:
                lock_sl = entry_price - max_favorable * 0.50
                current_sl = min(current_sl, lock_sl)

        if progress >= 1.0:
            new_tp_dist = a_amp * 1.20
            if a_dir == 1:
                current_tp = max(current_tp, entry_price + new_tp_dist)
                trail_sl = entry_price + max_favorable - a_amp * 0.15
                current_sl = max(current_sl, trail_sl)
            else:
                current_tp = min(current_tp, entry_price - new_tp_dist)
                trail_sl = entry_price - max_favorable + a_amp * 0.15
                current_sl = min(current_sl, trail_sl)

        if progress >= 1.5:
            new_tp_dist = a_amp * 1.60
            if a_dir == 1:
                current_tp = max(current_tp, entry_price + new_tp_dist)
                trail_sl = entry_price + max_favorable - a_amp * 0.10
                current_sl = max(current_sl, trail_sl)
            else:
                current_tp = min(current_tp, entry_price - new_tp_dist)
                trail_sl = entry_price - max_favorable + a_amp * 0.10
                current_sl = min(current_sl, trail_sl)

        if bars_elapsed > a_bars * 2 and progress < 0.50:
            if max_favorable > 0:
                shrink_tp_dist = max_favorable * 1.10
                if a_dir == 1:
                    current_tp = min(current_tp, entry_price + shrink_tp_dist)
                else:
                    current_tp = max(current_tp, entry_price - shrink_tp_dist)

        # Check TP
        if a_dir == 1:
            if h >= current_tp:
                exit_bar = bar
                exit_price = current_tp
                exit_reason = 'tp'
                break
        else:
            if l <= current_tp:
                exit_bar = bar
                exit_price = current_tp
                exit_reason = 'tp'
                break

    if a_dir == 1:
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price

    pnl_r = pnl / sl_distance if sl_distance > 0 else 0

    return {
        'exit_reason': exit_reason,
        'pnl': pnl,
        'pnl_r': pnl_r,
        'hold_bars': exit_bar - entry_bar,
    }


def process_pair_tf(args):
    """Run random baseline for one pair-TF."""
    pair, tf, n_trades, a_amps, a_bars_list, seed = args

    data = load_ohlcv(pair, tf)
    if data is None:
        return []

    opens, highs, lows, closes = data
    n_bars = len(closes)

    rng = np.random.RandomState(seed)

    results = []
    margin = 1000  # avoid edges

    for i in range(n_trades):
        # Random entry point (avoid first/last 1000 bars)
        entry_bar = rng.randint(margin, max(margin + 1, n_bars - margin))

        # Random direction
        a_dir = rng.choice([-1, 1])

        # Sample A amplitude from the actual distribution
        a_amp = a_amps[i % len(a_amps)]
        a_bars = a_bars_list[i % len(a_bars_list)]

        trade = simulate_trade(entry_bar, a_dir, a_amp, highs, lows, closes, a_bars)
        if trade is not None:
            results.append(trade)

    return results


def main():
    print("Loading ABC trade data to match distributions...")

    # Load actual ABC trades to get per-pair-TF distributions
    pair_tf_data = defaultdict(lambda: {'a_amps': [], 'a_bars': [], 'count': 0})

    with open(ABC_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['pair'], row['tf'])
            pair_tf_data[key]['a_amps'].append(float(row['a_amp']))
            pair_tf_data[key]['a_bars'].append(float(row['a_bars']))
            pair_tf_data[key]['count'] += 1

    print(f"Found {len(pair_tf_data)} pair-TF combinations")

    # Build tasks
    tasks = []
    total_trades = 0
    for (pair, tf), info in pair_tf_data.items():
        n = info['count']
        a_amps = np.array(info['a_amps'])
        a_bars = np.array(info['a_bars'])
        seed = hash(f"{pair}_{tf}") % (2**31)
        tasks.append((pair, tf, n, a_amps, a_bars, seed))
        total_trades += n

    print(f"Total random trades to simulate: {total_trades:,}")
    print(f"Running with 40 workers...")

    # Run
    all_results = []
    with Pool(40) as pool:
        for batch in pool.imap_unordered(process_pair_tf, tasks):
            all_results.extend(batch)

    print(f"\nCompleted: {len(all_results):,} random trades")

    # Analyze
    pnls = [r['pnl_r'] for r in all_results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')

    reasons = defaultdict(list)
    for r in all_results:
        reasons[r['exit_reason']].append(r['pnl_r'])

    print(f"\n{'='*70}")
    print(f"  RANDOM BASELINE RESULTS")
    print(f"{'='*70}")
    print(f"Total trades: {len(pnls):,}")
    print(f"Win rate: {len(wins)/len(pnls)*100:.1f}%")
    print(f"Avg PnL (R): {np.mean(pnls):.4f}")
    print(f"Median PnL (R): {np.median(pnls):.4f}")
    print(f"Std PnL (R): {np.std(pnls):.4f}")
    print(f"Avg Win (R): {np.mean(wins):.4f}" if wins else "No wins")
    print(f"Avg Loss (R): {np.mean(losses):.4f}" if losses else "No losses")
    print(f"Profit Factor: {pf:.2f}")

    print(f"\nBy exit reason:")
    for reason, rpnls in sorted(reasons.items()):
        wr = len([p for p in rpnls if p > 0]) / len(rpnls) * 100
        print(f"  {reason:10s}: n={len(rpnls):>8,}  WR={wr:.1f}%  "
              f"avg_R={np.mean(rpnls):.3f}  median_R={np.median(rpnls):.3f}")

    # Now compare with ABC
    print(f"\n{'='*70}")
    print(f"  COMPARISON: ABC vs RANDOM")
    print(f"{'='*70}")
    print(f"{'Metric':>20s} {'ABC':>12s} {'Random':>12s} {'Diff':>12s}")
    print(f"{'-'*56}")

    abc_stats = {
        'Win rate %': 49.7,
        'Avg PnL (R)': 0.5938,
        'Avg Win (R)': 1.5491,
        'Avg Loss (R)': -0.3489,
        'Profit Factor': 4.38,
    }

    rand_wr = len(wins)/len(pnls)*100
    rand_avgR = np.mean(pnls)
    rand_avgW = np.mean(wins) if wins else 0
    rand_avgL = np.mean(losses) if losses else 0

    rand_stats = {
        'Win rate %': rand_wr,
        'Avg PnL (R)': rand_avgR,
        'Avg Win (R)': rand_avgW,
        'Avg Loss (R)': rand_avgL,
        'Profit Factor': pf,
    }

    for metric in abc_stats:
        abc_v = abc_stats[metric]
        rand_v = rand_stats[metric]
        diff = abc_v - rand_v
        print(f"{metric:>20s} {abc_v:>12.4f} {rand_v:>12.4f} {diff:>+12.4f}")

    # Statistical test: is ABC mean_R significantly different from random mean_R?
    from math import erfc, sqrt
    abc_mean = 0.5938
    abc_std = 1.2179
    abc_n = 3649008
    rand_mean = np.mean(pnls)
    rand_std = np.std(pnls)
    rand_n = len(pnls)

    se = sqrt(abc_std**2 / abc_n + rand_std**2 / rand_n)
    t_stat = (abc_mean - rand_mean) / se if se > 0 else 0
    p_val = erfc(abs(t_stat) / sqrt(2))

    print(f"\nWelch's t-test (ABC mean_R vs Random mean_R):")
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_val:.6f}")
    print(f"  Significant at 0.01? {'YES' if p_val < 0.01 else 'NO'}")


if __name__ == '__main__':
    main()
