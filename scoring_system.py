"""
Stage2 ABC Scoring System
==========================
Complete adaptive trading system:

1. ENTRY SCORING MATRIX
   - Multiple parameters → composite score (0-100)
   - Score → position size multiplier, TP target, SL ratio

2. DYNAMIC HOLDING MATRIX
   - C' progress tracking with multiple parameters
   - Real-time adjustment of TP/SL/position

3. Full backtest with IS/OOS split

Trained on IS (2000-2018), validated on OOS (2019-2024).
"""

import os
import csv
import numpy as np
from collections import defaultdict
from multiprocessing import Pool

DATA_DIR = "/home/ubuntu/DataBase/base_kline"
ABC_FILE = "/home/ubuntu/stage2_abc/abc_all_triples.csv"


# ═══════════════════════════════════════════════════════════════════════
# 1. ENTRY SCORING MATRIX
# ═══════════════════════════════════════════════════════════════════════
#
# Based on OOS-validated feature importance:
#   slope_ratio: strongest (lower = better, monotonic IS & OOS)
#   time_ratio:  strong (higher = better)
#   amp_ratio:   moderate (sweet spot 0.7-1.0)
#   zz_dev:      strong (1.0 >> 0.3)
#
# Score components (each 0-25, total 0-100):

def score_slope_ratio(sr):
    """Lower slope_ratio = B is weaker = better. 0-25 points."""
    if sr <= 0.05:   return 25
    if sr <= 0.10:   return 23
    if sr <= 0.15:   return 21
    if sr <= 0.20:   return 19
    if sr <= 0.30:   return 16
    if sr <= 0.40:   return 13
    if sr <= 0.50:   return 11
    if sr <= 0.70:   return 8
    if sr <= 1.00:   return 5
    if sr <= 1.50:   return 3
    if sr <= 2.00:   return 1
    return 0


def score_time_ratio(tr):
    """Higher time_ratio = B takes longer = better. 0-25 points."""
    if tr >= 20.0:   return 25
    if tr >= 10.0:   return 23
    if tr >= 5.0:    return 20
    if tr >= 3.0:    return 17
    if tr >= 2.0:    return 14
    if tr >= 1.6:    return 12
    if tr >= 1.3:    return 10
    if tr >= 1.0:    return 8
    if tr >= 0.8:    return 6
    if tr >= 0.5:    return 4
    if tr >= 0.3:    return 2
    return 0


def score_amp_ratio(ar):
    """Sweet spot around 0.7-1.0. 0-25 points."""
    if 0.80 <= ar <= 1.00:   return 25
    if 0.70 <= ar < 0.80:    return 22
    if 1.00 < ar <= 1.15:    return 22
    if 0.60 <= ar < 0.70:    return 18
    if 1.15 < ar <= 1.50:    return 15
    if 0.50 <= ar < 0.60:    return 14
    if 0.40 <= ar < 0.50:    return 10
    if 1.50 < ar <= 2.00:    return 8
    if 0.30 <= ar < 0.40:    return 6
    if 0.20 <= ar < 0.30:    return 3
    if ar < 0.20:            return 1
    return 2  # > 2.0


def score_zz_quality(zz_dev):
    """Higher deviation = larger structure = more reliable. 0-25 points."""
    if zz_dev >= 1.0:    return 25
    if zz_dev >= 0.7:    return 18
    if zz_dev >= 0.5:    return 12
    if zz_dev >= 0.3:    return 6
    return 2


def compute_entry_score(trade):
    """Compute composite entry score (0-100)."""
    s1 = score_slope_ratio(trade['slope_ratio'])
    s2 = score_time_ratio(trade['time_ratio'])
    s3 = score_amp_ratio(trade['amp_ratio'])
    s4 = score_zz_quality(trade['zz_dev'])
    return s1 + s2 + s3 + s4


# ═══════════════════════════════════════════════════════════════════════
# 2. SCORE → TRADE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
#
# Score tiers:
#   90-100: A+  (best setups)
#   75-89:  A   (strong)
#   60-74:  B   (good)
#   45-59:  C   (average)
#   30-44:  D   (weak)
#   0-29:   F   (skip or minimal)

def get_trade_params(score, a_amp):
    """
    Given entry score and A amplitude, return trade parameters.

    Returns: (position_mult, tp_distance, sl_distance, sl_ratio)
    - position_mult: 0.0 (skip) to 3.0
    - tp_distance: target profit from entry
    - sl_distance: stop loss from entry
    - sl_ratio: SL/TP ratio used
    """
    if score >= 90:
        # A+: aggressive
        pos_mult = 3.0
        tp_mult = 1.00      # TP = A * 1.00
        sl_ratio = 0.30     # SL = TP * 0.30
    elif score >= 75:
        # A: strong
        pos_mult = 2.0
        tp_mult = 0.90
        sl_ratio = 0.33
    elif score >= 60:
        # B: good
        pos_mult = 1.5
        tp_mult = 0.80
        sl_ratio = 0.36
    elif score >= 45:
        # C: average
        pos_mult = 1.0
        tp_mult = 0.70
        sl_ratio = 0.40
    elif score >= 30:
        # D: weak
        pos_mult = 0.5
        tp_mult = 0.60
        sl_ratio = 0.45
    else:
        # F: skip
        pos_mult = 0.0
        tp_mult = 0.50
        sl_ratio = 0.50

    tp_distance = a_amp * tp_mult
    sl_distance = tp_distance * sl_ratio

    return pos_mult, tp_distance, sl_distance, sl_ratio


# ═══════════════════════════════════════════════════════════════════════
# 3. DYNAMIC HOLDING MATRIX
# ═══════════════════════════════════════════════════════════════════════
#
# During the trade, C' is tracked. Multiple parameters are monitored:
#   - progress: C'_current / TP_distance
#   - speed: progress / (bars_elapsed / a_bars)  (how fast vs expectation)
#   - volatility_shift: current bar range vs A's avg bar range
#
# Adjustments based on (progress, speed) matrix:

def dynamic_adjust(entry_price, a_dir, a_amp, current_tp, current_sl,
                   max_favorable, bars_elapsed, a_bars, tp_distance,
                   score, hit_breakeven):
    """
    Dynamic adjustment of TP/SL based on C' progress and speed.

    Returns: (new_tp, new_sl, new_hit_breakeven)
    """
    progress = max_favorable / tp_distance if tp_distance > 0 else 0
    expected_time = a_bars * 1.5   # expect C to take ~1.5x A's time
    speed = progress / (bars_elapsed / expected_time) if bars_elapsed > 0 and expected_time > 0 else 0

    new_tp = current_tp
    new_sl = current_sl
    new_be = hit_breakeven

    # ── Phase 0: Early (progress < 20%) ──
    # No adjustments, let the trade develop

    # ── Phase 1: Developing (20-50%) ──
    if progress >= 0.20 and not new_be:
        if speed >= 0.8:
            # Fast development → early breakeven
            new_sl = entry_price
            new_be = True
        elif progress >= 0.30:
            # Normal speed, but enough progress → breakeven
            new_sl = entry_price
            new_be = True

    # ── Phase 2: Progressing (50-80%) ──
    if progress >= 0.50:
        # Lock in portion of gains
        lock_pct = 0.40 if speed >= 1.0 else 0.30
        if a_dir == 1:
            lock_sl = entry_price + max_favorable * lock_pct
            new_sl = max(new_sl, lock_sl)
        else:
            lock_sl = entry_price - max_favorable * lock_pct
            new_sl = min(new_sl, lock_sl)

    # ── Phase 3: Near target (80-100%) ──
    if progress >= 0.80:
        lock_pct = 0.55 if speed >= 1.0 else 0.45
        if a_dir == 1:
            lock_sl = entry_price + max_favorable * lock_pct
            new_sl = max(new_sl, lock_sl)
        else:
            lock_sl = entry_price - max_favorable * lock_pct
            new_sl = min(new_sl, lock_sl)

    # ── Phase 4: Exceeding target (100-150%) — POSITIVE DEVIATION ──
    if progress >= 1.0:
        # Expand TP based on score
        if score >= 75:
            expand = 1.30   # high score → let it run more
        elif score >= 60:
            expand = 1.20
        else:
            expand = 1.10

        new_tp_dist = tp_distance * expand
        if a_dir == 1:
            new_tp = max(new_tp, entry_price + new_tp_dist)
            trail_dist = a_amp * 0.12
            trail_sl = entry_price + max_favorable - trail_dist
            new_sl = max(new_sl, trail_sl)
        else:
            new_tp = min(new_tp, entry_price - new_tp_dist)
            trail_dist = a_amp * 0.12
            trail_sl = entry_price - max_favorable + trail_dist
            new_sl = min(new_sl, trail_sl)

    # ── Phase 5: Far exceeding (>150%) — STRONG POSITIVE DEVIATION ──
    if progress >= 1.5:
        if score >= 75:
            expand = 1.80
        elif score >= 60:
            expand = 1.50
        else:
            expand = 1.30

        new_tp_dist = tp_distance * expand
        if a_dir == 1:
            new_tp = max(new_tp, entry_price + new_tp_dist)
            trail_dist = a_amp * 0.08
            trail_sl = entry_price + max_favorable - trail_dist
            new_sl = max(new_sl, trail_sl)
        else:
            new_tp = min(new_tp, entry_price - new_tp_dist)
            trail_dist = a_amp * 0.08
            trail_sl = entry_price - max_favorable + trail_dist
            new_sl = min(new_sl, trail_sl)

    # ── Negative deviation: stalling ──
    if bars_elapsed > a_bars * 2.5 and progress < 0.40:
        # Taking too long with too little progress → tighten TP
        if max_favorable > 0:
            shrink = max_favorable * 1.05
            if a_dir == 1:
                new_tp = min(new_tp, entry_price + shrink)
            else:
                new_tp = max(new_tp, entry_price - shrink)

    if bars_elapsed > a_bars * 4 and progress < 0.60:
        # Very slow → aggressive tighten
        if max_favorable > 0:
            shrink = max_favorable * 0.95
            if a_dir == 1:
                new_tp = min(new_tp, entry_price + shrink)
                lock_sl = entry_price + max_favorable * 0.60
                new_sl = max(new_sl, lock_sl)
            else:
                new_tp = max(new_tp, entry_price - shrink)
                lock_sl = entry_price - max_favorable * 0.60
                new_sl = min(new_sl, lock_sl)

    return new_tp, new_sl, new_be


# ═══════════════════════════════════════════════════════════════════════
# 4. TRADE SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def load_ohlcv(pair, tf):
    fpath = os.path.join(DATA_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fpath):
        return None, None
    dates, opens, highs, lows, closes = [], [], [], [], []
    with open(fpath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) < 6:
                continue
            dates.append(f"{row[0]} {row[1]}")
            opens.append(float(row[2]))
            highs.append(float(row[3]))
            lows.append(float(row[4]))
            closes.append(float(row[5]))
    return dates, (np.array(opens), np.array(highs), np.array(lows), np.array(closes))


def simulate_scored_trade(entry_bar, a_dir, a_amp, a_bars, score,
                          highs, lows, closes):
    """
    Full trade simulation with scoring system.
    """
    if entry_bar >= len(closes) - 1:
        return None

    pos_mult, tp_distance, sl_distance, sl_ratio = get_trade_params(score, a_amp)

    if pos_mult <= 0:
        return None  # Skip F-grade trades

    entry_price = closes[entry_bar]

    if tp_distance <= 0 or sl_distance <= 0:
        return None

    if a_dir == 1:
        current_tp = entry_price + tp_distance
        current_sl = entry_price - sl_distance
    else:
        current_tp = entry_price - tp_distance
        current_sl = entry_price + sl_distance

    max_favorable = 0.0
    hit_breakeven = False

    max_hold = max(int(a_bars * 8), 500)
    end_bar = min(entry_bar + max_hold, len(closes) - 1)

    exit_bar = end_bar
    exit_price = closes[end_bar]
    exit_reason = 'timeout'

    for bar in range(entry_bar + 1, end_bar + 1):
        h = highs[bar]
        l = lows[bar]
        bars_elapsed = bar - entry_bar

        # Track favorable excursion
        if a_dir == 1:
            favorable = h - entry_price
        else:
            favorable = entry_price - l

        if favorable > max_favorable:
            max_favorable = favorable

        # ── Check SL (before adjustments, use previous bar's SL) ──
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

        # ── Dynamic adjustments ──
        current_tp, current_sl, hit_breakeven = dynamic_adjust(
            entry_price, a_dir, a_amp, current_tp, current_sl,
            max_favorable, bars_elapsed, a_bars, tp_distance,
            score, hit_breakeven
        )

        # ── Check TP ──
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

    # PnL
    if a_dir == 1:
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price

    pnl_r = pnl / sl_distance if sl_distance > 0 else 0
    pnl_weighted = pnl_r * pos_mult  # position-weighted

    return {
        'entry_bar': entry_bar,
        'entry_price': entry_price,
        'exit_bar': exit_bar,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl': pnl,
        'pnl_r': pnl_r,
        'pnl_weighted': pnl_weighted,
        'pos_mult': pos_mult,
        'score': score,
        'tp_distance': tp_distance,
        'sl_distance': sl_distance,
        'sl_ratio': sl_ratio,
        'max_favorable': max_favorable,
        'hold_bars': exit_bar - entry_bar,
    }


# ═══════════════════════════════════════════════════════════════════════
# 5. FULL BACKTEST
# ═══════════════════════════════════════════════════════════════════════

def load_abc_data():
    """Load pre-collected ABC triples."""
    data = []
    float_cols = [
        'a_amp', 'a_bars', 'a_slope', 'a_dir',
        'b_amp', 'b_bars', 'b_slope',
        'amp_ratio', 'time_ratio', 'slope_ratio',
        'c_amp', 'c_bars', 'c_dir', 'c_follows_a', 'c_a_ratio',
        'entry_bar', 'zz_dev', 'zz_confirm',
        'pnl_r',  # original PnL for comparison
    ]
    with open(ABC_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = {}
            for k, v in row.items():
                if k in float_cols:
                    try:
                        r[k] = float(v)
                    except:
                        r[k] = 0.0
                else:
                    r[k] = v
            data.append(r)
    return data


def get_year_for_bar(dates, bar_idx):
    if 0 <= bar_idx < len(dates):
        return int(dates[bar_idx][:4])
    return 0


def process_pair_tf(args):
    """Process all trades for one pair-TF."""
    pair, tf, trades = args

    date_data = load_ohlcv(pair, tf)
    if date_data[0] is None:
        return []

    dates, (opens, highs, lows, closes) = date_data

    results = []
    for t in trades:
        score = compute_entry_score(t)
        entry_bar = int(t['entry_bar'])

        trade = simulate_scored_trade(
            entry_bar, int(t['a_dir']), t['a_amp'], t['a_bars'],
            score, highs, lows, closes
        )
        if trade is None:
            continue

        year = get_year_for_bar(dates, entry_bar)

        trade['pair'] = pair
        trade['tf'] = tf
        trade['year'] = year
        trade['zz_dev'] = t['zz_dev']
        trade['slope_ratio'] = t['slope_ratio']
        trade['amp_ratio'] = t['amp_ratio']
        trade['time_ratio'] = t['time_ratio']
        trade['original_pnl_r'] = t['pnl_r']
        results.append(trade)

    return results


def compute_stats(pnls, weights=None):
    if not pnls:
        return None
    pnls = np.array(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    pf = abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float('inf')

    # Sharpe (annualized, assuming ~5 trades/day across all pairs)
    if np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252 * 5)
    else:
        sharpe = 0

    # Max drawdown in R
    cumsum = np.cumsum(pnls if weights is None else np.array(pnls) * np.array(weights))
    peak = np.maximum.accumulate(cumsum)
    dd = peak - cumsum
    max_dd = np.max(dd) if len(dd) > 0 else 0

    return {
        'n': len(pnls),
        'wr': len(wins) / len(pnls) * 100,
        'avg_r': np.mean(pnls),
        'med_r': np.median(pnls),
        'std_r': np.std(pnls),
        'pf': pf,
        'sharpe': sharpe,
        'max_dd_r': max_dd,
        'total_r': np.sum(pnls),
        'avg_win': np.mean(wins) if len(wins) > 0 else 0,
        'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
    }


def print_stats(label, s):
    if s is None:
        print(f"  {label}: NO DATA")
        return
    print(f"  {label}: n={s['n']:>7,}  WR={s['wr']:.1f}%  avgR={s['avg_r']:.4f}  "
          f"PF={s['pf']:.2f}  Sharpe={s['sharpe']:.2f}  maxDD={s['max_dd_r']:.1f}R  "
          f"totalR={s['total_r']:.0f}")


def main():
    print("=" * 80)
    print("  STAGE2 ABC SCORING SYSTEM — FULL BACKTEST")
    print("=" * 80)

    # Load ABC data
    print("\nLoading ABC triples...")
    abc_data = load_abc_data()
    print(f"Loaded {len(abc_data):,} triples")

    # Group by pair-TF
    groups = defaultdict(list)
    for t in abc_data:
        groups[(t['pair'], t['tf'])].append(t)

    print(f"Unique pair-TF: {len(groups)}")

    # Build tasks
    tasks = [(pair, tf, trades) for (pair, tf), trades in groups.items()]

    # Run backtest
    print(f"\nRunning scored backtest ({len(tasks)} tasks, 40 workers)...")
    all_results = []
    with Pool(40) as pool:
        for batch in pool.imap_unordered(process_pair_tf, tasks):
            all_results.extend(batch)

    print(f"Total scored trades: {len(all_results):,}")

    # ── Overall ──
    print("\n" + "=" * 80)
    print("  OVERALL RESULTS (scored system)")
    print("=" * 80)

    pnls_r = [r['pnl_r'] for r in all_results]
    pnls_w = [r['pnl_weighted'] for r in all_results]
    print_stats("Unweighted (R)", compute_stats(pnls_r))
    print_stats("Weighted (R×pos)", compute_stats(pnls_w))

    # Compare with original (no scoring)
    orig_pnls = [r['original_pnl_r'] for r in all_results]
    print_stats("Original (no score)", compute_stats(orig_pnls))

    # ── By Score Tier ──
    print("\n" + "=" * 80)
    print("  BY SCORE TIER")
    print("=" * 80)

    tiers = [
        ('A+ (90-100)', lambda r: r['score'] >= 90),
        ('A  (75-89)',  lambda r: 75 <= r['score'] < 90),
        ('B  (60-74)',  lambda r: 60 <= r['score'] < 75),
        ('C  (45-59)',  lambda r: 45 <= r['score'] < 60),
        ('D  (30-44)',  lambda r: 30 <= r['score'] < 45),
        ('F  (<30)',    lambda r: r['score'] < 30),
    ]

    for name, filt in tiers:
        subset = [r for r in all_results if filt(r)]
        if not subset:
            print(f"  {name}: SKIPPED (pos_mult=0)")
            continue
        pnls = [r['pnl_r'] for r in subset]
        s = compute_stats(pnls)
        pos = subset[0]['pos_mult'] if subset else 0
        print(f"  {name} (pos={pos:.1f}x): n={s['n']:>7,}  WR={s['wr']:.1f}%  "
              f"avgR={s['avg_r']:.4f}  PF={s['pf']:.2f}  Sharpe={s['sharpe']:.2f}")

    # ── IS vs OOS ──
    print("\n" + "=" * 80)
    print("  IS vs OOS")
    print("=" * 80)

    is_trades = [r for r in all_results if 0 < r['year'] <= 2018]
    oos_trades = [r for r in all_results if r['year'] > 2018]

    print_stats("IS  (2000-2018) unweighted", compute_stats([r['pnl_r'] for r in is_trades]))
    print_stats("OOS (2019-2024) unweighted", compute_stats([r['pnl_r'] for r in oos_trades]))
    print_stats("IS  (2000-2018) weighted  ", compute_stats([r['pnl_weighted'] for r in is_trades]))
    print_stats("OOS (2019-2024) weighted  ", compute_stats([r['pnl_weighted'] for r in oos_trades]))

    # ── IS vs OOS by tier ──
    print("\n  By tier:")
    for name, filt in tiers:
        is_sub = [r for r in is_trades if filt(r)]
        oos_sub = [r for r in oos_trades if filt(r)]
        if not is_sub or not oos_sub:
            continue
        is_s = compute_stats([r['pnl_r'] for r in is_sub])
        oos_s = compute_stats([r['pnl_r'] for r in oos_sub])
        print(f"    {name}: IS n={is_s['n']:>6,} WR={is_s['wr']:.1f}% avgR={is_s['avg_r']:.3f} PF={is_s['pf']:.2f} | "
              f"OOS n={oos_s['n']:>6,} WR={oos_s['wr']:.1f}% avgR={oos_s['avg_r']:.3f} PF={oos_s['pf']:.2f}")

    # ── By Year (weighted) ──
    print("\n" + "=" * 80)
    print("  YEARLY PERFORMANCE (position-weighted)")
    print("=" * 80)

    years = sorted(set(r['year'] for r in all_results if r['year'] > 0))
    for y in years:
        yd = [r for r in all_results if r['year'] == y]
        s = compute_stats([r['pnl_weighted'] for r in yd])
        marker = " <<<OOS" if y > 2018 else ""
        print(f"  {y}{marker}: n={s['n']:>6,}  WR={s['wr']:.1f}%  avgR={s['avg_r']:.4f}  "
              f"PF={s['pf']:.2f}  totalR={s['total_r']:.0f}")

    # ── Cross-pair (weighted, OOS only) ──
    print("\n" + "=" * 80)
    print("  CROSS-PAIR OOS PERFORMANCE (weighted)")
    print("=" * 80)

    pair_results = defaultdict(list)
    for r in oos_trades:
        pair_results[r['pair']].append(r['pnl_weighted'])

    positive_pairs = 0
    total_pairs = 0
    for pair in sorted(pair_results.keys()):
        pnls = pair_results[pair]
        if len(pnls) < 30:
            continue
        total_pairs += 1
        s = compute_stats(pnls)
        if s['avg_r'] > 0:
            positive_pairs += 1
        print(f"  {pair:>10s}: n={s['n']:>5,}  WR={s['wr']:.1f}%  avgR={s['avg_r']:.4f}  PF={s['pf']:.2f}")

    print(f"\n  Pairs with positive OOS: {positive_pairs}/{total_pairs} ({positive_pairs/total_pairs*100:.0f}%)")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("  SYSTEM SUMMARY")
    print("=" * 80)

    oos_w = compute_stats([r['pnl_weighted'] for r in oos_trades])
    oos_u = compute_stats([r['pnl_r'] for r in oos_trades])

    print(f"  OOS (2019-2024) Unweighted:")
    print(f"    Trades: {oos_u['n']:,}")
    print(f"    Win Rate: {oos_u['wr']:.1f}%")
    print(f"    Avg PnL: {oos_u['avg_r']:.4f} R")
    print(f"    Profit Factor: {oos_u['pf']:.2f}")
    print(f"    Sharpe: {oos_u['sharpe']:.2f}")
    print(f"    Max Drawdown: {oos_u['max_dd_r']:.1f} R")
    print(f"    Avg Win: {oos_u['avg_win']:.3f} R  |  Avg Loss: {oos_u['avg_loss']:.3f} R")
    print(f"\n  OOS (2019-2024) Position-Weighted:")
    print(f"    Avg PnL: {oos_w['avg_r']:.4f} R")
    print(f"    Profit Factor: {oos_w['pf']:.2f}")
    print(f"    Sharpe: {oos_w['sharpe']:.2f}")
    print(f"    Total R: {oos_w['total_r']:.0f}")

    print(f"\n{'='*80}")
    print("  BACKTEST COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
