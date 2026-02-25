"""
Stage2 Delay Decay Curve
========================
For each ABC triple, simulate entry at delay=0,1,2,...,20 bars after pivot.
Measure how PnL decays as entry gets further from the theoretical turning point.

Key questions:
1. What's the decay shape? (linear? exponential? step function?)
2. At what delay is alpha still positive?
3. Does the decay rate depend on ABC conditions (slope_ratio, amp_ratio, time_ratio)?
4. For each delay, what TP/SL ratio is optimal?

This gives us the foundation for the entry-offset → exit-params transfer function.
"""

import os
import csv
import numpy as np
from collections import defaultdict
from multiprocessing import Pool

DATA_DIR = "/home/ubuntu/DataBase/base_kline"

MAX_DELAY = 25  # test delays 0..25


class OnlineZigZag:
    def __init__(self, deviation_pct=0.5, confirm_bars=5):
        self.dev = deviation_pct / 100.0
        self.confirm = confirm_bars
        self.pivots = []
        self._tent = None
        self._trend = 0
        self._ext_price = None
        self._ext_bar = None
        self._init_hi = None
        self._init_hi_bar = None
        self._init_lo = None
        self._init_lo_bar = None

    def process_bar(self, idx, high, low):
        if self._trend == 0:
            if self._init_hi is None:
                self._init_hi = high; self._init_hi_bar = idx
                self._init_lo = low;  self._init_lo_bar = idx
                return None
            if high > self._init_hi:
                self._init_hi = high; self._init_hi_bar = idx
            if low < self._init_lo:
                self._init_lo = low;  self._init_lo_bar = idx
            if self._init_lo > 0:
                rng = (self._init_hi - self._init_lo) / self._init_lo
            else:
                return None
            if rng < self.dev:
                return None
            if self._init_hi_bar > self._init_lo_bar:
                self.pivots.append((self._init_lo_bar, self._init_lo, -1, idx))
                self._trend = 1; self._ext_price = high; self._ext_bar = idx
            else:
                self.pivots.append((self._init_hi_bar, self._init_hi, 1, idx))
                self._trend = -1; self._ext_price = low; self._ext_bar = idx
            self._tent = None
            return None
        if self._trend == 1:
            if high > self._ext_price:
                self._ext_price = high; self._ext_bar = idx; self._tent = None
            drop = (self._ext_price - low) / self._ext_price if self._ext_price > 0 else 0
            if drop >= self.dev and self._tent is None:
                self._tent = (self._ext_bar, self._ext_price, 'H')
            if self._tent is not None:
                if high > self._tent[1]:
                    self._ext_price = high; self._ext_bar = idx; self._tent = None
                elif idx - self._tent[0] >= self.confirm:
                    pv = (self._tent[0], self._tent[1], +1, idx)
                    self.pivots.append(pv)
                    self._trend = -1; self._ext_price = low; self._ext_bar = idx
                    self._tent = None
                    return pv
            return None
        if self._trend == -1:
            if low < self._ext_price:
                self._ext_price = low; self._ext_bar = idx; self._tent = None
            rise = (high - self._ext_price) / self._ext_price if self._ext_price > 0 else 0
            if rise >= self.dev and self._tent is None:
                self._tent = (self._ext_bar, self._ext_price, 'L')
            if self._tent is not None:
                if low < self._tent[1]:
                    self._ext_price = low; self._ext_bar = idx; self._tent = None
                elif idx - self._tent[0] >= self.confirm:
                    pv = (self._tent[0], self._tent[1], -1, idx)
                    self.pivots.append(pv)
                    self._trend = 1; self._ext_price = high; self._ext_bar = idx
                    self._tent = None
                    return pv
            return None


def load_ohlcv(pair, tf):
    fpath = os.path.join(DATA_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fpath):
        return None
    highs, lows, closes = [], [], []
    with open(fpath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) < 6:
                continue
            highs.append(float(row[3]))
            lows.append(float(row[4]))
            closes.append(float(row[5]))
    return np.array(highs), np.array(lows), np.array(closes)


def simulate_at_delay(entry_bar, a_dir, a_amp, a_bars, tp_mult, sl_ratio,
                      highs, lows, closes):
    """
    Simulate a trade with given TP/SL parameters.
    TP = a_amp * tp_mult from entry
    SL = TP * sl_ratio from entry (reverse direction)
    With dynamic trailing stop.
    """
    if entry_bar >= len(closes) - 1:
        return None

    entry_price = closes[entry_bar]
    tp_distance = a_amp * tp_mult
    sl_distance = tp_distance * sl_ratio

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

        if a_dir == 1:
            favorable = h - entry_price
        else:
            favorable = entry_price - l
        if favorable > max_favorable:
            max_favorable = favorable

        progress = max_favorable / tp_distance if tp_distance > 0 else 0
        bars_elapsed = bar - entry_bar

        # SL check
        if a_dir == 1:
            if l <= current_sl:
                return {'pnl_r': (current_sl - entry_price) / sl_distance, 'reason': 'sl'}
        else:
            if h >= current_sl:
                return {'pnl_r': (entry_price - current_sl) / sl_distance, 'reason': 'sl'}

        # Dynamic: breakeven at 30%
        if progress >= 0.30 and not hit_breakeven:
            current_sl = entry_price
            hit_breakeven = True

        # Lock at 60%
        if progress >= 0.60:
            if a_dir == 1:
                current_sl = max(current_sl, entry_price + max_favorable * 0.50)
            else:
                current_sl = min(current_sl, entry_price - max_favorable * 0.50)

        # Trail at 100%+
        if progress >= 1.0:
            if a_dir == 1:
                current_sl = max(current_sl, entry_price + max_favorable - a_amp * 0.15)
            else:
                current_sl = min(current_sl, entry_price - max_favorable + a_amp * 0.15)

        # Stalling
        if bars_elapsed > a_bars * 2 and progress < 0.50 and max_favorable > 0:
            if a_dir == 1:
                current_tp = min(current_tp, entry_price + max_favorable * 1.10)
            else:
                current_tp = max(current_tp, entry_price - max_favorable * 1.10)

        # TP check
        if a_dir == 1:
            if h >= current_tp:
                return {'pnl_r': (current_tp - entry_price) / sl_distance, 'reason': 'tp'}
        else:
            if l <= current_tp:
                return {'pnl_r': (entry_price - current_tp) / sl_distance, 'reason': 'tp'}

    # Timeout
    if a_dir == 1:
        pnl = closes[end_bar] - entry_price
    else:
        pnl = entry_price - closes[end_bar]
    return {'pnl_r': pnl / sl_distance, 'reason': 'timeout'}


def process_pair_tf(args):
    """For each ABC triple, simulate at multiple delays and TP/SL configs."""
    pair, tf, zz_configs = args
    data = load_ohlcv(pair, tf)
    if data is None:
        return {}

    highs, lows, closes = data
    n_bars = len(closes)

    # Results: delay → list of (pnl_r, slope_ratio, amp_ratio, time_ratio, confirm_delay)
    results = {d: [] for d in range(MAX_DELAY + 1)}
    # Also track by condition bucket
    cond_results = {}  # (delay, condition_bucket) → [pnl_r]

    for dev_pct, confirm in zz_configs:
        zz = OnlineZigZag(deviation_pct=dev_pct, confirm_bars=confirm)
        for i in range(n_bars):
            zz.process_bar(i, highs[i], lows[i])

        pvs = zz.pivots  # (bar, price, dir, confirm_bar)
        if len(pvs) < 4:
            continue

        for i in range(len(pvs) - 3):
            p0_bar, p0_price, _, _ = pvs[i]
            p1_bar, p1_price, _, _ = pvs[i+1]
            p2_bar, p2_price, _, p2_conf = pvs[i+2]
            p3_bar, p3_price, _, _ = pvs[i+3]

            a_bars = p1_bar - p0_bar
            a_amp = abs(p1_price - p0_price)
            a_dir = 1 if p1_price > p0_price else -1
            b_bars = p2_bar - p1_bar
            b_amp = abs(p2_price - p1_price)

            if a_bars <= 0 or b_bars <= 0 or a_amp <= 0:
                continue

            a_slope = a_amp / a_bars
            b_slope = b_amp / b_bars
            slope_ratio = b_slope / a_slope if a_slope > 0 else 0
            amp_ratio = b_amp / a_amp
            time_ratio = b_bars / a_bars
            confirm_delay = p2_conf - p2_bar

            # Condition bucket: slope_ratio
            if slope_ratio < 0.3:
                cond = 'sr<0.3'
            elif slope_ratio < 0.7:
                cond = 'sr0.3-0.7'
            elif slope_ratio < 1.5:
                cond = 'sr0.7-1.5'
            else:
                cond = 'sr>1.5'

            # Simulate at each delay from pivot_bar
            for delay in range(MAX_DELAY + 1):
                entry_bar = p2_bar + delay
                if entry_bar >= n_bars - 1:
                    break

                # Adaptive TP/SL: as delay increases, reduce TP target
                # because we've already lost some of the move
                # For now, keep fixed to measure raw decay
                trade = simulate_at_delay(
                    entry_bar, a_dir, a_amp, a_bars,
                    tp_mult=0.80, sl_ratio=0.40,
                    highs=highs, lows=lows, closes=closes
                )
                if trade is not None:
                    results[delay].append({
                        'pnl_r': trade['pnl_r'],
                        'slope_ratio': slope_ratio,
                        'amp_ratio': amp_ratio,
                        'time_ratio': time_ratio,
                        'confirm_delay': confirm_delay,
                        'cond': cond,
                    })

    return results


def merge_results(all_batch_results):
    """Merge results from all workers."""
    merged = {d: [] for d in range(MAX_DELAY + 1)}
    for batch in all_batch_results:
        for d in range(MAX_DELAY + 1):
            if d in batch:
                merged[d].extend(batch[d])
    return merged


def main():
    print("=" * 80)
    print("  DELAY DECAY CURVE ANALYSIS")
    print("  Entry at pivot_bar + delay (delay = 0..25)")
    print("  Fixed TP=A*0.80, SL=TP*0.40, with dynamic trailing")
    print("=" * 80)

    pairs = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.csv'):
            parts = fname.replace('.csv', '').rsplit('_', 1)
            if len(parts) == 2:
                pairs.append(parts[0])
    pairs = sorted(set(pairs))

    tfs = ['H1', 'M30', 'M15']
    zz_configs = [(0.5, 5), (1.0, 5)]

    tasks = [(pair, tf, zz_configs) for pair in pairs for tf in tfs]
    print(f"Tasks: {len(tasks)} ({len(pairs)} pairs × {len(tfs)} TFs × {len(zz_configs)} ZZ)")
    print(f"Running with 40 workers...\n")

    all_results = []
    with Pool(40) as pool:
        for batch in pool.imap_unordered(process_pair_tf, tasks):
            all_results.append(batch)

    results = merge_results(all_results)

    # ── 1. Overall Decay Curve ──
    print("=" * 80)
    print("  1. OVERALL DELAY DECAY CURVE")
    print("=" * 80)
    print(f"\n{'Delay':>6s} {'N':>9s} {'WR%':>7s} {'AvgR':>8s} {'MedR':>8s} "
          f"{'PF':>7s} {'AvgWin':>8s} {'AvgLoss':>8s} {'Alpha':>8s}")
    print("-" * 80)

    baseline_pnl = None
    for delay in range(MAX_DELAY + 1):
        trades = results[delay]
        if not trades:
            continue
        pnls = [t['pnl_r'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999
        avg_r = np.mean(pnls)
        if baseline_pnl is None:
            baseline_pnl = avg_r

        alpha = avg_r  # alpha vs zero (random = ~0)
        marker = " ← pivot" if delay == 0 else ""
        marker = " ← confirm≈5" if delay == 5 else marker
        marker = " ← confirm≈9" if delay == 9 else marker

        print(f"{delay:>6d} {len(pnls):>9,} {len(wins)/len(pnls)*100:>7.1f} "
              f"{avg_r:>8.4f} {np.median(pnls):>8.4f} {pf:>7.2f} "
              f"{np.mean(wins):>8.4f} {np.mean(losses):>8.4f} "
              f"{alpha:>+8.4f}{marker}")

    # ── 2. Decay by Condition (slope_ratio buckets) ──
    print("\n" + "=" * 80)
    print("  2. DECAY BY CONDITION (slope_ratio buckets)")
    print("=" * 80)

    conds = ['sr<0.3', 'sr0.3-0.7', 'sr0.7-1.5', 'sr>1.5']
    for cond in conds:
        print(f"\n  --- {cond} ---")
        print(f"  {'Delay':>6s} {'N':>8s} {'WR%':>7s} {'AvgR':>8s} {'PF':>7s}")
        print(f"  {'-'*40}")
        for delay in range(MAX_DELAY + 1):
            trades = [t for t in results[delay] if t['cond'] == cond]
            if len(trades) < 100:
                continue
            pnls = [t['pnl_r'] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999
            print(f"  {delay:>6d} {len(pnls):>8,} {len(wins)/len(pnls)*100:>7.1f} "
                  f"{np.mean(pnls):>8.4f} {pf:>7.2f}")

    # ── 3. Where does alpha go to zero? ──
    print("\n" + "=" * 80)
    print("  3. ALPHA ZERO-CROSSING ANALYSIS")
    print("=" * 80)

    for cond in ['ALL'] + conds:
        delays_positive = []
        for delay in range(MAX_DELAY + 1):
            if cond == 'ALL':
                trades = results[delay]
            else:
                trades = [t for t in results[delay] if t['cond'] == cond]
            if len(trades) < 100:
                continue
            pnls = [t['pnl_r'] for t in trades]
            avg_r = np.mean(pnls)
            if avg_r > 0:
                delays_positive.append(delay)

        if delays_positive:
            print(f"  {cond:>12s}: alpha > 0 at delays {delays_positive}")
            print(f"               max profitable delay = {max(delays_positive)}")
        else:
            print(f"  {cond:>12s}: alpha never positive")

    # ── 4. Optimal TP/SL for different delays ──
    # Re-simulate delay=0 and delay=confirm with different TP/SL
    print("\n" + "=" * 80)
    print("  4. NOTE: To find optimal TP/SL per delay, run delay_curve_tpsl.py")
    print("     (this would add another dimension, making this script too slow)")
    print("=" * 80)

    print(f"\n{'='*80}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
