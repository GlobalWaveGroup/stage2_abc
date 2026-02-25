"""
Stage2 ABC Wave Collector
========================
Scan all 48 pairs × 3 TFs (H1, M30, M15) with online zigzag (zero look-ahead).
For every consecutive triple of zigzag legs (A, B, C), record full features
and simulate a trade with fixed SL/TP and dynamic TP adjustment.

Output: one large CSV with all ABC triples and their outcomes.

Usage:
    python abc_collector.py [--pairs EURUSD,GBPUSD] [--tfs H1,M30] [--workers 30]
"""

import os
import sys
import csv
import argparse
import numpy as np
from multiprocessing import Pool
from pathlib import Path

# ── Online ZigZag (embedded to avoid import issues on m5) ──────────────

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
                self.pivots.append((self._init_lo_bar, self._init_lo, -1))
                self._trend = 1
                self._ext_price = high
                self._ext_bar = idx
            else:
                self.pivots.append((self._init_hi_bar, self._init_hi, 1))
                self._trend = -1
                self._ext_price = low
                self._ext_bar = idx
            self._tent = None
            return None

        if self._trend == 1:
            if high > self._ext_price:
                self._ext_price = high
                self._ext_bar = idx
                self._tent = None
            drop = (self._ext_price - low) / self._ext_price if self._ext_price > 0 else 0
            if drop >= self.dev and self._tent is None:
                self._tent = (self._ext_bar, self._ext_price, 'H')
            if self._tent is not None:
                if high > self._tent[1]:
                    self._ext_price = high
                    self._ext_bar = idx
                    self._tent = None
                elif idx - self._tent[0] >= self.confirm:
                    pv = (self._tent[0], self._tent[1], +1)
                    self.pivots.append(pv)
                    self._trend = -1
                    self._ext_price = low
                    self._ext_bar = idx
                    self._tent = None
                    return (pv[0], pv[1], pv[2], idx)
            return None

        if self._trend == -1:
            if low < self._ext_price:
                self._ext_price = low
                self._ext_bar = idx
                self._tent = None
            rise = (high - self._ext_price) / self._ext_price if self._ext_price > 0 else 0
            if rise >= self.dev and self._tent is None:
                self._tent = (self._ext_bar, self._ext_price, 'L')
            if self._tent is not None:
                if low < self._tent[1]:
                    self._ext_price = low
                    self._ext_bar = idx
                    self._tent = None
                elif idx - self._tent[0] >= self.confirm:
                    pv = (self._tent[0], self._tent[1], -1)
                    self.pivots.append(pv)
                    self._trend = 1
                    self._ext_price = high
                    self._ext_bar = idx
                    self._tent = None
                    return (pv[0], pv[1], pv[2], idx)
            return None


# ── Data Loading ───────────────────────────────────────────────────────

DATA_DIR = "/home/ubuntu/DataBase/base_kline"

def load_ohlcv(pair, tf):
    """Load raw OHLCV from TSV. Returns (dates, opens, highs, lows, closes)."""
    fpath = os.path.join(DATA_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fpath):
        return None
    dates = []
    opens, highs, lows, closes = [], [], [], []
    with open(fpath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for row in reader:
            if len(row) < 6:
                continue
            dates.append(f"{row[0]} {row[1]}")
            opens.append(float(row[2]))
            highs.append(float(row[3]))
            lows.append(float(row[4]))
            closes.append(float(row[5]))
    return (dates,
            np.array(opens), np.array(highs),
            np.array(lows), np.array(closes))


# ── ABC Feature Extraction ────────────────────────────────────────────

def extract_abc_features(pivots):
    """
    Given a list of confirmed pivots [(bar, price, direction), ...],
    extract all consecutive ABC triples.

    A = pivots[i] → pivots[i+1]  (the trend leg)
    B = pivots[i+1] → pivots[i+2]  (the retracement)
    C = pivots[i+2] → pivots[i+3]  (the continuation — what actually happened)

    Returns list of dicts with features.
    """
    triples = []
    for i in range(len(pivots) - 3):
        p0 = pivots[i]    # A start
        p1 = pivots[i+1]  # A end / B start
        p2 = pivots[i+2]  # B end / C start (= entry point)
        p3 = pivots[i+3]  # C end (actual outcome)

        # A leg
        a_bars = p1[0] - p0[0]
        a_amp = abs(p1[1] - p0[1])
        a_dir = 1 if p1[1] > p0[1] else -1  # +1=up, -1=down

        # B leg
        b_bars = p2[0] - p1[0]
        b_amp = abs(p2[1] - p1[1])

        # C leg (actual)
        c_bars = p3[0] - p2[0]
        c_amp = abs(p3[1] - p2[1])
        c_dir = 1 if p3[1] > p2[1] else -1

        # Skip degenerate
        if a_bars <= 0 or b_bars <= 0 or a_amp <= 0:
            continue

        # Ratios
        amp_ratio = b_amp / a_amp                          # B/A amplitude ratio
        time_ratio = b_bars / a_bars                       # B/A time ratio
        a_slope = a_amp / a_bars                           # A slope (price/bar)
        b_slope = b_amp / b_bars                           # B slope
        slope_ratio = b_slope / a_slope if a_slope > 0 else 0  # B/A slope ratio

        # C follows A direction?
        c_follows_a = 1 if c_dir == a_dir else 0

        # C amplitude relative to A
        c_a_ratio = c_amp / a_amp if a_amp > 0 else 0

        triples.append({
            # Identifiers
            'a_start_bar': p0[0],
            'a_start_price': p0[1],
            'a_end_bar': p1[0],
            'a_end_price': p1[1],
            'b_end_bar': p2[0],
            'b_end_price': p2[1],
            'c_end_bar': p3[0],
            'c_end_price': p3[1],

            # A features
            'a_dir': a_dir,
            'a_amp': a_amp,
            'a_bars': a_bars,
            'a_slope': a_slope,

            # B features
            'b_amp': b_amp,
            'b_bars': b_bars,
            'b_slope': b_slope,

            # Ratios (key features for analysis)
            'amp_ratio': amp_ratio,       # B/A amplitude
            'time_ratio': time_ratio,     # B/A time
            'slope_ratio': slope_ratio,   # B/A slope

            # C outcome (what we want to predict/exploit)
            'c_dir': c_dir,
            'c_amp': c_amp,
            'c_bars': c_bars,
            'c_follows_a': c_follows_a,
            'c_a_ratio': c_a_ratio,
        })

    return triples


# ── Trade Simulation ──────────────────────────────────────────────────

def simulate_trade(triple, highs, lows, closes):
    """
    Simulate a trade entering at B end (C start), direction = A direction.

    Entry: close at bar after B confirmation (b_end_bar + 1)
    TP initial: A_amp * 0.80 from entry
    SL: TP / 2.5 = A_amp * 0.32 from entry (reverse direction)

    Dynamic TP adjustment based on C' progress:
    - progress > 100%: expand TP to A*1.20, trailing stop
    - progress > 150%: expand TP to A*1.60, tighter trail
    - time > A_bars*2 but progress < 50%: shrink TP to C'*1.1

    Returns trade result dict.
    """
    entry_bar = triple['b_end_bar'] + 1
    if entry_bar >= len(closes):
        return None

    a_amp = triple['a_amp']
    a_dir = triple['a_dir']  # +1 = go long, -1 = go short
    entry_price = closes[entry_bar]
    a_bars = triple['a_bars']

    # Initial TP/SL
    tp_distance = a_amp * 0.80
    sl_distance = tp_distance / 2.5  # = a_amp * 0.32

    if a_dir == 1:  # long
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
    else:  # short
        tp_price = entry_price - tp_distance
        sl_price = entry_price + sl_distance

    # Track C' progress
    max_favorable = 0.0    # max excursion in trade direction
    current_tp = tp_price
    current_sl = sl_price
    hit_breakeven = False

    max_bars = max(int(a_bars * 8), 500)  # max hold time
    end_bar = min(entry_bar + max_bars, len(closes) - 1)

    exit_bar = end_bar
    exit_price = closes[end_bar]
    exit_reason = 'timeout'

    for bar in range(entry_bar + 1, end_bar + 1):
        h = highs[bar]
        l = lows[bar]
        c = closes[bar]

        # Current favorable excursion
        if a_dir == 1:
            favorable = h - entry_price
            adverse = entry_price - l
        else:
            favorable = entry_price - l
            adverse = h - entry_price

        if favorable > max_favorable:
            max_favorable = favorable

        progress = max_favorable / tp_distance if tp_distance > 0 else 0
        bars_elapsed = bar - entry_bar

        # ── Check SL first (worst case within bar) ──
        if a_dir == 1:
            if l <= current_sl:
                exit_bar = bar
                exit_price = current_sl  # assume fill at SL
                exit_reason = 'sl'
                break
        else:
            if h >= current_sl:
                exit_bar = bar
                exit_price = current_sl
                exit_reason = 'sl'
                break

        # ── Dynamic adjustments ──
        # Phase 1: 30-60% progress → move SL to breakeven
        if progress >= 0.30 and not hit_breakeven:
            current_sl = entry_price  # breakeven
            hit_breakeven = True

        # Phase 2: 60-100% → lock in 50% of max favorable
        if progress >= 0.60:
            if a_dir == 1:
                lock_sl = entry_price + max_favorable * 0.50
                current_sl = max(current_sl, lock_sl)
            else:
                lock_sl = entry_price - max_favorable * 0.50
                current_sl = min(current_sl, lock_sl)

        # Phase 3: >100% → expand TP, trailing stop
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

        # Phase 4: >150% → further expand, tighter trail
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

        # Negative deviation: too slow
        if bars_elapsed > a_bars * 2 and progress < 0.50:
            # Shrink TP to current progress * 1.1
            if max_favorable > 0:
                shrink_tp_dist = max_favorable * 1.10
                if a_dir == 1:
                    current_tp = min(current_tp, entry_price + shrink_tp_dist)
                else:
                    current_tp = max(current_tp, entry_price - shrink_tp_dist)

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

    # Compute PnL
    if a_dir == 1:
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price

    pnl_r = pnl / sl_distance if sl_distance > 0 else 0  # in R-multiples

    return {
        'entry_bar': entry_bar,
        'entry_price': entry_price,
        'exit_bar': exit_bar,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl': pnl,
        'pnl_r': pnl_r,
        'max_favorable': max_favorable,
        'hold_bars': exit_bar - entry_bar,
        'tp_distance': tp_distance,
        'sl_distance': sl_distance,
    }


# ── Per-pair Worker ───────────────────────────────────────────────────

def process_pair_tf(args):
    """Process one pair-TF combination."""
    pair, tf, zz_configs = args
    data = load_ohlcv(pair, tf)
    if data is None:
        return []

    dates, opens, highs, lows, closes = data
    results = []

    for dev_pct, confirm in zz_configs:
        zz = OnlineZigZag(deviation_pct=dev_pct, confirm_bars=confirm)

        # Run zigzag
        for i in range(len(highs)):
            zz.process_bar(i, highs[i], lows[i])

        if len(zz.pivots) < 4:
            continue

        # Extract ABC triples
        triples = extract_abc_features(zz.pivots)

        for t in triples:
            # Simulate trade
            trade = simulate_trade(t, highs, lows, closes)
            if trade is None:
                continue

            row = {
                'pair': pair,
                'tf': tf,
                'zz_dev': dev_pct,
                'zz_confirm': confirm,
            }
            row.update(t)
            row.update(trade)
            results.append(row)

    print(f"  {pair}_{tf}: {len(results)} trades from {len(zz.pivots)} pivots")
    return results


# ── Main ──────────────────────────────────────────────────────────────

def get_all_pairs():
    """Get all unique pairs from data directory."""
    pairs = set()
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.csv'):
            # EURUSD_H1.csv → EURUSD
            parts = fname.replace('.csv', '').rsplit('_', 1)
            if len(parts) == 2:
                pairs.add(parts[0])
    return sorted(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=str, default=None,
                        help='Comma-separated pairs (default: all)')
    parser.add_argument('--tfs', type=str, default='H1,M30,M15',
                        help='Comma-separated timeframes')
    parser.add_argument('--workers', type=int, default=30)
    parser.add_argument('--output', type=str,
                        default='/home/ubuntu/stage2_abc/abc_all_triples.csv')
    args = parser.parse_args()

    pairs = args.pairs.split(',') if args.pairs else get_all_pairs()
    tfs = args.tfs.split(',')

    # ZigZag configs: test multiple sensitivities
    zz_configs = [
        (0.3, 3),   # sensitive
        (0.5, 5),   # medium
        (1.0, 5),   # conservative
    ]

    print(f"ABC Collector: {len(pairs)} pairs × {len(tfs)} TFs × {len(zz_configs)} ZZ configs")
    print(f"Pairs: {pairs[:5]}... ({len(pairs)} total)")
    print(f"TFs: {tfs}")
    print(f"ZZ configs: {zz_configs}")
    print(f"Workers: {args.workers}")
    print()

    # Build task list
    tasks = []
    for pair in pairs:
        for tf in tfs:
            tasks.append((pair, tf, zz_configs))

    print(f"Total tasks: {len(tasks)}")

    # Run parallel
    all_results = []
    with Pool(args.workers) as pool:
        for batch_results in pool.imap_unordered(process_pair_tf, tasks):
            all_results.extend(batch_results)

    print(f"\nTotal ABC triples collected: {len(all_results)}")

    if not all_results:
        print("ERROR: No results collected!")
        return

    # Save to CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = list(all_results[0].keys())
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"Saved to: {args.output}")

    # Quick summary
    pnls = [r['pnl_r'] for r in all_results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    print(f"\n=== Quick Summary (ALL, no filtering) ===")
    print(f"Trades: {len(pnls)}")
    print(f"Win rate: {len(wins)/len(pnls)*100:.1f}%")
    print(f"Avg PnL (R): {np.mean(pnls):.3f}")
    print(f"Median PnL (R): {np.median(pnls):.3f}")
    print(f"Avg win (R): {np.mean(wins):.3f}" if wins else "No wins")
    print(f"Avg loss (R): {np.mean(losses):.3f}" if losses else "No losses")
    print(f"Profit factor: {abs(sum(wins)/sum(losses)):.2f}" if losses and sum(losses) != 0 else "N/A")


if __name__ == '__main__':
    main()
