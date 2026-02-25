"""
Stage2 ABC V2 — Using Standard ZigZag(2,1,1) + Merge Levels
=============================================================
Key changes from V1:
1. ZigZag(2,1,1): depth=2, deviation=1 point, backstep=1
   - Near-zero confirmation delay (1 bar, not 5)
   - Captures every micro-swing at L0
2. Merge L0 → L1 → L2 → ... to get meaningful ABC structure
3. Entry at B-end price (the actual pivot price, not confirmed bar+1)
4. SL/TP recalculated from true entry price
5. Separate analysis of head start vs real edge

Runs on m5: 48 pairs × 3 TFs
"""

import os
import csv
import numpy as np
from multiprocessing import Pool
from collections import defaultdict
from math import erfc, sqrt

DATA_DIR = "/home/ubuntu/DataBase/base_kline"

# ═══════════════════════════════════════════════════════════════════════
# STANDARD ZIGZAG(2,1,1) — Online version
# ═══════════════════════════════════════════════════════════════════════

class ZigZag211:
    """
    MT5-standard ZigZag with depth=2, deviation=1 point, backstep=1.
    Online/streaming version — processes bar by bar, no lookahead.

    After each bar, check if a new pivot was confirmed 1 bar ago.
    Pivots are HIGH/LOW alternating.
    """

    def __init__(self, deviation_points=0.0001):
        """
        deviation_points: minimum price change to form a new leg.
        For zigzag(2,1,1), this is ~1 point = 0.0001 for most pairs.
        """
        self.dev = deviation_points
        self.pivots = []  # list of (bar_idx, price, type) type: +1=high, -1=low

        # State
        self._searching_for = 0  # +1=searching for high, -1=searching for low, 0=init
        self._last_high = -1e30
        self._last_high_bar = -1
        self._last_low = 1e30
        self._last_low_bar = -1
        self._prev_high = 0.0
        self._prev_low = 0.0
        self._bar_count = 0

    def process_bar(self, idx, high, low):
        """
        Process one bar. Returns list of newly confirmed pivots
        (can be 0 or 1, rarely 2 at initialization).

        Each pivot: (bar_idx, price, direction)
        direction: +1 = high pivot (price went UP to this point)
                   -1 = low pivot (price went DOWN to this point)
        """
        self._bar_count += 1
        new_pivots = []

        if self._searching_for == 0:
            # Initialization: need at least 2 bars
            if self._bar_count == 1:
                self._last_high = high
                self._last_high_bar = idx
                self._last_low = low
                self._last_low_bar = idx
                return []

            # Update extremes
            if high > self._last_high:
                self._last_high = high
                self._last_high_bar = idx
            if low < self._last_low:
                self._last_low = low
                self._last_low_bar = idx

            # Check if we have enough separation
            if self._last_high - self._last_low > self.dev:
                if self._last_low_bar < self._last_high_bar:
                    # Low came first → start with low pivot, search for high
                    self.pivots.append((self._last_low_bar, self._last_low, -1))
                    self._searching_for = 1  # now search for high
                    self._prev_high = self._last_high
                    self._prev_low = self._last_low
                else:
                    # High came first
                    self.pivots.append((self._last_high_bar, self._last_high, +1))
                    self._searching_for = -1  # now search for low
                    self._prev_high = self._last_high
                    self._prev_low = self._last_low
            return []

        # ── Normal operation ──

        if self._searching_for == 1:
            # Searching for HIGH pivot
            # backstep=1: if previous bar had a higher high than current bar,
            # that previous bar is a confirmed high pivot

            if high > self._last_high:
                self._last_high = high
                self._last_high_bar = idx
            else:
                # Current bar didn't exceed last high
                # Check if last_high qualifies as a pivot
                if self._last_high_bar == idx - 1:  # backstep=1: confirmed after 1 bar
                    last_pivot_price = self.pivots[-1][1] if self.pivots else 0
                    if self._last_high - last_pivot_price > self.dev:
                        pv = (self._last_high_bar, self._last_high, +1)
                        self.pivots.append(pv)
                        new_pivots.append(pv)
                        self._searching_for = -1
                        self._last_low = low
                        self._last_low_bar = idx

        elif self._searching_for == -1:
            # Searching for LOW pivot
            if low < self._last_low:
                self._last_low = low
                self._last_low_bar = idx
            else:
                if self._last_low_bar == idx - 1:
                    last_pivot_price = self.pivots[-1][1] if self.pivots else 0
                    if last_pivot_price - self._last_low > self.dev:
                        pv = (self._last_low_bar, self._last_low, -1)
                        self.pivots.append(pv)
                        new_pivots.append(pv)
                        self._searching_for = 1
                        self._last_high = high
                        self._last_high_bar = idx

        return new_pivots


def merge_pivots(pivots):
    """
    Merge L0 pivots into higher levels.
    
    Rule: For 3 consecutive points (H-L-H or L-H-L):
    - H-L-H: if p3 >= p1 (higher high), merge by removing p2 (the low)
      and keeping the higher of p1, p3
    - L-H-L: if p3 <= p1 (lower low), merge by removing p2 (the high)
      and keeping the lower of p1, p3

    Returns: merged pivot list (one level up)
    """
    if len(pivots) < 3:
        return pivots[:]

    result = []
    i = 0
    changed = False

    while i < len(pivots):
        if i + 2 >= len(pivots):
            result.extend(pivots[i:])
            break

        p1 = pivots[i]      # (bar, price, dir)
        p2 = pivots[i + 1]
        p3 = pivots[i + 2]

        can_merge = False
        if p1[2] == 1 and p2[2] == -1 and p3[2] == 1:
            # H-L-H: merge if p3 >= p1 (higher high)
            if p3[1] >= p1[1]:
                can_merge = True
        elif p1[2] == -1 and p2[2] == 1 and p3[2] == -1:
            # L-H-L: merge if p3 <= p1 (lower low)
            if p3[1] <= p1[1]:
                can_merge = True

        if can_merge:
            # Keep the more extreme point
            if p1[2] == 1:  # H-L-H
                kept = p3 if p3[1] > p1[1] else p1
            else:  # L-H-L
                kept = p3 if p3[1] < p1[1] else p1
            result.append(kept)
            i += 3
            changed = True
        else:
            result.append(p1)
            i += 1

    return result, changed


def merge_to_level(pivots, target_level):
    """Merge repeatedly until target level or no more merges possible."""
    current = pivots[:]
    for lv in range(target_level):
        merged, changed = merge_pivots(current)
        if not changed:
            break
        current = merged
    return current


# ═══════════════════════════════════════════════════════════════════════
# SCORING (same as before)
# ═══════════════════════════════════════════════════════════════════════

def compute_entry_score(slope_ratio, time_ratio, amp_ratio):
    slope_score = np.clip(1.0 - slope_ratio * 0.8, 0.0, 1.0)
    time_score = np.clip((np.log1p(time_ratio) - 0.2) / 2.2, 0.0, 1.0)
    if amp_ratio < 0.3: amp_score = amp_ratio / 0.3 * 0.4
    elif amp_ratio < 0.7: amp_score = 0.4 + (amp_ratio - 0.3) / 0.4 * 0.6
    elif amp_ratio <= 1.0: amp_score = 1.0
    elif amp_ratio <= 1.5: amp_score = 1.0 - (amp_ratio - 1.0) / 0.5 * 0.3
    else: amp_score = max(0.3, 0.7 - (amp_ratio - 1.5) * 0.2)
    return np.clip(0.40 * slope_score + 0.35 * time_score + 0.25 * amp_score, 0.0, 1.0)

def score_to_params(score):
    if score < 0.30: return None
    if score < 0.50:
        t = (score - 0.30) / 0.20
        return (0.5 + t * 0.2, 0.60 + t * 0.05, 0.45 - t * 0.03)
    elif score < 0.70:
        t = (score - 0.50) / 0.20
        return (0.7 + t * 0.3, 0.65 + t * 0.10, 0.42 - t * 0.04)
    elif score < 0.85:
        t = (score - 0.70) / 0.15
        return (1.0 + t * 0.5, 0.75 + t * 0.15, 0.38 - t * 0.03)
    else:
        t = min((score - 0.85) / 0.15, 1.0)
        return (1.5 + t * 0.5, 0.90 + t * 0.20, 0.35 - t * 0.03)


# ═══════════════════════════════════════════════════════════════════════
# TRADE SIMULATION — entry at close of B-end bar (the pivot bar itself)
# ═══════════════════════════════════════════════════════════════════════

def simulate_trade(entry_bar, entry_price, a_dir, a_amp, a_bars, tp_mult, sl_ratio, score, highs, lows, closes):
    """Simulate with dynamic management. Entry at the actual pivot bar's close."""
    if entry_bar >= len(closes) - 1:
        return None

    tp_distance = a_amp * tp_mult
    sl_distance = tp_distance * sl_ratio

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
    expected_bars = a_bars * 1.2

    max_hold = max(int(a_bars * 10), 500)
    end_bar = min(entry_bar + max_hold, len(closes) - 1)

    exit_bar = end_bar
    exit_price = closes[end_bar]
    exit_reason = 'timeout'

    for bar in range(entry_bar + 1, end_bar + 1):
        h = highs[bar]
        l = lows[bar]
        bars_elapsed = bar - entry_bar

        if a_dir == 1:
            favorable = h - entry_price
        else:
            favorable = entry_price - l
        if favorable > max_favorable:
            max_favorable = favorable

        progress = max_favorable / tp_distance if tp_distance > 0 else 0
        time_frac = bars_elapsed / expected_bars if expected_bars > 0 else 1
        speed = progress / time_frac if time_frac > 0.05 else progress * 20

        # Check SL
        if a_dir == 1:
            if l <= current_sl:
                exit_bar = bar; exit_price = current_sl; exit_reason = 'sl'; break
        else:
            if h >= current_sl:
                exit_bar = bar; exit_price = current_sl; exit_reason = 'sl'; break

        # Dynamic adjustments
        if progress >= 0.30 and not hit_breakeven:
            current_sl = entry_price; hit_breakeven = True
        if progress >= 0.60:
            if a_dir == 1: current_sl = max(current_sl, entry_price + max_favorable * 0.50)
            else: current_sl = min(current_sl, entry_price - max_favorable * 0.50)
        if progress >= 1.0:
            delta = a_amp * 0.30
            if a_dir == 1:
                current_tp = max(current_tp, current_tp + delta)
                current_sl = max(current_sl, entry_price + max_favorable * 0.65)
            else:
                current_tp = min(current_tp, current_tp - delta)
                current_sl = min(current_sl, entry_price - max_favorable * 0.65)
        if progress >= 1.5:
            if a_dir == 1: current_sl = max(current_sl, entry_price + max_favorable * 0.75)
            else: current_sl = min(current_sl, entry_price - max_favorable * 0.75)
        if progress >= 2.0:
            if a_dir == 1: current_sl = max(current_sl, entry_price + max_favorable * 0.85)
            else: current_sl = min(current_sl, entry_price - max_favorable * 0.85)
        if bars_elapsed > expected_bars * 2.5 and progress < 0.40 and max_favorable > 0:
            shrink = entry_price + max_favorable * 1.05 * a_dir
            if a_dir == 1: current_tp = min(current_tp, shrink)
            else: current_tp = max(current_tp, shrink)

        # Check TP
        if a_dir == 1:
            if h >= current_tp:
                exit_bar = bar; exit_price = current_tp; exit_reason = 'tp'; break
        else:
            if l <= current_tp:
                exit_bar = bar; exit_price = current_tp; exit_reason = 'tp'; break

    if a_dir == 1:
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price

    return {
        'exit_bar': exit_bar,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl': pnl,
        'pnl_r': pnl / sl_distance if sl_distance > 0 else 0,
        'hold_bars': exit_bar - entry_bar,
    }


# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

def load_ohlcv(pair, tf):
    fpath = os.path.join(DATA_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fpath):
        return None
    dates, opens, highs, lows, closes = [], [], [], [], []
    with open(fpath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) < 6: continue
            dates.append(f"{row[0]} {row[1]}")
            opens.append(float(row[2]))
            highs.append(float(row[3]))
            lows.append(float(row[4]))
            closes.append(float(row[5]))
    return dates, np.array(opens), np.array(highs), np.array(lows), np.array(closes)


# Spread table (same as before)
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
    return (info['spread'] + 1.0) * info['pip']  # spread + 0.5 pip slippage each way

def get_dev_points(pair):
    """Get deviation in price units (1 point)."""
    info = PAIR_INFO.get(pair, {'pip': 0.0001})
    return info['pip']  # 1 point = 1 pip for simplicity


# ═══════════════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════════════

def process_pair_tf(args):
    pair, tf, merge_levels = args

    data = load_ohlcv(pair, tf)
    if data is None:
        return []

    dates, opens, highs, lows, closes = data
    n = len(closes)
    dev_points = get_dev_points(pair)
    spread_cost = get_spread_cost(pair)

    # Run zigzag(2,1,1) on full data (batch mode for speed)
    zz = ZigZag211(deviation_points=dev_points)
    for i in range(n):
        zz.process_bar(i, highs[i], lows[i])

    l0_pivots = zz.pivots
    if len(l0_pivots) < 10:
        return []

    all_trades = []

    for merge_lv in merge_levels:
        # Merge to target level
        if merge_lv == 0:
            pivots = l0_pivots[:]
        else:
            pivots = merge_to_level(l0_pivots, merge_lv)

        if len(pivots) < 4:
            continue

        in_trade_until = -1

        for i in range(len(pivots) - 2):
            p0 = pivots[i]
            p1 = pivots[i + 1]
            p2 = pivots[i + 2]

            a_bars = p1[0] - p0[0]
            a_amp = abs(p1[1] - p0[1])
            a_dir = 1 if p1[1] > p0[1] else -1

            b_bars = p2[0] - p1[0]
            b_amp = abs(p2[1] - p1[1])

            if a_bars <= 0 or b_bars <= 0 or a_amp <= 0:
                continue

            amp_ratio = b_amp / a_amp
            time_ratio = b_bars / a_bars
            a_slope = a_amp / a_bars
            b_slope = b_amp / b_bars
            slope_ratio = b_slope / a_slope if a_slope > 0 else 999

            score = compute_entry_score(slope_ratio, time_ratio, amp_ratio)
            params = score_to_params(score)
            if params is None:
                continue

            pos_mult, tp_mult, sl_ratio = params

            # Entry: close of the bar AFTER B-end pivot
            # B pivot is at p2[0], confirmed at p2[0]+1 (backstep=1)
            # We can act at bar p2[0]+1 (next bar's open ≈ this bar's close)
            entry_bar = p2[0] + 1
            if entry_bar >= n - 5 or entry_bar <= in_trade_until:
                continue

            entry_price = closes[entry_bar]

            # Skip if spread too large relative to TP
            tp_distance = a_amp * tp_mult
            if spread_cost / tp_distance > 0.15:
                continue

            # Apply spread cost
            half_spread = spread_cost / 2
            if a_dir == 1:
                entry_price_adj = entry_price + half_spread
            else:
                entry_price_adj = entry_price - half_spread

            trade = simulate_trade(
                entry_bar, entry_price_adj, a_dir, a_amp, a_bars,
                tp_mult, sl_ratio, score, highs, lows, closes
            )
            if trade is None:
                continue

            # Apply spread on exit
            if a_dir == 1:
                trade['pnl'] -= half_spread
            else:
                trade['pnl'] -= half_spread

            sl_dist = tp_distance * sl_ratio
            trade['pnl_r'] = trade['pnl'] / sl_dist if sl_dist > 0 else 0

            in_trade_until = trade['exit_bar']

            # Head start measurement: how far has price moved from B pivot price?
            b_pivot_price = p2[1]
            if a_dir == 1:
                head_start = entry_price - b_pivot_price  # positive = price already moved in C dir
            else:
                head_start = b_pivot_price - entry_price

            head_start_pct = head_start / a_amp if a_amp > 0 else 0

            year = int(dates[entry_bar][:4]) if entry_bar < n else 0

            all_trades.append({
                'pair': pair, 'tf': tf, 'merge_level': merge_lv,
                'entry_bar': entry_bar, 'entry_year': year,
                'entry_price': entry_price_adj,
                'b_pivot_price': b_pivot_price,
                'head_start': head_start,
                'head_start_pct': head_start_pct,
                'a_dir': a_dir, 'a_amp': a_amp, 'a_bars': a_bars,
                'score': score, 'pos_mult': pos_mult,
                'slope_ratio': slope_ratio, 'amp_ratio': amp_ratio, 'time_ratio': time_ratio,
                'pnl_r': trade['pnl_r'],
                'pnl_weighted': trade['pnl_r'] * pos_mult,
                'exit_reason': trade['exit_reason'],
                'hold_bars': trade['hold_bars'],
            })

    return all_trades


def get_all_pairs():
    pairs = set()
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.csv'):
            parts = fname.replace('.csv', '').rsplit('_', 1)
            if len(parts) == 2: pairs.add(parts[0])
    return sorted(pairs)


def compute_stats(trades, key='pnl_r'):
    if not trades: return None
    pnls = [t[key] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
    return {
        'n': len(pnls), 'wr': len(wins)/len(pnls)*100,
        'avg_r': np.mean(pnls), 'pf': pf,
        'total_r': sum(pnls),
    }


def main():
    pairs = get_all_pairs()
    tfs = ['H1', 'M30', 'M15']
    merge_levels = [2, 3, 4, 5]  # Test different merge depths

    print(f"ABC V2 — ZigZag(2,1,1) + Merge")
    print(f"  Pairs: {len(pairs)}, TFs: {tfs}")
    print(f"  Merge levels: {merge_levels}")
    print(f"  Entry: close of bar after B pivot (1-bar delay)")
    print()

    tasks = [(pair, tf, merge_levels) for pair in pairs for tf in tfs]
    print(f"Tasks: {len(tasks)}, running with 40 workers...")

    all_trades = []
    with Pool(40) as pool:
        for batch in pool.imap_unordered(process_pair_tf, tasks):
            all_trades.extend(batch)

    print(f"Total trades: {len(all_trades):,}")

    # ── Analysis by merge level ──
    print(f"\n{'='*90}")
    print(f"  RESULTS BY MERGE LEVEL")
    print(f"{'='*90}")
    print(f"{'Level':>6s} | {'--- ALL ---':^30s} | {'--- IS ≤2018 ---':^30s} | {'--- OOS >2018 ---':^30s}")
    print(f"{'':>6s} | {'n':>7s} {'WR':>5s} {'avgR':>7s} {'PF':>6s} | {'n':>7s} {'WR':>5s} {'avgR':>7s} {'PF':>6s} | {'n':>7s} {'WR':>5s} {'avgR':>7s} {'PF':>6s}")
    print("-" * 90)

    for lv in merge_levels:
        lv_trades = [t for t in all_trades if t['merge_level'] == lv]
        is_t = [t for t in lv_trades if t['entry_year'] <= 2018]
        oos_t = [t for t in lv_trades if t['entry_year'] > 2018]
        s_all = compute_stats(lv_trades)
        s_is = compute_stats(is_t)
        s_oos = compute_stats(oos_t)

        def fmt(s):
            if s is None: return f"{'---':>7s} {'---':>5s} {'---':>7s} {'---':>6s}"
            return f"{s['n']:>7,} {s['wr']:>5.1f} {s['avg_r']:>7.4f} {s['pf']:>6.2f}"

        print(f"  L{lv:>3d} | {fmt(s_all)} | {fmt(s_is)} | {fmt(s_oos)}")

    # ── Head start analysis per merge level ──
    print(f"\n{'='*90}")
    print(f"  HEAD START ANALYSIS BY MERGE LEVEL")
    print(f"{'='*90}")
    print(f"{'Level':>6s} {'n':>8s} {'mean_hs':>9s} {'med_hs':>9s} {'hs>0%':>7s} {'hs_as_%TP':>10s} {'adj_avgR':>9s} {'adj_PF':>8s}")
    print("-" * 75)

    for lv in merge_levels:
        lv_trades = [t for t in all_trades if t['merge_level'] == lv]
        if not lv_trades: continue

        hs = np.array([t['head_start_pct'] for t in lv_trades])
        pnls = np.array([t['pnl_r'] for t in lv_trades])

        # Adjusted PnL: subtract head start in R units
        adj_pnls = []
        for t in lv_trades:
            sl_dist = t['a_amp'] * 0.80 * 0.40  # approximate sl_distance
            if sl_dist > 0:
                adj_r = t['pnl_r'] - t['head_start'] / sl_dist
            else:
                adj_r = t['pnl_r']
            adj_pnls.append(adj_r)
        adj = np.array(adj_pnls)
        adj_wins = adj[adj > 0]
        adj_losses = adj[adj <= 0]
        adj_pf = abs(np.sum(adj_wins) / np.sum(adj_losses)) if len(adj_losses) > 0 and np.sum(adj_losses) != 0 else float('inf')

        pct_positive = np.sum(hs > 0) / len(hs) * 100
        hs_as_tp = np.mean(hs) / 0.80 * 100  # as % of TP (A*0.80)

        print(f"  L{lv:>3d} {len(hs):>8,} {np.mean(hs):>9.4f} {np.median(hs):>9.4f} "
              f"{pct_positive:>7.1f} {hs_as_tp:>10.1f}% {np.mean(adj):>9.4f} {adj_pf:>8.2f}")

    # ── Best merge level: detailed breakdown ──
    # Find level with best OOS adjusted performance
    best_lv = None
    best_adj_r = -999
    for lv in merge_levels:
        lv_oos = [t for t in all_trades if t['merge_level'] == lv and t['entry_year'] > 2018]
        if not lv_oos: continue
        avg_r = np.mean([t['pnl_r'] for t in lv_oos])
        if avg_r > best_adj_r:
            best_adj_r = avg_r
            best_lv = lv

    if best_lv:
        print(f"\n{'='*90}")
        print(f"  BEST LEVEL: L{best_lv} — YEARLY BREAKDOWN (position-weighted)")
        print(f"{'='*90}")
        best_trades = [t for t in all_trades if t['merge_level'] == best_lv]
        years = sorted(set(t['entry_year'] for t in best_trades if t['entry_year'] > 0))
        print(f"{'Year':>6s} {'n':>7s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s}")
        print("-" * 40)
        for y in years:
            yt = [t for t in best_trades if t['entry_year'] == y]
            s = compute_stats(yt, 'pnl_weighted')
            if s:
                marker = " *OOS*" if y > 2018 else ""
                print(f"{y:>6d} {s['n']:>7,} {s['wr']:>6.1f} {s['avg_r']:>8.4f} {s['pf']:>7.2f}{marker}")

        # Pair breakdown OOS
        print(f"\n  OOS Pair Positive Rate:")
        oos_trades = [t for t in best_trades if t['entry_year'] > 2018]
        pair_stats = defaultdict(list)
        for t in oos_trades:
            pair_stats[t['pair']].append(t['pnl_weighted'])
        n_pos = sum(1 for p, pnls in pair_stats.items() if np.mean(pnls) > 0)
        print(f"  {n_pos}/{len(pair_stats)} pairs positive OOS ({n_pos/len(pair_stats)*100:.0f}%)")

    # Save
    out = "/home/ubuntu/stage2_abc/abc_v2_trades.csv"
    if all_trades:
        fields = list(all_trades[0].keys())
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in all_trades: w.writerow(t)
        print(f"\nSaved {len(all_trades):,} trades to {out}")


if __name__ == '__main__':
    main()
