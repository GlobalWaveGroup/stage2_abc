"""
Stage2 ABC V4 — Intra-B Entry with Dynamic Management
======================================================
Combines:
- V3: Enter DURING B wave (no confirmation wait)
- V1: DynamicManager for trailing stop, profit locking, TP expansion

Key insight: V3 was all negative because fixed TP/SL with no management.
Entering during B means entering against current move — you NEED dynamic
management to:
1. Move SL to breakeven when price turns in your favor
2. Lock profits progressively as C develops
3. Expand TP when momentum is strong
4. Exit early when stagnating

Uses zigzag(2,1,1) + merge for A wave structure detection.
Entry scoring based on B-wave development features.

Tests multiple merge levels (L2-L5) and multiple timeframes.
"""

import os
import csv
import numpy as np
from multiprocessing import Pool
from collections import defaultdict

DATA_DIR = "/home/ubuntu/DataBase/base_kline"

# ═══════════════════════════════════════════════════════════════════════
# BATCH ZIGZAG(2,1,1)
# ═══════════════════════════════════════════════════════════════════════

def zigzag_211_batch(highs, lows, dev_points=0.0001):
    """Batch zigzag(2,1,1). Returns list of (bar_idx, price, direction)."""
    n = len(highs)
    if n < 3:
        return []

    pivots = []
    if highs[0] >= highs[1]:
        pivots.append((0, highs[0], 1))
        searching = -1
        last_low = lows[0]
        last_low_bar = 0
    else:
        pivots.append((0, lows[0], -1))
        searching = 1
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
            elif last_high_bar == i - 1:
                if last_high - pivots[-1][1] > dev_points:
                    pivots.append((last_high_bar, last_high, 1))
                    searching = -1
                    last_low = lows[i]
                    last_low_bar = i
        else:
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
# DYNAMIC IN-TRADE MANAGER (from V1, adapted for intra-B entry)
# ═══════════════════════════════════════════════════════════════════════

class DynamicManager:
    """
    Manages SL/TP adjustments during a trade.
    
    Adapted for intra-B entry:
    - entry_price is during B wave (not at B end)
    - TP is calculated from entry position toward C target
    - SL accounts for B potentially continuing further
    """

    def __init__(self, entry_price, a_dir, a_amp, a_bars, tp_dist, sl_dist, entry_score):
        self.entry_price = entry_price
        self.a_dir = a_dir
        self.a_amp = a_amp
        self.a_bars = a_bars
        self.entry_score = entry_score

        self.tp_distance = tp_dist
        self.sl_distance = sl_dist
        self.initial_sl_dist = sl_dist  # for R calculation

        if a_dir == 1:
            self.tp_price = entry_price + tp_dist
            self.sl_price = entry_price - sl_dist
        else:
            self.tp_price = entry_price - tp_dist
            self.sl_price = entry_price + sl_dist

        self.max_favorable = 0.0
        self.hit_breakeven = False
        self.expected_bars = a_bars * 1.5  # more generous for intra-B (need B to finish + C)

    def update(self, bar_idx, high, low, bars_elapsed):
        """
        Update dynamic levels. Returns (new_sl, new_tp, should_close, close_price).
        """
        # Track favorable excursion
        if self.a_dir == 1:
            favorable = high - self.entry_price
            adverse = self.entry_price - low
        else:
            favorable = self.entry_price - low
            adverse = high - self.entry_price

        if favorable > self.max_favorable:
            self.max_favorable = favorable

        progress = self.max_favorable / self.tp_distance if self.tp_distance > 0 else 0
        time_frac = bars_elapsed / self.expected_bars if self.expected_bars > 0 else 1
        speed = progress / time_frac if time_frac > 0.05 else progress * 20

        # ── Check SL hit ──
        if self.a_dir == 1:
            if low <= self.sl_price:
                return self.sl_price, self.tp_price, True, self.sl_price
        else:
            if high >= self.sl_price:
                return self.sl_price, self.tp_price, True, self.sl_price

        # ── Phase 1: breakeven at 30% progress ──
        if progress >= 0.30 and not self.hit_breakeven:
            self.sl_price = self.entry_price
            self.hit_breakeven = True
            if speed > 1.2:
                expand = min(0.10, (speed - 1.2) * 0.05)
                self._expand_tp(expand)

        # ── Phase 2: lock profits at 60% ──
        if progress >= 0.60:
            self._move_sl_to_lock(0.50)
            if speed > 1.5:
                expand = min(0.25, (speed - 1.5) * 0.10)
                self._expand_tp(expand)
            elif speed < 0.5 and bars_elapsed > self.expected_bars:
                contract = min(0.20, (0.5 - speed) * 0.15)
                self._contract_tp(contract)

        # ── Phase 3: trailing at 100% ──
        if progress >= 1.0:
            self._expand_tp(0.30)
            self._move_sl_to_lock(0.65)
            if self.entry_score > 0.7:
                extra_room = (self.entry_score - 0.7) * 0.5
                self._expand_tp(extra_room)

        # ── Phase 4: tight trail at 150% ──
        if progress >= 1.5:
            self._expand_tp(0.50)
            self._move_sl_to_lock(0.75)

        # ── Phase 5: prepare exit at 200% ──
        if progress >= 2.0:
            self._move_sl_to_lock(0.85)

        # ── Stagnation check ──
        if bars_elapsed > self.expected_bars * 2.5 and progress < 0.40:
            if self.max_favorable > 0:
                shrink_target = self.entry_price + self.max_favorable * 1.05 * self.a_dir
                if self.a_dir == 1:
                    self.tp_price = min(self.tp_price, shrink_target)
                else:
                    self.tp_price = max(self.tp_price, shrink_target)

        # ── Timeout ──
        if bars_elapsed > self.a_bars * 10:
            # Force close at mid price
            mid = (high + low) / 2
            return self.sl_price, self.tp_price, True, mid

        # ── Check TP hit ──
        if self.a_dir == 1:
            if high >= self.tp_price:
                return self.sl_price, self.tp_price, True, self.tp_price
        else:
            if low <= self.tp_price:
                return self.sl_price, self.tp_price, True, self.tp_price

        return self.sl_price, self.tp_price, False, None

    def _expand_tp(self, fraction):
        delta = self.a_amp * fraction
        if self.a_dir == 1:
            self.tp_price = max(self.tp_price, self.tp_price + delta)
        else:
            self.tp_price = min(self.tp_price, self.tp_price - delta)
        self.tp_distance = abs(self.tp_price - self.entry_price)

    def _contract_tp(self, fraction):
        delta = self.a_amp * fraction
        if self.a_dir == 1:
            new_tp = self.tp_price - delta
            self.tp_price = max(new_tp, self.entry_price + self.max_favorable * 0.3)
        else:
            new_tp = self.tp_price + delta
            self.tp_price = min(new_tp, self.entry_price - self.max_favorable * 0.3)
        self.tp_distance = abs(self.tp_price - self.entry_price)

    def _move_sl_to_lock(self, lock_pct):
        lock = self.entry_price + self.max_favorable * lock_pct * self.a_dir
        if self.a_dir == 1:
            self.sl_price = max(self.sl_price, lock)
        else:
            self.sl_price = min(self.sl_price, lock)


# ═══════════════════════════════════════════════════════════════════════
# ENTRY SCORING (adapted for intra-B features)
# ═══════════════════════════════════════════════════════════════════════

def compute_intra_b_score(b_slope_ratio, b_time_ratio, b_depth_ratio, b_decel_ratio):
    """
    Score the quality of an intra-B entry.
    
    Key features from V3 analysis:
    - b_slope_ratio: B slope / A slope — lower = B is weaker (good)
    - b_time_ratio: B bars / A bars — higher = B taking longer (good)  
    - b_depth_ratio: B retracement / A amplitude — sweet spot 0.3-0.7
    - b_decel_ratio: late B slope / early B slope — lower = B slowing (good)
    """
    # Slope score: lower is better (B weaker than A)
    slope_score = np.clip(1.0 - b_slope_ratio * 1.0, 0.0, 1.0)

    # Time score: higher is better (B exhausting itself)
    time_score = np.clip((np.log1p(b_time_ratio) - 0.1) / 2.0, 0.0, 1.0)

    # Depth score: peak at 0.38-0.62 (fib zones), decline outside
    if b_depth_ratio < 0.15:
        depth_score = b_depth_ratio / 0.15 * 0.3  # too shallow
    elif b_depth_ratio < 0.38:
        depth_score = 0.3 + (b_depth_ratio - 0.15) / 0.23 * 0.7
    elif b_depth_ratio <= 0.62:
        depth_score = 1.0  # fib sweet spot
    elif b_depth_ratio <= 1.0:
        depth_score = 1.0 - (b_depth_ratio - 0.62) / 0.38 * 0.4  # getting deep
    else:
        depth_score = max(0.2, 0.6 - (b_depth_ratio - 1.0) * 0.3)  # too deep

    # Decel score: lower is better (B slowing down)
    decel_score = np.clip(1.0 - b_decel_ratio * 0.7, 0.0, 1.0)

    # Weighted: slope and decel most important for intra-B
    score = 0.35 * slope_score + 0.25 * time_score + 0.20 * depth_score + 0.20 * decel_score

    return np.clip(score, 0.0, 1.0)


def score_to_params_v4(score, a_amp, b_depth_ratio, entry_price, a_end_price, a_dir):
    """
    Map score to trade parameters for intra-B entry.
    
    TP target: entry → (A_end + C_target)
    - C_target = fraction of A_amp beyond A_end
    - Total TP distance = B_remaining + C_target
    
    SL: from entry, in B direction
    - Higher score → tighter SL (more confidence B has ended)
    - Lower score → wider SL (B might go further)
    """
    if score < 0.25:
        return None  # skip

    # Position sizing
    if score < 0.40:
        pos_mult = 0.5
    elif score < 0.55:
        pos_mult = 0.75
    elif score < 0.70:
        pos_mult = 1.0
    elif score < 0.85:
        pos_mult = 1.5
    else:
        pos_mult = 2.0

    # C target as fraction of A_amp (beyond A_end, in A direction)
    # Higher score → more ambitious C target
    if score < 0.40:
        c_target_frac = 0.30  # conservative: just 30% of A
    elif score < 0.55:
        c_target_frac = 0.50
    elif score < 0.70:
        c_target_frac = 0.70
    elif score < 0.85:
        c_target_frac = 0.85
    else:
        c_target_frac = 1.00  # full A replication

    # TP distance from entry price
    # = distance back to A_end + C extension beyond A_end
    if a_dir == 1:
        # A went up, B went down, C should go up
        # entry is below A_end
        b_remaining = a_end_price - entry_price  # positive if entry below A_end
        tp_dist = max(b_remaining, 0) + a_amp * c_target_frac
    else:
        # A went down, B went up, C should go down
        b_remaining = entry_price - a_end_price  # positive if entry above A_end
        tp_dist = max(b_remaining, 0) + a_amp * c_target_frac

    # SL distance from entry price (in B direction)
    # Base: fraction of A_amp, adjusted by score
    # Higher score → tighter SL (B should be nearly done)
    # Also: at least b_depth_overshoot room
    if score < 0.40:
        sl_frac = 0.50  # generous SL
    elif score < 0.55:
        sl_frac = 0.40
    elif score < 0.70:
        sl_frac = 0.35
    elif score < 0.85:
        sl_frac = 0.30
    else:
        sl_frac = 0.25  # tight SL, very confident

    sl_dist = a_amp * sl_frac

    # Ensure minimum R:R of 1.5:1
    if tp_dist < sl_dist * 1.5:
        sl_dist = tp_dist / 1.5

    # Safety: minimum distances
    min_dist = a_amp * 0.05
    if tp_dist < min_dist or sl_dist < min_dist:
        return None

    return pos_mult, tp_dist, sl_dist, c_target_frac


# ═══════════════════════════════════════════════════════════════════════
# PAIR INFO
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

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


def get_year(date_str):
    return int(date_str[:4])


def get_all_pairs():
    pairs = set()
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.csv'):
            parts = fname.replace('.csv', '').rsplit('_', 1)
            if len(parts) == 2: pairs.add(parts[0])
    return sorted(pairs)


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE — INTRA-B WITH DYNAMIC MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

def backtest_pair_tf(args):
    """Full backtest for one pair-TF-merge_level combo."""
    pair, tf, merge_level = args

    data = load_ohlcv(pair, tf)
    if data is None:
        return []

    dates, highs, lows, closes = data
    n_bars = len(closes)
    dev = get_dev(pair)
    spread = get_spread_cost(pair)
    half_spread = spread / 2

    # Run zigzag(2,1,1)
    l0 = zigzag_211_batch(highs, lows, dev)
    if len(l0) < 5:
        return []

    # Merge to target level
    pivots = merge_to_level(l0, merge_level)
    if len(pivots) < 3:
        return []

    trades = []
    active_trade_end = 0  # no-overlap: track when current trade ends

    for pi in range(len(pivots) - 1):
        p0 = pivots[pi]      # A start
        p1 = pivots[pi + 1]  # A end = B start

        a_bars = p1[0] - p0[0]
        a_amp = abs(p1[1] - p0[1])
        a_dir = 1 if p1[1] > p0[1] else -1
        a_end_bar = p1[0]
        a_end_price = p1[1]

        if a_bars < 2 or a_amp <= 0:
            continue

        a_slope = a_amp / a_bars

        # Track B wave bar by bar
        max_b_bars = min(int(a_bars * 10), 2000)
        max_b_depth = a_amp * 1.5

        b_extreme_price = a_end_price
        b_extreme_bar = a_end_bar
        entered = False

        for j in range(a_end_bar + 1, min(a_end_bar + max_b_bars, n_bars - 100)):
            if j <= active_trade_end:
                continue  # still in a previous trade

            c = closes[j]
            h = highs[j]
            l = lows[j]

            # B depth tracking
            if a_dir == 1:
                b_depth = a_end_price - l
                b_current = a_end_price - c
                if l < b_extreme_price:
                    b_extreme_price = l
                    b_extreme_bar = j
            else:
                b_depth = h - a_end_price
                b_current = c - a_end_price
                if h > b_extreme_price:
                    b_extreme_price = h
                    b_extreme_bar = j

            if b_depth > max_b_depth:
                break  # B too deep, pattern failed

            # B features
            b_depth_ratio = b_depth / a_amp if a_amp > 0 else 0
            b_current_ratio = b_current / a_amp if a_amp > 0 else 0
            b_bars_so_far = j - a_end_bar
            b_time_ratio = b_bars_so_far / a_bars if a_bars > 0 else 0

            b_amp_so_far = abs(b_extreme_price - a_end_price)
            b_bars_to_extreme = max(1, b_extreme_bar - a_end_bar)
            b_slope = b_amp_so_far / b_bars_to_extreme
            b_slope_ratio = b_slope / a_slope if a_slope > 0 else 999

            # Deceleration
            lookback = max(3, b_bars_so_far // 3)
            if b_bars_so_far >= lookback * 2:
                early_move = abs(closes[a_end_bar + lookback] - a_end_price)
                late_move = abs(c - closes[j - lookback])
                b_decel_ratio = late_move / early_move if early_move > 0 else 1.0
            else:
                b_decel_ratio = 1.0

            # Minimum B development to be interesting
            if b_depth_ratio < 0.10:
                continue  # B hasn't retraced enough

            # Must have some time development
            if b_time_ratio < 0.2:
                continue

            # ── Score this entry opportunity ──
            score = compute_intra_b_score(b_slope_ratio, b_time_ratio, b_depth_ratio, b_decel_ratio)

            # ── Get trade parameters ──
            entry_price = c
            # Apply spread
            if a_dir == 1:
                entry_adj = entry_price + half_spread
            else:
                entry_adj = entry_price - half_spread

            params = score_to_params_v4(score, a_amp, b_depth_ratio, entry_adj, a_end_price, a_dir)
            if params is None:
                continue

            pos_mult, tp_dist, sl_dist, c_target_frac = params

            # Spread cost check
            if spread / tp_dist > 0.15:
                continue

            # ── Create dynamic manager and simulate ──
            dm = DynamicManager(
                entry_price=entry_adj,
                a_dir=a_dir,
                a_amp=a_amp,
                a_bars=a_bars,
                tp_dist=tp_dist,
                sl_dist=sl_dist,
                entry_score=score,
            )

            max_hold = max(int(a_bars * 10), 500)
            end_bar = min(j + max_hold, n_bars - 1)

            exit_bar = end_bar
            exit_price = closes[end_bar]
            exit_reason = 'timeout'

            for bar in range(j + 1, end_bar + 1):
                bars_elapsed = bar - j
                sl, tp, should_close, close_price = dm.update(
                    bar, highs[bar], lows[bar], bars_elapsed
                )

                if should_close and close_price is not None:
                    exit_bar = bar
                    exit_price = close_price
                    # Determine reason
                    if close_price == dm.sl_price:
                        exit_reason = 'sl'
                    elif close_price == dm.tp_price:
                        exit_reason = 'tp'
                    else:
                        exit_reason = 'dynamic'
                    break

            # Apply exit spread
            if a_dir == 1:
                pnl = (exit_price - half_spread) - entry_adj
            else:
                pnl = entry_adj - (exit_price + half_spread)

            pnl_r = pnl / dm.initial_sl_dist if dm.initial_sl_dist > 0 else 0

            entry_year = get_year(dates[j]) if j < len(dates) else 0

            trades.append({
                'pair': pair,
                'tf': tf,
                'merge_level': merge_level,
                'entry_bar': j,
                'entry_year': entry_year,
                'entry_price': entry_adj,
                'exit_bar': exit_bar,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'a_dir': a_dir,
                'a_amp': round(a_amp, 6),
                'a_bars': a_bars,
                'score': round(score, 4),
                'pos_mult': pos_mult,
                'b_depth_ratio': round(b_depth_ratio, 4),
                'b_time_ratio': round(b_time_ratio, 4),
                'b_slope_ratio': round(b_slope_ratio, 4),
                'b_decel_ratio': round(b_decel_ratio, 4),
                'c_target_frac': c_target_frac,
                'tp_dist': round(tp_dist, 6),
                'sl_dist': round(sl_dist, 6),
                'pnl_r': round(pnl_r, 4),
                'pnl_weighted': round(pnl_r * pos_mult, 4),
                'hold_bars': exit_bar - j,
            })

            # Mark trade as active (no overlap)
            active_trade_end = exit_bar
            entered = True
            break  # only one entry per A wave

    return trades


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def compute_stats(trades, weighted=False):
    if not trades:
        return None
    if weighted:
        pnls = [t['pnl_weighted'] for t in trades]
    else:
        pnls = [t['pnl_r'] for t in trades]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')

    mean_r = np.mean(pnls)
    std_r = np.std(pnls)
    sharpe_per_trade = mean_r / std_r if std_r > 0 else 0

    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = np.max(dd) if len(dd) > 0 else 0

    return {
        'n': len(pnls),
        'wr': len(wins) / len(pnls) * 100,
        'avg_r': mean_r,
        'med_r': np.median(pnls),
        'pf': pf,
        'sharpe_trade': sharpe_per_trade,
        'total_r': sum(pnls),
        'max_dd_r': max_dd,
    }


def print_stats(label, s):
    if s is None:
        print(f"  {label}: NO DATA")
        return
    print(f"  {label}:")
    print(f"    Trades: {s['n']:,}   WR: {s['wr']:.1f}%   Avg R: {s['avg_r']:.4f}   Med R: {s['med_r']:.4f}")
    print(f"    PF: {s['pf']:.2f}   Total R: {s['total_r']:.1f}   Max DD: {s['max_dd_r']:.1f}R")
    print(f"    Sharpe/trade: {s['sharpe_trade']:.4f}")


def analyze_results(all_trades):
    print(f"\n{'='*90}")
    print(f"  V4 INTRA-B + DYNAMIC MANAGEMENT RESULTS")
    print(f"{'='*90}")

    # ── By merge level ──
    print(f"\n--- BY MERGE LEVEL ---")
    for ml in sorted(set(t['merge_level'] for t in all_trades)):
        subset = [t for t in all_trades if t['merge_level'] == ml]
        is_sub = [t for t in subset if t['entry_year'] <= 2018]
        oos_sub = [t for t in subset if t['entry_year'] > 2018]
        print(f"\n  Merge Level L{ml}:")
        print_stats(f"  IS  (2000-2018)", compute_stats(is_sub))
        print_stats(f"  OOS (2019-2024)", compute_stats(oos_sub))

    # ── By timeframe ──
    print(f"\n--- BY TIMEFRAME ---")
    for tf in sorted(set(t['tf'] for t in all_trades)):
        subset = [t for t in all_trades if t['tf'] == tf]
        is_sub = [t for t in subset if t['entry_year'] <= 2018]
        oos_sub = [t for t in subset if t['entry_year'] > 2018]
        print(f"\n  {tf}:")
        print_stats(f"  IS ", compute_stats(is_sub))
        print_stats(f"  OOS", compute_stats(oos_sub))

    # ── Overall IS/OOS ──
    is_trades = [t for t in all_trades if t['entry_year'] <= 2018]
    oos_trades = [t for t in all_trades if t['entry_year'] > 2018]

    print(f"\n--- OVERALL ---")
    print_stats("ALL", compute_stats(all_trades))
    print_stats("IS  (2000-2018)", compute_stats(is_trades))
    print_stats("OOS (2019-2024)", compute_stats(oos_trades))

    print(f"\n--- POSITION-WEIGHTED ---")
    print_stats("ALL weighted", compute_stats(all_trades, weighted=True))
    print_stats("IS  weighted", compute_stats(is_trades, weighted=True))
    print_stats("OOS weighted", compute_stats(oos_trades, weighted=True))

    # ── By score bucket ──
    print(f"\n--- BY ENTRY SCORE BUCKET ---")
    score_bins = [(0.25, 0.40, 'Low'), (0.40, 0.55, 'Med-Low'), (0.55, 0.70, 'Medium'),
                  (0.70, 0.85, 'High'), (0.85, 1.01, 'Elite')]
    for lo, hi, label in score_bins:
        bucket = [t for t in all_trades if lo <= t['score'] < hi]
        if not bucket:
            continue
        s = compute_stats(bucket)
        oos_b = [t for t in bucket if t['entry_year'] > 2018]
        s_oos = compute_stats(oos_b)
        print(f"\n  Score {lo:.2f}-{hi:.2f} ({label}):")
        if s:
            print(f"    ALL: n={s['n']:>7,}  WR={s['wr']:.1f}%  avgR={s['avg_r']:.4f}  PF={s['pf']:.2f}")
        if s_oos:
            print(f"    OOS: n={s_oos['n']:>7,}  WR={s_oos['wr']:.1f}%  avgR={s_oos['avg_r']:.4f}  PF={s_oos['pf']:.2f}")

    # ── By year ──
    print(f"\n--- BY YEAR ---")
    years = sorted(set(t['entry_year'] for t in all_trades))
    print(f"{'Year':>6s} {'n':>7s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s} {'totalR':>8s}")
    print("-" * 50)
    for y in years:
        yt = [t for t in all_trades if t['entry_year'] == y]
        s = compute_stats(yt, weighted=True)
        if s:
            marker = " *OOS*" if y > 2018 else ""
            print(f"{y:>6d} {s['n']:>7,} {s['wr']:>6.1f} {s['avg_r']:>8.4f} "
                  f"{s['pf']:>7.2f} {s['total_r']:>8.1f}{marker}")

    # ── Exit reason distribution ──
    print(f"\n--- EXIT REASONS ---")
    for period, trades in [("ALL", all_trades), ("OOS", oos_trades)]:
        reasons = defaultdict(list)
        for t in trades:
            reasons[t['exit_reason']].append(t['pnl_r'])
        print(f"\n  {period}:")
        for reason in sorted(reasons.keys()):
            rpnls = reasons[reason]
            wr = len([p for p in rpnls if p > 0]) / len(rpnls) * 100 if rpnls else 0
            print(f"    {reason:>10s}: n={len(rpnls):>7,}  WR={wr:.1f}%  avgR={np.mean(rpnls):.4f}")

    # ── Comparison vs V3 baseline (no dynamic management) ──
    print(f"\n{'='*90}")
    print(f"  KEY QUESTION: Does dynamic management rescue intra-B entry?")
    print(f"{'='*90}")
    print(f"  V3 (fixed SL/TP): WR=24.5%, avgR=-0.21, PF=0.73 (all negative)")
    s_all = compute_stats(all_trades)
    if s_all:
        delta = s_all['avg_r'] - (-0.21)
        print(f"  V4 (dynamic mgmt): WR={s_all['wr']:.1f}%, avgR={s_all['avg_r']:.4f}, PF={s_all['pf']:.2f}")
        print(f"  Delta avgR: {delta:+.4f} ({'+' if delta > 0 else ''}{delta / 0.21 * 100:.0f}% change)")
        if s_all['avg_r'] > 0:
            print(f"  >>> DYNAMIC MANAGEMENT RESCUES INTRA-B ENTRY! <<<")
        else:
            print(f"  >>> Still negative — may need better entry conditions or approach <<<")

    # ── Best merge level + TF combo ──
    print(f"\n--- BEST COMBO (OOS) ---")
    combos = []
    for ml in sorted(set(t['merge_level'] for t in all_trades)):
        for tf in sorted(set(t['tf'] for t in all_trades)):
            subset = [t for t in oos_trades if t['merge_level'] == ml and t['tf'] == tf]
            s = compute_stats(subset)
            if s and s['n'] >= 100:
                combos.append((ml, tf, s))
    combos.sort(key=lambda x: x[2]['avg_r'], reverse=True)
    print(f"{'ML':>4s} {'TF':>4s} {'n':>7s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s} {'totalR':>8s}")
    print("-" * 50)
    for ml, tf, s in combos:
        print(f"  L{ml:<3d} {tf:>4s} {s['n']:>7,} {s['wr']:>6.1f} {s['avg_r']:>8.4f} {s['pf']:>7.2f} {s['total_r']:>8.1f}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    pairs = get_all_pairs()
    tfs = ['H1', 'M30', 'M15']
    merge_levels = [2, 3, 4, 5]  # test multiple structural levels

    print(f"ABC V4 — Intra-B Entry with Dynamic Management")
    print(f"  Pairs: {len(pairs)}")
    print(f"  TFs: {tfs}")
    print(f"  ZigZag: (2,1,1)")
    print(f"  Merge levels: L{merge_levels}")
    print(f"  Entry: during B wave, scored + dynamic managed")
    print(f"  No-overlap mode: one trade at a time per A wave")
    print()

    tasks = [(pair, tf, ml)
             for pair in pairs
             for tf in tfs
             for ml in merge_levels]

    print(f"Total tasks: {len(tasks)}")
    print(f"Running with 60 workers...")

    all_trades = []
    done = 0
    with Pool(60) as pool:
        for batch in pool.imap_unordered(backtest_pair_tf, tasks):
            all_trades.extend(batch)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(tasks)} tasks done, {len(all_trades):,} trades so far")

    print(f"\nTotal trades: {len(all_trades):,}")

    # Sort
    all_trades.sort(key=lambda t: (t['entry_year'], t['entry_bar']))

    # Save
    out_path = "/home/ubuntu/stage2_abc/abc_v4_trades.csv"
    if all_trades:
        fieldnames = list(all_trades[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in all_trades:
                writer.writerow(t)
        print(f"Trades saved to: {out_path}")

    analyze_results(all_trades)


if __name__ == '__main__':
    main()
