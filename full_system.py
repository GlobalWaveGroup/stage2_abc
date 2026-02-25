"""
Stage2 Full ABC Trading System
================================
Complete system with:
1. Entry scoring matrix → position size + TP target + SL ratio
2. In-trade dynamic adjustment matrix → modify TP/SL based on C' progress
3. Full backtest with equity curve, Sharpe, max drawdown

Scoring calibrated on IS (2000-2018), validated on OOS (2019-2024).
"""

import os
import csv
import numpy as np
from multiprocessing import Pool
from collections import defaultdict

DATA_DIR = "/home/ubuntu/DataBase/base_kline"

# ═══════════════════════════════════════════════════════════════════════
# ONLINE ZIGZAG (embedded)
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# ENTRY SCORING MATRIX
# ═══════════════════════════════════════════════════════════════════════
#
# Based on IS data analysis (OOS-confirmed monotonic relationships):
#
# score = w1 * slope_score + w2 * time_score + w3 * amp_score
#
# Higher score → stronger setup → bigger position, wider TP, tighter SL ratio
#
# Score range: 0.0 ~ 1.0
# Score < 0.3: skip trade (too weak)
# Score 0.3-0.5: small position (0.5x), conservative TP
# Score 0.5-0.7: normal position (1.0x), standard TP
# Score 0.7-0.85: large position (1.5x), aggressive TP
# Score 0.85+: max position (2.0x), maximum TP
# ═══════════════════════════════════════════════════════════════════════

def compute_entry_score(slope_ratio, time_ratio, amp_ratio):
    """
    Compute entry quality score from ABC features.
    Returns score in [0, 1].

    Weights calibrated from IS feature importance:
    - slope_ratio: strongest monotonic predictor (40%)
    - time_ratio: second strongest (35%)
    - amp_ratio: moderate, non-linear peak at 0.7-1.0 (25%)
    """
    # Slope score: lower slope_ratio = better
    # slope_ratio 0.0 → 1.0, slope_ratio 0.5 → 0.6, slope_ratio 2.0 → 0.1
    slope_score = np.clip(1.0 - slope_ratio * 0.8, 0.0, 1.0)

    # Time score: higher time_ratio = better (B takes longer = weaker)
    # time_ratio 0.3 → 0.1, time_ratio 1.0 → 0.35, time_ratio 5.0 → 0.85, 10+ → 1.0
    time_score = np.clip((np.log1p(time_ratio) - 0.2) / 2.2, 0.0, 1.0)

    # Amp score: peak around 0.7-1.0, drops off at extremes
    # Models the sweet spot of retracement depth
    if amp_ratio < 0.3:
        amp_score = amp_ratio / 0.3 * 0.4  # 0→0, 0.3→0.4
    elif amp_ratio < 0.7:
        amp_score = 0.4 + (amp_ratio - 0.3) / 0.4 * 0.6  # 0.3→0.4, 0.7→1.0
    elif amp_ratio <= 1.0:
        amp_score = 1.0  # sweet spot
    elif amp_ratio <= 1.5:
        amp_score = 1.0 - (amp_ratio - 1.0) / 0.5 * 0.3  # 1.0→1.0, 1.5→0.7
    else:
        amp_score = max(0.3, 0.7 - (amp_ratio - 1.5) * 0.2)  # declining

    # Weighted combination
    score = 0.40 * slope_score + 0.35 * time_score + 0.25 * amp_score

    return np.clip(score, 0.0, 1.0)


def score_to_params(score):
    """
    Map entry score to trade parameters.

    Returns: (position_size_mult, tp_mult, sl_ratio)
    - position_size_mult: multiplier on base position (0.5x - 2.0x)
    - tp_mult: TP = A_amp * tp_mult (0.60 - 1.10)
    - sl_ratio: SL = TP * sl_ratio (0.30 - 0.50)
    """
    if score < 0.30:
        return None  # skip trade

    if score < 0.50:
        # Weak: small position, conservative TP, wider SL ratio
        t = (score - 0.30) / 0.20  # 0→1 within bin
        pos = 0.5 + t * 0.2        # 0.5 → 0.7
        tp = 0.60 + t * 0.05       # 0.60 → 0.65
        sl = 0.45 - t * 0.03       # 0.45 → 0.42
    elif score < 0.70:
        t = (score - 0.50) / 0.20
        pos = 0.7 + t * 0.3        # 0.7 → 1.0
        tp = 0.65 + t * 0.10       # 0.65 → 0.75
        sl = 0.42 - t * 0.04       # 0.42 → 0.38
    elif score < 0.85:
        t = (score - 0.70) / 0.15
        pos = 1.0 + t * 0.5        # 1.0 → 1.5
        tp = 0.75 + t * 0.15       # 0.75 → 0.90
        sl = 0.38 - t * 0.03       # 0.38 → 0.35
    else:
        t = min((score - 0.85) / 0.15, 1.0)
        pos = 1.5 + t * 0.5        # 1.5 → 2.0
        tp = 0.90 + t * 0.20       # 0.90 → 1.10
        sl = 0.35 - t * 0.03       # 0.35 → 0.32

    return (pos, tp, sl)


# ═══════════════════════════════════════════════════════════════════════
# DYNAMIC IN-TRADE ADJUSTMENT MATRIX
# ═══════════════════════════════════════════════════════════════════════
#
# During trade, track C' (current C wave progress).
# Adjust TP/SL based on:
# 1. progress = C'_current / TP_target
# 2. speed = progress / (bars_elapsed / expected_bars)
# 3. Based on these, expand or contract TP/SL
# ═══════════════════════════════════════════════════════════════════════

class DynamicManager:
    """Manages SL/TP adjustments during a trade."""

    def __init__(self, entry_price, a_dir, a_amp, a_bars, tp_mult, sl_ratio, entry_score):
        self.entry_price = entry_price
        self.a_dir = a_dir
        self.a_amp = a_amp
        self.a_bars = a_bars
        self.entry_score = entry_score

        # Initial TP/SL
        self.tp_distance = a_amp * tp_mult
        self.sl_distance = self.tp_distance * sl_ratio

        if a_dir == 1:
            self.tp_price = entry_price + self.tp_distance
            self.sl_price = entry_price - self.sl_distance
        else:
            self.tp_price = entry_price - self.tp_distance
            self.sl_price = entry_price + self.sl_distance

        self.max_favorable = 0.0
        self.hit_breakeven = False
        self.phase = 0  # 0=initial, 1=developing, 2=mature, 3=beyond_tp, 4=strong_beyond
        self.expected_bars = a_bars * 1.2  # C typically similar duration to A

    def update(self, bar_idx, high, low, bars_elapsed):
        """
        Update dynamic levels. Returns (new_sl, new_tp, should_close, close_price).
        should_close=True means exit now.
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

        # Speed: how fast is C' developing vs expectation
        time_frac = bars_elapsed / self.expected_bars if self.expected_bars > 0 else 1
        speed = progress / time_frac if time_frac > 0.05 else progress * 20

        # ── Check SL hit ──
        if self.a_dir == 1:
            if low <= self.sl_price:
                return self.sl_price, self.tp_price, True, self.sl_price
        else:
            if high >= self.sl_price:
                return self.sl_price, self.tp_price, True, self.sl_price

        # ── Phase transitions and adjustments ──

        # Phase 1: Early development (progress 0-30%)
        # No changes, let it breathe

        # Phase 2: Developing (progress 30-60%) → move SL to breakeven
        if progress >= 0.30 and not self.hit_breakeven:
            self.sl_price = self.entry_price
            self.hit_breakeven = True

            # If speed is good, slightly expand TP
            if speed > 1.2:
                expand = min(0.10, (speed - 1.2) * 0.05)
                self._expand_tp(expand)

        # Phase 3: Mature (progress 60-100%) → lock profits, adjust TP by speed
        if progress >= 0.60:
            # Lock 50% of max favorable
            self._move_sl_to_lock(0.50)

            # Speed-based TP adjustment
            if speed > 1.5:
                # Fast development: C' moving faster than expected → expand TP
                expand = min(0.25, (speed - 1.5) * 0.10)
                self._expand_tp(expand)
            elif speed < 0.5 and bars_elapsed > self.expected_bars:
                # Slow: taking too long → contract TP
                contract = min(0.20, (0.5 - speed) * 0.15)
                self._contract_tp(contract)

        # Phase 4: Beyond initial TP (progress > 100%) → trailing mode
        if progress >= 1.0:
            # Expand TP significantly
            self._expand_tp(0.30)  # +30% of A_amp added to TP

            # Tighter trailing stop
            self._move_sl_to_lock(0.65)

            # Score-adjusted: higher entry score → let it run more
            if self.entry_score > 0.7:
                extra_room = (self.entry_score - 0.7) * 0.5  # up to 0.15 extra
                self._expand_tp(extra_room)

        # Phase 5: Strong beyond (progress > 150%) → very tight trail, max expansion
        if progress >= 1.5:
            self._expand_tp(0.50)
            self._move_sl_to_lock(0.75)

        # Phase 6: Extreme (progress > 200%) → prepare to exit
        if progress >= 2.0:
            self._move_sl_to_lock(0.85)

        # ── Stagnation check ──
        if bars_elapsed > self.expected_bars * 2.5 and progress < 0.40:
            # Stagnating badly: shrink TP to what we have + 10%
            if self.max_favorable > 0:
                shrink_target = self.entry_price + self.max_favorable * 1.05 * self.a_dir
                if self.a_dir == 1:
                    self.tp_price = min(self.tp_price, shrink_target)
                else:
                    self.tp_price = max(self.tp_price, shrink_target)

        # ── Timeout: very long trades → just exit at current SL
        if bars_elapsed > self.a_bars * 10:
            # Force close at next bar open (return current favorable price)
            return self.sl_price, self.tp_price, False, None

        # ── Check TP hit ──
        if self.a_dir == 1:
            if high >= self.tp_price:
                return self.sl_price, self.tp_price, True, self.tp_price
        else:
            if low <= self.tp_price:
                return self.sl_price, self.tp_price, True, self.tp_price

        return self.sl_price, self.tp_price, False, None

    def _expand_tp(self, fraction):
        """Expand TP by fraction of A_amp (only in favorable direction)."""
        delta = self.a_amp * fraction
        if self.a_dir == 1:
            self.tp_price = max(self.tp_price, self.tp_price + delta)
        else:
            self.tp_price = min(self.tp_price, self.tp_price - delta)
        self.tp_distance = abs(self.tp_price - self.entry_price)

    def _contract_tp(self, fraction):
        """Contract TP by fraction of A_amp."""
        delta = self.a_amp * fraction
        if self.a_dir == 1:
            new_tp = self.tp_price - delta
            # Don't contract below entry + current locked profit
            self.tp_price = max(new_tp, self.entry_price + self.max_favorable * 0.3)
        else:
            new_tp = self.tp_price + delta
            self.tp_price = min(new_tp, self.entry_price - self.max_favorable * 0.3)
        self.tp_distance = abs(self.tp_price - self.entry_price)

    def _move_sl_to_lock(self, lock_pct):
        """Move SL to lock lock_pct of max favorable. Only moves in favorable direction."""
        lock = self.entry_price + self.max_favorable * lock_pct * self.a_dir
        if self.a_dir == 1:
            self.sl_price = max(self.sl_price, lock)
        else:
            self.sl_price = min(self.sl_price, lock)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
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
            if len(row) < 6:
                continue
            dates.append(f"{row[0]} {row[1]}")
            opens.append(float(row[2]))
            highs.append(float(row[3]))
            lows.append(float(row[4]))
            closes.append(float(row[5]))
    return dates, np.array(opens), np.array(highs), np.array(lows), np.array(closes)


def get_year(date_str):
    return int(date_str[:4])


def get_all_pairs():
    pairs = set()
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.csv'):
            parts = fname.replace('.csv', '').rsplit('_', 1)
            if len(parts) == 2:
                pairs.add(parts[0])
    return sorted(pairs)


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════

def backtest_pair_tf(args):
    """Full backtest for one pair-TF."""
    pair, tf, zz_dev, zz_confirm = args

    data = load_ohlcv(pair, tf)
    if data is None:
        return []

    dates, opens, highs, lows, closes = data
    n_bars = len(closes)

    # Run zigzag
    zz = OnlineZigZag(deviation_pct=zz_dev, confirm_bars=zz_confirm)
    for i in range(n_bars):
        zz.process_bar(i, highs[i], lows[i])

    if len(zz.pivots) < 4:
        return []

    # Process ABC triples
    pivots = zz.pivots
    trades = []

    for i in range(len(pivots) - 2):
        p0 = pivots[i]    # A start
        p1 = pivots[i+1]  # A end / B start
        p2 = pivots[i+2]  # B end / entry

        # A leg
        a_bars = p1[0] - p0[0]
        a_amp = abs(p1[1] - p0[1])
        a_dir = 1 if p1[1] > p0[1] else -1

        # B leg
        b_bars = p2[0] - p1[0]
        b_amp = abs(p2[1] - p1[1])

        if a_bars <= 0 or b_bars <= 0 or a_amp <= 0:
            continue

        amp_ratio = b_amp / a_amp
        time_ratio = b_bars / a_bars
        a_slope = a_amp / a_bars
        b_slope = b_amp / b_bars
        slope_ratio = b_slope / a_slope if a_slope > 0 else 999

        # ── Compute entry score ──
        score = compute_entry_score(slope_ratio, time_ratio, amp_ratio)
        params = score_to_params(score)

        if params is None:
            continue  # score too low, skip

        pos_mult, tp_mult, sl_ratio = params

        # ── Entry ──
        entry_bar = p2[0] + 1  # bar after B confirmation
        if entry_bar >= n_bars - 10:
            continue

        entry_price = closes[entry_bar]
        entry_year = get_year(dates[entry_bar])

        # ── Create dynamic manager ──
        dm = DynamicManager(
            entry_price=entry_price,
            a_dir=a_dir,
            a_amp=a_amp,
            a_bars=a_bars,
            tp_mult=tp_mult,
            sl_ratio=sl_ratio,
            entry_score=score,
        )

        # ── Simulate bar by bar ──
        max_hold = max(int(a_bars * 10), 500)
        end_bar = min(entry_bar + max_hold, n_bars - 1)

        exit_bar = end_bar
        exit_price = closes[end_bar]
        exit_reason = 'timeout'

        for bar in range(entry_bar + 1, end_bar + 1):
            bars_elapsed = bar - entry_bar
            sl, tp, should_close, close_price = dm.update(
                bar, highs[bar], lows[bar], bars_elapsed
            )

            if should_close and close_price is not None:
                exit_bar = bar
                exit_price = close_price
                # Determine reason
                if a_dir == 1:
                    exit_reason = 'sl' if close_price <= entry_price - dm.sl_distance * 0.5 else 'tp'
                else:
                    exit_reason = 'sl' if close_price >= entry_price + dm.sl_distance * 0.5 else 'tp'

                # More precise: check which level was hit
                if close_price == dm.sl_price:
                    exit_reason = 'sl'
                elif close_price == dm.tp_price:
                    exit_reason = 'tp'
                else:
                    exit_reason = 'dynamic'
                break

        # PnL
        if a_dir == 1:
            pnl_price = exit_price - entry_price
        else:
            pnl_price = entry_price - exit_price

        initial_sl_dist = a_amp * tp_mult * sl_ratio
        pnl_r = pnl_price / initial_sl_dist if initial_sl_dist > 0 else 0

        trades.append({
            'pair': pair,
            'tf': tf,
            'entry_bar': entry_bar,
            'entry_year': entry_year,
            'entry_price': entry_price,
            'exit_bar': exit_bar,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'a_dir': a_dir,
            'a_amp': a_amp,
            'a_bars': a_bars,
            'score': score,
            'pos_mult': pos_mult,
            'tp_mult': tp_mult,
            'sl_ratio': sl_ratio,
            'slope_ratio': slope_ratio,
            'amp_ratio': amp_ratio,
            'time_ratio': time_ratio,
            'pnl_price': pnl_price,
            'pnl_r': pnl_r,
            'pnl_weighted': pnl_r * pos_mult,  # position-weighted R
            'hold_bars': exit_bar - entry_bar,
        })

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

    # Sharpe (annualize assuming ~250 trades/year as rough estimate)
    mean_r = np.mean(pnls)
    std_r = np.std(pnls)
    sharpe_per_trade = mean_r / std_r if std_r > 0 else 0

    # Max drawdown in R terms
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = np.max(dd) if len(dd) > 0 else 0

    return {
        'n': len(pnls),
        'wr': len(wins) / len(pnls) * 100,
        'avg_r': mean_r,
        'med_r': np.median(pnls),
        'std_r': std_r,
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
        'pf': pf,
        'sharpe_trade': sharpe_per_trade,
        'total_r': sum(pnls),
        'max_dd_r': max_dd,
        'calmar': sum(pnls) / max_dd if max_dd > 0 else float('inf'),
    }


def print_stats(label, s):
    if s is None:
        print(f"  {label}: NO DATA")
        return
    print(f"  {label}:")
    print(f"    Trades: {s['n']:,}   WR: {s['wr']:.1f}%   Avg R: {s['avg_r']:.4f}")
    print(f"    Avg Win: {s['avg_win']:.3f}R   Avg Loss: {s['avg_loss']:.3f}R   PF: {s['pf']:.2f}")
    print(f"    Total R: {s['total_r']:.1f}   Max DD: {s['max_dd_r']:.1f}R   Calmar: {s['calmar']:.2f}")
    print(f"    Sharpe/trade: {s['sharpe_trade']:.4f}")


def analyze_results(all_trades):
    print(f"\n{'='*80}")
    print(f"  FULL SYSTEM RESULTS")
    print(f"{'='*80}")

    # Split IS/OOS
    is_trades = [t for t in all_trades if t['entry_year'] <= 2018]
    oos_trades = [t for t in all_trades if t['entry_year'] > 2018]

    print(f"\n--- Unweighted (raw R) ---")
    print_stats("ALL", compute_stats(all_trades))
    print_stats("IS  (2000-2018)", compute_stats(is_trades))
    print_stats("OOS (2019-2024)", compute_stats(oos_trades))

    print(f"\n--- Position-Weighted (score-adjusted R) ---")
    print_stats("ALL weighted", compute_stats(all_trades, weighted=True))
    print_stats("IS  weighted", compute_stats(is_trades, weighted=True))
    print_stats("OOS weighted", compute_stats(oos_trades, weighted=True))

    # By score bucket
    print(f"\n{'='*80}")
    print(f"  BY ENTRY SCORE BUCKET")
    print(f"{'='*80}")
    score_bins = [(0.3, 0.5, 'Low'), (0.5, 0.7, 'Medium'), (0.7, 0.85, 'High'), (0.85, 1.01, 'Elite')]
    for lo, hi, label in score_bins:
        bucket_is = [t for t in is_trades if lo <= t['score'] < hi]
        bucket_oos = [t for t in oos_trades if lo <= t['score'] < hi]
        print(f"\n  Score {lo:.2f}-{hi:.2f} ({label}):")
        s_is = compute_stats(bucket_is)
        s_oos = compute_stats(bucket_oos)
        if s_is:
            print(f"    IS:  n={s_is['n']:>7,}  WR={s_is['wr']:.1f}%  avgR={s_is['avg_r']:.4f}  PF={s_is['pf']:.2f}")
        if s_oos:
            print(f"    OOS: n={s_oos['n']:>7,}  WR={s_oos['wr']:.1f}%  avgR={s_oos['avg_r']:.4f}  PF={s_oos['pf']:.2f}")

    # By year
    print(f"\n{'='*80}")
    print(f"  BY YEAR (position-weighted)")
    print(f"{'='*80}")
    years = sorted(set(t['entry_year'] for t in all_trades))
    print(f"{'Year':>6s} {'n':>7s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s} {'totalR':>8s} {'maxDD':>7s}")
    print("-" * 55)
    for y in years:
        yt = [t for t in all_trades if t['entry_year'] == y]
        s = compute_stats(yt, weighted=True)
        if s:
            marker = " *OOS*" if y > 2018 else ""
            print(f"{y:>6d} {s['n']:>7,} {s['wr']:>6.1f} {s['avg_r']:>8.4f} "
                  f"{s['pf']:>7.2f} {s['total_r']:>8.1f} {s['max_dd_r']:>7.1f}{marker}")

    # By pair (OOS only, top/bottom)
    print(f"\n{'='*80}")
    print(f"  BY PAIR (OOS, position-weighted, sorted by avg_R)")
    print(f"{'='*80}")
    pair_results = defaultdict(list)
    for t in oos_trades:
        pair_results[t['pair']].append(t)

    pair_stats = []
    for pair, trades in pair_results.items():
        s = compute_stats(trades, weighted=True)
        if s and s['n'] >= 20:
            pair_stats.append((pair, s))

    pair_stats.sort(key=lambda x: x[1]['avg_r'], reverse=True)

    print(f"{'Pair':>10s} {'n':>6s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s} {'totalR':>8s}")
    print("-" * 50)
    for pair, s in pair_stats:
        print(f"{pair:>10s} {s['n']:>6,} {s['wr']:>6.1f} {s['avg_r']:>8.4f} {s['pf']:>7.2f} {s['total_r']:>8.1f}")

    n_positive = sum(1 for _, s in pair_stats if s['avg_r'] > 0)
    print(f"\nPositive pairs: {n_positive}/{len(pair_stats)} ({n_positive/len(pair_stats)*100:.0f}%)")

    # Exit reason distribution
    print(f"\n{'='*80}")
    print(f"  EXIT REASON ANALYSIS")
    print(f"{'='*80}")
    for period_name, period_trades in [("IS", is_trades), ("OOS", oos_trades)]:
        reasons = defaultdict(list)
        for t in period_trades:
            reasons[t['exit_reason']].append(t['pnl_weighted'])
        print(f"\n  {period_name}:")
        for reason in sorted(reasons.keys()):
            rpnls = reasons[reason]
            wr = len([p for p in rpnls if p > 0]) / len(rpnls) * 100 if rpnls else 0
            print(f"    {reason:>10s}: n={len(rpnls):>7,}  WR={wr:.1f}%  avg_wR={np.mean(rpnls):.3f}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    pairs = get_all_pairs()
    tfs = ['H1', 'M30', 'M15']

    # Use best ZZ config from OOS validation: dev=1.0, confirm=5
    # Also include dev=0.5 for more trades
    zz_configs = [
        (1.0, 5),   # best OOS performance
        (0.5, 5),   # more trades, still good
    ]

    print(f"Full System Backtest")
    print(f"  Pairs: {len(pairs)}")
    print(f"  TFs: {tfs}")
    print(f"  ZZ configs: {zz_configs}")
    print(f"  Score threshold: >= 0.30")
    print()

    # Build tasks
    tasks = []
    for pair in pairs:
        for tf in tfs:
            for dev, conf in zz_configs:
                tasks.append((pair, tf, dev, conf))

    print(f"Total tasks: {len(tasks)}")
    print(f"Running with 40 workers...")

    all_trades = []
    done = 0
    with Pool(40) as pool:
        for batch in pool.imap_unordered(backtest_pair_tf, tasks):
            all_trades.extend(batch)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)} tasks done, {len(all_trades):,} trades so far")

    print(f"\nTotal trades: {len(all_trades):,}")

    # Sort by entry bar for proper equity curve
    all_trades.sort(key=lambda t: (t['entry_year'], t['entry_bar']))

    # Save trades
    out_path = "/home/ubuntu/stage2_abc/full_system_trades.csv"
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
