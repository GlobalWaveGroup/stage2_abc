"""
Stage2 Realistic Backtest
=========================
Adds to full_system.py:
1. Spread costs (per-pair realistic spreads in pips)
2. No overlapping trades (same pair+TF: must close before re-entry)
3. Single ZZ config (dev=1.0 only — best OOS, fewer trades, less overlap)
4. Slippage: 0.5 pip per trade

This is the "honest" version before porting to MT5.
"""

import os
import csv
import numpy as np
from multiprocessing import Pool
from collections import defaultdict

DATA_DIR = "/home/ubuntu/DataBase/base_kline"

# ═══════════════════════════════════════════════════════════════════════
# REALISTIC SPREAD TABLE (in pips, typical retail broker)
# ═══════════════════════════════════════════════════════════════════════

# pip size: most pairs = 0.0001, JPY pairs = 0.01, metals = 0.01/$0.01
PAIR_INFO = {
    # Major pairs: tight spreads
    'EURUSD': {'spread': 1.2, 'pip': 0.0001},
    'GBPUSD': {'spread': 1.5, 'pip': 0.0001},
    'USDJPY': {'spread': 1.3, 'pip': 0.01},
    'USDCHF': {'spread': 1.5, 'pip': 0.0001},
    'AUDUSD': {'spread': 1.4, 'pip': 0.0001},
    'NZDUSD': {'spread': 1.8, 'pip': 0.0001},
    'USDCAD': {'spread': 1.6, 'pip': 0.0001},

    # Minor pairs: medium spreads
    'EURGBP': {'spread': 1.8, 'pip': 0.0001},
    'EURJPY': {'spread': 1.8, 'pip': 0.01},
    'GBPJPY': {'spread': 2.5, 'pip': 0.01},
    'EURAUD': {'spread': 2.5, 'pip': 0.0001},
    'EURNZD': {'spread': 3.0, 'pip': 0.0001},
    'EURCAD': {'spread': 2.5, 'pip': 0.0001},
    'EURCHF': {'spread': 2.0, 'pip': 0.0001},
    'GBPAUD': {'spread': 3.0, 'pip': 0.0001},
    'GBPCAD': {'spread': 3.0, 'pip': 0.0001},
    'GBPCHF': {'spread': 2.8, 'pip': 0.0001},
    'GBPNZD': {'spread': 4.0, 'pip': 0.0001},
    'AUDCAD': {'spread': 2.5, 'pip': 0.0001},
    'AUDCHF': {'spread': 2.5, 'pip': 0.0001},
    'AUDJPY': {'spread': 2.0, 'pip': 0.01},
    'AUDNZD': {'spread': 2.5, 'pip': 0.0001},
    'CADJPY': {'spread': 2.0, 'pip': 0.01},
    'CADCHF': {'spread': 2.5, 'pip': 0.0001},
    'CHFJPY': {'spread': 2.5, 'pip': 0.01},
    'NZDJPY': {'spread': 2.5, 'pip': 0.01},
    'NZDCAD': {'spread': 3.0, 'pip': 0.0001},
    'NZDCHF': {'spread': 3.0, 'pip': 0.0001},

    # Exotic pairs: wide spreads
    'EURNOK': {'spread': 25.0, 'pip': 0.0001},
    'EURSEK': {'spread': 30.0, 'pip': 0.0001},
    'EURPLN': {'spread': 25.0, 'pip': 0.0001},
    'EURTRY': {'spread': 50.0, 'pip': 0.0001},
    'EURHKD': {'spread': 20.0, 'pip': 0.0001},
    'EURCNH': {'spread': 30.0, 'pip': 0.0001},
    'GBPNOK': {'spread': 40.0, 'pip': 0.0001},
    'USDNOK': {'spread': 20.0, 'pip': 0.0001},
    'USDSEK': {'spread': 25.0, 'pip': 0.0001},
    'USDPLN': {'spread': 25.0, 'pip': 0.0001},
    'USDTRY': {'spread': 50.0, 'pip': 0.0001},
    'USDHUF': {'spread': 20.0, 'pip': 0.01},
    'USDMXN': {'spread': 30.0, 'pip': 0.0001},
    'USDZAR': {'spread': 40.0, 'pip': 0.0001},
    'USDSGD': {'spread': 3.0, 'pip': 0.0001},
    'USDCNH': {'spread': 30.0, 'pip': 0.0001},
    'USDRMB': {'spread': 30.0, 'pip': 0.0001},
    'USDHKD': {'spread': 5.0, 'pip': 0.0001},

    # Metals
    'XAUUSD': {'spread': 3.0, 'pip': 0.01},     # $0.30 spread
    'XAGUSD': {'spread': 3.0, 'pip': 0.001},     # $0.003 spread
}

SLIPPAGE_PIPS = 0.5  # per trade, one way

def get_spread_cost(pair):
    """Return round-trip cost (spread + slippage) in price units."""
    info = PAIR_INFO.get(pair, {'spread': 5.0, 'pip': 0.0001})
    total_pips = info['spread'] + SLIPPAGE_PIPS * 2  # slippage on entry and exit
    return total_pips * info['pip']


# ═══════════════════════════════════════════════════════════════════════
# ONLINE ZIGZAG (same as full_system.py)
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
                self._trend = 1; self._ext_price = high; self._ext_bar = idx
            else:
                self.pivots.append((self._init_hi_bar, self._init_hi, 1))
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
                    pv = (self._tent[0], self._tent[1], +1)
                    self.pivots.append(pv)
                    self._trend = -1; self._ext_price = low; self._ext_bar = idx
                    self._tent = None
                    return (pv[0], pv[1], pv[2], idx)
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
                    pv = (self._tent[0], self._tent[1], -1)
                    self.pivots.append(pv)
                    self._trend = 1; self._ext_price = high; self._ext_bar = idx
                    self._tent = None
                    return (pv[0], pv[1], pv[2], idx)
            return None


# ═══════════════════════════════════════════════════════════════════════
# SCORING (same as full_system.py)
# ═══════════════════════════════════════════════════════════════════════

def compute_entry_score(slope_ratio, time_ratio, amp_ratio):
    slope_score = np.clip(1.0 - slope_ratio * 0.8, 0.0, 1.0)
    time_score = np.clip((np.log1p(time_ratio) - 0.2) / 2.2, 0.0, 1.0)
    if amp_ratio < 0.3:
        amp_score = amp_ratio / 0.3 * 0.4
    elif amp_ratio < 0.7:
        amp_score = 0.4 + (amp_ratio - 0.3) / 0.4 * 0.6
    elif amp_ratio <= 1.0:
        amp_score = 1.0
    elif amp_ratio <= 1.5:
        amp_score = 1.0 - (amp_ratio - 1.0) / 0.5 * 0.3
    else:
        amp_score = max(0.3, 0.7 - (amp_ratio - 1.5) * 0.2)
    return np.clip(0.40 * slope_score + 0.35 * time_score + 0.25 * amp_score, 0.0, 1.0)


def score_to_params(score):
    if score < 0.30:
        return None
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
# DYNAMIC MANAGER (same as full_system.py)
# ═══════════════════════════════════════════════════════════════════════

class DynamicManager:
    def __init__(self, entry_price, a_dir, a_amp, a_bars, tp_mult, sl_ratio, entry_score):
        self.entry_price = entry_price
        self.a_dir = a_dir
        self.a_amp = a_amp
        self.a_bars = a_bars
        self.entry_score = entry_score
        self.tp_distance = a_amp * tp_mult
        self.sl_distance = self.tp_distance * sl_ratio
        self.initial_sl_distance = self.sl_distance
        if a_dir == 1:
            self.tp_price = entry_price + self.tp_distance
            self.sl_price = entry_price - self.sl_distance
        else:
            self.tp_price = entry_price - self.tp_distance
            self.sl_price = entry_price + self.sl_distance
        self.max_favorable = 0.0
        self.hit_breakeven = False
        self.expected_bars = a_bars * 1.2

    def update(self, bar_idx, high, low, bars_elapsed):
        if self.a_dir == 1:
            favorable = high - self.entry_price
        else:
            favorable = self.entry_price - low
        if favorable > self.max_favorable:
            self.max_favorable = favorable
        progress = self.max_favorable / self.tp_distance if self.tp_distance > 0 else 0
        time_frac = bars_elapsed / self.expected_bars if self.expected_bars > 0 else 1
        speed = progress / time_frac if time_frac > 0.05 else progress * 20

        # Check SL
        if self.a_dir == 1:
            if low <= self.sl_price:
                return self.sl_price, self.tp_price, True, self.sl_price
        else:
            if high >= self.sl_price:
                return self.sl_price, self.tp_price, True, self.sl_price

        if progress >= 0.30 and not self.hit_breakeven:
            self.sl_price = self.entry_price
            self.hit_breakeven = True
            if speed > 1.2:
                self._expand_tp(min(0.10, (speed - 1.2) * 0.05))

        if progress >= 0.60:
            self._move_sl_to_lock(0.50)
            if speed > 1.5:
                self._expand_tp(min(0.25, (speed - 1.5) * 0.10))
            elif speed < 0.5 and bars_elapsed > self.expected_bars:
                self._contract_tp(min(0.20, (0.5 - speed) * 0.15))

        if progress >= 1.0:
            self._expand_tp(0.30)
            self._move_sl_to_lock(0.65)
            if self.entry_score > 0.7:
                self._expand_tp((self.entry_score - 0.7) * 0.5)

        if progress >= 1.5:
            self._expand_tp(0.50)
            self._move_sl_to_lock(0.75)

        if progress >= 2.0:
            self._move_sl_to_lock(0.85)

        if bars_elapsed > self.expected_bars * 2.5 and progress < 0.40:
            if self.max_favorable > 0:
                shrink_target = self.entry_price + self.max_favorable * 1.05 * self.a_dir
                if self.a_dir == 1:
                    self.tp_price = min(self.tp_price, shrink_target)
                else:
                    self.tp_price = max(self.tp_price, shrink_target)

        # Check TP
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
            if len(row) < 6:
                continue
            dates.append(f"{row[0]} {row[1]}")
            opens.append(float(row[2]))
            highs.append(float(row[3]))
            lows.append(float(row[4]))
            closes.append(float(row[5]))
    return dates, np.array(opens), np.array(highs), np.array(lows), np.array(closes)


def get_all_pairs():
    pairs = set()
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.csv'):
            parts = fname.replace('.csv', '').rsplit('_', 1)
            if len(parts) == 2:
                pairs.add(parts[0])
    return sorted(pairs)


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST WITH COSTS AND NO OVERLAP
# ═══════════════════════════════════════════════════════════════════════

def backtest_pair_tf(args):
    pair, tf = args

    data = load_ohlcv(pair, tf)
    if data is None:
        return []

    dates, opens, highs, lows, closes = data
    n_bars = len(closes)
    spread_cost = get_spread_cost(pair)

    # Single ZZ config: best validated
    zz = OnlineZigZag(deviation_pct=1.0, confirm_bars=5)
    for i in range(n_bars):
        zz.process_bar(i, highs[i], lows[i])

    if len(zz.pivots) < 4:
        return []

    pivots = zz.pivots
    trades = []
    in_trade_until = -1  # bar index when current trade ends (no overlap)

    for i in range(len(pivots) - 2):
        p0 = pivots[i]
        p1 = pivots[i+1]
        p2 = pivots[i+2]

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

        entry_bar = p2[0] + 1
        if entry_bar >= n_bars - 10:
            continue

        # ── No overlap check ──
        if entry_bar <= in_trade_until:
            continue

        entry_price = closes[entry_bar]
        entry_year = int(dates[entry_bar][:4])

        # ── Apply spread cost to entry: worse fill ──
        # Long: buy at ask = close + half_spread
        # Short: sell at bid = close - half_spread
        half_spread = spread_cost / 2
        if a_dir == 1:
            entry_price_adj = entry_price + half_spread
        else:
            entry_price_adj = entry_price - half_spread

        # Check if spread cost is too large relative to TP
        tp_distance = a_amp * tp_mult
        if spread_cost / tp_distance > 0.15:  # spread > 15% of TP → skip
            continue

        dm = DynamicManager(
            entry_price=entry_price_adj,
            a_dir=a_dir,
            a_amp=a_amp,
            a_bars=a_bars,
            tp_mult=tp_mult,
            sl_ratio=sl_ratio,
            entry_score=score,
        )

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
                if close_price == dm.sl_price:
                    exit_reason = 'sl'
                elif close_price == dm.tp_price:
                    exit_reason = 'tp'
                else:
                    exit_reason = 'dynamic'
                break

        # ── Apply spread cost to exit ──
        if a_dir == 1:
            exit_price_adj = exit_price - half_spread  # sell at bid
        else:
            exit_price_adj = exit_price + half_spread  # buy at ask

        # PnL with costs
        if a_dir == 1:
            pnl_price = exit_price_adj - entry_price_adj
        else:
            pnl_price = entry_price_adj - exit_price_adj

        initial_sl_dist = dm.initial_sl_distance
        pnl_r = pnl_price / initial_sl_dist if initial_sl_dist > 0 else 0

        # Mark trade end for overlap prevention
        in_trade_until = exit_bar

        trades.append({
            'pair': pair,
            'tf': tf,
            'entry_bar': entry_bar,
            'entry_date': dates[entry_bar],
            'entry_year': entry_year,
            'entry_price': entry_price_adj,
            'exit_bar': exit_bar,
            'exit_date': dates[min(exit_bar, n_bars - 1)],
            'exit_price': exit_price_adj,
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
            'spread_cost': spread_cost,
            'pnl_price': pnl_price,
            'pnl_r': pnl_r,
            'pnl_weighted': pnl_r * pos_mult,
            'hold_bars': exit_bar - entry_bar,
        })

    return trades


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def compute_stats(trades, weighted=False):
    if not trades:
        return None
    pnls = [t['pnl_weighted' if weighted else 'pnl_r'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
    mean_r = np.mean(pnls)
    std_r = np.std(pnls)

    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = np.max(dd) if len(dd) > 0 else 0

    return {
        'n': len(pnls), 'wr': len(wins) / len(pnls) * 100,
        'avg_r': mean_r, 'std_r': std_r,
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
        'pf': pf, 'total_r': sum(pnls),
        'max_dd_r': max_dd,
        'calmar': sum(pnls) / max_dd if max_dd > 0 else float('inf'),
        'sharpe_trade': mean_r / std_r if std_r > 0 else 0,
    }


def analyze(all_trades):
    is_t = [t for t in all_trades if t['entry_year'] <= 2018]
    oos_t = [t for t in all_trades if t['entry_year'] > 2018]

    for label, trades, w in [
        ("UNWEIGHTED", all_trades, False),
        ("POSITION-WEIGHTED", all_trades, True),
    ]:
        s_all = compute_stats(trades, w)
        s_is = compute_stats([t for t in trades if t['entry_year'] <= 2018], w)
        s_oos = compute_stats([t for t in trades if t['entry_year'] > 2018], w)

        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")
        for name, s in [("ALL", s_all), ("IS 2000-2018", s_is), ("OOS 2019-2024", s_oos)]:
            if s:
                print(f"  {name:>15s}: n={s['n']:>6,} WR={s['wr']:.1f}% avgR={s['avg_r']:.4f} "
                      f"PF={s['pf']:.2f} totalR={s['total_r']:.0f} maxDD={s['max_dd_r']:.1f}R "
                      f"calmar={s['calmar']:.1f}")

    # By year
    print(f"\n{'='*80}")
    print(f"  BY YEAR (position-weighted)")
    print(f"{'='*80}")
    years = sorted(set(t['entry_year'] for t in all_trades))
    print(f"{'Year':>6s} {'n':>6s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s} {'totalR':>8s} {'maxDD':>7s}")
    print("-" * 55)
    for y in years:
        yt = [t for t in all_trades if t['entry_year'] == y]
        s = compute_stats(yt, weighted=True)
        if s:
            marker = " *OOS*" if y > 2018 else ""
            print(f"{y:>6d} {s['n']:>6,} {s['wr']:>6.1f} {s['avg_r']:>8.4f} "
                  f"{s['pf']:>7.2f} {s['total_r']:>8.1f} {s['max_dd_r']:>7.1f}{marker}")

    # By pair OOS
    print(f"\n{'='*80}")
    print(f"  BY PAIR (OOS, position-weighted)")
    print(f"{'='*80}")
    pair_results = defaultdict(list)
    for t in oos_t:
        pair_results[t['pair']].append(t)

    pair_stats = []
    for pair, trades in pair_results.items():
        s = compute_stats(trades, weighted=True)
        if s and s['n'] >= 5:
            pair_stats.append((pair, s))
    pair_stats.sort(key=lambda x: x[1]['avg_r'], reverse=True)

    print(f"{'Pair':>10s} {'n':>5s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s} {'totalR':>8s} {'spread':>8s}")
    print("-" * 60)
    n_pos = 0
    for pair, s in pair_stats:
        sp = get_spread_cost(pair)
        print(f"{pair:>10s} {s['n']:>5,} {s['wr']:>6.1f} {s['avg_r']:>8.4f} "
              f"{s['pf']:>7.2f} {s['total_r']:>8.1f} {sp:>8.5f}")
        if s['avg_r'] > 0:
            n_pos += 1
    print(f"\nPositive OOS pairs: {n_pos}/{len(pair_stats)} ({n_pos/len(pair_stats)*100:.0f}%)")

    # By score bucket
    print(f"\n{'='*80}")
    print(f"  BY SCORE BUCKET")
    print(f"{'='*80}")
    for lo, hi, label in [(0.3, 0.5, 'Low'), (0.5, 0.7, 'Med'), (0.7, 0.85, 'High'), (0.85, 1.01, 'Elite')]:
        b_is = [t for t in is_t if lo <= t['score'] < hi]
        b_oos = [t for t in oos_t if lo <= t['score'] < hi]
        s_is = compute_stats(b_is, True)
        s_oos = compute_stats(b_oos, True)
        print(f"  {label:>5s} ({lo:.2f}-{hi:.2f}):")
        if s_is:
            print(f"    IS:  n={s_is['n']:>5,} WR={s_is['wr']:.1f}% avgR={s_is['avg_r']:.4f} PF={s_is['pf']:.2f}")
        if s_oos:
            print(f"    OOS: n={s_oos['n']:>5,} WR={s_oos['wr']:.1f}% avgR={s_oos['avg_r']:.4f} PF={s_oos['pf']:.2f}")

    # Spread impact analysis
    print(f"\n{'='*80}")
    print(f"  SPREAD IMPACT ANALYSIS")
    print(f"{'='*80}")
    spread_costs_r = []
    for t in all_trades:
        cost_r = t['spread_cost'] / (t['a_amp'] * t['tp_mult'] * t['sl_ratio']) if t['a_amp'] > 0 else 0
        spread_costs_r.append(cost_r)
    sc = np.array(spread_costs_r)
    print(f"  Spread cost as fraction of initial SL:")
    print(f"    Mean: {np.mean(sc):.4f}R  Median: {np.median(sc):.4f}R")
    print(f"    P10: {np.percentile(sc, 10):.4f}R  P90: {np.percentile(sc, 90):.4f}R")
    print(f"    Total spread cost: {np.sum(sc):.0f}R across {len(sc):,} trades")

    # SL exit analysis
    print(f"\n{'='*80}")
    print(f"  SL EXIT BREAKDOWN")
    print(f"{'='*80}")
    sl_trades = [t for t in all_trades if t['exit_reason'] == 'sl']
    sl_pnls = [t['pnl_r'] for t in sl_trades]
    sl_arr = np.array(sl_pnls)
    print(f"  Total SL exits: {len(sl_arr):,}")
    print(f"  Real losses (pnl < -0.1): {np.sum(sl_arr < -0.1):,} ({np.sum(sl_arr < -0.1)/len(sl_arr)*100:.1f}%)")
    print(f"  Breakeven (-0.1 to 0.1): {np.sum((sl_arr >= -0.1) & (sl_arr <= 0.1)):,} ({np.sum((sl_arr >= -0.1) & (sl_arr <= 0.1))/len(sl_arr)*100:.1f}%)")
    print(f"  Locked profit (> 0.1): {np.sum(sl_arr > 0.1):,} ({np.sum(sl_arr > 0.1)/len(sl_arr)*100:.1f}%)")


def main():
    pairs = get_all_pairs()
    tfs = ['H1', 'M30', 'M15']

    print(f"Realistic Backtest (with costs, no overlap)")
    print(f"  Pairs: {len(pairs)}")
    print(f"  TFs: {tfs}")
    print(f"  ZZ: dev=1.0, confirm=5 (single config)")
    print(f"  Costs: spread + 0.5 pip slippage each way")
    print(f"  Overlap: NOT allowed (one trade per pair-TF at a time)")
    print()

    tasks = [(pair, tf) for pair in pairs for tf in tfs]
    print(f"Total tasks: {len(tasks)}")

    all_trades = []
    with Pool(40) as pool:
        for batch in pool.imap_unordered(backtest_pair_tf, tasks):
            all_trades.extend(batch)

    print(f"\nTotal trades: {len(all_trades):,}")

    all_trades.sort(key=lambda t: (t['entry_year'], t['entry_bar']))

    # Save
    out_path = "/home/ubuntu/stage2_abc/realistic_trades.csv"
    if all_trades:
        fieldnames = list(all_trades[0].keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in all_trades:
                writer.writerow(t)
        print(f"Saved to: {out_path}")

    analyze(all_trades)


if __name__ == '__main__':
    main()
