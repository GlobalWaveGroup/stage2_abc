"""
Stage2 ABC V5 — Full Ablation Study (Optimized)
=================================================
Core philosophy:
  - B wave maturity is THE core condition, no exception
  - zigzag(2,1,1) is the base measurement tool
  - Use ablation to find optimal boundary for each dimension
  - Conditions are interdependent: strong X relaxes threshold for Y

OPTIMIZATION vs previous attempt:
  Instead of running full trade simulation per entry, record MFE/MAE curves.
  MFE = max favorable excursion within N bars (tells us: IF we entered here,
  how far did price go in our favor?)
  MAE = max adverse excursion (how far against us?)
  
  From MFE/MAE we can compute results for ANY TP/SL combination analytically.

ALL CONDITION DIMENSIONS:
  1. merge_level (L3-L7)
  2. b_depth_ratio
  3. b_time_ratio  
  4. b_slope_ratio
  5. b_decel_ratio
  6. turn_bars (bars since B extreme — KEY maturity dimension)
  7. turn_depth_ratio (price rebound from B extreme — KEY maturity dimension)
  8. close_bullish (entry bar close in C direction)
  9. entry_from_extreme (distance from B extreme / A_amp)
"""

import os
import csv
import numpy as np
from multiprocessing import Pool
from collections import defaultdict

DATA_DIR = "/home/ubuntu/DataBase/base_kline"

# ═══════════════════════════════════════════════════════════════════════
# ZIGZAG + MERGE (identical to V3/V4)
# ═══════════════════════════════════════════════════════════════════════

def zigzag_211_batch(highs, lows, dev_points=0.0001):
    n = len(highs)
    if n < 3:
        return []
    pivots = []
    if highs[0] >= highs[1]:
        pivots.append((0, highs[0], 1))
        searching = -1
        last_low = lows[0]; last_low_bar = 0
        last_high = highs[0]; last_high_bar = 0
    else:
        pivots.append((0, lows[0], -1))
        searching = 1
        last_high = highs[0]; last_high_bar = 0
        last_low = lows[0]; last_low_bar = 0
    for i in range(1, n):
        if searching == 1:
            if highs[i] > last_high:
                last_high = highs[i]; last_high_bar = i
            elif last_high_bar == i - 1:
                if last_high - pivots[-1][1] > dev_points:
                    pivots.append((last_high_bar, last_high, 1))
                    searching = -1
                    last_low = lows[i]; last_low_bar = i
        else:
            if lows[i] < last_low:
                last_low = lows[i]; last_low_bar = i
            elif last_low_bar == i - 1:
                if pivots[-1][1] - last_low > dev_points:
                    pivots.append((last_low_bar, last_low, -1))
                    searching = 1
                    last_high = highs[i]; last_high_bar = i
    return pivots


def merge_pivots_once(pivots):
    if len(pivots) < 3:
        return pivots[:], False
    result = []; i = 0; changed = False
    while i < len(pivots):
        if i + 2 >= len(pivots):
            result.extend(pivots[i:]); break
        p1, p2, p3 = pivots[i], pivots[i+1], pivots[i+2]
        can_merge = False
        if p1[2] == 1 and p2[2] == -1 and p3[2] == 1:
            if p3[1] >= p1[1]: can_merge = True
        elif p1[2] == -1 and p2[2] == 1 and p3[2] == -1:
            if p3[1] <= p1[1]: can_merge = True
        if can_merge:
            kept = (p3 if (p3[1] > p1[1] if p1[2] == 1 else p3[1] < p1[1]) else p1)
            result.append(kept); i += 3; changed = True
        else:
            result.append(p1); i += 1
    return result, changed


def merge_to_level(pivots, levels):
    cur = pivots[:]
    for _ in range(levels):
        cur, changed = merge_pivots_once(cur)
        if not changed: break
    return cur


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
    'XAUUSD': {'spread': 3.0, 'pip': 0.01}, 'XAGUSD': {'spread': 3.0, 'pip': 0.001},
}

def get_spread_cost(pair):
    info = PAIR_INFO.get(pair, {'spread': 5.0, 'pip': 0.0001})
    return (info['spread'] + 1.0) * info['pip']

def get_dev(pair):
    return PAIR_INFO.get(pair, {'pip': 0.0001})['pip']

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

def get_all_pairs():
    pairs = set()
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.csv'):
            parts = fname.replace('.csv', '').rsplit('_', 1)
            if len(parts) == 2 and parts[0] in PAIR_INFO:
                pairs.add(parts[0])
    return sorted(pairs)


# ═══════════════════════════════════════════════════════════════════════
# CORE: COLLECT ENTRIES WITH MFE/MAE (fast — no trade simulation)
# ═══════════════════════════════════════════════════════════════════════

def compute_mfe_mae(highs, lows, entry_bar, a_dir, n_bars, max_hold):
    """
    Compute MFE and MAE over the trade window.
    Returns (mfe, mae) in price units.
    
    MFE = maximum favorable excursion (best case for us)
    MAE = maximum adverse excursion (worst case for us)
    """
    end = min(entry_bar + max_hold, n_bars - 1)
    mfe = 0.0
    mae = 0.0
    
    for k in range(entry_bar + 1, end + 1):
        if a_dir == 1:
            fav = highs[k] - 0  # placeholder, adjusted by caller
            adv = 0 - lows[k]
        else:
            fav = 0 - lows[k]
            adv = highs[k] - 0
    
    # Vectorized version (much faster)
    if end <= entry_bar:
        return 0.0, 0.0
    
    window_h = highs[entry_bar+1:end+1]
    window_l = lows[entry_bar+1:end+1]
    entry_price = (highs[entry_bar] + lows[entry_bar]) / 2  # use mid as proxy
    
    if a_dir == 1:
        mfe = float(np.max(window_h)) - entry_price  # max upside
        mae = entry_price - float(np.min(window_l))   # max downside
    else:
        mfe = entry_price - float(np.min(window_l))
        mae = float(np.max(window_h)) - entry_price
    
    return max(mfe, 0.0), max(mae, 0.0)


def collect_entries_fast(pair, tf, dates, opens, highs, lows, closes, pivots,
                         merge_level, spread_cost):
    """
    For each A wave, track B bar-by-bar.
    At maturity-relevant bars, record features + MFE/MAE.
    
    Sampling strategy:
    - turn_bars == 0: sample every Nth (B still extending)
    - turn_bars > 0: sample every bar (B potentially maturing)
    """
    n = len(closes)
    entries = []
    half_spread = spread_cost / 2

    for i in range(len(pivots) - 1):
        p0, p1 = pivots[i], pivots[i + 1]
        a_bars = p1[0] - p0[0]
        a_amp = abs(p1[1] - p0[1])
        a_dir = 1 if p1[1] > p0[1] else -1
        a_end_bar = p1[0]
        a_end_price = p1[1]

        if a_bars < 3 or a_amp <= 0:
            continue

        a_slope = a_amp / a_bars
        max_b_bars = min(int(a_bars * 8), 1500)
        max_b_depth = a_amp * 1.5

        b_extreme_price = a_end_price
        b_extreme_bar = a_end_bar
        early_sample = max(2, a_bars // 3)
        samples_this_a = 0
        max_samples_per_a = 60  # cap to control memory

        # MFE/MAE window = 5 * A_bars (enough for C to develop)
        mfe_window = max(int(a_bars * 5), 200)

        for j in range(a_end_bar + 1, min(a_end_bar + max_b_bars, n - mfe_window - 10)):
            if samples_this_a >= max_samples_per_a:
                break
            h, l, c, o = highs[j], lows[j], closes[j], opens[j]

            # Track B extreme
            if a_dir == 1:
                if l < b_extreme_price:
                    b_extreme_price = l; b_extreme_bar = j
                b_depth = a_end_price - b_extreme_price
            else:
                if h > b_extreme_price:
                    b_extreme_price = h; b_extreme_bar = j
                b_depth = b_extreme_price - a_end_price

            if b_depth > max_b_depth:
                break

            b_depth_ratio = b_depth / a_amp
            b_bars_so_far = j - a_end_bar

            if b_depth_ratio < 0.05 or b_bars_so_far < 2:
                continue

            turn_bars = j - b_extreme_bar

            # Sampling: sparse when B still extending, dense when maturing
            if turn_bars == 0:
                if b_bars_so_far % early_sample != 0:
                    continue
            # turn_bars > 0: sample every bar (maturity phase)

            b_time_ratio = b_bars_so_far / a_bars
            b_amp = abs(b_extreme_price - a_end_price)
            b_bars_to_ext = max(1, b_extreme_bar - a_end_bar)
            b_slope = b_amp / b_bars_to_ext
            b_slope_ratio = b_slope / a_slope if a_slope > 0 else 999.0

            # Deceleration
            lookback = max(3, b_bars_so_far // 3)
            if b_bars_so_far >= lookback * 2:
                early_move = abs(closes[a_end_bar + lookback] - a_end_price)
                late_move = abs(c - closes[j - lookback])
                b_decel = late_move / early_move if early_move > 0 else 1.0
            else:
                b_decel = 1.0

            # Turn features
            if a_dir == 1:
                turn_depth = c - b_extreme_price
            else:
                turn_depth = b_extreme_price - c
            turn_depth_ratio = turn_depth / a_amp

            close_bullish = 1 if ((a_dir == 1 and c > o) or (a_dir == -1 and c < o)) else 0
            entry_from_ext = abs(c - b_extreme_price) / a_amp

            # ═══ MFE / MAE (vectorized, fast) ═══
            end_idx = min(j + mfe_window, n - 1)
            if end_idx <= j:
                continue
            wh = highs[j+1:end_idx+1]
            wl = lows[j+1:end_idx+1]

            if a_dir == 1:
                entry_adj = c + half_spread
                mfe = float(np.max(wh)) - entry_adj
                mae = entry_adj - float(np.min(wl))
            else:
                entry_adj = c - half_spread
                mfe = entry_adj - float(np.min(wl))
                mae = float(np.max(wh)) - entry_adj

            # Normalize by A_amp
            mfe_r = mfe / a_amp
            mae_r = mae / a_amp

            year = int(dates[j][:4]) if j < len(dates) else 0

            entries.append((
                merge_level, year,
                round(a_amp, 6), a_bars, a_dir,
                round(b_depth_ratio, 4), round(b_time_ratio, 4),
                round(b_slope_ratio, 4), round(b_decel, 4),
                turn_bars, round(turn_depth_ratio, 4),
                close_bullish, round(entry_from_ext, 4),
                round(mfe_r, 4), round(mae_r, 4),
            ))
            samples_this_a += 1

    return entries


FIELDS = [
    'merge_level', 'year',
    'a_amp', 'a_bars', 'a_dir',
    'b_depth_ratio', 'b_time_ratio', 'b_slope_ratio', 'b_decel_ratio',
    'turn_bars', 'turn_depth_ratio',
    'close_bullish', 'entry_from_extreme',
    'mfe_r', 'mae_r',
]


def process_task(args):
    pair, tf, merge_level = args
    data = load_ohlcv(pair, tf)
    if data is None:
        return []
    dates, opens, highs, lows, closes = data
    dev = get_dev(pair)
    spread = get_spread_cost(pair)
    l0 = zigzag_211_batch(highs, lows, dev)
    if len(l0) < 5:
        return []
    pivots = merge_to_level(l0, merge_level)
    if len(pivots) < 3:
        return []
    entries = collect_entries_fast(pair, tf, dates, opens, highs, lows, closes,
                                   pivots, merge_level, spread)
    print(f"  {pair}_{tf}_L{merge_level}: {len(entries):,} samples")
    return entries


# ═══════════════════════════════════════════════════════════════════════
# ABLATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def load_data_np(path):
    """Load CSV into dict of numpy arrays."""
    data = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [[] for _ in header]
        for row in reader:
            for i, v in enumerate(row):
                try:
                    cols[i].append(float(v))
                except ValueError:
                    cols[i].append(v)
    for i, name in enumerate(header):
        try:
            data[name] = np.array(cols[i], dtype=float)
        except (ValueError, TypeError):
            data[name] = np.array(cols[i])
    return data


def sim_from_mfe_mae(mfe_r, mae_r, tp_r, sl_r):
    """
    Given MFE and MAE (in A_amp units), compute trade outcome for given TP/SL.
    
    Logic: if MAE > SL, we get stopped out (PnL = -SL).
           if MFE > TP and MAE <= SL, we hit TP (PnL = +TP).
           if neither, timeout at some intermediate value.
           
    Conservative assumption: if both SL and TP are reachable, SL hits first.
    This is the WORST CASE for us, i.e., results are a LOWER BOUND.
    
    Returns pnl in R units (pnl / sl_r).
    """
    if mae_r >= sl_r:
        return -1.0  # stopped out
    if mfe_r >= tp_r:
        return tp_r / sl_r  # TP hit
    # Neither: use (MFE - MAE) / 2 as rough estimate
    return (mfe_r - mae_r) / (2 * sl_r) if sl_r > 0 else 0


def stats_from_pnls(pnls):
    """Compute stats from array of pnl_r values."""
    if len(pnls) < 10:
        return None
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    pf = abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float('inf')
    return {
        'n': len(pnls),
        'wr': len(wins) / len(pnls) * 100,
        'avgR': float(np.mean(pnls)),
        'pf': pf,
        'medR': float(np.median(pnls)),
    }


def run_ablation(data):
    """Full ablation study using MFE/MAE data."""
    n = len(data['mfe_r'])
    mfe = data['mfe_r']
    mae = data['mae_r']
    years = data['year']
    
    is_mask = years <= 2018
    oos_mask = years > 2018
    
    # Use a reasonable TP/SL: TP = 0.6 * A_amp, SL = 0.35 * A_amp → R:R = 1.71:1
    tp_r = 0.60
    sl_r = 0.35
    
    # Vectorized PnL computation
    pnls = np.where(mae >= sl_r, -1.0,
            np.where(mfe >= tp_r, tp_r / sl_r,
                     (mfe - mae) / (2 * sl_r)))

    baseline = stats_from_pnls(pnls)
    
    print(f"\n{'#'*90}")
    print(f"  V5 ABLATION STUDY — B Wave Maturity as Core Condition")
    print(f"  Samples: {n:,}   TP={tp_r:.2f}A  SL={sl_r:.2f}A  R:R={tp_r/sl_r:.1f}:1")
    print(f"  Baseline: WR={baseline['wr']:.1f}%  avgR={baseline['avgR']:.4f}  PF={baseline['pf']:.2f}")
    print(f"{'#'*90}")
    
    def report(label, mask, show_oos=True):
        sub = pnls[mask]
        s = stats_from_pnls(sub)
        if s is None:
            print(f"  {label:>45s}  n/a")
            return
        line = f"  {label:>45s}  n={s['n']:>9,}  WR={s['wr']:>5.1f}%  avgR={s['avgR']:>+8.4f}  PF={s['pf']:>6.2f}"
        if show_oos:
            sub_oos = pnls[mask & oos_mask]
            so = stats_from_pnls(sub_oos)
            if so:
                line += f"  |OOS: n={so['n']:>7,} avgR={so['avgR']:>+8.4f} PF={so['pf']:>6.2f}"
        print(line)

    # ═══════════════════════════════════════════════════════════
    # PART 1: THE CORE — turn_bars (B maturity bars since extreme)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  PART 1: turn_bars — THE CORE maturity dimension")
    print(f"  (bars since B extreme = how long price has NOT made new extreme)")
    print(f"{'='*90}")
    
    tb = data['turn_bars']
    for lo, hi in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,7),(7,10),(10,15),(15,20),(20,30),(30,50),(50,999)]:
        report(f"turn_bars [{lo},{hi})", (tb >= lo) & (tb < hi))
    
    print(f"\n  Cumulative: turn_bars >= threshold")
    for thresh in [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50]:
        report(f"turn_bars >= {thresh}", tb >= thresh)

    # ═══════════════════════════════════════════════════════════
    # PART 2: turn_depth_ratio (price rebound from B extreme)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  PART 2: turn_depth_ratio — price rebound from B extreme")
    print(f"  (positive = price moved back toward C direction)")
    print(f"{'='*90}")
    
    td = data['turn_depth_ratio']
    for lo, hi in [(-0.5,-0.2),(-0.2,-0.1),(-0.1,-0.05),(-0.05,0.0),
                   (0.0,0.02),(0.02,0.05),(0.05,0.10),(0.10,0.15),(0.15,0.20),(0.20,0.30),(0.30,0.50)]:
        report(f"turn_depth [{lo:.2f},{hi:.2f})", (td >= lo) & (td < hi))
    
    print(f"\n  Cumulative: turn_depth_ratio >= threshold")
    for thresh in [-0.20, -0.10, -0.05, 0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
        report(f"turn_depth >= {thresh:.2f}", td >= thresh)

    # ═══════════════════════════════════════════════════════════
    # PART 3: Other dimensions (one at a time)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  PART 3: Other single dimensions")
    print(f"{'='*90}")
    
    # merge_level
    print(f"\n  merge_level:")
    ml = data['merge_level']
    for v in [3, 4, 5, 6, 7]:
        report(f"merge_level = {v}", ml == v)
    print(f"  Cumulative:")
    for v in [3, 4, 5, 6]:
        report(f"merge_level >= {v}", ml >= v)

    # b_slope_ratio
    print(f"\n  b_slope_ratio (lower = B weaker than A):")
    sr = data['b_slope_ratio']
    for lo, hi in [(0,0.05),(0.05,0.10),(0.10,0.20),(0.20,0.50),(0.50,1.0),(1.0,2.0),(2.0,999)]:
        report(f"slope [{lo:.2f},{hi:.2f})", (sr >= lo) & (sr < hi))
    print(f"  Cumulative:")
    for thresh in [999, 2.0, 1.0, 0.50, 0.30, 0.20, 0.15, 0.10, 0.05]:
        report(f"slope < {thresh:.2f}", sr < thresh)

    # b_depth_ratio
    print(f"\n  b_depth_ratio (B retracement depth):")
    dr = data['b_depth_ratio']
    for lo, hi in [(0.05,0.15),(0.15,0.25),(0.25,0.382),(0.382,0.50),(0.50,0.618),(0.618,0.786),(0.786,1.0),(1.0,1.5)]:
        report(f"depth [{lo:.3f},{hi:.3f})", (dr >= lo) & (dr < hi))

    # b_time_ratio
    print(f"\n  b_time_ratio (B duration / A duration):")
    tr = data['b_time_ratio']
    for lo, hi in [(0.2,0.5),(0.5,1.0),(1.0,2.0),(2.0,3.0),(3.0,5.0),(5.0,10.0)]:
        report(f"time [{lo:.1f},{hi:.1f})", (tr >= lo) & (tr < hi))

    # b_decel_ratio
    print(f"\n  b_decel_ratio (B deceleration):")
    dc = data['b_decel_ratio']
    for lo, hi in [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.0),(1.0,1.5),(1.5,3.0),(3.0,999)]:
        report(f"decel [{lo:.1f},{hi:.1f})", (dc >= lo) & (dc < hi))

    # close_bullish
    print(f"\n  close_bullish (entry bar close in C direction):")
    cb = data['close_bullish']
    report("close NOT in C dir (0)", cb == 0)
    report("close IN C dir (1)", cb == 1)

    # ═══════════════════════════════════════════════════════════
    # PART 4: PROGRESSIVE STACKING — building the condition matrix
    # Core logic: start with B maturity, then add supporting conditions
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  PART 4: PROGRESSIVE STACKING (B maturity first, then add support)")
    print(f"{'='*90}")

    # Stack A: turn_bars focused
    print(f"\n  Stack A: Start with turn_bars, add conditions")
    m = np.ones(n, dtype=bool)
    report("ALL (baseline)", m)
    for tb_thresh in [2, 3, 5, 8, 10, 15, 20]:
        m_tb = tb >= tb_thresh
        report(f"turn>={tb_thresh}", m_tb)
        
        # Now add slope
        for sl_thresh in [0.50, 0.30, 0.20, 0.10]:
            m2 = m_tb & (sr < sl_thresh)
            report(f"  turn>={tb_thresh} & slope<{sl_thresh}", m2)
        
        # Add turn_depth
        for td_thresh in [0.0, 0.05, 0.10]:
            m2 = m_tb & (td >= td_thresh)
            report(f"  turn>={tb_thresh} & tdepth>={td_thresh}", m2)

    # Stack B: Interaction — when turn is strong, can slope threshold relax?
    print(f"\n{'='*90}")
    print(f"  PART 5: INTERACTION — does strong turn relax slope requirement?")
    print(f"{'='*90}")
    
    print(f"\n  Slope threshold needed for positive avgR, at each turn_bars level:")
    print(f"  {'turn_bars':>12s} | ", end='')
    for s in [999, 1.0, 0.50, 0.30, 0.20, 0.10, 0.05]:
        print(f" slope<{s:<5.2f}", end='')
    print()
    print(f"  {'-'*90}")
    
    for tb_thresh in [0, 2, 3, 5, 8, 10, 15, 20, 30]:
        m_tb = tb >= tb_thresh
        print(f"  turn>={tb_thresh:>3d}     | ", end='')
        for s in [999, 1.0, 0.50, 0.30, 0.20, 0.10, 0.05]:
            m_s = m_tb & (sr < s)
            sub = pnls[m_s]
            if len(sub) < 200:
                print(f" {'n/a':>10s}", end='')
            else:
                print(f" {np.mean(sub):>+10.4f}", end='')
        print()

    # Stack C: 3-way: turn_bars × slope × turn_depth
    print(f"\n{'='*90}")
    print(f"  PART 6: 3-WAY — turn_bars × slope × turn_depth")
    print(f"{'='*90}")
    
    print(f"\n  Fixed slope<0.20, sweep turn_bars × turn_depth:")
    print(f"  {'':>12s} | ", end='')
    for td_t in [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]:
        print(f" td>={td_t:<+6.2f}", end='')
    print()
    print(f"  {'-'*100}")
    
    m_slope = sr < 0.20
    for tb_thresh in [0, 2, 3, 5, 8, 10, 15, 20]:
        m_tb = (tb >= tb_thresh) & m_slope
        print(f"  turn>={tb_thresh:>3d}     | ", end='')
        for td_t in [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]:
            m_td = m_tb & (td >= td_t)
            sub = pnls[m_td]
            if len(sub) < 100:
                print(f" {'n/a':>10s}", end='')
            else:
                avgr = np.mean(sub)
                marker = "*" if avgr > 0 else " "
                print(f" {avgr:>+9.4f}{marker}", end='')
        print()

    # Same but slope < 0.10
    print(f"\n  Fixed slope<0.10, sweep turn_bars × turn_depth:")
    print(f"  {'':>12s} | ", end='')
    for td_t in [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]:
        print(f" td>={td_t:<+6.2f}", end='')
    print()
    print(f"  {'-'*100}")
    
    m_slope = sr < 0.10
    for tb_thresh in [0, 2, 3, 5, 8, 10, 15, 20]:
        m_tb = (tb >= tb_thresh) & m_slope
        print(f"  turn>={tb_thresh:>3d}     | ", end='')
        for td_t in [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]:
            m_td = m_tb & (td >= td_t)
            sub = pnls[m_td]
            if len(sub) < 100:
                print(f" {'n/a':>10s}", end='')
            else:
                avgr = np.mean(sub)
                marker = "*" if avgr > 0 else " "
                print(f" {avgr:>+9.4f}{marker}", end='')
        print()

    # ═══════════════════════════════════════════════════════════
    # PART 7: Multiple TP/SL levels — which management helps most?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  PART 7: TP/SL sensitivity (at best condition combo)")
    print(f"{'='*90}")
    
    # Use a promising condition: turn>=5 & slope<0.20 & turn_depth>=0.05
    cond_mask = (tb >= 5) & (sr < 0.20) & (td >= 0.05)
    cond_mfe = mfe[cond_mask]
    cond_mae = mae[cond_mask]
    
    print(f"  Samples matching condition: {np.sum(cond_mask):,}")
    print(f"\n  {'TP':>6s} {'SL':>6s} {'R:R':>5s} | {'n':>8s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s} | OOS: {'n':>7s} {'avgR':>8s} {'PF':>7s}")
    print(f"  {'-'*85}")
    
    cond_mfe_oos = mfe[cond_mask & oos_mask]
    cond_mae_oos = mae[cond_mask & oos_mask]
    
    for tp_test in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]:
        for sl_test in [0.20, 0.30, 0.40, 0.50]:
            p = np.where(cond_mae >= sl_test, -1.0,
                np.where(cond_mfe >= tp_test, tp_test / sl_test,
                         (cond_mfe - cond_mae) / (2 * sl_test)))
            s = stats_from_pnls(p)
            if s is None: continue
            
            p_oos = np.where(cond_mae_oos >= sl_test, -1.0,
                    np.where(cond_mfe_oos >= tp_test, tp_test / sl_test,
                             (cond_mfe_oos - cond_mae_oos) / (2 * sl_test)))
            so = stats_from_pnls(p_oos)
            oos_str = f"{so['n']:>7,} {so['avgR']:>+8.4f} {so['pf']:>7.2f}" if so else "n/a"
            
            marker = " <<<" if s['avgR'] > 0 else ""
            print(f"  {tp_test:>6.2f} {sl_test:>6.2f} {tp_test/sl_test:>5.1f} | "
                  f"{s['n']:>8,} {s['wr']:>6.1f} {s['avgR']:>+8.4f} {s['pf']:>7.2f} | {oos_str}{marker}")

    # ═══════════════════════════════════════════════════════════
    # PART 8: The MFE/MAE profile — raw edge measurement
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  PART 8: RAW MFE/MAE PROFILE by condition")
    print(f"  (Higher MFE/MAE ratio = stronger directional edge)")
    print(f"{'='*90}")
    
    def mfe_mae_report(label, mask):
        m = mfe[mask]; a = mae[mask]
        if len(m) < 100: return
        ratio = np.mean(m) / np.mean(a) if np.mean(a) > 0 else float('inf')
        p_fav = np.mean(m > a) * 100  # % where MFE > MAE
        print(f"  {label:>45s}  n={len(m):>9,}  MFE={np.mean(m):>.4f}  MAE={np.mean(a):>.4f}"
              f"  MFE/MAE={ratio:>.3f}  P(MFE>MAE)={p_fav:>.1f}%")
    
    mfe_mae_report("ALL", np.ones(n, dtype=bool))
    for tb_thresh in [0, 2, 3, 5, 8, 10, 15, 20]:
        mfe_mae_report(f"turn>={tb_thresh}", tb >= tb_thresh)
    print()
    for tb_thresh in [5, 10]:
        for sl_thresh in [0.50, 0.20, 0.10]:
            mfe_mae_report(f"turn>={tb_thresh} & slope<{sl_thresh}", (tb >= tb_thresh) & (sr < sl_thresh))
        for td_thresh in [0.0, 0.05, 0.10]:
            mfe_mae_report(f"turn>={tb_thresh} & tdepth>={td_thresh}", (tb >= tb_thresh) & (td >= td_thresh))


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    import sys
    out_path = "/home/ubuntu/stage2_abc/abc_v5_samples.csv"

    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        print("Loading pre-collected data...")
        data = load_data_np(out_path)
        run_ablation(data)
        return

    pairs = get_all_pairs()
    tfs = ['H1', 'M30']
    merge_levels = [3, 4, 5, 6, 7]

    print(f"ABC V5 — Ablation Data Collection (MFE/MAE, optimized)")
    print(f"  Pairs: {len(pairs)}, TFs: {tfs}")
    print(f"  Merge levels: {merge_levels}")
    print(f"  Sampling: sparse during B extension, dense during B maturity")
    print(f"  Output: features + MFE_r + MAE_r per entry")
    print()

    tasks = [(p, tf, ml) for p in pairs for tf in tfs for ml in merge_levels]
    print(f"Total tasks: {len(tasks)}")
    print(f"Running with 60 workers...\n")

    # Stream write to avoid memory explosion
    total_samples = 0
    done = 0
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        with Pool(60) as pool:
            for batch in pool.imap_unordered(process_task, tasks):
                for e in batch:
                    writer.writerow(e)
                total_samples += len(batch)
                done += 1
                if done % 30 == 0:
                    f.flush()
                    print(f"  {done}/{len(tasks)} done, {total_samples:,} samples")

    print(f"\nTotal: {total_samples:,} samples")
    print(f"Saved: {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1e9:.2f} GB")

    if total_samples > 0:
        print("\n\nRunning ablation analysis...\n")
        data = load_data_np(out_path)
        run_ablation(data)


if __name__ == '__main__':
    main()
