"""
Stage2 ABC Collector V2 — FIXED ENTRY TIMING
=============================================
Critical fix: entry at confirm_bar + 1, NOT pivot_bar + 1.

The OnlineZigZag confirms a pivot at confirm_bar = pivot_bar + confirm_bars.
You can only ACT at confirm_bar + 1 (the next bar after confirmation).

This version stores confirm_bar for each pivot and uses it correctly.
"""

import os
import csv
import argparse
import numpy as np
from multiprocessing import Pool

DATA_DIR = "/home/ubuntu/DataBase/base_kline"


class OnlineZigZag:
    def __init__(self, deviation_pct=0.5, confirm_bars=5):
        self.dev = deviation_pct / 100.0
        self.confirm = confirm_bars
        self.pivots = []          # (bar, price, direction, confirm_bar)
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
                    pv = (self._tent[0], self._tent[1], +1, idx)  # (pivot_bar, price, dir, confirm_bar)
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
    return (dates, np.array(opens), np.array(highs),
            np.array(lows), np.array(closes))


def extract_abc_with_confirm(pivots):
    """
    Extract ABC triples using confirm_bar for entry timing.
    pivots: [(bar, price, dir, confirm_bar), ...]
    
    A = pivots[i] → pivots[i+1]
    B = pivots[i+1] → pivots[i+2]
    C = pivots[i+2] → pivots[i+3]
    
    Entry: at pivots[i+2].confirm_bar + 1 (first bar AFTER B is confirmed)
    """
    triples = []
    for i in range(len(pivots) - 3):
        p0_bar, p0_price, p0_dir, p0_conf = pivots[i]
        p1_bar, p1_price, p1_dir, p1_conf = pivots[i + 1]
        p2_bar, p2_price, p2_dir, p2_conf = pivots[i + 2]  # B end
        p3_bar, p3_price, p3_dir, p3_conf = pivots[i + 3]  # C end

        a_bars = p1_bar - p0_bar
        a_amp = abs(p1_price - p0_price)
        a_dir = 1 if p1_price > p0_price else -1

        b_bars = p2_bar - p1_bar
        b_amp = abs(p2_price - p1_price)

        c_bars = p3_bar - p2_bar
        c_amp = abs(p3_price - p2_price)
        c_dir = 1 if p3_price > p2_price else -1

        if a_bars <= 0 or b_bars <= 0 or a_amp <= 0:
            continue

        a_slope = a_amp / a_bars
        b_slope = b_amp / b_bars
        amp_ratio = b_amp / a_amp
        time_ratio = b_bars / a_bars
        slope_ratio = b_slope / a_slope if a_slope > 0 else 0

        # CRITICAL: entry uses confirm_bar, not pivot_bar
        entry_bar = p2_conf + 1   # ← FIXED: confirm_bar + 1

        triples.append({
            'a_start_bar': p0_bar, 'a_start_price': p0_price,
            'a_end_bar': p1_bar, 'a_end_price': p1_price,
            'b_end_bar': p2_bar, 'b_end_price': p2_price,
            'b_confirm_bar': p2_conf,  # NEW: store confirm bar
            'c_end_bar': p3_bar, 'c_end_price': p3_price,
            'a_dir': a_dir, 'a_amp': a_amp, 'a_bars': a_bars, 'a_slope': a_slope,
            'b_amp': b_amp, 'b_bars': b_bars, 'b_slope': b_slope,
            'amp_ratio': amp_ratio, 'time_ratio': time_ratio, 'slope_ratio': slope_ratio,
            'c_dir': c_dir, 'c_amp': c_amp, 'c_bars': c_bars,
            'c_follows_a': 1 if c_dir == a_dir else 0,
            'c_a_ratio': c_amp / a_amp if a_amp > 0 else 0,
            'entry_bar': entry_bar,
            'lookahead_bars': p2_conf - p2_bar,  # how many bars of delay
        })

    return triples


def simulate_trade(triple, highs, lows, closes):
    """Trade simulation — identical logic but using correct entry_bar."""
    entry_bar = triple['entry_bar']
    if entry_bar >= len(closes) - 1:
        return None

    a_amp = triple['a_amp']
    a_dir = triple['a_dir']
    a_bars = triple['a_bars']
    entry_price = closes[entry_bar]

    tp_distance = a_amp * 0.80
    sl_distance = tp_distance / 2.5

    if tp_distance <= 0 or sl_distance <= 0:
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
                exit_bar = bar; exit_price = current_sl; exit_reason = 'sl'; break
        else:
            if h >= current_sl:
                exit_bar = bar; exit_price = current_sl; exit_reason = 'sl'; break

        # Dynamic adjustments
        if progress >= 0.30 and not hit_breakeven:
            current_sl = entry_price; hit_breakeven = True

        if progress >= 0.60:
            if a_dir == 1:
                current_sl = max(current_sl, entry_price + max_favorable * 0.50)
            else:
                current_sl = min(current_sl, entry_price - max_favorable * 0.50)

        if progress >= 1.0:
            new_tp_dist = a_amp * 1.20
            if a_dir == 1:
                current_tp = max(current_tp, entry_price + new_tp_dist)
                current_sl = max(current_sl, entry_price + max_favorable - a_amp * 0.15)
            else:
                current_tp = min(current_tp, entry_price - new_tp_dist)
                current_sl = min(current_sl, entry_price - max_favorable + a_amp * 0.15)

        if progress >= 1.5:
            new_tp_dist = a_amp * 1.60
            if a_dir == 1:
                current_tp = max(current_tp, entry_price + new_tp_dist)
                current_sl = max(current_sl, entry_price + max_favorable - a_amp * 0.10)
            else:
                current_tp = min(current_tp, entry_price - new_tp_dist)
                current_sl = min(current_sl, entry_price - max_favorable + a_amp * 0.10)

        if bars_elapsed > a_bars * 2 and progress < 0.50 and max_favorable > 0:
            shrink = max_favorable * 1.10
            if a_dir == 1:
                current_tp = min(current_tp, entry_price + shrink)
            else:
                current_tp = max(current_tp, entry_price - shrink)

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

    pnl_r = pnl / sl_distance if sl_distance > 0 else 0

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


def process_pair_tf(args):
    pair, tf, zz_configs = args
    data = load_ohlcv(pair, tf)
    if data is None:
        return []

    dates, opens, highs, lows, closes = data
    results = []

    for dev_pct, confirm in zz_configs:
        zz = OnlineZigZag(deviation_pct=dev_pct, confirm_bars=confirm)
        for i in range(len(highs)):
            zz.process_bar(i, highs[i], lows[i])

        if len(zz.pivots) < 4:
            continue

        triples = extract_abc_with_confirm(zz.pivots)

        for t in triples:
            trade = simulate_trade(t, highs, lows, closes)
            if trade is None:
                continue
            row = {'pair': pair, 'tf': tf, 'zz_dev': dev_pct, 'zz_confirm': confirm}
            row.update(t)
            row.update(trade)
            results.append(row)

    print(f"  {pair}_{tf}: {len(results)} trades")
    return results


def get_all_pairs():
    pairs = set()
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.csv'):
            parts = fname.replace('.csv', '').rsplit('_', 1)
            if len(parts) == 2:
                pairs.add(parts[0])
    return sorted(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=str, default=None)
    parser.add_argument('--tfs', type=str, default='H1,M30,M15')
    parser.add_argument('--workers', type=int, default=40)
    parser.add_argument('--output', type=str,
                        default='/home/ubuntu/stage2_abc/abc_v2_triples.csv')
    args = parser.parse_args()

    pairs = args.pairs.split(',') if args.pairs else get_all_pairs()
    tfs = args.tfs.split(',')
    zz_configs = [(0.3, 3), (0.5, 5), (1.0, 5)]

    print(f"ABC Collector V2 (FIXED ENTRY TIMING)")
    print(f"  {len(pairs)} pairs × {len(tfs)} TFs × {len(zz_configs)} ZZ configs")
    print(f"  Entry = confirm_bar + 1 (NOT pivot_bar + 1)")
    print()

    tasks = [(pair, tf, zz_configs) for pair in pairs for tf in tfs]

    all_results = []
    with Pool(args.workers) as pool:
        for batch in pool.imap_unordered(process_pair_tf, tasks):
            all_results.extend(batch)

    print(f"\nTotal trades: {len(all_results):,}")

    if not all_results:
        print("ERROR: No results!")
        return

    # Save
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
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')

    # Also compare with anti-A direction
    # (re-simulate with reversed direction to test direction signal)
    print(f"\n{'='*70}")
    print(f"  V2 RESULTS (correct entry timing)")
    print(f"{'='*70}")
    print(f"Trades: {len(pnls):,}")
    print(f"Win rate: {len(wins)/len(pnls)*100:.1f}%")
    print(f"Avg PnL (R): {np.mean(pnls):.4f}")
    print(f"Median PnL (R): {np.median(pnls):.4f}")
    print(f"Avg Win (R): {np.mean(wins):.4f}" if wins else "No wins")
    print(f"Avg Loss (R): {np.mean(losses):.4f}" if losses else "No losses")
    print(f"Profit Factor: {pf:.2f}")

    # Lookahead stats
    delays = [r['lookahead_bars'] for r in all_results]
    print(f"\nConfirmation delay: mean={np.mean(delays):.1f} bars, "
          f"median={np.median(delays):.0f}, max={np.max(delays)}")

    # By exit reason
    from collections import defaultdict
    reasons = defaultdict(list)
    for r in all_results:
        reasons[r['exit_reason']].append(r['pnl_r'])
    print(f"\nBy exit reason:")
    for reason, rpnls in sorted(reasons.items()):
        wr = len([p for p in rpnls if p > 0]) / len(rpnls) * 100
        print(f"  {reason:10s}: n={len(rpnls):>8,}  WR={wr:.1f}%  "
              f"avg_R={np.mean(rpnls):.4f}")

    # By ZZ config
    zz_groups = defaultdict(list)
    for r in all_results:
        zz_groups[(r['zz_dev'], r['zz_confirm'])].append(r['pnl_r'])
    print(f"\nBy ZZ config:")
    for key in sorted(zz_groups.keys()):
        p = zz_groups[key]
        w = [x for x in p if x > 0]
        l = [x for x in p if x <= 0]
        pf_z = abs(sum(w) / sum(l)) if l and sum(l) != 0 else float('inf')
        print(f"  dev={key[0]:.1f} confirm={key[1]}: n={len(p):>8,}  "
              f"WR={len(w)/len(p)*100:.1f}%  avg_R={np.mean(p):.4f}  PF={pf_z:.2f}")


if __name__ == '__main__':
    main()
