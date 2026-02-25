"""
Stage2 ABC Bug Audit
=====================
Systematically test every possible source of bias/bug in the backtest.

BUG HYPOTHESIS 1: DIRECTION IS FREE
  In zigzag, A-B-C alternate direction. If A is up, C MUST start going up
  (by definition, the pivot at B's end means price reversed from B's direction).
  But does this mean we get a "free" direction signal?
  TEST: at B's confirmed end, does going in A's direction actually help?
  Compare: trade in A's direction vs trade in ANTI-A direction (same everything else).

BUG HYPOTHESIS 2: ENTRY TIMING LOOK-AHEAD
  B's end is confirmed by online zigzag with `confirm_bars` delay.
  We enter at `b_end_bar + 1`. But is `b_end_bar` the bar where the pivot
  actually occurred, or the bar where it was confirmed?
  In the collector: b_end_bar = pivot[i+2][0] = the actual pivot bar.
  But confirmation happens at pivot_confirm_bar = pivot bar + confirm_bars.
  SO WE MAY BE ENTERING BEFORE B IS ACTUALLY CONFIRMED!
  This is potentially a CRITICAL look-ahead bug.

BUG HYPOTHESIS 3: BAR-INTERNAL ORDER BIAS
  Within a bar, we check SL before TP. If both could be hit in the same bar,
  we always take the SL. This should bias AGAINST us, not for us.
  But also: we check SL using the current bar's low (for longs), which means
  SL is checked at the worst possible intra-bar price. This is conservative.

BUG HYPOTHESIS 4: OVERLAPPING TRADES
  3 ZZ configs × 48 pairs × 3 TFs = many trades that overlap in time.
  Each trade is simulated independently. In reality you can't enter all of them.
  This doesn't cause a PnL bias (each trade is evaluated correctly), but the
  TOTAL R figure is misleading (you'd need capital for all simultaneous trades).

BUG HYPOTHESIS 5: SPREAD/SLIPPAGE
  We enter at close[entry_bar] and exit at SL/TP exact price.
  No spread, no slippage. For H1 this matters less; for M15 it matters more.

BUG HYPOTHESIS 6: RANDOM BASELINE FAIRNESS
  Random baseline uses a_amp sampled from actual ABC trades. This means the
  TP/SL distances match. But direction is random, and entry point is random.
  Is this fair? The ABC entry at a confirmed zigzag pivot is NOT a random bar.
  The pivot is at a local extreme — this means the subsequent bar is likely
  to move in the opposite direction of B (= A's direction).
  This is the WHOLE POINT of the strategy, not a bug.
  BUT: does the confirmation delay actually capture this properly?
"""

import os
import csv
import numpy as np
from collections import defaultdict
from multiprocessing import Pool

DATA_DIR = "/home/ubuntu/DataBase/base_kline"
ABC_FILE = "/home/ubuntu/stage2_abc/abc_all_triples.csv"


class OnlineZigZag:
    """Copy from abc_collector.py"""
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
                    return (pv[0], pv[1], pv[2], idx)  # (pivot_bar, price, dir, confirm_bar)
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


def simulate_trade_simple(entry_bar, direction, tp_dist, sl_dist,
                          highs, lows, closes, max_hold=500):
    """Minimal trade sim — no dynamic adjustment, pure fixed SL/TP."""
    if entry_bar >= len(closes) - 1:
        return None
    entry_price = closes[entry_bar]
    if tp_dist <= 0 or sl_dist <= 0:
        return None

    end_bar = min(entry_bar + max_hold, len(closes) - 1)

    for bar in range(entry_bar + 1, end_bar + 1):
        h = highs[bar]
        l = lows[bar]
        if direction == 1:
            if l <= entry_price - sl_dist:
                return -1.0  # SL hit
            if h >= entry_price + tp_dist:
                return tp_dist / sl_dist  # TP hit in R
        else:
            if h >= entry_price + sl_dist:
                return -1.0
            if l <= entry_price - tp_dist:
                return tp_dist / sl_dist

    # Timeout: PnL at close
    if direction == 1:
        pnl = closes[end_bar] - entry_price
    else:
        pnl = entry_price - closes[end_bar]
    return pnl / sl_dist


def test_bug1_direction_free():
    """
    BUG 1: Is trading in A's direction actually better than anti-A?
    Test with SIMPLE fixed SL/TP (no dynamic adjustment) to isolate the
    direction signal from the trade management.
    """
    print("=" * 80)
    print("  BUG TEST 1: Direction Signal — A direction vs Anti-A vs Random")
    print("  Using SIMPLE fixed SL/TP (no dynamic adjustment)")
    print("=" * 80)

    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'EURJPY',
             'GBPJPY', 'USDCHF', 'EURGBP', 'XAUUSD', 'NZDUSD']
    tfs = ['H1', 'M30']
    zz_configs = [(0.5, 5), (1.0, 5)]

    results_a = []   # A direction
    results_anti = [] # Anti-A direction
    results_rand = [] # Random direction

    rng = np.random.RandomState(42)

    for pair in pairs:
        for tf in tfs:
            data = load_ohlcv(pair, tf)
            if data is None:
                continue
            highs, lows, closes = data

            for dev, confirm in zz_configs:
                zz = OnlineZigZag(deviation_pct=dev, confirm_bars=confirm)
                events = []  # (pivot_bar, price, direction, confirm_bar)
                for i in range(len(highs)):
                    ev = zz.process_bar(i, highs[i], lows[i])
                    if ev:
                        events.append(ev)

                # Build pivots with confirm bars
                # zz.pivots has (bar, price, dir) but no confirm_bar
                # events has (pivot_bar, price, dir, confirm_bar)
                # We need ABC triples using confirm_bar for entry timing

                if len(events) < 3:
                    continue

                for i in range(len(events) - 2):
                    # A = events[i], B = events[i+1], entry after B confirmed
                    a_pv_bar, a_price, a_dir, a_confirm = events[i]
                    b_pv_bar, b_price, b_dir, b_confirm = events[i + 1]

                    # Entry at confirm_bar + 1 of B (CORRECT timing)
                    entry_bar = b_confirm + 1

                    a_amp = abs(b_price - a_price)  # A amplitude (A_start to A_end=B_start... wait)

                    # Actually: A goes from previous pivot to events[i]
                    # B goes from events[i] to events[i+1]
                    # So A_amp = price change of leg ending at events[i]
                    # B_amp = price change of leg ending at events[i+1]

                    # For simplicity, use b_amp as the leg we just saw
                    # and a_amp as the leg before that
                    if i == 0:
                        continue  # need at least 2 events for A

                    prev_bar, prev_price, prev_dir, prev_confirm = events[i - 1]
                    a_amp_val = abs(a_price - prev_price)  # A leg
                    b_amp_val = abs(b_price - a_price)     # B leg

                    if a_amp_val <= 0:
                        continue

                    # Trade direction: A was from prev→events[i]
                    # A direction = +1 if a_price > prev_price else -1
                    trade_dir = 1 if a_price > prev_price else -1

                    tp_dist = a_amp_val * 0.80
                    sl_dist = tp_dist / 2.5

                    # A direction trade
                    r_a = simulate_trade_simple(entry_bar, trade_dir,
                                               tp_dist, sl_dist,
                                               highs, lows, closes)
                    if r_a is not None:
                        results_a.append(r_a)

                    # Anti-A direction
                    r_anti = simulate_trade_simple(entry_bar, -trade_dir,
                                                  tp_dist, sl_dist,
                                                  highs, lows, closes)
                    if r_anti is not None:
                        results_anti.append(r_anti)

                    # Random direction
                    rand_dir = rng.choice([-1, 1])
                    r_rand = simulate_trade_simple(entry_bar, rand_dir,
                                                  tp_dist, sl_dist,
                                                  highs, lows, closes)
                    if r_rand is not None:
                        results_rand.append(r_rand)

    def stats(pnls, label):
        pnls = np.array(pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        pf = abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else 999
        print(f"  {label:20s}: n={len(pnls):>6,}  WR={len(wins)/len(pnls)*100:.1f}%  "
              f"avgR={np.mean(pnls):.4f}  PF={pf:.2f}")

    print(f"\n  Simple fixed SL/TP (TP=A*0.80, SL=TP/2.5), no dynamic adjustments:")
    stats(results_a, "A direction")
    stats(results_anti, "Anti-A direction")
    stats(results_rand, "Random direction")

    print(f"\n  If A direction >> Anti-A: direction signal is real")
    print(f"  If A direction ≈ Anti-A: the edge comes from TIMING, not direction")
    print(f"  If both >> Random dir at random time: the edge is in entry timing")


def test_bug2_entry_lookahead():
    """
    BUG 2: Does the collector use pivot_bar or confirm_bar for entry?
    Check: in abc_collector.py, b_end_bar is pivots[i+2][0] = the PIVOT bar,
    NOT the confirmation bar. This means entry is at pivot_bar + 1,
    but the pivot isn't confirmed until pivot_bar + confirm_bars.

    WE ARE ENTERING confirm_bars EARLY = LOOK-AHEAD BUG!
    """
    print("\n" + "=" * 80)
    print("  BUG TEST 2: ENTRY TIMING LOOK-AHEAD")
    print("=" * 80)

    # Verify by checking the collector code logic
    print("""
  In abc_collector.py:
    extract_abc_features() uses pivots from zz.pivots
    zz.pivots stores: (bar, price, direction) where bar = PIVOT BAR

    entry_bar = triple['b_end_bar'] + 1
    b_end_bar = pivots[i+2][0] = the BAR WHERE THE EXTREME OCCURRED

    But OnlineZigZag confirms a pivot at: idx - pivot_bar >= confirm_bars
    The CONFIRM bar = pivot_bar + confirm_bars
    The EARLIEST you can act = confirm_bar + 1 = pivot_bar + confirm_bars + 1

    Current code enters at: pivot_bar + 1
    Should enter at: pivot_bar + confirm_bars + 1

    DIFFERENCE: confirm_bars bars of look-ahead!

    For confirm_bars=5: entering 5 bars too early
    For confirm_bars=3: entering 3 bars too early

    THIS IS A CRITICAL BUG. The backtest enters before the signal is available.
    """)

    # Quantify the impact: what happens in those confirm_bars?
    print("  Quantifying look-ahead impact on EURUSD H1:")

    data = load_ohlcv('EURUSD', 'H1')
    if data is None:
        print("  ERROR: Could not load data")
        return

    highs, lows, closes = data
    zz = OnlineZigZag(deviation_pct=0.5, confirm_bars=5)
    events = []
    for i in range(len(highs)):
        ev = zz.process_bar(i, highs[i], lows[i])
        if ev:
            events.append(ev)

    # Compare entry at pivot_bar+1 vs confirm_bar+1
    results_early = []
    results_correct = []

    for i in range(1, len(events) - 1):
        prev_bar, prev_price, prev_dir, prev_confirm = events[i - 1]
        b_bar, b_price, b_dir, b_confirm = events[i]

        a_amp = abs(b_price - prev_price)
        if a_amp <= 0:
            continue

        trade_dir = 1 if b_price > prev_price else -1
        # Wait, b_price is the end of A/start of B... no.
        # events[i-1] = end of leg before A
        # events[i] = end of A / start of B ... actually end of B
        # Let me reconsider:
        # In zigzag: pivots alternate H, L, H, L
        # events[i-1] → events[i] = one zigzag leg
        # events[i] → events[i+1] = next leg
        # For ABC: A = events[i-1]→events[i], B = events[i]→events[i+1]
        # Trade direction = A direction

        a_dir = 1 if events[i][1] > events[i-1][1] else -1

        tp_dist = a_amp * 0.80
        sl_dist = tp_dist / 2.5

        # Early entry (bug): pivot_bar + 1
        early_bar = events[i][0] + 1
        r_early = simulate_trade_simple(early_bar, a_dir, tp_dist, sl_dist,
                                        highs, lows, closes)
        if r_early is not None:
            results_early.append(r_early)

        # Correct entry: confirm_bar + 1
        correct_bar = events[i][3] + 1  # confirm_bar + 1
        r_correct = simulate_trade_simple(correct_bar, a_dir, tp_dist, sl_dist,
                                          highs, lows, closes)
        if r_correct is not None:
            results_correct.append(r_correct)

    def stats(pnls, label):
        pnls = np.array(pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        pf = abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else 999
        print(f"  {label:30s}: n={len(pnls):>5,}  WR={len(wins)/len(pnls)*100:.1f}%  "
              f"avgR={np.mean(pnls):.4f}  PF={pf:.2f}")

    print(f"\n  EURUSD H1, dev=0.5 confirm=5, simple fixed SL/TP:")
    stats(results_early, "EARLY (pivot_bar+1) BUG")
    stats(results_correct, "CORRECT (confirm_bar+1)")

    # Also check the magnitude of the look-ahead bars
    delays = []
    for ev in events:
        delays.append(ev[3] - ev[0])  # confirm_bar - pivot_bar
    delays = np.array(delays)
    print(f"\n  Confirmation delay stats:")
    print(f"    Mean: {np.mean(delays):.1f} bars")
    print(f"    Median: {np.median(delays):.0f} bars")
    print(f"    Min: {np.min(delays)}, Max: {np.max(delays)}")


def test_bug3_bar_order():
    """
    BUG 3: Within a bar, if both SL and TP could be hit, which wins?
    Current code checks SL first → conservative (biased against us).
    But for a more accurate test, we should also check the reverse.
    """
    print("\n" + "=" * 80)
    print("  BUG TEST 3: BAR-INTERNAL ORDER (SL checked before TP)")
    print("=" * 80)
    print("""
  Current logic: check SL first, then dynamic adjustments, then TP.
  If both SL and TP could be triggered in the same bar:
    → SL wins (conservative)

  This biases AGAINST the strategy, which means our results
  are a LOWER BOUND. Not a bug in our favor.
  
  However, the dynamic SL adjustment happens DURING the bar scan:
    1. Check SL (using previous bar's SL level)
    2. Update max_favorable with current bar's high/low
    3. Dynamic adjust SL/TP
    4. Check TP

  So TP is checked against the UPDATED TP level, which could have
  been expanded due to positive deviation detected in step 2-3.
  This means within the SAME bar, we could:
    - See a new high (updating max_favorable)
    - Expand TP due to progress > 100%
    - NOT hit the new TP (because it was just expanded)
  
  This is actually CORRECT behavior — if progress exceeds target,
  we want to let it run. The expansion happens at the right time.
  
  VERDICT: No bug. SL-first is conservative (biased against us).
  """)


def test_bug4_overlap():
    """
    BUG 4: Overlapping trades inflate total R.
    """
    print("\n" + "=" * 80)
    print("  BUG TEST 4: OVERLAPPING TRADES")
    print("=" * 80)

    # Load a sample to check overlap
    float_cols = ['entry_bar', 'exit_bar', 'hold_bars', 'pnl_r', 'zz_dev', 'zz_confirm']
    trades_by_pair = defaultdict(list)

    with open(ABC_FILE, 'r') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if row['pair'] == 'EURUSD' and row['tf'] == 'H1':
                trades_by_pair[(row['pair'], row['tf'], row['zz_dev'])].append({
                    'entry': int(float(row['entry_bar'])),
                    'exit': int(float(row['exit_bar'])),
                    'pnl_r': float(row['pnl_r']),
                })

    for key in sorted(trades_by_pair.keys()):
        trades = sorted(trades_by_pair[key], key=lambda t: t['entry'])
        overlaps = 0
        for i in range(len(trades) - 1):
            if trades[i + 1]['entry'] < trades[i]['exit']:
                overlaps += 1

        print(f"  {key}: {len(trades)} trades, {overlaps} overlapping ({overlaps/max(1,len(trades))*100:.0f}%)")

    print(f"""
  With 3 ZZ configs per pair-TF, many trades overlap in time.
  This means the "total R" figure is NOT achievable with a single account.
  However, the PER-TRADE statistics (WR, avg_R, PF) are still valid.
  
  For a realistic system, you would:
  1. Pick ONE ZZ config per pair-TF (probably dev=1.0 based on results)
  2. Or implement a priority system when signals conflict
  """)


def test_bug5_spread():
    """
    BUG 5: Impact of realistic spread/slippage.
    """
    print("\n" + "=" * 80)
    print("  BUG TEST 5: SPREAD IMPACT ESTIMATE")
    print("=" * 80)

    # Load ABC data and estimate spread impact
    float_cols = ['a_amp', 'sl_distance', 'tp_distance', 'pnl_r']

    # Typical spreads in pips (approximate)
    spreads = {
        'EURUSD': 1.0, 'GBPUSD': 1.5, 'USDJPY': 1.2, 'AUDUSD': 1.5,
        'EURJPY': 2.0, 'GBPJPY': 3.0, 'USDCHF': 1.5, 'EURGBP': 2.0,
        'XAUUSD': 30.0, 'XAGUSD': 3.0,
    }
    default_spread = 3.0  # pips for exotics

    # pip values differ by pair
    # For XXX/YYY where YYY is the quote currency:
    # 1 pip = 0.0001 for most pairs, 0.01 for JPY pairs, 0.1 for XAUUSD

    sample_trades = []
    with open(ABC_FILE, 'r') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if count > 500000:
                break
            pair = row['pair']
            entry_price = float(row.get('entry_price', 0))
            a_amp = float(row['a_amp'])
            sl_dist = float(row['sl_distance']) if 'sl_distance' in row else a_amp * 0.32

            if a_amp <= 0 or entry_price <= 0 or sl_dist <= 0:
                continue

            # Get spread in price units
            spread_pips = spreads.get(pair, default_spread)
            if 'JPY' in pair:
                spread_price = spread_pips * 0.01
            elif pair.startswith('XAU'):
                spread_price = spread_pips * 0.1
            elif pair.startswith('XAG'):
                spread_price = spread_pips * 0.01
            else:
                spread_price = spread_pips * 0.0001

            # Spread as fraction of SL distance
            spread_r = spread_price / sl_dist  # spread cost in R units

            sample_trades.append({
                'pair': pair,
                'spread_r': spread_r,
                'pnl_r': float(row.get('pnl_r', 0)),
            })
            count += 1

    spreads_r = [t['spread_r'] for t in sample_trades]
    pnls = [t['pnl_r'] for t in sample_trades]
    adjusted = [t['pnl_r'] - t['spread_r'] for t in sample_trades]

    print(f"  Sample: {len(sample_trades):,} trades")
    print(f"  Spread as R: mean={np.mean(spreads_r):.4f}, "
          f"median={np.median(spreads_r):.4f}, "
          f"p95={np.percentile(spreads_r, 95):.4f}")
    print(f"  Original avg_R: {np.mean(pnls):.4f}")
    print(f"  After spread:   {np.mean(adjusted):.4f}")
    print(f"  Spread impact:  {np.mean(spreads_r):.4f} R per trade")

    wins_orig = len([p for p in pnls if p > 0]) / len(pnls) * 100
    wins_adj = len([p for p in adjusted if p > 0]) / len(adjusted) * 100
    print(f"  WR original: {wins_orig:.1f}%")
    print(f"  WR after spread: {wins_adj:.1f}%")


def main():
    test_bug1_direction_free()
    test_bug2_entry_lookahead()
    test_bug3_bar_order()
    test_bug4_overlap()
    test_bug5_spread()

    print("\n" + "=" * 80)
    print("  BUG AUDIT COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
