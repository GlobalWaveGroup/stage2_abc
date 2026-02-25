"""
Pyramid Entry/Exit Strategy
============================
Core philosophy:
- Not precise, but ROBUST
- Pyramid sizing: score -> position size mapping
- Don't chase WR, chase E[PnL] = P(win) * avg_win_size - P(loss) * avg_loss_size
- Higher score = higher conviction = larger position
- Multi-level resonance = layered entries

Key design:
1. ENTRY: score determines position size (not binary in/out)
   - Base position for score >= threshold
   - Scale up for higher scores (pyramid)
   - Multiple entries allowed at same bar if multi-level signals fire

2. EXIT: dynamic, based on C' development vs predicted C
   - Initial SL based on B structure
   - As C' develops favorably, trail stop using B/A ratio logic
   - If C' exceeds predicted C, this C becomes new A, and next B becomes opportunity to add
   - If C' fails (becomes new reverse B), reduce/exit

3. Comparison: pyramid vs flat sizing
"""
import sys, os, csv, numpy as np, pickle
from collections import defaultdict

DATA_DIR = "/home/ubuntu/DataBase/base_kline"
ZIG_DIR = "/home/ubuntu/database2/build_zig_all/cl128"

FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618, 2.000, 2.618]


def fib_distance(ratio):
    return min(abs(ratio - f) for f in FIB_LEVELS)


def load_ohlcv(pair, tf):
    fpath = os.path.join(DATA_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fpath):
        return None, None
    dates, highs, lows, closes = [], [], [], []
    with open(fpath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 6:
                continue
            dates.append(row[0])
            highs.append(float(row[3]))
            lows.append(float(row[4]))
            closes.append(float(row[5]))
    return dates, (np.array(highs), np.array(lows), np.array(closes))


def load_zigzag_topo(pair, tf):
    fpath = os.path.join(ZIG_DIR, tf, f"{pair}_{tf}.npz")
    if not os.path.exists(fpath):
        return None, None
    d = np.load(fpath, allow_pickle=True)
    feats = d["features"]
    edges = pickle.loads(d["edges_bytes"].tobytes())
    return feats, edges


def extract_signals(pair, tf):
    """Extract all entry signals with scores and trade parameters."""
    dates, price_data = load_ohlcv(pair, tf)
    if dates is None:
        return []
    highs, lows, closes = price_data
    n_bars = len(closes)

    feats, edges = load_zigzag_topo(pair, tf)
    if feats is None:
        return []

    win_bar_to_node = defaultdict(dict)
    for i in range(len(feats)):
        win = int(feats[i, 1])
        rel_idx = int(feats[i, 2])
        ml = int(feats[i, 5])
        if rel_idx not in win_bar_to_node[win] or ml > feats[win_bar_to_node[win][rel_idx], 5]:
            win_bar_to_node[win][rel_idx] = i

    DELAY = 5
    signals = []

    for i in range(len(feats)):
        bar_idx = int(feats[i, 0])
        win_id = int(feats[i, 1])
        rel_idx = int(feats[i, 2])
        max_level = int(feats[i, 5])

        if bar_idx < 50 or bar_idx >= n_bars - 100:
            continue

        in_edges_i, out_edges_i = edges[i]
        if not out_edges_i:
            continue

        in_by_lv = defaultdict(list)
        out_by_lv = defaultdict(list)
        for e in in_edges_i:
            if e["duration"] > 0:
                in_by_lv[e["level"]].append(e)
        for e in out_edges_i:
            if e["duration"] > 0:
                out_by_lv[e["level"]].append(e)

        level_scores = []
        for lv in range(max_level + 1):
            b_list = [e for e in in_by_lv.get(lv, []) if e["end_idx"] == rel_idx]
            c_list = [e for e in out_by_lv.get(lv, []) if e["start_idx"] == rel_idx]
            if not c_list or not b_list:
                continue

            c = c_list[0]
            b = b_list[0]

            b_src_rel = b["start_idx"]
            src_idx = win_bar_to_node.get(win_id, {}).get(b_src_rel, None)
            if src_idx is None:
                continue
            src_in, _ = edges[src_idx]
            a_cands = [e for e in src_in
                       if e["level"] == lv and e["end_idx"] == b_src_rel and e["duration"] > 0]
            if not a_cands:
                continue

            a = a_cands[0]
            a_amp = abs(a["amplitude_pct"])
            b_amp = abs(b["amplitude_pct"])
            a_mod = a["modulus"]
            b_mod = b["modulus"]

            if a_amp < 1e-6 or a_mod < 1e-6:
                continue

            ba_amp_ratio = b_amp / a_amp
            ba_mod_ratio = b_mod / a_mod

            score = 0.0
            score += 0.5 + lv * 0.5
            fib_d = fib_distance(ba_amp_ratio)
            score += max(0, 2.0 * (1.0 - fib_d / 0.10))
            fib_d_mod = fib_distance(ba_mod_ratio)
            score += max(0, 1.0 * (1.0 - fib_d_mod / 0.10))
            if ba_amp_ratio < 0.382:
                score += 1.5
            elif ba_amp_ratio < 0.618:
                score += 0.8
            elif ba_amp_ratio < 1.0:
                score += 0.0
            else:
                score -= 0.5
            c_continues = 1 if (a["direction"] == c["direction"]) else 0
            score += 1.0 if c_continues else -0.5

            level_scores.append({
                "lv": lv,
                "score": score,
                "c_amp": c["amplitude_pct"],
                "c_dir": c["direction"],
                "c_mod": c["modulus"],
                "c_dur": c["duration"],
                "a_amp": a_amp,
                "a_mod": a_mod,
                "b_amp": b_amp,
                "b_mod": b_mod,
            })

        if not level_scores:
            continue

        n_lv = len(level_scores)
        total_score = sum(ls["score"] for ls in level_scores)

        dirs = [ls["c_dir"] for ls in level_scores]
        consensus = abs(sum(dirs)) / len(dirs)
        direction_sum = sum(ls["c_dir"] * ls["score"] for ls in level_scores)
        direction = 1 if direction_sum > 0 else -1

        if consensus >= 0.9:
            total_score += n_lv * 0.5
        elif consensus < 0.5:
            total_score -= n_lv * 0.3

        # Target amplitude: weighted average of C amplitudes
        w_amp = sum(ls["score"] * abs(ls["c_amp"]) for ls in level_scores if ls["score"] > 0)
        w_total = sum(ls["score"] for ls in level_scores if ls["score"] > 0)
        if w_total <= 0:
            continue
        v_amp = w_amp / w_total

        # A amplitude for SL reference
        a_amp_avg = np.mean([ls["a_amp"] for ls in level_scores])

        if v_amp < 0.01:
            continue

        eb = bar_idx + DELAY
        if eb >= n_bars - 100:
            continue

        year = int(dates[min(bar_idx, len(dates) - 1)][:4])

        signals.append({
            "bar_idx": bar_idx,
            "entry_bar": eb,
            "direction": direction,
            "score": total_score,
            "n_lv": n_lv,
            "consensus": consensus,
            "v_amp": v_amp,
            "a_amp_avg": a_amp_avg,
            "year": year,
        })

    print(f"  {pair}_{tf}: {len(signals):,} signals")
    return signals, highs, lows, closes, dates


def simulate_flat(signals, highs, lows, closes):
    """Flat sizing: every trade gets size=1."""
    results = []
    for sig in signals:
        eb = sig["entry_bar"]
        direction = sig["direction"]
        v_amp = sig["v_amp"]

        decay = np.exp(-0.08 * 5)
        tp_pct = v_amp * 0.75 * decay
        sl_pct = tp_pct * 0.4
        if tp_pct < 0.01:
            continue

        ep = closes[eb]
        tp_d = ep * tp_pct / 100.0
        sl_d = ep * sl_pct / 100.0
        if tp_d <= 0 or sl_d <= 0:
            continue

        pnl_r = run_trade(eb, direction, tp_d, sl_d, highs, lows, closes)
        results.append({
            "score": sig["score"],
            "pnl_r": pnl_r,
            "pnl_sized": pnl_r * 1.0,  # flat size = 1
            "size": 1.0,
            "year": sig["year"],
        })
    return results


def simulate_pyramid(signals, highs, lows, closes):
    """Pyramid sizing: score -> position size.
    
    Size mapping (continuous, not discrete):
    - score < 5: size = 0.5 (low conviction, small position)
    - score 5-10: size = 1.0 (base)
    - score 10-15: size = 1.5
    - score 15-20: size = 2.0
    - score 20-30: size = 2.5
    - score > 30: size = 3.0 (max)
    
    This is a simple monotonic mapping. The key insight:
    higher score -> more capital deployed -> E[PnL] amplified on good signals.
    """
    results = []
    for sig in signals:
        eb = sig["entry_bar"]
        direction = sig["direction"]
        v_amp = sig["v_amp"]
        score = sig["score"]

        decay = np.exp(-0.08 * 5)
        tp_pct = v_amp * 0.75 * decay
        sl_pct = tp_pct * 0.4
        if tp_pct < 0.01:
            continue

        ep = closes[eb]
        tp_d = ep * tp_pct / 100.0
        sl_d = ep * sl_pct / 100.0
        if tp_d <= 0 or sl_d <= 0:
            continue

        # Pyramid sizing
        if score < 5:
            size = 0.5
        elif score < 10:
            size = 1.0
        elif score < 15:
            size = 1.5
        elif score < 20:
            size = 2.0
        elif score < 30:
            size = 2.5
        else:
            size = 3.0

        pnl_r = run_trade(eb, direction, tp_d, sl_d, highs, lows, closes)
        results.append({
            "score": score,
            "pnl_r": pnl_r,
            "pnl_sized": pnl_r * size,
            "size": size,
            "year": sig["year"],
        })
    return results


def simulate_pyramid_v2(signals, highs, lows, closes):
    """Pyramid V2: also adjust TP/SL based on score.
    
    Higher score -> more aggressive TP (let winners run further)
    Higher score -> slightly wider SL (give more room for high-conviction trades)
    """
    results = []
    for sig in signals:
        eb = sig["entry_bar"]
        direction = sig["direction"]
        v_amp = sig["v_amp"]
        score = sig["score"]

        decay = np.exp(-0.08 * 5)

        # Score-adjusted TP: higher score -> aim for more of predicted C
        # Base: 0.75 * v_amp. High score: up to 1.0 * v_amp
        tp_mult = 0.6 + min(score / 30.0, 1.0) * 0.4  # 0.6 to 1.0
        tp_pct = v_amp * tp_mult * decay

        # Score-adjusted SL: higher score -> slightly wider SL (more conviction)
        # Base: 0.4 * tp. High score: 0.5 * tp (wider room)
        sl_mult = 0.35 + min(score / 40.0, 1.0) * 0.15  # 0.35 to 0.50
        sl_pct = tp_pct * sl_mult

        if tp_pct < 0.01:
            continue

        ep = closes[eb]
        tp_d = ep * tp_pct / 100.0
        sl_d = ep * sl_pct / 100.0
        if tp_d <= 0 or sl_d <= 0:
            continue

        # Same pyramid sizing
        if score < 5:
            size = 0.5
        elif score < 10:
            size = 1.0
        elif score < 15:
            size = 1.5
        elif score < 20:
            size = 2.0
        elif score < 30:
            size = 2.5
        else:
            size = 3.0

        pnl_r = run_trade(eb, direction, tp_d, sl_d, highs, lows, closes)
        results.append({
            "score": score,
            "pnl_r": pnl_r,
            "pnl_sized": pnl_r * size,
            "size": size,
            "year": sig["year"],
        })
    return results


def run_trade(eb, direction, tp_d, sl_d, highs, lows, closes):
    """Execute trade with dynamic trailing stop."""
    n_bars = len(closes)
    ep = closes[eb]
    if direction == 1:
        tp_price = ep + tp_d
        sl_price = ep - sl_d
    else:
        tp_price = ep - tp_d
        sl_price = ep + sl_d

    max_hold = 200
    mf = 0.0
    be = False
    end_bar = min(eb + max_hold, n_bars - 1)

    for bar in range(eb + 1, end_bar + 1):
        h = highs[bar]
        l = lows[bar]
        fav = (h - ep) if direction == 1 else (ep - l)
        if fav > mf:
            mf = fav
        prog = mf / tp_d if tp_d > 0 else 0
        elapsed = bar - eb

        # SL check
        if direction == 1:
            if l <= sl_price:
                return (sl_price - ep) / sl_d
        else:
            if h >= sl_price:
                return (ep - sl_price) / sl_d

        # Dynamic trailing
        if prog >= 0.25 and not be:
            sl_price = ep
            be = True
        if prog >= 0.50:
            if direction == 1:
                sl_price = max(sl_price, ep + mf * 0.40)
            else:
                sl_price = min(sl_price, ep - mf * 0.40)

        # TP check
        if direction == 1:
            if h >= tp_price:
                return (tp_price - ep) / sl_d
        else:
            if l <= tp_price:
                return (ep - tp_price) / sl_d

    pnl = (closes[end_bar] - ep) if direction == 1 else (ep - closes[end_bar])
    return pnl / sl_d


def analyze_results(name, results):
    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(f"{'=' * 80}")

    n = len(results)
    if n == 0:
        print("  No results")
        return

    # Raw PnL (per unit)
    pnl_raw = [r["pnl_r"] for r in results]
    w_raw = [x for x in pnl_raw if x > 0]
    l_raw = [x for x in pnl_raw if x <= 0]
    pf_raw = abs(sum(w_raw) / sum(l_raw)) if l_raw and sum(l_raw) != 0 else 999

    # Sized PnL
    pnl_sized = [r["pnl_sized"] for r in results]
    w_sized = [x for x in pnl_sized if x > 0]
    l_sized = [x for x in pnl_sized if x <= 0]
    pf_sized = abs(sum(w_sized) / sum(l_sized)) if l_sized and sum(l_sized) != 0 else 999

    avg_size = np.mean([r["size"] for r in results])

    print(f"  Trades: {n:,}")
    print(f"  Raw:   WR={len(w_raw)/n*100:.1f}%  avgR={np.mean(pnl_raw):.4f}  PF={pf_raw:.2f}  sumR={sum(pnl_raw):.1f}")
    print(f"  Sized: WR={len(w_sized)/n*100:.1f}%  avgR={np.mean(pnl_sized):.4f}  PF={pf_sized:.2f}  sumR={sum(pnl_sized):.1f}")
    print(f"  Avg size: {avg_size:.2f}")
    print(f"  Sized/Raw sumR ratio: {sum(pnl_sized)/sum(pnl_raw):.2f}x" if sum(pnl_raw) != 0 else "")

    # IS vs OOS
    is_data = [r for r in results if 0 < r["year"] <= 2018]
    oos_data = [r for r in results if r["year"] > 2018]

    for label, data in [("IS (<=2018)", is_data), ("OOS (>2018)", oos_data)]:
        if not data:
            continue
        p = [r["pnl_sized"] for r in data]
        w = [x for x in p if x > 0]
        l = [x for x in p if x <= 0]
        pf = abs(sum(w) / sum(l)) if l and sum(l) != 0 else 999
        print(f"  {label}: n={len(data):,}  WR={len(w)/len(data)*100:.1f}%  "
              f"avgR={np.mean(p):.4f}  PF={pf:.2f}  sumR={sum(p):.1f}")

    # By score tier
    print(f"\n  By score tier (sized PnL):")
    tiers = [(0, 5, "<5"), (5, 10, "5-10"), (10, 15, "10-15"),
             (15, 20, "15-20"), (20, 30, "20-30"), (30, 100, "30+")]
    for lo, hi, label in tiers:
        sub = [r for r in results if lo <= r["score"] < hi]
        if not sub:
            continue
        p = [r["pnl_sized"] for r in sub]
        w = [x for x in p if x > 0]
        l = [x for x in p if x <= 0]
        pf = abs(sum(w) / sum(l)) if l and sum(l) != 0 else 999
        avg_sz = np.mean([r["size"] for r in sub])
        print(f"    score {label:>5}: n={len(sub):>8,}  size={avg_sz:.1f}  "
              f"WR={len(w)/len(sub)*100:.1f}%  avgR={np.mean(p):.4f}  PF={pf:.2f}  sumR={sum(p):.1f}")


def main():
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

    all_signals = []
    all_highs = None
    all_lows = None
    all_closes = None

    # Process each pair separately (since OHLCV is per-pair)
    all_flat = []
    all_pyramid = []
    all_pyramid_v2 = []

    for pair in pairs:
        sigs, highs, lows, closes, dates = extract_signals(pair, "H1")
        if not sigs:
            continue

        flat = simulate_flat(sigs, highs, lows, closes)
        pyramid = simulate_pyramid(sigs, highs, lows, closes)
        pyramid_v2 = simulate_pyramid_v2(sigs, highs, lows, closes)

        all_flat.extend(flat)
        all_pyramid.extend(pyramid)
        all_pyramid_v2.extend(pyramid_v2)

    # Compare strategies
    analyze_results("STRATEGY A: FLAT SIZING (size=1 for all)", all_flat)
    analyze_results("STRATEGY B: PYRAMID SIZING (score -> size)", all_pyramid)
    analyze_results("STRATEGY C: PYRAMID + ADAPTIVE TP/SL", all_pyramid_v2)

    # Expected value analysis
    print(f"\n{'=' * 80}")
    print(f"  EXPECTED VALUE COMPARISON")
    print(f"{'=' * 80}")

    for name, data in [("Flat", all_flat), ("Pyramid", all_pyramid), ("Pyramid+Adaptive", all_pyramid_v2)]:
        if not data:
            continue
        sized_pnl = [r["pnl_sized"] for r in data]
        total_size = sum(r["size"] for r in data)
        total_pnl = sum(sized_pnl)
        # E[PnL] per unit of capital deployed
        ev_per_unit = total_pnl / total_size if total_size > 0 else 0
        print(f"  {name:>20}: total_sized_PnL={total_pnl:>10.1f}  "
              f"total_capital={total_size:>10.1f}  E[PnL]/unit={ev_per_unit:.4f}")


if __name__ == "__main__":
    main()
