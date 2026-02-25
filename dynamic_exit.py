"""
Dynamic Exit + Pure Signal Pool (score 10-30)
=============================================
Exit logic based on C' development vs predicted C:

1. STRUCTURAL EXIT: C' forms a reverse micro-structure
   - Track MFE (max favorable excursion) as C' develops
   - If C' retraces from MFE by a fib ratio of the move so far,
     this retracement IS a new B'. If B'/C' ratio hits fib level -> exit
   - This is "the reason to hold disappears"

2. TIME-BASED DECAY: if C' hasn't reached predicted target within
   expected time, reduce conviction (tighten stops)

3. PROGRESSIVE TARGETS: instead of one TP, use multiple targets
   - T1: 0.382 * predicted_C -> take 30% off
   - T2: 0.618 * predicted_C -> take 30% off  
   - T3: 1.000 * predicted_C -> take 25% off
   - T4: 1.618 * predicted_C -> let remainder run with tight trail

Compare with:
- Static exit (current)
- Dynamic structural exit
- Progressive partial exits
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
    """Extract signals with score 10-30 only."""
    dates, price_data = load_ohlcv(pair, tf)
    if dates is None:
        return [], None, None, None, None
    highs, lows, closes = price_data
    n_bars = len(closes)

    feats, edges = load_zigzag_topo(pair, tf)
    if feats is None:
        return [], None, None, None, None

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

        if bar_idx < 50 or bar_idx >= n_bars - 200:
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
                "lv": lv, "score": score,
                "c_amp": c["amplitude_pct"], "c_dir": c["direction"],
                "c_dur": c["duration"], "c_mod": c["modulus"],
                "a_amp": a_amp, "b_amp": b_amp,
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

        # FILTER: only score 10-30
        if total_score < 10 or total_score >= 30:
            continue

        w_amp = sum(ls["score"] * abs(ls["c_amp"]) for ls in level_scores if ls["score"] > 0)
        w_total = sum(ls["score"] for ls in level_scores if ls["score"] > 0)
        if w_total <= 0:
            continue
        v_amp = w_amp / w_total

        # Predicted C duration (weighted)
        w_dur = sum(ls["score"] * ls["c_dur"] for ls in level_scores if ls["score"] > 0)
        v_dur = w_dur / w_total

        if v_amp < 0.01:
            continue

        eb = bar_idx + DELAY
        if eb >= n_bars - 200:
            continue

        year = int(dates[min(bar_idx, len(dates) - 1)][:4])

        # Pyramid sizing
        if total_score < 15:
            size = 1.5
        elif total_score < 20:
            size = 2.0
        else:
            size = 2.5

        signals.append({
            "bar_idx": bar_idx,
            "entry_bar": eb,
            "direction": direction,
            "score": total_score,
            "n_lv": n_lv,
            "v_amp": v_amp,
            "v_dur": v_dur,
            "size": size,
            "year": year,
        })

    print(f"  {pair}_{tf}: {len(signals):,} signals (score 10-30)")
    return signals, highs, lows, closes, dates


# ============================================================
# EXIT STRATEGY A: Static (current baseline)
# ============================================================
def exit_static(sig, highs, lows, closes):
    eb = sig["entry_bar"]
    direction = sig["direction"]
    v_amp = sig["v_amp"]
    n_bars = len(closes)

    decay = np.exp(-0.08 * 5)
    tp_mult = 0.6 + min(sig["score"] / 30.0, 1.0) * 0.4
    tp_pct = v_amp * tp_mult * decay
    sl_mult = 0.35 + min(sig["score"] / 40.0, 1.0) * 0.15
    sl_pct = tp_pct * sl_mult
    if tp_pct < 0.01:
        return None

    ep = closes[eb]
    tp_d = ep * tp_pct / 100.0
    sl_d = ep * sl_pct / 100.0
    if tp_d <= 0 or sl_d <= 0:
        return None

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

        if direction == 1:
            if l <= sl_price:
                return (sl_price - ep) / sl_d
        else:
            if h >= sl_price:
                return (ep - sl_price) / sl_d

        if prog >= 0.25 and not be:
            sl_price = ep
            be = True
        if prog >= 0.50:
            if direction == 1:
                sl_price = max(sl_price, ep + mf * 0.40)
            else:
                sl_price = min(sl_price, ep - mf * 0.40)

        if direction == 1:
            if h >= tp_price:
                return (tp_price - ep) / sl_d
        else:
            if l <= tp_price:
                return (ep - tp_price) / sl_d

    pnl = (closes[end_bar] - ep) if direction == 1 else (ep - closes[end_bar])
    return pnl / sl_d


# ============================================================
# EXIT STRATEGY B: Structural (C' reverse detection)
# ============================================================
def exit_structural(sig, highs, lows, closes):
    """
    Track C' development. If C' retraces from MFE by a significant
    fib ratio, the structure is breaking -> exit.
    
    Key idea: we don't just trail mechanically.
    We ask: "has C' formed a reverse B' that kills the thesis?"
    
    - Retrace > 0.618 of C' so far -> structure broken, exit
    - Retrace > 0.382 after time decay -> reducing conviction
    - No retrace, hitting targets -> let it run
    """
    eb = sig["entry_bar"]
    direction = sig["direction"]
    v_amp = sig["v_amp"]
    v_dur = sig["v_dur"]
    n_bars = len(closes)

    decay = np.exp(-0.08 * 5)
    tp_pct = v_amp * 0.85 * decay  # slightly more aggressive TP
    sl_pct = tp_pct * 0.45  # slightly wider SL for structural exit
    if tp_pct < 0.01:
        return None

    ep = closes[eb]
    tp_d = ep * tp_pct / 100.0
    sl_d = ep * sl_pct / 100.0
    if tp_d <= 0 or sl_d <= 0:
        return None

    if direction == 1:
        sl_price = ep - sl_d
    else:
        sl_price = ep + sl_d

    max_hold = max(int(v_dur * 3), 200)
    mf = 0.0  # max favorable excursion
    mf_bar = eb  # bar where MFE occurred
    end_bar = min(eb + max_hold, n_bars - 1)

    for bar in range(eb + 1, end_bar + 1):
        h = highs[bar]
        l = lows[bar]
        elapsed = bar - eb
        time_ratio = elapsed / max_hold if max_hold > 0 else 1

        # Update MFE
        fav = (h - ep) if direction == 1 else (ep - l)
        if fav > mf:
            mf = fav
            mf_bar = bar

        # Current favorable excursion THIS bar
        if direction == 1:
            current_fav = l - ep  # negative when below entry
        else:
            current_fav = ep - h  # negative when above entry

        # Retrace from MFE: how much of the favorable move has been given back
        if mf > sl_d * 0.5:  # only measure retrace when MFE is meaningful
            retrace_from_peak = mf - max(0, current_fav if direction == 1 else current_fav)
            # For long: peak was ep+mf, current adverse is l, retrace = (ep+mf) - l
            # But we want retrace of the MOVE (mf), not absolute
            if direction == 1:
                retrace_from_peak = (ep + mf) - l
            else:
                retrace_from_peak = h - (ep - mf)
            retrace_ratio = retrace_from_peak / mf if mf > 0 else 0
        else:
            retrace_ratio = 0

        # Hard SL
        if direction == 1:
            if l <= sl_price:
                return (sl_price - ep) / sl_d
        else:
            if h >= sl_price:
                return (ep - sl_price) / sl_d

        prog = mf / tp_d if tp_d > 0 else 0

        # STRUCTURAL EXIT RULES:

        # Rule 1: Only after C' has made SIGNIFICANT progress (prog > 0.5)
        # AND retraces > 0.618 of that move -> structure may be breaking
        # This means C' formed, then a reverse B' ate 61.8% of it
        if prog >= 0.5 and retrace_ratio > 0.618:
            # Exit at approximate retrace level
            kept = mf * (1 - retrace_ratio)
            if direction == 1:
                exit_pnl = max(0, kept)  # at least breakeven if possible
            else:
                exit_pnl = max(0, kept)
            return exit_pnl / sl_d

        # Rule 2: After very strong progress (prog > 1.0), tighter retrace
        # threshold: 0.382 retrace of move = structure weakening
        if prog >= 1.0 and retrace_ratio > 0.382:
            kept = mf * (1 - retrace_ratio)
            return max(0, kept) / sl_d

        # Rule 3: Time decay - if > 60% of time used, < 30% of target
        if time_ratio > 0.6 and prog < 0.3:
            if direction == 1:
                sl_price = max(sl_price, ep)
            else:
                sl_price = min(sl_price, ep)

        # Rule 4: Progressive trailing (fib-based)
        if prog >= 0.25:
            # Move to breakeven
            if direction == 1:
                sl_price = max(sl_price, ep)
            else:
                sl_price = min(sl_price, ep)
        if prog >= 0.5:
            # Trail at 0.382 retrace (keep 61.8%)
            trail = mf * 0.618
            if direction == 1:
                sl_price = max(sl_price, ep + trail)
            else:
                sl_price = min(sl_price, ep - trail)
        if prog >= 1.0:
            # Trail at 0.236 retrace (keep 76.4%)
            trail = mf * 0.764
            if direction == 1:
                sl_price = max(sl_price, ep + trail)
            else:
                sl_price = min(sl_price, ep - trail)

        # Cap at 2.5x predicted
        if prog >= 2.5:
            return mf / sl_d

    pnl = (closes[end_bar] - ep) if direction == 1 else (ep - closes[end_bar])
    return pnl / sl_d


# ============================================================
# EXIT STRATEGY C: Progressive partial exits
# ============================================================
def exit_progressive(sig, highs, lows, closes):
    """
    Multiple targets, partial exits:
    T1: 0.382 * predicted -> close 30%
    T2: 0.618 * predicted -> close 30%
    T3: 1.000 * predicted -> close 25%
    T4: remainder trails with tight stop (0.236 retrace)
    
    Returns weighted average PnL across partials.
    """
    eb = sig["entry_bar"]
    direction = sig["direction"]
    v_amp = sig["v_amp"]
    n_bars = len(closes)

    decay = np.exp(-0.08 * 5)
    base_tp_pct = v_amp * 0.85 * decay
    sl_pct = base_tp_pct * 0.45
    if base_tp_pct < 0.01:
        return None

    ep = closes[eb]
    base_tp_d = ep * base_tp_pct / 100.0
    sl_d = ep * sl_pct / 100.0
    if base_tp_d <= 0 or sl_d <= 0:
        return None

    # Targets (as fraction of base_tp_d)
    targets = [
        (0.382, 0.30),  # T1: 38.2% of predicted C, close 30%
        (0.618, 0.30),  # T2: 61.8%, close 30%
        (1.000, 0.25),  # T3: 100%, close 25%
    ]
    remaining_weight = 0.15  # T4: 15% rides the trend

    if direction == 1:
        sl_price = ep - sl_d
    else:
        sl_price = ep + sl_d

    max_hold = 200
    mf = 0.0
    end_bar = min(eb + max_hold, n_bars - 1)

    total_pnl = 0.0
    total_weight = 0.0
    targets_hit = 0

    for bar in range(eb + 1, end_bar + 1):
        h = highs[bar]
        l = lows[bar]
        fav = (h - ep) if direction == 1 else (ep - l)
        if fav > mf:
            mf = fav
        prog = mf / base_tp_d if base_tp_d > 0 else 0

        # SL check (applies to all remaining position)
        if direction == 1:
            if l <= sl_price:
                # Close all remaining at SL
                remaining_w = sum(t[1] for t in targets[targets_hit:]) + remaining_weight
                pnl_here = (sl_price - ep) / sl_d
                total_pnl += pnl_here * remaining_w
                total_weight += remaining_w
                break
        else:
            if h >= sl_price:
                remaining_w = sum(t[1] for t in targets[targets_hit:]) + remaining_weight
                pnl_here = (ep - sl_price) / sl_d
                total_pnl += pnl_here * remaining_w
                total_weight += remaining_w
                break

        # Check targets
        while targets_hit < len(targets):
            t_level, t_weight = targets[targets_hit]
            t_price = base_tp_d * t_level
            if direction == 1:
                if h >= ep + t_price:
                    pnl_here = t_price / sl_d
                    total_pnl += pnl_here * t_weight
                    total_weight += t_weight
                    targets_hit += 1
                else:
                    break
            else:
                if l <= ep - t_price:
                    pnl_here = t_price / sl_d
                    total_pnl += pnl_here * t_weight
                    total_weight += t_weight
                    targets_hit += 1
                else:
                    break

        # Trail for remaining position after all targets hit
        if targets_hit >= len(targets):
            # Trail using 0.236 retrace of MFE
            if mf > 0:
                trail_level = mf * (1 - 0.236)
                if direction == 1:
                    sl_price = max(sl_price, ep + trail_level)
                else:
                    sl_price = min(sl_price, ep - trail_level)

        # Progressive trailing for partial position
        if targets_hit >= 1:
            # Move to breakeven after T1
            if direction == 1:
                sl_price = max(sl_price, ep)
            else:
                sl_price = min(sl_price, ep)
        if targets_hit >= 2 and mf > 0:
            # After T2, trail at 0.382 retrace
            trail = mf * (1 - 0.382)
            if direction == 1:
                sl_price = max(sl_price, ep + trail)
            else:
                sl_price = min(sl_price, ep - trail)

    else:
        # Timeout: close all remaining at market
        remaining_w = sum(t[1] for t in targets[targets_hit:]) + remaining_weight
        pnl = (closes[end_bar] - ep) if direction == 1 else (ep - closes[end_bar])
        total_pnl += (pnl / sl_d) * remaining_w
        total_weight += remaining_w

    if total_weight <= 0:
        return 0
    return total_pnl / total_weight  # weighted average PnL per unit


def analyze(name, results):
    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(f"{'=' * 80}")
    if not results:
        print("  No results")
        return

    n = len(results)
    pnl = [r["pnl_sized"] for r in results]
    w = [x for x in pnl if x > 0]
    l = [x for x in pnl if x <= 0]
    pf = abs(sum(w) / sum(l)) if l and sum(l) != 0 else 999
    total_capital = sum(r["size"] for r in results)

    print(f"  n={n:,}  WR={len(w)/n*100:.1f}%  avgR={np.mean(pnl):.4f}  "
          f"PF={pf:.2f}  sumR={sum(pnl):,.1f}")
    print(f"  total_capital={total_capital:,.0f}  E[PnL]/unit={sum(pnl)/total_capital:.4f}")

    # IS vs OOS
    for label, fn in [("IS <=2018", lambda r: 0 < r["year"] <= 2018),
                      ("OOS >2018", lambda r: r["year"] > 2018)]:
        sub = [r for r in results if fn(r)]
        if not sub:
            continue
        p = [r["pnl_sized"] for r in sub]
        w2 = [x for x in p if x > 0]
        l2 = [x for x in p if x <= 0]
        pf2 = abs(sum(w2) / sum(l2)) if l2 and sum(l2) != 0 else 999
        cap = sum(r["size"] for r in sub)
        print(f"  {label}: n={len(sub):>7,}  WR={len(w2)/len(sub)*100:.1f}%  "
              f"avgR={np.mean(p):.4f}  PF={pf2:.2f}  sumR={sum(p):>10,.1f}  E/u={sum(p)/cap:.4f}")

    # By score tier
    print(f"\n  By score tier:")
    for lo, hi, label in [(10, 15, "10-15"), (15, 20, "15-20"), (20, 30, "20-30")]:
        sub = [r for r in results if lo <= r["score"] < hi]
        if not sub:
            continue
        p = [r["pnl_sized"] for r in sub]
        w2 = [x for x in p if x > 0]
        l2 = [x for x in p if x <= 0]
        pf2 = abs(sum(w2) / sum(l2)) if l2 and sum(l2) != 0 else 999
        print(f"    {label}: n={len(sub):>7,}  WR={len(w2)/len(sub)*100:.1f}%  "
              f"avgR={np.mean(p):.4f}  PF={pf2:.2f}  sumR={sum(p):>10,.1f}")

    # Distribution of outcomes
    raw_pnl = [r["pnl_r"] for r in results]
    print(f"\n  PnL distribution (raw):")
    pcts = np.percentile(raw_pnl, [5, 25, 50, 75, 95])
    print(f"    p5={pcts[0]:.2f}  p25={pcts[1]:.2f}  median={pcts[2]:.2f}  "
          f"p75={pcts[3]:.2f}  p95={pcts[4]:.2f}")


def main():
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

    all_static = []
    all_structural = []
    all_progressive = []

    for pair in pairs:
        sigs, highs, lows, closes, dates = extract_signals(pair, "H1")
        if not sigs:
            continue

        for sig in sigs:
            size = sig["size"]
            score = sig["score"]
            year = sig["year"]

            r1 = exit_static(sig, highs, lows, closes)
            if r1 is not None:
                all_static.append({"pnl_r": r1, "pnl_sized": r1 * size,
                                   "size": size, "score": score, "year": year})

            r2 = exit_structural(sig, highs, lows, closes)
            if r2 is not None:
                all_structural.append({"pnl_r": r2, "pnl_sized": r2 * size,
                                       "size": size, "score": score, "year": year})

            r3 = exit_progressive(sig, highs, lows, closes)
            if r3 is not None:
                all_progressive.append({"pnl_r": r3, "pnl_sized": r3 * size,
                                        "size": size, "score": score, "year": year})

    analyze("EXIT A: STATIC (current baseline)", all_static)
    analyze("EXIT B: STRUCTURAL (C' reverse detection)", all_structural)
    analyze("EXIT C: PROGRESSIVE PARTIAL EXITS", all_progressive)

    # Summary comparison
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'=' * 80}")
    for name, data in [("Static", all_static), ("Structural", all_structural),
                       ("Progressive", all_progressive)]:
        if not data:
            continue
        pnl = [r["pnl_sized"] for r in data]
        cap = sum(r["size"] for r in data)
        w = [x for x in pnl if x > 0]
        l = [x for x in pnl if x <= 0]
        pf = abs(sum(w) / sum(l)) if l and sum(l) != 0 else 999
        print(f"  {name:>12}: sumR={sum(pnl):>10,.1f}  PF={pf:.2f}  "
              f"WR={len(w)/len(pnl)*100:.1f}%  E/u={sum(pnl)/cap:.4f}")


if __name__ == "__main__":
    main()
