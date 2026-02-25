"""
AB Comparability & Turning Point Quality Analysis
===================================================
Two core concepts:

1. COMPARABILITY C(A,B): Not all adjacent legs form a meaningful pair.
   - amp_symmetry: min(A,B)/max(A,B)
   - dur_symmetry: min(durA,durB)/max(durA,durB)
   - slope_symmetry: min(slopeA,slopeB)/max(slopeA,slopeB)
   Combined into comparability score [0,1].

2. TURNING POINT QUALITY Q(P):
   - arrival_type: adjustment vs trend arrival
   - completion_degree: ABC structural symmetry
   - exhaustion: B leg deceleration

3. EXTENDED RAG: Add these to KDTree features.

Optimized: trade simulation inside worker process, parallel across 48 pairs.
"""
import sys, os, csv, numpy as np, pickle
from collections import defaultdict
from scipy.spatial import KDTree
from multiprocessing import Pool

DATA_DIR = "/home/ubuntu/DataBase/base_kline"
ZIG_DIR = "/home/ubuntu/database2/build_zig_all/cl128"
FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618, 2.000, 2.618]
DELAY = 5


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
            if len(row) < 6: continue
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


def compute_comparability(a_amp, b_amp, a_dur, b_dur, a_slope, b_slope):
    if max(a_amp, b_amp) > 1e-8:
        amp_sym = min(a_amp, b_amp) / max(a_amp, b_amp)
    else:
        amp_sym = 0.0
    if max(a_dur, b_dur) > 0:
        dur_sym = min(a_dur, b_dur) / max(a_dur, b_dur)
    else:
        dur_sym = 0.0
    if max(a_slope, b_slope) > 1e-10:
        slope_sym = min(a_slope, b_slope) / max(a_slope, b_slope)
    else:
        slope_sym = 0.0
    comparability = (amp_sym * dur_sym * slope_sym) ** (1.0 / 3.0)
    return comparability, amp_sym, dur_sym, slope_sym


def compute_completion_degree(a_amp, b_amp, c_amp, a_dur, b_dur, c_dur):
    if a_amp < 1e-8:
        return 0.0, 0.0, 0.0, 0.0
    ca_ratio = c_amp / a_amp
    ba_ratio = b_amp / a_amp
    fib_dist = fib_distance(ca_ratio)
    fib_score = max(0, 1.0 - fib_dist / 0.15)
    if max(a_dur, c_dur) > 0:
        time_sym = min(a_dur, c_dur) / max(a_dur, c_dur)
    else:
        time_sym = 0.0
    if ba_ratio < 0.15:
        b_quality = ba_ratio / 0.15
    elif ba_ratio <= 1.0:
        b_quality = 1.0
    elif ba_ratio <= 2.0:
        b_quality = max(0, 1.0 - (ba_ratio - 1.0) / 1.0)
    else:
        b_quality = 0.0
    completion = (fib_score * 0.4 + time_sym * 0.3 + b_quality * 0.3)
    return completion, fib_score, time_sym, b_quality


def _run_trade(eb, direction, tp_pct, sl_pct, ep, highs, lows, closes):
    n_bars = len(closes)
    if tp_pct < 0.005 or sl_pct < 0.005: return None
    tp_d = ep * tp_pct / 100.0
    sl_d = ep * sl_pct / 100.0
    if tp_d <= 0 or sl_d <= 0: return None
    if direction == 1:
        tp_price = ep + tp_d; sl_price = ep - sl_d
    else:
        tp_price = ep - tp_d; sl_price = ep + sl_d
    max_hold = 200; mf = 0.0; be = False
    end_bar = min(eb + max_hold, n_bars - 1)
    for bar in range(eb + 1, end_bar + 1):
        h = highs[bar]; l = lows[bar]
        fav = (h - ep) if direction == 1 else (ep - l)
        if fav > mf: mf = fav
        prog = mf / tp_d if tp_d > 0 else 0
        if direction == 1:
            if l <= sl_price: return (sl_price - ep) / sl_d
        else:
            if h >= sl_price: return (ep - sl_price) / sl_d
        if prog >= 0.25 and not be: sl_price = ep; be = True
        if prog >= 0.50:
            if direction == 1: sl_price = max(sl_price, ep + mf * 0.40)
            else: sl_price = min(sl_price, ep - mf * 0.40)
        if direction == 1:
            if h >= tp_price: return (tp_price - ep) / sl_d
        else:
            if l <= tp_price: return (ep - tp_price) / sl_d
    pnl = (closes[end_bar] - ep) if direction == 1 else (ep - closes[end_bar])
    return pnl / sl_d


def process_pair(args):
    """Extract ABC chains + compute comparability/quality + run trades. All in one worker."""
    pair, tf = args
    dates, price_data = load_ohlcv(pair, tf)
    if dates is None: return []
    highs, lows, closes = price_data
    n_bars = len(closes)
    feats, edges = load_zigzag_topo(pair, tf)
    if feats is None: return []

    win_bar_to_node = defaultdict(dict)
    for i in range(len(feats)):
        win = int(feats[i, 1])
        rel_idx = int(feats[i, 2])
        ml = int(feats[i, 5])
        if rel_idx not in win_bar_to_node[win] or ml > feats[win_bar_to_node[win][rel_idx], 5]:
            win_bar_to_node[win][rel_idx] = i

    decay = np.exp(-0.08 * DELAY)
    results = []

    for i in range(len(feats)):
        bar_idx = int(feats[i, 0])
        win_id = int(feats[i, 1])
        rel_idx = int(feats[i, 2])
        max_level = int(feats[i, 5])
        if bar_idx < 50 or bar_idx >= n_bars - 300: continue

        in_edges_i, out_edges_i = edges[i]
        if not out_edges_i: continue

        in_by_lv = defaultdict(list)
        out_by_lv = defaultdict(list)
        for e in in_edges_i:
            if e["duration"] > 0: in_by_lv[e["level"]].append(e)
        for e in out_edges_i:
            if e["duration"] > 0: out_by_lv[e["level"]].append(e)

        level_data = []
        for lv in range(max_level + 1):
            b_list = [e for e in in_by_lv.get(lv, []) if e["end_idx"] == rel_idx]
            c_list = [e for e in out_by_lv.get(lv, []) if e["start_idx"] == rel_idx]
            if not c_list or not b_list: continue
            c_edge = c_list[0]; b_edge = b_list[0]

            b_src_rel = b_edge["start_idx"]
            src_idx = win_bar_to_node.get(win_id, {}).get(b_src_rel, None)
            if src_idx is None: continue
            src_in, _ = edges[src_idx]
            a_cands = [e for e in src_in if e["level"] == lv and e["end_idx"] == b_src_rel and e["duration"] > 0]
            if not a_cands: continue
            a_edge = a_cands[0]

            a_amp = abs(a_edge["amplitude_pct"])
            b_amp = abs(b_edge["amplitude_pct"])
            c_amp = abs(c_edge["amplitude_pct"])
            a_dur = max(a_edge["duration"], 1)
            b_dur = max(b_edge["duration"], 1)
            c_dur = max(c_edge["duration"], 1)
            a_mod = a_edge["modulus"]
            b_mod = b_edge["modulus"]
            if a_amp < 1e-6 or a_mod < 1e-6: continue

            a_slope = a_amp / a_dur
            b_slope = b_amp / b_dur

            comp, amp_sym, dur_sym, slope_sym = compute_comparability(
                a_amp, b_amp, a_dur, b_dur, a_slope, b_slope)
            completion, fib_score, time_sym, b_quality = compute_completion_degree(
                a_amp, b_amp, c_amp, a_dur, b_dur, c_dur)

            # Arrival type (simplified — no OHLCV access needed for basic version)
            b_complexity = min(lv / 7.0, 1.0)
            dur_ratio_raw = b_dur / a_dur
            if dur_ratio_raw > 3.0: b_dur_score = 1.0
            elif dur_ratio_raw > 1.5: b_dur_score = 0.7
            elif dur_ratio_raw > 0.5: b_dur_score = 0.3
            else: b_dur_score = 0.0

            # Exhaustion from OHLCV
            b_start_bar = bar_idx - b_dur
            b_mid_bar = bar_idx - b_dur // 2
            exhaustion = 0.5
            if b_start_bar >= 0 and b_mid_bar >= 0 and bar_idx < n_bars and b_dur >= 4:
                p_start = closes[max(0, b_start_bar)]
                p_mid = closes[max(0, min(b_mid_bar, n_bars-1))]
                p_end = closes[min(bar_idx, n_bars-1)]
                half1 = abs(p_mid - p_start)
                half2 = abs(p_end - p_mid)
                total = half1 + half2
                if total > 1e-8:
                    exhaustion = half1 / total

            arrival = b_complexity * 0.3 + b_dur_score * 0.4 + (1.0 - exhaustion) * 0.3

            c_continues = 1 if (a_edge["direction"] == c_edge["direction"]) else 0
            amp_r = b_amp / a_amp

            # Scoring
            s = 0.5 + lv * 0.5
            s += max(0, 2.0 * (1.0 - fib_distance(amp_r) / 0.10))
            s += max(0, 1.0 * (1.0 - fib_distance(b_mod / a_mod) / 0.10))
            if amp_r < 0.382: s += 1.5
            elif amp_r < 0.618: s += 0.8
            elif amp_r >= 1.0: s -= 0.5
            s += 1.0 if c_continues else -0.5

            level_data.append({
                "lv": lv, "score": s,
                "a_amp": a_amp, "b_amp": b_amp, "c_amp": c_amp,
                "a_dur": a_dur, "b_dur": b_dur, "c_dur": c_dur,
                "a_mod": a_mod, "b_mod": b_mod,
                "a_slope": a_slope, "b_slope": b_slope,
                "comp": comp, "amp_sym": amp_sym, "dur_sym": dur_sym, "slope_sym": slope_sym,
                "completion": completion, "fib_score": fib_score,
                "time_sym": time_sym, "b_quality": b_quality,
                "arrival": arrival, "b_exhaustion": exhaustion,
                "c_continues": c_continues, "ca_amp_r": c_amp / a_amp,
                "c_dir": c_edge["direction"], "amp_r": amp_r,
                "ab_vec": np.array([
                    np.log1p(b_amp / a_amp),
                    np.log1p(b_dur / a_dur),
                    np.log1p(b_slope / a_slope) if a_slope > 1e-10 else 0,
                    np.log1p(b_mod / a_mod) if a_mod > 1e-6 else 0,
                ]),
            })

        if not level_data: continue
        n_lv = len(level_data)

        total_score = sum(d["score"] for d in level_data)
        dirs = [d["c_dir"] for d in level_data]
        consensus = abs(sum(dirs)) / len(dirs)
        if consensus >= 0.9: total_score += n_lv * 0.5
        elif consensus < 0.5: total_score -= n_lv * 0.3
        if total_score < 10 or total_score >= 30: continue

        direction_sum = sum(d["c_dir"] * d["score"] for d in level_data)
        direction = 1 if direction_sum > 0 else -1

        ws = [d["score"] for d in level_data if d["score"] > 0]
        w_total = sum(ws)
        if w_total <= 0: continue

        def wmean(key):
            return sum(d["score"] * d[key] for d in level_data if d["score"] > 0) / w_total

        ab_vec_agg = np.zeros(4)
        for d in level_data:
            if d["score"] > 0:
                ab_vec_agg += d["score"] * d["ab_vec"]
        ab_vec_agg /= w_total

        year = int(dates[min(bar_idx, len(dates)-1)][:4])

        # ---- Run trade immediately ----
        eb = bar_idx + DELAY
        if eb >= n_bars - 200: continue
        ep = closes[eb]
        tp_pct = wmean("c_amp") * 1.0 * decay
        sl_pct = wmean("b_amp") * 0.8 * decay
        pnl_r = _run_trade(eb, direction, tp_pct, sl_pct, ep, highs, lows, closes)
        if pnl_r is None: continue

        results.append({
            "pnl_r": round(pnl_r, 4),
            "pair": pair, "year": year, "bar_idx": bar_idx,
            "n_lv": n_lv, "max_level": max_level,
            "total_score": round(total_score, 2),
            "direction": direction,
            "ab_vec": ab_vec_agg,
            "amp_r": round(wmean("amp_r"), 4),
            "comp": round(wmean("comp"), 4),
            "amp_sym": round(wmean("amp_sym"), 4),
            "dur_sym": round(wmean("dur_sym"), 4),
            "slope_sym": round(wmean("slope_sym"), 4),
            "completion": round(wmean("completion"), 4),
            "fib_score": round(wmean("fib_score"), 4),
            "time_sym": round(wmean("time_sym"), 4),
            "b_quality": round(wmean("b_quality"), 4),
            "arrival": round(wmean("arrival"), 4),
            "b_exhaustion": round(wmean("b_exhaustion"), 4),
            "c_continues": 1 if sum(d["c_continues"] * d["score"] for d in level_data if d["score"] > 0) / w_total > 0.5 else 0,
            "ca_amp_r": round(wmean("ca_amp_r"), 4),
            "c_amp": round(abs(wmean("c_amp")), 4),
            "a_amp": round(wmean("a_amp"), 4),
            "b_amp": round(wmean("b_amp"), 4),
        })

    print(f"  {pair}_{tf}: {len(results):,} trades (from {len(feats):,} nodes)")
    return results


def pf_wr(pnl_list):
    if not pnl_list: return 0, 0, 0, 0, 0
    w = [x for x in pnl_list if x > 0]
    lo = [x for x in pnl_list if x <= 0]
    pf = abs(sum(w) / sum(lo)) if lo and sum(lo) != 0 else 999
    return len(pnl_list), len(w)/len(pnl_list)*100, np.mean(pnl_list), pf, sum(pnl_list)


def main():
    print("=" * 80)
    print("  COMPARABILITY & TURNING POINT QUALITY ANALYSIS")
    print("=" * 80)

    tf = "H1"
    tf_path = os.path.join(ZIG_DIR, tf)
    tasks = []
    for fname in sorted(os.listdir(tf_path)):
        if not fname.endswith(".npz"): continue
        pair = fname.replace(f"_{tf}.npz", "")
        if os.path.exists(os.path.join(DATA_DIR, f"{pair}_{tf}.csv")):
            tasks.append((pair, tf))
    
    print(f"  Pairs: {len(tasks)}")
    
    all_trades = []
    with Pool(min(48, len(tasks))) as pool:
        for batch in pool.imap_unordered(process_pair, tasks):
            all_trades.extend(batch)
    
    print(f"\n  Total trades: {len(all_trades):,}")
    
    is_trades = [t for t in all_trades if 0 < t["year"] <= 2018]
    oos_trades = [t for t in all_trades if t["year"] > 2018]
    print(f"  IS: {len(is_trades):,}  OOS: {len(oos_trades):,}")
    
    n, wr, ar, pf, sr = pf_wr([t["pnl_r"] for t in all_trades])
    print(f"  ALL: n={n:,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}  sumR={sr:,.0f}")
    _, _, _, pf_i, _ = pf_wr([t["pnl_r"] for t in is_trades])
    _, _, _, pf_o, _ = pf_wr([t["pnl_r"] for t in oos_trades])
    print(f"  IS PF={pf_i:.2f}  OOS PF={pf_o:.2f}")

    qs = max(1, len(all_trades) // 5)

    # =============================================
    # PHASE 2: Comparability -> PF
    # =============================================
    print(f"\n{'='*80}")
    print("  PHASE 2: COMPARABILITY -> ALPHA")
    print(f"{'='*80}")

    print(f"\n  2a. COMPARABILITY QUINTILES (geometric mean of 3 symmetries)")
    tr_sorted = sorted(all_trades, key=lambda x: x["comp"])
    for qi in range(5):
        chunk = tr_sorted[qi*qs:(qi+1)*qs] if qi < 4 else tr_sorted[qi*qs:]
        pnl = [t["pnl_r"] for t in chunk]
        n, wr, ar, pf, sr = pf_wr(pnl)
        lo_v = chunk[0]["comp"]; hi_v = chunk[-1]["comp"]
        is_p = [t["pnl_r"] for t in chunk if 0 < t["year"] <= 2018]
        oos_p = [t["pnl_r"] for t in chunk if t["year"] > 2018]
        _, _, _, pf_i, _ = pf_wr(is_p)
        _, _, _, pf_o, _ = pf_wr(oos_p)
        print(f"    Q{qi+1} comp=[{lo_v:.3f},{hi_v:.3f}]: n={n:>7,}  WR={wr:.1f}%  "
              f"avgR={ar:.4f}  PF={pf:.2f}  IS={pf_i:.2f}  OOS={pf_o:.2f}")

    for dim_name in ["amp_sym", "dur_sym", "slope_sym"]:
        print(f"\n  2b. {dim_name.upper()} QUINTILES")
        tr_sorted = sorted(all_trades, key=lambda x: x[dim_name])
        for qi in range(5):
            chunk = tr_sorted[qi*qs:(qi+1)*qs] if qi < 4 else tr_sorted[qi*qs:]
            pnl = [t["pnl_r"] for t in chunk]
            n, wr, ar, pf, sr = pf_wr(pnl)
            lo_v = chunk[0][dim_name]; hi_v = chunk[-1][dim_name]
            is_p = [t["pnl_r"] for t in chunk if 0 < t["year"] <= 2018]
            oos_p = [t["pnl_r"] for t in chunk if t["year"] > 2018]
            _, _, _, pf_i, _ = pf_wr(is_p)
            _, _, _, pf_o, _ = pf_wr(oos_p)
            print(f"    Q{qi+1} [{lo_v:.3f},{hi_v:.3f}]: n={n:>7,}  WR={wr:.1f}%  "
                  f"avgR={ar:.4f}  PF={pf:.2f}  IS={pf_i:.2f}  OOS={pf_o:.2f}")

    # =============================================
    # PHASE 3: Turning Point Quality -> PF
    # =============================================
    print(f"\n{'='*80}")
    print("  PHASE 3: TURNING POINT QUALITY -> ALPHA")
    print(f"{'='*80}")

    for dim_name, desc in [("completion", "COMPLETION DEGREE"),
                           ("arrival", "ARRIVAL TYPE (0=adj, 1=trend)"),
                           ("b_exhaustion", "B EXHAUSTION (>0.5=decelerate)"),
                           ("fib_score", "FIB SCORE"),
                           ("time_sym", "TIME SYMMETRY"),
                           ("b_quality", "B RETRACEMENT QUALITY")]:
        print(f"\n  {desc} QUINTILES")
        tr_sorted = sorted(all_trades, key=lambda x: x[dim_name])
        for qi in range(5):
            chunk = tr_sorted[qi*qs:(qi+1)*qs] if qi < 4 else tr_sorted[qi*qs:]
            pnl = [t["pnl_r"] for t in chunk]
            n, wr, ar, pf, sr = pf_wr(pnl)
            lo_v = chunk[0][dim_name]; hi_v = chunk[-1][dim_name]
            is_p = [t["pnl_r"] for t in chunk if 0 < t["year"] <= 2018]
            oos_p = [t["pnl_r"] for t in chunk if t["year"] > 2018]
            _, _, _, pf_i, _ = pf_wr(is_p)
            _, _, _, pf_o, _ = pf_wr(oos_p)
            print(f"    Q{qi+1} [{lo_v:.3f},{hi_v:.3f}]: n={n:>7,}  WR={wr:.1f}%  "
                  f"avgR={ar:.4f}  PF={pf:.2f}  IS={pf_i:.2f}  OOS={pf_o:.2f}")

    # =============================================
    # PHASE 4: 2D Cross
    # =============================================
    print(f"\n{'='*80}")
    print("  PHASE 4: 2D CROSS: COMPARABILITY x QUALITY")
    print(f"{'='*80}")
    
    comp_med = np.median([t["comp"] for t in all_trades])
    compl_med = np.median([t["completion"] for t in all_trades])
    arr_med = np.median([t["arrival"] for t in all_trades])
    exh_med = np.median([t["b_exhaustion"] for t in all_trades])
    
    print(f"\n  Comp × Completion (medians: comp={comp_med:.3f}, compl={compl_med:.3f})")
    for cl, cf in [("LowComp", lambda t: t["comp"] < comp_med), ("HighComp", lambda t: t["comp"] >= comp_med)]:
        for ql, qf in [("LowCompl", lambda t: t["completion"] < compl_med), ("HighCompl", lambda t: t["completion"] >= compl_med)]:
            sub = [t["pnl_r"] for t in all_trades if cf(t) and qf(t)]
            n, wr, ar, pf, sr = pf_wr(sub)
            is_s = [t["pnl_r"] for t in all_trades if cf(t) and qf(t) and 0 < t["year"] <= 2018]
            oos_s = [t["pnl_r"] for t in all_trades if cf(t) and qf(t) and t["year"] > 2018]
            _, _, _, pi, _ = pf_wr(is_s)
            _, _, _, po, _ = pf_wr(oos_s)
            if n < 100: continue
            print(f"    {cl:>8} × {ql:>9}: n={n:>7,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}  IS={pi:.2f}  OOS={po:.2f}")

    print(f"\n  Comp × Arrival (medians: comp={comp_med:.3f}, arr={arr_med:.3f})")
    for cl, cf in [("LowComp", lambda t: t["comp"] < comp_med), ("HighComp", lambda t: t["comp"] >= comp_med)]:
        for al, af in [("Adjust", lambda t: t["arrival"] < arr_med), ("Trend", lambda t: t["arrival"] >= arr_med)]:
            sub = [t["pnl_r"] for t in all_trades if cf(t) and af(t)]
            n, wr, ar, pf, sr = pf_wr(sub)
            is_s = [t["pnl_r"] for t in all_trades if cf(t) and af(t) and 0 < t["year"] <= 2018]
            oos_s = [t["pnl_r"] for t in all_trades if cf(t) and af(t) and t["year"] > 2018]
            _, _, _, pi, _ = pf_wr(is_s)
            _, _, _, po, _ = pf_wr(oos_s)
            if n < 100: continue
            print(f"    {cl:>8} × {al:>9}: n={n:>7,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}  IS={pi:.2f}  OOS={po:.2f}")

    print(f"\n  Comp × Exhaustion (medians: comp={comp_med:.3f}, exh={exh_med:.3f})")
    for cl, cf in [("LowComp", lambda t: t["comp"] < comp_med), ("HighComp", lambda t: t["comp"] >= comp_med)]:
        for el, ef in [("Accel", lambda t: t["b_exhaustion"] < exh_med), ("Exhaust", lambda t: t["b_exhaustion"] >= exh_med)]:
            sub = [t["pnl_r"] for t in all_trades if cf(t) and ef(t)]
            n, wr, ar, pf, sr = pf_wr(sub)
            if n < 100: continue
            print(f"    {cl:>8} × {el:>9}: n={n:>7,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # =============================================
    # PHASE 5: Extended RAG (5 major pairs subset)
    # =============================================
    print(f"\n{'='*80}")
    print("  PHASE 5: EXTENDED RAG COMPARISON")
    print(f"{'='*80}")

    # Use 5 major pairs for RAG (memory/speed)
    major = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"}
    rag_trades = [t for t in all_trades if t["pair"] in major]
    rag_is = [t for t in rag_trades if 0 < t["year"] <= 2018]
    rag_oos = [t for t in rag_trades if t["year"] > 2018]
    
    print(f"  RAG subset (5 majors): IS={len(rag_is):,} OOS={len(rag_oos):,}")

    if len(rag_is) > 100 and len(rag_oos) > 100:
        def make_vec_4d(t): return t["ab_vec"]
        def make_vec_7d_comp(t): return np.concatenate([t["ab_vec"], [t["comp"], t["amp_sym"], t["dur_sym"]]])
        def make_vec_7d_qual(t): return np.concatenate([t["ab_vec"], [t["completion"], t["arrival"], t["b_exhaustion"]]])
        def make_vec_10d(t): return np.concatenate([t["ab_vec"], [t["comp"], t["amp_sym"], t["dur_sym"]], [t["completion"], t["arrival"], t["b_exhaustion"]]])

        K = 50
        for name, make_vec in [("Baseline 4D", make_vec_4d),
                                ("+Comp 7D", make_vec_7d_comp),
                                ("+Quality 7D", make_vec_7d_qual),
                                ("+Both 10D", make_vec_10d)]:
            is_vecs = np.array([make_vec(t) for t in rag_is])
            oos_vecs = np.array([make_vec(t) for t in rag_oos])
            
            tree = KDTree(is_vecs)
            distances, indices = tree.query(oos_vecs, k=min(K, len(rag_is)-1))
            
            pred_ca = []; actual_ca = []; pred_dir = []; actual_dir = []
            for qi in range(len(rag_oos)):
                ni = indices[qi]
                pred_ca.append(np.median([rag_is[n]["ca_amp_r"] for n in ni]))
                actual_ca.append(rag_oos[qi]["ca_amp_r"])
                pred_dir.append(np.mean([rag_is[n]["c_continues"] for n in ni]))
                actual_dir.append(rag_oos[qi]["c_continues"])
            
            pred_ca = np.array(pred_ca)
            actual_ca = np.array(actual_ca)
            pred_dir = np.array(pred_dir)
            actual_dir = np.array(actual_dir)
            
            corr = np.corrcoef(pred_ca, actual_ca)[0, 1]
            
            # Direction accuracy
            hi60 = pred_dir >= 0.6; hi70 = pred_dir >= 0.7
            acc60 = actual_dir[hi60].mean() if hi60.sum() > 0 else 0
            acc70 = actual_dir[hi70].mean() if hi70.sum() > 0 else 0
            
            # Trading: filter by RAG direction
            oos_rag_pnl = [rag_oos[qi]["pnl_r"] for qi in range(len(rag_oos)) if pred_dir[qi] >= 0.6]
            oos_all_pnl = [t["pnl_r"] for t in rag_oos]
            n_f, wr_f, ar_f, pf_f, _ = pf_wr(oos_rag_pnl)
            n_a, wr_a, ar_a, pf_a, _ = pf_wr(oos_all_pnl)
            
            print(f"\n  {name}:")
            print(f"    Median dist: {np.median(distances[:,0]):.4f}  C/A corr: {corr:.4f}")
            print(f"    Dir P>=0.6: n={hi60.sum():,} acc={acc60:.3f}  P>=0.7: n={hi70.sum():,} acc={acc70:.3f}")
            print(f"    OOS all:     n={n_a:,} PF={pf_a:.2f} avgR={ar_a:.4f}")
            print(f"    OOS RAG>=0.6: n={n_f:,} PF={pf_f:.2f} avgR={ar_f:.4f}")

    # =============================================
    # PHASE 6: Trade cost filter
    # =============================================
    print(f"\n{'='*80}")
    print("  PHASE 6: TRADE COST FILTER")
    print(f"{'='*80}")
    for min_camp in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
        sub = [t["pnl_r"] for t in all_trades if t["c_amp"] >= min_camp]
        n, wr, ar, pf, sr = pf_wr(sub)
        is_s = [t["pnl_r"] for t in all_trades if t["c_amp"] >= min_camp and 0 < t["year"] <= 2018]
        oos_s = [t["pnl_r"] for t in all_trades if t["c_amp"] >= min_camp and t["year"] > 2018]
        _, _, _, pi, _ = pf_wr(is_s)
        _, _, _, po, _ = pf_wr(oos_s)
        print(f"    min_c_amp>={min_camp:.2f}%: n={n:>7,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}  IS={pi:.2f}  OOS={po:.2f}")

    # =============================================
    # PHASE 7: Distribution stats
    # =============================================
    print(f"\n{'='*80}")
    print("  PHASE 7: DISTRIBUTIONS")
    print(f"{'='*80}")
    for dim in ["comp", "amp_sym", "dur_sym", "slope_sym",
                "completion", "fib_score", "time_sym", "b_quality",
                "arrival", "b_exhaustion"]:
        vals = [t[dim] for t in all_trades]
        print(f"  {dim:>15}: mean={np.mean(vals):.3f} std={np.std(vals):.3f} "
              f"p25={np.percentile(vals,25):.3f} p50={np.median(vals):.3f} p75={np.percentile(vals,75):.3f}")

    print(f"\n{'='*80}")
    print("  COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
