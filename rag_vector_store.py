"""
ABC(DEF...) Vector Store RAG System
====================================
Build a vector store of historical price structure sequences.
When a new AB appears, retrieve nearest neighbors and use their
ACTUAL C distribution for TP/SL calibration.

Extend to ABCD, ABCDE sequences for deeper context.

Vector representation per leg:
- amp (normalized by price level)
- dur (bars)
- slope (amp/dur)  
- mod (vector modulus)
All as RATIOS between consecutive legs (scale-invariant).

For AB pair: (amp_r, dur_r, slope_r, mod_r) = 4D vector
For ABC triple: (ab_amp_r, ab_dur_r, ab_slope_r, ab_mod_r,
                  bc_amp_r, bc_dur_r, bc_slope_r, bc_mod_r) = 8D vector
For ABCD: 12D, etc.

Retrieval: L2 distance in ratio space (or cosine).
Use scipy.spatial.KDTree for fast nearest-neighbor.

Output: for each query AB, return statistics of retrieved Cs:
- C amplitude distribution (p25, p50, p75)
- C duration distribution
- C direction probability
- Suggested TP/SL based on actual outcomes
"""
import sys, os, csv, numpy as np, pickle
from collections import defaultdict
from scipy.spatial import KDTree

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


def extract_abc_chains(pair, tf):
    """
    Extract all ABC chains with full feature vectors.
    Each chain includes the AB vector (for query) and C outcome (for retrieval).
    Also extract ABCD chains where possible.
    """
    dates, price_data = load_ohlcv(pair, tf)
    if dates is None: return [], None, None, None, None
    highs, lows, closes = price_data
    n_bars = len(closes)
    feats, edges = load_zigzag_topo(pair, tf)
    if feats is None: return [], None, None, None, None

    win_bar_to_node = defaultdict(dict)
    for i in range(len(feats)):
        win = int(feats[i, 1])
        rel_idx = int(feats[i, 2])
        ml = int(feats[i, 5])
        if rel_idx not in win_bar_to_node[win] or ml > feats[win_bar_to_node[win][rel_idx], 5]:
            win_bar_to_node[win][rel_idx] = i

    chains = []
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

        for lv in range(max_level + 1):
            b_list = [e for e in in_by_lv.get(lv, []) if e["end_idx"] == rel_idx]
            c_list = [e for e in out_by_lv.get(lv, []) if e["start_idx"] == rel_idx]
            if not c_list or not b_list: continue
            c = c_list[0]; b = b_list[0]

            # Trace A
            b_src_rel = b["start_idx"]
            src_idx = win_bar_to_node.get(win_id, {}).get(b_src_rel, None)
            if src_idx is None: continue
            src_in, src_out = edges[src_idx]
            a_cands = [e for e in src_in if e["level"] == lv and e["end_idx"] == b_src_rel and e["duration"] > 0]
            if not a_cands: continue
            a = a_cands[0]

            # Extract features
            a_amp = abs(a["amplitude_pct"]); b_amp = abs(b["amplitude_pct"]); c_amp = abs(c["amplitude_pct"])
            a_dur = max(a["duration"], 1); b_dur = max(b["duration"], 1); c_dur = max(c["duration"], 1)
            a_mod = a["modulus"]; b_mod = b["modulus"]; c_mod = c["modulus"]
            if a_amp < 1e-6 or a_mod < 1e-6: continue

            a_slope = a_amp / a_dur
            b_slope = b_amp / b_dur
            c_slope = c_amp / c_dur

            # AB ratios (query vector)
            ab_vec = np.array([
                np.log1p(b_amp / a_amp),      # log scale for ratios
                np.log1p(b_dur / a_dur),
                np.log1p(b_slope / a_slope) if a_slope > 1e-10 else 0,
                np.log1p(b_mod / a_mod) if a_mod > 1e-6 else 0,
            ])

            # C outcome (what we want to predict)
            c_continues = 1 if (a["direction"] == c["direction"]) else 0
            ca_amp_r = c_amp / a_amp
            cb_amp_r = c_amp / b_amp if b_amp > 1e-6 else 1.0
            ca_slope_r = c_slope / a_slope if a_slope > 1e-10 else 1.0

            # Try to trace D (next leg after C)
            c_end_rel = c["end_idx"]
            c_end_node = win_bar_to_node.get(win_id, {}).get(c_end_rel, None)
            has_d = False
            d_info = {}
            if c_end_node is not None:
                _, c_end_out = edges[c_end_node]
                d_cands = [e for e in c_end_out if e["level"] == lv and e["start_idx"] == c_end_rel and e["duration"] > 0]
                if d_cands:
                    d = d_cands[0]
                    d_amp = abs(d["amplitude_pct"])
                    d_dur = max(d["duration"], 1)
                    d_slope = d_amp / d_dur
                    has_d = True
                    d_info = {
                        "d_amp": d_amp, "d_dur": d_dur,
                        "cd_amp_r": d_amp / c_amp if c_amp > 1e-6 else 1.0,
                        "cd_slope_r": d_slope / c_slope if c_slope > 1e-10 else 1.0,
                        "d_continues_c": 1 if (c["direction"] == d["direction"]) else 0,
                    }

            chain = {
                "bar_idx": bar_idx, "lv": lv, "win_id": win_id,
                "ab_vec": ab_vec,
                "a_amp": a_amp, "b_amp": b_amp, "c_amp": c_amp,
                "a_dur": a_dur, "b_dur": b_dur, "c_dur": c_dur,
                "a_mod": a_mod, "b_mod": b_mod, "c_mod": c_mod,
                "c_dir": c["direction"],
                "c_continues": c_continues,
                "ca_amp_r": ca_amp_r, "cb_amp_r": cb_amp_r,
                "ca_slope_r": ca_slope_r,
                "has_d": has_d,
                "year": int(dates[min(bar_idx, len(dates)-1)][:4]),
            }
            chain.update(d_info)
            chains.append(chain)

    print(f"  {pair}_{tf}: {len(chains):,} ABC chains ({sum(1 for c in chains if c['has_d']):,} with D)")
    return chains, highs, lows, closes, dates


def build_and_test_rag(all_chains, highs_map, lows_map, closes_map):
    """
    Split into IS/OOS.
    Build KDTree from IS chains.
    For each OOS chain, retrieve K nearest IS neighbors.
    Use neighbor C distributions to set TP/SL.
    Compare vs fixed TP/SL.
    """
    is_chains = [c for c in all_chains if 0 < c["year"] <= 2018]
    oos_chains = [c for c in all_chains if c["year"] > 2018]

    print(f"\n  IS chains: {len(is_chains):,}")
    print(f"  OOS chains: {len(oos_chains):,}")

    # Build KDTree from IS
    is_vecs = np.array([c["ab_vec"] for c in is_chains])
    tree = KDTree(is_vecs)

    # For each OOS chain, find K nearest IS neighbors
    K = 50
    oos_vecs = np.array([c["ab_vec"] for c in oos_chains])

    print(f"\n  Querying {len(oos_chains):,} OOS points with K={K}...")
    distances, indices = tree.query(oos_vecs, k=K)
    print(f"  Done. Median distance to nearest: {np.median(distances[:,0]):.4f}")

    # =============================================
    # Strategy 1: Fixed TP/SL (baseline) - SL=0.8*B
    # =============================================
    # Strategy 2: RAG-driven TP/SL from neighbor C distribution
    # =============================================

    results_fixed = []
    results_rag = []
    results_rag_adaptive = []

    for qi, chain in enumerate(oos_chains):
        pair = chain.get("pair", "EURUSD")  # we'll handle this below
        bar_idx = chain["bar_idx"]
        direction = chain["c_dir"]  # predicted direction
        b_amp = chain["b_amp"]
        c_amp_predicted = chain["c_amp"]

        # Get neighbor C outcomes
        neighbor_idx = indices[qi]
        neighbor_ca_amp_r = [is_chains[ni]["ca_amp_r"] for ni in neighbor_idx]
        neighbor_c_continues = [is_chains[ni]["c_continues"] for ni in neighbor_idx]
        neighbor_cb_amp_r = [is_chains[ni]["cb_amp_r"] for ni in neighbor_idx]
        # Also get what C amplitude actually was in neighbors (relative to B)
        neighbor_c_over_b = [is_chains[ni]["c_amp"] / is_chains[ni]["b_amp"]
                             for ni in neighbor_idx if is_chains[ni]["b_amp"] > 1e-6]

        # RAG predictions
        rag_direction_prob = np.mean(neighbor_c_continues)  # P(C continues A)
        rag_ca_amp_median = np.median(neighbor_ca_amp_r)
        rag_ca_amp_p25 = np.percentile(neighbor_ca_amp_r, 25)
        rag_ca_amp_p75 = np.percentile(neighbor_ca_amp_r, 75)
        if neighbor_c_over_b:
            rag_cb_median = np.median(neighbor_c_over_b)
        else:
            rag_cb_median = 1.0

        # Store for analysis
        results_rag.append({
            "bar_idx": bar_idx,
            "direction": direction,
            "rag_dir_prob": rag_direction_prob,
            "rag_ca_median": rag_ca_amp_median,
            "rag_ca_p25": rag_ca_amp_p25,
            "rag_ca_p75": rag_ca_amp_p75,
            "rag_cb_median": rag_cb_median,
            "actual_ca": chain["ca_amp_r"],
            "actual_continues": chain["c_continues"],
            "c_amp": c_amp_predicted,
            "b_amp": b_amp,
            "a_amp": chain["a_amp"],
            "lv": chain["lv"],
            "dist_nearest": distances[qi, 0],
            "dist_median": np.median(distances[qi]),
        })

    return results_rag


def main():
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

    # Extract all chains, keeping pair info
    all_chains = []
    for pair in pairs:
        chains, h, l, c, d = extract_abc_chains(pair, "H1")
        for ch in chains:
            ch["pair"] = pair
        all_chains.extend(chains)

    print(f"\nTotal ABC chains: {len(all_chains):,}")
    print(f"  With D extension: {sum(1 for c in all_chains if c['has_d']):,}")

    # =============================================
    # Part 1: RAG retrieval quality
    # =============================================
    is_chains = [c for c in all_chains if 0 < c["year"] <= 2018]
    oos_chains = [c for c in all_chains if c["year"] > 2018]

    print(f"\nIS: {len(is_chains):,}  OOS: {len(oos_chains):,}")

    # Build KDTree
    is_vecs = np.array([c["ab_vec"] for c in is_chains])
    tree = KDTree(is_vecs)

    K = 50
    oos_vecs = np.array([c["ab_vec"] for c in oos_chains])
    print(f"Querying K={K} neighbors...")
    distances, indices = tree.query(oos_vecs, k=K)
    print(f"Done. Median nearest dist: {np.median(distances[:,0]):.4f}")

    # =============================================
    # Part 2: Does RAG predict C distribution?
    # =============================================
    print(f"\n{'='*80}")
    print("  PART 2: RAG PREDICTION QUALITY")
    print(f"{'='*80}")

    # For each OOS point, compare RAG-predicted C/A ratio vs actual
    rag_medians = []
    actual_cas = []
    rag_dir_probs = []
    actual_continues = []

    for qi in range(len(oos_chains)):
        neighbor_idx = indices[qi]
        # RAG prediction: median C/A ratio of neighbors
        neighbor_ca = [is_chains[ni]["ca_amp_r"] for ni in neighbor_idx]
        rag_med = np.median(neighbor_ca)
        rag_medians.append(rag_med)
        actual_cas.append(oos_chains[qi]["ca_amp_r"])

        # Direction prediction
        neighbor_cont = [is_chains[ni]["c_continues"] for ni in neighbor_idx]
        rag_dir_probs.append(np.mean(neighbor_cont))
        actual_continues.append(oos_chains[qi]["c_continues"])

    rag_medians = np.array(rag_medians)
    actual_cas = np.array(actual_cas)
    rag_dir_probs = np.array(rag_dir_probs)
    actual_continues = np.array(actual_continues)

    # Correlation between predicted and actual C/A ratio
    corr = np.corrcoef(rag_medians, actual_cas)[0, 1]
    print(f"\n  RAG C/A ratio prediction:")
    print(f"    Correlation(predicted, actual): {corr:.4f}")
    print(f"    Mean predicted: {np.mean(rag_medians):.3f}  Mean actual: {np.mean(actual_cas):.3f}")

    # Direction prediction quality
    # High rag_dir_prob should predict c_continues=1
    print(f"\n  RAG direction prediction:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        mask_hi = rag_dir_probs >= thresh
        mask_lo = rag_dir_probs < thresh
        if mask_hi.sum() > 0 and mask_lo.sum() > 0:
            acc_hi = actual_continues[mask_hi].mean()
            acc_lo = actual_continues[mask_lo].mean()
            print(f"    P(continue) >= {thresh:.1f}: n={mask_hi.sum():>7,}  actual_continue={acc_hi:.3f}"
                  f"  |  < {thresh:.1f}: n={mask_lo.sum():>7,}  actual_continue={acc_lo:.3f}")

    # =============================================
    # Part 3: RAG-calibrated TP/SL vs fixed
    # =============================================
    print(f"\n{'='*80}")
    print("  PART 3: RAG-CALIBRATED TRADING (OOS only)")
    print(f"{'='*80}")

    # For trading, we need per-pair price data
    pair_data = {}
    for pair in pairs:
        dates, price_data = load_ohlcv(pair, "H1")
        if dates:
            pair_data[pair] = (dates, price_data)

    DELAY = 5
    decay = np.exp(-0.08 * DELAY)

    results_fixed = []    # SL=0.8*B baseline
    results_rag_tp = []   # RAG-calibrated TP, SL=0.8*B
    results_rag_full = [] # RAG-calibrated TP AND SL
    results_rag_dir = []  # RAG direction filter + RAG TP/SL

    for qi in range(len(oos_chains)):
        chain = oos_chains[qi]
        pair = chain["pair"]
        if pair not in pair_data: continue
        _, (h, l, c) = pair_data[pair]
        n_bars = len(c)

        bar_idx = chain["bar_idx"]
        eb = bar_idx + DELAY
        if eb >= n_bars - 200: continue

        direction = chain["c_dir"]
        b_amp = chain["b_amp"]
        a_amp = chain["a_amp"]
        c_amp_pred = chain["c_amp"]

        # RAG retrieval
        neighbor_idx = indices[qi]
        neighbor_ca = np.array([is_chains[ni]["ca_amp_r"] for ni in neighbor_idx])
        neighbor_cb = np.array([is_chains[ni]["c_amp"] / max(is_chains[ni]["b_amp"], 1e-6)
                                for ni in neighbor_idx])
        rag_dir_prob = np.mean([is_chains[ni]["c_continues"] for ni in neighbor_idx])

        # RAG-predicted C amplitude (as fraction of A)
        rag_ca_p50 = np.median(neighbor_ca)
        rag_ca_p25 = np.percentile(neighbor_ca, 25)
        rag_ca_p75 = np.percentile(neighbor_ca, 75)
        # RAG-predicted C/B ratio
        rag_cb_p50 = np.median(neighbor_cb)

        ep = c[eb]

        # ---- Fixed strategy (SL=0.8*B) ----
        tp_fixed = c_amp_pred * 1.0 * decay
        sl_fixed = b_amp * 0.8 * decay
        r_fixed = _run_trade(eb, direction, tp_fixed, sl_fixed, ep, h, l, c)
        if r_fixed is not None:
            results_fixed.append(r_fixed)

        # ---- RAG TP (use p50 of neighbor C/A * A as TP) ----
        rag_tp = rag_ca_p50 * a_amp * decay
        sl_rag1 = b_amp * 0.8 * decay
        r_rag_tp = _run_trade(eb, direction, rag_tp, sl_rag1, ep, h, l, c)
        if r_rag_tp is not None:
            results_rag_tp.append(r_rag_tp)

        # ---- RAG full (TP from neighbors, SL from neighbor MAE) ----
        # SL: use p25 of neighbor C/B as minimum expected C relative to B
        # If C typically is 1.5x B, then SL at 0.8*B is reasonable
        # But if neighbors show C is typically small, tighten SL
        rag_sl = b_amp * min(0.8, rag_cb_p50 * 0.5) * decay
        rag_sl = max(rag_sl, b_amp * 0.2 * decay)  # floor
        r_rag_full = _run_trade(eb, direction, rag_tp, rag_sl, ep, h, l, c)
        if r_rag_full is not None:
            results_rag_full.append(r_rag_full)

        # ---- RAG direction filter ----
        # Only trade if RAG says direction is likely (prob > 0.55)
        if rag_dir_prob > 0.55:
            r_rag_dir = _run_trade(eb, direction, rag_tp, sl_rag1, ep, h, l, c)
            if r_rag_dir is not None:
                results_rag_dir.append(r_rag_dir)

    # Print results
    for name, results in [("Fixed (SL=0.8B)", results_fixed),
                          ("RAG TP only", results_rag_tp),
                          ("RAG TP+SL", results_rag_full),
                          ("RAG TP + direction filter", results_rag_dir)]:
        if not results: continue
        pnl = [r["pnl_r"] for r in results]
        w = [x for x in pnl if x > 0]
        lo = [x for x in pnl if x <= 0]
        pf = abs(sum(w) / sum(lo)) if lo and sum(lo) != 0 else 999
        print(f"\n  {name}:")
        print(f"    n={len(pnl):>7,}  WR={len(w)/len(pnl)*100:.1f}%  "
              f"avgR={np.mean(pnl):.4f}  PF={pf:.2f}  sumR={sum(pnl):,.1f}")

    # =============================================
    # Part 4: ABCD extension
    # =============================================
    print(f"\n{'='*80}")
    print("  PART 4: ABCD EXTENSION (does knowing C improve D prediction?)")
    print(f"{'='*80}")

    abcd_chains = [c for c in all_chains if c["has_d"]]
    print(f"  ABCD chains: {len(abcd_chains):,}")

    if len(abcd_chains) > 1000:
        is_abcd = [c for c in abcd_chains if 0 < c["year"] <= 2018]
        oos_abcd = [c for c in abcd_chains if c["year"] > 2018]

        # Build 8D vectors: AB ratios + BC ratios
        def make_abcd_vec(ch):
            a_amp = ch["a_amp"]; b_amp = ch["b_amp"]; c_amp = ch["c_amp"]
            a_dur = ch["a_dur"]; b_dur = ch["b_dur"]; c_dur = ch["c_dur"]
            a_mod = ch["a_mod"]; b_mod = ch["b_mod"]; c_mod = ch["c_mod"]
            return np.array([
                np.log1p(b_amp / max(a_amp, 1e-6)),
                np.log1p(b_dur / max(a_dur, 1)),
                np.log1p(b_mod / max(a_mod, 1e-6)),
                np.log1p(c_amp / max(b_amp, 1e-6)),
                np.log1p(c_dur / max(b_dur, 1)),
                np.log1p(c_mod / max(b_mod, 1e-6)),
                np.log1p(c_amp / max(a_amp, 1e-6)),
                1.0 if ch["c_continues"] else 0.0,
            ])

        is_abcd_vecs = np.array([make_abcd_vec(c) for c in is_abcd])
        oos_abcd_vecs = np.array([make_abcd_vec(c) for c in oos_abcd])

        if len(is_abcd_vecs) > 0 and len(oos_abcd_vecs) > 0:
            tree_abcd = KDTree(is_abcd_vecs)
            K_abcd = min(50, len(is_abcd) - 1)
            dist_abcd, idx_abcd = tree_abcd.query(oos_abcd_vecs, k=K_abcd)

            print(f"  IS ABCD: {len(is_abcd):,}  OOS ABCD: {len(oos_abcd):,}")

            # Predict D properties from neighbors
            pred_d_continues = []
            actual_d_continues = []
            pred_cd_amp_r = []
            actual_cd_amp_r = []

            for qi in range(len(oos_abcd)):
                ni = idx_abcd[qi]
                # Predict D from neighbors
                neighbor_d_cont = [is_abcd[n]["d_continues_c"] for n in ni if "d_continues_c" in is_abcd[n]]
                if neighbor_d_cont:
                    pred_d_continues.append(np.mean(neighbor_d_cont))
                    actual_d_continues.append(oos_abcd[qi].get("d_continues_c", 0))
                neighbor_cd_r = [is_abcd[n].get("cd_amp_r", 1.0) for n in ni]
                pred_cd_amp_r.append(np.median(neighbor_cd_r))
                actual_cd_amp_r.append(oos_abcd[qi].get("cd_amp_r", 1.0))

            if pred_d_continues:
                pred_d = np.array(pred_d_continues)
                act_d = np.array(actual_d_continues)
                print(f"\n  D direction prediction (from ABCD neighbors):")
                for thresh in [0.4, 0.5, 0.6, 0.7]:
                    hi = pred_d >= thresh
                    lo = pred_d < thresh
                    if hi.sum() > 0 and lo.sum() > 0:
                        print(f"    P(D cont C) >= {thresh}: n={hi.sum():>6,} actual={act_d[hi].mean():.3f}"
                              f"  |  < {thresh}: n={lo.sum():>6,} actual={act_d[lo].mean():.3f}")

            corr_d = np.corrcoef(pred_cd_amp_r, actual_cd_amp_r)[0, 1]
            print(f"\n  D amplitude prediction:")
            print(f"    Corr(predicted C/D ratio, actual): {corr_d:.4f}")


def _run_trade(eb, direction, tp_pct, sl_pct, ep, highs, lows, closes):
    """Simple trade execution."""
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
            if l <= sl_price: return {"pnl_r": (sl_price - ep) / sl_d, "reason": "sl"}
        else:
            if h >= sl_price: return {"pnl_r": (ep - sl_price) / sl_d, "reason": "sl"}

        if prog >= 0.25 and not be: sl_price = ep; be = True
        if prog >= 0.50:
            if direction == 1: sl_price = max(sl_price, ep + mf * 0.40)
            else: sl_price = min(sl_price, ep - mf * 0.40)

        if direction == 1:
            if h >= tp_price: return {"pnl_r": (tp_price - ep) / sl_d, "reason": "tp"}
        else:
            if l <= tp_price: return {"pnl_r": (ep - tp_price) / sl_d, "reason": "tp"}

    pnl = (closes[end_bar] - ep) if direction == 1 else (ep - closes[end_bar])
    return {"pnl_r": pnl / sl_d, "reason": "timeout"}


if __name__ == "__main__":
    main()
