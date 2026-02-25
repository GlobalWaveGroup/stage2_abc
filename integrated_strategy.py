"""
Integrated Strategy V1
======================
Combines all discoveries:

ENTRY:
- Multi-level ABC scoring (score 10-30 sweet spot)
- RAG direction filter (rag_dir_prob > threshold)
- Pyramid sizing based on score + RAG confidence

EXIT:
- SL = 0.8 * B amplitude (structural, best calibration found)
- TP = progressive partial exits at fib levels of predicted C
- Dynamic trail using fib retracement of MFE

This is NOT the final system. This is a comprehensive test of whether
all components work together better than individually.
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


def process_pair(pair, tf):
    """Extract all per-level ABC data for both RAG store and signal generation."""
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

    # Phase 1: extract all per-level ABC chains (for RAG store)
    all_level_chains = []
    # Phase 2: extract multi-level signals (for trading)
    node_level_data = defaultdict(list)  # (bar_idx, win_id) -> list of level data

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
            b_src_rel = b["start_idx"]
            src_idx = win_bar_to_node.get(win_id, {}).get(b_src_rel, None)
            if src_idx is None: continue
            src_in, _ = edges[src_idx]
            a_cands = [e for e in src_in if e["level"] == lv and e["end_idx"] == b_src_rel and e["duration"] > 0]
            if not a_cands: continue
            a = a_cands[0]

            a_amp = abs(a["amplitude_pct"]); b_amp = abs(b["amplitude_pct"]); c_amp = abs(c["amplitude_pct"])
            a_dur = max(a["duration"], 1); b_dur = max(b["duration"], 1)
            a_mod = a["modulus"]; b_mod = b["modulus"]
            if a_amp < 1e-6 or a_mod < 1e-6: continue

            a_slope = a_amp / a_dur; b_slope = b_amp / b_dur
            amp_r = b_amp / a_amp
            dur_r = b_dur / a_dur
            slope_r = b_slope / a_slope if a_slope > 1e-10 else 1.0
            mod_r = b_mod / a_mod

            c_continues = 1 if (a["direction"] == c["direction"]) else 0
            ca_amp_r = c_amp / a_amp

            ab_vec = np.array([
                np.log1p(amp_r), np.log1p(dur_r),
                np.log1p(slope_r), np.log1p(mod_r),
            ])

            year = int(dates[min(bar_idx, len(dates)-1)][:4])

            chain = {
                "bar_idx": bar_idx, "lv": lv, "win_id": win_id,
                "ab_vec": ab_vec,
                "a_amp": a_amp, "b_amp": b_amp, "c_amp": c_amp,
                "a_dur": a_dur, "b_dur": b_dur,
                "a_mod": a_mod, "b_mod": b_mod,
                "c_dir": c["direction"], "c_continues": c_continues,
                "ca_amp_r": ca_amp_r, "amp_r": amp_r,
                "slope_r": slope_r, "year": year, "pair": pair,
            }
            all_level_chains.append(chain)

            # Scoring for multi-level signal
            score = 0.5 + lv * 0.5
            score += max(0, 2.0 * (1.0 - fib_distance(amp_r) / 0.10))
            score += max(0, 1.0 * (1.0 - fib_distance(mod_r) / 0.10))
            if amp_r < 0.382: score += 1.5
            elif amp_r < 0.618: score += 0.8
            elif amp_r < 1.0: score += 0.0
            else: score -= 0.5
            score += 1.0 if c_continues else -0.5

            ld = {
                "lv": lv, "score": score,
                "c_amp": c["amplitude_pct"], "c_dir": c["direction"],
                "a_amp": a_amp, "b_amp": b_amp, "ab_vec": ab_vec,
                "slope_r": slope_r,
            }
            node_level_data[(bar_idx, win_id)].append(ld)

    # Build multi-level signals
    DELAY = 5
    signals = []
    for (bar_idx, win_id), levels in node_level_data.items():
        if len(levels) < 1: continue
        n_lv = len(levels)
        total_score = sum(d["score"] for d in levels)
        dirs = [d["c_dir"] for d in levels]
        consensus = abs(sum(dirs)) / len(dirs)
        if consensus >= 0.9: total_score += n_lv * 0.5
        elif consensus < 0.5: total_score -= n_lv * 0.3
        if total_score < 10 or total_score >= 30: continue

        direction_sum = sum(d["c_dir"] * d["score"] for d in levels)
        direction = 1 if direction_sum > 0 else -1

        w_total = sum(d["score"] for d in levels if d["score"] > 0)
        if w_total <= 0: continue
        v_amp = sum(d["score"] * abs(d["c_amp"]) for d in levels if d["score"] > 0) / w_total
        v_b_amp = sum(d["score"] * d["b_amp"] for d in levels if d["score"] > 0) / w_total
        v_a_amp = sum(d["score"] * d["a_amp"] for d in levels if d["score"] > 0) / w_total

        # Weighted AB vector for RAG query
        ab_vec_agg = np.zeros(4)
        for d in levels:
            if d["score"] > 0:
                ab_vec_agg += d["score"] * d["ab_vec"]
        ab_vec_agg /= w_total

        if abs(v_amp) < 0.01: continue
        eb = bar_idx + DELAY
        if eb >= n_bars - 300: continue

        year = int(dates[min(bar_idx, len(dates)-1)][:4])

        signals.append({
            "bar_idx": bar_idx, "entry_bar": eb,
            "direction": direction, "score": total_score,
            "n_lv": n_lv, "consensus": consensus,
            "v_amp": v_amp, "v_b_amp": v_b_amp, "v_a_amp": v_a_amp,
            "ab_vec": ab_vec_agg, "year": year, "pair": pair,
        })

    print(f"  {pair}_{tf}: {len(all_level_chains):,} chains, {len(signals):,} signals")
    return all_level_chains, signals, highs, lows, closes, dates


def run_progressive_trade(eb, direction, tp_pct, sl_pct, ep, highs, lows, closes, score):
    """Progressive partial exit with structural SL."""
    n_bars = len(closes)
    if tp_pct < 0.005 or sl_pct < 0.005: return None
    tp_d = ep * tp_pct / 100.0
    sl_d = ep * sl_pct / 100.0
    if tp_d <= 0 or sl_d <= 0: return None

    if direction == 1:
        sl_price = ep - sl_d
    else:
        sl_price = ep + sl_d

    # Progressive targets
    targets = [
        (0.382, 0.25),  # T1
        (0.618, 0.25),  # T2
        (1.000, 0.25),  # T3
    ]
    remaining_weight = 0.25  # T4: rides with trail

    max_hold = 200
    mf = 0.0
    end_bar = min(eb + max_hold, n_bars - 1)
    total_pnl = 0.0
    total_weight = 0.0
    targets_hit = 0

    for bar in range(eb + 1, end_bar + 1):
        h = highs[bar]; l = lows[bar]
        fav = (h - ep) if direction == 1 else (ep - l)
        if fav > mf: mf = fav
        prog = mf / tp_d if tp_d > 0 else 0

        # SL check
        if direction == 1:
            if l <= sl_price:
                remaining_w = sum(t[1] for t in targets[targets_hit:]) + remaining_weight
                total_pnl += ((sl_price - ep) / sl_d) * remaining_w
                total_weight += remaining_w
                break
        else:
            if h >= sl_price:
                remaining_w = sum(t[1] for t in targets[targets_hit:]) + remaining_weight
                total_pnl += ((ep - sl_price) / sl_d) * remaining_w
                total_weight += remaining_w
                break

        # Check targets
        while targets_hit < len(targets):
            t_level, t_weight = targets[targets_hit]
            t_price_offset = tp_d * t_level
            hit = False
            if direction == 1:
                hit = h >= ep + t_price_offset
            else:
                hit = l <= ep - t_price_offset
            if hit:
                total_pnl += (t_price_offset / sl_d) * t_weight
                total_weight += t_weight
                targets_hit += 1
            else:
                break

        # Trail management
        if targets_hit >= 1:
            if direction == 1: sl_price = max(sl_price, ep)
            else: sl_price = min(sl_price, ep)
        if targets_hit >= 2 and mf > 0:
            trail = mf * 0.618
            if direction == 1: sl_price = max(sl_price, ep + trail)
            else: sl_price = min(sl_price, ep - trail)
        if targets_hit >= 3 and mf > 0:
            trail = mf * 0.764
            if direction == 1: sl_price = max(sl_price, ep + trail)
            else: sl_price = min(sl_price, ep - trail)
    else:
        remaining_w = sum(t[1] for t in targets[targets_hit:]) + remaining_weight
        pnl = (closes[end_bar] - ep) if direction == 1 else (ep - closes[end_bar])
        total_pnl += (pnl / sl_d) * remaining_w
        total_weight += remaining_w

    if total_weight <= 0: return None
    return total_pnl / total_weight


def main():
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

    all_chains = []
    all_signals = []
    pair_price = {}

    for pair in pairs:
        result = process_pair(pair, "H1")
        chains, signals = result[0], result[1]
        h, l, c, d = result[2], result[3], result[4], result[5]
        all_chains.extend(chains)
        all_signals.extend(signals)
        if h is not None:
            pair_price[pair] = (h, l, c)

    print(f"\nTotal: {len(all_chains):,} chains, {len(all_signals):,} signals")

    # Build RAG store from IS chains
    is_chains = [c for c in all_chains if 0 < c["year"] <= 2018]
    is_vecs = np.array([c["ab_vec"] for c in is_chains])
    tree = KDTree(is_vecs)
    print(f"RAG store: {len(is_chains):,} IS chains")

    # Split signals
    is_signals = [s for s in all_signals if 0 < s["year"] <= 2018]
    oos_signals = [s for s in all_signals if s["year"] > 2018]
    print(f"Signals: {len(is_signals):,} IS, {len(oos_signals):,} OOS")

    DELAY = 5
    decay = np.exp(-0.08 * DELAY)
    K = 50

    # =============================================
    # Run strategies on both IS and OOS
    # =============================================
    strategies = {
        "A_baseline": {},       # SL=0.8B, TP=1.0*v_amp, flat exit
        "B_progressive": {},    # SL=0.8B, progressive partial exit
        "C_rag_filter": {},     # B + RAG direction filter (prob > 0.6)
        "D_rag_pyramid": {},    # C + pyramid sizing from score + RAG
        "E_full": {},           # D + RAG-adjusted TP
    }

    for split_name, signals in [("IS", is_signals), ("OOS", oos_signals)]:
        for strat in strategies:
            strategies[strat][split_name] = []

        for sig in signals:
            pair = sig["pair"]
            if pair not in pair_price: continue
            h, l, c = pair_price[pair]
            n_bars = len(c)

            eb = sig["entry_bar"]
            if eb >= n_bars - 200: continue
            ep = c[eb]
            direction = sig["direction"]
            score = sig["score"]
            v_amp = sig["v_amp"]
            v_b_amp = sig["v_b_amp"]
            v_a_amp = sig["v_a_amp"]

            tp_pct = abs(v_amp) * 1.0 * decay
            sl_pct = v_b_amp * 0.8 * decay
            if tp_pct < 0.005 or sl_pct < 0.005: continue

            # RAG query
            ab_vec = sig["ab_vec"].reshape(1, -1)
            dists, idxs = tree.query(ab_vec, k=K)
            rag_dir_prob = np.mean([is_chains[ni]["c_continues"] for ni in idxs[0]])
            rag_ca_median = np.median([is_chains[ni]["ca_amp_r"] for ni in idxs[0]])

            # Pyramid sizing
            if score < 15: base_size = 1.5
            elif score < 20: base_size = 2.0
            else: base_size = 2.5

            # A: Baseline (flat, SL=0.8B)
            tp_d = ep * tp_pct / 100.0
            sl_d = ep * sl_pct / 100.0
            if tp_d > 0 and sl_d > 0:
                if direction == 1:
                    tp_price = ep + tp_d; sl_price = ep - sl_d
                else:
                    tp_price = ep - tp_d; sl_price = ep + sl_d
                mf = 0.0; be = False
                end_bar = min(eb + 200, n_bars - 1)
                pnl_a = 0.0
                for bar in range(eb + 1, end_bar + 1):
                    hh = h[bar]; ll = l[bar]
                    fav = (hh - ep) if direction == 1 else (ep - ll)
                    if fav > mf: mf = fav
                    prog = mf / tp_d
                    if direction == 1:
                        if ll <= sl_price: pnl_a = (sl_price - ep) / sl_d; break
                    else:
                        if hh >= sl_price: pnl_a = (ep - sl_price) / sl_d; break
                    if prog >= 0.25 and not be: sl_price = ep; be = True
                    if prog >= 0.50:
                        if direction == 1: sl_price = max(sl_price, ep + mf * 0.40)
                        else: sl_price = min(sl_price, ep - mf * 0.40)
                    if direction == 1:
                        if hh >= tp_price: pnl_a = (tp_price - ep) / sl_d; break
                    else:
                        if ll <= tp_price: pnl_a = (ep - tp_price) / sl_d; break
                else:
                    pnl_val = (c[end_bar] - ep) if direction == 1 else (ep - c[end_bar])
                    pnl_a = pnl_val / sl_d

                strategies["A_baseline"][split_name].append({
                    "pnl_r": pnl_a, "pnl_sized": pnl_a * 1.0, "size": 1.0, "score": score})

            # B: Progressive partial exit
            pnl_b = run_progressive_trade(eb, direction, tp_pct, sl_pct, ep, h, l, c, score)
            if pnl_b is not None:
                strategies["B_progressive"][split_name].append({
                    "pnl_r": pnl_b, "pnl_sized": pnl_b * 1.0, "size": 1.0, "score": score})

            # C: RAG direction filter
            if rag_dir_prob > 0.6 and pnl_b is not None:
                strategies["C_rag_filter"][split_name].append({
                    "pnl_r": pnl_b, "pnl_sized": pnl_b * 1.0, "size": 1.0, "score": score})

            # D: RAG pyramid (filter + sizing)
            if rag_dir_prob > 0.6 and pnl_b is not None:
                # Adjust size by RAG confidence
                rag_bonus = 1.0 + max(0, (rag_dir_prob - 0.6)) * 2.0  # 1.0 to 1.8
                size_d = base_size * rag_bonus
                strategies["D_rag_pyramid"][split_name].append({
                    "pnl_r": pnl_b, "pnl_sized": pnl_b * size_d, "size": size_d, "score": score})

            # E: Full (RAG-adjusted TP + pyramid)
            if rag_dir_prob > 0.6:
                # Use RAG C/A median to adjust TP
                rag_tp_pct = rag_ca_median * v_a_amp * decay
                rag_tp_pct = max(rag_tp_pct, tp_pct * 0.5)  # floor at 50% of original
                rag_tp_pct = min(rag_tp_pct, tp_pct * 2.0)  # cap at 200%
                pnl_e = run_progressive_trade(eb, direction, rag_tp_pct, sl_pct, ep, h, l, c, score)
                if pnl_e is not None:
                    size_e = base_size * rag_bonus
                    strategies["E_full"][split_name].append({
                        "pnl_r": pnl_e, "pnl_sized": pnl_e * size_e, "size": size_e, "score": score})

    # =============================================
    # Results
    # =============================================
    print(f"\n{'='*80}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*80}")

    for strat_name in strategies:
        print(f"\n  === {strat_name} ===")
        for split in ["IS", "OOS"]:
            data = strategies[strat_name].get(split, [])
            if not data: continue
            pnl = [d["pnl_sized"] for d in data]
            w = [x for x in pnl if x > 0]
            lo = [x for x in pnl if x <= 0]
            pf = abs(sum(w) / sum(lo)) if lo and sum(lo) != 0 else 999
            total_cap = sum(d["size"] for d in data)
            avg_size = np.mean([d["size"] for d in data])
            print(f"    {split:>3}: n={len(pnl):>7,}  WR={len(w)/len(pnl)*100:.1f}%  "
                  f"avgR={np.mean(pnl):.4f}  PF={pf:.2f}  sumR={sum(pnl):>10,.1f}  "
                  f"avg_size={avg_size:.2f}  E/u={sum(pnl)/total_cap:.4f}")

    # By score tier for best strategy
    print(f"\n{'='*80}")
    print("  BEST STRATEGY BY SCORE TIER (OOS)")
    print(f"{'='*80}")
    for strat_name in ["D_rag_pyramid", "E_full"]:
        oos_data = strategies[strat_name].get("OOS", [])
        if not oos_data: continue
        print(f"\n  {strat_name}:")
        for lo_s, hi_s in [(10, 15), (15, 20), (20, 30)]:
            sub = [d for d in oos_data if lo_s <= d["score"] < hi_s]
            if not sub: continue
            pnl = [d["pnl_sized"] for d in sub]
            w = [x for x in pnl if x > 0]
            l = [x for x in pnl if x <= 0]
            pf = abs(sum(w) / sum(l)) if l and sum(l) != 0 else 999
            print(f"    score {lo_s}-{hi_s}: n={len(sub):>6,}  WR={len(w)/len(sub)*100:.1f}%  "
                  f"PF={pf:.2f}  sumR={sum(pnl):>8,.1f}")


if __name__ == "__main__":
    main()
