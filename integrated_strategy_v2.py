"""
Integrated Strategy V2 — Final Comprehensive Backtest
=====================================================
Combines ALL findings (F1-F16):

SIGNAL GENERATION:
- Multi-level ABC scoring (score 10-30 sweet spot) [F6]
- Cost filter: c_amp >= 0.20% [F16]
- Quality: low completion = better (don't filter out, use for sizing) [F13a]
- Arrival: trend arrival = better [F13d]
- Exhaustion: U-shape, extremes better [F13e]
- Time symmetry: high = better [F13c]

EXIT:
- SL = 0.8 * B amplitude (structural stop) [F7]
- Progressive partial exits at fib levels [F9]
- TP = 1.0 * predicted C amplitude

SIZING:
- Base size from score tier
- Quality multiplier from completion (inverse) + time_sym + arrival
- RAG confidence multiplier from direction probability

RAG:
- 7D vector: 4D AB ratios + 3D quality (completion, arrival, exhaustion) [F15]
- Used as confidence adjuster (sizing), NOT for TP/SL setting [F10]
- Direction filter: only trade if rag_dir_prob > 0.55

VALIDATION:
- 48 pairs × H1
- IS: year <= 2018, OOS: year > 2018
- Walk-forward: per-year results
- Per-pair breakdown
- Drawdown analysis

Architecture: all computation in worker processes (parallel across 48 pairs).
RAG built from IS chains of ALL pairs, queried for OOS.
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


def compute_quality(a_amp, b_amp, c_amp, a_dur, b_dur, c_dur,
                    a_slope, b_slope, lv, bar_idx, closes, n_bars):
    """Compute turning point quality metrics."""
    # Completion degree (low = better for trading, F13a)
    ca_ratio = c_amp / a_amp if a_amp > 1e-8 else 1.0
    fib_dist = fib_distance(ca_ratio)
    fib_score = max(0, 1.0 - fib_dist / 0.15)
    time_sym = min(a_dur, c_dur) / max(a_dur, c_dur) if max(a_dur, c_dur) > 0 else 0
    ba_ratio = b_amp / a_amp if a_amp > 1e-8 else 1.0
    if ba_ratio < 0.15: b_quality = ba_ratio / 0.15
    elif ba_ratio <= 1.0: b_quality = 1.0
    elif ba_ratio <= 2.0: b_quality = max(0, 1.0 - (ba_ratio - 1.0))
    else: b_quality = 0.0
    completion = fib_score * 0.4 + time_sym * 0.3 + b_quality * 0.3

    # Arrival type (trend = better, F13d)
    b_complexity = min(lv / 7.0, 1.0)
    dur_ratio_raw = b_dur / max(a_dur, 1)
    if dur_ratio_raw > 3.0: b_dur_score = 1.0
    elif dur_ratio_raw > 1.5: b_dur_score = 0.7
    elif dur_ratio_raw > 0.5: b_dur_score = 0.3
    else: b_dur_score = 0.0

    # Exhaustion (U-shape = extremes better, F13e)
    exhaustion = 0.5
    b_start_bar = bar_idx - b_dur
    b_mid_bar = bar_idx - b_dur // 2
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

    return completion, arrival, exhaustion, time_sym, fib_score


def run_progressive_trade(eb, direction, tp_pct, sl_pct, ep, highs, lows, closes):
    """Progressive partial exit with structural SL. Returns weighted R."""
    n_bars = len(closes)
    if tp_pct < 0.005 or sl_pct < 0.005: return None
    tp_d = ep * tp_pct / 100.0
    sl_d = ep * sl_pct / 100.0
    if tp_d <= 0 or sl_d <= 0: return None

    if direction == 1:
        sl_price = ep - sl_d
    else:
        sl_price = ep + sl_d

    targets = [(0.382, 0.25), (0.618, 0.25), (1.000, 0.25)]
    remaining_weight = 0.25

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

        # SL check
        if direction == 1:
            if l <= sl_price:
                rem_w = sum(t[1] for t in targets[targets_hit:]) + remaining_weight
                total_pnl += ((sl_price - ep) / sl_d) * rem_w
                total_weight += rem_w
                break
        else:
            if h >= sl_price:
                rem_w = sum(t[1] for t in targets[targets_hit:]) + remaining_weight
                total_pnl += ((ep - sl_price) / sl_d) * rem_w
                total_weight += rem_w
                break

        # Check targets
        while targets_hit < len(targets):
            t_level, t_weight = targets[targets_hit]
            t_offset = tp_d * t_level
            hit = (h >= ep + t_offset) if direction == 1 else (l <= ep - t_offset)
            if hit:
                total_pnl += (t_offset / sl_d) * t_weight
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
        rem_w = sum(t[1] for t in targets[targets_hit:]) + remaining_weight
        pnl = (closes[end_bar] - ep) if direction == 1 else (ep - closes[end_bar])
        total_pnl += (pnl / sl_d) * rem_w
        total_weight += rem_w

    if total_weight <= 0: return None
    return total_pnl / total_weight


def extract_pair_data(args):
    """
    Worker: extract all ABC chains + signals for one pair.
    Returns (chains_for_rag, signals_for_trading).
    Chains = per-level ABC records (for RAG store).
    Signals = aggregated multi-level signals (for trading).
    """
    pair, tf = args
    dates, price_data = load_ohlcv(pair, tf)
    if dates is None: return pair, [], []
    highs, lows, closes = price_data
    n_bars = len(closes)
    feats, edges = load_zigzag_topo(pair, tf)
    if feats is None: return pair, [], []

    win_bar_to_node = defaultdict(dict)
    for i in range(len(feats)):
        win = int(feats[i, 1])
        rel_idx = int(feats[i, 2])
        ml = int(feats[i, 5])
        if rel_idx not in win_bar_to_node[win] or ml > feats[win_bar_to_node[win][rel_idx], 5]:
            win_bar_to_node[win][rel_idx] = i

    decay = np.exp(-0.08 * DELAY)
    chains = []
    signals = []

    node_levels = defaultdict(list)  # bar_idx -> list of level data

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

            amp_r = b_amp / a_amp
            mod_r = b_mod / a_mod

            c_continues = 1 if (a_edge["direction"] == c_edge["direction"]) else 0
            ca_amp_r = c_amp / a_amp

            # Quality metrics
            completion, arrival, exhaustion, time_sym, fib_sc = compute_quality(
                a_amp, b_amp, c_amp, a_dur, b_dur, c_dur,
                a_slope, b_slope, lv, bar_idx, closes, n_bars)

            # 7D RAG vector: 4D AB ratios + 3D quality [F15]
            ab_vec_7d = np.array([
                np.log1p(b_amp / a_amp),
                np.log1p(b_dur / a_dur),
                np.log1p(b_slope / a_slope) if a_slope > 1e-10 else 0,
                np.log1p(b_mod / a_mod) if a_mod > 1e-6 else 0,
                completion,
                arrival,
                exhaustion,
            ])

            year = int(dates[min(bar_idx, len(dates)-1)][:4])

            # Store for RAG
            chains.append({
                "ab_vec": ab_vec_7d,
                "c_continues": c_continues,
                "ca_amp_r": ca_amp_r,
                "year": year,
            })

            # Scoring
            score = 0.5 + lv * 0.5
            score += max(0, 2.0 * (1.0 - fib_distance(amp_r) / 0.10))
            score += max(0, 1.0 * (1.0 - fib_distance(mod_r) / 0.10))
            if amp_r < 0.382: score += 1.5
            elif amp_r < 0.618: score += 0.8
            elif amp_r >= 1.0: score -= 0.5
            score += 1.0 if c_continues else -0.5

            node_levels[bar_idx].append({
                "lv": lv, "score": score,
                "c_amp": c_edge["amplitude_pct"], "c_dir": c_edge["direction"],
                "a_amp": a_amp, "b_amp": b_amp, "c_amp_abs": c_amp,
                "a_dur": a_dur, "b_dur": b_dur,
                "ab_vec": ab_vec_7d,
                "completion": completion, "arrival": arrival,
                "exhaustion": exhaustion, "time_sym": time_sym,
                "c_continues": c_continues,
            })

    # Build multi-level signals
    for bar_idx, levels in node_levels.items():
        if not levels: continue
        n_lv = len(levels)
        total_score = sum(d["score"] for d in levels)
        dirs = [d["c_dir"] for d in levels]
        consensus = abs(sum(dirs)) / len(dirs)
        if consensus >= 0.9: total_score += n_lv * 0.5
        elif consensus < 0.5: total_score -= n_lv * 0.3
        if total_score < 10 or total_score >= 30: continue

        direction_sum = sum(d["c_dir"] * d["score"] for d in levels)
        direction = 1 if direction_sum > 0 else -1

        ws = [d["score"] for d in levels if d["score"] > 0]
        w_total = sum(ws)
        if w_total <= 0: continue

        def wmean(key):
            return sum(d["score"] * d[key] for d in levels if d["score"] > 0) / w_total

        v_amp = abs(wmean("c_amp"))
        v_b_amp = wmean("b_amp")
        v_a_amp = wmean("a_amp")
        c_amp_abs = wmean("c_amp_abs")

        # Cost filter [F16]: skip tiny signals
        if c_amp_abs < 0.20: continue

        # Aggregate 7D vector
        ab_vec_agg = np.zeros(7)
        for d in levels:
            if d["score"] > 0:
                ab_vec_agg += d["score"] * d["ab_vec"]
        ab_vec_agg /= w_total

        # Aggregate quality
        completion = wmean("completion")
        arrival = wmean("arrival")
        exhaustion = wmean("exhaustion")
        time_sym = wmean("time_sym")

        # Quality-based sizing multiplier [F13]
        # Low completion = better (Q1 PF=2.43 vs Q5 PF=1.61)
        compl_mult = 1.0 + max(0, 0.6 - completion) * 1.5  # range ~1.0-1.9
        # High time symmetry = better (Q4 PF=2.25 vs Q1 PF=1.28)
        tsym_mult = 0.7 + time_sym * 0.6  # range 0.7-1.3
        # Trend arrival = better (Q3-Q5 PF~2.0 vs Q1 PF=1.53)
        arr_mult = 0.8 + arrival * 0.5  # range 0.8-1.3
        # Exhaustion U-shape: extremes better
        exh_deviation = abs(exhaustion - 0.5)  # 0=neutral, 0.5=extreme
        exh_mult = 0.9 + exh_deviation * 0.4  # range 0.9-1.1

        quality_mult = compl_mult * tsym_mult * arr_mult * exh_mult

        eb = bar_idx + DELAY
        if eb >= n_bars - 200: continue
        ep = closes[eb]
        year = int(dates[min(bar_idx, len(dates)-1)][:4])

        # TP/SL
        tp_pct = v_amp * 1.0 * decay
        sl_pct = v_b_amp * 0.8 * decay

        # Run progressive trade
        pnl_r = run_progressive_trade(eb, direction, tp_pct, sl_pct, ep, highs, lows, closes)
        if pnl_r is None: continue

        signals.append({
            "bar_idx": bar_idx, "year": year, "pair": pair,
            "direction": direction, "score": round(total_score, 2),
            "n_lv": n_lv, "pnl_r": round(pnl_r, 4),
            "ab_vec": ab_vec_agg,
            "completion": round(completion, 4),
            "arrival": round(arrival, 4),
            "exhaustion": round(exhaustion, 4),
            "time_sym": round(time_sym, 4),
            "quality_mult": round(quality_mult, 4),
            "tp_pct": round(tp_pct, 4),
            "sl_pct": round(sl_pct, 4),
            "c_amp": round(c_amp_abs, 4),
        })

    print(f"  {pair}_{tf}: {len(chains):,} chains, {len(signals):,} signals")
    return pair, chains, signals


def pf_wr(pnl_list):
    if not pnl_list: return 0, 0, 0, 0, 0
    w = [x for x in pnl_list if x > 0]
    lo = [x for x in pnl_list if x <= 0]
    pf = abs(sum(w) / sum(lo)) if lo and sum(lo) != 0 else 999
    return len(pnl_list), len(w)/len(pnl_list)*100, np.mean(pnl_list), pf, sum(pnl_list)


def sharpe_annual(pnl_list, trades_per_year=250):
    """Approximate annualized Sharpe from per-trade R-multiples."""
    if len(pnl_list) < 10: return 0
    arr = np.array(pnl_list)
    mu = arr.mean()
    std = arr.std()
    if std < 1e-8: return 0
    return mu / std * np.sqrt(trades_per_year)


def max_drawdown_r(pnl_list):
    """Max drawdown in R-multiple terms (cumulative)."""
    if not pnl_list: return 0
    cum = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return abs(dd.min())


def main():
    print("=" * 80)
    print("  INTEGRATED STRATEGY V2 — FINAL COMPREHENSIVE BACKTEST")
    print("  48 pairs × H1 | IS: <=2018 | OOS: >2018")
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

    # Phase 1: Extract data in parallel
    all_chains = []
    all_signals = []
    pair_signals = defaultdict(list)

    with Pool(min(48, len(tasks))) as pool:
        for pair, chains, signals in pool.imap_unordered(extract_pair_data, tasks):
            all_chains.extend(chains)
            all_signals.extend(signals)
            pair_signals[pair] = signals

    print(f"\n  Total: {len(all_chains):,} RAG chains, {len(all_signals):,} signals")

    is_chains = [c for c in all_chains if 0 < c["year"] <= 2018]
    oos_chains = [c for c in all_chains if c["year"] > 2018]
    is_signals = [s for s in all_signals if 0 < s["year"] <= 2018]
    oos_signals = [s for s in all_signals if s["year"] > 2018]
    print(f"  RAG: IS={len(is_chains):,} OOS={len(oos_chains):,}")
    print(f"  Signals: IS={len(is_signals):,} OOS={len(oos_signals):,}")

    # Phase 2: Build RAG store from IS chains (7D)
    print(f"\n  Building RAG KDTree (7D)...")
    is_vecs = np.array([c["ab_vec"] for c in is_chains])
    tree = KDTree(is_vecs)
    print(f"  KDTree built: {len(is_vecs):,} vectors, 7D")

    K = 50

    # Phase 3: Run strategies
    # A = Baseline: flat size, no RAG, progressive exit
    # B = +Quality sizing (quality_mult from completion/arrival/exhaustion/time_sym)
    # C = +RAG filter (rag_dir_prob > 0.55)
    # D = +RAG confidence sizing (quality_mult × rag_bonus)
    # E = Final: D + higher RAG threshold (>0.60) + score-based sizing

    print(f"\n{'='*80}")
    print("  RUNNING 5 STRATEGY VARIANTS")
    print(f"{'='*80}")

    results = {s: {"IS": [], "OOS": []} for s in ["A_base", "B_quality", "C_rag_filt", "D_rag_conf", "E_final"]}

    for split_name, sigs in [("IS", is_signals), ("OOS", oos_signals)]:
        # Batch RAG query
        if not sigs: continue
        sig_vecs = np.array([s["ab_vec"] for s in sigs])
        print(f"  RAG querying {split_name}: {len(sigs):,} signals...")
        dists, idxs = tree.query(sig_vecs, k=min(K, len(is_chains)-1))

        for qi, sig in enumerate(sigs):
            pnl_r = sig["pnl_r"]
            score = sig["score"]
            quality_mult = sig["quality_mult"]

            # RAG predictions
            ni = idxs[qi]
            rag_dir_prob = np.mean([is_chains[n]["c_continues"] for n in ni])

            # Score-based base size
            if score < 15: base_size = 1.0
            elif score < 20: base_size = 1.5
            else: base_size = 2.0

            # A: Baseline (flat size=1.0, no quality, no RAG)
            results["A_base"][split_name].append({
                "pnl_r": pnl_r, "size": 1.0,
                "pnl_sized": pnl_r * 1.0,
                "year": sig["year"], "pair": sig["pair"],
                "score": score,
            })

            # B: Quality sizing only
            size_b = quality_mult
            results["B_quality"][split_name].append({
                "pnl_r": pnl_r, "size": size_b,
                "pnl_sized": pnl_r * size_b,
                "year": sig["year"], "pair": sig["pair"],
                "score": score,
            })

            # C: RAG filter (>0.55) + flat size
            if rag_dir_prob > 0.55:
                results["C_rag_filt"][split_name].append({
                    "pnl_r": pnl_r, "size": 1.0,
                    "pnl_sized": pnl_r * 1.0,
                    "year": sig["year"], "pair": sig["pair"],
                    "score": score,
                })

            # D: RAG confidence + quality sizing
            if rag_dir_prob > 0.55:
                rag_bonus = 1.0 + max(0, rag_dir_prob - 0.55) * 2.0  # 1.0-1.9
                size_d = quality_mult * rag_bonus
                results["D_rag_conf"][split_name].append({
                    "pnl_r": pnl_r, "size": size_d,
                    "pnl_sized": pnl_r * size_d,
                    "year": sig["year"], "pair": sig["pair"],
                    "score": score,
                })

            # E: Final (tighter RAG >0.60 + score-based + quality + RAG confidence)
            if rag_dir_prob > 0.60:
                rag_bonus = 1.0 + max(0, rag_dir_prob - 0.60) * 2.5  # 1.0-2.0
                size_e = base_size * quality_mult * rag_bonus
                results["E_final"][split_name].append({
                    "pnl_r": pnl_r, "size": size_e,
                    "pnl_sized": pnl_r * size_e,
                    "year": sig["year"], "pair": sig["pair"],
                    "score": score,
                })

    # =============================================
    # REPORT 1: Strategy comparison
    # =============================================
    print(f"\n{'='*80}")
    print("  REPORT 1: STRATEGY COMPARISON")
    print(f"{'='*80}")

    for strat in results:
        print(f"\n  === {strat} ===")
        for split in ["IS", "OOS"]:
            data = results[strat][split]
            if not data: continue
            pnl = [d["pnl_sized"] for d in data]
            pnl_r = [d["pnl_r"] for d in data]
            n, wr, ar, pf, sr = pf_wr(pnl)
            _, wr_r, ar_r, pf_r, sr_r = pf_wr(pnl_r)
            avg_size = np.mean([d["size"] for d in data])
            sh = sharpe_annual(pnl_r, trades_per_year=max(1, n // max(1, len(set(d["year"] for d in data)))))
            md = max_drawdown_r(pnl)
            print(f"    {split:>3}: n={n:>7,}  WR_r={wr_r:.1f}%  PF_r={pf_r:.2f}  "
                  f"PF_sized={pf:.2f}  avgR={ar:.4f}  sumR={sr:>10,.0f}  "
                  f"avg_sz={avg_size:.2f}  maxDD={md:,.0f}")

    # =============================================
    # REPORT 2: Walk-forward by year (E_final)
    # =============================================
    print(f"\n{'='*80}")
    print("  REPORT 2: WALK-FORWARD BY YEAR (E_final)")
    print(f"{'='*80}")

    all_e = results["E_final"]["IS"] + results["E_final"]["OOS"]
    years = sorted(set(d["year"] for d in all_e))
    for year in years:
        yr_data = [d for d in all_e if d["year"] == year]
        if not yr_data: continue
        pnl = [d["pnl_sized"] for d in yr_data]
        pnl_r = [d["pnl_r"] for d in yr_data]
        n, wr, ar, pf, sr = pf_wr(pnl)
        _, wr_r, _, pf_r, _ = pf_wr(pnl_r)
        md = max_drawdown_r(pnl)
        oos_tag = " [OOS]" if year > 2018 else ""
        print(f"    {year}{oos_tag:>6}: n={n:>6,}  WR={wr_r:.1f}%  PF_r={pf_r:.2f}  "
              f"PF_sz={pf:.2f}  sumR={sr:>8,.0f}  maxDD={md:>6,.0f}")

    # =============================================
    # REPORT 3: Per-pair summary (E_final, OOS only)
    # =============================================
    print(f"\n{'='*80}")
    print("  REPORT 3: PER-PAIR SUMMARY (E_final, OOS)")
    print(f"{'='*80}")

    oos_e = results["E_final"]["OOS"]
    pairs_in_oos = sorted(set(d["pair"] for d in oos_e))
    pair_stats = []
    for pair in pairs_in_oos:
        pdata = [d for d in oos_e if d["pair"] == pair]
        if not pdata: continue
        pnl = [d["pnl_sized"] for d in pdata]
        pnl_r = [d["pnl_r"] for d in pdata]
        n, wr, ar, pf, sr = pf_wr(pnl)
        _, wr_r, _, pf_r, _ = pf_wr(pnl_r)
        pair_stats.append((pair, n, wr_r, pf_r, pf, sr))

    pair_stats.sort(key=lambda x: -x[5])
    profitable = sum(1 for p in pair_stats if p[5] > 0)
    print(f"  Profitable pairs: {profitable}/{len(pair_stats)}")
    for pair, n, wr, pfr, pfs, sr in pair_stats[:10]:
        print(f"    {pair:>8}: n={n:>5,}  WR={wr:.1f}%  PF_r={pfr:.2f}  PF_sz={pfs:.2f}  sumR={sr:>7,.0f}")
    print(f"    ...")
    for pair, n, wr, pfr, pfs, sr in pair_stats[-5:]:
        print(f"    {pair:>8}: n={n:>5,}  WR={wr:.1f}%  PF_r={pfr:.2f}  PF_sz={pfs:.2f}  sumR={sr:>7,.0f}")

    # =============================================
    # REPORT 4: Score tier analysis (E_final)
    # =============================================
    print(f"\n{'='*80}")
    print("  REPORT 4: SCORE TIER (E_final)")
    print(f"{'='*80}")
    for split in ["IS", "OOS"]:
        print(f"\n  {split}:")
        data = results["E_final"][split]
        for lo_s, hi_s in [(10, 13), (13, 16), (16, 20), (20, 25), (25, 30)]:
            sub = [d for d in data if lo_s <= d["score"] < hi_s]
            if not sub: continue
            pnl = [d["pnl_sized"] for d in sub]
            n, wr, ar, pf, sr = pf_wr(pnl)
            print(f"    score {lo_s:>2}-{hi_s:>2}: n={n:>6,}  WR={wr:.1f}%  PF={pf:.2f}  "
                  f"avgR={ar:.4f}  sumR={sr:>8,.0f}")

    # =============================================
    # REPORT 5: Quality tier analysis (E_final, OOS)
    # =============================================
    print(f"\n{'='*80}")
    print("  REPORT 5: QUALITY MULTIPLIER TIERS (E_final, OOS)")
    print(f"{'='*80}")
    oos_e = results["E_final"]["OOS"]
    if oos_e:
        sizes = sorted([d["size"] for d in oos_e])
        qs = max(1, len(sizes) // 5)
        sz_sorted = sorted(oos_e, key=lambda x: x["size"])
        for qi in range(5):
            chunk = sz_sorted[qi*qs:(qi+1)*qs] if qi < 4 else sz_sorted[qi*qs:]
            pnl = [d["pnl_sized"] for d in chunk]
            pnl_r = [d["pnl_r"] for d in chunk]
            n, wr, ar, pf, sr = pf_wr(pnl)
            _, wr_r, _, pf_r, _ = pf_wr(pnl_r)
            sz_lo = chunk[0]["size"]; sz_hi = chunk[-1]["size"]
            print(f"    Q{qi+1} size=[{sz_lo:.2f},{sz_hi:.2f}]: n={n:>6,}  "
                  f"WR_r={wr_r:.1f}%  PF_r={pf_r:.2f}  PF_sz={pf:.2f}  sumR={sr:>8,.0f}")

    # =============================================
    # REPORT 6: Summary statistics
    # =============================================
    print(f"\n{'='*80}")
    print("  REPORT 6: FINAL SUMMARY")
    print(f"{'='*80}")

    best = results["E_final"]
    for split in ["IS", "OOS"]:
        data = best[split]
        if not data: continue
        pnl = [d["pnl_sized"] for d in data]
        pnl_r = [d["pnl_r"] for d in data]
        n, wr, ar, pf, sr = pf_wr(pnl)
        _, wr_r, ar_r, pf_r, sr_r = pf_wr(pnl_r)
        avg_size = np.mean([d["size"] for d in data])
        md = max_drawdown_r(pnl)
        n_years = len(set(d["year"] for d in data))
        tpy = n / max(n_years, 1)
        sh = sharpe_annual(pnl_r, trades_per_year=int(tpy))

        print(f"\n  {split}:")
        print(f"    Trades: {n:,} over {n_years} years ({tpy:.0f}/yr)")
        print(f"    WR (per R): {wr_r:.1f}%")
        print(f"    PF (per R): {pf_r:.2f}")
        print(f"    PF (sized): {pf:.2f}")
        print(f"    Mean R (sized): {ar:.4f}")
        print(f"    Sum R (sized): {sr:,.0f}")
        print(f"    Avg position size: {avg_size:.2f}")
        print(f"    Max drawdown (R): {md:,.0f}")
        print(f"    Approx Sharpe: {sh:.2f}")

    print(f"\n{'='*80}")
    print("  COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
