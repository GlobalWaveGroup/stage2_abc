"""
Fuzzy scoring system: multiple soft criteria stacked together.
Each criterion contributes a continuous score, not a hard cutoff.
The combined score should show monotonic relationship with alpha.

Key insights from prior analysis:
- Level (L0-L7): higher level = stronger signal (C_cont 55%->85%, PF 0.64->1.15)
- Fib proximity: closer to fib = better (delta_PF +0.13 to +0.19)
- BA amp ratio: lower = better (shallow retracement = strong trend)
- Multi-level resonance (n_lv): more levels = better (PF 0.79->1.45)
- Direction consensus: all levels agree = better

All are fuzzy, continuous, and should be combined additively.
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


def process_pair_tf(args):
    pair, tf = args
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
    samples = []

    for i in range(len(feats)):
        bar_idx = int(feats[i, 0])
        win_id = int(feats[i, 1])
        rel_idx = int(feats[i, 2])
        max_level = int(feats[i, 5])

        if bar_idx < 50 or bar_idx >= n_bars - 50:
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

        # Per-level data collection
        level_scores = []
        level_c_amps = []
        level_c_dirs = []
        level_a_amps = []

        for lv in range(max_level + 1):
            b_list = [e for e in in_by_lv.get(lv, []) if e["end_idx"] == rel_idx]
            c_list = [e for e in out_by_lv.get(lv, []) if e["start_idx"] == rel_idx]
            if not c_list:
                continue

            c = c_list[0]
            level_c_amps.append(c["amplitude_pct"])
            level_c_dirs.append(c["direction"])

            if not b_list:
                continue
            b = b_list[0]

            # Trace A
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

            level_a_amps.append(a_amp)

            # === FUZZY SCORING per level ===
            score = 0.0

            # S1: Level contribution (higher level = more weight)
            # L0=0.5, L1=1.0, L2=1.5, L3=2.0, L4=2.5, L5=3.0
            s_level = 0.5 + lv * 0.5
            score += s_level

            # S2: Fib proximity (amp ratio)
            # Distance 0 -> +2.0, distance 0.1 -> +0, distance >0.1 -> slightly negative
            fib_d = fib_distance(ba_amp_ratio)
            s_fib = max(0, 2.0 * (1.0 - fib_d / 0.10))
            score += s_fib

            # S3: Fib proximity (mod ratio)
            fib_d_mod = fib_distance(ba_mod_ratio)
            s_fib_mod = max(0, 1.0 * (1.0 - fib_d_mod / 0.10))
            score += s_fib_mod

            # S4: Shallow retracement bonus
            # ba_amp < 0.382 -> +1.5, ba_amp > 1.0 -> -0.5
            if ba_amp_ratio < 0.382:
                s_shallow = 1.5
            elif ba_amp_ratio < 0.618:
                s_shallow = 0.8
            elif ba_amp_ratio < 1.0:
                s_shallow = 0.0
            else:
                s_shallow = -0.5
            score += s_shallow

            # S5: Direction continuation at this level
            c_continues = 1 if (a["direction"] == c["direction"]) else 0
            s_cont = 1.0 if c_continues else -0.5
            score += s_cont

            level_scores.append({
                "lv": lv,
                "score": score,
                "s_level": s_level,
                "s_fib": s_fib,
                "s_fib_mod": s_fib_mod,
                "s_shallow": s_shallow,
                "s_cont": s_cont,
                "ba_amp_r": ba_amp_ratio,
                "c_amp": c["amplitude_pct"],
                "c_dir": c["direction"],
                "a_amp": a_amp,
            })

        if not level_scores:
            continue

        n_lv = len(level_scores)

        # Aggregate: sum of per-level scores
        total_score = sum(ls["score"] for ls in level_scores)

        # Weighted direction
        direction_sum = sum(ls["c_dir"] * ls["score"] for ls in level_scores)
        direction = 1 if direction_sum > 0 else -1

        # Direction consensus
        dirs = [ls["c_dir"] for ls in level_scores]
        consensus = abs(sum(dirs)) / len(dirs)

        # Consensus bonus to total score
        if consensus >= 0.9:
            total_score += n_lv * 0.5
        elif consensus < 0.5:
            total_score -= n_lv * 0.3

        # Weighted target amplitude
        w_amp = sum(ls["score"] * abs(ls["c_amp"]) for ls in level_scores if ls["score"] > 0)
        w_total = sum(ls["score"] for ls in level_scores if ls["score"] > 0)
        if w_total <= 0:
            continue
        v_amp = w_amp / w_total
        if v_amp < 0.01:
            continue

        # Simulate at delay=5
        eb = bar_idx + DELAY
        if eb >= n_bars - 50:
            continue

        decay = np.exp(-0.08 * DELAY)
        tp_pct = v_amp * 0.75 * decay
        sl_pct = tp_pct * 0.4
        if tp_pct < 0.01:
            continue

        ep = closes[eb]
        tp_d = ep * tp_pct / 100.0
        sl_d = ep * sl_pct / 100.0
        if tp_d <= 0 or sl_d <= 0:
            continue

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
        pnl_r = 0.0
        reason = "timeout"
        hold = 0

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
                    pnl_r = (sl_price - ep) / sl_d
                    reason = "sl"
                    hold = elapsed
                    break
            else:
                if h >= sl_price:
                    pnl_r = (ep - sl_price) / sl_d
                    reason = "sl"
                    hold = elapsed
                    break

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
                    pnl_r = (tp_price - ep) / sl_d
                    reason = "tp"
                    hold = elapsed
                    break
            else:
                if l <= tp_price:
                    pnl_r = (ep - tp_price) / sl_d
                    reason = "tp"
                    hold = elapsed
                    break
        else:
            pnl = (closes[end_bar] - ep) if direction == 1 else (ep - closes[end_bar])
            pnl_r = pnl / sl_d
            hold = end_bar - eb

        year = int(dates[min(bar_idx, len(dates) - 1)][:4])

        samples.append({
            "score": round(total_score, 2),
            "n_lv": n_lv,
            "consensus": round(consensus, 2),
            "pnl_r": round(pnl_r, 4),
            "reason": reason,
            "year": year,
        })

    print(f"  {pair}_{tf}: {len(samples):,} samples")
    return samples


def pf_wr(data):
    if not data:
        return 0, 0, 0, 0
    p = [d["pnl_r"] for d in data]
    w = [x for x in p if x > 0]
    l = [x for x in p if x <= 0]
    pf = abs(sum(w) / sum(l)) if l and sum(l) != 0 else 999
    return len(p), len(w) / len(p) * 100 if p else 0, np.mean(p), pf


def print_section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def main():
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    tasks = [(p, "H1") for p in pairs]

    all_s = []
    for t in tasks:
        all_s.extend(process_pair_tf(t))

    print(f"\nTotal samples: {len(all_s):,}")

    # 1. Score distribution
    print_section("1. SCORE DISTRIBUTION")
    scores = [s["score"] for s in all_s]
    pcts = np.percentile(scores, [10, 25, 50, 75, 90, 95, 99])
    print(f"  min={min(scores):.1f}  p10={pcts[0]:.1f}  p25={pcts[1]:.1f}  "
          f"median={pcts[2]:.1f}  p75={pcts[3]:.1f}  p90={pcts[4]:.1f}  "
          f"p95={pcts[5]:.1f}  p99={pcts[6]:.1f}  max={max(scores):.1f}")

    # 2. Score deciles -> alpha (THE KEY TEST)
    print_section("2. SCORE DECILES -> ALPHA (monotonic = scoring works)")
    sorted_s = sorted(all_s, key=lambda x: x["score"])
    decile_size = len(sorted_s) // 10
    for di in range(10):
        chunk = sorted_s[di * decile_size:(di + 1) * decile_size] if di < 9 else sorted_s[di * decile_size:]
        n, wr, ar, pf = pf_wr(chunk)
        smin = chunk[0]["score"]
        smax = chunk[-1]["score"]
        print(f"  D{di + 1:>2d} (score {smin:>6.1f} to {smax:>6.1f}): "
              f"n={n:>8,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # 3. Score quintiles for IS vs OOS
    print_section("3. SCORE QUINTILES: IS (<=2018) vs OOS (>2018)")
    quintile_size = len(sorted_s) // 5
    for qi in range(5):
        chunk = sorted_s[qi * quintile_size:(qi + 1) * quintile_size] if qi < 4 else sorted_s[qi * quintile_size:]
        smin = chunk[0]["score"]
        smax = chunk[-1]["score"]
        is_data = [s for s in chunk if 0 < s["year"] <= 2018]
        oos_data = [s for s in chunk if s["year"] > 2018]
        ni, wri, ari, pfi = pf_wr(is_data)
        no, wro, aro, pfo = pf_wr(oos_data)
        print(f"  Q{qi + 1} ({smin:>5.1f}-{smax:>5.1f}):")
        if ni > 0:
            print(f"    IS:  n={ni:>7,}  WR={wri:.1f}%  avgR={ari:.4f}  PF={pfi:.2f}")
        if no > 0:
            print(f"    OOS: n={no:>7,}  WR={wro:.1f}%  avgR={aro:.4f}  PF={pfo:.2f}")

    # 4. Top 20% only
    print_section("4. TOP 20% BY SCORE")
    top20 = sorted_s[int(len(sorted_s) * 0.8):]
    n, wr, ar, pf = pf_wr(top20)
    print(f"  n={n:>8,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")
    print(f"  score range: {top20[0]['score']:.1f} to {top20[-1]['score']:.1f}")
    is_top = [s for s in top20 if 0 < s["year"] <= 2018]
    oos_top = [s for s in top20 if s["year"] > 2018]
    ni, wri, ari, pfi = pf_wr(is_top)
    no, wro, aro, pfo = pf_wr(oos_top)
    if ni > 0:
        print(f"  IS:  n={ni:>7,}  WR={wri:.1f}%  avgR={ari:.4f}  PF={pfi:.2f}")
    if no > 0:
        print(f"  OOS: n={no:>7,}  WR={wro:.1f}%  avgR={aro:.4f}  PF={pfo:.2f}")

    # 5. Top 10%
    print_section("5. TOP 10% BY SCORE")
    top10 = sorted_s[int(len(sorted_s) * 0.9):]
    n, wr, ar, pf = pf_wr(top10)
    print(f"  n={n:>8,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")
    print(f"  score range: {top10[0]['score']:.1f} to {top10[-1]['score']:.1f}")
    is_t10 = [s for s in top10 if 0 < s["year"] <= 2018]
    oos_t10 = [s for s in top10 if s["year"] > 2018]
    ni, wri, ari, pfi = pf_wr(is_t10)
    no, wro, aro, pfo = pf_wr(oos_t10)
    if ni > 0:
        print(f"  IS:  n={ni:>7,}  WR={wri:.1f}%  avgR={ari:.4f}  PF={pfi:.2f}")
    if no > 0:
        print(f"  OOS: n={no:>7,}  WR={wro:.1f}%  avgR={aro:.4f}  PF={pfo:.2f}")

    # 6. Score component contribution analysis
    print_section("6. MARGINAL CONTRIBUTION OF EACH COMPONENT")
    print("  Testing: does adding each criterion improve the scoring?")

    # Baseline: just n_lv
    for nl_min in [1, 3, 5]:
        sub = [s for s in all_s if s["n_lv"] >= nl_min]
        n, wr, ar, pf = pf_wr(sub)
        print(f"  n_lv>={nl_min}: n={n:>8,}  PF={pf:.2f}  avgR={ar:.4f}")

    # Just consensus >= 0.9
    sub = [s for s in all_s if s["consensus"] >= 0.9]
    n, wr, ar, pf = pf_wr(sub)
    print(f"  consensus>=0.9: n={n:>8,}  PF={pf:.2f}  avgR={ar:.4f}")

    # n_lv>=3 + consensus>=0.9
    sub = [s for s in all_s if s["n_lv"] >= 3 and s["consensus"] >= 0.9]
    n, wr, ar, pf = pf_wr(sub)
    print(f"  n_lv>=3 & cons>=0.9: n={n:>8,}  PF={pf:.2f}  avgR={ar:.4f}")

    # Score > median
    med = np.median(scores)
    sub = [s for s in all_s if s["score"] > med]
    n, wr, ar, pf = pf_wr(sub)
    print(f"  score>{med:.1f}(median): n={n:>8,}  PF={pf:.2f}  avgR={ar:.4f}")


if __name__ == "__main__":
    main()
