"""
Fibonacci hypothesis test:
1. B/A ratio proximity to fib levels -> does it predict C quality?
2. C vector symmetry with A (modulus ratio, centered on B) -> does it exist?
3. Large-level pivots conform MORE to fib -> scale invariance with fib affinity?

Core idea: it's not about ba_ratio being in a range,
it's about ba_ratio being CLOSE TO a specific fib level.
ratio=0.382 and ratio=0.618 are both "good", ratio=0.45 is "bad".
"""
import sys, os, csv, numpy as np, pickle
from collections import defaultdict

DATA_DIR = "/home/ubuntu/DataBase/base_kline"
ZIG_DIR = "/home/ubuntu/database2/build_zig_all/cl128"

# Fibonacci levels (retracement and extension)
FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618, 2.000, 2.618]


def fib_distance(ratio):
    """Minimum distance from ratio to any fib level."""
    return min(abs(ratio - f) for f in FIB_LEVELS)


def fib_nearest(ratio):
    """Nearest fib level."""
    return min(FIB_LEVELS, key=lambda f: abs(ratio - f))


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

        # Per-level ABC extraction
        for lv in range(max_level + 1):
            b_list = [e for e in in_by_lv.get(lv, []) if e["end_idx"] == rel_idx]
            c_list = [e for e in out_by_lv.get(lv, []) if e["start_idx"] == rel_idx]
            if not c_list or not b_list:
                continue

            c = c_list[0]
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

            # Compute ratios
            a_amp = abs(a["amplitude_pct"])
            b_amp = abs(b["amplitude_pct"])
            c_amp = abs(c["amplitude_pct"])
            a_mod = a["modulus"]
            b_mod = b["modulus"]
            c_mod = c["modulus"]
            a_dur = a["duration"]
            b_dur = b["duration"]
            c_dur = c["duration"]

            if a_amp < 1e-6 or a_mod < 1e-6 or a_dur < 1:
                continue

            ba_amp_ratio = b_amp / a_amp
            ba_mod_ratio = b_mod / a_mod
            ba_dur_ratio = b_dur / a_dur

            # C/A ratios (symmetry: is C similar to A in scale?)
            ca_amp_ratio = c_amp / a_amp
            ca_mod_ratio = c_mod / a_mod
            ca_dur_ratio = c_dur / a_dur if a_dur > 0 else 0

            # Fib distances
            fib_dist_amp = fib_distance(ba_amp_ratio)
            fib_dist_mod = fib_distance(ba_mod_ratio)
            fib_nearest_amp = fib_nearest(ba_amp_ratio)

            # C direction: does C continue A's direction? (trend continuation)
            # A goes up, B retraces down, C should go up again
            c_continues = 1 if (a["direction"] == c["direction"]) else 0

            # Trade simulation at delay=5
            eb = bar_idx + DELAY
            if eb >= n_bars - 50:
                continue

            # Use C's actual amplitude as the "truth" for measuring prediction quality
            # But also simulate a trade
            direction = c["direction"]
            tp_pct = a_amp * 0.5 * np.exp(-0.08 * DELAY)  # target: 50% of A
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

            samples.append({
                "lv": lv,
                "max_lv": max_level,
                "ba_amp_r": round(ba_amp_ratio, 4),
                "ba_mod_r": round(ba_mod_ratio, 4),
                "ba_dur_r": round(ba_dur_ratio, 4),
                "ca_amp_r": round(ca_amp_ratio, 4),
                "ca_mod_r": round(ca_mod_ratio, 4),
                "ca_dur_r": round(ca_dur_ratio, 4),
                "fib_dist_amp": round(fib_dist_amp, 4),
                "fib_dist_mod": round(fib_dist_mod, 4),
                "fib_near": fib_nearest_amp,
                "c_continues": c_continues,
                "a_amp": round(a_amp, 4),
                "a_mod": round(a_mod, 4),
                "pnl_r": round(pnl_r, 4),
                "reason": reason,
            })

    print(f"  {pair}_{tf}: {len(samples):,} per-level samples")
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

    print(f"\nTotal per-level ABC samples: {len(all_s):,}")

    # =============================================
    # TEST 1: Fib distance vs alpha
    # =============================================
    print_section("TEST 1: FIB DISTANCE (ba_amp) vs ALPHA")
    print("  Does proximity to fib level predict better C?")
    print("  fib_dist=0 means ratio IS a fib level exactly")
    bins = [(0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.08), (0.08, 0.12), (0.12, 0.20)]
    for lo, hi in bins:
        sub = [s for s in all_s if lo <= s["fib_dist_amp"] < hi]
        n, wr, ar, pf = pf_wr(sub)
        if n < 100:
            continue
        print(f"  fib_dist [{lo:.2f},{hi:.2f}): n={n:>9,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # Also for modulus ratio
    print_section("TEST 1b: FIB DISTANCE (ba_mod) vs ALPHA")
    for lo, hi in bins:
        sub = [s for s in all_s if lo <= s["fib_dist_mod"] < hi]
        n, wr, ar, pf = pf_wr(sub)
        if n < 100:
            continue
        print(f"  fib_dist [{lo:.2f},{hi:.2f}): n={n:>9,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # =============================================
    # TEST 2: Per fib level
    # =============================================
    print_section("TEST 2: ALPHA BY NEAREST FIB LEVEL (ba_amp)")
    print("  Which fib level produces best C?")
    for fib in FIB_LEVELS:
        sub = [s for s in all_s if s["fib_near"] == fib and s["fib_dist_amp"] < 0.05]
        n, wr, ar, pf = pf_wr(sub)
        if n < 100:
            continue
        print(f"  fib={fib:.3f} (dist<0.05): n={n:>9,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # =============================================
    # TEST 3: C/A symmetry - is C actually similar to A?
    # =============================================
    print_section("TEST 3: C/A SYMMETRY (does C mirror A in scale?)")
    print("  ca_amp_ratio ~ 1.0 means C amplitude equals A")
    ca_bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 1.618), (1.618, 3.0), (3.0, 100)]
    for lo, hi in ca_bins:
        sub = [s for s in all_s if lo <= s["ca_amp_r"] < hi]
        n, wr, ar, pf = pf_wr(sub)
        if n < 100:
            continue
        # Also check what % of cases C continues A direction
        cont = sum(1 for s in sub if s["c_continues"]) / len(sub) * 100
        print(f"  ca_amp [{lo:.1f},{hi:.1f}): n={n:>9,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}  C_cont={cont:.0f}%")

    # C/A modulus symmetry
    print_section("TEST 3b: C/A MODULUS SYMMETRY")
    for lo, hi in ca_bins:
        sub = [s for s in all_s if lo <= s["ca_mod_r"] < hi]
        n, wr, ar, pf = pf_wr(sub)
        if n < 100:
            continue
        cont = sum(1 for s in sub if s["c_continues"]) / len(sub) * 100
        print(f"  ca_mod [{lo:.1f},{hi:.1f}): n={n:>9,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}  C_cont={cont:.0f}%")

    # =============================================
    # TEST 4: C direction continuation rate
    # =============================================
    print_section("TEST 4: C DIRECTION CONTINUATION (does C follow A's direction?)")
    print("  If A is up, B retraces, does C go up?")
    cont = [s for s in all_s if s["c_continues"] == 1]
    rev = [s for s in all_s if s["c_continues"] == 0]
    n1, wr1, ar1, pf1 = pf_wr(cont)
    n0, wr0, ar0, pf0 = pf_wr(rev)
    total = len(all_s)
    print(f"  C continues A: n={n1:>9,} ({n1/total*100:.1f}%)  WR={wr1:.1f}%  avgR={ar1:.4f}  PF={pf1:.2f}")
    print(f"  C reverses A:  n={n0:>9,} ({n0/total*100:.1f}%)  WR={wr0:.1f}%  avgR={ar0:.4f}  PF={pf0:.2f}")

    # By level
    print("\n  By fractal level:")
    for lv in range(8):
        sub = [s for s in all_s if s["lv"] == lv]
        if len(sub) < 100:
            continue
        cont_rate = sum(1 for s in sub if s["c_continues"]) / len(sub) * 100
        n, wr, ar, pf = pf_wr(sub)
        print(f"    L{lv}: n={n:>9,}  C_cont={cont_rate:.1f}%  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # =============================================
    # TEST 5: Large level pivots conform MORE to fib?
    # =============================================
    print_section("TEST 5: FIB CONFORMITY BY LEVEL (large level = more fib?)")
    print("  Mean fib_distance by level (lower = closer to fib)")
    for lv in range(8):
        sub = [s for s in all_s if s["lv"] == lv]
        if len(sub) < 100:
            continue
        mean_fd = np.mean([s["fib_dist_amp"] for s in sub])
        median_fd = np.median([s["fib_dist_amp"] for s in sub])
        close_fib = sum(1 for s in sub if s["fib_dist_amp"] < 0.03) / len(sub) * 100
        print(f"    L{lv}: n={len(sub):>9,}  mean_fib_dist={mean_fd:.4f}  "
              f"median={median_fd:.4f}  pct_within_0.03={close_fib:.1f}%")

    # =============================================
    # TEST 6: CROSS fib_distance x level x alpha
    # =============================================
    print_section("TEST 6: CROSS: FIB_PROXIMITY x LEVEL -> ALPHA")
    print("  Does fib proximity matter MORE at higher levels?")
    for lv in [0, 1, 2, 3, 4, 5]:
        lv_data = [s for s in all_s if s["lv"] == lv]
        if len(lv_data) < 200:
            continue
        close = [s for s in lv_data if s["fib_dist_amp"] < 0.03]
        far = [s for s in lv_data if s["fib_dist_amp"] >= 0.08]
        nc, wrc, arc, pfc = pf_wr(close)
        nf, wrf, arf, pff = pf_wr(far)
        if nc >= 50 and nf >= 50:
            print(f"    L{lv}: close_fib(d<0.03) n={nc:>7,} PF={pfc:.2f} avgR={arc:.4f}"
                  f"  |  far_fib(d>0.08) n={nf:>7,} PF={pff:.2f} avgR={arf:.4f}"
                  f"  |  delta_PF={pfc-pff:+.2f}")

    # =============================================
    # TEST 7: fib proximity x n_lv cross
    # =============================================
    print_section("TEST 7: FIB PROXIMITY x MULTI-LEVEL RESONANCE")
    for nlv_lo, nlv_hi, nlv_label in [(1, 2, "1-2"), (3, 4, "3-4"), (5, 10, "5+")]:
        close = [s for s in all_s if nlv_lo <= s["max_lv"] <= nlv_hi and s["fib_dist_amp"] < 0.03]
        far = [s for s in all_s if nlv_lo <= s["max_lv"] <= nlv_hi and s["fib_dist_amp"] >= 0.08]
        nc, wrc, arc, pfc = pf_wr(close)
        nf, wrf, arf, pff = pf_wr(far)
        if nc >= 50 and nf >= 50:
            print(f"  max_lv={nlv_label}: close_fib n={nc:>8,} PF={pfc:.2f} avgR={arc:.4f}"
                  f"  |  far_fib n={nf:>8,} PF={pff:.2f} avgR={arf:.4f}")


if __name__ == "__main__":
    main()
