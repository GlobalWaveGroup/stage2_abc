"""
Deep analysis: what AB conditions produce alpha in C?
Approach: enrich each trade with per-level AB features,
then do conditional analysis at delay=5.

Key question (user framework):
- Constrain C shape -> which AB structures produce it?
- Multi-level resonance (fractal nesting) -> mechanism?
"""
import sys, os, csv, json, numpy as np, pickle
from collections import defaultdict
from multiprocessing import Pool

DATA_DIR = "/home/ubuntu/DataBase/base_kline"
ZIG_DIR = "/home/ubuntu/database2/build_zig_all/cl128"


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

    # Build window-relative lookup
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

        # Collect per-level ABC data
        level_records = {}
        for lv in range(max_level + 1):
            b_list = [e for e in in_by_lv.get(lv, []) if e["end_idx"] == rel_idx]
            c_list = [e for e in out_by_lv.get(lv, []) if e["start_idx"] == rel_idx]
            if not c_list:
                continue

            c = c_list[0]
            rec = {
                "c_amp": c["amplitude_pct"],
                "c_dur": c["duration"],
                "c_mod": c["modulus"],
                "c_slope": c.get("slope", 0),
                "c_dir": c["direction"],
            }

            if b_list:
                b = b_list[0]
                rec["b_amp"] = b["amplitude_pct"]
                rec["b_dur"] = b["duration"]
                rec["b_mod"] = b["modulus"]
                rec["b_slope"] = b.get("slope", 0)

                # Trace A
                b_src_rel = b["start_idx"]
                src_idx = win_bar_to_node.get(win_id, {}).get(b_src_rel, None)
                if src_idx is not None:
                    src_in, _ = edges[src_idx]
                    a_cands = [e for e in src_in
                               if e["level"] == lv and e["end_idx"] == b_src_rel and e["duration"] > 0]
                    if a_cands:
                        a = a_cands[0]
                        rec["a_amp"] = a["amplitude_pct"]
                        rec["a_dur"] = a["duration"]
                        rec["a_mod"] = a["modulus"]
                        rec["a_slope"] = a.get("slope", 0)

                        aa = abs(a["amplitude_pct"])
                        ba = abs(b["amplitude_pct"])
                        if aa > 0:
                            rec["ba_amp_ratio"] = ba / aa
                        if a["duration"] > 0:
                            rec["ba_dur_ratio"] = b["duration"] / a["duration"]
                        if a["modulus"] > 0:
                            rec["ba_mod_ratio"] = b["modulus"] / a["modulus"]
                        if abs(a.get("slope", 0)) > 1e-10:
                            rec["ba_slope_ratio"] = abs(b.get("slope", 0)) / abs(a["slope"])

            level_records[lv] = rec

        if not level_records:
            continue
        n_lv = len(level_records)

        # Aggregate C direction/amplitude
        c_amps = [level_records[lv]["c_amp"] for lv in level_records]
        direction = 1 if sum(c_amps) > 0 else -1

        # Key AB features across levels
        ba_amp_ratios = [level_records[lv].get("ba_amp_ratio", None) for lv in level_records]
        ba_amp_ratios = [x for x in ba_amp_ratios if x is not None]
        ba_slope_ratios = [level_records[lv].get("ba_slope_ratio", None) for lv in level_records]
        ba_slope_ratios = [x for x in ba_slope_ratios if x is not None]
        ba_dur_ratios = [level_records[lv].get("ba_dur_ratio", None) for lv in level_records]
        ba_dur_ratios = [x for x in ba_dur_ratios if x is not None]

        has_abc = sum(1 for lv in level_records if "a_amp" in level_records[lv])
        has_bc = sum(1 for lv in level_records if "b_amp" in level_records[lv])

        # Weighted target C
        w_amp = 0.0
        w_total = 0.0
        for lv, d in level_records.items():
            w = 1.0 + lv * 0.5
            if "ba_slope_ratio" in d and d["ba_slope_ratio"] < 0.3:
                w *= 2.5
            w_amp += w * d["c_amp"]
            w_total += w
        v_amp = abs(w_amp / w_total) if w_total > 0 else 0
        if v_amp < 0.01:
            continue

        # Simulate at delay=5
        eb = bar_idx + DELAY
        if eb >= n_bars - 50:
            continue

        decay = np.exp(-0.08 * DELAY)
        tp_pct = v_amp * 0.75 * decay
        sl_pct = tp_pct * (0.35 + 0.015 * DELAY)
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
            if prog >= 1.0:
                if direction == 1:
                    sl_price = max(sl_price, ep + mf - tp_d * 0.15)
                else:
                    sl_price = min(sl_price, ep - mf + tp_d * 0.15)

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

        # Direction consensus across levels
        dir_signs = []
        for lv in level_records:
            d = level_records[lv]
            if d["c_amp"] > 0:
                dir_signs.append(1)
            else:
                dir_signs.append(-1)
        consensus = abs(sum(dir_signs)) / len(dir_signs) if dir_signs else 0

        samples.append({
            "n_lv": n_lv,
            "has_abc": has_abc,
            "has_bc": has_bc,
            "ba_amp_r_mean": round(np.mean(ba_amp_ratios), 4) if ba_amp_ratios else -1,
            "ba_amp_r_min": round(min(ba_amp_ratios), 4) if ba_amp_ratios else -1,
            "ba_slope_r_mean": round(np.mean(ba_slope_ratios), 4) if ba_slope_ratios else -1,
            "ba_slope_r_min": round(min(ba_slope_ratios), 4) if ba_slope_ratios else -1,
            "ba_dur_r_mean": round(np.mean(ba_dur_ratios), 4) if ba_dur_ratios else -1,
            "v_amp": round(v_amp, 4),
            "consensus": round(consensus, 2),
            "pnl_r": round(pnl_r, 4),
            "reason": reason,
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

    all_samples = []
    for t in tasks:
        all_samples.extend(process_pair_tf(t))

    print(f"\nTotal samples (delay=5): {len(all_samples):,}")

    # 1. BY N_LEVELS
    print_section("1. BY NUMBER OF RESONATING LEVELS")
    for nl in sorted(set(s["n_lv"] for s in all_samples)):
        sub = [s for s in all_samples if s["n_lv"] == nl]
        n, wr, ar, pf = pf_wr(sub)
        if n < 100:
            continue
        print(f"  n_lv={nl}: n={n:>8,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # 2. BY HAS_ABC COUNT
    print_section("2. BY COMPLETE ABC CHAIN COUNT")
    for abc in sorted(set(s["has_abc"] for s in all_samples)):
        sub = [s for s in all_samples if s["has_abc"] == abc]
        n, wr, ar, pf = pf_wr(sub)
        if n < 100:
            continue
        print(f"  abc_chains={abc}: n={n:>8,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # 3. BA_AMP_RATIO
    print_section("3. BY B/A AMPLITUDE RATIO (mean across levels)")
    print("     low ratio = shallow B = strong trend continuation signal")
    valid = [s for s in all_samples if s["ba_amp_r_mean"] >= 0]
    if valid:
        bins = [(0, 0.2), (0.2, 0.382), (0.382, 0.5), (0.5, 0.618),
                (0.618, 0.786), (0.786, 1.0), (1.0, 2.0), (2.0, 100)]
        for lo, hi in bins:
            sub = [s for s in valid if lo <= s["ba_amp_r_mean"] < hi]
            n, wr, ar, pf = pf_wr(sub)
            if n < 100:
                continue
            print(f"  ba_amp [{lo:.3f},{hi:.3f}): n={n:>8,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # 4. BA_SLOPE_RATIO
    print_section("4. BY B/A SLOPE RATIO (mean across levels)")
    print("     low ratio = B is slow/weak = strong A trend")
    valid = [s for s in all_samples if s["ba_slope_r_mean"] >= 0]
    if valid:
        bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8),
                (0.8, 1.0), (1.0, 1.5), (1.5, 3.0), (3.0, 100)]
        for lo, hi in bins:
            sub = [s for s in valid if lo <= s["ba_slope_r_mean"] < hi]
            n, wr, ar, pf = pf_wr(sub)
            if n < 100:
                continue
            print(f"  ba_slope [{lo:.1f},{hi:.1f}): n={n:>8,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # 5. BA_DUR_RATIO (time structure)
    print_section("5. BY B/A DURATION RATIO (mean across levels)")
    print("     low ratio = B is fast = time compression in retracement")
    valid = [s for s in all_samples if s["ba_dur_r_mean"] >= 0]
    if valid:
        bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.0),
                (1.0, 1.5), (1.5, 2.0), (2.0, 100)]
        for lo, hi in bins:
            sub = [s for s in valid if lo <= s["ba_dur_r_mean"] < hi]
            n, wr, ar, pf = pf_wr(sub)
            if n < 100:
                continue
            print(f"  ba_dur [{lo:.1f},{hi:.1f}): n={n:>8,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # 6. DIRECTION CONSENSUS
    print_section("6. BY DIRECTION CONSENSUS ACROSS LEVELS")
    print("     1.0 = all levels agree on direction")
    bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        sub = [s for s in all_samples if lo <= s["consensus"] < hi]
        n, wr, ar, pf = pf_wr(sub)
        if n < 100:
            continue
        print(f"  consensus [{lo:.1f},{hi:.1f}): n={n:>8,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # 7. CROSS: n_lv x ba_slope_ratio
    print_section("7. CROSS: N_LEVELS x BA_SLOPE_RATIO")
    slope_bins = [(0, 0.3, "<0.3"), (0.3, 0.6, "0.3-0.6"),
                  (0.6, 1.0, "0.6-1.0"), (1.0, 100, ">1.0")]
    for nl in [1, 2, 3, 4, 5, 6]:
        parts = []
        for lo, hi, label in slope_bins:
            sub = [s for s in all_samples if s["n_lv"] == nl
                   and s["ba_slope_r_mean"] >= 0 and lo <= s["ba_slope_r_mean"] < hi]
            if len(sub) >= 50:
                n, wr, ar, pf = pf_wr(sub)
                parts.append(f"slope {label}: n={n:>6,} PF={pf:.2f} avgR={ar:.3f}")
            else:
                parts.append(f"slope {label}: n<50")
        print(f"  n_lv={nl}: " + " | ".join(parts))

    # 8. CROSS: n_lv x ba_amp_ratio
    print_section("8. CROSS: N_LEVELS x BA_AMP_RATIO")
    amp_bins = [(0, 0.382, "<0.382"), (0.382, 0.618, "0.382-0.618"),
                (0.618, 1.0, "0.618-1.0"), (1.0, 100, ">1.0")]
    for nl in [1, 2, 3, 4, 5, 6]:
        parts = []
        for lo, hi, label in amp_bins:
            sub = [s for s in all_samples if s["n_lv"] == nl
                   and s["ba_amp_r_mean"] >= 0 and lo <= s["ba_amp_r_mean"] < hi]
            if len(sub) >= 50:
                n, wr, ar, pf = pf_wr(sub)
                parts.append(f"amp {label}: n={n:>6,} PF={pf:.2f} avgR={ar:.3f}")
            else:
                parts.append(f"amp {label}: n<50")
        print(f"  n_lv={nl}: " + " | ".join(parts))

    # 9. COMBINED FILTERS
    print_section("9. COMBINED FILTERS (best conditions)")

    filters = [
        ("n_lv>=3 & slope<0.6 & amp<0.618",
         lambda s: s["n_lv"] >= 3 and 0 <= s["ba_slope_r_mean"] < 0.6 and 0 <= s["ba_amp_r_mean"] < 0.618),
        ("n_lv>=4 & slope<0.6",
         lambda s: s["n_lv"] >= 4 and 0 <= s["ba_slope_r_mean"] < 0.6),
        ("n_lv>=5 & slope<0.6",
         lambda s: s["n_lv"] >= 5 and 0 <= s["ba_slope_r_mean"] < 0.6),
        ("n_lv>=3 & consensus>=0.7",
         lambda s: s["n_lv"] >= 3 and s["consensus"] >= 0.7),
        ("n_lv>=3 & consensus>=0.7 & slope<0.6",
         lambda s: s["n_lv"] >= 3 and s["consensus"] >= 0.7 and 0 <= s["ba_slope_r_mean"] < 0.6),
        ("n_lv>=4 & consensus=1.0 & slope<0.6",
         lambda s: s["n_lv"] >= 4 and s["consensus"] >= 0.99 and 0 <= s["ba_slope_r_mean"] < 0.6),
        ("n_lv>=3 & dur<0.5 & slope<0.6",
         lambda s: s["n_lv"] >= 3 and 0 <= s["ba_dur_r_mean"] < 0.5 and 0 <= s["ba_slope_r_mean"] < 0.6),
    ]

    for label, fn in filters:
        sub = [s for s in all_samples if fn(s)]
        n, wr, ar, pf = pf_wr(sub)
        print(f"  {label}")
        print(f"    n={n:>8,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")


if __name__ == "__main__":
    main()
