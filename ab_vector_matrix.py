"""
AB Vector Relationship Matrix -> C Distribution
=================================================
No preconception about which is trend/correction.
A and B are just two consecutive legs. Their RELATIVE properties
determine the probability distribution of C.

Key relative features (all continuous ratios):
- amp_ratio = |B_amp| / |A_amp|  (who moved more in price?)
- dur_ratio = B_dur / A_dur  (who took longer?)  
- slope_ratio = |B_slope| / |A_slope|  (who moved faster?)
- mod_ratio = B_mod / A_mod  (who had bigger vector modulus?)

These 4 ratios define a point in the AB relationship space.
We want to know: given this point, what does C look like?
And critically: what TP/SL calibration works best?

Also: the diagnosis showed MFE median = 2.45x predicted C,
MAE median = 2.46x. This suggests v_amp is too small as reference.
Need to understand WHY and recalibrate.
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


def process_pair(pair, tf):
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
        if bar_idx < 50 or bar_idx >= n_bars - 300:
            continue
        in_edges_i, out_edges_i = edges[i]
        if not out_edges_i:
            continue

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
            if not c_list or not b_list:
                continue
            c = c_list[0]; b = b_list[0]
            b_src_rel = b["start_idx"]
            src_idx = win_bar_to_node.get(win_id, {}).get(b_src_rel, None)
            if src_idx is None: continue
            src_in, _ = edges[src_idx]
            a_cands = [e for e in src_in if e["level"] == lv and e["end_idx"] == b_src_rel and e["duration"] > 0]
            if not a_cands: continue
            a = a_cands[0]

            a_amp = abs(a["amplitude_pct"]); b_amp = abs(b["amplitude_pct"])
            c_amp = abs(c["amplitude_pct"])
            a_dur = a["duration"]; b_dur = b["duration"]; c_dur = c["duration"]
            a_mod = a["modulus"]; b_mod = b["modulus"]; c_mod = c["modulus"]

            if a_amp < 1e-6 or a_dur < 1 or a_mod < 1e-6:
                continue

            a_slope = a_amp / a_dur
            b_slope = b_amp / max(b_dur, 1)
            c_slope = c_amp / max(c_dur, 1)

            # The 4 key ratios (AB relationship)
            amp_r = b_amp / a_amp
            dur_r = b_dur / a_dur
            slope_r = b_slope / a_slope if a_slope > 1e-10 else 1.0
            mod_r = b_mod / a_mod

            # C/A ratios (what C actually did relative to A)
            ca_amp_r = c_amp / a_amp
            ca_dur_r = c_dur / a_dur if a_dur > 0 else 1.0
            ca_slope_r = c_slope / a_slope if a_slope > 1e-10 else 1.0
            ca_mod_r = c_mod / a_mod

            # C/B ratios (what C did relative to B)
            cb_amp_r = c_amp / b_amp if b_amp > 1e-6 else 1.0
            cb_slope_r = c_slope / b_slope if b_slope > 1e-10 else 1.0

            c_continues_a = (a["direction"] == c["direction"])

            # Scoring
            score = 0.5 + lv * 0.5
            score += max(0, 2.0 * (1.0 - fib_distance(amp_r) / 0.10))
            score += max(0, 1.0 * (1.0 - fib_distance(mod_r) / 0.10))
            if amp_r < 0.382: score += 1.5
            elif amp_r < 0.618: score += 0.8
            elif amp_r < 1.0: score += 0.0
            else: score -= 0.5
            score += 1.0 if c_continues_a else -0.5

            level_data.append({
                "lv": lv, "score": score,
                "amp_r": amp_r, "dur_r": dur_r, "slope_r": slope_r, "mod_r": mod_r,
                "ca_amp_r": ca_amp_r, "ca_dur_r": ca_dur_r,
                "ca_slope_r": ca_slope_r, "ca_mod_r": ca_mod_r,
                "cb_amp_r": cb_amp_r, "cb_slope_r": cb_slope_r,
                "c_continues_a": c_continues_a,
                "c_amp": c["amplitude_pct"], "c_dir": c["direction"],
                "c_dur": c_dur, "c_mod": c_mod,
                "a_amp": a_amp, "b_amp": b_amp,
                "a_dur": a_dur, "b_dur": b_dur,
                "a_mod": a_mod, "b_mod": b_mod,
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

        # Weighted means of AB ratios across levels
        ws = [d["score"] for d in level_data if d["score"] > 0]
        w_total = sum(ws)
        if w_total <= 0: continue

        def wmean(key):
            return sum(d["score"] * d[key] for d in level_data if d["score"] > 0) / w_total

        v_amp = wmean("c_amp")
        # For TP/SL: use A amplitude as reference scale
        v_a_amp = wmean("a_amp")
        v_b_amp = wmean("b_amp")
        v_a_mod = wmean("a_mod")
        v_c_mod = wmean("c_mod")

        if abs(v_amp) < 0.01: continue

        eb = bar_idx + DELAY
        if eb >= n_bars - 300: continue
        year = int(dates[min(bar_idx, len(dates) - 1)][:4])

        ep = closes[eb]

        # ============================================
        # MULTIPLE TP/SL calibrations
        # ============================================
        decay = np.exp(-0.08 * DELAY)
        calibrations = {}

        # Cal 1: Current (v_amp based, tight)
        calibrations["curr"] = (abs(v_amp) * 0.75 * decay, abs(v_amp) * 0.75 * 0.4 * decay)

        # Cal 2: A-scale TP, B-scale SL
        # TP = fraction of A (expect C to be proportional to A)
        # SL = B amplitude (if price goes beyond B, structure invalidated)
        calibrations["a_tp_b_sl"] = (v_a_amp * 0.618 * decay, v_b_amp * 1.0 * decay)

        # Cal 3: A-scale both
        calibrations["a_scale"] = (v_a_amp * 0.618 * decay, v_a_amp * 0.382 * decay)

        # Cal 4: Modulus based
        calibrations["mod"] = (v_a_mod * 0.618 * decay, v_a_mod * 0.382 * decay)

        # Cal 5: Wide (diagnosis showed MFE >> predicted)
        calibrations["wide"] = (abs(v_amp) * 2.0 * decay, abs(v_amp) * 1.0 * decay)

        # Cal 6: B as SL, C as TP (natural structure)
        calibrations["struct"] = (abs(v_amp) * 1.0 * decay, v_b_amp * 0.8 * decay)

        results = {}
        for cal_name, (tp_pct, sl_pct) in calibrations.items():
            if tp_pct < 0.005 or sl_pct < 0.005: continue
            tp_d = ep * tp_pct / 100.0
            sl_d = ep * sl_pct / 100.0
            if tp_d <= 0 or sl_d <= 0: continue

            if direction == 1:
                tp_price = ep + tp_d; sl_price = ep - sl_d
            else:
                tp_price = ep - tp_d; sl_price = ep + sl_d

            max_hold = 200; mf = 0.0; be = False
            end_bar = min(eb + max_hold, n_bars - 1)
            pnl_r = 0.0

            for bar in range(eb + 1, end_bar + 1):
                h = highs[bar]; l = lows[bar]
                fav = (h - ep) if direction == 1 else (ep - l)
                if fav > mf: mf = fav
                prog = mf / tp_d if tp_d > 0 else 0

                if direction == 1:
                    if l <= sl_price: pnl_r = (sl_price - ep) / sl_d; break
                else:
                    if h >= sl_price: pnl_r = (ep - sl_price) / sl_d; break

                if prog >= 0.25 and not be: sl_price = ep; be = True
                if prog >= 0.50:
                    if direction == 1: sl_price = max(sl_price, ep + mf * 0.40)
                    else: sl_price = min(sl_price, ep - mf * 0.40)

                if direction == 1:
                    if h >= tp_price: pnl_r = (tp_price - ep) / sl_d; break
                else:
                    if l <= tp_price: pnl_r = (ep - tp_price) / sl_d; break
            else:
                pnl = (closes[end_bar] - ep) if direction == 1 else (ep - closes[end_bar])
                pnl_r = pnl / sl_d

            results[cal_name] = round(pnl_r, 4)

        if not results: continue

        samples.append({
            "score": round(total_score, 2), "n_lv": n_lv, "year": year,
            "amp_r": round(wmean("amp_r"), 4),
            "dur_r": round(wmean("dur_r"), 4),
            "slope_r": round(wmean("slope_r"), 4),
            "mod_r": round(wmean("mod_r"), 4),
            "ca_amp_r": round(wmean("ca_amp_r"), 4),
            "ca_slope_r": round(wmean("ca_slope_r"), 4),
            "cb_amp_r": round(wmean("cb_amp_r"), 4),
            "cb_slope_r": round(wmean("cb_slope_r"), 4),
            **{f"pnl_{k}": v for k, v in results.items()},
        })

    print(f"  {pair}_{tf}: {len(samples):,}")
    return samples


def pf_wr(pnl_list):
    if not pnl_list: return 0, 0, 0, 0
    w = [x for x in pnl_list if x > 0]
    l = [x for x in pnl_list if x <= 0]
    pf = abs(sum(w) / sum(l)) if l and sum(l) != 0 else 999
    return len(pnl_list), len(w) / len(pnl_list) * 100, np.mean(pnl_list), pf


def main():
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    all_s = []
    for pair in pairs:
        all_s.extend(process_pair(pair, "H1"))
    print(f"\nTotal: {len(all_s):,}")

    # =============================================
    # 1. TP/SL CALIBRATION HEAD-TO-HEAD
    # =============================================
    print(f"\n{'='*80}")
    print("  1. TP/SL CALIBRATION COMPARISON")
    print(f"{'='*80}")
    cal_keys = ["curr", "a_tp_b_sl", "a_scale", "mod", "wide", "struct"]
    cal_desc = {
        "curr": "current (v_amp*0.75, SL=40%TP)",
        "a_tp_b_sl": "TP=0.618*A, SL=1.0*B",
        "a_scale": "TP=0.618*A, SL=0.382*A",
        "mod": "TP=0.618*A_mod, SL=0.382*A_mod",
        "wide": "TP=2.0*v_amp, SL=1.0*v_amp",
        "struct": "TP=1.0*v_amp, SL=0.8*B",
    }
    for key in cal_keys:
        col = f"pnl_{key}"
        pnl = [s[col] for s in all_s if col in s]
        n, wr, ar, pf = pf_wr(pnl)
        is_pnl = [s[col] for s in all_s if col in s and 0 < s["year"] <= 2018]
        oos_pnl = [s[col] for s in all_s if col in s and s["year"] > 2018]
        _, _, _, pf_is = pf_wr(is_pnl)
        _, _, _, pf_oos = pf_wr(oos_pnl)
        desc = cal_desc.get(key, key)
        print(f"  {desc}")
        print(f"    n={n:>7,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}  IS_PF={pf_is:.2f}  OOS_PF={pf_oos:.2f}")

    # =============================================
    # 2. AB RELATIONSHIP MATRIX -> C
    # =============================================
    # Using best calibration from above (we'll pick after seeing results)
    # For now use "curr" as baseline

    print(f"\n{'='*80}")
    print("  2. AB VECTOR RELATIONSHIP -> ALPHA")
    print("     No preconception about trend/correction")
    print(f"{'='*80}")

    # 2a. amp_ratio: B bigger or smaller than A?
    print("\n  2a. B/A amplitude ratio (who moved more?)")
    for lo, hi, label in [(0, 0.382, "B<<A"), (0.382, 0.618, "B<A"),
                          (0.618, 1.0, "B~A"), (1.0, 1.618, "B>A"),
                          (1.618, 3.0, "B>>A"), (3.0, 100, "B>>>A")]:
        sub = [s["pnl_curr"] for s in all_s if lo <= s["amp_r"] < hi and "pnl_curr" in s]
        n, wr, ar, pf = pf_wr(sub)
        if n < 100: continue
        print(f"    {label:>8} [{lo:.3f},{hi:.3f}): n={n:>7,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # 2b. dur_ratio: B faster or slower than A?
    print("\n  2b. B/A duration ratio (who took longer?)")
    for lo, hi, label in [(0, 0.382, "B fast"), (0.382, 0.618, "B moderate"),
                          (0.618, 1.0, "B~A"), (1.0, 1.618, "B slower"),
                          (1.618, 3.0, "B slow"), (3.0, 100, "B v.slow")]:
        sub = [s["pnl_curr"] for s in all_s if lo <= s["dur_r"] < hi and "pnl_curr" in s]
        n, wr, ar, pf = pf_wr(sub)
        if n < 100: continue
        print(f"    {label:>12} [{lo:.3f},{hi:.3f}): n={n:>7,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # 2c. slope_ratio: B faster or slower speed?
    print("\n  2c. B/A slope ratio (who moved faster per bar?)")
    for lo, hi, label in [(0, 0.3, "B much slower"), (0.3, 0.6, "B slower"),
                          (0.6, 1.0, "B~A"), (1.0, 1.5, "B faster"),
                          (1.5, 3.0, "B much faster"), (3.0, 100, "B explosive")]:
        sub = [s["pnl_curr"] for s in all_s if lo <= s["slope_r"] < hi and "pnl_curr" in s]
        n, wr, ar, pf = pf_wr(sub)
        if n < 100: continue
        print(f"    {label:>15} [{lo:.1f},{hi:.1f}): n={n:>7,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # =============================================
    # 3. WHAT C ACTUALLY DID vs A and B
    # =============================================
    print(f"\n{'='*80}")
    print("  3. WHAT C ACTUALLY DID (C/A and C/B ratios)")
    print(f"{'='*80}")

    print("\n  3a. C/A amplitude ratio (C compared to A)")
    for lo, hi, label in [(0, 0.382, "C<<A"), (0.382, 0.618, "C<A"),
                          (0.618, 1.0, "C~A"), (1.0, 1.618, "C>A"),
                          (1.618, 3.0, "C>>A"), (3.0, 100, "C>>>A")]:
        sub = [s for s in all_s if lo <= s["ca_amp_r"] < hi and "pnl_curr" in s]
        if len(sub) < 100: continue
        n, wr, ar, pf = pf_wr([s["pnl_curr"] for s in sub])
        # Also: what % of these trades had C faster than A?
        c_faster = sum(1 for s in sub if s["ca_slope_r"] > 1.0) / len(sub) * 100
        print(f"    {label:>8}: n={n:>7,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}  C_faster_than_A={c_faster:.0f}%")

    print("\n  3b. C/B amplitude ratio (C compared to B)")
    for lo, hi, label in [(0, 0.382, "C<<B"), (0.382, 0.618, "C<B"),
                          (0.618, 1.0, "C~B"), (1.0, 1.618, "C>B"),
                          (1.618, 3.0, "C>>B"), (3.0, 100, "C>>>B")]:
        sub = [s for s in all_s if lo <= s["cb_amp_r"] < hi and "pnl_curr" in s]
        if len(sub) < 100: continue
        n, wr, ar, pf = pf_wr([s["pnl_curr"] for s in sub])
        print(f"    {label:>8}: n={n:>7,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # =============================================
    # 4. CROSS: amp_ratio x slope_ratio (the 2D AB space)
    # =============================================
    print(f"\n{'='*80}")
    print("  4. 2D AB SPACE: amp_ratio x slope_ratio")
    print(f"{'='*80}")
    amp_bins = [(0, 0.618, "B<A"), (0.618, 1.618, "B~A"), (1.618, 100, "B>A")]
    slope_bins = [(0, 0.6, "B slower"), (0.6, 1.5, "B~A speed"), (1.5, 100, "B faster")]
    for amp_lo, amp_hi, amp_label in amp_bins:
        for sl_lo, sl_hi, sl_label in slope_bins:
            sub = [s["pnl_curr"] for s in all_s
                   if amp_lo <= s["amp_r"] < amp_hi and sl_lo <= s["slope_r"] < sl_hi
                   and "pnl_curr" in s]
            n, wr, ar, pf = pf_wr(sub)
            if n < 100: continue
            print(f"  {amp_label:>5} + {sl_label:>12}: n={n:>7,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # =============================================
    # 5. BEST CALIBRATION per AB type
    # =============================================
    print(f"\n{'='*80}")
    print("  5. BEST CALIBRATION PER AB RELATIONSHIP TYPE")
    print(f"{'='*80}")
    ab_types = [
        ("B<<A & B slower", lambda s: s["amp_r"] < 0.618 and s["slope_r"] < 0.6),
        ("B<<A & B faster", lambda s: s["amp_r"] < 0.618 and s["slope_r"] >= 1.5),
        ("B~A", lambda s: 0.618 <= s["amp_r"] < 1.618 and 0.6 <= s["slope_r"] < 1.5),
        ("B>A & B slower", lambda s: s["amp_r"] >= 1.618 and s["slope_r"] < 0.6),
        ("B>A & B faster", lambda s: s["amp_r"] >= 1.618 and s["slope_r"] >= 1.5),
    ]
    for label, fn in ab_types:
        sub = [s for s in all_s if fn(s)]
        if len(sub) < 100: continue
        print(f"\n  {label} (n={len(sub):,}):")
        for key in cal_keys:
            col = f"pnl_{key}"
            pnl = [s[col] for s in sub if col in s]
            n, wr, ar, pf = pf_wr(pnl)
            if n < 50: continue
            print(f"    {cal_desc.get(key,''):>35}: PF={pf:.2f}  avgR={ar:.4f}  WR={wr:.1f}%")


if __name__ == "__main__":
    main()
