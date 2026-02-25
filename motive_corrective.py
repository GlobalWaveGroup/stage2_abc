"""
Motive vs Corrective analysis + TP/SL recalibration
=====================================================
Test user's hypothesis:
- A corrective + B motive -> C likely corrective (expected)
- But if C becomes motive -> deviation from expectation -> stronger signal
- The interplay between expectation and deviation IS the market

Classify each leg as motive/corrective using:
- slope = amplitude / duration (speed)
- A leg with higher slope = more motive (driven, impulsive)
- A leg with lower slope = more corrective (grinding, hesitant)

Also: recalibrate TP/SL based on diagnosis findings:
- Current v_amp predicts only ~40% of actual MFE median
- MAE is huge relative to v_amp
- Need to use MODULUS-based targets, not just amplitude
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


def classify_leg(amp, dur, slope):
    """
    Classify a leg as motive or corrective.
    Returns motive_score: higher = more motive/impulsive.
    
    Not binary - continuous score based on slope intensity.
    Slope = amplitude / duration = speed of price change.
    """
    if dur <= 0:
        return 0
    # slope is already amplitude_pct / duration
    # Higher abs(slope) = more motive
    return abs(slope) if slope is not None else abs(amp) / max(dur, 1)


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
            if a_amp < 1e-6: continue

            # Motive scores
            a_slope = classify_leg(a["amplitude_pct"], a["duration"], a.get("slope", None))
            b_slope = classify_leg(b["amplitude_pct"], b["duration"], b.get("slope", None))
            c_slope = classify_leg(c["amplitude_pct"], c["duration"], c.get("slope", None))

            # Relative: is B more motive than A?
            if a_slope > 1e-10:
                ba_slope_ratio = b_slope / a_slope
            else:
                ba_slope_ratio = 1.0

            # Classify pattern
            # A corrective (low slope), B motive (high slope) -> ba_slope_ratio > 1
            # A motive (high slope), B corrective (low slope) -> ba_slope_ratio < 1
            a_type = "motive" if a_slope > b_slope else "corrective"
            b_type = "motive" if b_slope > a_slope else "corrective"

            # What is C actually?
            if c_slope > 1e-10 and a_slope > 1e-10:
                ca_slope_ratio = c_slope / a_slope
            else:
                ca_slope_ratio = 1.0

            c_is_motive_vs_a = c_slope > a_slope
            c_continues_a = (a["direction"] == c["direction"])

            # Scoring (same as before)
            score = 0.5 + lv * 0.5
            ba_amp_ratio = b_amp / a_amp
            ba_mod_ratio = b["modulus"] / a["modulus"] if a["modulus"] > 1e-6 else 1.0
            score += max(0, 2.0 * (1.0 - fib_distance(ba_amp_ratio) / 0.10))
            score += max(0, 1.0 * (1.0 - fib_distance(ba_mod_ratio) / 0.10))
            if ba_amp_ratio < 0.382: score += 1.5
            elif ba_amp_ratio < 0.618: score += 0.8
            elif ba_amp_ratio < 1.0: score += 0.0
            else: score -= 0.5
            score += 1.0 if c_continues_a else -0.5

            level_data.append({
                "lv": lv, "score": score,
                "a_slope": a_slope, "b_slope": b_slope, "c_slope": c_slope,
                "ba_slope_ratio": ba_slope_ratio, "ca_slope_ratio": ca_slope_ratio,
                "a_type": a_type, "b_type": b_type,
                "c_motive_vs_a": c_is_motive_vs_a,
                "c_continues_a": c_continues_a,
                "c_amp": c["amplitude_pct"], "c_dir": c["direction"],
                "c_dur": c["duration"], "c_mod": c["modulus"],
                "a_amp": a_amp, "a_mod": a["modulus"],
                "b_amp": b_amp, "b_mod": b["modulus"],
                "a_dur": a["duration"], "b_dur": b["duration"],
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

        # Aggregate motive/corrective pattern
        # Dominant pattern across levels
        n_a_corr = sum(1 for d in level_data if d["a_type"] == "corrective")
        n_b_mot = sum(1 for d in level_data if d["b_type"] == "motive")
        n_c_mot_vs_a = sum(1 for d in level_data if d["c_motive_vs_a"])

        # Pattern classification
        # "Expected": A corrective + B motive -> C should be corrective
        # "Deviation": A corrective + B motive -> but C is motive (stronger signal)
        a_mostly_corrective = n_a_corr > n_lv / 2
        b_mostly_motive = n_b_mot > n_lv / 2
        c_mostly_motive = n_c_mot_vs_a > n_lv / 2

        if a_mostly_corrective and b_mostly_motive:
            if c_mostly_motive:
                pattern = "deviation"  # C motive when expected corrective
            else:
                pattern = "expected"   # C corrective as expected
        elif not a_mostly_corrective and not b_mostly_motive:
            if c_mostly_motive:
                pattern = "trend_continuation"  # A motive, B corrective, C motive
            else:
                pattern = "trend_weakening"     # A motive, B corrective, C corrective
        else:
            pattern = "mixed"

        # Weighted amplitudes for TP/SL
        w_amp = sum(d["score"] * abs(d["c_amp"]) for d in level_data if d["score"] > 0)
        w_mod = sum(d["score"] * d["c_mod"] for d in level_data if d["score"] > 0)
        w_a_amp = sum(d["score"] * d["a_amp"] for d in level_data if d["score"] > 0)
        w_a_mod = sum(d["score"] * d["a_mod"] for d in level_data if d["score"] > 0)
        w_b_amp = sum(d["score"] * d["b_amp"] for d in level_data if d["score"] > 0)
        w_total = sum(d["score"] for d in level_data if d["score"] > 0)
        if w_total <= 0: continue

        v_amp = w_amp / w_total
        v_mod = w_mod / w_total
        v_a_amp = w_a_amp / w_total
        v_a_mod = w_a_mod / w_total
        v_b_amp = w_b_amp / w_total
        if v_amp < 0.01: continue

        # Mean slope ratios
        mean_ba_slope = np.mean([d["ba_slope_ratio"] for d in level_data])
        mean_ca_slope = np.mean([d["ca_slope_ratio"] for d in level_data])

        eb = bar_idx + DELAY
        if eb >= n_bars - 300: continue
        year = int(dates[min(bar_idx, len(dates) - 1)][:4])

        # Simulate with MULTIPLE TP/SL calibrations
        ep = closes[eb]

        results_by_calibration = {}
        for cal_name, tp_base, sl_base in [
            # Current: v_amp based
            ("v_amp_tight", v_amp * 0.75, v_amp * 0.75 * 0.4),
            # A-amplitude based (SL = fraction of A)
            ("a_amp_sl", v_amp * 0.85, v_a_amp * 0.5),
            # B-amplitude based (SL = B amplitude, natural structure)
            ("b_amp_sl", v_amp * 0.85, v_b_amp * 1.0),
            # Modulus based
            ("modulus", v_mod * 0.75, v_mod * 0.75 * 0.5),
            # Wide: larger TP, wider SL
            ("wide", v_amp * 1.5, v_amp * 0.8),
        ]:
            decay = np.exp(-0.08 * DELAY)
            tp_pct = tp_base * decay
            sl_pct = sl_base * decay
            if tp_pct < 0.005 or sl_pct < 0.005: continue

            tp_d = ep * tp_pct / 100.0
            sl_d = ep * sl_pct / 100.0
            if tp_d <= 0 or sl_d <= 0: continue

            if direction == 1:
                tp_price = ep + tp_d; sl_price = ep - sl_d
            else:
                tp_price = ep - tp_d; sl_price = ep + sl_d

            max_hold = 200
            mf = 0.0; be = False
            end_bar = min(eb + max_hold, n_bars - 1)
            pnl_r = 0.0; reason = "timeout"

            for bar in range(eb + 1, end_bar + 1):
                h = highs[bar]; l = lows[bar]
                fav = (h - ep) if direction == 1 else (ep - l)
                if fav > mf: mf = fav
                prog = mf / tp_d if tp_d > 0 else 0

                if direction == 1:
                    if l <= sl_price:
                        pnl_r = (sl_price - ep) / sl_d; reason = "sl"; break
                else:
                    if h >= sl_price:
                        pnl_r = (ep - sl_price) / sl_d; reason = "sl"; break

                if prog >= 0.25 and not be:
                    sl_price = ep; be = True
                if prog >= 0.50:
                    if direction == 1: sl_price = max(sl_price, ep + mf * 0.40)
                    else: sl_price = min(sl_price, ep - mf * 0.40)

                if direction == 1:
                    if h >= tp_price:
                        pnl_r = (tp_price - ep) / sl_d; reason = "tp"; break
                else:
                    if l <= tp_price:
                        pnl_r = (ep - tp_price) / sl_d; reason = "tp"; break
            else:
                pnl = (closes[end_bar] - ep) if direction == 1 else (ep - closes[end_bar])
                pnl_r = pnl / sl_d

            results_by_calibration[cal_name] = round(pnl_r, 4)

        if not results_by_calibration: continue

        samples.append({
            "score": round(total_score, 2),
            "pattern": pattern,
            "n_lv": n_lv,
            "mean_ba_slope": round(mean_ba_slope, 4),
            "mean_ca_slope": round(mean_ca_slope, 4),
            "v_amp": round(v_amp, 4),
            "v_a_amp": round(v_a_amp, 4),
            "v_b_amp": round(v_b_amp, 4),
            "year": year,
            **{f"pnl_{k}": v for k, v in results_by_calibration.items()},
        })

    print(f"  {pair}_{tf}: {len(samples):,} samples")
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
    # 1. PATTERN DISTRIBUTION AND ALPHA
    # =============================================
    print(f"\n{'='*80}")
    print("  1. MOTIVE/CORRECTIVE PATTERN -> ALPHA")
    print(f"{'='*80}")
    print("  Using v_amp_tight calibration:")
    for pattern in ["expected", "deviation", "trend_continuation", "trend_weakening", "mixed"]:
        sub = [s for s in all_s if s["pattern"] == pattern]
        if not sub: continue
        n, wr, ar, pf = pf_wr([s["pnl_v_amp_tight"] for s in sub])
        print(f"  {pattern:>22}: n={n:>7,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # =============================================
    # 2. TP/SL CALIBRATION COMPARISON
    # =============================================
    print(f"\n{'='*80}")
    print("  2. TP/SL CALIBRATION COMPARISON")
    print(f"{'='*80}")
    cal_keys = ["v_amp_tight", "a_amp_sl", "b_amp_sl", "modulus", "wide"]
    for key in cal_keys:
        col = f"pnl_{key}"
        pnl_list = [s[col] for s in all_s if col in s]
        n, wr, ar, pf = pf_wr(pnl_list)
        print(f"  {key:>15}: n={n:>7,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}  sumR={sum(pnl_list):>10,.1f}")

    # =============================================
    # 3. PATTERN x CALIBRATION CROSS
    # =============================================
    print(f"\n{'='*80}")
    print("  3. BEST CALIBRATION PER PATTERN")
    print(f"{'='*80}")
    for pattern in ["expected", "deviation", "trend_continuation", "trend_weakening", "mixed"]:
        sub = [s for s in all_s if s["pattern"] == pattern]
        if len(sub) < 100: continue
        print(f"\n  Pattern: {pattern} (n={len(sub):,})")
        for key in cal_keys:
            col = f"pnl_{key}"
            pnl_list = [s[col] for s in sub if col in s]
            if not pnl_list: continue
            n, wr, ar, pf = pf_wr(pnl_list)
            print(f"    {key:>15}: WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # =============================================
    # 4. IS vs OOS for best calibration
    # =============================================
    print(f"\n{'='*80}")
    print("  4. IS vs OOS BY CALIBRATION")
    print(f"{'='*80}")
    for key in cal_keys:
        col = f"pnl_{key}"
        is_pnl = [s[col] for s in all_s if col in s and 0 < s["year"] <= 2018]
        oos_pnl = [s[col] for s in all_s if col in s and s["year"] > 2018]
        ni, wri, ari, pfi = pf_wr(is_pnl)
        no, wro, aro, pfo = pf_wr(oos_pnl)
        print(f"  {key:>15}: IS PF={pfi:.2f} avgR={ari:.4f}  |  OOS PF={pfo:.2f} avgR={aro:.4f}")

    # =============================================
    # 5. C slope deviation from expectation as signal
    # =============================================
    print(f"\n{'='*80}")
    print("  5. C SLOPE DEVIATION: ca_slope_ratio as predictor")
    print(f"     ca_slope > 1 = C more motive than A (deviation)")
    print(f"     ca_slope < 1 = C less motive than A (expected)")
    print(f"{'='*80}")
    for lo, hi, label in [(0, 0.3, "<0.3"), (0.3, 0.6, "0.3-0.6"), (0.6, 1.0, "0.6-1.0"),
                          (1.0, 1.5, "1.0-1.5"), (1.5, 3.0, "1.5-3.0"), (3.0, 100, ">3.0")]:
        sub = [s for s in all_s if lo <= s["mean_ca_slope"] < hi]
        if len(sub) < 100: continue
        n, wr, ar, pf = pf_wr([s["pnl_v_amp_tight"] for s in sub])
        print(f"  ca_slope {label:>8}: n={n:>7,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")

    # =============================================
    # 6. B/A slope ratio as entry quality
    # =============================================
    print(f"\n{'='*80}")
    print("  6. B/A SLOPE RATIO: is B more motive than A?")
    print(f"     ba_slope > 1 = B more impulsive (B is driving)")
    print(f"     ba_slope < 1 = B less impulsive (B is correcting)")
    print(f"{'='*80}")
    for lo, hi, label in [(0, 0.3, "<0.3"), (0.3, 0.6, "0.3-0.6"), (0.6, 1.0, "0.6-1.0"),
                          (1.0, 1.5, "1.0-1.5"), (1.5, 3.0, "1.5-3.0"), (3.0, 100, ">3.0")]:
        sub = [s for s in all_s if lo <= s["mean_ba_slope"] < hi]
        if len(sub) < 100: continue
        n, wr, ar, pf = pf_wr([s["pnl_v_amp_tight"] for s in sub])
        print(f"  ba_slope {label:>8}: n={n:>7,}  WR={wr:.1f}%  avgR={ar:.4f}  PF={pf:.2f}")


if __name__ == "__main__":
    main()
