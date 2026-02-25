"""
Multi-Level Vector Aggregation V2
==================================
Key changes from V1:
- NO dedup: every window is an independent sample
- Process ALL nodes in each window (not just unique bar_idx)
- Much more efficient ABC chain extraction
- Relaxed ABC completeness: allow partial chains (B+C without A)
"""

import os
import csv
import json
import numpy as np
import pickle
from collections import defaultdict
from multiprocessing import Pool

DATA_DIR = "/home/ubuntu/DataBase/base_kline"
ZIG_DIR = "/home/ubuntu/database2/build_zig_all/cl128"
OUT_DIR = "/home/ubuntu/stage2_abc/multilevel"

DELAYS = [0, 1, 2, 3, 5, 8, 12]


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


def extract_all_nodes(feats, edges):
    """
    For every node, extract multi-level feature set.
    No dedup — every node in every window is a sample.
    
    For efficiency, build a lookup: (window_id, bar_idx) -> node_idx
    so we can quickly trace A from B's source.
    """
    # Build lookup using WINDOW-RELATIVE index (col2), not global bar_idx (col0)
    # col0 = global bar index (for OHLCV positioning)
    # col2 = window-relative index (matches edge start_idx/end_idx)
    win_bar_to_node = defaultdict(dict)
    for i in range(len(feats)):
        win = int(feats[i, 1])
        rel_idx = int(feats[i, 2])  # window-relative index
        ml = int(feats[i, 5])
        # If multiple nodes at same (win, rel_idx), keep highest max_level
        if rel_idx not in win_bar_to_node[win] or ml > feats[win_bar_to_node[win][rel_idx], 5]:
            win_bar_to_node[win][rel_idx] = i

    samples = []

    for i in range(len(feats)):
        bar_idx = int(feats[i, 0])      # global bar index (for OHLCV)
        win_id = int(feats[i, 1])
        rel_idx = int(feats[i, 2])      # window-relative index (for edge matching)
        is_high = int(feats[i, 3])
        price = float(feats[i, 4])
        max_level = int(feats[i, 5])

        in_edges, out_edges = edges[i]
        if not out_edges:
            continue

        # Group by level
        in_by_lv = defaultdict(list)
        out_by_lv = defaultdict(list)
        for e in in_edges:
            if e["duration"] > 0:
                in_by_lv[e["level"]].append(e)
        for e in out_edges:
            if e["duration"] > 0:
                out_by_lv[e["level"]].append(e)

        level_data = {}
        for lv in range(max_level + 1):
            # Use rel_idx (window-relative) to match edges, NOT bar_idx (global)
            b_list = [e for e in in_by_lv.get(lv, []) if e["end_idx"] == rel_idx]
            c_list = [e for e in out_by_lv.get(lv, []) if e["start_idx"] == rel_idx]

            if not c_list:
                continue

            c = c_list[0]
            ld = {
                "c_amp": c["amplitude_pct"],
                "c_dur": c["duration"],
                "c_mod": c["modulus"],
                "c_theta": c["theta"],
                "c_dir": c["direction"],
                "has_b": False,
                "has_a": False,
            }

            if b_list:
                b = b_list[0]
                ld["b_amp"] = b["amplitude_pct"]
                ld["b_dur"] = b["duration"]
                ld["b_mod"] = b["modulus"]
                ld["b_theta"] = b["theta"]
                ld["b_dir"] = b["direction"]
                ld["b_slope"] = b["slope"]
                ld["has_b"] = True

                # Trace A: b["start_idx"] is window-relative
                b_src_rel = b["start_idx"]
                src_idx = win_bar_to_node.get(win_id, {}).get(b_src_rel, None)
                if src_idx is not None:
                    src_in, _ = edges[src_idx]
                    a_cands = [e for e in src_in
                               if e["level"] == lv and e["end_idx"] == b_src_rel and e["duration"] > 0]
                    if a_cands:
                        a = a_cands[0]
                        ld["a_amp"] = a["amplitude_pct"]
                        ld["a_dur"] = a["duration"]
                        ld["a_mod"] = a["modulus"]
                        ld["a_theta"] = a["theta"]
                        ld["a_dir"] = a["direction"]
                        ld["a_slope"] = a["slope"]
                        ld["has_a"] = True

                        # B/A ratios
                        aa = abs(a["amplitude_pct"])
                        ba = abs(b["amplitude_pct"])
                        if aa > 0:
                            ld["ba_amp_ratio"] = ba / aa
                        if a["duration"] > 0:
                            ld["ba_dur_ratio"] = b["duration"] / a["duration"]
                        if a["modulus"] > 0:
                            ld["ba_mod_ratio"] = b["modulus"] / a["modulus"]
                        if abs(a["slope"]) > 1e-10:
                            ld["ba_slope_ratio"] = abs(b["slope"]) / abs(a["slope"])

            level_data[lv] = ld

        if not level_data:
            continue

        samples.append({
            "bar_idx": bar_idx,
            "win_id": win_id,
            "price": price,
            "max_level": max_level,
            "levels": level_data,
        })

    return samples


def compute_target(sample):
    """Weighted aggregation of per-level C vectors."""
    levels = sample["levels"]
    w_mod = 0.0; w_amp = 0.0; w_dur = 0.0; w_total = 0.0

    for lv, d in levels.items():
        w = 1.0 + lv * 0.5

        if d["has_a"]:
            sr = d.get("ba_slope_ratio", 1.0)
            mr = d.get("ba_mod_ratio", 1.0)
            # Quality: low slope ratio AND low mod ratio = B is weak
            if sr < 0.3 and mr < 0.5:
                w *= 2.5
            elif sr < 0.5:
                w *= 1.8
            elif sr < 0.7:
                w *= 1.3
            elif sr > 1.5:
                w *= 0.6
        elif d["has_b"]:
            w *= 0.7
        else:
            w *= 0.4

        c_mod = abs(d["c_mod"])
        c_amp = d["c_amp"]  # signed
        c_dur = d["c_dur"]

        w_mod += w * c_mod
        w_amp += w * c_amp
        w_dur += w * c_dur
        w_total += w

    if w_total <= 0:
        return None

    return {
        "v_mod": w_mod / w_total,
        "v_amp": w_amp / w_total,
        "v_dur": w_dur / w_total,
        "direction": 1 if w_amp > 0 else -1,
        "n_levels": len(levels),
        "w_total": w_total,
    }


def simulate_trade(entry_bar, direction, tp_pct, sl_pct, max_hold,
                   highs, lows, closes):
    if entry_bar >= len(closes) - 1:
        return None
    ep = closes[entry_bar]
    tp_d = ep * tp_pct / 100.0
    sl_d = ep * sl_pct / 100.0
    if tp_d <= 0 or sl_d <= 0:
        return None

    if direction == 1:
        tp = ep + tp_d; sl = ep - sl_d
    else:
        tp = ep - tp_d; sl = ep + sl_d

    mf = 0.0; be = False
    end = min(entry_bar + max_hold, len(closes) - 1)

    for bar in range(entry_bar + 1, end + 1):
        h = highs[bar]; l = lows[bar]
        fav = (h - ep) if direction == 1 else (ep - l)
        if fav > mf:
            mf = fav
        prog = mf / tp_d if tp_d > 0 else 0
        elapsed = bar - entry_bar

        # SL
        if direction == 1:
            if l <= sl:
                return (sl - ep) / sl_d, "sl", elapsed
        else:
            if h >= sl:
                return (ep - sl) / sl_d, "sl", elapsed

        # Dynamic
        if prog >= 0.25 and not be:
            sl = ep; be = True
        if prog >= 0.50:
            if direction == 1:
                sl = max(sl, ep + mf * 0.40)
            else:
                sl = min(sl, ep - mf * 0.40)
        if prog >= 1.0:
            if direction == 1:
                sl = max(sl, ep + mf - tp_d * 0.15)
            else:
                sl = min(sl, ep - mf + tp_d * 0.15)
        if elapsed > max_hold * 0.4 and prog < 0.35 and mf > 0:
            if direction == 1:
                tp = min(tp, ep + mf * 1.05)
            else:
                tp = max(tp, ep - mf * 1.05)

        # TP
        if direction == 1:
            if h >= tp:
                return (tp - ep) / sl_d, "tp", elapsed
        else:
            if l <= tp:
                return (ep - tp) / sl_d, "tp", elapsed

    pnl = (closes[end] - ep) if direction == 1 else (ep - closes[end])
    return pnl / sl_d, "timeout", end - entry_bar


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

    samples = extract_all_nodes(feats, edges)

    results = []
    for s in samples:
        bar_idx = s["bar_idx"]
        if bar_idx < 50 or bar_idx >= n_bars - 50:
            continue

        tgt = compute_target(s)
        if tgt is None:
            continue

        v_amp = abs(tgt["v_amp"])
        v_mod = tgt["v_mod"]
        v_dur = tgt["v_dur"]
        direction = tgt["direction"]

        if v_amp < 0.01:
            continue

        year = int(dates[min(bar_idx, len(dates)-1)][:4])

        for delay in DELAYS:
            eb = bar_idx + delay
            if eb >= n_bars - 50:
                continue

            decay = np.exp(-0.08 * delay)
            tp_pct = v_amp * 0.75 * decay
            sl_pct = tp_pct * (0.35 + 0.015 * delay)

            if tp_pct < 0.01:
                continue

            max_hold = max(int(v_dur * 5), 200)

            r = simulate_trade(eb, direction, tp_pct, sl_pct, max_hold,
                               highs, lows, closes)
            if r is None:
                continue

            pnl_r, reason, hold = r
            results.append({
                "pair": pair, "tf": tf, "bar": bar_idx, "year": year,
                "delay": delay, "dir": direction,
                "n_lv": tgt["n_levels"], "v_mod": round(v_mod, 4),
                "v_amp": round(v_amp, 4), "v_dur": round(v_dur, 1),
                "w": round(tgt["w_total"], 2),
                "tp": round(tp_pct, 4), "sl": round(sl_pct, 4),
                "pnl_r": round(pnl_r, 4), "reason": reason, "hold": hold,
            })

    print(f"  {pair}_{tf}: {len(samples)} nodes -> {len(results)} trades")
    return results


def analyze(results):
    print(f"\n{'='*80}")
    print(f"  RESULTS BY DELAY")
    print(f"{'='*80}")

    for delay in DELAYS:
        t = [r for r in results if r["delay"] == delay]
        if not t: continue
        p = [r["pnl_r"] for r in t]
        w = [x for x in p if x > 0]
        l = [x for x in p if x <= 0]
        pf = abs(sum(w)/sum(l)) if l and sum(l) != 0 else 999
        print(f"  d={delay:>2d}: n={len(p):>8,}  WR={len(w)/len(p)*100:.1f}%  "
              f"avgR={np.mean(p):.4f}  PF={pf:.2f}")

    print(f"\n{'='*80}")
    print(f"  BY N_LEVELS (delay=5)")
    print(f"{'='*80}")
    d5 = [r for r in results if r["delay"] == 5]
    for nl in sorted(set(r["n_lv"] for r in d5)):
        t = [r for r in d5 if r["n_lv"] == nl]
        if len(t) < 100: continue
        p = [r["pnl_r"] for r in t]
        w = [x for x in p if x > 0]
        l = [x for x in p if x <= 0]
        pf = abs(sum(w)/sum(l)) if l and sum(l) != 0 else 999
        print(f"  n_lv={nl}: n={len(p):>7,}  WR={len(w)/len(p)*100:.1f}%  "
              f"avgR={np.mean(p):.4f}  PF={pf:.2f}")

    print(f"\n{'='*80}")
    print(f"  IS vs OOS BY DELAY")
    print(f"{'='*80}")
    for delay in DELAYS:
        is_t = [r["pnl_r"] for r in results if r["delay"]==delay and 0<r["year"]<=2018]
        oos_t = [r["pnl_r"] for r in results if r["delay"]==delay and r["year"]>2018]
        if len(is_t)<100 or len(oos_t)<100: continue
        is_w = [x for x in is_t if x>0]; is_l = [x for x in is_t if x<=0]
        oos_w = [x for x in oos_t if x>0]; oos_l = [x for x in oos_t if x<=0]
        is_pf = abs(sum(is_w)/sum(is_l)) if is_l and sum(is_l)!=0 else 999
        oos_pf = abs(sum(oos_w)/sum(oos_l)) if oos_l and sum(oos_l)!=0 else 999
        print(f"  d={delay:>2d}: IS n={len(is_t):>7,} WR={len(is_w)/len(is_t)*100:.1f}% "
              f"avgR={np.mean(is_t):.4f} PF={is_pf:.2f} | "
              f"OOS n={len(oos_t):>6,} WR={len(oos_w)/len(oos_t)*100:.1f}% "
              f"avgR={np.mean(oos_t):.4f} PF={oos_pf:.2f}")

    # By weight (quality) quintiles at delay=5
    print(f"\n{'='*80}")
    print(f"  BY QUALITY WEIGHT QUINTILES (delay=5)")
    print(f"{'='*80}")
    d5 = sorted([r for r in results if r["delay"]==5], key=lambda x: x["w"])
    if len(d5) >= 500:
        qs = len(d5) // 5
        for qi in range(5):
            chunk = d5[qi*qs:(qi+1)*qs] if qi < 4 else d5[qi*qs:]
            p = [r["pnl_r"] for r in chunk]
            w = [x for x in p if x > 0]
            l = [x for x in p if x <= 0]
            pf = abs(sum(w)/sum(l)) if l and sum(l)!=0 else 999
            wmin = chunk[0]["w"]; wmax = chunk[-1]["w"]
            print(f"  Q{qi+1} (w={wmin:.1f}-{wmax:.1f}): n={len(p):>6,}  "
                  f"WR={len(w)/len(p)*100:.1f}%  avgR={np.mean(p):.4f}  PF={pf:.2f}")


def main():
    print("="*80)
    print("  MULTI-LEVEL V2: No dedup, all windows")
    print("="*80)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Find available pairs
    tasks = []
    for tf in ["H1"]:
        tf_path = os.path.join(ZIG_DIR, tf)
        if not os.path.isdir(tf_path):
            continue
        for fname in sorted(os.listdir(tf_path)):
            if not fname.endswith(".npz"):
                continue
            pair = fname.replace(f"_{tf}.npz", "")
            if os.path.exists(os.path.join(DATA_DIR, f"{pair}_{tf}.csv")):
                tasks.append((pair, tf))

    print(f"Tasks: {len(tasks)}")

    all_results = []
    with Pool(30) as pool:
        for batch in pool.imap_unordered(process_pair_tf, tasks):
            all_results.extend(batch)

    print(f"\nTotal: {len(all_results):,} trades")

    if all_results:
        analyze(all_results)

    print(f"\n{'='*80}")
    print("  COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
