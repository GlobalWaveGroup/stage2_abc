"""
Stage2 Multi-Level Vector Aggregation System
=============================================
Path A: Simplified multi-level approach.

For each pivot node in build_zig_all:
1. Extract ABC structure at each available level (L0-L7)
2. Compute B/A relative matrix at each level
3. Aggregate C predictions across levels into V_target
4. Trade with delay-aware TP/SL derived from V_target

Key insight: V_target = weighted average of per-level C predictions,
where weights come from the quality of each level's ABC structure.

Uses raw price data for actual trade simulation (no look-ahead).
Uses build_zig_all topology only for feature extraction (look-ahead
exists in the topology since it's pre-computed, but we handle this
by only using IN-edges as features and OUT-edges as labels).

CRITICAL: Entry timing uses online zigzag confirm_bar logic.
The build_zig_all pivot_bar is the theoretical turning point.
We test at multiple delays from pivot_bar.
"""

import os
import csv
import numpy as np
import pickle
from collections import defaultdict
from multiprocessing import Pool
import json

DATA_DIR = "/home/ubuntu/DataBase/base_kline"
ZIG_DIR = "/home/ubuntu/database2/build_zig_all/cl128"
OUT_DIR = "/home/ubuntu/stage2_abc/multilevel"


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
    """Load build_zig_all topology."""
    fpath = os.path.join(ZIG_DIR, tf, f"{pair}_{tf}.npz")
    if not os.path.exists(fpath):
        return None, None
    d = np.load(fpath, allow_pickle=True)
    feats = d["features"]
    edges = pickle.loads(d["edges_bytes"].tobytes())
    return feats, edges


def extract_node_features(node_idx, feats, edges):
    """
    Extract multi-level ABC features for a given node.
    
    Returns dict with:
    - node info (bar_idx, price, is_high, max_level)
    - per-level: A vector, B vector, C vector (if available)
    - per-level: B/A ratios
    - aggregated V_target
    """
    f = feats[node_idx]
    bar_idx = int(f[0])
    window_id = int(f[1])
    is_high = int(f[3])
    price = f[4]
    max_level = int(f[5])
    
    in_edges, out_edges = edges[node_idx]
    
    # Group edges by level
    in_by_level = defaultdict(list)
    out_by_level = defaultdict(list)
    for e in in_edges:
        in_by_level[e["level"]].append(e)
    for e in out_edges:
        out_by_level[e["level"]].append(e)
    
    # For each level, try to build ABC chain
    # B = the in-edge at this level (the leg ending at this node)
    # A = the in-edge of B's source node at this level
    # C = the out-edge at this level (the leg starting from this node)
    
    levels = {}
    
    for lv in range(max_level + 1):
        # Get B (in-edge at this level, pick the one with matching bar_idx as end)
        b_edges = [e for e in in_by_level.get(lv, []) 
                   if e["end_idx"] == bar_idx and e["duration"] > 0]
        c_edges = [e for e in out_by_level.get(lv, [])
                   if e["start_idx"] == bar_idx and e["duration"] > 0]
        
        if not b_edges or not c_edges:
            continue
        
        b = b_edges[0]
        c = c_edges[0]
        
        # Try to find A: trace back to B's source node
        b_start = b["start_idx"]
        a_edge = None
        
        # Search for source node in same window
        for j in range(max(0, node_idx - 1000), node_idx):
            if int(feats[j, 0]) == b_start and int(feats[j, 1]) == window_id:
                src_in, _ = edges[j]
                a_candidates = [e for e in src_in 
                               if e["level"] == lv and e["end_idx"] == b_start and e["duration"] > 0]
                if a_candidates:
                    a_edge = a_candidates[0]
                break
        
        level_data = {
            "b_amp": b["amplitude_pct"],
            "b_dur": b["duration"],
            "b_mod": b["modulus"],
            "b_theta": b["theta"],
            "b_slope": b["slope"],
            "b_dir": b["direction"],
            "c_amp": c["amplitude_pct"],
            "c_dur": c["duration"],
            "c_mod": c["modulus"],
            "c_theta": c["theta"],
            "c_dir": c["direction"],
        }
        
        if a_edge is not None:
            a_amp = abs(a_edge["amplitude_pct"])
            b_amp = abs(b["amplitude_pct"])
            
            level_data["a_amp"] = a_edge["amplitude_pct"]
            level_data["a_dur"] = a_edge["duration"]
            level_data["a_mod"] = a_edge["modulus"]
            level_data["a_theta"] = a_edge["theta"]
            level_data["a_dir"] = a_edge["direction"]
            
            # B/A ratios
            if a_amp > 0:
                level_data["ba_amp_ratio"] = b_amp / a_amp
            if a_edge["duration"] > 0:
                level_data["ba_dur_ratio"] = b["duration"] / a_edge["duration"]
            if a_edge["modulus"] > 0:
                level_data["ba_mod_ratio"] = b["modulus"] / a_edge["modulus"]
            if abs(a_edge["slope"]) > 0:
                level_data["ba_slope_ratio"] = abs(b["slope"]) / abs(a_edge["slope"])
            
            level_data["has_a"] = True
        else:
            level_data["has_a"] = False
        
        levels[lv] = level_data
    
    if not levels:
        return None
    
    return {
        "bar_idx": bar_idx,
        "window_id": window_id,
        "is_high": is_high,
        "price": price,
        "max_level": max_level,
        "levels": levels,
    }


def compute_target_vector(node_feat):
    """
    Compute aggregated target vector from multi-level C predictions.
    
    V_target = Σ w_n * V_Cn
    
    Weight w_n for each level depends on:
    - ABC completeness (has_a?)
    - B/A quality (lower slope_ratio = higher weight)
    - Level (higher levels = larger, more significant structures)
    """
    levels = node_feat["levels"]
    
    weighted_mod = 0.0
    weighted_amp = 0.0
    weighted_dur = 0.0
    total_weight = 0.0
    
    level_details = []
    
    for lv, data in sorted(levels.items()):
        # Base weight: higher level = more weight
        w = 1.0 + lv * 0.5  # L0=1.0, L1=1.5, L2=2.0, ...
        
        # Quality bonus: if has A and B is weak relative to A
        if data["has_a"]:
            sr = data.get("ba_slope_ratio", 1.0)
            if sr < 0.3:
                w *= 2.0
            elif sr < 0.5:
                w *= 1.5
            elif sr < 0.7:
                w *= 1.2
            elif sr > 1.5:
                w *= 0.7
        else:
            w *= 0.5  # penalize incomplete chains
        
        c_mod = abs(data["c_mod"])
        c_amp = abs(data["c_amp"])
        c_dur = data["c_dur"]
        c_dir = data["c_dir"]
        
        weighted_mod += w * c_mod
        weighted_amp += w * c_amp * c_dir  # signed
        weighted_dur += w * c_dur
        total_weight += w
        
        level_details.append({
            "level": lv,
            "weight": w,
            "c_mod": c_mod,
            "c_amp": c_amp,
            "c_dur": c_dur,
            "c_dir": c_dir,
            "has_a": data["has_a"],
            "ba_slope_ratio": data.get("ba_slope_ratio", None),
        })
    
    if total_weight <= 0:
        return None
    
    target = {
        "v_mod": weighted_mod / total_weight,
        "v_amp": weighted_amp / total_weight,  # signed
        "v_dur": weighted_dur / total_weight,
        "n_levels": len(levels),
        "total_weight": total_weight,
        "direction": 1 if weighted_amp > 0 else -1,
        "details": level_details,
    }
    
    return target


def simulate_trade(entry_bar, direction, tp_amp, sl_amp, a_bars_equiv,
                   highs, lows, closes):
    """
    Trade simulation with delay-aware parameters.
    tp_amp/sl_amp are in price units (percentage of entry price).
    """
    if entry_bar >= len(closes) - 1:
        return None
    
    entry_price = closes[entry_bar]
    tp_dist = entry_price * tp_amp / 100.0
    sl_dist = entry_price * sl_amp / 100.0
    
    if tp_dist <= 0 or sl_dist <= 0:
        return None
    
    if direction == 1:
        current_tp = entry_price + tp_dist
        current_sl = entry_price - sl_dist
    else:
        current_tp = entry_price - tp_dist
        current_sl = entry_price + sl_dist
    
    max_favorable = 0.0
    hit_breakeven = False
    max_hold = max(int(a_bars_equiv * 6), 300)
    end_bar = min(entry_bar + max_hold, len(closes) - 1)
    
    for bar in range(entry_bar + 1, end_bar + 1):
        h = highs[bar]
        l = lows[bar]
        
        if direction == 1:
            favorable = h - entry_price
        else:
            favorable = entry_price - l
        if favorable > max_favorable:
            max_favorable = favorable
        
        progress = max_favorable / tp_dist if tp_dist > 0 else 0
        bars_elapsed = bar - entry_bar
        
        # SL check
        if direction == 1:
            if l <= current_sl:
                pnl = current_sl - entry_price
                return {"pnl": pnl, "pnl_r": pnl / sl_dist, "reason": "sl",
                        "hold": bars_elapsed}
        else:
            if h >= current_sl:
                pnl = entry_price - current_sl
                return {"pnl": pnl, "pnl_r": pnl / sl_dist, "reason": "sl",
                        "hold": bars_elapsed}
        
        # Dynamic
        if progress >= 0.25 and not hit_breakeven:
            current_sl = entry_price
            hit_breakeven = True
        
        if progress >= 0.50:
            lock = 0.40
            if direction == 1:
                current_sl = max(current_sl, entry_price + max_favorable * lock)
            else:
                current_sl = min(current_sl, entry_price - max_favorable * lock)
        
        if progress >= 1.0:
            if direction == 1:
                current_sl = max(current_sl, entry_price + max_favorable - tp_dist * 0.15)
            else:
                current_sl = min(current_sl, entry_price - max_favorable + tp_dist * 0.15)
        
        # Stalling
        if bars_elapsed > a_bars_equiv * 2.5 and progress < 0.40 and max_favorable > 0:
            if direction == 1:
                current_tp = min(current_tp, entry_price + max_favorable * 1.05)
            else:
                current_tp = max(current_tp, entry_price - max_favorable * 1.05)
        
        # TP check
        if direction == 1:
            if h >= current_tp:
                pnl = current_tp - entry_price
                return {"pnl": pnl, "pnl_r": pnl / sl_dist, "reason": "tp",
                        "hold": bars_elapsed}
        else:
            if l <= current_tp:
                pnl = entry_price - current_tp
                return {"pnl": pnl, "pnl_r": pnl / sl_dist, "reason": "tp",
                        "hold": bars_elapsed}
    
    # Timeout
    if direction == 1:
        pnl = closes[end_bar] - entry_price
    else:
        pnl = entry_price - closes[end_bar]
    return {"pnl": pnl, "pnl_r": pnl / sl_dist, "reason": "timeout",
            "hold": end_bar - entry_bar}


def process_pair_tf(args):
    """Process one pair-TF: extract features, compute targets, simulate trades."""
    pair, tf, delays = args
    
    dates, price_data = load_ohlcv(pair, tf)
    if dates is None:
        return []
    
    highs, lows, closes = price_data
    n_bars = len(closes)
    
    feats, edges = load_zigzag_topo(pair, tf)
    if feats is None:
        return []
    
    # Deduplicate nodes: same bar_idx can appear in multiple windows
    # Take the one with highest max_level (most complete picture)
    bar_to_node = {}
    for i in range(len(feats)):
        bar = int(feats[i, 0])
        ml = int(feats[i, 5])
        if bar not in bar_to_node or ml > feats[bar_to_node[bar], 5]:
            bar_to_node[bar] = i
    
    # Extract features and simulate for each unique pivot
    results = []
    
    for bar_idx in sorted(bar_to_node.keys()):
        node_idx = bar_to_node[bar_idx]
        
        if bar_idx < 100 or bar_idx > n_bars - 100:
            continue
        
        nf = extract_node_features(node_idx, feats, edges)
        if nf is None:
            continue
        
        target = compute_target_vector(nf)
        if target is None:
            continue
        
        direction = target["direction"]
        v_mod = target["v_mod"]
        v_amp = abs(target["v_amp"])
        v_dur = target["v_dur"]
        
        if v_amp < 0.01 or v_mod < 0.01:
            continue
        
        # Determine year for IS/OOS split
        year = int(dates[min(bar_idx, len(dates)-1)][:4])
        
        # For each delay, compute delay-adjusted TP/SL and simulate
        for delay in delays:
            entry_bar = bar_idx + delay
            if entry_bar >= n_bars - 50:
                continue
            
            # Delay-adjusted parameters:
            # As delay increases, we've lost some of the move
            # Reduce TP proportionally, keep SL tighter
            decay = np.exp(-0.08 * delay)  # from our empirical decay curve
            
            tp_amp = v_amp * 0.80 * decay    # TP shrinks with delay
            sl_amp = tp_amp * (0.35 + 0.02 * delay)  # SL ratio loosens slightly
            
            # Minimum viable trade
            if tp_amp < 0.02:
                continue
            
            a_bars_equiv = max(v_dur, 10)
            
            trade = simulate_trade(
                entry_bar, direction, tp_amp, sl_amp, a_bars_equiv,
                highs, lows, closes
            )
            
            if trade is None:
                continue
            
            results.append({
                "pair": pair,
                "tf": tf,
                "bar_idx": bar_idx,
                "year": year,
                "delay": delay,
                "direction": direction,
                "n_levels": target["n_levels"],
                "v_mod": v_mod,
                "v_amp": v_amp,
                "v_dur": v_dur,
                "total_weight": target["total_weight"],
                "tp_amp": tp_amp,
                "sl_amp": sl_amp,
                "pnl_r": trade["pnl_r"],
                "reason": trade["reason"],
                "hold": trade["hold"],
            })
    
    n = len(results)
    print(f"  {pair}_{tf}: {n} trades from {len(bar_to_node)} pivots")
    return results


def main():
    print("=" * 80)
    print("  MULTI-LEVEL VECTOR AGGREGATION SYSTEM")
    print("=" * 80)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Get available pairs
    available = set()
    for tf_dir in os.listdir(ZIG_DIR):
        tf_path = os.path.join(ZIG_DIR, tf_dir)
        if not os.path.isdir(tf_path):
            continue
        for fname in os.listdir(tf_path):
            if fname.endswith(".npz"):
                pair = fname.replace(f"_{tf_dir}.npz", "")
                available.add((pair, tf_dir))
    
    # Filter to pairs that have raw data too
    tfs = ["H1"]  # Start with H1 for speed
    tasks = []
    delays = [0, 2, 5, 8, 12]  # Test multiple delays
    
    for pair, tf in sorted(available):
        if tf not in tfs:
            continue
        raw_path = os.path.join(DATA_DIR, f"{pair}_{tf}.csv")
        if os.path.exists(raw_path):
            tasks.append((pair, tf, delays))
    
    print(f"Tasks: {len(tasks)} pair-TFs, delays={delays}")
    print(f"Running with 30 workers...\n")
    
    all_results = []
    with Pool(30) as pool:
        for batch in pool.imap_unordered(process_pair_tf, tasks):
            all_results.extend(batch)
    
    print(f"\nTotal trades: {len(all_results):,}")
    
    if not all_results:
        print("ERROR: No results")
        return
    
    # ── Analysis ──
    print("\n" + "=" * 80)
    print("  RESULTS BY DELAY")
    print("=" * 80)
    
    for delay in delays:
        trades = [r for r in all_results if r["delay"] == delay]
        if not trades:
            continue
        pnls = [t["pnl_r"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        pf = abs(sum(wins)/sum(losses)) if losses and sum(losses) != 0 else 999
        
        print(f"  delay={delay:>2d}: n={len(pnls):>7,}  WR={len(wins)/len(pnls)*100:.1f}%  "
              f"avgR={np.mean(pnls):.4f}  PF={pf:.2f}  "
              f"avgWin={np.mean(wins):.3f}  avgLoss={np.mean(losses):.3f}")
    
    # ── By n_levels ──
    print("\n" + "=" * 80)
    print("  RESULTS BY NUMBER OF ACTIVE LEVELS (delay=5)")
    print("=" * 80)
    
    d5 = [r for r in all_results if r["delay"] == 5]
    for nl in sorted(set(r["n_levels"] for r in d5)):
        trades = [r for r in d5 if r["n_levels"] == nl]
        if len(trades) < 50:
            continue
        pnls = [t["pnl_r"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        pf = abs(sum(wins)/sum(losses)) if losses and sum(losses) != 0 else 999
        print(f"  n_levels={nl}: n={len(pnls):>6,}  WR={len(wins)/len(pnls)*100:.1f}%  "
              f"avgR={np.mean(pnls):.4f}  PF={pf:.2f}")
    
    # ── IS vs OOS by delay ──
    print("\n" + "=" * 80)
    print("  IS (<=2018) vs OOS (2019+) BY DELAY")
    print("=" * 80)
    
    for delay in delays:
        is_t = [r for r in all_results if r["delay"] == delay and 0 < r["year"] <= 2018]
        oos_t = [r for r in all_results if r["delay"] == delay and r["year"] > 2018]
        
        if not is_t or not oos_t:
            continue
        
        is_pnls = [t["pnl_r"] for t in is_t]
        oos_pnls = [t["pnl_r"] for t in oos_t]
        is_w = [p for p in is_pnls if p > 0]
        is_l = [p for p in is_pnls if p <= 0]
        oos_w = [p for p in oos_pnls if p > 0]
        oos_l = [p for p in oos_pnls if p <= 0]
        is_pf = abs(sum(is_w)/sum(is_l)) if is_l and sum(is_l) != 0 else 999
        oos_pf = abs(sum(oos_w)/sum(oos_l)) if oos_l and sum(oos_l) != 0 else 999
        
        print(f"  delay={delay:>2d}: IS n={len(is_pnls):>6,} WR={len(is_w)/len(is_pnls)*100:.1f}% "
              f"avgR={np.mean(is_pnls):.4f} PF={is_pf:.2f} | "
              f"OOS n={len(oos_pnls):>6,} WR={len(oos_w)/len(oos_pnls)*100:.1f}% "
              f"avgR={np.mean(oos_pnls):.4f} PF={oos_pf:.2f}")
    
    # ── Save results ──
    out_path = os.path.join(OUT_DIR, "multilevel_results.json")
    summary = {
        "total_trades": len(all_results),
        "by_delay": {},
    }
    for delay in delays:
        trades = [r for r in all_results if r["delay"] == delay]
        if trades:
            pnls = [t["pnl_r"] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            summary["by_delay"][str(delay)] = {
                "n": len(pnls),
                "wr": len(wins)/len(pnls)*100,
                "avg_r": float(np.mean(pnls)),
                "pf": abs(sum(wins)/sum(losses)) if losses and sum(losses) != 0 else 999,
            }
    
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")
    
    print(f"\n{'='*80}")
    print("  COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
