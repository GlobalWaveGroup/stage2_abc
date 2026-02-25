"""
Exit Diagnosis: understand HOW C' develops in winning vs losing trades.

Back to first principles:
- Price that went up a lot tends to come back (mean reversion in amplitude)
- Price that went up for long tends to reverse (mean reversion in time)
- But we want to ride the trend (C) while it lasts
- Exit when the STRUCTURE says the trend is over, not at arbitrary levels

Key questions:
1. In winning trades, what is the max retrace C' experiences BEFORE hitting TP?
   (This tells us how much "noise" we must tolerate)
2. In losing trades, what did the path look like?
   (Did it slowly drift to SL, or spike?)
3. What is the relationship between MFE (max favorable) and MAE (max adverse)?
4. At what point in time does a trade "decide" - win or lose?
5. Does C' form micro-zigzags, and do their proportions predict outcome?
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


def extract_signals_scored(pair, tf):
    """Same signal extraction as before, score 10-30 only."""
    dates, price_data = load_ohlcv(pair, tf)
    if dates is None:
        return [], None, None, None, None
    highs, lows, closes = price_data
    n_bars = len(closes)
    feats, edges = load_zigzag_topo(pair, tf)
    if feats is None:
        return [], None, None, None, None

    win_bar_to_node = defaultdict(dict)
    for i in range(len(feats)):
        win = int(feats[i, 1])
        rel_idx = int(feats[i, 2])
        ml = int(feats[i, 5])
        if rel_idx not in win_bar_to_node[win] or ml > feats[win_bar_to_node[win][rel_idx], 5]:
            win_bar_to_node[win][rel_idx] = i

    DELAY = 5
    signals = []
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

        level_scores = []
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
            a_amp = abs(a["amplitude_pct"]); b_amp = abs(b["amplitude_pct"])
            a_mod = a["modulus"]; b_mod = b["modulus"]
            if a_amp < 1e-6 or a_mod < 1e-6: continue
            ba_amp_ratio = b_amp / a_amp; ba_mod_ratio = b_mod / a_mod
            score = 0.5 + lv * 0.5
            score += max(0, 2.0 * (1.0 - fib_distance(ba_amp_ratio) / 0.10))
            score += max(0, 1.0 * (1.0 - fib_distance(ba_mod_ratio) / 0.10))
            if ba_amp_ratio < 0.382: score += 1.5
            elif ba_amp_ratio < 0.618: score += 0.8
            elif ba_amp_ratio < 1.0: score += 0.0
            else: score -= 0.5
            score += 1.0 if (a["direction"] == c["direction"]) else -0.5
            level_scores.append({"lv": lv, "score": score, "c_amp": c["amplitude_pct"],
                                 "c_dir": c["direction"], "c_dur": c["duration"]})

        if not level_scores: continue
        n_lv = len(level_scores)
        total_score = sum(ls["score"] for ls in level_scores)
        dirs = [ls["c_dir"] for ls in level_scores]
        consensus = abs(sum(dirs)) / len(dirs)
        if consensus >= 0.9: total_score += n_lv * 0.5
        elif consensus < 0.5: total_score -= n_lv * 0.3
        if total_score < 10 or total_score >= 30: continue

        direction_sum = sum(ls["c_dir"] * ls["score"] for ls in level_scores)
        direction = 1 if direction_sum > 0 else -1
        w_amp = sum(ls["score"] * abs(ls["c_amp"]) for ls in level_scores if ls["score"] > 0)
        w_total = sum(ls["score"] for ls in level_scores if ls["score"] > 0)
        if w_total <= 0: continue
        v_amp = w_amp / w_total
        if v_amp < 0.01: continue

        eb = bar_idx + DELAY
        if eb >= n_bars - 300: continue
        year = int(dates[min(bar_idx, len(dates) - 1)][:4])

        signals.append({"entry_bar": eb, "direction": direction, "score": total_score,
                        "v_amp": v_amp, "year": year})

    print(f"  {pair}_{tf}: {len(signals):,} signals")
    return signals, highs, lows, closes, dates


def trace_trade_path(sig, highs, lows, closes):
    """
    Instead of simulating a trade, trace the full bar-by-bar path of C'.
    Record MFE, MAE, retraces, micro-zigzag structure.
    """
    eb = sig["entry_bar"]
    direction = sig["direction"]
    v_amp = sig["v_amp"]
    n_bars = len(closes)

    ep = closes[eb]
    # Use predicted C amplitude as the reference unit
    ref = ep * v_amp / 100.0  # predicted C in price units
    if ref <= 0:
        return None

    max_hold = 200
    end_bar = min(eb + max_hold, n_bars - 1)

    # Track path
    mfe = 0.0  # max favorable excursion (in ref units)
    mae = 0.0  # max adverse excursion
    mfe_bar = 0
    mae_bar = 0

    # Track micro-zigzag: sequence of swings from entry
    # A "swing" is when price reverses direction by > threshold
    swing_threshold = 0.1  # 10% of ref = minimum swing to count
    last_extreme_fav = 0.0  # last swing high (favorable)
    last_extreme_adv = 0.0  # last swing low (adverse)
    swings = []  # list of (bar_offset, fav_excursion_in_ref_units)
    current_fav_peak = 0.0
    current_fav_trough = 0.0

    # Track retrace episodes
    max_retrace_from_mfe = 0.0  # largest retrace ratio from MFE
    retrace_episodes = []  # (mfe_at_time, retrace_amount, recovered?)

    in_retrace = False
    retrace_start_mfe = 0.0

    for bar in range(eb + 1, end_bar + 1):
        h = highs[bar]
        l = lows[bar]
        elapsed = bar - eb

        if direction == 1:
            fav = (h - ep) / ref  # favorable in ref units
            adv = (ep - l) / ref  # adverse in ref units
            close_fav = (closes[bar] - ep) / ref
        else:
            fav = (ep - l) / ref
            adv = (h - ep) / ref
            close_fav = (ep - closes[bar]) / ref

        if fav > mfe:
            mfe = fav
            mfe_bar = elapsed
        if adv > mae:
            mae = adv
            mae_bar = elapsed

        # Track retrace from MFE
        if mfe > swing_threshold:
            current_retrace = mfe - close_fav
            retrace_ratio = current_retrace / mfe if mfe > 0 else 0

            if retrace_ratio > max_retrace_from_mfe:
                max_retrace_from_mfe = retrace_ratio

            # Detect retrace episodes
            if not in_retrace and retrace_ratio > 0.15:
                in_retrace = True
                retrace_start_mfe = mfe
            elif in_retrace and retrace_ratio < 0.05:
                # Retrace ended, price recovered
                retrace_episodes.append((retrace_start_mfe, max_retrace_from_mfe, True))
                in_retrace = False

    if in_retrace:
        retrace_episodes.append((retrace_start_mfe, max_retrace_from_mfe, False))

    # Final outcome (close at end)
    if direction == 1:
        final_pnl = (closes[end_bar] - ep) / ref
    else:
        final_pnl = (ep - closes[end_bar]) / ref

    return {
        "mfe": round(mfe, 3),  # in multiples of predicted C
        "mae": round(mae, 3),
        "mfe_bar": mfe_bar,
        "mae_bar": mae_bar,
        "max_retrace_ratio": round(max_retrace_from_mfe, 3),
        "n_retrace_episodes": len(retrace_episodes),
        "n_recovered_retraces": sum(1 for _, _, r in retrace_episodes if r),
        "final_pnl": round(final_pnl, 3),
        "score": sig["score"],
        "year": sig["year"],
    }


def main():
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

    all_paths = []
    for pair in pairs:
        sigs, highs, lows, closes, dates = extract_signals_scored(pair, "H1")
        if not sigs:
            continue
        for sig in sigs:
            path = trace_trade_path(sig, highs, lows, closes)
            if path:
                all_paths.append(path)

    print(f"\nTotal paths traced: {len(all_paths):,}")

    # =============================================
    # 1. MFE distribution: how far does C' actually go?
    # =============================================
    print(f"\n{'='*80}")
    print("  1. MFE DISTRIBUTION (multiples of predicted C)")
    print(f"{'='*80}")
    mfes = [p["mfe"] for p in all_paths]
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{pct}: {np.percentile(mfes, pct):.3f}x predicted C")

    print(f"\n  How often does C' reach predicted target?")
    for threshold in [0.25, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.0]:
        pct = sum(1 for m in mfes if m >= threshold) / len(mfes) * 100
        print(f"    MFE >= {threshold:.3f}x: {pct:.1f}%")

    # =============================================
    # 2. MAE distribution: how much adverse before winning?
    # =============================================
    print(f"\n{'='*80}")
    print("  2. MAE DISTRIBUTION (max adverse excursion)")
    print(f"{'='*80}")
    maes = [p["mae"] for p in all_paths]
    for pct in [10, 25, 50, 75, 90, 95]:
        print(f"  p{pct}: {np.percentile(maes, pct):.3f}x predicted C")

    # MAE for winners (MFE > 1.0) vs losers (MFE < 0.5)
    winners = [p for p in all_paths if p["mfe"] >= 1.0]
    losers = [p for p in all_paths if p["mfe"] < 0.382]
    if winners:
        print(f"\n  Winners (MFE >= 1.0, n={len(winners):,}):")
        w_maes = [p["mae"] for p in winners]
        print(f"    MAE median={np.median(w_maes):.3f}  p75={np.percentile(w_maes, 75):.3f}  p90={np.percentile(w_maes, 90):.3f}")
    if losers:
        print(f"  Losers (MFE < 0.382, n={len(losers):,}):")
        l_maes = [p["mae"] for p in losers]
        print(f"    MAE median={np.median(l_maes):.3f}  p75={np.percentile(l_maes, 75):.3f}  p90={np.percentile(l_maes, 90):.3f}")

    # =============================================
    # 3. MFE timing: WHEN does the peak happen?
    # =============================================
    print(f"\n{'='*80}")
    print("  3. MFE TIMING (bar offset of peak)")
    print(f"{'='*80}")
    mfe_bars = [p["mfe_bar"] for p in all_paths]
    for pct in [10, 25, 50, 75, 90]:
        print(f"  p{pct}: bar {np.percentile(mfe_bars, pct):.0f}")

    if winners:
        w_bars = [p["mfe_bar"] for p in winners]
        print(f"\n  Winners MFE timing: median bar {np.median(w_bars):.0f}, p75={np.percentile(w_bars, 75):.0f}")
    if losers:
        l_bars = [p["mfe_bar"] for p in losers]
        print(f"  Losers MFE timing: median bar {np.median(l_bars):.0f}, p75={np.percentile(l_bars, 75):.0f}")

    # =============================================
    # 4. MAX RETRACE: how much does C' retrace before continuing?
    # =============================================
    print(f"\n{'='*80}")
    print("  4. MAX RETRACE FROM MFE (key for exit design)")
    print(f"{'='*80}")
    # Only for trades where MFE > 0.25 (some progress was made)
    progressed = [p for p in all_paths if p["mfe"] >= 0.25]
    if progressed:
        retraces = [p["max_retrace_ratio"] for p in progressed]
        print(f"  Trades with MFE >= 0.25x: n={len(progressed):,}")
        for pct in [10, 25, 50, 75, 90, 95]:
            print(f"    p{pct}: {np.percentile(retraces, pct):.3f} retrace ratio")

        # Crucial: among trades that eventually won (MFE >= 1.0),
        # what was their max retrace DURING the trade?
        winners_prog = [p for p in progressed if p["mfe"] >= 1.0]
        losers_prog = [p for p in progressed if p["final_pnl"] < 0]
        if winners_prog:
            w_ret = [p["max_retrace_ratio"] for p in winners_prog]
            print(f"\n  WINNERS (MFE>=1.0, n={len(winners_prog):,}):")
            print(f"    max retrace: median={np.median(w_ret):.3f}  p75={np.percentile(w_ret, 75):.3f}  p90={np.percentile(w_ret, 90):.3f}")
            print(f"    retrace > 0.382: {sum(1 for r in w_ret if r > 0.382)/len(w_ret)*100:.1f}%")
            print(f"    retrace > 0.500: {sum(1 for r in w_ret if r > 0.500)/len(w_ret)*100:.1f}%")
            print(f"    retrace > 0.618: {sum(1 for r in w_ret if r > 0.618)/len(w_ret)*100:.1f}%")
            print(f"    retrace > 0.786: {sum(1 for r in w_ret if r > 0.786)/len(w_ret)*100:.1f}%")
            print(f"    retrace > 1.000: {sum(1 for r in w_ret if r > 1.000)/len(w_ret)*100:.1f}%")
        if losers_prog:
            l_ret = [p["max_retrace_ratio"] for p in losers_prog]
            print(f"\n  LOSERS (final_pnl<0, n={len(losers_prog):,}):")
            print(f"    max retrace: median={np.median(l_ret):.3f}  p75={np.percentile(l_ret, 75):.3f}  p90={np.percentile(l_ret, 90):.3f}")
            print(f"    retrace > 0.382: {sum(1 for r in l_ret if r > 0.382)/len(l_ret)*100:.1f}%")
            print(f"    retrace > 0.618: {sum(1 for r in l_ret if r > 0.618)/len(l_ret)*100:.1f}%")
            print(f"    retrace > 1.000: {sum(1 for r in l_ret if r > 1.000)/len(l_ret)*100:.1f}%")

    # =============================================
    # 5. RETRACE RECOVERY: does retrace predict failure?
    # =============================================
    print(f"\n{'='*80}")
    print("  5. RETRACE AS PREDICTOR: P(win | max_retrace < threshold)")
    print(f"{'='*80}")
    print("  'win' = final_pnl > 0")
    progressed = [p for p in all_paths if p["mfe"] >= 0.25]
    if progressed:
        for thresh in [0.2, 0.3, 0.382, 0.5, 0.618, 0.786, 1.0]:
            below = [p for p in progressed if p["max_retrace_ratio"] <= thresh]
            above = [p for p in progressed if p["max_retrace_ratio"] > thresh]
            if below and above:
                wr_below = sum(1 for p in below if p["final_pnl"] > 0) / len(below) * 100
                wr_above = sum(1 for p in above if p["final_pnl"] > 0) / len(above) * 100
                avg_below = np.mean([p["final_pnl"] for p in below])
                avg_above = np.mean([p["final_pnl"] for p in above])
                print(f"  retrace <= {thresh:.3f}: n={len(below):>7,}  WR={wr_below:.1f}%  avgPnL={avg_below:.3f}"
                      f"  |  > {thresh:.3f}: n={len(above):>7,}  WR={wr_above:.1f}%  avgPnL={avg_above:.3f}")

    # =============================================
    # 6. MFE vs MAE scatter: optimal SL level
    # =============================================
    print(f"\n{'='*80}")
    print("  6. OPTIMAL SL: what MAE threshold separates winners from losers?")
    print(f"{'='*80}")
    print("  If we set SL at X * predicted_C, what happens?")
    for sl_mult in [0.1, 0.15, 0.2, 0.25, 0.3, 0.382, 0.5, 0.618]:
        stopped = [p for p in all_paths if p["mae"] >= sl_mult]
        survived = [p for p in all_paths if p["mae"] < sl_mult]
        if survived:
            # Among survived: how many eventually win?
            sv_wr = sum(1 for p in survived if p["final_pnl"] > 0) / len(survived) * 100
            sv_avg = np.mean([p["final_pnl"] for p in survived])
            sv_mfe = np.median([p["mfe"] for p in survived])
        else:
            sv_wr = 0; sv_avg = 0; sv_mfe = 0
        print(f"  SL={sl_mult:.3f}: stopped={len(stopped):>7,} ({len(stopped)/len(all_paths)*100:.1f}%)  "
              f"survived={len(survived):>7,}  surv_WR={sv_wr:.1f}%  surv_avgPnL={sv_avg:.3f}  surv_medMFE={sv_mfe:.3f}")

    # =============================================
    # 7. TIME-BASED: early decision point
    # =============================================
    print(f"\n{'='*80}")
    print("  7. EARLY DECISION: MFE at bar 10/20/50 as predictor of outcome")
    print(f"{'='*80}")

    # We need bar-by-bar MFE, so re-trace a subset
    # Use the first pair's data for this detailed analysis
    pair = "EURUSD"
    sigs, highs, lows, closes, dates = extract_signals_scored(pair, "H1")
    if sigs:
        early_data = []
        for sig in sigs[:50000]:  # sample
            eb = sig["entry_bar"]
            d = sig["direction"]
            ep_val = closes[eb]
            ref_val = ep_val * sig["v_amp"] / 100.0
            if ref_val <= 0: continue
            n = len(closes)
            max_hold = min(200, n - eb - 1)

            mfe_10 = 0; mfe_20 = 0; mfe_50 = 0; mfe_total = 0
            for bar in range(eb + 1, min(eb + max_hold + 1, n)):
                if d == 1:
                    fav_val = (highs[bar] - ep_val) / ref_val
                else:
                    fav_val = (ep_val - lows[bar]) / ref_val
                offset = bar - eb
                if fav_val > mfe_total: mfe_total = fav_val
                if offset <= 10 and fav_val > mfe_10: mfe_10 = fav_val
                if offset <= 20 and fav_val > mfe_20: mfe_20 = fav_val
                if offset <= 50 and fav_val > mfe_50: mfe_50 = fav_val

            if d == 1:
                final_val = (closes[min(eb + max_hold, n - 1)] - ep_val) / ref_val
            else:
                final_val = (ep_val - closes[min(eb + max_hold, n - 1)]) / ref_val

            early_data.append({"mfe_10": mfe_10, "mfe_20": mfe_20,
                               "mfe_50": mfe_50, "mfe_total": mfe_total, "final": final_val})

        print(f"  EURUSD sample: {len(early_data):,} trades")
        for check_bar, key in [(10, "mfe_10"), (20, "mfe_20"), (50, "mfe_50")]:
            for thresh in [0.0, 0.1, 0.25, 0.5]:
                above = [e for e in early_data if e[key] >= thresh]
                if above:
                    wr = sum(1 for e in above if e["final"] > 0) / len(above) * 100
                    avg = np.mean([e["final"] for e in above])
                    med_mfe = np.median([e["mfe_total"] for e in above])
                    print(f"    MFE@bar{check_bar} >= {thresh:.1f}x: n={len(above):>6,}  "
                          f"final_WR={wr:.1f}%  avg_final={avg:.3f}  med_totalMFE={med_mfe:.3f}")
            print()


if __name__ == "__main__":
    main()
