"""
Stage2 ABC Analysis
===================
Analyze the ABC triples collected by abc_collector.py.
Generate heatmaps in (amp_ratio, time_ratio) space to find optimal entry zones.

Output: text-based analysis + saved PNG heatmaps.
"""

import os
import csv
import numpy as np
from collections import defaultdict

INPUT = "/home/ubuntu/stage2_abc/abc_all_triples.csv"
OUT_DIR = "/home/ubuntu/stage2_abc/analysis"


def load_data():
    """Load CSV into list of dicts with float conversion."""
    data = []
    float_cols = [
        'a_amp', 'a_bars', 'a_slope', 'a_dir',
        'b_amp', 'b_bars', 'b_slope',
        'amp_ratio', 'time_ratio', 'slope_ratio',
        'c_amp', 'c_bars', 'c_dir', 'c_follows_a', 'c_a_ratio',
        'pnl', 'pnl_r', 'max_favorable', 'hold_bars',
        'tp_distance', 'sl_distance',
        'entry_price', 'exit_price',
        'a_start_price', 'a_end_price', 'b_end_price', 'c_end_price',
        'a_start_bar', 'a_end_bar', 'b_end_bar', 'c_end_bar',
        'entry_bar', 'exit_bar',
        'zz_dev', 'zz_confirm',
    ]
    with open(INPUT, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = {}
            for k, v in row.items():
                if k in float_cols:
                    try:
                        r[k] = float(v)
                    except:
                        r[k] = 0.0
                else:
                    r[k] = v
            data.append(r)
    return data


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze_overall(data):
    """Overall statistics."""
    print_section("OVERALL STATISTICS (no filtering)")

    pnls = [d['pnl_r'] for d in data]
    c_follows = [d['c_follows_a'] for d in data]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    print(f"Total trades: {len(data):,}")
    print(f"C follows A direction: {sum(c_follows)/len(c_follows)*100:.1f}%")
    print(f"Win rate: {len(wins)/len(pnls)*100:.1f}%")
    print(f"Avg PnL (R): {np.mean(pnls):.4f}")
    print(f"Median PnL (R): {np.median(pnls):.4f}")
    print(f"Std PnL (R): {np.std(pnls):.4f}")
    print(f"Avg Win (R): {np.mean(wins):.4f}" if wins else "No wins")
    print(f"Avg Loss (R): {np.mean(losses):.4f}" if losses else "No losses")
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
    print(f"Profit Factor: {pf:.2f}")

    # By exit reason
    reasons = defaultdict(list)
    for d in data:
        reasons[d['exit_reason']].append(d['pnl_r'])
    print(f"\nBy exit reason:")
    for reason, rpnls in sorted(reasons.items()):
        wr = len([p for p in rpnls if p > 0]) / len(rpnls) * 100
        print(f"  {reason:10s}: n={len(rpnls):>8,}  WR={wr:.1f}%  "
              f"avg_R={np.mean(rpnls):.3f}  median_R={np.median(rpnls):.3f}")


def analyze_by_zz_config(data):
    """Break down by zigzag configuration."""
    print_section("BY ZIGZAG CONFIG")

    groups = defaultdict(list)
    for d in data:
        key = (d['zz_dev'], d['zz_confirm'])
        groups[key].append(d['pnl_r'])

    for key in sorted(groups.keys()):
        pnls = groups[key]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        print(f"  dev={key[0]:.1f} confirm={key[1]:.0f}: "
              f"n={len(pnls):>8,}  WR={len(wins)/len(pnls)*100:.1f}%  "
              f"avg_R={np.mean(pnls):.3f}  PF={pf:.2f}")


def analyze_by_tf(data):
    """Break down by timeframe."""
    print_section("BY TIMEFRAME")

    groups = defaultdict(list)
    for d in data:
        groups[d['tf']].append(d['pnl_r'])

    for tf in sorted(groups.keys()):
        pnls = groups[tf]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        print(f"  {tf:5s}: n={len(pnls):>8,}  WR={len(wins)/len(pnls)*100:.1f}%  "
              f"avg_R={np.mean(pnls):.3f}  PF={pf:.2f}")


def heatmap_text(data, val_key='pnl_r', agg='mean',
                 amp_bins=None, time_bins=None, min_count=100):
    """
    Text-based heatmap of val_key aggregated by (amp_ratio, time_ratio) bins.
    """
    if amp_bins is None:
        amp_bins = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
                    0.90, 1.00, 1.15, 1.50, 2.00, 5.00]
    if time_bins is None:
        time_bins = [0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0, 5.0, 10.0, 50.0]

    grid = defaultdict(list)
    for d in data:
        ar = d['amp_ratio']
        tr = d['time_ratio']
        # Find bins
        ai = len(amp_bins) - 2
        for j in range(len(amp_bins) - 1):
            if ar < amp_bins[j + 1]:
                ai = j
                break
        ti = len(time_bins) - 2
        for j in range(len(time_bins) - 1):
            if tr < time_bins[j + 1]:
                ti = j
                break
        grid[(ai, ti)].append(d[val_key])

    # Print header
    col_labels = []
    for j in range(len(time_bins) - 1):
        col_labels.append(f"T{time_bins[j]:.1f}-{time_bins[j+1]:.1f}")

    header = f"{'amp\\time':>16s}"
    for cl in col_labels:
        header += f" {cl:>12s}"
    print(header)
    print("-" * len(header))

    results_grid = {}
    for i in range(len(amp_bins) - 1):
        row_label = f"A{amp_bins[i]:.2f}-{amp_bins[i+1]:.2f}"
        row = f"{row_label:>16s}"
        for j in range(len(time_bins) - 1):
            vals = grid.get((i, j), [])
            if len(vals) < min_count:
                row += f" {'---':>12s}"
            else:
                if agg == 'mean':
                    v = np.mean(vals)
                elif agg == 'count':
                    v = len(vals)
                elif agg == 'winrate':
                    v = len([x for x in vals if x > 0]) / len(vals) * 100
                elif agg == 'c_follows':
                    v = np.mean(vals) * 100
                row += f" {v:>12.3f}" if agg != 'count' else f" {int(v):>12,}"
                results_grid[(i, j)] = (v, len(vals))
            
        print(row)

    return results_grid


def analyze_heatmaps(data):
    """Generate multiple heatmaps."""

    print_section("HEATMAP: Mean PnL (R) by (amp_ratio, time_ratio)")
    print("Each cell: mean PnL in R-multiples. '---' = <100 samples.\n")
    pnl_grid = heatmap_text(data, 'pnl_r', 'mean')

    print_section("HEATMAP: Win Rate (%) by (amp_ratio, time_ratio)")
    print("Each cell: percentage of trades with PnL > 0.\n")
    wr_grid = heatmap_text(data, 'pnl_r', 'winrate')

    print_section("HEATMAP: C follows A direction (%) by (amp_ratio, time_ratio)")
    print("Each cell: % of cases where C goes in A's direction.\n")
    heatmap_text(data, 'c_follows_a', 'c_follows')

    print_section("HEATMAP: Sample Count by (amp_ratio, time_ratio)")
    print("Each cell: number of ABC triples.\n")
    heatmap_text(data, 'pnl_r', 'count', min_count=0)

    return pnl_grid, wr_grid


def analyze_slope_ratio(data):
    """Analyze by slope_ratio (B slope / A slope)."""
    print_section("ANALYSIS BY SLOPE RATIO (B_slope / A_slope)")
    print("slope_ratio < 1 means B is 'weaker' than A (your hypothesis)\n")

    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            1.2, 1.5, 2.0, 5.0, 100.0]

    groups = defaultdict(list)
    for d in data:
        sr = d['slope_ratio']
        for j in range(len(bins) - 1):
            if sr < bins[j + 1]:
                groups[j].append(d)
                break

    print(f"{'slope_ratio':>16s} {'n':>8s} {'WR%':>7s} {'avgR':>8s} {'PF':>7s} "
          f"{'C_fol%':>7s} {'C/A_amp':>8s}")
    print("-" * 70)

    for j in range(len(bins) - 1):
        rows = groups.get(j, [])
        if len(rows) < 100:
            continue
        pnls = [r['pnl_r'] for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        c_fol = np.mean([r['c_follows_a'] for r in rows]) * 100
        c_a = np.mean([r['c_a_ratio'] for r in rows])

        label = f"{bins[j]:.1f}-{bins[j+1]:.1f}"
        print(f"{label:>16s} {len(rows):>8,} {len(wins)/len(pnls)*100:>7.1f} "
              f"{np.mean(pnls):>8.3f} {pf:>7.2f} {c_fol:>7.1f} {c_a:>8.3f}")


def analyze_combined_quality(data):
    """
    Combined quality score: how 'weak' B is relative to A.
    quality = (1 - slope_ratio) * (1 - amp_ratio/2) * time_ratio_factor
    Higher = B is weaker → better entry per hypothesis.
    """
    print_section("ANALYSIS BY COMBINED B-WEAKNESS SCORE")
    print("Score = how much weaker B is relative to A (higher = weaker B)\n")

    scored = []
    for d in data:
        sr = d['slope_ratio']
        ar = d['amp_ratio']
        tr = d['time_ratio']

        # B weakness: low slope ratio, low amp ratio, high time ratio
        # Normalize each to [0, 1] range approximately
        slope_score = max(0, min(1, 1.0 - sr))          # 1 if B slope = 0
        amp_score = max(0, min(1, 1.0 - ar))             # 1 if B retraces 0%
        time_score = min(1, tr / 5.0)                     # 1 if B takes 5x as long

        # Combined: geometric mean so all must be somewhat good
        quality = (slope_score * 0.4 + amp_score * 0.3 + time_score * 0.3)
        d['quality'] = quality
        scored.append(d)

    # Sort by quality and bin
    scored.sort(key=lambda x: x['quality'])
    n = len(scored)
    decile_size = n // 10

    print(f"{'Decile':>8s} {'quality':>10s} {'n':>8s} {'WR%':>7s} {'avgR':>8s} "
          f"{'PF':>7s} {'C_fol%':>7s}")
    print("-" * 65)

    for dec in range(10):
        start = dec * decile_size
        end = start + decile_size if dec < 9 else n
        chunk = scored[start:end]
        pnls = [r['pnl_r'] for r in chunk]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        c_fol = np.mean([r['c_follows_a'] for r in chunk]) * 100
        avg_q = np.mean([r['quality'] for r in chunk])

        print(f"{'D'+str(dec+1):>8s} {avg_q:>10.3f} {len(chunk):>8,} "
              f"{len(wins)/len(pnls)*100:>7.1f} {np.mean(pnls):>8.3f} "
              f"{pf:>7.2f} {c_fol:>7.1f}")


def find_optimal_boundary(data):
    """
    Search for the optimal boundary in (amp_ratio, time_ratio) space.
    Test the hypothesis: amp_ratio >= k / time_ratio
    for various k values.
    """
    print_section("OPTIMAL ENTRY BOUNDARY: amp_ratio >= k / time_ratio")
    print("Testing your hypothesis that B needs less retracement when it takes longer\n")

    k_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]

    print(f"{'k':>6s} {'n_pass':>8s} {'n_fail':>8s} {'WR_pass':>8s} {'WR_fail':>8s} "
          f"{'R_pass':>8s} {'R_fail':>8s} {'PF_pass':>8s} {'PF_fail':>8s} {'diff_R':>8s}")
    print("-" * 95)

    for k in k_values:
        passed = []
        failed = []
        for d in data:
            threshold = k / d['time_ratio'] if d['time_ratio'] > 0 else 999
            if d['amp_ratio'] >= threshold:
                passed.append(d['pnl_r'])
            else:
                failed.append(d['pnl_r'])

        if len(passed) < 100 or len(failed) < 100:
            continue

        wins_p = [p for p in passed if p > 0]
        losses_p = [p for p in passed if p <= 0]
        pf_p = abs(sum(wins_p) / sum(losses_p)) if losses_p and sum(losses_p) != 0 else float('inf')

        wins_f = [p for p in failed if p > 0]
        losses_f = [p for p in failed if p <= 0]
        pf_f = abs(sum(wins_f) / sum(losses_f)) if losses_f and sum(losses_f) != 0 else float('inf')

        wr_p = len(wins_p) / len(passed) * 100
        wr_f = len(wins_f) / len(failed) * 100
        r_p = np.mean(passed)
        r_f = np.mean(failed)

        print(f"{k:>6.1f} {len(passed):>8,} {len(failed):>8,} "
              f"{wr_p:>8.1f} {wr_f:>8.1f} "
              f"{r_p:>8.3f} {r_f:>8.3f} "
              f"{pf_p:>8.2f} {pf_f:>8.2f} "
              f"{r_p - r_f:>8.3f}")


def statistical_significance(data):
    """
    Test if the best regions are statistically significant vs baseline.
    """
    print_section("STATISTICAL SIGNIFICANCE")

    # Overall baseline
    all_pnls = np.array([d['pnl_r'] for d in data])
    baseline_mean = np.mean(all_pnls)
    baseline_std = np.std(all_pnls)

    print(f"Baseline: mean_R={baseline_mean:.4f}, std_R={baseline_std:.4f}, n={len(all_pnls):,}")

    # Test various filters
    filters = {
        'amp_ratio 0.10-0.50 & time_ratio 1.0-5.0': 
            lambda d: 0.10 <= d['amp_ratio'] < 0.50 and 1.0 <= d['time_ratio'] < 5.0,
        'amp_ratio 0.20-0.60 & time_ratio 0.5-2.0':
            lambda d: 0.20 <= d['amp_ratio'] < 0.60 and 0.5 <= d['time_ratio'] < 2.0,
        'amp_ratio 0.30-0.70 & slope_ratio < 0.5':
            lambda d: 0.30 <= d['amp_ratio'] < 0.70 and d['slope_ratio'] < 0.5,
        'slope_ratio < 0.3 (B very weak)':
            lambda d: d['slope_ratio'] < 0.3,
        'slope_ratio < 0.5':
            lambda d: d['slope_ratio'] < 0.5,
        'amp >= 0.8/time_ratio (your curve)':
            lambda d: d['amp_ratio'] >= 0.8 / d['time_ratio'] if d['time_ratio'] > 0 else False,
        'amp >= 0.5/time_ratio':
            lambda d: d['amp_ratio'] >= 0.5 / d['time_ratio'] if d['time_ratio'] > 0 else False,
    }

    print(f"\n{'Filter':>50s} {'n':>8s} {'WR%':>7s} {'avgR':>8s} {'vs_base':>8s} {'t-stat':>8s} {'p-val':>8s}")
    print("-" * 100)

    for name, filt in filters.items():
        subset = [d['pnl_r'] for d in data if filt(d)]
        if len(subset) < 100:
            continue

        sub_mean = np.mean(subset)
        sub_std = np.std(subset)
        n = len(subset)
        
        # Welch's t-test vs baseline
        se = np.sqrt(sub_std**2 / n + baseline_std**2 / len(all_pnls))
        t_stat = (sub_mean - baseline_mean) / se if se > 0 else 0

        # Approximate p-value (two-tailed, normal approx for large n)
        from math import erfc, sqrt
        p_val = erfc(abs(t_stat) / sqrt(2))

        wins = len([p for p in subset if p > 0])
        wr = wins / n * 100

        print(f"{name:>50s} {n:>8,} {wr:>7.1f} {sub_mean:>8.4f} "
              f"{sub_mean - baseline_mean:>+8.4f} {t_stat:>8.2f} {p_val:>8.4f}")


def main():
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data):,} ABC triples\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    analyze_overall(data)
    analyze_by_zz_config(data)
    analyze_by_tf(data)
    analyze_heatmaps(data)
    analyze_slope_ratio(data)
    analyze_combined_quality(data)
    find_optimal_boundary(data)
    statistical_significance(data)

    print(f"\n{'='*70}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
