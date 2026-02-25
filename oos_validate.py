"""
Stage2 OOS Validation + Feature Importance
============================================
1. Time-series OOS: train on 2000-2018, test on 2019-2024
   - Confirm ABC edge is stable out-of-sample
   - Confirm key parameters (slope_ratio etc) maintain ranking

2. Feature importance: which parameters truly separate good from bad trades
   - Univariate: each feature vs PnL correlation + binned performance
   - Check monotonicity and stability across IS/OOS

Uses the pre-collected abc_all_triples.csv (3.65M trades).
Entry bar timestamps needed — we'll use bar index + pair/TF to infer date.
"""

import os
import csv
import numpy as np
from collections import defaultdict
from math import erfc, sqrt

INPUT = "/home/ubuntu/stage2_abc/abc_all_triples.csv"
DATA_DIR = "/home/ubuntu/DataBase/base_kline"


def load_date_index(pair, tf):
    """Load bar_index → date mapping for a pair-TF."""
    fpath = os.path.join(DATA_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fpath):
        return {}
    dates = {}
    with open(fpath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # header
        for i, row in enumerate(reader):
            if len(row) >= 2:
                # row[0] = "2000.01.03", row[1] = "00:00:00"
                year = int(row[0][:4])
                dates[i] = year
    return dates


def load_data_with_dates():
    """Load all triples and tag with year based on entry_bar."""
    print("Loading date indices for all pairs...")

    # Pre-load all date mappings
    date_cache = {}
    pairs_tfs = set()

    # First pass: find all unique pair-TF combos
    with open(INPUT, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs_tfs.add((row['pair'], row['tf']))

    print(f"  Loading {len(pairs_tfs)} date indices...")
    for pair, tf in pairs_tfs:
        date_cache[(pair, tf)] = load_date_index(pair, tf)

    # Second pass: load data with year tags
    print("Loading trades with year tags...")
    data = []
    float_cols = [
        'a_amp', 'a_bars', 'a_slope', 'a_dir',
        'b_amp', 'b_bars', 'b_slope',
        'amp_ratio', 'time_ratio', 'slope_ratio',
        'c_amp', 'c_bars', 'c_dir', 'c_follows_a', 'c_a_ratio',
        'pnl', 'pnl_r', 'max_favorable', 'hold_bars',
        'tp_distance', 'sl_distance',
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

            # Tag with year
            entry_bar = int(r['entry_bar'])
            dm = date_cache.get((r['pair'], r['tf']), {})
            r['year'] = dm.get(entry_bar, 0)
            data.append(r)

    return data


def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def oos_split(data, train_end=2018):
    """Split into IS (<=train_end) and OOS (>train_end)."""
    is_data = [d for d in data if 0 < d['year'] <= train_end]
    oos_data = [d for d in data if d['year'] > train_end]
    return is_data, oos_data


def compute_stats(trades):
    """Compute trading stats for a list of trades."""
    if not trades:
        return {'n': 0}
    pnls = [t['pnl_r'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
    return {
        'n': len(pnls),
        'wr': len(wins) / len(pnls) * 100,
        'avg_r': np.mean(pnls),
        'med_r': np.median(pnls),
        'std_r': np.std(pnls),
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
        'pf': pf,
    }


def print_stats(label, stats):
    if stats['n'] == 0:
        print(f"  {label}: NO DATA")
        return
    print(f"  {label}: n={stats['n']:>8,}  WR={stats['wr']:.1f}%  "
          f"avg_R={stats['avg_r']:.4f}  PF={stats['pf']:.2f}  "
          f"avg_win={stats['avg_win']:.3f}  avg_loss={stats['avg_loss']:.3f}")


def validate_oos(data):
    """Main OOS validation."""
    is_data, oos_data = oos_split(data, train_end=2018)

    print_section("TIME-SERIES OOS SPLIT: Train ≤2018, Test 2019-2024")
    print_stats("IS  (2000-2018)", compute_stats(is_data))
    print_stats("OOS (2019-2024)", compute_stats(oos_data))

    # By year
    print_section("PERFORMANCE BY YEAR")
    years = sorted(set(d['year'] for d in data if d['year'] > 0))
    for y in years:
        yd = [d for d in data if d['year'] == y]
        s = compute_stats(yd)
        marker = " <<<OOS" if y > 2018 else ""
        print_stats(f"{y}{marker}", s)

    # Key filters: check if they hold OOS
    print_section("KEY FILTERS: IS vs OOS STABILITY")

    filters = {
        'ALL (no filter)': lambda d: True,
        'slope_ratio < 0.1': lambda d: d['slope_ratio'] < 0.1,
        'slope_ratio < 0.3': lambda d: d['slope_ratio'] < 0.3,
        'slope_ratio < 0.5': lambda d: d['slope_ratio'] < 0.5,
        'amp >= 0.8/time': lambda d: d['amp_ratio'] >= 0.8 / d['time_ratio'] if d['time_ratio'] > 0 else False,
        'slope<0.3 & amp>=0.8/t': lambda d: d['slope_ratio'] < 0.3 and (d['amp_ratio'] >= 0.8 / d['time_ratio'] if d['time_ratio'] > 0 else False),
        'dev=1.0 confirm=5': lambda d: d['zz_dev'] == 1.0 and d['zz_confirm'] == 5,
        'dev=1.0 & slope<0.3': lambda d: d['zz_dev'] == 1.0 and d['slope_ratio'] < 0.3,
    }

    print(f"\n{'Filter':>30s} | {'--- IS (2000-2018) ---':^36s} | {'--- OOS (2019-2024) ---':^36s} | {'Stable?':>7s}")
    print(f"{'':>30s} | {'n':>8s} {'WR%':>6s} {'avgR':>7s} {'PF':>6s} | {'n':>8s} {'WR%':>6s} {'avgR':>7s} {'PF':>6s} |")
    print("-" * 115)

    for name, filt in filters.items():
        is_f = [d for d in is_data if filt(d)]
        oos_f = [d for d in oos_data if filt(d)]
        is_s = compute_stats(is_f)
        oos_s = compute_stats(oos_f)

        if is_s['n'] < 50 or oos_s['n'] < 50:
            continue

        # Stability: OOS avg_R within 50% of IS avg_R
        if is_s['avg_r'] > 0:
            ratio = oos_s['avg_r'] / is_s['avg_r']
            stable = "YES" if 0.5 < ratio < 2.0 else "WEAK" if 0.3 < ratio < 3.0 else "NO"
        else:
            stable = "N/A"

        print(f"{name:>30s} | {is_s['n']:>8,} {is_s['wr']:>6.1f} {is_s['avg_r']:>7.4f} {is_s['pf']:>6.2f} | "
              f"{oos_s['n']:>8,} {oos_s['wr']:>6.1f} {oos_s['avg_r']:>7.4f} {oos_s['pf']:>6.2f} | {stable:>7s}")


def feature_importance(data):
    """Rank features by their ability to separate good trades from bad."""
    print_section("FEATURE IMPORTANCE: Correlation with PnL(R)")

    features = ['slope_ratio', 'amp_ratio', 'time_ratio', 'a_amp', 'a_bars',
                'b_amp', 'b_bars', 'b_slope', 'a_slope']

    is_data, oos_data = oos_split(data, train_end=2018)

    results = []
    for feat in features:
        vals_is = np.array([d[feat] for d in is_data])
        pnls_is = np.array([d['pnl_r'] for d in is_data])
        vals_oos = np.array([d[feat] for d in oos_data])
        pnls_oos = np.array([d['pnl_r'] for d in oos_data])

        # Clip outliers for correlation
        mask_is = np.abs(vals_is) < np.percentile(np.abs(vals_is), 99)
        mask_oos = np.abs(vals_oos) < np.percentile(np.abs(vals_oos), 99)

        corr_is = np.corrcoef(vals_is[mask_is], pnls_is[mask_is])[0, 1]
        corr_oos = np.corrcoef(vals_oos[mask_oos], pnls_oos[mask_oos])[0, 1]

        results.append((feat, corr_is, corr_oos))

    # Sort by abs IS correlation
    results.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'Feature':>15s} {'Corr(IS)':>10s} {'Corr(OOS)':>10s} {'Same sign?':>12s} {'Rank stable?':>12s}")
    print("-" * 65)
    for feat, ci, co in results:
        same_sign = "YES" if (ci > 0) == (co > 0) else "NO"
        stable = "YES" if abs(co) > abs(ci) * 0.3 else "WEAK"
        print(f"{feat:>15s} {ci:>10.4f} {co:>10.4f} {same_sign:>12s} {stable:>12s}")


def feature_binned_analysis(data):
    """Binned analysis for top features — check for monotonic relationships."""
    print_section("BINNED FEATURE ANALYSIS (IS vs OOS)")

    is_data, oos_data = oos_split(data, train_end=2018)

    # slope_ratio
    print(f"\n--- slope_ratio (B weakness indicator) ---")
    bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0, 100.0]
    print(f"{'Bin':>14s} | {'IS n':>8s} {'IS WR':>6s} {'IS avgR':>8s} {'IS PF':>7s} | "
          f"{'OOS n':>8s} {'OOS WR':>6s} {'OOS avgR':>8s} {'OOS PF':>7s}")
    print("-" * 95)

    for i in range(len(bins) - 1):
        is_f = [d for d in is_data if bins[i] <= d['slope_ratio'] < bins[i+1]]
        oos_f = [d for d in oos_data if bins[i] <= d['slope_ratio'] < bins[i+1]]
        is_s = compute_stats(is_f)
        oos_s = compute_stats(oos_f)
        if is_s['n'] < 50:
            continue
        label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
        oos_str = (f"{oos_s['n']:>8,} {oos_s['wr']:>6.1f} {oos_s['avg_r']:>8.4f} {oos_s['pf']:>7.2f}"
                   if oos_s['n'] >= 50 else "  insufficient data")
        print(f"{label:>14s} | {is_s['n']:>8,} {is_s['wr']:>6.1f} {is_s['avg_r']:>8.4f} {is_s['pf']:>7.2f} | {oos_str}")

    # amp_ratio
    print(f"\n--- amp_ratio (B/A amplitude) ---")
    bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.15, 1.5, 2.0, 5.0]
    print(f"{'Bin':>14s} | {'IS n':>8s} {'IS WR':>6s} {'IS avgR':>8s} {'IS PF':>7s} | "
          f"{'OOS n':>8s} {'OOS WR':>6s} {'OOS avgR':>8s} {'OOS PF':>7s}")
    print("-" * 95)

    for i in range(len(bins) - 1):
        is_f = [d for d in is_data if bins[i] <= d['amp_ratio'] < bins[i+1]]
        oos_f = [d for d in oos_data if bins[i] <= d['amp_ratio'] < bins[i+1]]
        is_s = compute_stats(is_f)
        oos_s = compute_stats(oos_f)
        if is_s['n'] < 50:
            continue
        label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
        oos_str = (f"{oos_s['n']:>8,} {oos_s['wr']:>6.1f} {oos_s['avg_r']:>8.4f} {oos_s['pf']:>7.2f}"
                   if oos_s['n'] >= 50 else "  insufficient data")
        print(f"{label:>14s} | {is_s['n']:>8,} {is_s['wr']:>6.1f} {is_s['avg_r']:>8.4f} {is_s['pf']:>7.2f} | {oos_str}")

    # time_ratio
    print(f"\n--- time_ratio (B/A duration) ---")
    bins = [0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0, 5.0, 10.0, 50.0]
    print(f"{'Bin':>14s} | {'IS n':>8s} {'IS WR':>6s} {'IS avgR':>8s} {'IS PF':>7s} | "
          f"{'OOS n':>8s} {'OOS WR':>6s} {'OOS avgR':>8s} {'OOS PF':>7s}")
    print("-" * 95)

    for i in range(len(bins) - 1):
        is_f = [d for d in is_data if bins[i] <= d['time_ratio'] < bins[i+1]]
        oos_f = [d for d in oos_data if bins[i] <= d['time_ratio'] < bins[i+1]]
        is_s = compute_stats(is_f)
        oos_s = compute_stats(oos_f)
        if is_s['n'] < 50:
            continue
        label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
        oos_str = (f"{oos_s['n']:>8,} {oos_s['wr']:>6.1f} {oos_s['avg_r']:>8.4f} {oos_s['pf']:>7.2f}"
                   if oos_s['n'] >= 50 else "  insufficient data")
        print(f"{label:>14s} | {is_s['n']:>8,} {is_s['wr']:>6.1f} {is_s['avg_r']:>8.4f} {is_s['pf']:>7.2f} | {oos_str}")


def cross_pair_stability(data):
    """Check if edge is consistent across pairs or concentrated in a few."""
    print_section("CROSS-PAIR STABILITY (slope_ratio < 0.3)")

    is_data, oos_data = oos_split(data, train_end=2018)

    filt = lambda d: d['slope_ratio'] < 0.3

    # By pair
    pair_stats = defaultdict(lambda: {'is': [], 'oos': []})
    for d in is_data:
        if filt(d):
            pair_stats[d['pair']]['is'].append(d['pnl_r'])
    for d in oos_data:
        if filt(d):
            pair_stats[d['pair']]['oos'].append(d['pnl_r'])

    print(f"{'Pair':>10s} | {'IS n':>7s} {'IS WR':>6s} {'IS avgR':>8s} | {'OOS n':>7s} {'OOS WR':>6s} {'OOS avgR':>8s} | {'Both+?':>6s}")
    print("-" * 75)

    both_positive = 0
    total_pairs = 0
    for pair in sorted(pair_stats.keys()):
        is_pnls = pair_stats[pair]['is']
        oos_pnls = pair_stats[pair]['oos']
        if len(is_pnls) < 30:
            continue
        total_pairs += 1
        is_s = compute_stats([{'pnl_r': p} for p in is_pnls])
        oos_s = compute_stats([{'pnl_r': p} for p in oos_pnls]) if len(oos_pnls) >= 30 else {'n': 0, 'wr': 0, 'avg_r': 0}

        both = "YES" if is_s['avg_r'] > 0 and oos_s.get('avg_r', 0) > 0 else "NO"
        if both == "YES":
            both_positive += 1

        oos_str = f"{oos_s['n']:>7,} {oos_s['wr']:>6.1f} {oos_s['avg_r']:>8.4f}" if oos_s['n'] > 0 else "  insufficient"
        print(f"{pair:>10s} | {is_s['n']:>7,} {is_s['wr']:>6.1f} {is_s['avg_r']:>8.4f} | {oos_str} | {both:>6s}")

    print(f"\nPairs with positive avg_R in BOTH IS and OOS: {both_positive}/{total_pairs} ({both_positive/total_pairs*100:.0f}%)")


def main():
    data = load_data_with_dates()
    print(f"Loaded {len(data):,} trades")

    # Check year distribution
    years = [d['year'] for d in data if d['year'] > 0]
    print(f"Year range: {min(years)}-{max(years)}")
    print(f"Trades with year tag: {len(years):,} / {len(data):,}")

    validate_oos(data)
    feature_importance(data)
    feature_binned_analysis(data)
    cross_pair_stability(data)

    print(f"\n{'='*80}")
    print("  VALIDATION COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
