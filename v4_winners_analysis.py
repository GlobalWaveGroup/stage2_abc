"""
V4 Winners Deep Analysis
========================
V4 had 3.16M trades, 97% hit SL, but 2.8% hit TP with avgR=2.89.
The TP trades are valuable signal — what makes them different?

Also: analyze what "head-start equivalent" conditions look like.
The V1 confirmation delay gave ~22% of A_amp head start.
Can we find condition combinations that replicate this?

Core idea from user:
- zigzag(2,1,1) is the base measurement tool
- The "turn" confirmation is just ONE condition in the matrix
- When other conditions are strong enough, we can relax the turn requirement
- 79.5% bias = real price info that conditions can capture partially
"""

import csv
import numpy as np
from collections import defaultdict

def load_v4_trades():
    """Load V4 trades from CSV."""
    trades = []
    with open('/home/ubuntu/stage2_abc/abc_v4_trades.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = {}
            for k, v in row.items():
                try:
                    t[k] = float(v)
                except ValueError:
                    t[k] = v
            trades.append(t)
    return trades


def analyze_winners_vs_losers(trades):
    """Deep comparison of TP-exits vs SL-exits."""
    
    tp_trades = [t for t in trades if t['exit_reason'] == 'tp']
    sl_trades = [t for t in trades if t['exit_reason'] == 'sl']
    dyn_trades = [t for t in trades if t['exit_reason'] == 'dynamic']
    
    print(f"{'='*90}")
    print(f"  V4 WINNERS vs LOSERS DEEP ANALYSIS")
    print(f"{'='*90}")
    print(f"  Total: {len(trades):,}")
    print(f"  TP exits:      {len(tp_trades):>10,} ({len(tp_trades)/len(trades)*100:.1f}%)")
    print(f"  Dynamic exits:  {len(dyn_trades):>10,} ({len(dyn_trades)/len(trades)*100:.1f}%)")
    print(f"  SL exits:      {len(sl_trades):>10,} ({len(sl_trades)/len(trades)*100:.1f}%)")
    
    # ── Feature comparison ──
    features = ['b_depth_ratio', 'b_time_ratio', 'b_slope_ratio', 'b_decel_ratio', 'score']
    
    print(f"\n{'='*90}")
    print(f"  FEATURE DISTRIBUTION: TP vs SL exits")
    print(f"{'='*90}")
    print(f"{'Feature':>18s} {'TP mean':>10s} {'TP med':>10s} {'SL mean':>10s} {'SL med':>10s} {'Dyn mean':>10s} {'Separation':>10s}")
    print("-" * 80)
    
    for feat in features:
        tp_vals = np.array([t[feat] for t in tp_trades])
        sl_vals = np.array([t[feat] for t in sl_trades])
        dyn_vals = np.array([t[feat] for t in dyn_trades]) if dyn_trades else np.array([0])
        
        # Separation: Cohen's d
        pooled_std = np.sqrt((np.var(tp_vals) + np.var(sl_vals)) / 2)
        cohens_d = (np.mean(tp_vals) - np.mean(sl_vals)) / pooled_std if pooled_std > 0 else 0
        
        print(f"{feat:>18s} {np.mean(tp_vals):>10.4f} {np.median(tp_vals):>10.4f} "
              f"{np.mean(sl_vals):>10.4f} {np.median(sl_vals):>10.4f} "
              f"{np.mean(dyn_vals):>10.4f} {cohens_d:>+10.4f}")
    
    # ── What B_depth_ratio ranges produce best results? ──
    print(f"\n{'='*90}")
    print(f"  BY b_depth_ratio BUCKET — which retracement levels work?")
    print(f"{'='*90}")
    
    depth_bins = [(0.10, 0.20), (0.20, 0.30), (0.30, 0.382), (0.382, 0.50), 
                  (0.50, 0.618), (0.618, 0.786), (0.786, 1.00), (1.00, 1.50)]
    
    print(f"{'depth':>14s} {'n':>8s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s} {'TP%':>6s}")
    print("-" * 55)
    for lo, hi in depth_bins:
        sub = [t for t in trades if lo <= t['b_depth_ratio'] < hi]
        if len(sub) < 1000: continue
        p = np.array([t['pnl_r'] for t in sub])
        w = p[p > 0]; l = p[p <= 0]
        pf = abs(np.sum(w)/np.sum(l)) if np.sum(l) != 0 else float('inf')
        tp_pct = len([t for t in sub if t['exit_reason'] == 'tp']) / len(sub) * 100
        label = f"{lo:.3f}-{hi:.3f}"
        print(f"{label:>14s} {len(p):>8,} {np.sum(p>0)/len(p)*100:>6.1f} {np.mean(p):>8.4f} {pf:>7.2f} {tp_pct:>6.1f}")
    
    # ── The key question: can conditions CREATE positive expectancy? ──
    print(f"\n{'='*90}")
    print(f"  CONDITION MATRIX SEARCH — finding positive-expectancy zones")
    print(f"{'='*90}")
    print(f"  Logic: find combinations where OTHER conditions compensate for lack of confirmation")
    print(f"  The 79.5% head-start means: at V1 entry, price had moved ~22% of A_amp toward C")
    print(f"  Question: can slope/decel/time/depth conditions identify similar high-probability zones?")
    
    # Systematic grid search
    best_combos = []
    
    slope_cuts = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    depth_ranges = [(0.30, 0.50), (0.382, 0.618), (0.50, 0.786), (0.20, 0.50), (0.30, 0.70)]
    time_cuts = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    decel_cuts = [0.2, 0.3, 0.4, 0.5, 0.7]
    
    print(f"\n  Phase 1: Single strong conditions")
    print(f"{'Condition':>45s} {'n':>8s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s}")
    print("-" * 78)
    
    # Single conditions
    single_conditions = []
    for sc in slope_cuts:
        single_conditions.append((f'slope_ratio < {sc}', lambda t, s=sc: t['b_slope_ratio'] < s))
    for tc in time_cuts:
        single_conditions.append((f'time_ratio > {tc}', lambda t, c=tc: t['b_time_ratio'] > c))
    for dc in decel_cuts:
        single_conditions.append((f'decel_ratio < {dc}', lambda t, c=dc: t['b_decel_ratio'] < c))
    for lo, hi in depth_ranges:
        single_conditions.append((f'depth [{lo:.3f}-{hi:.3f}]', lambda t, l=lo, h=hi: l <= t['b_depth_ratio'] < h))
    
    for name, filt in single_conditions:
        sub = [t for t in trades if filt(t)]
        if len(sub) < 500: continue
        p = np.array([t['pnl_r'] for t in sub])
        w = p[p > 0]; l = p[p <= 0]
        pf = abs(np.sum(w)/np.sum(l)) if np.sum(l) != 0 else float('inf')
        avgr = np.mean(p)
        wr = np.sum(p > 0) / len(p) * 100
        print(f"{name:>45s} {len(p):>8,} {wr:>6.1f} {avgr:>8.4f} {pf:>7.2f}")
        if avgr > -0.10:
            best_combos.append((name, len(p), wr, avgr, pf))
    
    # ── Phase 2: Two-condition combos ──
    print(f"\n  Phase 2: Two-condition combos (best single conditions)")
    print(f"{'Condition':>55s} {'n':>8s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s}")
    print("-" * 88)
    
    two_combos = [
        ('slope<0.10 & time>2.0', 
         lambda t: t['b_slope_ratio'] < 0.10 and t['b_time_ratio'] > 2.0),
        ('slope<0.10 & decel<0.3',
         lambda t: t['b_slope_ratio'] < 0.10 and t['b_decel_ratio'] < 0.3),
        ('slope<0.10 & depth[0.38-0.62]',
         lambda t: t['b_slope_ratio'] < 0.10 and 0.382 <= t['b_depth_ratio'] < 0.618),
        ('slope<0.15 & time>3.0',
         lambda t: t['b_slope_ratio'] < 0.15 and t['b_time_ratio'] > 3.0),
        ('slope<0.15 & decel<0.3',
         lambda t: t['b_slope_ratio'] < 0.15 and t['b_decel_ratio'] < 0.3),
        ('slope<0.15 & depth[0.38-0.62]',
         lambda t: t['b_slope_ratio'] < 0.15 and 0.382 <= t['b_depth_ratio'] < 0.618),
        ('slope<0.20 & time>3.0 ',
         lambda t: t['b_slope_ratio'] < 0.20 and t['b_time_ratio'] > 3.0),
        ('slope<0.20 & decel<0.3',
         lambda t: t['b_slope_ratio'] < 0.20 and t['b_decel_ratio'] < 0.3),
        ('time>3.0 & decel<0.3',
         lambda t: t['b_time_ratio'] > 3.0 and t['b_decel_ratio'] < 0.3),
        ('time>5.0 & decel<0.3',
         lambda t: t['b_time_ratio'] > 5.0 and t['b_decel_ratio'] < 0.3),
        ('time>3.0 & depth[0.38-0.62]',
         lambda t: t['b_time_ratio'] > 3.0 and 0.382 <= t['b_depth_ratio'] < 0.618),
        ('decel<0.2 & depth[0.38-0.62]',
         lambda t: t['b_decel_ratio'] < 0.2 and 0.382 <= t['b_depth_ratio'] < 0.618),
        ('slope<0.05 & time>1.0',
         lambda t: t['b_slope_ratio'] < 0.05 and t['b_time_ratio'] > 1.0),
        ('slope<0.05 & decel<0.5',
         lambda t: t['b_slope_ratio'] < 0.05 and t['b_decel_ratio'] < 0.5),
    ]
    
    for name, filt in two_combos:
        sub = [t for t in trades if filt(t)]
        if len(sub) < 200: continue
        p = np.array([t['pnl_r'] for t in sub])
        w = p[p > 0]; l = p[p <= 0]
        pf = abs(np.sum(w)/np.sum(l)) if np.sum(l) != 0 else float('inf')
        avgr = np.mean(p)
        wr = np.sum(p > 0) / len(p) * 100
        print(f"{name:>55s} {len(p):>8,} {wr:>6.1f} {avgr:>8.4f} {pf:>7.2f}")
        if avgr > -0.05:
            best_combos.append((name, len(p), wr, avgr, pf))
    
    # ── Phase 3: Three-condition combos ──
    print(f"\n  Phase 3: Three-condition combos")
    print(f"{'Condition':>65s} {'n':>8s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s}")
    print("-" * 98)
    
    three_combos = [
        ('slope<0.10 & time>2.0 & decel<0.3',
         lambda t: t['b_slope_ratio'] < 0.10 and t['b_time_ratio'] > 2.0 and t['b_decel_ratio'] < 0.3),
        ('slope<0.10 & time>3.0 & decel<0.4',
         lambda t: t['b_slope_ratio'] < 0.10 and t['b_time_ratio'] > 3.0 and t['b_decel_ratio'] < 0.4),
        ('slope<0.10 & time>2.0 & depth[0.38-0.62]',
         lambda t: t['b_slope_ratio'] < 0.10 and t['b_time_ratio'] > 2.0 and 0.382 <= t['b_depth_ratio'] < 0.618),
        ('slope<0.15 & time>3.0 & decel<0.3',
         lambda t: t['b_slope_ratio'] < 0.15 and t['b_time_ratio'] > 3.0 and t['b_decel_ratio'] < 0.3),
        ('slope<0.15 & time>2.0 & depth[0.38-0.62]',
         lambda t: t['b_slope_ratio'] < 0.15 and t['b_time_ratio'] > 2.0 and 0.382 <= t['b_depth_ratio'] < 0.618),
        ('slope<0.15 & decel<0.3 & depth[0.30-0.62]',
         lambda t: t['b_slope_ratio'] < 0.15 and t['b_decel_ratio'] < 0.3 and 0.30 <= t['b_depth_ratio'] < 0.618),
        ('slope<0.20 & time>3.0 & decel<0.3',
         lambda t: t['b_slope_ratio'] < 0.20 and t['b_time_ratio'] > 3.0 and t['b_decel_ratio'] < 0.3),
        ('slope<0.05 & time>1.5 & decel<0.5',
         lambda t: t['b_slope_ratio'] < 0.05 and t['b_time_ratio'] > 1.5 and t['b_decel_ratio'] < 0.5),
        ('slope<0.05 & time>2.0 & depth[0.30-0.70]',
         lambda t: t['b_slope_ratio'] < 0.05 and t['b_time_ratio'] > 2.0 and 0.30 <= t['b_depth_ratio'] < 0.70),
        ('slope<0.10 & decel<0.2 & depth[0.30-0.70]',
         lambda t: t['b_slope_ratio'] < 0.10 and t['b_decel_ratio'] < 0.2 and 0.30 <= t['b_depth_ratio'] < 0.70),
    ]
    
    for name, filt in three_combos:
        sub = [t for t in trades if filt(t)]
        if len(sub) < 100: continue
        p = np.array([t['pnl_r'] for t in sub])
        w = p[p > 0]; l = p[p <= 0]
        pf = abs(np.sum(w)/np.sum(l)) if np.sum(l) != 0 else float('inf')
        avgr = np.mean(p)
        wr = np.sum(p > 0) / len(p) * 100
        print(f"{name:>65s} {len(p):>8,} {wr:>6.1f} {avgr:>8.4f} {pf:>7.2f}")
        if avgr > 0:
            best_combos.append(('***' + name, len(p), wr, avgr, pf))
        elif avgr > -0.05:
            best_combos.append((name, len(p), wr, avgr, pf))
    
    # ── Phase 4: Four-condition combos (the full matrix) ──
    print(f"\n  Phase 4: Four-condition combos (full matrix)")
    print(f"{'Condition':>75s} {'n':>8s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s}")
    print("-" * 108)
    
    four_combos = [
        ('slope<0.10 & time>2.0 & decel<0.3 & depth[0.38-0.62]',
         lambda t: t['b_slope_ratio'] < 0.10 and t['b_time_ratio'] > 2.0 and t['b_decel_ratio'] < 0.3 and 0.382 <= t['b_depth_ratio'] < 0.618),
        ('slope<0.10 & time>3.0 & decel<0.4 & depth[0.30-0.70]',
         lambda t: t['b_slope_ratio'] < 0.10 and t['b_time_ratio'] > 3.0 and t['b_decel_ratio'] < 0.4 and 0.30 <= t['b_depth_ratio'] < 0.70),
        ('slope<0.15 & time>2.0 & decel<0.3 & depth[0.38-0.62]',
         lambda t: t['b_slope_ratio'] < 0.15 and t['b_time_ratio'] > 2.0 and t['b_decel_ratio'] < 0.3 and 0.382 <= t['b_depth_ratio'] < 0.618),
        ('slope<0.15 & time>3.0 & decel<0.4 & depth[0.30-0.70]',
         lambda t: t['b_slope_ratio'] < 0.15 and t['b_time_ratio'] > 3.0 and t['b_decel_ratio'] < 0.4 and 0.30 <= t['b_depth_ratio'] < 0.70),
        ('slope<0.05 & time>1.5 & decel<0.4 & depth[0.38-0.62]',
         lambda t: t['b_slope_ratio'] < 0.05 and t['b_time_ratio'] > 1.5 and t['b_decel_ratio'] < 0.4 and 0.382 <= t['b_depth_ratio'] < 0.618),
        ('slope<0.05 & time>2.0 & decel<0.3 & depth[0.30-0.62]',
         lambda t: t['b_slope_ratio'] < 0.05 and t['b_time_ratio'] > 2.0 and t['b_decel_ratio'] < 0.3 and 0.30 <= t['b_depth_ratio'] < 0.618),
        ('slope<0.10 & time>5.0 & decel<0.5 & depth[0.30-0.70]',
         lambda t: t['b_slope_ratio'] < 0.10 and t['b_time_ratio'] > 5.0 and t['b_decel_ratio'] < 0.5 and 0.30 <= t['b_depth_ratio'] < 0.70),
    ]
    
    for name, filt in four_combos:
        sub = [t for t in trades if filt(t)]
        if len(sub) < 50: continue
        p = np.array([t['pnl_r'] for t in sub])
        w = p[p > 0]; l = p[p <= 0]
        pf = abs(np.sum(w)/np.sum(l)) if np.sum(l) != 0 else float('inf')
        avgr = np.mean(p)
        wr = np.sum(p > 0) / len(p) * 100
        print(f"{name:>75s} {len(p):>8,} {wr:>6.1f} {avgr:>8.4f} {pf:>7.2f}")
        if avgr > 0:
            best_combos.append(('****' + name, len(p), wr, avgr, pf))
        elif avgr > -0.05:
            best_combos.append((name, len(p), wr, avgr, pf))
    
    # ── Summary of best combos ──
    if best_combos:
        print(f"\n{'='*90}")
        print(f"  BEST COMBOS SUMMARY (avgR > -0.10 or positive)")
        print(f"{'='*90}")
        best_combos.sort(key=lambda x: x[3], reverse=True)
        print(f"{'Condition':>65s} {'n':>8s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s}")
        print("-" * 98)
        for name, n, wr, avgr, pf in best_combos[:30]:
            marker = " <<<POSITIVE>>>" if avgr > 0 else ""
            print(f"{name:>65s} {n:>8,} {wr:>6.1f} {avgr:>8.4f} {pf:>7.2f}{marker}")
    
    # ── OOS check for best combos ──
    if best_combos:
        print(f"\n{'='*90}")
        print(f"  OOS VALIDATION of top combos")
        print(f"{'='*90}")
        # Re-evaluate top combos on IS vs OOS
        # But we need the filter functions... let's just check IS vs OOS split
        is_trades = [t for t in trades if t['entry_year'] <= 2018]
        oos_trades = [t for t in trades if t['entry_year'] > 2018]
        
        # Re-do a few key combos with IS/OOS split
        key_tests = [
            ('slope<0.10 & time>3.0 & decel<0.4 & depth[0.30-0.70]',
             lambda t: t['b_slope_ratio'] < 0.10 and t['b_time_ratio'] > 3.0 and t['b_decel_ratio'] < 0.4 and 0.30 <= t['b_depth_ratio'] < 0.70),
            ('slope<0.10 & time>2.0 & decel<0.3',
             lambda t: t['b_slope_ratio'] < 0.10 and t['b_time_ratio'] > 2.0 and t['b_decel_ratio'] < 0.3),
            ('slope<0.05 & time>2.0 & depth[0.30-0.70]',
             lambda t: t['b_slope_ratio'] < 0.05 and t['b_time_ratio'] > 2.0 and 0.30 <= t['b_depth_ratio'] < 0.70),
            ('slope<0.10 & decel<0.2 & depth[0.30-0.70]',
             lambda t: t['b_slope_ratio'] < 0.10 and t['b_decel_ratio'] < 0.2 and 0.30 <= t['b_depth_ratio'] < 0.70),
            ('slope<0.05 & time>1.5 & decel<0.4 & depth[0.38-0.62]',
             lambda t: t['b_slope_ratio'] < 0.05 and t['b_time_ratio'] > 1.5 and t['b_decel_ratio'] < 0.4 and 0.382 <= t['b_depth_ratio'] < 0.618),
        ]
        
        print(f"\n{'Condition':>65s} | {'IS n':>7s} {'IS avgR':>8s} {'IS PF':>7s} | {'OOS n':>7s} {'OOS avgR':>8s} {'OOS PF':>7s}")
        print("-" * 120)
        for name, filt in key_tests:
            is_sub = [t for t in is_trades if filt(t)]
            oos_sub = [t for t in oos_trades if filt(t)]
            
            if len(is_sub) < 50 or len(oos_sub) < 20: continue
            
            is_p = np.array([t['pnl_r'] for t in is_sub])
            oos_p = np.array([t['pnl_r'] for t in oos_sub])
            
            is_w = is_p[is_p > 0]; is_l = is_p[is_p <= 0]
            oos_w = oos_p[oos_p > 0]; oos_l = oos_p[oos_p <= 0]
            
            is_pf = abs(np.sum(is_w)/np.sum(is_l)) if np.sum(is_l) != 0 else float('inf')
            oos_pf = abs(np.sum(oos_w)/np.sum(oos_l)) if np.sum(oos_l) != 0 else float('inf')
            
            print(f"{name:>65s} | {len(is_p):>7,} {np.mean(is_p):>8.4f} {is_pf:>7.2f} | "
                  f"{len(oos_p):>7,} {np.mean(oos_p):>8.4f} {oos_pf:>7.2f}")


    # ── Analysis by merge level for best conditions ──
    print(f"\n{'='*90}")
    print(f"  BEST CONDITIONS × MERGE LEVEL × TF")
    print(f"{'='*90}")
    
    # Use the strongest condition found
    strong_cond = lambda t: t['b_slope_ratio'] < 0.10 and t['b_time_ratio'] > 2.0 and t['b_decel_ratio'] < 0.3
    strong_sub = [t for t in trades if strong_cond(t)]
    
    if strong_sub:
        print(f"\n  Condition: slope<0.10 & time>2.0 & decel<0.3")
        print(f"{'ML':>4s} {'TF':>4s} {'n':>7s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s}")
        print("-" * 40)
        for ml in [2, 3, 4, 5]:
            for tf in ['H1', 'M30', 'M15']:
                sub = [t for t in strong_sub if t['merge_level'] == ml and t['tf'] == tf]
                if len(sub) < 30: continue
                p = np.array([t['pnl_r'] for t in sub])
                w = p[p > 0]; l = p[p <= 0]
                pf = abs(np.sum(w)/np.sum(l)) if np.sum(l) != 0 else float('inf')
                print(f"  L{ml:<3.0f} {tf:>4s} {len(p):>7,} {np.sum(p>0)/len(p)*100:>6.1f} {np.mean(p):>8.4f} {pf:>7.2f}")


def main():
    print("Loading V4 trades...")
    trades = load_v4_trades()
    print(f"Loaded {len(trades):,} trades")
    
    analyze_winners_vs_losers(trades)


if __name__ == '__main__':
    main()
