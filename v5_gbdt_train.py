"""
Route 1: GBDT on 46.8M V5 samples
===================================
Train LightGBM to learn soft condition boundaries.

Label: binary classification
  - positive = MFE/MAE > threshold (directional edge exists)
  - We test multiple thresholds to find optimal

Features: all condition dimensions from V5
  - b_depth_ratio, b_time_ratio, b_slope_ratio, b_decel_ratio
  - turn_bars, turn_depth_ratio
  - close_bullish, entry_from_extreme
  - merge_level

Key outputs:
  1. Feature importance ranking (which dims matter most?)
  2. SHAP-style interaction analysis (which combos matter?)
  3. Probability calibration → entry score
  4. IS/OOS validation (2000-2018 train, 2019-2024 test)
  5. Backtest: use model probability as entry filter
"""

import numpy as np
import csv
import os
import time

def load_data(path):
    """Load V5 CSV into numpy arrays. Memory-efficient: read in chunks."""
    print(f"Loading {path}...")
    t0 = time.time()
    
    # First pass: count lines
    with open(path, 'r') as f:
        header = f.readline().strip().split(',')
        n_lines = sum(1 for _ in f)
    
    print(f"  {n_lines:,} rows, {len(header)} columns")
    
    # Pre-allocate arrays
    n_feat = len(header)
    data = np.zeros((n_lines, n_feat), dtype=np.float32)
    str_cols = {}  # for non-numeric columns
    
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in enumerate(reader):
            for j, v in enumerate(row):
                try:
                    data[i, j] = float(v)
                except ValueError:
                    data[i, j] = 0  # placeholder
                    if j not in str_cols:
                        str_cols[j] = []
                    # only track if needed
            if (i + 1) % 5_000_000 == 0:
                print(f"  loaded {i+1:,} rows...")
    
    print(f"  Done in {time.time()-t0:.1f}s, shape={data.shape}")
    return header, data


def main():
    path = "/home/ubuntu/stage2_abc/abc_v5_samples.csv"
    header, data = load_data(path)
    
    # Column indices
    col = {name: i for i, name in enumerate(header)}
    
    # Features
    feat_names = ['merge_level', 'b_depth_ratio', 'b_time_ratio', 'b_slope_ratio',
                  'b_decel_ratio', 'turn_bars', 'turn_depth_ratio',
                  'close_bullish', 'entry_from_extreme']
    feat_idx = [col[f] for f in feat_names]
    
    X = data[:, feat_idx]
    mfe = data[:, col['mfe_r']]
    mae = data[:, col['mae_r']]
    years = data[:, col['year']]
    
    # IS/OOS split
    is_mask = years <= 2018
    oos_mask = years > 2018
    
    print(f"\nIS: {np.sum(is_mask):,}, OOS: {np.sum(oos_mask):,}")
    
    # ═══ Try importing LightGBM ═══
    try:
        import lightgbm as lgb
        use_lgb = True
        print("Using LightGBM")
    except ImportError:
        try:
            from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
            use_lgb = False
            print("LightGBM not available, using sklearn HistGradientBoosting")
        except ImportError:
            print("ERROR: No gradient boosting library available!")
            return
    
    # ═══ Multiple label definitions ═══
    label_defs = [
        ("MFE>MAE (any edge)", mfe > mae),
        ("MFE>MAE*1.2", mfe > mae * 1.2),
        ("MFE>MAE*1.5", mfe > mae * 1.5),
        ("MFE>MAE*2.0", mfe > mae * 2.0),
        ("MFE>0.5A & MAE<0.35A", (mfe > 0.5) & (mae < 0.35)),
        ("MFE>0.6A & MAE<0.35A", (mfe > 0.6) & (mae < 0.35)),
    ]
    
    print(f"\n{'='*90}")
    print(f"  LABEL DISTRIBUTION")
    print(f"{'='*90}")
    for name, label in label_defs:
        pos_rate = np.mean(label) * 100
        pos_is = np.mean(label[is_mask]) * 100
        pos_oos = np.mean(label[oos_mask]) * 100
        print(f"  {name:>30s}: {pos_rate:.1f}% (IS={pos_is:.1f}%, OOS={pos_oos:.1f}%)")
    
    # ═══ Train on primary label: MFE > MAE * 1.5 ═══
    # This means "price moved 50% more in our favor than against us"
    primary_label = (mfe > mae * 1.5).astype(int)
    
    X_is = X[is_mask]
    y_is = primary_label[is_mask]
    X_oos = X[oos_mask]
    y_oos = primary_label[oos_mask]
    
    print(f"\n{'='*90}")
    print(f"  TRAINING GBDT: label = MFE > MAE * 1.5")
    print(f"  IS: {len(X_is):,} samples, pos_rate={np.mean(y_is)*100:.1f}%")
    print(f"  OOS: {len(X_oos):,} samples, pos_rate={np.mean(y_oos)*100:.1f}%")
    print(f"{'='*90}")
    
    # Subsample IS for speed (46M is too much for full training)
    max_train = 5_000_000
    if len(X_is) > max_train:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_is), max_train, replace=False)
        X_train = X_is[idx]
        y_train = y_is[idx]
        print(f"  Subsampled IS to {max_train:,} for training speed")
    else:
        X_train = X_is
        y_train = y_is
    
    # Subsample OOS for validation
    max_val = 2_000_000
    if len(X_oos) > max_val:
        rng = np.random.RandomState(99)
        idx = rng.choice(len(X_oos), max_val, replace=False)
        X_val = X_oos[idx]
        y_val = y_oos[idx]
    else:
        X_val = X_oos
        y_val = y_oos
    
    t0 = time.time()
    
    if use_lgb:
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feat_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feat_names, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': 60,
        }
        
        model = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[lgb.log_evaluation(50)],
        )
        
        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        
        # Predictions
        prob_is = model.predict(X_is)
        prob_oos = model.predict(X_oos)
        
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=300, max_leaf_nodes=63, learning_rate=0.05,
            min_samples_leaf=100, random_state=42,
        )
        model.fit(X_train, y_train)
        
        # No clean feature importance for HGB, use permutation later
        importance = np.zeros(len(feat_names))
        
        prob_is = model.predict_proba(X_is)[:, 1]
        prob_oos = model.predict_proba(X_oos)[:, 1]
    
    print(f"\n  Training done in {time.time()-t0:.1f}s")
    
    # ═══ Feature Importance ═══
    print(f"\n{'='*90}")
    print(f"  FEATURE IMPORTANCE (gain)")
    print(f"{'='*90}")
    imp_order = np.argsort(importance)[::-1]
    total_imp = np.sum(importance)
    for rank, i in enumerate(imp_order):
        pct = importance[i] / total_imp * 100 if total_imp > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"  {rank+1}. {feat_names[i]:>22s}: {importance[i]:>12.0f} ({pct:>5.1f}%) {bar}")
    
    # ═══ Model probability → trading performance ═══
    print(f"\n{'='*90}")
    print(f"  MODEL PROBABILITY → TRADING PERFORMANCE")
    print(f"  (At each probability threshold, what's the simulated PnL?)")
    print(f"{'='*90}")
    
    # Compute PnL for each sample at different TP/SL
    tp_r, sl_r = 0.60, 0.35  # standard from V5
    pnl_all = np.where(mae >= sl_r, -1.0,
              np.where(mfe >= tp_r, tp_r / sl_r,
                       (mfe - mae) / (2 * sl_r)))
    
    pnl_is = pnl_all[is_mask]
    pnl_oos = pnl_all[oos_mask]
    
    print(f"\n  TP={tp_r:.2f}A, SL={sl_r:.2f}A, R:R={tp_r/sl_r:.1f}:1")
    print(f"\n  {'prob_thresh':>12s} | {'IS n':>9s} {'IS WR':>6s} {'IS avgR':>8s} {'IS PF':>7s} | "
          f"{'OOS n':>9s} {'OOS WR':>6s} {'OOS avgR':>8s} {'OOS PF':>7s}")
    print(f"  {'-'*95}")
    
    for thresh in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]:
        is_sel = prob_is >= thresh
        oos_sel = prob_oos >= thresh
        
        is_pnl = pnl_is[is_sel]
        oos_pnl = pnl_oos[oos_sel]
        
        if len(is_pnl) < 100 or len(oos_pnl) < 50:
            continue
        
        is_wr = np.mean(is_pnl > 0) * 100
        oos_wr = np.mean(oos_pnl > 0) * 100
        
        is_w = is_pnl[is_pnl > 0]; is_l = is_pnl[is_pnl <= 0]
        oos_w = oos_pnl[oos_pnl > 0]; oos_l = oos_pnl[oos_pnl <= 0]
        is_pf = abs(np.sum(is_w)/np.sum(is_l)) if np.sum(is_l) != 0 else float('inf')
        oos_pf = abs(np.sum(oos_w)/np.sum(oos_l)) if np.sum(oos_l) != 0 else float('inf')
        
        marker = " <<<" if np.mean(oos_pnl) > 0 else ""
        print(f"  {thresh:>12.2f} | {len(is_pnl):>9,} {is_wr:>6.1f} {np.mean(is_pnl):>+8.4f} {is_pf:>7.2f} | "
              f"{len(oos_pnl):>9,} {oos_wr:>6.1f} {np.mean(oos_pnl):>+8.4f} {oos_pf:>7.2f}{marker}")
    
    # ═══ Try multiple TP/SL at best probability threshold ═══
    print(f"\n{'='*90}")
    print(f"  TP/SL SWEEP at model probability >= 0.50 (OOS)")
    print(f"{'='*90}")
    
    oos_high_mask = prob_oos >= 0.50
    oos_mfe_h = mfe[oos_mask][oos_high_mask]
    oos_mae_h = mae[oos_mask][oos_high_mask]
    
    if len(oos_mfe_h) >= 50:
        print(f"  Samples: {len(oos_mfe_h):,}")
        print(f"  MFE: mean={np.mean(oos_mfe_h):.4f}, MAE: mean={np.mean(oos_mae_h):.4f}, "
              f"MFE/MAE={np.mean(oos_mfe_h)/np.mean(oos_mae_h):.3f}")
        print(f"\n  {'TP':>6s} {'SL':>6s} {'R:R':>5s} {'n':>8s} {'WR%':>6s} {'avgR':>8s} {'PF':>7s}")
        print(f"  {'-'*52}")
        for tp in [0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]:
            for sl in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
                p = np.where(oos_mae_h >= sl, -1.0,
                    np.where(oos_mfe_h >= tp, tp / sl,
                             (oos_mfe_h - oos_mae_h) / (2 * sl)))
                if len(p) < 50: continue
                w = p[p > 0]; l = p[p <= 0]
                pf = abs(np.sum(w)/np.sum(l)) if np.sum(l) != 0 else float('inf')
                avgr = np.mean(p)
                marker = " <<<" if avgr > 0 else ""
                print(f"  {tp:>6.2f} {sl:>6.2f} {tp/sl:>5.1f} {len(p):>8,} "
                      f"{np.mean(p>0)*100:>6.1f} {avgr:>+8.4f} {pf:>7.2f}{marker}")
    else:
        print(f"  Only {len(oos_mfe_h)} samples at prob>=0.50, too few")
    
    # ═══ Probability-calibrated MFE/MAE profile ═══
    print(f"\n{'='*90}")
    print(f"  MODEL PROBABILITY → RAW MFE/MAE EDGE (OOS)")
    print(f"{'='*90}")
    
    prob_bins = [(0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), 
                 (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]
    
    print(f"  {'prob_range':>14s} {'n':>9s} {'MFE':>8s} {'MAE':>8s} {'MFE/MAE':>8s} {'P(MFE>MAE)':>11s}")
    print(f"  {'-'*60}")
    for lo, hi in prob_bins:
        sel = (prob_oos >= lo) & (prob_oos < hi)
        m = mfe[oos_mask][sel]
        a = mae[oos_mask][sel]
        if len(m) < 100: continue
        ratio = np.mean(m) / np.mean(a) if np.mean(a) > 0 else 0
        pfav = np.mean(m > a) * 100
        marker = " <<<" if ratio > 1.0 else ""
        print(f"  [{lo:.2f},{hi:.2f})  {len(m):>9,} {np.mean(m):>8.4f} {np.mean(a):>8.4f} "
              f"{ratio:>8.3f} {pfav:>10.1f}%{marker}")
    
    # ═══ What the model learned: top feature interaction ═══
    if use_lgb:
        print(f"\n{'='*90}")
        print(f"  TOP SPLIT VALUES (what thresholds did the model learn?)")
        print(f"{'='*90}")
        
        # Analyze split points from tree
        trees_info = model.dump_model()
        split_counts = {f: {} for f in feat_names}
        
        def count_splits(node, feat_names_list):
            if 'split_feature' in node:
                fname = feat_names_list[node['split_feature']]
                val = round(node['threshold'], 4)
                split_counts[fname][val] = split_counts[fname].get(val, 0) + 1
                if 'left_child' in node:
                    count_splits(node['left_child'], feat_names_list)
                if 'right_child' in node:
                    count_splits(node['right_child'], feat_names_list)
        
        for tree in trees_info['tree_info']:
            count_splits(tree['tree_structure'], feat_names)
        
        for fname in feat_names:
            splits = split_counts[fname]
            if not splits:
                continue
            top = sorted(splits.items(), key=lambda x: -x[1])[:5]
            top_str = ', '.join([f"{v:.4f}({c})" for v, c in top])
            print(f"  {fname:>22s}: {top_str}")


if __name__ == '__main__':
    main()
