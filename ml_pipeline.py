#!/usr/bin/env python3
"""
ML Pipeline — 天枢FSD 机器学习训练框架

数据流:
  Raw OHLC → FSD Engine (逐K线) → StateSnapshot → to_vector(63d) + Oracle labels
  → Train/Val/Test split (时间序列, 不shuffle)
  → Multiple models (XGBoost, MLP, LSTM)
  → 评估: accuracy, 策略模拟, vs random baseline

目标:
  1. dir_50: 未来50bar方向 (+1/-1) — 分类
  2. best_rr_50: 最优R:R — 回归
  3. best_tp_dir_50: 最优止盈方向 — 分类

用法:
  # 小规模验证 (2000 bars, ~30s)
  python3 ml_pipeline.py --bars 2000

  # 全量训练 (155K bars, ~40min)
  python3 ml_pipeline.py --bars 0

  # 只生成数据
  python3 ml_pipeline.py --bars 2000 --data-only
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/stage2_abc')


# ============================================================
# 1. 数据生成
# ============================================================

def generate_dataset(csv_path: str, limit_bars: int = 0, 
                     output_dir: str = '/home/ubuntu/stage2_abc/ml_data',
                     chunk_size: int = 10000) -> str:
    """
    运行FSD引擎, 生成训练数据 (状态向量 + Oracle标注)
    
    Args:
        csv_path: K线数据路径
        limit_bars: 0=全量
        output_dir: 输出目录
        chunk_size: 分块写入大小
    
    Returns:
        output_path: .npz 文件路径
    """
    from merge_engine_v3 import load_kline
    from fsd_engine import FSDEngine, OracleLabeler
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_kline(csv_path, limit=limit_bars if limit_bars > 0 else None)
    n = len(df)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    opens = df['open'].values.astype(float)
    closes = df['close'].values.astype(float)
    
    print(f"数据: {n} bars, {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    
    # --- Phase 1: 跑FSD引擎, 收集snapshots ---
    print(f"\n[Phase 1] FSD Engine ({n} bars)...")
    engine = FSDEngine(start_pred=50, max_trajs=30, pred_horizon=50)
    snapshots = []
    
    t0 = time.time()
    for i in range(n):
        snap = engine.step(highs[i], lows[i], opens[i], closes[i])
        snapshots.append(snap)
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"  bar {i+1}/{n} ({elapsed:.0f}s, {rate:.0f} bar/s, ETA {eta:.0f}s) "
                  f"trajs={snap.n_active_trajs}")
    
    elapsed = time.time() - t0
    print(f"  完成: {n} bars in {elapsed:.1f}s ({n/elapsed:.0f} bar/s)")
    
    # --- Phase 2: Oracle标注 ---
    print(f"\n[Phase 2] Oracle Labeling...")
    t1 = time.time()
    oracle_labels = OracleLabeler.label_batch(snapshots, highs, lows, closes)
    print(f"  完成: {time.time()-t1:.1f}s")
    
    # --- Phase 3: 向量化 ---
    print(f"\n[Phase 3] Vectorizing {n} snapshots...")
    t2 = time.time()
    
    # 状态向量
    max_trajs = 8
    vec_dim = 15 + max_trajs * 6  # 63
    X = np.zeros((n, vec_dim), dtype=np.float32)
    for i, snap in enumerate(snapshots):
        X[i] = snap.to_vector(max_trajs=max_trajs)
    
    # Oracle标注 → 多个target
    # Y_dir50: 方向 (0=down, 1=up) — 二分类
    # Y_rr50: 最优R:R — 回归
    # Y_tp_dir50: 最优TP方向 (0=down, 1=up) — 二分类  
    # Y_mfe_up50: 上行MFE (pips) — 回归
    # Y_mfe_dn50: 下行MFE (pips) — 回归
    
    Y_dir50 = np.array([1 if l.get('dir_50', 0) >= 0 else 0 for l in oracle_labels], dtype=np.int32)
    Y_tp_dir50 = np.array([1 if l.get('best_tp_dir_50', 0) >= 0 else 0 for l in oracle_labels], dtype=np.int32)
    Y_rr50 = np.array([l.get('best_rr_50', 0.0) for l in oracle_labels], dtype=np.float32)
    Y_mfe_up50 = np.array([l.get('mfe_up_50', 0.0) for l in oracle_labels], dtype=np.float32)
    Y_mfe_dn50 = np.array([l.get('mfe_dn_50', 0.0) for l in oracle_labels], dtype=np.float32)
    
    # 额外的meta信息 (不用于训练, 用于分析)
    bars = np.array([s.bar for s in snapshots], dtype=np.int32)
    closes_arr = np.array([s.close for s in snapshots], dtype=np.float32)
    n_trajs = np.array([s.n_active_trajs for s in snapshots], dtype=np.int32)
    
    print(f"  X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  Y_dir50 分布: up={np.sum(Y_dir50==1)}, dn={np.sum(Y_dir50==0)}")
    print(f"  Y_tp_dir50 分布: up={np.sum(Y_tp_dir50==1)}, dn={np.sum(Y_tp_dir50==0)}")
    print(f"  Y_rr50: mean={np.mean(Y_rr50):.2f}, median={np.median(Y_rr50):.2f}")
    print(f"  向量化完成: {time.time()-t2:.1f}s")
    
    # --- Phase 4: 保存 ---
    fname = f"fsd_dataset_{n}bars.npz"
    out_path = os.path.join(output_dir, fname)
    np.savez_compressed(out_path,
        X=X,
        Y_dir50=Y_dir50,
        Y_tp_dir50=Y_tp_dir50,
        Y_rr50=Y_rr50,
        Y_mfe_up50=Y_mfe_up50,
        Y_mfe_dn50=Y_mfe_dn50,
        bars=bars,
        closes=closes_arr,
        n_trajs=n_trajs,
    )
    fsize = os.path.getsize(out_path)
    print(f"\n保存: {out_path} ({fsize/1024/1024:.1f}MB)")
    
    return out_path


# ============================================================
# 2. 数据加载 & 切分
# ============================================================

def load_dataset(npz_path: str, train_ratio=0.70, val_ratio=0.15):
    """
    加载数据集, 按时间序列切分 (不shuffle!)
    
    Returns:
        dict with train/val/test splits, each containing X, y_dir, y_rr, y_tp_dir, etc.
    """
    data = np.load(npz_path)
    X = data['X']
    n = len(X)
    
    # 跳过前50 bars (无预测, 特征全0)
    start = 50
    X = X[start:]
    Y_dir50 = data['Y_dir50'][start:]
    Y_tp_dir50 = data['Y_tp_dir50'][start:]
    Y_rr50 = data['Y_rr50'][start:]
    Y_mfe_up50 = data['Y_mfe_up50'][start:]
    Y_mfe_dn50 = data['Y_mfe_dn50'][start:]
    closes = data['closes'][start:]
    n_trajs = data['n_trajs'][start:]
    
    n_eff = len(X)
    
    # 时间序列切分 (不shuffle!)
    train_end = int(n_eff * train_ratio)
    val_end = int(n_eff * (train_ratio + val_ratio))
    
    def make_split(start_idx, end_idx, name):
        return {
            'name': name,
            'X': X[start_idx:end_idx],
            'y_dir': Y_dir50[start_idx:end_idx],
            'y_tp_dir': Y_tp_dir50[start_idx:end_idx],
            'y_rr': Y_rr50[start_idx:end_idx],
            'y_mfe_up': Y_mfe_up50[start_idx:end_idx],
            'y_mfe_dn': Y_mfe_dn50[start_idx:end_idx],
            'closes': closes[start_idx:end_idx],
            'n_trajs': n_trajs[start_idx:end_idx],
        }
    
    splits = {
        'train': make_split(0, train_end, 'train'),
        'val': make_split(train_end, val_end, 'val'),
        'test': make_split(val_end, n_eff, 'test'),
    }
    
    print(f"数据集: {n_eff} 有效样本 (跳过前50)")
    for k, v in splits.items():
        n_up = np.sum(v['y_dir'] == 1)
        n_dn = np.sum(v['y_dir'] == 0)
        print(f"  {k}: {len(v['X'])} 样本, up={n_up}({n_up/len(v['X'])*100:.1f}%), "
              f"dn={n_dn}({n_dn/len(v['X'])*100:.1f}%)")
    
    return splits


# ============================================================
# 3. 特征工程 (增强63维基础向量)
# ============================================================

def engineer_features(X_raw: np.ndarray) -> np.ndarray:
    """
    在63维基础向量上添加衍生特征:
    - 价格变动率 (OHLC → return)
    - 轨迹特征的统计量 (mean/std/max of deviations)
    - 多空比差
    
    这个函数保持纯numpy, 不依赖FSD engine
    """
    n = X_raw.shape[0]
    
    # 基础63维的结构:
    # [0]: bar_norm
    # [1:5]: O,H,L,C
    # [5:11]: ZG状态
    # [11:15]: 共识(bull_ratio, bear_ratio, best_dev, best_prog)
    # [15:63]: 8条轨迹 × 6维(dir, prog, dev, mfe, mae, amp_pips)
    
    features = [X_raw.copy()]
    
    # --- 衍生特征 ---
    
    # 1. 多空比差 (bull_ratio - bear_ratio)
    bull_bear_diff = X_raw[:, 11] - X_raw[:, 12]
    features.append(bull_bear_diff.reshape(-1, 1))
    
    # 2. 轨迹特征聚合
    traj_devs = []
    traj_progs = []
    traj_mfes = []
    traj_maes = []
    traj_amps = []
    for j in range(8):
        base = 15 + j * 6
        traj_devs.append(X_raw[:, base + 2])   # deviation
        traj_progs.append(X_raw[:, base + 1])   # progress
        traj_mfes.append(X_raw[:, base + 3])    # MFE
        traj_maes.append(X_raw[:, base + 4])    # MAE
        traj_amps.append(X_raw[:, base + 5])    # amp_pips
    
    traj_devs = np.stack(traj_devs, axis=1)   # (n, 8)
    traj_progs = np.stack(traj_progs, axis=1)
    traj_mfes = np.stack(traj_mfes, axis=1)
    traj_maes = np.stack(traj_maes, axis=1)
    traj_amps = np.stack(traj_amps, axis=1)
    
    # 活跃轨迹掩码 (amp > 0 → 活跃)
    active_mask = (traj_amps > 0).astype(float)
    n_active = active_mask.sum(axis=1, keepdims=True)
    
    # 安全均值: 避免除0
    def safe_mean(arr, mask):
        s = (arr * mask).sum(axis=1, keepdims=True)
        c = np.maximum(mask.sum(axis=1, keepdims=True), 1e-10)
        return s / c
    
    def safe_std(arr, mask):
        m = safe_mean(arr, mask)
        sq = ((arr - m) ** 2 * mask).sum(axis=1, keepdims=True)
        c = np.maximum(mask.sum(axis=1, keepdims=True) - 1, 1)
        return np.sqrt(sq / c)
    
    # 离差统计
    features.append(safe_mean(traj_devs, active_mask))     # mean_dev
    features.append(safe_std(traj_devs, active_mask))      # std_dev
    features.append(np.max(traj_devs * active_mask, axis=1, keepdims=True))  # max_dev
    features.append(np.min(np.where(active_mask > 0, traj_devs, 99), axis=1, keepdims=True))  # min_dev
    
    # MFE/MAE比
    mean_mfe = safe_mean(traj_mfes, active_mask)
    mean_mae = safe_mean(traj_maes, active_mask)
    mfe_mae_ratio = mean_mfe / np.maximum(mean_mae, 0.01)
    features.append(mfe_mae_ratio)
    
    # 进度统计
    features.append(safe_mean(traj_progs, active_mask))    # mean_prog
    
    # 方向加权离差 (多头轨迹的离差均值 vs 空头)
    traj_dirs = np.stack([X_raw[:, 15 + j * 6 + 0] for j in range(8)], axis=1)
    bull_mask = (traj_dirs > 0).astype(float) * active_mask
    bear_mask = (traj_dirs < 0).astype(float) * active_mask
    bull_dev = safe_mean(traj_devs, bull_mask)
    bear_dev = safe_mean(traj_devs, bear_mask)
    features.append(bull_dev)
    features.append(bear_dev)
    
    X_eng = np.hstack(features)
    return X_eng


# ============================================================
# 4. 模型训练
# ============================================================

def train_xgboost(splits, target='y_dir', task='classification'):
    """XGBoost baseline"""
    import xgboost as xgb
    
    X_train = engineer_features(splits['train']['X'])
    X_val = engineer_features(splits['val']['X'])
    X_test = engineer_features(splits['test']['X'])
    y_train = splits['train'][target]
    y_val = splits['val'][target]
    y_test = splits['test'][target]
    
    print(f"\n{'='*60}")
    print(f"XGBoost | target={target} | task={task}")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Features: {X_train.shape[1]} dims")
    
    if task == 'classification':
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 10,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'tree_method': 'hist',
            'device': 'cuda',
            'n_jobs': -1,
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model = xgb.train(
            params, dtrain, 
            num_boost_round=500,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=30,
            verbose_eval=50,
        )
        
        # 评估
        pred_prob_val = model.predict(dval)
        pred_prob_test = model.predict(dtest)
        pred_val = (pred_prob_val > 0.5).astype(int)
        pred_test = (pred_prob_test > 0.5).astype(int)
        
        acc_val = np.mean(pred_val == y_val)
        acc_test = np.mean(pred_test == y_test)
        
        # 对比随机
        random_acc = max(np.mean(y_test == 1), np.mean(y_test == 0))
        
        print(f"\n  结果:")
        print(f"    Val  acc: {acc_val:.4f}")
        print(f"    Test acc: {acc_test:.4f}")
        print(f"    Random baseline: {random_acc:.4f} (always majority)")
        print(f"    Lift over random: {(acc_test - random_acc) / random_acc * 100:+.1f}%")
        
        # 分析置信度
        high_conf_mask = np.abs(pred_prob_test - 0.5) > 0.15  # >0.65 or <0.35
        if np.sum(high_conf_mask) > 0:
            hc_acc = np.mean(pred_test[high_conf_mask] == y_test[high_conf_mask])
            print(f"    High-confidence ({np.sum(high_conf_mask)} samples): acc={hc_acc:.4f}")
        
        return model, {'acc_val': acc_val, 'acc_test': acc_test, 'random': random_acc,
                       'pred_prob_test': pred_prob_test, 'pred_test': pred_test}
    
    else:  # regression
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 10,
            'tree_method': 'hist',
            'device': 'cuda',
            'n_jobs': -1,
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model = xgb.train(
            params, dtrain,
            num_boost_round=500,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=30,
            verbose_eval=50,
        )
        
        pred_val = model.predict(dval)
        pred_test = model.predict(dtest)
        
        from sklearn.metrics import mean_squared_error, r2_score
        rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
        rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
        r2_val = r2_score(y_val, pred_val)
        r2_test = r2_score(y_test, pred_test)
        
        # 对比: 总是预测均值
        naive_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, np.mean(y_train))))
        
        print(f"\n  结果:")
        print(f"    Val  RMSE={rmse_val:.4f}, R²={r2_val:.4f}")
        print(f"    Test RMSE={rmse_test:.4f}, R²={r2_test:.4f}")
        print(f"    Naive mean RMSE: {naive_rmse:.4f}")
        print(f"    Improvement: {(naive_rmse - rmse_test) / naive_rmse * 100:.1f}%")
        
        return model, {'rmse_val': rmse_val, 'rmse_test': rmse_test, 'r2_test': r2_test,
                       'naive_rmse': naive_rmse, 'pred_test': pred_test}


def train_lightgbm(splits, target='y_dir', task='classification'):
    """LightGBM baseline"""
    import lightgbm as lgb
    
    X_train = engineer_features(splits['train']['X'])
    X_val = engineer_features(splits['val']['X'])
    X_test = engineer_features(splits['test']['X'])
    y_train = splits['train'][target]
    y_val = splits['val'][target]
    y_test = splits['test'][target]
    
    print(f"\n{'='*60}")
    print(f"LightGBM | target={target} | task={task}")
    
    if task == 'classification':
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbose': -1,
            # 'device': 'gpu',  # requires OpenCL
        }
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        callbacks = [
            lgb.early_stopping(30),
            lgb.log_evaluation(50),
        ]
        
        model = lgb.train(params, dtrain, num_boost_round=500,
                         valid_sets=[dtrain, dval], valid_names=['train', 'val'],
                         callbacks=callbacks)
        
        pred_prob_test = model.predict(X_test)
        pred_test = (pred_prob_test > 0.5).astype(int)
        acc_test = np.mean(pred_test == y_test)
        random_acc = max(np.mean(y_test == 1), np.mean(y_test == 0))
        
        pred_prob_val = model.predict(X_val)
        pred_val = (pred_prob_val > 0.5).astype(int)
        acc_val = np.mean(pred_val == y_val)
        
        print(f"\n  结果:")
        print(f"    Val  acc: {acc_val:.4f}")
        print(f"    Test acc: {acc_test:.4f}")
        print(f"    Random baseline: {random_acc:.4f}")
        print(f"    Lift: {(acc_test - random_acc) / random_acc * 100:+.1f}%")
        
        return model, {'acc_val': acc_val, 'acc_test': acc_test, 'random': random_acc}
    
    return None, {}


def train_pytorch_mlp(splits, target='y_dir', task='classification', 
                      hidden_dims=[256, 128, 64], epochs=100, lr=1e-3, batch_size=512):
    """PyTorch MLP (GPU)"""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    X_train = engineer_features(splits['train']['X'])
    X_val = engineer_features(splits['val']['X'])
    X_test = engineer_features(splits['test']['X'])
    y_train = splits['train'][target]
    y_val = splits['val'][target]
    y_test = splits['test'][target]
    
    # 标准化
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / std
    X_val = (X_val - mu) / std
    X_test = (X_test - mu) / std
    
    input_dim = X_train.shape[1]
    
    print(f"\n{'='*60}")
    print(f"PyTorch MLP | target={target} | task={task} | device={device}")
    print(f"  Input: {input_dim}d, Hidden: {hidden_dims}")
    
    # 数据
    X_tr_t = torch.FloatTensor(X_train).to(device)
    X_va_t = torch.FloatTensor(X_val).to(device)
    X_te_t = torch.FloatTensor(X_test).to(device)
    
    if task == 'classification':
        y_tr_t = torch.LongTensor(y_train.astype(np.int64)).to(device)
        y_va_t = torch.LongTensor(y_val.astype(np.int64)).to(device)
        y_te_t = torch.LongTensor(y_test.astype(np.int64)).to(device)
        output_dim = 2
    else:
        y_tr_t = torch.FloatTensor(y_train).to(device)
        y_va_t = torch.FloatTensor(y_val).to(device)
        y_te_t = torch.FloatTensor(y_test).to(device)
        output_dim = 1
    
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # 模型
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)])
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    model = nn.Sequential(*layers).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    if task == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_metric = -1 if task == 'classification' else float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            out = model(xb)
            if task == 'regression':
                out = out.squeeze()
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        avg_loss = total_loss / len(X_tr_t)
        
        # Validation
        model.eval()
        with torch.no_grad():
            v_out = model(X_va_t)
            if task == 'classification':
                v_pred = v_out.argmax(dim=1)
                v_acc = (v_pred == y_va_t).float().mean().item()
                metric = v_acc
                scheduler.step(-v_acc)
            else:
                v_out = v_out.squeeze()
                v_loss = criterion(v_out, y_va_t).item()
                metric = -v_loss
                scheduler.step(v_loss)
        
        improved = (metric > best_val_metric) if task == 'classification' else (metric > best_val_metric)
        if improved:
            best_val_metric = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            if task == 'classification':
                print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, val_acc={v_acc:.4f}, lr={lr_now:.1e}")
            else:
                print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, val_rmse={np.sqrt(-metric):.4f}, lr={lr_now:.1e}")
        
        if patience_counter >= 20:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # 加载最优模型, 评估test
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        t_out = model(X_te_t)
        if task == 'classification':
            t_pred = t_out.argmax(dim=1)
            acc_test = (t_pred == y_te_t).float().mean().item()
            random_acc = max(np.mean(y_test == 1), np.mean(y_test == 0))
            
            # 分析置信度
            t_prob = torch.softmax(t_out, dim=1)[:, 1].cpu().numpy()
            high_conf = np.abs(t_prob - 0.5) > 0.15
            
            print(f"\n  最终结果:")
            print(f"    Test acc: {acc_test:.4f}")
            print(f"    Random baseline: {random_acc:.4f}")
            print(f"    Lift: {(acc_test - random_acc) / random_acc * 100:+.1f}%")
            if np.sum(high_conf) > 0:
                hc_acc = np.mean((t_pred.cpu().numpy()[high_conf]) == y_test[high_conf])
                print(f"    High-confidence ({np.sum(high_conf)} samples): acc={hc_acc:.4f}")
            
            return model, {'acc_test': acc_test, 'random': random_acc, 'pred_prob': t_prob}
        
        else:
            t_out = t_out.squeeze()
            rmse = np.sqrt(nn.MSELoss()(t_out, y_te_t).item())
            naive_rmse = np.sqrt(np.mean((y_test - np.mean(y_train)) ** 2))
            print(f"\n  最终结果: RMSE={rmse:.4f}, naive={naive_rmse:.4f}")
            return model, {'rmse_test': rmse, 'naive_rmse': naive_rmse}


# ============================================================
# 5. 策略模拟评估
# ============================================================

def evaluate_strategy(splits, pred_probs, threshold=0.55):
    """
    用模型预测模拟交易策略, 评估盈利能力。
    
    规则:
    - pred > threshold → 做多
    - pred < (1-threshold) → 做空
    - 否则 → 不做
    - 持仓50根K线后平仓
    - 不考虑点差/手续费 (先看raw signal)
    
    Returns:
        dict with PnL stats
    """
    test = splits['test']
    closes = test['closes']
    y_mfe_up = test['y_mfe_up']
    y_mfe_dn = test['y_mfe_dn']
    n = len(closes)
    
    trades = []
    for i in range(n - 50):
        if pred_probs[i] > threshold:
            # 做多: PnL = close[i+50] - close[i] (in pips)
            pnl = (closes[min(i+50, n-1)] - closes[i]) * 10000
            trades.append({'dir': 'long', 'entry_bar': i, 'pnl_pips': pnl,
                          'mfe_up': y_mfe_up[i], 'mfe_dn': y_mfe_dn[i]})
        elif pred_probs[i] < (1 - threshold):
            # 做空
            pnl = (closes[i] - closes[min(i+50, n-1)]) * 10000
            trades.append({'dir': 'short', 'entry_bar': i, 'pnl_pips': pnl,
                          'mfe_up': y_mfe_up[i], 'mfe_dn': y_mfe_dn[i]})
    
    if not trades:
        print("  无交易信号")
        return {}
    
    pnls = np.array([t['pnl_pips'] for t in trades])
    longs = [t for t in trades if t['dir'] == 'long']
    shorts = [t for t in trades if t['dir'] == 'short']
    
    win_rate = np.mean(pnls > 0)
    avg_win = np.mean(pnls[pnls > 0]) if np.any(pnls > 0) else 0
    avg_loss = np.mean(pnls[pnls <= 0]) if np.any(pnls <= 0) else 0
    
    cumsum = np.cumsum(pnls)
    max_dd = np.max(np.maximum.accumulate(cumsum) - cumsum)
    
    print(f"\n  策略模拟 (threshold={threshold}):")
    print(f"    总交易: {len(trades)} (多:{len(longs)}, 空:{len(shorts)})")
    print(f"    胜率: {win_rate:.3f}")
    print(f"    平均盈利: {avg_win:.1f} pips, 平均亏损: {avg_loss:.1f} pips")
    print(f"    Profit Factor: {abs(avg_win * np.sum(pnls>0)) / max(abs(avg_loss * np.sum(pnls<=0)), 0.01):.2f}")
    print(f"    总PnL: {np.sum(pnls):.1f} pips")
    print(f"    最大回撤: {max_dd:.1f} pips")
    print(f"    Sharpe (近似): {np.mean(pnls) / max(np.std(pnls), 0.01) * np.sqrt(252):.2f}")
    
    # 随机对比: 如果随机选择同样数量的交易
    np.random.seed(42)
    random_pnls = []
    for _ in range(100):
        ridx = np.random.choice(n - 50, size=len(trades), replace=True)
        rdir = np.random.choice([1, -1], size=len(trades))
        rpnl = [(closes[min(j+50, n-1)] - closes[j]) * 10000 * d for j, d in zip(ridx, rdir)]
        random_pnls.append(np.sum(rpnl))
    
    print(f"    随机策略PnL: mean={np.mean(random_pnls):.1f}, std={np.std(random_pnls):.1f}")
    z_score = (np.sum(pnls) - np.mean(random_pnls)) / max(np.std(random_pnls), 0.01)
    print(f"    Z-score vs random: {z_score:.2f}")
    
    return {
        'n_trades': len(trades),
        'win_rate': win_rate,
        'total_pnl': np.sum(pnls),
        'max_dd': max_dd,
        'z_score': z_score,
    }


# ============================================================
# 6. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='FSD ML Pipeline')
    parser.add_argument('--bars', type=int, default=2000, help='Bars to process (0=all)')
    parser.add_argument('--data-only', action='store_true', help='Only generate data')
    parser.add_argument('--npz', type=str, default=None, help='Use existing .npz file')
    parser.add_argument('--pair', type=str, default='EURUSD_H1', help='Currency pair + TF')
    args = parser.parse_args()
    
    print("=" * 60)
    print("天枢 FSD ML Pipeline")
    print("=" * 60)
    
    csv_path = f'/home/ubuntu/DataBase/base_kline/{args.pair}.csv'
    
    # 生成或加载数据
    if args.npz:
        npz_path = args.npz
    else:
        npz_path = generate_dataset(csv_path, limit_bars=args.bars)
    
    if args.data_only:
        print("\n[data-only mode] 数据已生成, 退出")
        return
    
    # 加载
    splits = load_dataset(npz_path)
    
    # === 多目标训练 ===
    results = {}
    
    # Task 1: 方向预测 (dir_50)
    print(f"\n{'#'*60}")
    print("# Task 1: 方向预测 (dir_50)")
    print(f"{'#'*60}")
    
    xgb_model, xgb_res = train_xgboost(splits, target='y_dir', task='classification')
    results['xgb_dir'] = xgb_res
    
    lgb_model, lgb_res = train_lightgbm(splits, target='y_dir', task='classification')
    results['lgb_dir'] = lgb_res
    
    mlp_model, mlp_res = train_pytorch_mlp(splits, target='y_dir', task='classification',
                                            hidden_dims=[256, 128, 64], epochs=100)
    results['mlp_dir'] = mlp_res
    
    # Task 2: 最优TP方向 (tp_dir_50)
    print(f"\n{'#'*60}")
    print("# Task 2: 最优TP方向 (tp_dir_50)")
    print(f"{'#'*60}")
    
    xgb2, xgb2_res = train_xgboost(splits, target='y_tp_dir', task='classification')
    results['xgb_tp_dir'] = xgb2_res
    
    # Task 3: R:R回归 (best_rr_50)
    print(f"\n{'#'*60}")
    print("# Task 3: R:R 回归 (best_rr_50)")
    print(f"{'#'*60}")
    
    xgb3, xgb3_res = train_xgboost(splits, target='y_rr', task='regression')
    results['xgb_rr'] = xgb3_res
    
    # === 策略模拟 ===
    print(f"\n{'#'*60}")
    print("# 策略模拟")
    print(f"{'#'*60}")
    
    # 用最优模型做策略
    if 'pred_prob_test' in xgb_res:
        for thresh in [0.52, 0.55, 0.60]:
            strat = evaluate_strategy(splits, xgb_res['pred_prob_test'], threshold=thresh)
            results[f'strat_xgb_{thresh}'] = strat
    
    if 'pred_prob' in mlp_res:
        for thresh in [0.52, 0.55, 0.60]:
            strat = evaluate_strategy(splits, mlp_res['pred_prob'], threshold=thresh)
            results[f'strat_mlp_{thresh}'] = strat
    
    # === 总结 ===
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    for k, v in results.items():
        if isinstance(v, dict):
            summary = {kk: vv for kk, vv in v.items() 
                      if isinstance(vv, (int, float)) and not isinstance(vv, bool)}
            if summary:
                print(f"  {k}: {summary}")


if __name__ == '__main__':
    main()
