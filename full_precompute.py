#!/usr/bin/env python3
"""
全量预计算脚本 — 动态引擎 + Z outcome采集

逐K线推进，每个拐点确认时刻:
  1. 计算当前对称谱快照 (top spectra)
  2. 聚合为特征向量 (多维描述当前"对称状态")
  3. 记录Z outcome (未来10/20/50/100 bars的价格发展)

输出:
  features.pkl — 特征矩阵 (N_samples × D_features)
  z_outcomes.pkl — Z标签矩阵 (N_samples × D_z)
  metadata.pkl — 时间戳、bar index等元信息
"""

import numpy as np
import pandas as pd
import pickle
import time
import sys
import os
from collections import Counter
from dynamic_engine import DynamicEngine, load_kline


# =============================================================================
# 特征提取: 从对称谱快照 → 固定维度特征向量
# =============================================================================

def spectra_to_features(spectra, n_pool_segs, n_confirmed_pivots):
    """
    将对称谱列表压缩为固定维度特征向量。
    
    思路:
    - 对称谱是变长的 (可能有0个到几万个)
    - 需要聚合为固定维度的统计描述
    - 同时保留"最重要的"个体谱信息
    
    特征维度:
    - 基础统计 (4): n_spectra, n_mirror, n_center, mirror_ratio
    - amp_ratio分布 (5): mean, std, median, p10, p90
    - time_ratio分布 (5): mean, std, median, p10, p90
    - mod_ratio分布 (5): mean, std, median, p10, p90
    - slope_ratio分布 (5): mean, std, median, p10, p90
    - sym_closeness分布 (5): mean, std, median, p10, p90
    - Top5对称谱的详细向量 (5×4=20): amp_ratio, time_ratio, mod_ratio, slope_ratio
    - 池状态 (2): n_pool_segs, n_confirmed_pivots
    
    总计: 51维
    """
    N = len(spectra)
    
    if N == 0:
        return np.zeros(51, dtype=np.float32)
    
    # 提取数组
    amp_ratios = np.array([s['amp_ratio'] for s in spectra], dtype=np.float32)
    time_ratios = np.array([s['time_ratio'] for s in spectra], dtype=np.float32)
    mod_ratios = np.array([s['mod_ratio'] for s in spectra], dtype=np.float32)
    slope_ratios = np.array([s['slope_ratio'] for s in spectra], dtype=np.float32)
    sym_scores = np.array([s['sym_closeness'] for s in spectra], dtype=np.float32)
    types = [s['type'] for s in spectra]
    
    n_mirror = sum(1 for t in types if t == 'mirror')
    n_center = N - n_mirror
    
    def _dist_stats(arr):
        """5个分布统计量"""
        return [
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.median(arr)),
            float(np.percentile(arr, 10)),
            float(np.percentile(arr, 90)),
        ]
    
    features = []
    
    # 基础统计
    features.extend([N, n_mirror, n_center, n_mirror / max(N, 1)])
    
    # 各维度分布
    features.extend(_dist_stats(amp_ratios))
    features.extend(_dist_stats(time_ratios))
    features.extend(_dist_stats(mod_ratios))
    features.extend(_dist_stats(slope_ratios))
    features.extend(_dist_stats(sym_scores))
    
    # Top5对称谱的详细向量 (按sym_closeness排序,取前5)
    sorted_spectra = sorted(spectra, key=lambda s: -s['sym_closeness'])
    for i in range(5):
        if i < len(sorted_spectra):
            s = sorted_spectra[i]
            features.extend([s['amp_ratio'], s['time_ratio'], 
                           s['mod_ratio'], s['slope_ratio']])
        else:
            features.extend([0, 0, 0, 0])
    
    # 池状态
    features.extend([n_pool_segs, n_confirmed_pivots])
    
    return np.array(features, dtype=np.float32)


# =============================================================================
# Z outcome采集
# =============================================================================

def compute_z_outcomes(close, high, low, current_bar, horizons=[10, 20, 50, 100]):
    """
    计算Z outcome — 当前bar之后的价格发展。
    
    对每个前瞻距离h:
    - direction: close[t+h] vs close[t] 的方向 (+1/-1)
    - max_up: max(high[t+1:t+h+1]) - close[t]  (最大上行)
    - max_down: close[t] - min(low[t+1:t+h+1])  (最大下行)
    - close_change: close[t+h] - close[t]
    - volatility: std(close[t+1:t+h+1])
    
    总计: 5×len(horizons) = 20维 (默认4个horizon)
    """
    n = len(close)
    z = []
    
    for h in horizons:
        end = current_bar + h + 1
        if end > n:
            # 数据不足，填NaN
            z.extend([np.nan] * 5)
            continue
        
        future_close = close[current_bar + 1: end]
        future_high = high[current_bar + 1: end]
        future_low = low[current_bar + 1: end]
        
        if len(future_close) == 0:
            z.extend([np.nan] * 5)
            continue
        
        current_price = close[current_bar]
        
        direction = 1.0 if future_close[-1] > current_price else -1.0
        max_up = float(np.max(future_high) - current_price)
        max_down = float(current_price - np.min(future_low))
        close_change = float(future_close[-1] - current_price)
        volatility = float(np.std(future_close))
        
        z.extend([direction, max_up, max_down, close_change, volatility])
    
    return np.array(z, dtype=np.float32)


# =============================================================================
# 全量预计算
# =============================================================================

def full_precompute(filepath, output_dir, limit=None,
                    window_size=200,
                    stride=50,
                    spectra_max_pool=500,
                    z_horizons=[10, 20, 50, 100],
                    warmup_bars=200):
    """
    全量预计算 — 滑动窗口方式。
    
    核心架构:
    - 用滑动窗口[t-window+1, t], 在每个窗口上用静态引擎完整计算
    - 静态引擎: ZG → 归并 → pool_fusion → 对称谱 (在窗口内完成)
    - 每个窗口的对称谱 → 特征向量
    - Z outcome: 窗口结束后的价格发展
    
    为什么用滑动窗口而不是纯动态:
    - 纯动态引擎的pool_fusion随池增大而爆炸 (O(degree^3))
    - 滑动窗口每次在固定大小的池上计算, 性能可控
    - 窗口内的计算完全无前瞻 (只用窗口内的K线)
    
    Args:
        filepath: K线数据文件路径
        output_dir: 输出目录
        limit: 限制K线数量
        window_size: 窗口大小 (K线数)
        stride: 滑动步长
        spectra_max_pool: 对称谱搜索的max_pool_size
        z_horizons: Z outcome前瞻距离
        warmup_bars: 前N根K线不采样
    """
    from merge_engine_v3 import (calculate_base_zg, full_merge_engine,
                                  build_segment_pool, compute_pivot_importance,
                                  pool_fusion, compute_symmetry_spectrum)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"全量预计算 (滑动窗口): {filepath}")
    print(f"输出目录: {output_dir}")
    print("=" * 70)
    
    # 加载数据
    df = load_kline(filepath, limit=limit)
    n = len(df)
    close = df['close'].values.astype(np.float64)
    high_arr = df['high'].values.astype(np.float64)
    low_arr = df['low'].values.astype(np.float64)
    datetimes = df['datetime'].values
    
    print(f"K线: {n}, {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"配置: window={window_size}, stride={stride}, max_pool={spectra_max_pool}")
    print(f"Z horizons: {z_horizons}")
    print(f"预计窗口数: {max(0, (n - warmup_bars - window_size) // stride + 1)}")
    
    all_features = []
    all_z = []
    all_meta = []
    
    t0 = time.time()
    last_report = t0
    sample_count = 0
    
    # 滑动窗口
    start = warmup_bars
    while start + window_size <= n:
        end = start + window_size
        
        # 窗口内的K线
        w_high = high_arr[start:end]
        w_low = low_arr[start:end]
        
        # 静态引擎: 在窗口上完整计算
        base_pivots = calculate_base_zg(w_high, w_low)
        
        if len(base_pivots) < 4:
            start += stride
            continue
        
        results = full_merge_engine(base_pivots)
        pivot_info = compute_pivot_importance(results, total_bars=window_size)
        pool = build_segment_pool(results, pivot_info)
        full_pool, _, _ = pool_fusion(pool, pivot_info)
        spectra = compute_symmetry_spectrum(full_pool, pivot_info, max_pool_size=spectra_max_pool)
        
        # 特征向量
        feat = spectra_to_features(spectra, len(full_pool), len(base_pivots))
        
        # Z outcome: 窗口结束时刻(bar=end-1)之后的价格发展
        z = compute_z_outcomes(close, high_arr, low_arr, end - 1, z_horizons)
        
        all_features.append(feat)
        all_z.append(z)
        all_meta.append({
            'window_start': start,
            'window_end': end,
            'bar': end - 1,
            'datetime': str(datetimes[end - 1]),
            'n_pivots': len(base_pivots),
            'n_segs': len(full_pool),
            'n_spectra': len(spectra),
            'price': float(close[end - 1]),
        })
        sample_count += 1
        
        # 进度报告
        now = time.time()
        if now - last_report > 10 or start + stride + window_size > n:
            elapsed = now - t0
            total_windows = max(1, (n - warmup_bars - window_size) // stride + 1)
            pct = sample_count / total_windows * 100
            speed = sample_count / elapsed if elapsed > 0 else 0
            eta = (total_windows - sample_count) / speed if speed > 0 else 0
            print(f"  win {sample_count:5d} [{start:6d}:{end:6d}] | "
                  f"pivots={len(base_pivots):3d} segs={len(full_pool):5d} "
                  f"spectra={len(spectra):5d} | "
                  f"{pct:.1f}% {elapsed:.0f}s ({speed:.1f}w/s ETA {eta:.0f}s)")
            last_report = now
        
        start += stride
    
    elapsed = time.time() - t0
    print(f"\n完成: {sample_count} windows, {elapsed:.1f}s ({sample_count/elapsed:.1f} w/s)")
    
    # 转为numpy
    features = np.array(all_features, dtype=np.float32) if all_features else np.empty((0, 51))
    z_outcomes = np.array(all_z, dtype=np.float32) if all_z else np.empty((0, len(z_horizons) * 5))
    
    print(f"\n特征矩阵: {features.shape}")
    print(f"Z标签矩阵: {z_outcomes.shape}")
    
    # 有效样本 (Z没有NaN的行)
    valid_mask = ~np.any(np.isnan(z_outcomes), axis=1) if len(z_outcomes) > 0 else np.array([])
    n_valid = int(np.sum(valid_mask)) if len(valid_mask) > 0 else 0
    print(f"有效样本 (Z完整): {n_valid}/{sample_count}")
    
    # 保存
    output = {
        'features': features,
        'z_outcomes': z_outcomes,
        'metadata': all_meta,
        'valid_mask': valid_mask,
        'config': {
            'filepath': filepath,
            'n_bars': n,
            'window_size': window_size,
            'stride': stride,
            'spectra_max_pool': spectra_max_pool,
            'z_horizons': z_horizons,
            'warmup_bars': warmup_bars,
            'feature_dim': 51,
            'z_dim': len(z_horizons) * 5,
        },
    }
    
    pkl_path = os.path.join(output_dir, 'precomputed.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(output, f)
    
    print(f"\n已保存: {pkl_path} ({os.path.getsize(pkl_path) / 1024 / 1024:.1f} MB)")
    
    # 基础统计
    if n_valid > 0:
        valid_features = features[valid_mask]
        valid_z = z_outcomes[valid_mask]
        
        print(f"\n=== 特征统计 (有效样本) ===")
        print(f"  n_spectra: mean={valid_features[:,0].mean():.0f} "
              f"std={valid_features[:,0].std():.0f}")
        print(f"  amp_ratio_mean: mean={valid_features[:,4].mean():.3f} "
              f"std={valid_features[:,4].std():.3f}")
        print(f"  time_ratio_mean: mean={valid_features[:,9].mean():.3f} "
              f"std={valid_features[:,9].std():.3f}")
        
        print(f"\n=== Z outcome统计 (有效样本) ===")
        for idx, h in enumerate(z_horizons):
            base = idx * 5
            dir_col = valid_z[:, base]
            change_col = valid_z[:, base + 3]
            up_pct = np.mean(dir_col > 0) * 100
            print(f"  H{h:3d}: up={up_pct:.1f}% "
                  f"change mean={change_col.mean():.6f} std={change_col.std():.6f}")
    
    return output


# =============================================================================
# 小规模测试
# =============================================================================

def test_small(filepath, limit=2000):
    """小规模测试 — 2000 bars"""
    output_dir = '/home/ubuntu/stage2_abc/precompute_test'
    result = full_precompute(
        filepath, output_dir,
        limit=limit,
        window_size=200,
        stride=50,
        spectra_max_pool=500,
        warmup_bars=0,
    )
    return result


if __name__ == '__main__':
    filepath = "/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv"
    
    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        # 全量运行
        output_dir = '/home/ubuntu/stage2_abc/precompute_h1'
        full_precompute(
            filepath, output_dir,
            limit=None,
            window_size=200,
            stride=50,
            spectra_max_pool=500,
            warmup_bars=0,
        )
    else:
        # 小规模测试
        test_small(filepath, limit=2000)
