#!/usr/bin/env python3
"""
向量库构建 — 原始对称谱向量 + Z outcome

不做任何统计压缩，每个对称谱作为独立向量存入。
多核并行计算。

向量编码 (每个spectrum → 1个向量):
  [type_mirror,     # 0=center, 1=mirror
   dir_L,           # +1/-1
   amp_ratio, time_ratio, mod_ratio, slope_ratio,  # 核心4维ratio
   amp_log, time_log, mod_log, slope_log,           # log空间
   complexity_diff,  # L复杂度 - R复杂度
   center_span_ratio, # center_span / (L_time + R_time)  
   sym_closeness,    # 标量对称度
   L_time_norm,      # L时间 / window_size (归一化)
   R_time_norm,      # R时间 / window_size
  ]
  → 15维 float32

附带metadata:
  window_end_bar: 窗口末尾在全局K线中的位置
  → 用于关联Z outcome

Z outcome (每个窗口一个, 共享给该窗口的所有spectrum):
  和之前相同: 4 horizons × 5 metrics = 20维
"""

import numpy as np
import pickle
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from merge_engine_v3 import (load_kline, calculate_base_zg, full_merge_engine,
                              build_segment_pool, compute_pivot_importance,
                              pool_fusion, compute_symmetry_spectrum)


# =============================================================================
# 单窗口处理
# =============================================================================

def spectrum_to_vector(s, window_size):
    """将一个spectrum dict编码为15维float向量"""
    is_mirror = 1.0 if s['type'] == 'mirror' else 0.0
    center_total = s.get('center_span', 0)
    lr_total = max(s['L_time'] + s['R_time'], 1)
    center_ratio = center_total / lr_total
    
    return np.array([
        is_mirror,
        float(s['dir_L']),
        s['amp_ratio'], s['time_ratio'], s['mod_ratio'], s['slope_ratio'],
        s['amp_log'], s['time_log'], s['mod_log'], s['slope_log'],
        float(s.get('complexity_diff', 0)),
        center_ratio,
        s['sym_closeness'],
        s['L_time'] / window_size,
        s['R_time'] / window_size,
    ], dtype=np.float32)


def process_window(args):
    """处理单个窗口 (用于multiprocessing)"""
    global_start, window_size, high_window, low_window, spectra_max_pool = args
    
    try:
        pivots = calculate_base_zg(high_window, low_window)
        if len(pivots) < 4:
            return None
        
        results = full_merge_engine(pivots)
        pi = compute_pivot_importance(results, total_bars=window_size)
        pool = build_segment_pool(results, pi)
        fp, _, _ = pool_fusion(pool, pi)
        spectra = compute_symmetry_spectrum(fp, pi, max_pool_size=spectra_max_pool)
        
        if not spectra:
            return None
        
        # 编码所有spectrum为向量
        vectors = np.array([spectrum_to_vector(s, window_size) for s in spectra],
                          dtype=np.float32)
        
        return {
            'global_end_bar': global_start + window_size - 1,
            'n_spectra': len(spectra),
            'vectors': vectors,
            'n_pivots': len(pivots),
            'n_segs': len(fp),
        }
    except Exception as e:
        return None


def _process_window_wrapper(args):
    """wrapper for multiprocessing — 解包numpy数组"""
    idx, global_start, window_size, high_slice, low_slice, max_pool = args
    result = process_window((global_start, window_size, high_slice, low_slice, max_pool))
    if result:
        result['window_idx'] = idx
    return result


# =============================================================================
# 主流程
# =============================================================================

def build_store(filepath, output_path, 
                window_size=50, stride=5,
                spectra_max_pool=500,
                z_horizons=[10, 20, 50, 100],
                n_workers=None, limit=None):
    """
    构建向量库。
    
    输出格式:
    {
        'vectors': np.array (N_total_spectra × 15), float32
        'window_ids': np.array (N_total_spectra,), int32 — 每个向量属于哪个窗口
        'window_end_bars': np.array (N_windows,), int32
        'z_outcomes': np.array (N_windows × 20), float32
        'config': dict
    }
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 8)
    
    print("=" * 70)
    print(f"向量库构建")
    print(f"  数据: {filepath}")
    print(f"  窗口: {window_size}, 步进: {stride}")
    print(f"  并行: {n_workers} workers")
    print("=" * 70)
    
    # 加载数据
    df = load_kline(filepath, limit=limit)
    n = len(df)
    close = df['close'].values.astype(np.float64)
    high_arr = df['high'].values.astype(np.float64)
    low_arr = df['low'].values.astype(np.float64)
    
    print(f"K线: {n}")
    
    # 生成窗口参数
    tasks = []
    idx = 0
    start = 0
    while start + window_size <= n:
        tasks.append((
            idx, start, window_size,
            high_arr[start:start+window_size].copy(),
            low_arr[start:start+window_size].copy(),
            spectra_max_pool,
        ))
        idx += 1
        start += stride
    
    n_windows = len(tasks)
    print(f"窗口数: {n_windows}")
    
    # 多核并行
    t0 = time.time()
    
    if n_workers > 1:
        with Pool(n_workers) as pool:
            # 用imap_unordered + 进度报告
            results_list = []
            done = 0
            last_report = t0
            for result in pool.imap_unordered(_process_window_wrapper, tasks, chunksize=50):
                results_list.append(result)
                done += 1
                now = time.time()
                if now - last_report > 10 or done == n_windows:
                    elapsed = now - t0
                    speed = done / elapsed
                    eta = (n_windows - done) / speed if speed > 0 else 0
                    print(f"  {done:6d}/{n_windows} ({done/n_windows*100:.1f}%) "
                          f"{elapsed:.0f}s {speed:.0f}w/s ETA {eta:.0f}s")
                    last_report = now
    else:
        results_list = []
        for task in tasks:
            results_list.append(_process_window_wrapper(task))
            if len(results_list) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {len(results_list):6d}/{n_windows} {elapsed:.0f}s")
    
    elapsed = time.time() - t0
    print(f"\n计算完成: {elapsed:.1f}s ({n_windows/elapsed:.0f} w/s)")
    
    # 整理结果
    valid_results = [r for r in results_list if r is not None]
    valid_results.sort(key=lambda r: r['window_idx'])
    
    print(f"有效窗口: {len(valid_results)}/{n_windows}")
    
    # 汇总向量
    all_vectors = []
    all_window_ids = []
    window_end_bars = []
    
    for r in valid_results:
        win_id = len(window_end_bars)
        n_spec = r['n_spectra']
        all_vectors.append(r['vectors'])
        all_window_ids.extend([win_id] * n_spec)
        window_end_bars.append(r['global_end_bar'])
    
    if not all_vectors:
        print("ERROR: 没有有效结果")
        return None
    
    vectors = np.concatenate(all_vectors, axis=0)
    window_ids = np.array(all_window_ids, dtype=np.int32)
    end_bars = np.array(window_end_bars, dtype=np.int32)
    
    print(f"\n向量库: {vectors.shape[0]} vectors × {vectors.shape[1]}D")
    print(f"窗口数: {len(end_bars)}")
    print(f"平均每窗口: {vectors.shape[0] / len(end_bars):.0f} spectra")
    
    # Z outcome
    z_list = []
    for end_bar in end_bars:
        z = _compute_z(close, high_arr, low_arr, end_bar, z_horizons)
        z_list.append(z)
    z_outcomes = np.array(z_list, dtype=np.float32)
    
    valid_z = ~np.any(np.isnan(z_outcomes), axis=1)
    print(f"Z有效: {int(np.sum(valid_z))}/{len(end_bars)}")
    
    # 保存
    store = {
        'vectors': vectors,
        'window_ids': window_ids,
        'window_end_bars': end_bars,
        'z_outcomes': z_outcomes,
        'z_valid': valid_z,
        'config': {
            'filepath': filepath,
            'n_bars': n,
            'window_size': window_size,
            'stride': stride,
            'spectra_max_pool': spectra_max_pool,
            'z_horizons': z_horizons,
            'vec_dim': vectors.shape[1],
            'z_dim': z_outcomes.shape[1],
        },
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(store, f, protocol=4)
    
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n已保存: {output_path} ({size_mb:.1f} MB)")
    
    # 基础统计
    print(f"\n=== 向量统计 ===")
    for i, name in enumerate(['is_mirror', 'dir_L', 
                               'amp_r', 'time_r', 'mod_r', 'slope_r',
                               'amp_log', 'time_log', 'mod_log', 'slope_log',
                               'complex_diff', 'center_ratio', 'sym_close',
                               'L_time_n', 'R_time_n']):
        col = vectors[:, i]
        print(f"  {name:>12s}: mean={col.mean():8.3f} std={col.std():8.3f} "
              f"[{col.min():8.3f}, {col.max():8.3f}]")
    
    return store


def _compute_z(close, high, low, current_bar, horizons):
    """计算Z outcome"""
    n = len(close)
    z = []
    for h in horizons:
        end = current_bar + h + 1
        if end > n:
            z.extend([np.nan] * 5)
            continue
        fc = close[current_bar + 1: end]
        fh = high[current_bar + 1: end]
        fl = low[current_bar + 1: end]
        if len(fc) == 0:
            z.extend([np.nan] * 5)
            continue
        cp = close[current_bar]
        direction = 1.0 if fc[-1] > cp else -1.0
        max_up = float(np.max(fh) - cp)
        max_down = float(cp - np.min(fl))
        close_change = float(fc[-1] - cp)
        volatility = float(np.std(fc))
        z.extend([direction, max_up, max_down, close_change, volatility])
    return np.array(z, dtype=np.float32)


# =============================================================================

if __name__ == '__main__':
    filepath = "/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv"
    
    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        build_store(
            filepath,
            output_path='/home/ubuntu/stage2_abc/vecstore_h1/store.pkl',
            window_size=50, stride=5,
            spectra_max_pool=500,
            n_workers=8,
        )
    else:
        # 小测试
        build_store(
            filepath,
            output_path='/home/ubuntu/stage2_abc/vecstore_test/store.pkl',
            window_size=50, stride=5,
            spectra_max_pool=500,
            n_workers=4,
            limit=2000,
        )
