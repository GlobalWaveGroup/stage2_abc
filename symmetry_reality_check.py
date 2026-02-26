#!/usr/bin/env python3
"""
对称客观性检验 — 真实行情 vs 随机序列

核心问题: 我们在波段池中发现的高对称结构,是行情的客观规律还是统计幻觉?

方法:
1. 对真实EURUSD H1数据, 构建波段池, 计算对称度分布
2. 生成N组随机游走序列(保持相同的波动率和长度), 同样构建波段池, 计算对称度分布
3. 对比: 真实行情的高对称度(>0.8, >0.9, >0.95)出现频率 vs 随机序列的频率
4. 用permutation test / bootstrap检验显著性

如果真实行情的对称度分布和随机游走没有显著差异 → 对称是统计幻觉
如果显著更高 → 对称客观存在
如果显著更低 → 市场反对称(也有意义)
"""

import numpy as np
import sys
import time
sys.path.insert(0, '/home/ubuntu/stage2_abc')
from merge_engine_v3 import *

def run_pipeline(high, low, total_bars):
    """完整pipeline: ZG → 归并 → 融合 → 对称识别, 返回对称度列表"""
    base = calculate_base_zg(high, low)
    if len(base) < 6:
        return []
    results = full_merge_engine(base)
    pivot_info = compute_pivot_importance(results, total_bars=total_bars)
    pool = build_segment_pool(results, pivot_info)
    full_pool, _, _ = pool_fusion(pool, pivot_info)
    
    # 对称识别 — 返回所有结构(不截断top_n), 用较大的max_pool_size
    structures = find_symmetric_structures(full_pool, pivot_info, top_n=9999, max_pool_size=600)
    
    sym_scores = [s['sym_score'] for s in structures]
    return sym_scores

def generate_random_walk(n, volatility, start_price=1.04):
    """生成随机游走序列, 模拟K线的OHLC"""
    returns = np.random.normal(0, volatility, n)
    close = start_price * np.exp(np.cumsum(returns))
    
    # 生成OHLC
    high = np.zeros(n)
    low = np.zeros(n)
    for i in range(n):
        intra_vol = volatility * 0.5
        h_dev = abs(np.random.normal(0, intra_vol))
        l_dev = abs(np.random.normal(0, intra_vol))
        high[i] = close[i] * (1 + h_dev)
        low[i] = close[i] * (1 - l_dev)
        if i > 0:
            # 确保high >= max(open, close), low <= min(open, close)
            high[i] = max(high[i], close[i], close[i-1])
            low[i] = min(low[i], close[i], close[i-1])
    
    return high, low

def generate_shuffled_returns(real_close):
    """打乱真实收益率序列 — 保持相同的收益率分布但破坏时序结构"""
    returns = np.diff(np.log(real_close))
    np.random.shuffle(returns)
    close = real_close[0] * np.exp(np.cumsum(np.concatenate([[0], returns])))
    
    n = len(close)
    vol = np.std(returns)
    high = np.zeros(n)
    low = np.zeros(n)
    for i in range(n):
        intra_dev = abs(np.random.normal(0, vol * 0.3))
        high[i] = close[i] * (1 + intra_dev)
        low[i] = close[i] * (1 - intra_dev)
        if i > 0:
            high[i] = max(high[i], close[i], close[i-1])
            low[i] = min(low[i], close[i], close[i-1])
    
    return high, low

# =========================================================
# 主检验
# =========================================================

print("=" * 70)
print("对称客观性检验 — 真实行情 vs 随机序列")
print("=" * 70)

# 1. 真实数据
filepath = "/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv"
df = load_kline(filepath, limit=200)
real_high = df['high'].values
real_low = df['low'].values
real_close = df['close'].values

print(f"\n真实数据: EURUSD H1, {len(df)} bars")
print(f"  波动率(std of log returns): {np.std(np.diff(np.log(real_close))):.6f}")

t0 = time.time()
real_syms = run_pipeline(real_high, real_low, len(df))
t1 = time.time()

print(f"  对称结构数: {len(real_syms)} ({t1-t0:.2f}s)")
print(f"  对称度均值: {np.mean(real_syms):.4f}")
print(f"  对称度中位: {np.median(real_syms):.4f}")
print(f"  >0.80: {sum(1 for s in real_syms if s > 0.80)} ({sum(1 for s in real_syms if s > 0.80)/max(len(real_syms),1)*100:.1f}%)")
print(f"  >0.90: {sum(1 for s in real_syms if s > 0.90)} ({sum(1 for s in real_syms if s > 0.90)/max(len(real_syms),1)*100:.1f}%)")
print(f"  >0.95: {sum(1 for s in real_syms if s > 0.95)} ({sum(1 for s in real_syms if s > 0.95)/max(len(real_syms),1)*100:.1f}%)")

# 2. 随机对照组 — 两种方式
N_TRIALS = 30

print(f"\n--- 对照组A: 随机游走 (相同波动率, {N_TRIALS}次) ---")
vol = np.std(np.diff(np.log(real_close)))
rw_results = {'n_struct': [], 'mean_sym': [], 'gt80': [], 'gt90': [], 'gt95': []}

t0 = time.time()
for trial in range(N_TRIALS):
    np.random.seed(trial * 137 + 42)
    h, l = generate_random_walk(200, vol)
    syms = run_pipeline(h, l, 200)
    rw_results['n_struct'].append(len(syms))
    rw_results['mean_sym'].append(np.mean(syms) if syms else 0)
    rw_results['gt80'].append(sum(1 for s in syms if s > 0.80))
    rw_results['gt90'].append(sum(1 for s in syms if s > 0.90))
    rw_results['gt95'].append(sum(1 for s in syms if s > 0.95))
t1 = time.time()

print(f"  耗时: {t1-t0:.1f}s")
print(f"  结构数: {np.mean(rw_results['n_struct']):.0f} ± {np.std(rw_results['n_struct']):.0f}")
print(f"  对称度均值: {np.mean(rw_results['mean_sym']):.4f} ± {np.std(rw_results['mean_sym']):.4f}")
print(f"  >0.80: {np.mean(rw_results['gt80']):.1f} ± {np.std(rw_results['gt80']):.1f}")
print(f"  >0.90: {np.mean(rw_results['gt90']):.1f} ± {np.std(rw_results['gt90']):.1f}")
print(f"  >0.95: {np.mean(rw_results['gt95']):.1f} ± {np.std(rw_results['gt95']):.1f}")

print(f"\n--- 对照组B: 打乱收益率 (保持分布,破坏时序, {N_TRIALS}次) ---")
sf_results = {'n_struct': [], 'mean_sym': [], 'gt80': [], 'gt90': [], 'gt95': []}

t0 = time.time()
for trial in range(N_TRIALS):
    np.random.seed(trial * 251 + 17)
    h, l = generate_shuffled_returns(real_close)
    syms = run_pipeline(h, l, 200)
    sf_results['n_struct'].append(len(syms))
    sf_results['mean_sym'].append(np.mean(syms) if syms else 0)
    sf_results['gt80'].append(sum(1 for s in syms if s > 0.80))
    sf_results['gt90'].append(sum(1 for s in syms if s > 0.90))
    sf_results['gt95'].append(sum(1 for s in syms if s > 0.95))
t1 = time.time()

print(f"  耗时: {t1-t0:.1f}s")
print(f"  结构数: {np.mean(sf_results['n_struct']):.0f} ± {np.std(sf_results['n_struct']):.0f}")
print(f"  对称度均值: {np.mean(sf_results['mean_sym']):.4f} ± {np.std(sf_results['mean_sym']):.4f}")
print(f"  >0.80: {np.mean(sf_results['gt80']):.1f} ± {np.std(sf_results['gt80']):.1f}")
print(f"  >0.90: {np.mean(sf_results['gt90']):.1f} ± {np.std(sf_results['gt90']):.1f}")
print(f"  >0.95: {np.mean(sf_results['gt95']):.1f} ± {np.std(sf_results['gt95']):.1f}")

# 3. 统计显著性
print(f"\n" + "=" * 70)
print("统计对比")
print("=" * 70)

real_gt80 = sum(1 for s in real_syms if s > 0.80)
real_gt90 = sum(1 for s in real_syms if s > 0.90)
real_gt95 = sum(1 for s in real_syms if s > 0.95)

def percentile_rank(real_val, random_vals):
    """real_val在random_vals中的百分位排名"""
    return sum(1 for v in random_vals if v < real_val) / len(random_vals) * 100

print(f"\n{'指标':>15s} | {'真实':>8s} | {'随机游走':>12s} | {'打乱收益':>12s} | {'vs RW rank':>10s} | {'vs SF rank':>10s}")
print("-" * 85)

metrics = [
    ('结构数', len(real_syms), rw_results['n_struct'], sf_results['n_struct']),
    ('均值sym', round(np.mean(real_syms),4), rw_results['mean_sym'], sf_results['mean_sym']),
    ('>0.80数', real_gt80, rw_results['gt80'], sf_results['gt80']),
    ('>0.90数', real_gt90, rw_results['gt90'], sf_results['gt90']),
    ('>0.95数', real_gt95, rw_results['gt95'], sf_results['gt95']),
]

for name, real_val, rw_vals, sf_vals in metrics:
    rw_mean = np.mean(rw_vals)
    sf_mean = np.mean(sf_vals)
    rw_std = np.std(rw_vals)
    sf_std = np.std(sf_vals)
    rw_rank = percentile_rank(real_val, rw_vals)
    sf_rank = percentile_rank(real_val, sf_vals)
    print(f"{name:>15s} | {real_val:>8} | {rw_mean:>6.1f}±{rw_std:>4.1f} | {sf_mean:>6.1f}±{sf_std:>4.1f} | {rw_rank:>8.0f}% | {sf_rank:>8.0f}%")

print(f"\n解读:")
print(f"  rank > 95%: 真实显著高于随机 → 对称客观存在")
print(f"  rank 30-70%: 无显著差异 → 对称是统计幻觉")
print(f"  rank < 5%: 真实显著低于随机 → 市场反对称")
