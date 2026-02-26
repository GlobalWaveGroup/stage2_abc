#!/usr/bin/env python3
"""
聚类分析 — 对称谱特征 → faiss聚类 → Z outcome分布检验

核心问题: 相似的对称谱状态 → Z的发展是否集中？
如果集中 → 对称谱有预测力
如果不集中 → 对称谱和Z无关

检验方法:
1. 对特征矩阵做k-means聚类 (N=100/300/1000)
2. 对每个cluster, 看Z的分布:
   - 方向一致性: cluster内Z方向的众数占比 (越高越好, baseline=50%)
   - 幅度集中度: cluster内Z幅度的变异系数 (越小越好)
   - 整体熵: H(Z|cluster) vs H(Z) — 条件熵应该小于无条件熵
3. 统计显著性: 和随机聚类对比 (shuffle特征后重新聚类)
"""

import numpy as np
import pickle
import os
import sys
import time
from scipy import stats


def load_precomputed(pkl_path):
    """加载预计算数据"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"加载: {pkl_path}")
    print(f"  特征: {data['features'].shape}")
    print(f"  Z: {data['z_outcomes'].shape}")
    print(f"  配置: {data['config']}")
    
    # 有效样本
    valid = data['valid_mask']
    n_valid = int(np.sum(valid))
    print(f"  有效样本: {n_valid}/{len(valid)}")
    
    return data


def normalize_features(features):
    """特征标准化 (z-score)"""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-10] = 1.0  # 避免除零
    return (features - mean) / std, mean, std


def faiss_kmeans(features, n_clusters, n_iter=50, seed=42):
    """用faiss做k-means聚类"""
    import faiss
    
    d = features.shape[1]
    kmeans = faiss.Kmeans(d, n_clusters, niter=n_iter, seed=seed, verbose=False)
    kmeans.train(features.astype(np.float32))
    
    # 分配每个样本到最近的聚类中心
    _, labels = kmeans.index.search(features.astype(np.float32), 1)
    labels = labels.flatten()
    
    return labels, kmeans


def analyze_clusters(features, z_outcomes, labels, z_horizons=[10, 20, 50, 100],
                     label_name="kmeans"):
    """
    分析聚类结果。
    
    对每个cluster检查Z的分布:
    - 方向一致性
    - 幅度集中度
    - 条件熵 vs 无条件熵
    """
    n_clusters = len(set(labels))
    n_samples = len(labels)
    
    results = {}
    
    for h_idx, h in enumerate(z_horizons):
        base = h_idx * 5
        z_dir = z_outcomes[:, base]        # 方向 (+1/-1)
        z_change = z_outcomes[:, base + 3]  # close变化
        z_maxup = z_outcomes[:, base + 1]   # 最大上行
        z_maxdown = z_outcomes[:, base + 2]  # 最大下行
        
        # 全局baseline
        global_up_pct = np.mean(z_dir > 0)
        global_change_std = np.std(z_change)
        global_change_mean = np.mean(z_change)
        
        # 逐cluster统计
        cluster_stats = []
        weighted_entropy = 0
        
        for c in range(n_clusters):
            mask = labels == c
            n_c = np.sum(mask)
            if n_c < 3:
                continue
            
            c_dir = z_dir[mask]
            c_change = z_change[mask]
            
            # 方向一致性
            up_pct = np.mean(c_dir > 0)
            direction_consensus = max(up_pct, 1 - up_pct)  # [0.5, 1.0]
            
            # 幅度集中度 (变异系数倒数)
            c_std = np.std(c_change)
            c_mean = np.mean(c_change)
            
            # 该cluster的Z方向熵
            p_up = max(up_pct, 0.001)
            p_down = max(1 - up_pct, 0.001)
            entropy = -(p_up * np.log2(p_up) + p_down * np.log2(p_down))
            weighted_entropy += (n_c / n_samples) * entropy
            
            cluster_stats.append({
                'cluster': c,
                'n': int(n_c),
                'up_pct': float(up_pct),
                'consensus': float(direction_consensus),
                'mean_change': float(c_mean),
                'std_change': float(c_std),
                'entropy': float(entropy),
            })
        
        # 全局熵
        p_up_global = max(global_up_pct, 0.001)
        p_down_global = max(1 - global_up_pct, 0.001)
        global_entropy = -(p_up_global * np.log2(p_up_global) + 
                          p_down_global * np.log2(p_down_global))
        
        # 互信息 = H(Z) - H(Z|cluster)
        mutual_info = global_entropy - weighted_entropy
        
        # 方向一致性统计
        consensuses = [s['consensus'] for s in cluster_stats]
        
        results[f'H{h}'] = {
            'global_up_pct': float(global_up_pct),
            'global_entropy': float(global_entropy),
            'weighted_entropy': float(weighted_entropy),
            'mutual_info': float(mutual_info),
            'mean_consensus': float(np.mean(consensuses)) if consensuses else 0.5,
            'max_consensus': float(np.max(consensuses)) if consensuses else 0.5,
            'n_high_consensus': sum(1 for c in consensuses if c > 0.65),
            'n_clusters_valid': len(cluster_stats),
            'cluster_stats': cluster_stats,
        }
    
    return results


def significance_test(features, z_outcomes, n_clusters, z_horizons=[10, 20, 50, 100],
                     n_shuffles=100, seed=42):
    """
    显著性检验: 真实聚类 vs 随机聚类。
    
    方法:
    1. 对真实特征做k-means → 计算互信息MI_real
    2. 随机打乱特征 → 重新聚类 → 计算MI_shuffle (重复n_shuffles次)
    3. MI_real在shuffle分布中的百分位 = 显著性
    
    如果MI_real > 95%分位 → 特征和Z有显著关联
    """
    rng = np.random.RandomState(seed)
    
    # 标准化
    feat_norm, _, _ = normalize_features(features)
    
    # 真实聚类
    real_labels, _ = faiss_kmeans(feat_norm, n_clusters, seed=seed)
    real_results = analyze_clusters(features, z_outcomes, real_labels, z_horizons)
    
    # 提取真实MI
    real_mi = {h: real_results[h]['mutual_info'] for h in real_results}
    
    # shuffle测试
    shuffle_mis = {h: [] for h in real_results}
    
    for i in range(n_shuffles):
        # 打乱特征行 (保持Z不变)
        perm = rng.permutation(len(features))
        shuffled_feat = feat_norm[perm]
        
        # 重新聚类
        shuf_labels, _ = faiss_kmeans(shuffled_feat, n_clusters, seed=seed + i + 1)
        shuf_results = analyze_clusters(features, z_outcomes, shuf_labels, z_horizons)
        
        for h in shuf_results:
            shuffle_mis[h].append(shuf_results[h]['mutual_info'])
    
    # 百分位排名
    sig_results = {}
    for h in real_mi:
        shuf_arr = np.array(shuffle_mis[h])
        percentile = np.mean(shuf_arr < real_mi[h]) * 100
        sig_results[h] = {
            'real_mi': real_mi[h],
            'shuffle_mean': float(np.mean(shuf_arr)),
            'shuffle_std': float(np.std(shuf_arr)),
            'percentile': float(percentile),
            'significant': percentile > 95,
        }
    
    return sig_results, real_results, real_labels


def full_analysis(pkl_path, n_clusters_list=[50, 100, 300], n_shuffles=50):
    """完整分析流程"""
    print("=" * 70)
    print("聚类分析 + 显著性检验")
    print("=" * 70)
    
    data = load_precomputed(pkl_path)
    
    features = data['features']
    z_outcomes = data['z_outcomes']
    valid = data['valid_mask']
    z_horizons = data['config']['z_horizons']
    
    # 只用有效样本
    feat_valid = features[valid]
    z_valid = z_outcomes[valid]
    
    print(f"\n有效样本: {len(feat_valid)}")
    
    if len(feat_valid) < 100:
        print("WARNING: 样本太少 (<100), 结果不可靠")
    
    # 对每个聚类数量
    for n_clusters in n_clusters_list:
        if n_clusters > len(feat_valid) // 3:
            print(f"\n跳过 N={n_clusters} (样本太少, 每cluster<3)")
            continue
        
        print(f"\n{'='*60}")
        print(f"N_clusters = {n_clusters}")
        print(f"{'='*60}")
        
        t0 = time.time()
        sig_results, cluster_results, labels = significance_test(
            feat_valid, z_valid, n_clusters, z_horizons, n_shuffles=n_shuffles
        )
        t1 = time.time()
        print(f"耗时: {t1-t0:.1f}s (含{n_shuffles}次shuffle)")
        
        # 显著性汇总
        print(f"\n--- 显著性检验 ---")
        print(f"{'Horizon':>8s} | {'MI_real':>8s} | {'MI_shuf':>8s} | {'pct':>6s} | {'sig':>3s}")
        print(f"{'-'*48}")
        for h in sorted(sig_results.keys()):
            s = sig_results[h]
            sig_mark = '***' if s['percentile'] > 99 else ('**' if s['percentile'] > 95 else ('*' if s['percentile'] > 90 else ''))
            print(f"{h:>8s} | {s['real_mi']:.6f} | {s['shuffle_mean']:.6f} | "
                  f"{s['percentile']:5.1f}% | {sig_mark}")
        
        # 聚类详情 (只显示H20)
        if 'H20' in cluster_results:
            r = cluster_results['H20']
            print(f"\n--- H20 聚类详情 ---")
            print(f"全局: up={r['global_up_pct']*100:.1f}% entropy={r['global_entropy']:.4f}")
            print(f"条件熵: {r['weighted_entropy']:.4f} (MI={r['mutual_info']:.6f})")
            print(f"方向一致性: mean={r['mean_consensus']:.3f} max={r['max_consensus']:.3f}")
            print(f"高一致性cluster(>65%): {r['n_high_consensus']}/{r['n_clusters_valid']}")
            
            # Top10方向性最强的cluster
            sorted_clusters = sorted(r['cluster_stats'], 
                                    key=lambda x: -x['consensus'])
            print(f"\nTop10方向性最强的cluster (H20):")
            print(f"  {'clust':>5s} {'n':>5s} {'up%':>6s} {'cons':>6s} {'mean_chg':>10s} {'std_chg':>10s}")
            for cs in sorted_clusters[:10]:
                dir_mark = '↑' if cs['up_pct'] > 0.5 else '↓'
                print(f"  {cs['cluster']:5d} {cs['n']:5d} {cs['up_pct']*100:5.1f}% "
                      f"{cs['consensus']:.3f} {cs['mean_change']:+10.6f} "
                      f"{cs['std_change']:10.6f} {dir_mark}")
    
    return sig_results, cluster_results, labels


if __name__ == '__main__':
    # 先测试小规模数据
    test_path = '/home/ubuntu/stage2_abc/precompute_test/precomputed.pkl'
    full_path = '/home/ubuntu/stage2_abc/precompute_h1/precomputed.pkl'
    
    if os.path.exists(full_path):
        pkl_path = full_path
        n_shuffles = 100
    elif os.path.exists(test_path):
        pkl_path = test_path
        n_shuffles = 20
    else:
        print("ERROR: 没有预计算数据，请先运行 full_precompute.py")
        sys.exit(1)
    
    full_analysis(pkl_path, 
                  n_clusters_list=[10, 50, 100],
                  n_shuffles=n_shuffles)
