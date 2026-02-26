#!/usr/bin/env python3
"""
向量库查询 — 用faiss做最近邻检索

核心逻辑:
  1. 从向量库中取一个"查询窗口"的所有spectra向量
  2. 在库中找每个向量的K个最近邻
  3. 最近邻对应的窗口 → 那些窗口的Z outcome
  4. 汇总: 这些相似窗口的Z是否有一致的方向？

验证方法:
  - 留出最后20%数据作为测试集
  - 用前80%建库
  - 对测试集每个窗口做查询, 看最近邻Z的方向多数投票 vs 实际方向
"""

import numpy as np
import faiss
import pickle
import time
import sys
import os


def load_store(path):
    """加载向量库"""
    with open(path, 'rb') as f:
        store = pickle.load(f)
    print(f"加载: {path}")
    print(f"  向量: {store['vectors'].shape}")
    print(f"  窗口: {len(store['window_end_bars'])}")
    print(f"  Z有效: {int(np.sum(store['z_valid']))}")
    return store


def build_faiss_index(vectors, index_type='flat'):
    """构建faiss索引"""
    d = vectors.shape[1]
    n = vectors.shape[0]
    
    if index_type == 'flat':
        index = faiss.IndexFlatL2(d)
    elif index_type == 'ivf':
        nlist = min(int(np.sqrt(n)), 1000)
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(vectors)
        index.nprobe = min(nlist // 4, 50)
    
    index.add(vectors)
    return index


def query_window(store, index, query_window_idx, k=50, 
                 exclude_nearby_windows=10):
    """
    查询一个窗口的所有spectra的最近邻。
    
    Args:
        store: 向量库
        index: faiss索引
        query_window_idx: 查询窗口的index
        k: 每个向量的K近邻数
        exclude_nearby_windows: 排除相邻窗口（避免自相关）
    
    Returns:
        neighbor_window_ids: 被投票的窗口ID集合
        z_outcomes: 对应的Z outcome
    """
    vectors = store['vectors']
    window_ids = store['window_ids']
    
    # 找到查询窗口的所有向量
    mask = window_ids == query_window_idx
    query_vecs = vectors[mask]
    
    if len(query_vecs) == 0:
        return np.array([]), np.array([])
    
    # 搜索
    D, I = index.search(query_vecs, k)
    
    # 收集邻居的窗口ID
    neighbor_vec_ids = I.flatten()
    neighbor_vec_ids = neighbor_vec_ids[neighbor_vec_ids >= 0]  # 过滤无效
    
    neighbor_win_ids = window_ids[neighbor_vec_ids]
    
    # 排除自身和相邻窗口
    valid = np.abs(neighbor_win_ids - query_window_idx) > exclude_nearby_windows
    neighbor_win_ids = neighbor_win_ids[valid]
    
    # 去重并计票
    unique_wins, counts = np.unique(neighbor_win_ids, return_counts=True)
    
    return unique_wins, counts


def evaluate_store(store_path, k=50, exclude_nearby=10, 
                   test_ratio=0.2, horizon_idx=1):
    """
    评估向量库的预测力。
    
    方法:
    - 前80%窗口建库
    - 后20%窗口做查询
    - 每个查询窗口: 找最近邻窗口 → 加权投票Z方向 → vs 实际Z
    
    Args:
        horizon_idx: 0=H10, 1=H20, 2=H50, 3=H100
    """
    store = load_store(store_path)
    
    vectors = store['vectors']
    window_ids = store['window_ids']
    end_bars = store['window_end_bars']
    z_outcomes = store['z_outcomes']
    z_valid = store['z_valid']
    z_horizons = store['config']['z_horizons']
    
    n_windows = len(end_bars)
    
    # 标准化向量
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    std[std < 1e-10] = 1.0
    vectors_norm = ((vectors - mean) / std).astype(np.float32)
    
    # 分割训练/测试
    split = int(n_windows * (1 - test_ratio))
    
    # 训练集: 窗口index < split 的所有向量
    train_mask = window_ids < split
    train_vecs = vectors_norm[train_mask]
    
    # 构建索引
    print(f"\n构建faiss索引: {train_vecs.shape[0]} vectors...")
    t0 = time.time()
    
    if train_vecs.shape[0] > 500000:
        index = build_faiss_index(train_vecs, 'ivf')
    else:
        index = build_faiss_index(train_vecs, 'flat')
    
    print(f"  耗时: {time.time()-t0:.1f}s")
    
    # 测试
    h = z_horizons[horizon_idx]
    z_dir_col = horizon_idx * 5  # direction column
    z_chg_col = horizon_idx * 5 + 3  # close_change column
    
    print(f"\n评估: H{h}, 训练窗口={split}, 测试窗口={n_windows - split}")
    print(f"  k={k}, exclude_nearby={exclude_nearby}")
    
    correct = 0
    total = 0
    predictions = []
    actuals = []
    
    t0 = time.time()
    
    for test_win in range(split, n_windows):
        if not z_valid[test_win]:
            continue
        
        actual_dir = z_outcomes[test_win, z_dir_col]
        actual_chg = z_outcomes[test_win, z_chg_col]
        
        # 查询向量
        mask = window_ids == test_win
        query_vecs = vectors_norm[mask]
        
        if len(query_vecs) == 0:
            continue
        
        # 搜索
        D, I = index.search(query_vecs, k)
        
        # 收集邻居窗口
        neighbor_vec_ids = I.flatten()
        neighbor_vec_ids = neighbor_vec_ids[(neighbor_vec_ids >= 0) & (neighbor_vec_ids < len(window_ids[train_mask]))]
        
        # 映射到训练集的window_ids
        train_win_ids = window_ids[train_mask]
        neighbor_win_ids = train_win_ids[neighbor_vec_ids]
        
        # 排除相邻窗口
        valid = np.abs(neighbor_win_ids - test_win) > exclude_nearby
        neighbor_win_ids = neighbor_win_ids[valid]
        
        if len(neighbor_win_ids) == 0:
            continue
        
        # 对应Z
        unique_wins, counts = np.unique(neighbor_win_ids, return_counts=True)
        
        # 加权投票
        weighted_dir = 0
        total_weight = 0
        for w, c in zip(unique_wins, counts):
            if z_valid[w]:
                weighted_dir += z_outcomes[w, z_dir_col] * c
                total_weight += c
        
        if total_weight == 0:
            continue
        
        pred_dir = 1.0 if weighted_dir > 0 else -1.0
        
        predictions.append(pred_dir)
        actuals.append(actual_dir)
        
        if pred_dir == actual_dir:
            correct += 1
        total += 1
    
    elapsed = time.time() - t0
    
    if total == 0:
        print("ERROR: 没有有效测试样本")
        return
    
    accuracy = correct / total
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    print(f"\n=== 结果: H{h} ===")
    print(f"  测试样本: {total}")
    print(f"  准确率: {accuracy:.4f} ({correct}/{total})")
    print(f"  baseline (全猜多): {np.mean(actuals > 0):.4f}")
    print(f"  lift: {accuracy - max(np.mean(actuals > 0), 1 - np.mean(actuals > 0)):.4f}")
    print(f"  耗时: {elapsed:.1f}s")
    
    # 分组看: 预测涨的实际涨了多少, 预测跌的实际跌了多少
    pred_up = predictions > 0
    pred_down = predictions < 0
    
    if np.sum(pred_up) > 0:
        actual_when_pred_up = actuals[pred_up]
        up_acc = np.mean(actual_when_pred_up > 0)
        print(f"\n  预测涨 ({int(np.sum(pred_up))}次): 实际涨 {up_acc:.4f}")
    
    if np.sum(pred_down) > 0:
        actual_when_pred_down = actuals[pred_down]
        down_acc = np.mean(actual_when_pred_down < 0)
        print(f"  预测跌 ({int(np.sum(pred_down))}次): 实际跌 {down_acc:.4f}")
    
    return {
        'accuracy': accuracy,
        'total': total,
        'predictions': predictions,
        'actuals': actuals,
    }


if __name__ == '__main__':
    test_path = '/home/ubuntu/stage2_abc/vecstore_test/store.pkl'
    full_path = '/home/ubuntu/stage2_abc/vecstore_h1/store.pkl'
    
    path = full_path if os.path.exists(full_path) else test_path
    
    if not os.path.exists(path):
        print("ERROR: 没有向量库, 请先运行 build_vector_store.py")
        sys.exit(1)
    
    # 对所有horizon评估
    for h_idx in range(4):
        print("\n" + "=" * 60)
        evaluate_store(path, k=50, exclude_nearby=10, 
                      test_ratio=0.2, horizon_idx=h_idx)
