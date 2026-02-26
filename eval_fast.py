#!/usr/bin/env python3
"""
快速评估向量库: 采样200向量/窗口, nprobe=10, k=30
先跑4个horizon的accuracy, 再做轻量shuffle test
"""
import pickle, numpy as np, faiss, time, sys

print('=== 快速向量库评估 ===')
print('加载...')
t0 = time.time()
with open('/home/ubuntu/stage2_abc/vecstore_h1/store.pkl','rb') as f:
    store = pickle.load(f)
print(f'加载: {time.time()-t0:.1f}s')

vectors = store['vectors']
window_ids = store['window_ids']
n_windows = len(store['window_end_bars'])
z_outcomes = store['z_outcomes']
z_valid = store['z_valid']
z_horizons = store['config']['z_horizons']
split = int(n_windows * 0.8)

# 标准化
print('标准化...')
mean = vectors.mean(axis=0)
std = vectors.std(axis=0)
std[std < 1e-10] = 1.0
vectors_norm = ((vectors - mean) / std).astype(np.float32)

train_mask = window_ids < split
train_vecs = vectors_norm[train_mask]
train_win_ids = window_ids[train_mask]

print(f'训练: {train_vecs.shape[0]} vectors, {split} windows')
print(f'测试: {n_windows - split} windows')

# 构建索引
faiss.omp_set_num_threads(80)
d = 15
nlist = 1000
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
t0 = time.time()
index.train(train_vecs)
index.add(train_vecs)
index.nprobe = 10
print(f'索引构建: {time.time()-t0:.1f}s')

# 预计算测试窗口位置
print('预计算窗口向量位置...')
test_positions = {}
for i, wid in enumerate(window_ids):
    if wid >= split:
        if wid not in test_positions:
            test_positions[wid] = []
        test_positions[wid].append(i)
print(f'测试窗口: {len(test_positions)}')

max_sample = 200
k = 30
exclude_nearby = 10
np.random.seed(42)

# ====== Phase 1: 对每个测试窗口做一次search, 保存投票结果 ======
# 这样shuffle test只需要重新统计, 不需要重新search
print('\n--- Phase 1: Search all test windows ---')
t0 = time.time()

# 每个测试窗口 → 邻居窗口投票 {win_id: count}
window_votes = {}  # test_win → dict{train_win: count}
skipped = 0

for idx, win_idx in enumerate(range(split, n_windows)):
    if not z_valid[win_idx] or win_idx not in test_positions:
        skipped += 1
        continue
    
    positions = test_positions[win_idx]
    if len(positions) > max_sample:
        sampled = np.random.choice(positions, max_sample, replace=False)
    else:
        sampled = np.array(positions)
    
    query_vecs = vectors_norm[sampled]
    D, I = index.search(query_vecs, k)
    
    valid_mask = (I >= 0) & (I < len(train_win_ids))
    neighbor_wins = np.full_like(I, -1)
    neighbor_wins[valid_mask] = train_win_ids[I[valid_mask]]
    neighbor_wins[(np.abs(neighbor_wins - win_idx) <= exclude_nearby)] = -1
    
    flat_wins = neighbor_wins.flatten()
    flat_wins = flat_wins[flat_wins >= 0]
    
    if len(flat_wins) == 0:
        skipped += 1
        continue
    
    unique_wins, counts = np.unique(flat_wins, return_counts=True)
    # 只保留z_valid的邻居
    votes = {}
    for w, c in zip(unique_wins, counts):
        if z_valid[w]:
            votes[int(w)] = int(c)
    
    if len(votes) > 0:
        window_votes[win_idx] = votes
    else:
        skipped += 1
    
    if (idx + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        eta = (n_windows - split - idx - 1) / rate
        print(f'  {idx+1}/{n_windows-split} ({elapsed:.0f}s, {rate:.0f}w/s, ETA {eta:.0f}s)')

elapsed = time.time() - t0
print(f'Search完成: {elapsed:.1f}s, 有效窗口: {len(window_votes)}, 跳过: {skipped}')

# ====== Phase 2: 对4个horizon评估 ======
print('\n--- Phase 2: Evaluate all horizons ---')

results = {}
for h_idx in range(4):
    h = z_horizons[h_idx]
    z_dir_col = h_idx * 5
    
    correct = 0
    total = 0
    preds = []
    actuals = []
    confs = []
    
    for win_idx, votes in window_votes.items():
        actual_dir = z_outcomes[win_idx, z_dir_col]
        
        weighted_dir = 0.0
        total_weight = 0.0
        for w, c in votes.items():
            weighted_dir += z_outcomes[w, z_dir_col] * c
            total_weight += c
        
        if total_weight == 0:
            continue
        
        conf = abs(weighted_dir) / total_weight
        pred_dir = 1.0 if weighted_dir > 0 else -1.0
        
        preds.append(pred_dir)
        actuals.append(actual_dir)
        confs.append(conf)
        
        if pred_dir == actual_dir:
            correct += 1
        total += 1
    
    preds = np.array(preds)
    actuals = np.array(actuals)
    confs = np.array(confs)
    
    accuracy = correct / total if total > 0 else 0
    up_ratio = np.mean(actuals > 0)
    baseline = max(up_ratio, 1 - up_ratio)
    
    print(f'\n{"="*60}')
    print(f'H{h}: accuracy={accuracy:.4f} ({correct}/{total})')
    print(f'  baseline={baseline:.4f}, lift={accuracy - baseline:+.4f}')
    print(f'  预测涨: {int(np.sum(preds>0))}, 预测跌: {int(np.sum(preds<0))}')
    
    for ct in [0.6, 0.7, 0.8]:
        hc = confs >= ct
        if hc.sum() > 20:
            hc_acc = np.mean(preds[hc] == actuals[hc])
            print(f'  conf>={ct}: {hc_acc:.4f} (n={int(hc.sum())})')
    
    if np.sum(preds>0) > 0:
        print(f'  预测涨→实际涨: {np.mean(actuals[preds>0]>0):.4f} (n={int(np.sum(preds>0))})')
    if np.sum(preds<0) > 0:
        print(f'  预测跌→实际跌: {np.mean(actuals[preds<0]<0):.4f} (n={int(np.sum(preds<0))})')
    
    results[h] = {'accuracy': accuracy, 'preds': preds, 'actuals': actuals, 'confs': confs}

# ====== Phase 3: Shuffle test (轻量, 只重新统计投票) ======
print('\n' + '='*60)
print('=== SHUFFLE TEST (1000 permutations, H20) ===')

h_idx = 1
h = z_horizons[h_idx]
z_dir_col = h_idx * 5
true_acc = results[h]['accuracy']

n_shuffle = 1000
shuffle_accs = np.zeros(n_shuffle)

# 获取有效测试窗口和邻居列表
test_wins = sorted(window_votes.keys())
# 预取z方向
all_z_dirs = z_outcomes[:, z_dir_col].copy()

for si in range(n_shuffle):
    # 只打乱训练集的Z方向
    shuffled_dirs = all_z_dirs[:split].copy()
    np.random.shuffle(shuffled_dirs)
    
    sc = 0
    st = 0
    for win_idx in test_wins:
        actual_dir = all_z_dirs[win_idx]  # 测试集实际方向不打乱
        votes = window_votes[win_idx]
        
        weighted_dir = 0.0
        total_weight = 0.0
        for w, c in votes.items():
            weighted_dir += shuffled_dirs[w] * c  # 使用打乱的训练集方向
            total_weight += c
        
        if total_weight == 0:
            continue
        
        pred_dir = 1.0 if weighted_dir > 0 else -1.0
        if pred_dir == actual_dir:
            sc += 1
        st += 1
    
    shuffle_accs[si] = sc / st if st > 0 else 0

percentile = np.mean(shuffle_accs <= true_acc) * 100
p_value = 1 - percentile / 100

print(f'True accuracy: {true_acc:.4f}')
print(f'Shuffle: mean={shuffle_accs.mean():.4f}, std={shuffle_accs.std():.4f}')
print(f'Shuffle: min={shuffle_accs.min():.4f}, max={shuffle_accs.max():.4f}')
print(f'Percentile: {percentile:.1f}%')
print(f'p-value: {p_value:.4f}')

if p_value < 0.01:
    print('*** SIGNIFICANT at p<0.01 ***')
elif p_value < 0.05:
    print('** SIGNIFICANT at p<0.05 **')
elif p_value < 0.10:
    print('* MARGINAL at p<0.10 *')
else:
    print('NOT SIGNIFICANT')

print('\nDONE.')
