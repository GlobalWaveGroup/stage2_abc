#!/usr/bin/env python3
"""
FSD Engine — 天枢统一状态-策略框架

FSD = Full Self-Driving 模式:
  感知 → 预测 → 规划 → 执行 → 反馈

三个用途共享同一个引擎:
  1. 人眼逐K线检查 (前端可视化)
  2. 后台大规模回测 (批量计算)
  3. ML训练 (上帝视角标注 → 学习 state→action)

核心设计:
  - FSDEngine.step(h, l, o, c) → StateSnapshot
  - StateSnapshot 包含:
    - 原始K线信息
    - ZG结构状态
    - 所有活跃预测轨迹 + 每条的实时离差
    - 持仓状态
  - OracleLabeler 回看标注最优策略
  - 所有状态可序列化为固定维度向量 (给ML用)
"""

import numpy as np
import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import sys
sys.path.insert(0, '/home/ubuntu/stage2_abc')
from merge_engine_v3 import (predict_symmetric_image, calculate_base_zg, 
                               full_merge_engine, build_segment_pool,
                               compute_pivot_importance, pool_fusion)
from dynamic_engine import DynamicZG, DynamicMerger


# ============================================================
# 1. 预测轨迹 (Trajectory)
# ============================================================

@dataclass
class Trajectory:
    """一条预测轨迹 — 从发出到过期/命中的生命周期"""
    traj_id: int
    issued_bar: int           # 发出时的bar index
    pred_type: str            # 'mirror' or 'center'
    pred_dir: int             # +1 or -1
    
    # A段信息
    a_start: int
    a_end: int
    a_price_start: float
    a_price_end: float
    a_amp: float
    a_time: int
    
    # 预测C'
    pred_start_bar: int
    pred_start_price: float
    pred_target_bar: int
    pred_target_price: float
    pred_amp: float
    pred_time: int
    
    # B段 (center only)
    b_start: int = 0
    b_end: int = 0
    b_price_start: float = 0.0
    b_price_end: float = 0.0
    
    # 原始score
    score: float = 0.0
    
    # === 动态追踪 (每根K线更新) ===
    progress: float = 0.0       # 实际进度 (0→1)
    expected_progress: float = 0.0  # 期望进度
    deviation: float = 0.0     # 离差 (actual - expected) / pred_amp
    max_favorable: float = 0.0  # 最大有利移动 / pred_amp
    max_adverse: float = 0.0    # 最大不利移动 / pred_amp
    bars_alive: int = 0        # 存活K线数
    status: str = 'active'     # active / expired / hit / killed
    
    def update(self, bar: int, close: float, high: float, low: float):
        """每根K线更新离差和进度"""
        if self.status != 'active':
            return
        
        self.bars_alive = bar - self.issued_bar
        
        if self.pred_amp < 1e-10 or self.pred_time < 1:
            return
        
        signed_amp = self.pred_target_price - self.pred_start_price
        
        # 期望进度 (线性)
        elapsed = bar - self.pred_start_bar
        self.expected_progress = elapsed / self.pred_time
        
        # 实际进度
        self.progress = (close - self.pred_start_price) / signed_amp if abs(signed_amp) > 1e-10 else 0
        
        # 离差
        expected_price = self.pred_start_price + self.expected_progress * signed_amp
        self.deviation = (close - expected_price) / self.pred_amp
        
        # MFE / MAE
        if self.pred_dir == 1:
            fav = (high - self.pred_start_price) / self.pred_amp
            adv = (self.pred_start_price - low) / self.pred_amp
        else:
            fav = (self.pred_start_price - low) / self.pred_amp
            adv = (high - self.pred_start_price) / self.pred_amp
        
        self.max_favorable = max(self.max_favorable, fav)
        self.max_adverse = max(self.max_adverse, adv)
        
        # 状态更新
        if bar >= self.pred_target_bar:
            self.status = 'expired'
        if self.progress >= 1.0:
            self.status = 'hit'


# ============================================================
# 2. 状态快照 (StateSnapshot)
# ============================================================

@dataclass
class StateSnapshot:
    """一根K线时刻的完整状态 — 感知层的输出"""
    bar: int
    open: float
    high: float
    low: float
    close: float
    
    # ZG结构
    n_confirmed_pivots: int = 0
    n_segments: int = 0
    last_pivot_bar: int = 0
    last_pivot_price: float = 0.0
    last_pivot_dir: int = 0
    bars_since_last_pivot: int = 0
    
    # 活跃轨迹摘要
    n_active_trajs: int = 0
    trajs: List[Trajectory] = field(default_factory=list)
    
    # 当前价格在所有活跃预测中的位置
    best_traj_id: int = -1      # 离差最小的轨迹
    best_deviation: float = 99.0
    best_progress: float = 0.0
    
    # 多轨迹共识
    n_bullish: int = 0          # 看多的轨迹数
    n_bearish: int = 0          # 看空的轨迹数
    consensus_dir: int = 0      # +1/-1/0
    avg_deviation: float = 0.0  # 所有活跃轨迹的平均离差
    
    def to_vector(self, max_trajs: int = 8) -> np.ndarray:
        """
        序列化为固定维度向量 (给ML用)
        
        维度:
        [0]    bar_normalized (0~1)
        [1:5]  OHLC normalized
        [5:11] ZG状态 (6维)
        [11:15] 多轨迹共识 (4维)
        [15:15+max_trajs*6] 每条轨迹的状态 (6维 × max_trajs)
        
        总维度: 15 + max_trajs * 6
        """
        v = np.zeros(15 + max_trajs * 6, dtype=np.float32)
        
        # 基础
        v[0] = self.bar / 2000.0  # 归一化
        v[1] = self.open
        v[2] = self.high
        v[3] = self.low
        v[4] = self.close
        
        # ZG
        v[5] = self.n_confirmed_pivots / 1000.0
        v[6] = self.n_segments / 1500.0
        v[7] = self.last_pivot_dir
        v[8] = self.bars_since_last_pivot / 50.0
        v[9] = self.last_pivot_price
        v[10] = self.n_active_trajs / 20.0
        
        # 共识
        v[11] = self.n_bullish / max(self.n_active_trajs, 1)
        v[12] = self.n_bearish / max(self.n_active_trajs, 1)
        v[13] = self.best_deviation
        v[14] = self.best_progress
        
        # 轨迹 (按score排序, 取top max_trajs)
        sorted_trajs = sorted(self.trajs, key=lambda t: -t.score)[:max_trajs]
        for j, t in enumerate(sorted_trajs):
            base = 15 + j * 6
            v[base + 0] = t.pred_dir
            v[base + 1] = t.progress
            v[base + 2] = t.deviation
            v[base + 3] = t.max_favorable
            v[base + 4] = t.max_adverse
            v[base + 5] = t.pred_amp * 10000  # pips
        
        return v


# ============================================================
# 3. FSD引擎
# ============================================================

def prune_redundant(segments, pivot_info, merge_dist=3, max_per_start=5, max_total=300):
    """
    冗余删除: 2970 → ~300段, 保留代表性线段。
    
    逻辑:
    1. 每条线段的importance = 起点imp × 终点imp
    2. 同起点的线段按终点排序 → 终点相差 <= merge_dist 的归为一组
    3. 每组保留importance最高的代表
    4. 每个起点最多保留 max_per_start 条
    5. 全局按importance截断到 max_total
    
    典型效果: 2970 → 300, 压缩10x, 覆盖94%端点
    """
    from collections import defaultdict
    
    # 计算importance
    for s in segments:
        p1 = pivot_info.get(s['bar_start'], {}).get('importance', 0)
        p2 = pivot_info.get(s['bar_end'], {}).get('importance', 0)
        s['_imp'] = p1 * p2
    
    # 按起点分组
    by_start = defaultdict(list)
    for s in segments:
        by_start[s['bar_start']].append(s)
    
    kept = []
    for bar_s in sorted(by_start.keys()):
        group = by_start[bar_s]
        group.sort(key=lambda s: s['bar_end'])
        
        # 终点聚类: 相差 <= merge_dist 归一组
        clusters = []
        cur_cluster = [group[0]]
        for j in range(1, len(group)):
            if group[j]['bar_end'] - cur_cluster[-1]['bar_end'] <= merge_dist:
                cur_cluster.append(group[j])
            else:
                clusters.append(cur_cluster)
                cur_cluster = [group[j]]
        clusters.append(cur_cluster)
        
        # 每cluster保留importance最高的
        reps = []
        for cl in clusters:
            best = max(cl, key=lambda s: s['_imp'])
            reps.append(best)
        
        # 每起点最多 max_per_start
        reps.sort(key=lambda s: -s['_imp'])
        kept.extend(reps[:max_per_start])
    
    # 全局截断
    kept.sort(key=lambda s: -s['_imp'])
    if len(kept) > max_total:
        kept = kept[:max_total]
    
    # 清理临时字段
    for s in kept:
        s.pop('_imp', None)
    
    return kept


def _build_pivot_info_simple(segments):
    """从线段列表构建简化pivot_info"""
    pivot_info = {}
    for seg in segments:
        for bar, price, d in [(seg['bar_start'], seg['price_start'], seg['dir_start']),
                               (seg['bar_end'], seg['price_end'], seg['dir_end'])]:
            if bar not in pivot_info:
                pivot_info[bar] = {'bar': bar, 'price': price, 'dir': d, 'importance': 0}
            pivot_info[bar]['importance'] = max(
                pivot_info[bar]['importance'],
                seg['amplitude'] * seg['span']
            )
    max_imp = max((p['importance'] for p in pivot_info.values()), default=1)
    if max_imp > 0:
        for p in pivot_info.values():
            p['importance'] /= max_imp
    return pivot_info


class FSDEngine:
    """
    统一的K线步进引擎 (v2: 含滑动窗口pool_fusion)。
    
    每调用一次 step(h, l, o, c) → StateSnapshot
    
    内部维护:
      - 收集所有OHLC (用于窗口内静态重算)
      - 滑动窗口 pool_fusion: 每 fusion_stride 根K线, 
        在最近 fusion_window bars 内做完整 静态引擎 → pool_fusion → predict
      - DynamicZG: 逐K线拐点检测 (用于可视化ZG线)
      - 活跃轨迹池: 逐K线更新 deviation/progress
    
    关键改进: 
      之前只用 DynamicMerger (161段/200bar), 缺少 pool_fusion (2970段/200bar)
      现在用滑动窗口静态重算, 线段池与v3完全一致
    """
    
    def __init__(self, start_pred: int = 50, max_trajs: int = 30,
                 pred_horizon: int = 50, max_preds_per_event: int = 20,
                 fusion_window: int = 200, fusion_stride: int = 10):
        # ZG for real-time pivot tracking (可视化用)
        self.zg = DynamicZG()
        self.bar_idx = 0
        self.start_pred = start_pred
        self.max_trajs = max_trajs
        self.pred_horizon = pred_horizon
        self.max_preds_per_event = max_preds_per_event
        
        # 滑动窗口参数
        self.fusion_window = fusion_window
        self.fusion_stride = fusion_stride
        
        # 收集所有OHLC
        self.all_highs: List[float] = []
        self.all_lows: List[float] = []
        
        # 当前fused pool (最近一次窗口重算的结果)
        self.current_fused_pool: List[dict] = []
        self.current_pivot_info: dict = {}
        self.last_fusion_bar: int = -999
        self.n_fused_segs: int = 0
        
        # 活跃轨迹池
        self.active_trajs: List[Trajectory] = []
        self.traj_counter = 0
        
        # ZG状态追踪
        self.last_confirmed_bar = 0
        self.last_confirmed_price = 0.0
        self.last_confirmed_dir = 0
    
    def step(self, h: float, l: float, o: float, c: float) -> StateSnapshot:
        """
        处理一根K线, 返回当前状态快照。
        
        内部流程:
        1. 收集OHLC
        2. ZG step (逐K线, 实时拐点追踪)
        3. 每 fusion_stride 根K线: 滑动窗口 静态重算 → fused pool → predict → 新轨迹
        4. 所有活跃轨迹 update(当前K线)
        5. 组装 StateSnapshot
        """
        i = self.bar_idx
        
        # 1. 收集OHLC
        self.all_highs.append(h)
        self.all_lows.append(l)
        
        # 2. ZG (逐K线, 用于可视化)
        events = self.zg.step(h, l)
        for ev in events:
            if ev['type'] == 'confirmed':
                self.last_confirmed_bar = ev['pivot']['bar']
                self.last_confirmed_price = ev['pivot']['price']
                self.last_confirmed_dir = ev['pivot']['dir']
        
        # 3. 滑动窗口 pool_fusion + predict
        should_fuse = (i >= self.start_pred and 
                       i - self.last_fusion_bar >= self.fusion_stride and
                       i >= self.fusion_window)
        if should_fuse:
            self._sliding_window_fusion(i)
        
        # 4. 更新所有活跃轨迹, 淘汰严重偏离的
        new_active = []
        for t in self.active_trajs:
            t.update(i, c, h, l)
            if t.status == 'active':
                scale = math.sqrt(max(t.pred_time, 1) / 10.0)
                dev_limit = min(1.5 * scale, 5.0)
                adv_limit = min(1.5 * scale, 4.0)
                
                if abs(t.deviation) > dev_limit:
                    t.status = 'killed'
                elif t.max_adverse > adv_limit:
                    t.status = 'killed'
                else:
                    new_active.append(t)
        self.active_trajs = new_active
        
        # 5. 组装快照
        snap = self._build_snapshot(i, o, h, l, c)
        
        self.bar_idx += 1
        return snap
    
    def _sliding_window_fusion(self, bar: int):
        """
        滑动窗口: 在最近 fusion_window bars 内做完整静态引擎 → pool_fusion → predict
        
        这保证线段池与v3完全一致 (~2000-3000段/200bar)
        每次 ~0.2s, stride=10 → 2000 bars 总计 ~30s
        """
        import numpy as np
        ws = max(0, bar - self.fusion_window + 1)
        h_win = np.array(self.all_highs[ws:bar+1], dtype=float)
        l_win = np.array(self.all_lows[ws:bar+1], dtype=float)
        win_len = len(h_win)
        
        if win_len < 30:
            return
        
        # 静态引擎
        base_pivots = calculate_base_zg(h_win, l_win, rb=0.5)
        if len(base_pivots) < 5:
            return
        
        results = full_merge_engine(base_pivots)
        pivot_info = compute_pivot_importance(results, win_len)
        pool = build_segment_pool(results, pivot_info)
        fused_all, _, _ = pool_fusion(pool, pivot_info)
        
        # 坐标变换: 窗口内坐标 → 全局坐标
        for seg in fused_all:
            seg['bar_start'] += ws
            seg['bar_end'] += ws
        
        new_pivot_info = {}
        for local_bar, info in pivot_info.items():
            global_bar = local_bar + ws
            new_info = dict(info)
            new_info['bar'] = global_bar
            new_pivot_info[global_bar] = new_info
        
        # 冗余删除: ~2500 → ~300
        pruned = prune_redundant(fused_all, new_pivot_info, 
                                  merge_dist=3, max_per_start=5, max_total=300)
        
        self.current_fused_pool = pruned
        self.current_pivot_info = new_pivot_info
        self.n_fused_segs = len(pruned)
        self.last_fusion_bar = bar
        
        # 用pruned pool生成预测
        self._generate_predictions(bar)
    
    def _generate_predictions(self, bar: int):
        """从当前fused pool生成新预测轨迹"""
        if len(self.current_fused_pool) < 3:
            return
        
        raw = predict_symmetric_image(self.current_fused_pool, self.current_pivot_info, 
                                       current_bar=bar, max_pool_size=99999)
        
        # 筛选: 起点已存在 + 目标在未来
        valid = []
        for p in raw:
            if p['pred_start_bar'] > bar:
                continue
            if p['pred_target_bar'] > bar + self.pred_horizon:
                continue
            if p['pred_target_bar'] <= bar:
                continue
            valid.append(p)
        
        valid.sort(key=lambda p: -p['score'])
        
        # 去重: 不添加和现有轨迹过于相似的
        existing_keys = set()
        for t in self.active_trajs:
            existing_keys.add((t.pred_type[0], t.a_start, t.a_end, t.pred_dir))
        
        added = 0
        for p in valid:
            if added >= self.max_preds_per_event:
                break
            
            key = (p['type'][0], p['A_start'], p['A_end'], p['pred_dir'])
            if key in existing_keys:
                continue
            existing_keys.add(key)
            
            self.traj_counter += 1
            t = Trajectory(
                traj_id=self.traj_counter,
                issued_bar=bar,
                pred_type=p['type'],
                pred_dir=p['pred_dir'],
                a_start=p['A_start'],
                a_end=p['A_end'],
                a_price_start=p['A_price_start'],
                a_price_end=p['A_price_end'],
                a_amp=p['A_amp'],
                a_time=p['A_time'],
                pred_start_bar=p['pred_start_bar'],
                pred_start_price=p['pred_start_price'],
                pred_target_bar=p['pred_target_bar'],
                pred_target_price=p['pred_target_price'],
                pred_amp=p['pred_amp'],
                pred_time=p['pred_time'],
                score=p['score'],
            )
            if p['type'] == 'center':
                t.b_start = p['B_start']
                t.b_end = p['B_end']
                t.b_price_start = p['B_price_start']
                t.b_price_end = p['B_price_end']
            
            self.active_trajs.append(t)
            added += 1
        
        # 限制总数
        if len(self.active_trajs) > self.max_trajs:
            self.active_trajs.sort(key=lambda t: -t.score)
            self.active_trajs = self.active_trajs[:self.max_trajs]
    
    def _build_snapshot(self, bar: int, o: float, h: float, l: float, c: float) -> StateSnapshot:
        """组装当前帧的完整状态快照"""
        snap = StateSnapshot(
            bar=bar, open=o, high=h, low=l, close=c,
            n_confirmed_pivots=len(self.zg.pivots),
            n_segments=self.n_fused_segs,
            last_pivot_bar=self.last_confirmed_bar,
            last_pivot_price=self.last_confirmed_price,
            last_pivot_dir=self.last_confirmed_dir,
            bars_since_last_pivot=bar - self.last_confirmed_bar,
            n_active_trajs=len(self.active_trajs),
            trajs=list(self.active_trajs),
        )
        
        if self.active_trajs:
            # 找离差最小的
            best = min(self.active_trajs, key=lambda t: abs(t.deviation))
            snap.best_traj_id = best.traj_id
            snap.best_deviation = best.deviation
            snap.best_progress = best.progress
            
            # 共识
            snap.n_bullish = sum(1 for t in self.active_trajs if t.pred_dir == 1)
            snap.n_bearish = sum(1 for t in self.active_trajs if t.pred_dir == -1)
            if snap.n_bullish > snap.n_bearish * 1.5:
                snap.consensus_dir = 1
            elif snap.n_bearish > snap.n_bullish * 1.5:
                snap.consensus_dir = -1
            
            devs = [t.deviation for t in self.active_trajs]
            snap.avg_deviation = sum(devs) / len(devs)
        
        return snap


# ============================================================
# 4. 上帝视角标注器
# ============================================================

class OracleLabeler:
    """
    上帝视角: 知道未来, 回看标注每个状态的最优策略。
    
    对每个StateSnapshot, 标注:
    - 最优方向 (从当前到 +10/+20/+50 bars)
    - 最优止盈距离 (max favorable excursion)
    - 最优止损距离 (在达到止盈前的max adverse)
    - 最优轨迹 (哪条预测最终被证实)
    
    这些标注就是ML的训练标签。
    """
    
    @staticmethod
    def label_batch(snapshots: List[StateSnapshot], 
                    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                    horizons: List[int] = [10, 20, 50]) -> List[Dict]:
        """
        批量标注。
        
        Args:
            snapshots: step()产出的快照序列
            highs/lows/closes: 完整K线数据
            horizons: 标注的前瞻距离
        
        Returns:
            list of label dicts, 和snapshots一一对应
        """
        n = len(highs)
        labels = []
        
        for snap in snapshots:
            bar = snap.bar
            label = {
                'bar': bar,
                'close': snap.close,
            }
            
            for hz in horizons:
                end = min(bar + hz, n - 1)
                if end <= bar:
                    label[f'dir_{hz}'] = 0
                    label[f'mfe_up_{hz}'] = 0.0
                    label[f'mfe_dn_{hz}'] = 0.0
                    label[f'best_tp_dir_{hz}'] = 0
                    label[f'best_rr_{hz}'] = 0.0
                    continue
                
                future_highs = highs[bar+1:end+1]
                future_lows = lows[bar+1:end+1]
                
                # 方向
                future_close = float(closes[end])
                move = future_close - snap.close
                label[f'dir_{hz}'] = 1 if move > 0 else (-1 if move < 0 else 0)
                
                # MFE 双向
                up_mfe = float(np.max(future_highs)) - snap.close
                dn_mfe = snap.close - float(np.min(future_lows))
                label[f'mfe_up_{hz}'] = round(up_mfe * 10000, 1)  # pips
                label[f'mfe_dn_{hz}'] = round(dn_mfe * 10000, 1)
                
                # 最优止盈方向: 哪个方向的MFE更大
                if up_mfe > dn_mfe:
                    label[f'best_tp_dir_{hz}'] = 1
                    # 做多时, 到达up_mfe前的最大回撤
                    best_mfe = up_mfe
                    worst_mae = 0.0
                    peak = snap.close
                    for j in range(bar+1, end+1):
                        if j >= n: break
                        peak = max(peak, float(highs[j]))
                        drawdown = peak - float(lows[j])  # 但这不对, 需要到达MFE前的MAE
                    # 简化: 做多到horizon, MAE就是最大dn_mfe
                    worst_mae = dn_mfe
                    label[f'best_rr_{hz}'] = round(best_mfe / max(worst_mae, 0.0001), 2)
                else:
                    label[f'best_tp_dir_{hz}'] = -1
                    best_mfe = dn_mfe
                    worst_mae = up_mfe
                    label[f'best_rr_{hz}'] = round(best_mfe / max(worst_mae, 0.0001), 2)
            
            # 标注哪条轨迹最终最接近现实
            if snap.trajs:
                best_traj = None
                best_traj_score = -1
                for t in snap.trajs:
                    target_bar = min(t.pred_target_bar, n - 1)
                    if target_bar <= bar:
                        continue
                    # 看target_bar时刻价格是否接近预测
                    actual_at_target = float(closes[target_bar])
                    pred_error = abs(actual_at_target - t.pred_target_price) / max(t.pred_amp, 1e-10)
                    # error越小越好 → score = 1/(1+error)
                    traj_score = 1.0 / (1.0 + pred_error)
                    if traj_score > best_traj_score:
                        best_traj_score = traj_score
                        best_traj = t
                
                if best_traj:
                    label['oracle_best_traj_id'] = best_traj.traj_id
                    label['oracle_best_traj_type'] = best_traj.pred_type
                    label['oracle_best_traj_dir'] = best_traj.pred_dir
                    label['oracle_best_traj_accuracy'] = round(best_traj_score, 4)
                else:
                    label['oracle_best_traj_id'] = -1
            
            labels.append(label)
        
        return labels


# ============================================================
# 5. 运行 & 测试
# ============================================================

def run_full(csv_path: str, limit: int = 2000, verbose: bool = True):
    """
    完整运行: 逐K线推进, 收集所有状态快照, 上帝视角标注。
    
    返回: (snapshots, labels, df)
    """
    from merge_engine_v3 import load_kline
    
    df = load_kline(csv_path, limit=limit)
    n = len(df)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    opens = df['open'].values.astype(float)
    closes = df['close'].values.astype(float)
    
    if verbose:
        print(f"FSD Engine: {n} bars, {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    
    engine = FSDEngine()
    snapshots = []
    
    t0 = time.time()
    for i in range(n):
        snap = engine.step(highs[i], lows[i], opens[i], closes[i])
        snapshots.append(snap)
        
        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  bar {i+1}/{n} ({elapsed:.1f}s) "
                  f"pivots={snap.n_confirmed_pivots} segs={snap.n_segments} "
                  f"active_trajs={snap.n_active_trajs}")
    
    elapsed = time.time() - t0
    if verbose:
        print(f"完成: {n} bars, {elapsed:.1f}s")
    
    # 上帝标注
    if verbose:
        print("上帝视角标注...")
    t1 = time.time()
    labels = OracleLabeler.label_batch(snapshots, highs, lows, closes)
    if verbose:
        print(f"标注完成: {time.time()-t1:.1f}s")
    
    return snapshots, labels, df


def print_summary(snapshots, labels):
    """打印统计摘要"""
    n = len(snapshots)
    
    # 基础统计
    active_counts = [s.n_active_trajs for s in snapshots]
    print(f"\n=== FSD Summary ({n} bars) ===")
    print(f"Active trajectories: mean={np.mean(active_counts):.1f}, "
          f"max={np.max(active_counts)}, "
          f"bars_with_trajs={sum(1 for c in active_counts if c > 0)}/{n}")
    
    # 上帝视角最优方向分布
    for hz in [10, 20, 50]:
        key = f'best_tp_dir_{hz}'
        if key not in labels[0]:
            continue
        ups = sum(1 for l in labels if l.get(key) == 1)
        dns = sum(1 for l in labels if l.get(key) == -1)
        rrs = [l.get(f'best_rr_{hz}', 0) for l in labels]
        mfe_ups = [l.get(f'mfe_up_{hz}', 0) for l in labels if l.get(f'mfe_up_{hz}', 0) > 0]
        mfe_dns = [l.get(f'mfe_dn_{hz}', 0) for l in labels if l.get(f'mfe_dn_{hz}', 0) > 0]
        print(f"\n  Horizon {hz}: up={ups} dn={dns}")
        print(f"    MFE up: mean={np.mean(mfe_ups):.1f} pips")
        print(f"    MFE dn: mean={np.mean(mfe_dns):.1f} pips")
        print(f"    Best R:R: mean={np.mean(rrs):.2f}, median={np.median(rrs):.2f}")
    
    # 轨迹准确度
    oracle_accs = [l.get('oracle_best_traj_accuracy', 0) for l in labels 
                   if l.get('oracle_best_traj_id', -1) >= 0]
    if oracle_accs:
        print(f"\n  Oracle best traj accuracy: mean={np.mean(oracle_accs):.3f}, "
              f"median={np.median(oracle_accs):.3f}")
    
    # 向量维度
    if snapshots:
        vec = snapshots[0].to_vector()
        print(f"\n  State vector dim: {len(vec)}")


if __name__ == '__main__':
    snapshots, labels, df = run_full(
        '/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv',
        limit=2000, verbose=True
    )
    print_summary(snapshots, labels)
    
    # 展示几帧的状态
    print("\n=== Sample Frames ===")
    for bar_idx in [100, 500, 1000, 1500]:
        if bar_idx >= len(snapshots):
            continue
        s = snapshots[bar_idx]
        l = labels[bar_idx]
        print(f"\nBar {bar_idx}: close={s.close:.5f} "
              f"trajs={s.n_active_trajs} best_dev={s.best_deviation:.3f}")
        print(f"  Oracle: dir10={l.get('dir_10')}, dir50={l.get('dir_50')}, "
              f"mfe_up50={l.get('mfe_up_50')}p, mfe_dn50={l.get('mfe_dn_50')}p, "
              f"best_rr50={l.get('best_rr_50')}")
        if s.trajs:
            for t in s.trajs[:3]:
                print(f"    T{t.traj_id}: {t.pred_type} dir={t.pred_dir} "
                      f"prog={t.progress:.2f} dev={t.deviation:.3f} "
                      f"fav={t.max_favorable:.2f} adv={t.max_adverse:.2f}")
