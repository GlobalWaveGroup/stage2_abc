#!/usr/bin/env python3
"""
动态引擎 v1.0 — 逐K线推进，无前瞻

核心思想:
  静态引擎是"事后分析" — 看到所有K线后一次性计算
  动态引擎是"实时推进" — 每根K线到来时增量更新

  关键区别:
  1. zigzag拐点有birth_bar(确认时刻)，在确认前不存在
  2. 归并在新拐点确认时增量执行，已固定的不重算
  3. pool_fusion在新线段产生时增量搜索
  4. 对称谱在新线段加入时增量计算
  5. 每条线段/每个对称谱都有birth_bar，标记"何时开始存在"

  这意味着: 在bar_t时刻能看到的线段/对称谱，
  严格是用[0, bar_t]数据能确认产生的那些。
  没有任何前瞻。

架构:
  DynamicZG        — 逐K线zigzag, 维护前向状态, 记录birth_bar
  DynamicMerger    — 增量归并(幅度+横向), 固定段不重算
  DynamicPool      — 增量pool_fusion + 对称谱
  DynamicEngine    — 整合以上三者, 提供step()接口
"""

import numpy as np
import math
import time as _time
from collections import defaultdict
from merge_engine_v3 import load_kline


# =============================================================================
# 1. 动态ZG — 逐K线zigzag
# =============================================================================

class DynamicZG:
    """
    逐K线推进的zigzag。
    
    和静态版的核心区别:
    - 没有backward pass (那是事后修复，实时中不存在)
    - 维护前向状态，每次feed一根K线
    - 最后一个拐点是"临时的"(tentative)，可能被后续K线修改
    - 当方向反转时，前一个临时拐点变为"确认的"(confirmed)
    - birth_bar: 拐点被确认的那根K线的index
    
    拐点格式: {
        'bar': int,          # 拐点所在K线index
        'price': float,      # 拐点价格
        'dir': int,          # +1=峰, -1=谷
        'birth_bar': int,    # 被确认的K线index (对于临时拐点 = -1)
        'confirmed': bool,   # 是否已确认
    }
    """
    
    def __init__(self, rb=0.5):
        self.rb = rb
        self.pivots = []          # 所有确认的拐点
        self.tentative = None     # 当前临时拐点 (最后一个, 未确认)
        
        # KZig前向状态
        self.last_state = 0       # 0=未初始化, 1=上升, -1=下降
        self.last_pos = 0         # 最后一个拐点的bar
        self.p_pos = 0            # 前一个拐点的bar (for range计算)
        self.last_range = 9999.0
        
        # K线数据
        self.high = []
        self.low = []
        self.bar_count = 0
        
        # 事件记录: 每次step产生的变化
        self.last_events = []     # [{'type': 'confirmed'|'updated'|'new_tentative', 'pivot': {...}}]
    
    def get_all_pivots(self):
        """返回所有拐点(确认+临时)"""
        result = list(self.pivots)
        if self.tentative:
            result.append(self.tentative)
        return result
    
    def get_confirmed_pivots(self):
        """只返回已确认的拐点"""
        return list(self.pivots)
    
    def step(self, h, l):
        """
        送入一根新K线的high/low。
        
        返回: list of events (本次step产生的变化)
        """
        self.high.append(h)
        self.low.append(l)
        i = self.bar_count
        self.bar_count += 1
        self.last_events = []
        
        if i == 0:
            # 第一根K线 — 初始化
            self.last_state = 1
            self.last_pos = 0
            self.p_pos = 0
            self.last_range = 9999.0
            self.tentative = {
                'bar': 0, 'price': h, 'dir': 1,
                'birth_bar': -1, 'confirmed': False,
            }
            self.last_events.append({'type': 'new_tentative', 'pivot': dict(self.tentative)})
            return self.last_events
        
        # KZig前向逻辑 (和静态版相同)
        is_up = False
        is_down = False
        
        prev_h = self.high[i-1]
        prev_l = self.low[i-1]
        
        if ((h > prev_h and l >= prev_l) or
            (self.last_state == -1 and h - self.low[self.last_pos] > self.last_range * self.rb and
             not (h < prev_h and l < prev_l)) or
            (self.last_state == -1 and i - self.last_pos > 1 and h > self.low[self.last_pos] and
             not (h < prev_h and l < prev_l))):
            is_up = True
        
        if ((h <= prev_h and l < prev_l) or
            (self.last_state == 1 and self.high[self.last_pos] - l > self.last_range * self.rb and
             not (h > prev_h and l > prev_l)) or
            (self.last_state == 1 and i - self.last_pos > 1 and l < self.high[self.last_pos] and
             not (h > prev_h and l > prev_l))):
            is_down = True
        
        if is_up:
            if self.last_state == 1:
                # 同方向延续 — 只有新高才更新临时拐点
                if h < self.high[self.last_pos]:
                    return self.last_events  # 没有新高，忽略
                # 更新临时拐点位置
                old_tentative = dict(self.tentative) if self.tentative else None
                self.tentative = {
                    'bar': i, 'price': h, 'dir': 1,
                    'birth_bar': -1, 'confirmed': False,
                }
                self.last_pos = i
                self.last_range = h - self.low[self.p_pos]
                self.last_events.append({'type': 'updated', 'pivot': dict(self.tentative),
                                         'old': old_tentative})
            else:
                # 方向反转: 下降→上升
                # 前一个临时拐点(谷)被确认
                if self.tentative and not self.tentative['confirmed']:
                    self.tentative['confirmed'] = True
                    self.tentative['birth_bar'] = i  # 被这根K线确认
                    self.pivots.append(self.tentative)
                    self.last_events.append({'type': 'confirmed', 'pivot': dict(self.tentative)})
                
                self.p_pos = self.last_pos
                # 新的临时拐点(峰)
                self.tentative = {
                    'bar': i, 'price': h, 'dir': 1,
                    'birth_bar': -1, 'confirmed': False,
                }
                self.last_pos = i
                self.last_state = 1
                self.last_range = h - self.low[self.p_pos]
                self.last_events.append({'type': 'new_tentative', 'pivot': dict(self.tentative)})
        
        elif is_down:
            if self.last_state == -1:
                # 同方向延续 — 只有新低才更新临时拐点
                if l > self.low[self.last_pos]:
                    return self.last_events  # 没有新低，忽略
                old_tentative = dict(self.tentative) if self.tentative else None
                self.tentative = {
                    'bar': i, 'price': l, 'dir': -1,
                    'birth_bar': -1, 'confirmed': False,
                }
                self.last_pos = i
                self.last_range = self.high[self.p_pos] - l
                self.last_events.append({'type': 'updated', 'pivot': dict(self.tentative),
                                         'old': old_tentative})
            else:
                # 方向反转: 上升→下降
                # 前一个临时拐点(峰)被确认
                if self.tentative and not self.tentative['confirmed']:
                    self.tentative['confirmed'] = True
                    self.tentative['birth_bar'] = i
                    self.pivots.append(self.tentative)
                    self.last_events.append({'type': 'confirmed', 'pivot': dict(self.tentative)})
                
                self.p_pos = self.last_pos
                self.tentative = {
                    'bar': i, 'price': l, 'dir': -1,
                    'birth_bar': -1, 'confirmed': False,
                }
                self.last_pos = i
                self.last_state = -1
                self.last_range = self.high[self.p_pos] - l
                self.last_events.append({'type': 'new_tentative', 'pivot': dict(self.tentative)})
        
        return self.last_events


# =============================================================================
# 2. 动态归并器 — 增量幅度+横向归并
# =============================================================================

class DynamicMerger:
    """
    增量归并器。
    
    当DynamicZG确认新拐点时，触发增量归并检查。
    已固定的归并结果不变，只处理尾部新变化的部分。
    
    核心逻辑:
    - 维护一个"活跃拐点序列" (和静态版的full_merge_engine中的current对应)
    - 但不是从头重算，而是只检查尾部4个拐点是否触发归并
    - 归并产生的新线段记录birth_bar
    
    和静态版的对应关系:
    - 静态: full_merge_engine() 对整个序列做多轮交替迭代
    - 动态: 每次新拐点加入后，只检查尾部，可能触发连锁归并
    """
    
    def __init__(self):
        # 多级归并链 — 每级一个拐点序列
        # levels[0] = base ZG确认拐点序列
        # levels[1] = 第一轮归并后的序列
        # ...
        # 动态版中我们用扁平方式: 维护一个"工作序列"
        # 当尾部4点满足归并条件时立即执行
        self.work_seq = []           # 当前工作拐点序列 [(bar, price, dir), ...]
        self.segments = []           # 所有产生的线段 [{...}]
        self.seg_set = set()         # 去重集合 (bar_start, bar_end)
        self.last_new_segments = []  # 上次step产生的新线段
    
    def _try_amp_merge_tail(self):
        """检查尾部4拐点是否满足幅度归并。如满足，执行并返回True"""
        seq = self.work_seq
        if len(seq) < 4:
            return False
        
        p1, p2, p3, p4 = seq[-4], seq[-3], seq[-2], seq[-1]
        prices = [p1[1], p2[1], p3[1], p4[1]]
        max_idx = prices.index(max(prices))
        min_idx = prices.index(min(prices))
        
        if (max_idx == 0 and min_idx == 3) or (max_idx == 3 and min_idx == 0):
            # 记录被归并的线段 (p1→p4)
            self._record_segment(p1, p4, 'amp')
            # 也记录中间的三段作为extra
            self._record_segment(p1, p2, 'base')
            self._record_segment(p2, p3, 'base')
            self._record_segment(p3, p4, 'base')
            # 删除中间两点
            self.work_seq = seq[:-3] + [seq[-4], seq[-1]]
            # 实际上是: 保留p1和p4, 删除p2和p3
            # 但p1已经在-4位置，所以: seq[:-4] + [p1, p4]
            self.work_seq = list(seq[:-4]) + [p1, p4]
            return True
        return False
    
    def _try_lat_merge_tail(self):
        """检查尾部4拐点是否满足横向归并"""
        seq = self.work_seq
        if len(seq) < 4:
            return False
        
        p1, p2, p3, p4 = seq[-4], seq[-3], seq[-2], seq[-1]
        
        amp_a = abs(p2[1] - p1[1])
        amp_b = abs(p3[1] - p2[1])
        amp_c = abs(p4[1] - p3[1])
        
        merge = False
        
        # 收敛
        if amp_a > amp_b and amp_b > amp_c:
            merge = True
        # 扩张
        elif amp_a < amp_b and amp_b < amp_c:
            merge = True
        # b最长，检查交错
        elif amp_a <= amp_b and amp_b >= amp_c:
            time_span = p4[0] - p1[0]
            if time_span > 0:
                slope = (p4[1] - p1[1]) / time_span
                if p1[2] == 1:
                    merge = slope <= 0
                else:
                    merge = slope >= 0
        else:
            # fallback
            time_span = p4[0] - p1[0]
            if time_span > 0:
                slope = (p4[1] - p1[1]) / time_span
                if p1[2] == 1:
                    merge = slope <= 0
                else:
                    merge = slope >= 0
        
        if merge:
            self._record_segment(p1, p4, 'lat')
            self._record_segment(p1, p2, 'base')
            self._record_segment(p2, p3, 'base')
            self._record_segment(p3, p4, 'base')
            self.work_seq = list(seq[:-4]) + [p1, p4]
            return True
        return False
    
    def _record_segment(self, p1, p2, source, birth_bar=None):
        """记录一条线段（如果不存在）"""
        key = (p1[0], p2[0])
        if key in self.seg_set:
            return
        if p1[0] >= p2[0]:
            return
        
        self.seg_set.add(key)
        seg = {
            'bar_start': p1[0], 'price_start': p1[1], 'dir_start': p1[2],
            'bar_end': p2[0], 'price_end': p2[1], 'dir_end': p2[2],
            'span': p2[0] - p1[0],
            'amplitude': abs(p2[1] - p1[1]),
            'source': source,
            'source_label': source,
            'birth_bar': birth_bar if birth_bar is not None else -1,
            'importance': 0,  # 暂时为0，后续计算
        }
        self.segments.append(seg)
        self.last_new_segments.append(seg)
    
    def add_pivot(self, pivot_dict, current_bar):
        """
        添加一个新确认的拐点，触发增量归并。
        
        Args:
            pivot_dict: {'bar': int, 'price': float, 'dir': int, 'birth_bar': int}
            current_bar: 当前K线index
        
        Returns:
            list of new segments produced
        """
        self.last_new_segments = []
        p = (pivot_dict['bar'], pivot_dict['price'], pivot_dict['dir'])
        self.work_seq.append(p)
        
        # 记录基础线段 (和前一个拐点构成)
        if len(self.work_seq) >= 2:
            self._record_segment(self.work_seq[-2], self.work_seq[-1], 'base', 
                                 birth_bar=current_bar)
        
        # 尝试连锁归并: 先幅度后横向，循环直到无法归并
        max_chain = 50
        for _ in range(max_chain):
            if self._try_amp_merge_tail():
                # 幅度归并成功，标记birth_bar
                if self.last_new_segments:
                    for seg in self.last_new_segments:
                        if seg['birth_bar'] == -1:
                            seg['birth_bar'] = current_bar
                continue
            if self._try_lat_merge_tail():
                if self.last_new_segments:
                    for seg in self.last_new_segments:
                        if seg['birth_bar'] == -1:
                            seg['birth_bar'] = current_bar
                continue
            break
        
        return self.last_new_segments


# =============================================================================
# 3. 动态波段池 — 增量pool_fusion + 对称谱
# =============================================================================

class DynamicPool:
    """
    动态波段池。
    
    当DynamicMerger产出新线段时，触发增量pool_fusion:
    - 只搜索涉及新线段的三波组合
    - 新fusion线段再触发进一步搜索
    - 循环直到无新线段
    
    同时维护对称谱:
    - 新线段加入后，搜索涉及它的新对称对
    """
    
    def __init__(self, min_amp_for_spectra=0.0):
        self.pool = []               # 所有线段
        self.pool_dict = {}          # (bar_start, bar_end) → seg
        self.end_at = defaultdict(list)    # bar → [seg ending here]
        self.start_at = defaultdict(list)  # bar → [seg starting here]
        
        self.spectra = []            # 所有对称谱
        self.spectra_set = set()     # 去重
        
        self.last_new_segments = []  # 上次add产生的新线段(含fusion)
        self.last_new_spectra = []   # 上次add产生的新对称谱
        
        # 对称谱搜索过滤: 只搜索amplitude >= min_amp_for_spectra的线段
        # 这等价于静态版的max_pool_size (按重要性截断)
        self.min_amp_for_spectra = min_amp_for_spectra
        # 索引: 只包含满足过滤条件的线段 (用于对称谱搜索)
        self.spec_end_at = defaultdict(list)
        self.spec_start_at = defaultdict(list)
    
    def _add_to_index(self, seg):
        """将线段加入索引"""
        self.end_at[seg['bar_end']].append(seg)
        self.start_at[seg['bar_start']].append(seg)
        # 对称谱过滤索引
        if seg['amplitude'] >= self.min_amp_for_spectra:
            self.spec_end_at[seg['bar_end']].append(seg)
            self.spec_start_at[seg['bar_start']].append(seg)
    
    def add_segments(self, new_segs, current_bar, compute_spectra=False):
        """
        添加新线段到池中，触发增量fusion。
        
        Args:
            new_segs: 来自DynamicMerger的新线段列表
            current_bar: 当前K线index
            compute_spectra: 是否触发增量对称谱搜索 (默认False，按需调用)
        
        Returns:
            (all_new_segments, all_new_spectra) — 本次添加产生的所有新内容
        """
        self.last_new_segments = []
        self.last_new_spectra = []
        
        # 1. 加入基础线段
        truly_new = []
        for seg in new_segs:
            key = (seg['bar_start'], seg['bar_end'])
            if key not in self.pool_dict:
                self.pool_dict[key] = seg
                self.pool.append(seg)
                self._add_to_index(seg)
                truly_new.append(seg)
                self.last_new_segments.append(seg)
        
        if not truly_new:
            return self.last_new_segments, self.last_new_spectra
        
        # 2. 增量pool_fusion — 只搜索涉及新线段的三波
        pending = list(truly_new)
        max_rounds = 20
        for _ in range(max_rounds):
            new_fusion = self._incremental_fusion(pending, current_bar)
            if not new_fusion:
                break
            pending = new_fusion
        
        # 3. 增量对称谱 (仅当显式请求时)
        if compute_spectra:
            self._incremental_spectra(self.last_new_segments, current_bar)
        
        return self.last_new_segments, self.last_new_spectra
    
    def _incremental_fusion(self, new_segs, current_bar):
        """
        增量fusion: 只搜索涉及new_segs的三波组合。
        
        新线段可以作为三波中的任何一段(A, B, C):
        - 作为A: new_seg(p1→p2) + B(p2→p3) + C(p3→p4) → p1→p4
        - 作为B: A(p1→p2) + new_seg(p2→p3) + C(p3→p4) → p1→p4
        - 作为C: A(p1→p2) + B(p2→p3) + new_seg(p3→p4) → p1→p4
        """
        fusion_results = []
        
        for seg in new_segs:
            p_start = seg['bar_start']
            p_end = seg['bar_end']
            
            # Case 1: new_seg作为A (p1→p2)
            # 需要: B starts at p_end, C starts at B.end
            for seg_B in self.start_at.get(p_end, []):
                p3 = seg_B['bar_end']
                for seg_C in self.start_at.get(p3, []):
                    p4 = seg_C['bar_end']
                    if p_start < p_end < p3 < p4:
                        self._try_add_fusion(p_start, p4, seg, seg_C, 
                                            current_bar, fusion_results)
            
            # Case 2: new_seg作为B (p2→p3)
            # 需要: A ends at p_start, C starts at p_end
            for seg_A in self.end_at.get(p_start, []):
                p1 = seg_A['bar_start']
                for seg_C in self.start_at.get(p_end, []):
                    p4 = seg_C['bar_end']
                    if p1 < p_start < p_end < p4:
                        self._try_add_fusion(p1, p4, seg_A, seg_C,
                                            current_bar, fusion_results)
            
            # Case 3: new_seg作为C (p3→p4)
            # 需要: B ends at p_start, A ends at B.start
            for seg_B in self.end_at.get(p_start, []):
                p2 = seg_B['bar_start']
                for seg_A in self.end_at.get(p2, []):
                    p1 = seg_A['bar_start']
                    if p1 < p2 < p_start < p_end:
                        self._try_add_fusion(p1, p_end, seg_A, seg,
                                            current_bar, fusion_results)
        
        return fusion_results
    
    def _try_add_fusion(self, bar_start, bar_end, seg_first, seg_last, 
                        current_bar, fusion_results):
        """尝试添加一条fusion线段"""
        key = (bar_start, bar_end)
        if key in self.pool_dict:
            return
        
        new_seg = {
            'bar_start': bar_start,
            'price_start': seg_first['price_start'],
            'dir_start': seg_first['dir_start'],
            'bar_end': bar_end,
            'price_end': seg_last['price_end'],
            'dir_end': seg_last['dir_end'],
            'span': bar_end - bar_start,
            'amplitude': abs(seg_last['price_end'] - seg_first['price_start']),
            'source': 'fusion',
            'source_label': f'dyn_F',
            'birth_bar': current_bar,
            'importance': 0,
        }
        
        self.pool_dict[key] = new_seg
        self.pool.append(new_seg)
        self._add_to_index(new_seg)
        self.last_new_segments.append(new_seg)
        fusion_results.append(new_seg)
    
    def _incremental_spectra(self, new_segs, current_bar):
        """
        增量对称谱: 搜索涉及new_segs的新对称对。
        
        使用spec_end_at/spec_start_at过滤索引,只搜索amplitude >= min_amp_for_spectra的线段。
        """
        for seg in new_segs:
            # 如果新线段本身不够大，跳过
            if seg['amplitude'] < self.min_amp_for_spectra:
                continue
            
            p_start = seg['bar_start']
            p_end = seg['bar_end']
            seg_dir = 1 if seg['price_end'] > seg['price_start'] else -1
            
            # === Mirror: seg作为左臂, 在p_end处折返 ===
            for seg_R in self.spec_start_at.get(p_end, []):
                dir_R = 1 if seg_R['price_end'] > seg_R['price_start'] else -1
                if dir_R == seg_dir:
                    continue  # 镜像需要方向相反
                if seg_R['bar_end'] <= p_end:
                    continue
                key = ('M', p_start, p_end, seg_R['bar_end'])
                if key in self.spectra_set:
                    continue
                self.spectra_set.add(key)
                spec = self._make_spectrum(seg, seg_R, 'mirror', 
                                          pivot_bar=p_end, birth_bar=current_bar)
                if spec:
                    self.spectra.append(spec)
                    self.last_new_spectra.append(spec)
            
            # === Mirror: seg作为右臂, 在p_start处折返 ===
            for seg_L in self.spec_end_at.get(p_start, []):
                dir_L = 1 if seg_L['price_end'] > seg_L['price_start'] else -1
                if dir_L == seg_dir:
                    continue
                if seg_L['bar_start'] >= p_start:
                    continue
                key = ('M', seg_L['bar_start'], p_start, p_end)
                if key in self.spectra_set:
                    continue
                self.spectra_set.add(key)
                spec = self._make_spectrum(seg_L, seg, 'mirror',
                                          pivot_bar=p_start, birth_bar=current_bar)
                if spec:
                    self.spectra.append(spec)
                    self.last_new_spectra.append(spec)
            
            # === Center: seg作为左臂, 通过中心段连接右臂 ===
            for center in self.spec_start_at.get(p_end, []):
                dir_c = 1 if center['price_end'] > center['price_start'] else -1
                if dir_c == seg_dir:
                    continue  # 中心方向应与左/右臂相反
                p3 = center['bar_end']
                for seg_R in self.spec_start_at.get(p3, []):
                    dir_R = 1 if seg_R['price_end'] > seg_R['price_start'] else -1
                    if dir_R != seg_dir:
                        continue  # 左右臂同向
                    p4 = seg_R['bar_end']
                    if not (p_start < p_end < p3 < p4):
                        continue
                    key = ('C', p_start, p_end, p3, p4)
                    if key in self.spectra_set:
                        continue
                    self.spectra_set.add(key)
                    spec = self._make_spectrum(seg, seg_R, 'center',
                                             center_seg=center, birth_bar=current_bar)
                    if spec:
                        self.spectra.append(spec)
                        self.last_new_spectra.append(spec)
            
            # === Center: seg作为右臂 ===
            for center in self.spec_end_at.get(p_start, []):
                dir_c = 1 if center['price_end'] > center['price_start'] else -1
                if dir_c == seg_dir:
                    continue
                p2 = center['bar_start']
                for seg_L in self.spec_end_at.get(p2, []):
                    dir_L = 1 if seg_L['price_end'] > seg_L['price_start'] else -1
                    if dir_L != seg_dir:
                        continue
                    p1 = seg_L['bar_start']
                    if not (p1 < p2 < p_start < p_end):
                        continue
                    key = ('C', p1, p2, p_start, p_end)
                    if key in self.spectra_set:
                        continue
                    self.spectra_set.add(key)
                    spec = self._make_spectrum(seg_L, seg, 'center',
                                             center_seg=center, birth_bar=current_bar)
                    if spec:
                        self.spectra.append(spec)
                        self.last_new_spectra.append(spec)
            
            # === Center: seg作为中心段 ===
            for seg_L in self.spec_end_at.get(p_start, []):
                dir_L = 1 if seg_L['price_end'] > seg_L['price_start'] else -1
                if dir_L == seg_dir:
                    continue  # 左臂应和中心相反
                for seg_R in self.spec_start_at.get(p_end, []):
                    dir_R = 1 if seg_R['price_end'] > seg_R['price_start'] else -1
                    if dir_R == seg_dir:
                        continue  # 右臂应和中心相反
                    if dir_L != dir_R:
                        continue  # 左右臂同向
                    p1 = seg_L['bar_start']
                    p4 = seg_R['bar_end']
                    if not (p1 < p_start < p_end < p4):
                        continue
                    key = ('C', p1, p_start, p_end, p4)
                    if key in self.spectra_set:
                        continue
                    self.spectra_set.add(key)
                    spec = self._make_spectrum(seg_L, seg_R, 'center',
                                             center_seg=seg, birth_bar=current_bar)
                    if spec:
                        self.spectra.append(spec)
                        self.last_new_spectra.append(spec)
    
    def _make_spectrum(self, seg_L, seg_R, sym_type, center_seg=None, 
                       pivot_bar=None, birth_bar=-1):
        """构造对称谱向量 (简化版，和静态版compute_symmetry_spectrum兼容)"""
        amp_L = seg_L['amplitude']
        amp_R = seg_R['amplitude']
        time_L = seg_L['span']
        time_R = seg_R['span']
        
        if amp_L < 1e-10 or amp_R < 1e-10:
            return None
        if time_L < 1 or time_R < 1:
            return None
        
        # 简单归一化模长 (用当前池的全局统计)
        # 注意: 动态版中全局统计是随时间变化的，这里用简化版
        slope_L = amp_L / max(time_L, 1)
        slope_R = amp_R / max(time_R, 1)
        
        def _safe_ratio(a, b):
            if b < 1e-10 and a < 1e-10:
                return 1.0, 0.0
            if b < 1e-10:
                return 999.0, math.log(999)
            r = a / b
            return r, math.log(max(r, 1e-10))
        
        amp_ratio, amp_log = _safe_ratio(amp_L, amp_R)
        time_ratio, time_log = _safe_ratio(time_L, time_R)
        slope_ratio, slope_log = _safe_ratio(slope_L, slope_R)
        
        # 模长 (用slope_ratio近似, 避免归一化问题)
        mod_L = math.sqrt(amp_L**2 + time_L**2) 
        mod_R = math.sqrt(amp_R**2 + time_R**2)
        mod_ratio, mod_log = _safe_ratio(mod_L, mod_R)
        
        sym_closeness = 1.0 / (1.0 + abs(amp_log) + abs(time_log) + abs(mod_log) + abs(slope_log))
        
        dir_L = 1 if seg_L['price_end'] > seg_L['price_start'] else -1
        dir_R = 1 if seg_R['price_end'] > seg_R['price_start'] else -1
        
        center_amp = center_seg['amplitude'] if center_seg else 0
        center_span = center_seg['span'] if center_seg else 0
        center_bar_start = center_seg['bar_start'] if center_seg else 0
        center_bar_end = center_seg['bar_end'] if center_seg else 0
        
        return {
            'type': sym_type,
            'dir_L': dir_L,
            'dir_R': dir_R,
            'L_start': seg_L['bar_start'], 'L_end': seg_L['bar_end'],
            'L_price_start': seg_L['price_start'], 'L_price_end': seg_L['price_end'],
            'L_amp': round(amp_L, 5), 'L_time': time_L,
            'R_start': seg_R['bar_start'], 'R_end': seg_R['bar_end'],
            'R_price_start': seg_R['price_start'], 'R_price_end': seg_R['price_end'],
            'R_amp': round(amp_R, 5), 'R_time': time_R,
            'amp_ratio': round(amp_ratio, 4),
            'time_ratio': round(time_ratio, 4),
            'mod_ratio': round(mod_ratio, 4),
            'slope_ratio': round(slope_ratio, 4),
            'amp_log': round(amp_log, 4),
            'time_log': round(time_log, 4),
            'mod_log': round(mod_log, 4),
            'slope_log': round(slope_log, 4),
            'center_bar_start': center_bar_start,
            'center_bar_end': center_bar_end,
            'center_amp': round(center_amp, 5),
            'center_span': center_span,
            'pivot_bar': pivot_bar if pivot_bar else 0,
            'sym_closeness': round(sym_closeness, 4),
            'birth_bar': birth_bar,
        }


# =============================================================================
# 4. 动态引擎 — 整合
# =============================================================================

class DynamicEngine:
    """
    完整动态引擎。
    
    用法:
        engine = DynamicEngine()
        for i in range(len(high)):
            snapshot = engine.step(high[i], low[i])
            # snapshot包含当前时刻的所有信息
    """
    
    def __init__(self, rb=0.5, min_amp_for_spectra=0.0):
        self.zg = DynamicZG(rb=rb)
        self.merger = DynamicMerger()
        self.pool = DynamicPool(min_amp_for_spectra=min_amp_for_spectra)
        self.bar_count = 0
        self.min_amp_for_spectra = min_amp_for_spectra
        
        # 历史记录: 每根K线的快照摘要
        self.history = []  # [{bar, n_confirmed_pivots, n_segments, n_spectra, events}]
    
    def step(self, h, l):
        """
        送入一根新K线。
        
        返回: snapshot dict {
            'bar': int,
            'zg_events': list,
            'new_segments': list,
            'n_confirmed_pivots': int,
            'n_total_segments': int,
            'has_new_pivot': bool,
        }
        """
        current_bar = self.bar_count
        self.bar_count += 1
        
        # 1. ZG更新
        zg_events = self.zg.step(h, l)
        
        # 2. 对每个新确认的拐点，触发归并
        all_new_segments = []
        has_new_pivot = False
        for event in zg_events:
            if event['type'] == 'confirmed':
                has_new_pivot = True
                new_segs = self.merger.add_pivot(event['pivot'], current_bar)
                all_new_segments.extend(new_segs)
        
        # 3. 新线段加入池，触发fusion (不触发对称谱 — 按需计算)
        if all_new_segments:
            new_segs, _ = self.pool.add_segments(all_new_segments, current_bar)
            all_new_segments = new_segs  # 包含fusion产出的
        
        snapshot = {
            'bar': current_bar,
            'zg_events': zg_events,
            'new_segments': all_new_segments,
            'n_confirmed_pivots': len(self.zg.pivots),
            'n_tentative': 1 if self.zg.tentative else 0,
            'n_total_segments': len(self.pool.pool),
            'has_new_pivot': has_new_pivot,
        }
        
        self.history.append(snapshot)
        return snapshot
    
    def compute_spectra_now(self, max_pool_size=800):
        """
        在当前池上计算对称谱（按需调用）。
        
        这是一个"即时快照" — 用当前池中重要性最高的线段计算。
        不累积，每次调用都是独立的。
        
        和静态版compute_symmetry_spectrum()逻辑相同，但只用已birth的线段。
        """
        from merge_engine_v3 import compute_symmetry_spectrum
        
        # 构建临时pivot_info (简化版: 只用bar→{importance: amplitude})
        # 动态版没有完整的8维importance，用amplitude*span近似
        pivot_info = {}
        for seg in self.pool.pool:
            for bar, price, d in [(seg['bar_start'], seg['price_start'], seg['dir_start']),
                                   (seg['bar_end'], seg['price_end'], seg['dir_end'])]:
                if bar not in pivot_info:
                    pivot_info[bar] = {
                        'bar': bar, 'price': price, 'dir': d,
                        'importance': 0,
                    }
                # 累加经过该点的线段amplitude作为简化importance
                pivot_info[bar]['importance'] = max(
                    pivot_info[bar]['importance'],
                    seg['amplitude'] * seg['span']
                )
        
        # 归一化importance到0~1
        max_imp = max((p['importance'] for p in pivot_info.values()), default=1)
        if max_imp > 0:
            for p in pivot_info.values():
                p['importance'] = p['importance'] / max_imp
        
        spectra = compute_symmetry_spectrum(self.pool.pool, pivot_info, 
                                           max_pool_size=max_pool_size)
        return spectra
    
    def run_all(self, high, low, verbose=True):
        """
        批量运行所有K线。
        
        Args:
            high: array of high prices
            low: array of low prices
            verbose: 是否打印进度
        
        Returns:
            list of snapshots
        """
        n = len(high)
        t0 = _time.time()
        
        for i in range(n):
            self.step(high[i], low[i])
            
            if verbose and (i+1) % 50 == 0:
                snap = self.history[-1]
                elapsed = _time.time() - t0
                print(f"  bar {i+1:5d}/{n}: pivots={snap['n_confirmed_pivots']:3d} "
                      f"segs={snap['n_total_segments']:5d} "
                      f"spectra={snap['n_total_spectra']:6d} "
                      f"({elapsed:.1f}s)")
        
        if verbose:
            elapsed = _time.time() - t0
            final = self.history[-1]
            print(f"\n完成: {n} bars, {elapsed:.2f}s")
            print(f"  确认拐点: {final['n_confirmed_pivots']}")
            print(f"  总线段: {final['n_total_segments']}")
            print(f"  总对称谱: {final['n_total_spectra']}")
        
        return self.history


# =============================================================================
# 5. 验证: 动态 vs 静态
# =============================================================================

def validate_dynamic_vs_static(filepath, limit=200):
    """
    在同一组数据上运行动态引擎和静态引擎，对比结果。
    
    期望: 
    - 动态引擎的确认拐点序列 ≈ 静态引擎的基础ZG拐点序列
      (不完全相同，因为静态版有backward pass，动态版没有)
    - 动态引擎的最终线段池应是静态版的子集
      (因为动态版只用确认时能看到的信息)
    """
    from merge_engine_v3 import (calculate_base_zg, full_merge_engine, 
                                  build_segment_pool, compute_pivot_importance,
                                  pool_fusion, compute_symmetry_spectrum)
    
    print("=" * 70)
    print(f"验证: 动态 vs 静态 ({filepath}, limit={limit})")
    print("=" * 70)
    
    df = load_kline(filepath, limit=limit)
    high = df['high'].values
    low = df['low'].values
    print(f"K线: {len(df)}, {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    
    # === 静态引擎 ===
    print("\n--- 静态引擎 ---")
    t0 = _time.time()
    static_pivots = calculate_base_zg(high, low)
    results = full_merge_engine(static_pivots)
    pivot_info = compute_pivot_importance(results, total_bars=len(df))
    pool = build_segment_pool(results, pivot_info)
    full_pool, _, _ = pool_fusion(pool, pivot_info)
    spectra = compute_symmetry_spectrum(full_pool, pivot_info)
    t1 = _time.time()
    
    print(f"  拐点: {len(static_pivots)}")
    print(f"  线段: {len(full_pool)}")
    print(f"  对称谱: {len(spectra)}")
    print(f"  耗时: {t1-t0:.3f}s")
    
    # === 动态引擎 ===
    print("\n--- 动态引擎 ---")
    # min_amp_for_spectra: 过滤小幅度线段，避免对称谱爆炸
    # 静态版用max_pool_size=800截断(2970条取top800)
    # 经验: 全局范围的5%大约过滤掉60-70%的微小线段
    price_range = high.max() - low.min()
    min_amp = price_range * 0.05  # 全局范围的5%
    print(f"  min_amp_for_spectra: {min_amp:.5f} ({price_range:.5f} * 5%)")
    
    t0 = _time.time()
    engine = DynamicEngine(min_amp_for_spectra=min_amp)
    engine.run_all(high, low, verbose=True)
    t1 = _time.time()
    
    dyn_pivots = engine.zg.get_confirmed_pivots()
    dyn_all_pivots = engine.zg.get_all_pivots()
    dyn_segs = engine.pool.pool
    dyn_spectra = engine.pool.spectra
    
    print(f"\n  确认拐点: {len(dyn_pivots)}")
    print(f"  拐点(含临时): {len(dyn_all_pivots)}")
    print(f"  线段: {len(dyn_segs)}")
    print(f"  对称谱: {len(dyn_spectra)}")
    print(f"  耗时: {t1-t0:.3f}s")
    
    # === 对比 ===
    print("\n--- 对比 ---")
    
    # 拐点对比
    static_bars = [(p[0], p[2]) for p in static_pivots]
    dyn_bars = [(p['bar'], p['dir']) for p in dyn_pivots]
    
    # 找交集
    static_set = set(static_bars)
    dyn_set = set(dyn_bars)
    common = static_set & dyn_set
    only_static = static_set - dyn_set
    only_dyn = dyn_set - static_set
    
    print(f"\n拐点对比:")
    print(f"  静态: {len(static_bars)}, 动态确认: {len(dyn_bars)}")
    print(f"  共有: {len(common)}")
    print(f"  仅静态: {len(only_static)}")
    if only_static:
        for b, d in sorted(only_static):
            print(f"    bar={b} dir={'+' if d==1 else '-'}")
    print(f"  仅动态: {len(only_dyn)}")
    if only_dyn:
        for b, d in sorted(only_dyn):
            print(f"    bar={b} dir={'+' if d==1 else '-'}")
    
    # 线段对比
    static_seg_keys = {(s['bar_start'], s['bar_end']) for s in full_pool}
    dyn_seg_keys = {(s['bar_start'], s['bar_end']) for s in dyn_segs}
    common_segs = static_seg_keys & dyn_seg_keys
    only_static_segs = static_seg_keys - dyn_seg_keys
    only_dyn_segs = dyn_seg_keys - static_seg_keys
    
    print(f"\n线段对比:")
    print(f"  静态: {len(static_seg_keys)}, 动态: {len(dyn_seg_keys)}")
    print(f"  共有: {len(common_segs)}")
    print(f"  仅静态: {len(only_static_segs)}")
    print(f"  仅动态: {len(only_dyn_segs)}")
    
    # 动态引擎的birth_bar分布
    print(f"\n动态引擎birth_bar信息:")
    confirmed = engine.zg.get_confirmed_pivots()
    if confirmed:
        birth_bars = [p['birth_bar'] for p in confirmed]
        print(f"  拐点birth_bar: min={min(birth_bars)} max={max(birth_bars)} "
              f"mean={sum(birth_bars)/len(birth_bars):.1f}")
    
    seg_births = [s['birth_bar'] for s in dyn_segs if s.get('birth_bar', -1) >= 0]
    if seg_births:
        print(f"  线段birth_bar: min={min(seg_births)} max={max(seg_births)} "
              f"mean={sum(seg_births)/len(seg_births):.1f}")
    
    spec_births = [s['birth_bar'] for s in dyn_spectra if s.get('birth_bar', -1) >= 0]
    if spec_births:
        print(f"  对称谱birth_bar: min={min(spec_births)} max={max(spec_births)} "
              f"mean={sum(spec_births)/len(spec_births):.1f}")
    
    # 打印前20个确认拐点(动态)，显示birth_bar
    print(f"\n动态引擎 — 前20个确认拐点:")
    print(f"  {'bar':>4s} {'price':>9s} {'dir':>3s} {'birth':>5s} {'delay':>5s}")
    for p in confirmed[:20]:
        delay = p['birth_bar'] - p['bar']
        dir_s = 'H' if p['dir'] == 1 else 'L'
        print(f"  {p['bar']:4d} {p['price']:.5f} {dir_s:>3s} {p['birth_bar']:5d} {delay:+5d}")
    
    return engine


# =============================================================================
# 主程序
# =============================================================================

def performance_test(filepath, limit=200):
    """性能测试: 无对称谱，只跑ZG+归并+fusion"""
    print("=" * 70)
    print(f"性能测试: 只跑ZG+归并+fusion (无对称谱)")
    print("=" * 70)
    
    df = load_kline(filepath, limit=limit)
    high = df['high'].values
    low = df['low'].values
    
    # 无对称谱 — 设置一个极高阈值
    engine = DynamicEngine(min_amp_for_spectra=9999.0)
    t0 = _time.time()
    engine.run_all(high, low, verbose=True)
    t1 = _time.time()
    
    print(f"\n纯ZG+归并+fusion耗时: {t1-t0:.3f}s")
    print(f"线段: {len(engine.pool.pool)}, 对称谱: {len(engine.pool.spectra)}")
    
    return engine


if __name__ == '__main__':
    filepath = "/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv"
    
    # 性能测试 — 只跑核心部分
    performance_test(filepath, limit=200)
    print()
    
    # 完整验证
    engine = validate_dynamic_vs_static(filepath, limit=200)
