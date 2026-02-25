#!/usr/bin/env python3
"""
完整归并引擎 v1.0
- 基础ZG (KZig风格)
- 幅度归并（多轮贪心）
- 横向归并（类型1/2/3连接，类型4不连）
- 分层交替迭代直到不动点
"""

import numpy as np
import pandas as pd
import sys
import time

# =============================================================================
# 1. 数据加载
# =============================================================================

def load_kline(filepath, limit=None):
    """加载K线数据，返回DataFrame (time, open, high, low, close)"""
    df = pd.read_csv(filepath, sep='\t', 
                     names=['date','time','open','high','low','close','tickvol','vol','spread'],
                     skiprows=1)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df[['datetime','open','high','low','close']].reset_index(drop=True)
    if limit:
        df = df.tail(limit).reset_index(drop=True)
    return df

# =============================================================================
# 2. 基础ZG计算 (KZig风格，移植自MQL5)
# =============================================================================

def calculate_base_zg(high, low, rb=0.5):
    """
    计算基础zigzag拐点。移植自MQL5的CalculateBaseZG。
    
    返回: list of (bar_index, price, direction)
      direction: 1=高点(峰), -1=低点(谷)
    """
    n = len(high)
    if n < 3:
        return []
    
    # goddsZg数组 — 存储拐点价格，其他位置为None
    zg = [None] * n
    
    last_state = 1  # 1=上, -1=下
    last_pos = 0    # 当前拐点位置（从最老的开始）
    p_pos = 0       # 前一个拐点位置
    last_range = 9999.0
    
    # 第一遍：从左到右（从老到新）扫描
    for i in range(1, n):
        is_up = False
        is_down = False
        
        # 上行判定（3个条件之一）
        if ((high[i] > high[i-1] and low[i] >= low[i-1]) or
            (last_state == -1 and high[i] - low[last_pos] > last_range * rb and 
             not (high[i] < high[i-1] and low[i] < low[i-1])) or
            (last_state == -1 and i - last_pos > 1 and high[i] > low[last_pos] and
             not (high[i] < high[i-1] and low[i] < low[i-1]))):
            is_up = True
        
        # 下行判定（3个条件之一）
        if ((high[i] <= high[i-1] and low[i] < low[i-1]) or
            (last_state == 1 and high[last_pos] - low[i] > last_range * rb and
             not (high[i] > high[i-1] and low[i] > low[i-1])) or
            (last_state == 1 and i - last_pos > 1 and low[i] < high[last_pos] and
             not (high[i] > high[i-1] and low[i] > low[i-1]))):
            is_down = True
        
        if is_up:
            if last_state == 1:
                # 同方向延伸：新高必须更高
                if high[i] < high[last_pos]:
                    continue
                zg[last_pos] = None  # 清除旧拐点
            else:
                p_pos = last_pos
            zg[i] = high[i]
            last_pos = i
            last_state = 1
            last_range = high[i] - low[p_pos]
        elif is_down:
            if last_state == -1:
                # 同方向延伸：新低必须更低
                if low[i] > low[last_pos]:
                    continue
                zg[last_pos] = None
            else:
                p_pos = last_pos
            zg[i] = low[i]
            last_pos = i
            last_state = -1
            last_range = high[p_pos] - low[i]
    
    # 第二遍：从左到右修正方向交替一致性
    # (MQL5原版是从右到左，但因为我们数组方向相反，这里从左到右)
    pre_price = zg[last_pos]
    pre_pos = last_pos
    fix_state = last_state
    
    # 找到最新的拐点，然后向左（更老方向）修正
    # 实际上MQL5的第二遍是从lastPos向更老的方向扫描
    # 因为AS_SERIES=true, lastPos+1 = 更老
    # 我们的数组是正序（0=最老），所以从last_pos向0扫描
    for i in range(last_pos - 1, -1, -1):
        if zg[i] is not None:
            if fix_state == -1:
                fix_state = 1
                if zg[i] < pre_price:
                    zg[i] = None
                    zg[pre_pos] = pre_price
                else:
                    pre_price = zg[i]
                    pre_pos = i
            elif fix_state == 1:
                fix_state = -1
                if zg[i] > pre_price:
                    zg[i] = None
                    zg[pre_pos] = pre_price
                else:
                    pre_price = zg[i]
                    pre_pos = i
        else:
            if fix_state == 1 and low[i] < pre_price:
                pre_price = low[i]
                pre_pos = i
            elif fix_state == -1 and high[i] > pre_price:
                pre_price = high[i]
                pre_pos = i
    
    # 收集拐点
    pivots = []
    for i in range(n):
        if zg[i] is not None:
            # 判断方向：价格等于high[i]则为峰(1)，等于low[i]则为谷(-1)
            if abs(zg[i] - high[i]) < 1e-10:
                pivots.append((i, zg[i], 1))
            elif abs(zg[i] - low[i]) < 1e-10:
                pivots.append((i, zg[i], -1))
            else:
                # 不确定，跳过
                pass
    
    # 确保方向交替
    cleaned = []
    for p in pivots:
        if not cleaned or cleaned[-1][2] != p[2]:
            cleaned.append(p)
        else:
            # 同方向：保留更极端的
            if p[2] == 1 and p[1] > cleaned[-1][1]:
                cleaned[-1] = p
            elif p[2] == -1 and p[1] < cleaned[-1][1]:
                cleaned[-1] = p
    
    return cleaned

# =============================================================================
# 3. 幅度归并
# =============================================================================

def amplitude_merge_one_pass(pivots):
    """
    单轮幅度归并。
    
    规则：相邻4个拐点(P1,P2,P3,P4)构成3个波段。
    交替序列: P1(dir_a), P2(dir_b), P3(dir_a), P4(dir_b)
    
    归并条件（MQL5逻辑）：P1和P4分别是4个点中的极值。
    等价于：P4超过P2(同方向) AND P1超过P3(同方向)。
    即：第三段延续了第一段的趋势，中间的回撤被吞没。
    
    合并：删除P2和P3，保留P1和P4。
    
    返回: (新拐点列表, 是否有变化)
    """
    if len(pivots) < 4:
        return pivots, False
    
    result = []
    changed = False
    i = 0
    
    while i < len(pivots):
        if i + 3 < len(pivots):
            p1 = pivots[i]      # (bar_idx, price, dir)
            p2 = pivots[i+1]
            p3 = pivots[i+2]    # 同方向于P1
            p4 = pivots[i+3]    # 同方向于P2
            
            # 找4个点中的max和min
            prices = [p1[1], p2[1], p3[1], p4[1]]
            max_price = max(prices)
            min_price = min(prices)
            max_idx = prices.index(max_price)  # 0=P1, 1=P2, 2=P3, 3=P4
            min_idx = prices.index(min_price)
            
            # MQL5条件：首尾分别是极值
            # 情况1: P1是最大, P4是最小 (下降趋势吞没)
            # 情况2: P1是最小, P4是最大 (上升趋势吞没)
            merge = False
            if (max_idx == 0 and min_idx == 3) or (max_idx == 3 and min_idx == 0):
                merge = True
            
            if merge:
                # 删除P2和P3，保留P1，P4在下次循环处理
                result.append(p1)
                i += 3
                changed = True
                continue
        
        result.append(pivots[i])
        i += 1
    
    return result, changed

def amplitude_merge_full(pivots):
    """多轮幅度归并直到无变化"""
    current = pivots
    total_rounds = 0
    while True:
        new_pivots, changed = amplitude_merge_one_pass(current)
        total_rounds += 1
        if not changed:
            break
        current = new_pivots
    return current, total_rounds

# =============================================================================
# 4. 横向归并
# =============================================================================

def classify_three_segments(pivots, i):
    """
    分类相邻三段(a,b,c)的几何结构。
    pivots[i], pivots[i+1], pivots[i+2], pivots[i+3] = P1,P2,P3,P4
    a = P1→P2, b = P2→P3, c = P3→P4
    
    前提：已经过幅度归并，所以P4不超过P1。
    
    返回: ('converging'|'expanding'|'crossover'|'no_crossover', 详情dict)
    """
    if i + 3 >= len(pivots):
        return None, {}
    
    p1_idx, p1_price, p1_dir = pivots[i]
    p2_idx, p2_price, p2_dir = pivots[i+1]
    p3_idx, p3_price, p3_dir = pivots[i+2]
    p4_idx, p4_price, p4_dir = pivots[i+3]
    
    amp_a = abs(p2_price - p1_price)
    amp_b = abs(p3_price - p2_price)
    amp_c = abs(p4_price - p3_price)
    
    # 类型1：收敛 |a| > |b| > |c|
    if amp_a > amp_b and amp_b > amp_c:
        return 'converging', {'amp_a': amp_a, 'amp_b': amp_b, 'amp_c': amp_c}
    
    # 类型2：扩张 |a| < |b| < |c|
    if amp_a < amp_b and amp_b < amp_c:
        return 'expanding', {'amp_a': amp_a, 'amp_b': amp_b, 'amp_c': amp_c}
    
    # b最长的情况：|a| < |b| 且 |b| > |c|
    if amp_a <= amp_b and amp_b >= amp_c:
        # 检查ac是否有交错
        # 连接a起点(P1)到c终点(P4)
        # 峰→谷线（P1高P4低）：角度 <= 0 则有交错
        # 谷→峰线（P1低P4高）：角度 >= 0 则有交错
        
        time_span = p4_idx - p1_idx
        if time_span == 0:
            return 'no_crossover', {}
        
        price_diff = p4_price - p1_price  # P4 - P1
        slope = price_diff / time_span
        
        if p1_dir == 1:
            # P1是峰（高点），P4是谷（低点）→ 峰→谷线
            # 角度 <= 0（下降或水平）→ 有交错
            if slope <= 0:
                return 'crossover', {'slope': slope}
            else:
                return 'no_crossover', {'slope': slope}
        else:
            # P1是谷（低点），P4是峰（高点）→ 谷→峰线
            # 角度 >= 0（上升或水平）→ 有交错
            if slope >= 0:
                return 'crossover', {'slope': slope}
            else:
                return 'no_crossover', {'slope': slope}
    
    # 未覆盖的情况：a最长(|a|>|b|, |a|>|c|, 但非收敛)，或其他边界
    # 这些情况下，检查ac是否有交错，有则连，无则不连
    time_span = p4_idx - p1_idx
    if time_span == 0:
        return 'no_crossover', {}
    price_diff = p4_price - p1_price
    slope = price_diff / time_span
    
    if p1_dir == 1:
        if slope <= 0:
            return 'crossover', {'slope': slope, 'note': 'fallback'}
        else:
            return 'no_crossover', {'slope': slope, 'note': 'fallback'}
    else:
        if slope >= 0:
            return 'crossover', {'slope': slope, 'note': 'fallback'}
        else:
            return 'no_crossover', {'slope': slope, 'note': 'fallback'}

def lateral_merge_one_pass(pivots):
    """
    单轮横向归并。
    
    扫描所有相邻三段组，对类型1/2/3进行连接（删除中间两个拐点）。
    类型4不连接。
    
    返回: (新拐点列表, 是否有变化)
    """
    if len(pivots) < 4:
        return pivots, False
    
    result = []
    changed = False
    i = 0
    
    while i < len(pivots):
        if i + 3 < len(pivots):
            seg_type, info = classify_three_segments(pivots, i)
            
            if seg_type in ('converging', 'expanding', 'crossover'):
                # 连接：保留P1，删除P2和P3，P4将在下次处理
                result.append(pivots[i])
                i += 3  # 跳过P2和P3
                changed = True
                continue
        
        result.append(pivots[i])
        i += 1
    
    return result, changed

def lateral_merge_full(pivots):
    """多轮横向归并直到无变化"""
    current = pivots
    total_rounds = 0
    while True:
        new_pivots, changed = lateral_merge_one_pass(current)
        total_rounds += 1
        if not changed:
            break
        current = new_pivots
    return current, total_rounds

# =============================================================================
# 5. 完整归并引擎：分层交替迭代
# =============================================================================

def full_merge_engine(pivots, max_iterations=50):
    """
    完整归并引擎。
    
    分层交替迭代：
    while(有变化) {
        幅度归并穷尽;
        横向归并一轮;
    }
    
    返回: dict，包含每个阶段的中间结果
    """
    results = {
        'base_pivots': len(pivots),
        'iterations': [],
        'levels': [pivots.copy()],  # level 0 = 基础ZG
    }
    
    current = pivots.copy()
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        iter_info = {'iteration': iteration}
        
        # 阶段1：幅度归并穷尽
        before_amp = len(current)
        current, amp_rounds = amplitude_merge_full(current)
        after_amp = len(current)
        iter_info['amp_before'] = before_amp
        iter_info['amp_after'] = after_amp
        iter_info['amp_rounds'] = amp_rounds
        iter_info['amp_removed'] = before_amp - after_amp
        
        if before_amp != after_amp:
            results['levels'].append(current.copy())
        
        # 阶段2：横向归并一轮（不是穷尽——每轮让权给幅度归并检查）
        before_lat = len(current)
        current, lat_changed = lateral_merge_one_pass(current)
        after_lat = len(current)
        iter_info['lat_before'] = before_lat
        iter_info['lat_after'] = after_lat
        iter_info['lat_removed'] = before_lat - after_lat
        
        results['iterations'].append(iter_info)
        
        if before_lat != after_lat:
            results['levels'].append(current.copy())
        
        # 检查是否达到不动点
        total_change = (before_amp - after_amp) + (before_lat - after_lat)
        if total_change == 0:
            break
    
    results['final_pivots'] = len(current)
    results['total_iterations'] = iteration
    results['final'] = current
    
    return results

# =============================================================================
# 6. 主程序
# =============================================================================

def main():
    print("=" * 70)
    print("完整归并引擎 v1.0")
    print("=" * 70)
    
    # 加载数据
    filepath = "/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv"
    limit = 2000  # 先用2000根K线测试
    
    print(f"\n加载数据: {filepath} (最后{limit}根)")
    df = load_kline(filepath, limit=limit)
    print(f"数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"K线数量: {len(df)}")
    
    high = df['high'].values
    low = df['low'].values
    
    # 计算基础ZG
    t0 = time.time()
    base_pivots = calculate_base_zg(high, low)
    t1 = time.time()
    print(f"\n基础ZG: {len(base_pivots)} 个拐点 ({t1-t0:.3f}s)")
    print(f"平均 {len(df)/max(len(base_pivots),1):.2f} K线/拐点")
    
    # 展示前10个拐点
    print("\n前10个拐点:")
    for idx, (bar, price, direction) in enumerate(base_pivots[:10]):
        dir_str = "峰" if direction == 1 else "谷"
        print(f"  [{idx}] bar={bar}, price={price:.5f}, {dir_str}, time={df['datetime'].iloc[bar]}")
    
    # 运行完整归并引擎
    print("\n" + "=" * 70)
    print("运行完整归并引擎...")
    print("=" * 70)
    
    t0 = time.time()
    results = full_merge_engine(base_pivots)
    t1 = time.time()
    
    print(f"\n完成! 耗时: {t1-t0:.3f}s")
    print(f"基础拐点: {results['base_pivots']}")
    print(f"最终拐点: {results['final_pivots']}")
    print(f"总迭代轮次: {results['total_iterations']}")
    print(f"产生级别数: {len(results['levels'])}")
    
    print("\n各迭代详情:")
    for it in results['iterations']:
        print(f"  迭代{it['iteration']}: "
              f"幅度归并 {it['amp_before']}→{it['amp_after']} (去除{it['amp_removed']}, {it['amp_rounds']}轮), "
              f"横向归并 {it['lat_before']}→{it['lat_after']} (去除{it['lat_removed']})")
    
    print("\n各级别拐点数:")
    for i, level in enumerate(results['levels']):
        print(f"  Level {i}: {len(level)} 个拐点")
    
    # 最终拐点
    final = results['final']
    print(f"\n最终拐点 (共{len(final)}个):")
    for idx, (bar, price, direction) in enumerate(final[:20]):
        dir_str = "峰" if direction == 1 else "谷"
        print(f"  [{idx}] bar={bar}, price={price:.5f}, {dir_str}, time={df['datetime'].iloc[bar]}")
    if len(final) > 20:
        print(f"  ... 还有 {len(final)-20} 个")

if __name__ == '__main__':
    main()
