#!/usr/bin/env python3
"""
v32 Server — 可调窗口 + 实时归并计算 + 逐K步进

启动: python3 visualize_v32_server.py [--bars 5000] [--port 8765]
访问: http://localhost:8765

架构: FastAPI + WebSocket
  前端每次操作(步进/窗口调整/toggle) → WebSocket请求
  → 服务器实时计算 base_zg → full_merge_engine (+ 可选 fusion/predict)
  → 返回 JSON → 前端重绘

TODO: 后续集成 FSD 系统
"""

import sys
import json
import time
import math
import argparse

sys.path.insert(0, '/home/ubuntu/stage2_abc')

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn
import numpy as np

from merge_engine_v3 import (
    load_kline, calculate_base_zg, full_merge_engine,
    compute_pivot_importance, build_segment_pool,
    pool_fusion, predict_symmetric_image, find_symmetric_structures
)
from fsd_engine import prune_redundant

app = FastAPI()

# 全局数据 (启动时加载)
G = {}


@app.on_event("startup")
def startup():
    bars = getattr(app, '_bars', 5000)
    print(f"Loading EURUSD H1 data (limit={bars})...")
    df = load_kline("/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv", limit=bars)
    G['df'] = df
    G['highs'] = df['high'].values.astype(float)
    G['lows'] = df['low'].values.astype(float)
    G['opens'] = df['open'].values.astype(float)
    G['closes'] = df['close'].values.astype(float)
    G['n'] = len(df)
    G['datetimes'] = [str(d) for d in df['datetime'].values]
    print(f"Loaded: {G['n']} bars | {G['datetimes'][0]} ~ {G['datetimes'][-1]}")


@app.get("/")
def index():
    return FileResponse("/home/ubuntu/stage2_abc/visualize_v32.html",
                        media_type="text/html")


def compute_window(end_bar, window, show_lat=False, show_fusion=False,
                   show_pred=False, show_pvt=False, show_sym=False,
                   show_extra=False, min_imp=0.0, peak_n=10, valley_n=10):
    """
    对指定窗口执行完整v3 pipeline，返回前端需要的数据。
    
    始终计算: base_zg → full_merge_engine → snapshots
    可选计算: importance → pool → fusion → prune → predict → symmetry
    """
    t0 = time.time()
    
    # 边界
    window = max(10, min(window, 1000, G['n']))
    end_bar = max(window, min(end_bar, G['n']))
    start = end_bar - window
    
    hw = G['highs'][start:end_bar]
    lw = G['lows'][start:end_bar]
    
    # --- 基础计算 (每次都做) ---
    base = calculate_base_zg(hw, lw, rb=0.5)
    results = full_merge_engine(base)
    
    # 提取快照: L0 + A1-A6 (始终), T1-T2 (可选)
    snapshots = []
    for snap_type, label, pvts in results['all_snapshots']:
        if snap_type == 'base' or snap_type == 'amp':
            points = [[int(p[0]) + start, round(float(p[1]), 5), int(p[2])]
                      for p in pvts]
            snapshots.append({'type': snap_type, 'label': label, 'pts': points})
        elif snap_type == 'lat' and show_lat:
            points = [[int(p[0]) + start, round(float(p[1]), 5), int(p[2])]
                      for p in pvts]
            snapshots.append({'type': snap_type, 'label': label, 'pts': points})
    
    # K线
    klines = []
    for i in range(start, end_bar):
        klines.append([
            round(float(G['opens'][i]), 5),
            round(float(G['highs'][i]), 5),
            round(float(G['lows'][i]), 5),
            round(float(G['closes'][i]), 5),
        ])
    
    # --- Extra segments (greedy-skipped) ---
    extra_data = None
    if show_extra:
        extra_segs = results.get('extra_segments', [])
        extra_by_label = {}
        for p_start, p_end, label in extra_segs:
            if label not in extra_by_label:
                extra_by_label[label] = []
            extra_by_label[label].append({
                'b1': int(p_start[0]) + start, 'p1': round(float(p_start[1]), 5),
                'b2': int(p_end[0]) + start,   'p2': round(float(p_end[1]), 5),
            })
        extra_data = []
        for label in sorted(extra_by_label.keys()):
            segs = extra_by_label[label]
            src = 'amp' if (label.startswith('A') and '_mid' not in label) else 'lat'
            extra_data.append({'label': label, 'src': src, 'segs': segs})
    
    # --- 可选计算 ---
    pool_data = None
    pred_data = None
    pivot_data = None
    sym_data = None
    
    need_pi = show_fusion or show_pred or show_pvt or show_sym
    pi = None
    pruned = None
    
    if need_pi:
        pi = compute_pivot_importance(results, window)
    
    if show_fusion or show_pred or show_sym:
        pool = build_segment_pool(results, pi)
        fused_all, _, _ = pool_fusion(pool, pi)
        pruned = prune_redundant(fused_all, pi, merge_dist=3,
                                 max_per_start=5, max_total=300)
        
        if show_fusion:
            # 全量显示 (不截断，时间轴拉大后自然稀疏)
            pool_data = []
            for seg in pruned:
                if min_imp > 0 and seg.get('importance', 0) < min_imp:
                    continue
                dr = 1 if seg['price_end'] > seg['price_start'] else -1
                pool_data.append({
                    'bs': seg['bar_start'] + start,
                    'be': seg['bar_end'] + start,
                    'ps': round(seg['price_start'], 5),
                    'pe': round(seg['price_end'], 5),
                    'src': seg['source'][0],  # b/a/l/f/e
                    'imp': round(seg.get('importance', 0), 5),
                    'dr': dr,
                })
        
        if show_pred:
            preds = predict_symmetric_image(pruned, pi, current_bar=window - 1,
                                            max_pool_size=99999)
            # 只保留短程预测
            short_preds = [p for p in preds if p['pred_time'] <= 50]
            short_preds.sort(key=lambda p: -p['score'])
            
            # 归一化参数 (用于模长弧线计算)
            # global_amp = 窗口内的价格极差, global_span = 窗口长度
            global_amp = float(hw.max() - lw.min())
            global_span = float(window)
            
            pred_data = []
            for p in short_preds:  # 全量(不截断)
                # 归一化模长 R = sqrt((amp/global_amp)² + (time/global_span)²)
                norm_amp = p['A_amp'] / max(global_amp, 1e-10)
                norm_time = p['A_time'] / max(global_span, 1)
                mod_R = math.sqrt(norm_amp**2 + norm_time**2)
                
                # 类型编码: m=mirror, c=center, t=triangle, o=modonly
                tp_code = {'mirror': 'm', 'center': 'c',
                           'triangle': 't', 'modonly': 'o'}.get(p['type'], p['type'][0])
                
                pg = {
                    'tp': tp_code,
                    'pd': p['pred_dir'],
                    'as': p['A_start'] + start,
                    'ae': p['A_end'] + start,
                    'aps': round(p['A_price_start'], 5),
                    'ape': round(p['A_price_end'], 5),
                    'psb': p['pred_start_bar'] + start,
                    'psp': round(p['pred_start_price'], 5),
                    'ptb': p['pred_target_bar'] + start,
                    'ptp': round(p['pred_target_price'], 5),
                    'pa': round(p['pred_amp'], 5),
                    'pt': p['pred_time'],
                    'sc': round(p['score'], 5),
                    'prog': round(p.get('actual_progress', 0), 3),  # 进度
                    # 弧线参数
                    'aa': round(p['A_amp'], 5),     # A段原始幅度
                    'at': p['A_time'],               # A段原始时间跨度
                    'mr': round(mod_R, 6),           # 归一化模长 R
                    'ga': round(global_amp, 5),      # 全局幅度
                    'gs': global_span,               # 全局时间跨度
                }
                
                # 对称中心/轴 (mirror有center_bar/center_price, center有B段)
                if p['type'] == 'mirror':
                    pg['cb'] = round(p.get('center_bar', 0), 1)
                    pg['cp'] = round(p.get('center_price', 0), 5)
                
                # B段信息 (center, triangle, modonly 都有)
                if p['type'] in ('center', 'triangle', 'modonly'):
                    pg['bs'] = p['B_start'] + start
                    pg['be'] = p['B_end'] + start
                    pg['bps'] = round(p['B_price_start'], 5)
                    pg['bpe'] = round(p['B_price_end'], 5)
                
                # 三角专属参数
                if p['type'] == 'triangle':
                    pg['st'] = p['tri_type'][0]  # 'c'=converging, 'd'=diverging
                    pg['ar'] = p['amp_ratio']     # B/A幅度比
                    pg['tr'] = p['time_ratio']    # B/A时间比
                    # 三角边界线 (偏移到全局坐标)
                    pg['bt'] = [[b[0] + start, round(b[1], 5)]
                                for b in p['boundary_top']]
                    pg['bb'] = [[b[0] + start, round(b[1], 5)]
                                for b in p['boundary_bot']]
                
                # ModOnly专属参数
                if p['type'] == 'modonly':
                    pg['ms'] = p.get('mod_sym', 0)    # 模长对称度
                    pg['asy'] = p.get('amp_sym', 0)    # 幅度对称度
                
                pred_data.append(pg)
        
        # --- Symmetry structures ---
        if show_sym and pruned:
            try:
                syms = find_symmetric_structures(pruned, pi, top_n=200,
                                                  max_pool_size=300)
                sym_data = []
                for s in syms:
                    sym_data.append({
                        'p1': s['p1'] + start, 'p2': s['p2'] + start,
                        'p3': s['p3'] + start, 'p4': s['p4'] + start,
                        'pp1': round(s['price_p1'], 5), 'pp2': round(s['price_p2'], 5),
                        'pp3': round(s['price_p3'], 5), 'pp4': round(s['price_p4'], 5),
                        'sc': round(s['score'], 4),
                        'sym': round(s['sym_score'], 3),
                        'imp': round(s['endpoint_imp'], 4),
                        'va': round(s['vec']['amp'], 3),
                        'vt': round(s['vec']['time'], 3),
                        'vm': round(s['vec']['mod'], 3),
                        'vs': round(s['vec']['slope'], 3),
                        'vc': round(s['vec']['complexity'], 3),
                        'dr': s['dir'],
                        'tp': s['type'],
                    })
            except Exception as e:
                print(f"Symmetry error: {e}")
                sym_data = None
    
    if show_pvt and pi:
        # 分别排序peaks和valleys
        peaks = sorted([v for v in pi.values() if v['dir'] == 1],
                       key=lambda x: -x['importance'])
        valleys = sorted([v for v in pi.values() if v['dir'] == -1],
                         key=lambda x: -x['importance'])
        pivot_data = {
            'peaks': [{'b': pv['bar'] + start, 'p': round(pv['price'], 5),
                        'imp': round(pv['importance'], 4)}
                       for pv in peaks[:peak_n]],
            'valleys': [{'b': pv['bar'] + start, 'p': round(pv['price'], 5),
                          'imp': round(pv['importance'], 4)}
                         for pv in valleys[:valley_n]],
            'total_peaks': len(peaks),
            'total_valleys': len(valleys),
        }
    
    elapsed = time.time() - t0
    
    return {
        'start_bar': start,
        'end_bar': end_bar,
        'window': window,
        'total_bars': G['n'],
        'klines': klines,
        'snapshots': snapshots,
        'extra': extra_data,
        'pool': pool_data,
        'predictions': pred_data,
        'pivots': pivot_data,
        'symmetry': sym_data,
        'compute_ms': round(elapsed * 1000, 1),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")
    
    try:
        while True:
            msg = await websocket.receive_text()
            req = json.loads(msg)
            
            end_bar = int(req.get('end_bar', 100))
            window = int(req.get('window', 25))
            
            result = compute_window(
                end_bar=end_bar,
                window=window,
                show_lat=req.get('show_lat', False),
                show_fusion=req.get('show_fusion', False),
                show_pred=req.get('show_pred', False),
                show_pvt=req.get('show_pvt', False),
                show_sym=req.get('show_sym', False),
                show_extra=req.get('show_extra', False),
                min_imp=float(req.get('min_imp', 0)),
                peak_n=int(req.get('peak_n', 10)),
                valley_n=int(req.get('valley_n', 10)),
            )
            
            await websocket.send_text(json.dumps(result))
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='v32 Visualizer Server')
    parser.add_argument('--bars', type=int, default=5000, help='Number of bars to load')
    parser.add_argument('--port', type=int, default=8765, help='Server port')
    args = parser.parse_args()
    
    app._bars = args.bars
    print(f"Starting v32 server on port {args.port}...")
    uvicorn.run(app, host='0.0.0.0', port=args.port)
