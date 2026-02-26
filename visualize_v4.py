#!/usr/bin/env python3
"""
动态对称预测可视化 v4.0

核心特性:
1. 逐K线动态推进 — DynamicEngine step-by-step
2. 四窗口分离: 大尺度 / 中尺度 / 小尺度 / 预测日志
3. 预测生命周期跟踪: 诞生→展开→验证/失效, 边界记录

每个step:
  K线喂入 → ZG更新 → merge更新 → pool更新 
  → predict_symmetric_image → 预测生命周期更新
"""

import json
import sys
import time
import copy
sys.path.insert(0, '/home/ubuntu/stage2_abc')
from merge_engine_v3 import (load_kline, predict_symmetric_image,
                              compute_pivot_importance, full_merge_engine,
                              build_segment_pool, pool_fusion, calculate_base_zg)
from dynamic_engine import DynamicEngine


def build_pivot_info_from_pool(pool):
    """从pool构建简化版pivot_info (动态版没有完整的8维importance)"""
    pivot_info = {}
    for seg in pool:
        for bar, price, d in [(seg['bar_start'], seg['price_start'], seg['dir_start']),
                               (seg['bar_end'], seg['price_end'], seg['dir_end'])]:
            if bar not in pivot_info:
                pivot_info[bar] = {
                    'bar': bar, 'price': price, 'dir': d,
                    'importance': 0,
                }
            pivot_info[bar]['importance'] = max(
                pivot_info[bar]['importance'],
                seg['amplitude'] * seg['span']
            )
    
    max_imp = max((p['importance'] for p in pivot_info.values()), default=1)
    if max_imp > 0:
        for p in pivot_info.values():
            p['importance'] /= max_imp
    
    return pivot_info


class PredictionTracker:
    """
    预测生命周期管理器。
    
    每个预测有唯一ID, 生命周期状态:
    - active: 预测正在展开, 尚未达到目标或失效
    - hit: 价格到达预测目标区间
    - invalidated: 预测被新结构否定 (对称中心被突破, 或反向大幅偏离)
    - expired: 预测目标时间已过但未达到目标价格
    
    边界(boundary): 预测仍然有效的条件
    - price_boundary: 价格不能突破对称中心 (mirror) 或B段极值 (center) 
    - time_boundary: 不能超过pred_target_bar * 2
    """
    
    def __init__(self):
        self.all_predictions = {}  # id → prediction dict (含lifecycle)
        self.next_id = 0
        self.log = []  # [(bar, event_type, pred_id, detail)]
    
    def register(self, pred, birth_bar):
        """注册一个新预测"""
        pid = self.next_id
        self.next_id += 1
        
        # 计算边界
        if pred['type'] == 'mirror':
            # Mirror: 价格不能反向突破A的起点
            if pred['pred_dir'] == 1:  # 预测涨
                invalidate_price = pred['center_price'] - pred['A_amp'] * 0.5  # 跌破中心-50%A
            else:
                invalidate_price = pred['center_price'] + pred['A_amp'] * 0.5
        else:
            # Center: 价格不能突破B段与A段极值
            if pred['pred_dir'] == 1:
                invalidate_price = pred['pred_start_price'] - pred['A_amp'] * 0.5
            else:
                invalidate_price = pred['pred_start_price'] + pred['A_amp'] * 0.5
        
        # 目标区间: 不是精确一个点, 而是 ±50% A_amp (宽松: 对称是近似的)
        target_range = pred['A_amp'] * 0.5
        target_high = pred['pred_target_price'] + target_range
        target_low = pred['pred_target_price'] - target_range
        
        # 时间边界: 2倍预测时间
        time_limit = pred['pred_start_bar'] + pred['pred_time'] * 2
        
        entry = {
            'id': pid,
            'pred': pred,
            'birth_bar': birth_bar,
            'status': 'active',
            'invalidate_price': invalidate_price,
            'target_high': target_high,
            'target_low': target_low,
            'time_limit': time_limit,
            'peak_progress': 0.0,
            'history': [(birth_bar, 'born', 0.0)],
            'death_bar': None,
            'death_reason': None,
        }
        
        self.all_predictions[pid] = entry
        self.log.append((birth_bar, 'born', pid, 
                        f"{pred['type'][0].upper()} A:{pred['A_start']}→{pred['A_end']} → target={pred['pred_target_price']:.5f}"))
        
        return pid
    
    def update(self, bar, current_high, current_low, current_close):
        """用当前K线更新所有active预测的状态"""
        events = []
        
        for pid, entry in self.all_predictions.items():
            if entry['status'] != 'active':
                continue
            
            pred = entry['pred']
            
            # 检查是否被失效 (价格突破边界)
            if pred['pred_dir'] == 1:
                # 预测涨, 如果价格跌破invalidate_price
                if current_low < entry['invalidate_price']:
                    entry['status'] = 'invalidated'
                    entry['death_bar'] = bar
                    entry['death_reason'] = f'price broke {entry["invalidate_price"]:.5f}'
                    entry['history'].append((bar, 'invalidated', 0.0))
                    self.log.append((bar, 'invalidated', pid, entry['death_reason']))
                    events.append(('invalidated', pid))
                    continue
            else:
                # 预测跌, 如果价格涨破invalidate_price
                if current_high > entry['invalidate_price']:
                    entry['status'] = 'invalidated'
                    entry['death_bar'] = bar
                    entry['death_reason'] = f'price broke {entry["invalidate_price"]:.5f}'
                    entry['history'].append((bar, 'invalidated', 0.0))
                    self.log.append((bar, 'invalidated', pid, entry['death_reason']))
                    events.append(('invalidated', pid))
                    continue
            
            # 检查是否hit目标
            if current_low <= entry['target_high'] and current_high >= entry['target_low']:
                # 价格进入目标区间
                entry['status'] = 'hit'
                entry['death_bar'] = bar
                entry['death_reason'] = 'target reached'
                entry['history'].append((bar, 'hit', 1.0))
                self.log.append((bar, 'hit', pid, 
                               f'target zone [{entry["target_low"]:.5f}, {entry["target_high"]:.5f}]'))
                events.append(('hit', pid))
                continue
            
            # 检查时间过期
            if bar > entry['time_limit']:
                entry['status'] = 'expired'
                entry['death_bar'] = bar
                entry['death_reason'] = f'time limit bar {entry["time_limit"]}'
                entry['history'].append((bar, 'expired', entry['peak_progress']))
                self.log.append((bar, 'expired', pid, entry['death_reason']))
                events.append(('expired', pid))
                continue
            
            # 更新progress
            if pred['A_amp'] > 0:
                if pred['pred_dir'] == 1:
                    progress = (current_high - pred['pred_start_price']) / pred['A_amp']
                else:
                    progress = (pred['pred_start_price'] - current_low) / pred['A_amp']
                progress = max(0, min(progress, 1.5))
                entry['peak_progress'] = max(entry['peak_progress'], progress)
                entry['history'].append((bar, 'update', progress))
        
        return events
    
    def get_active(self):
        """获取所有active预测"""
        return [e for e in self.all_predictions.values() if e['status'] == 'active']
    
    def get_summary(self):
        """统计摘要"""
        statuses = {}
        for e in self.all_predictions.values():
            statuses[e['status']] = statuses.get(e['status'], 0) + 1
        return statuses


def run_dynamic_predictions(df, pred_interval=5, min_pool_for_pred=20,
                            top_pred_per_step=30):
    """
    运行完整的动态预测流程。
    
    Args:
        df: K线数据
        pred_interval: 每隔多少根K线做一次预测 (不是每根K线都做, 太贵)
        min_pool_for_pred: 池中至少多少线段才开始做预测
        top_pred_per_step: 每次取top N个预测注册
    
    Returns:
        frames: 每根K线的状态快照 (用于可视化回放)
        tracker: PredictionTracker
    """
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    engine = DynamicEngine()
    tracker = PredictionTracker()
    
    frames = []  # 每根K线一帧
    
    # 已注册预测的key set (避免重复注册相同结构)
    registered_keys = set()
    
    t0 = time.time()
    
    for i in range(n):
        h, l = float(high[i]), float(low[i])
        c = float(close[i])
        
        # 1. DynamicEngine step
        snap = engine.step(h, l)
        
        # 2. 更新所有active预测
        tracker.update(i, h, l, c)
        
        # 3. 定期做新预测 (不是每根K线都做)
        new_preds = []
        if i > 0 and i % pred_interval == 0:
            pool = engine.pool.pool
            if len(pool) >= min_pool_for_pred:
                pivot_info = build_pivot_info_from_pool(pool)
                predictions = predict_symmetric_image(pool, pivot_info, current_bar=i)
                
                # 只注册top N个, 且避免重复
                count = 0
                for pred in predictions:
                    if count >= top_pred_per_step:
                        break
                    # 用 (type, A_start, A_end, center_bar, pred_dir) 作为唯一key
                    key = (pred['type'], pred['A_start'], pred['A_end'], 
                           int(pred['center_bar']), pred['pred_dir'])
                    if key in registered_keys:
                        continue
                    registered_keys.add(key)
                    
                    # 预测起点必须已经是过去的 (A和center/pivot已确认)
                    if pred['pred_start_bar'] > i:
                        continue
                    
                    pid = tracker.register(pred, birth_bar=i)
                    new_preds.append(pid)
                    count += 1
        
        # 4. 收集当前帧的全部ZG拐点 (包括tentative)
        confirmed_pivots = [(p['bar'], p['price'], p['dir']) 
                           for p in engine.zg.pivots]
        tent = engine.zg.tentative
        if tent:
            confirmed_pivots.append((tent['bar'], tent['price'], tent['dir']))
        
        # 5. 构建帧
        active_preds = tracker.get_active()
        frame = {
            'bar': i,
            'kline': {'o': round(float(df.iloc[i]['open']), 5),
                      'h': round(h, 5), 'l': round(l, 5),
                      'c': round(c, 5)},
            'n_pivots': len(engine.zg.pivots),
            'n_segments': len(engine.pool.pool),
            'n_active_preds': len(active_preds),
            'new_preds': new_preds,
            'has_event': snap['has_new_pivot'] or len(new_preds) > 0,
            # 拐点序列 (只存confirmed + tentative)
            'pivots': [(p[0], round(p[1], 5), p[2]) for p in confirmed_pivots],
        }
        
        frames.append(frame)
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            summ = tracker.get_summary()
            print(f"  bar {i+1}/{n} ({elapsed:.1f}s) | "
                  f"pivots={snap['n_confirmed_pivots']} segs={snap['n_total_segments']} | "
                  f"preds: {summ}")
    
    elapsed = time.time() - t0
    print(f"\n完成: {n} bars, {elapsed:.1f}s")
    print(f"最终: pivots={len(engine.zg.pivots)}, segments={len(engine.pool.pool)}")
    print(f"预测统计: {tracker.get_summary()}")
    print(f"日志条目: {len(tracker.log)}")
    
    return frames, tracker, engine


def generate_dynamic_html(df, frames, tracker, output_path):
    """
    生成四窗口动态可视化HTML。
    
    四窗口布局:
    - 左上: K线 + ZG + 大尺度预测 (A_time >= 30)
    - 右上: K线 + ZG + 中尺度预测 (10 <= A_time < 30)
    - 左下: K线 + ZG + 小尺度预测 (A_time < 10)
    - 右下: 预测生命周期日志
    """
    
    # K线数据
    kline_data = []
    for i in range(len(df)):
        kline_data.append({
            'o': round(df.iloc[i]['open'], 5),
            'h': round(df.iloc[i]['high'], 5),
            'l': round(df.iloc[i]['low'], 5),
            'c': round(df.iloc[i]['close'], 5),
        })
    
    # 每帧的pivot序列
    frame_pivots = [f['pivots'] for f in frames]
    
    # 预测数据 — 每个预测的完整生命周期
    pred_data = []
    for pid, entry in tracker.all_predictions.items():
        p = entry['pred']
        pd_entry = {
            'id': pid,
            'type': p['type'],
            'A_s': p['A_start'], 'A_e': p['A_end'],
            'A_ps': round(p['A_price_start'], 5), 'A_pe': round(p['A_price_end'], 5),
            'A_time': p['A_time'], 'A_amp': round(p['A_amp'], 5),
            'A_dir': p['A_dir'],
            'pd': p['pred_dir'],
            'ps_bar': p['pred_start_bar'],
            'ps_prc': round(p['pred_start_price'], 5),
            'pt_prc': round(p['pred_target_price'], 5),
            'pt_bar': p['pred_target_bar'],
            'score': round(p['score'], 5),
            'birth': entry['birth_bar'],
            'status': entry['status'],
            'death': entry['death_bar'] if entry['death_bar'] is not None else -1,
            'inv_prc': round(entry['invalidate_price'], 5),
            'tgt_hi': round(entry['target_high'], 5),
            'tgt_lo': round(entry['target_low'], 5),
            'tlimit': entry['time_limit'],
            'peak_prog': round(entry['peak_progress'], 3),
        }
        if p['type'] == 'center':
            pd_entry['B_s'] = p['B_start']
            pd_entry['B_e'] = p['B_end']
            pd_entry['B_ps'] = round(p['B_price_start'], 5)
            pd_entry['B_pe'] = round(p['B_price_end'], 5)
        pred_data.append(pd_entry)
    
    # 日志数据
    log_data = [(bar, evt, pid, detail) for bar, evt, pid, detail in tracker.log]
    
    # 统计
    summ = tracker.get_summary()
    n_total = len(pred_data)
    n_hit = summ.get('hit', 0)
    n_inv = summ.get('invalidated', 0)
    n_exp = summ.get('expired', 0)
    n_act = summ.get('active', 0)
    hit_rate = n_hit / max(n_hit + n_inv + n_exp, 1)
    
    n_bars = len(kline_data)
    dt_start = df['datetime'].iloc[0]
    dt_end = df['datetime'].iloc[-1]
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>动态对称预测 v4.0 | EURUSD H1</title>
<style>
* {{ box-sizing: border-box; }}
body {{ background: #0a0a1a; color: #ddd; font-family: 'Consolas', monospace; margin: 0; padding: 8px; }}
h2 {{ color: #7eb8da; margin: 3px 0; font-size: 15px; }}
.info {{ background: #12122a; padding: 6px 10px; margin: 3px 0; border-radius: 4px; font-size: 12px; border-left: 3px solid #4488aa; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 4px; height: calc(100vh - 120px); }}
.panel {{ background: #0d0d20; border: 1px solid #1a1a3a; border-radius: 4px; position: relative; overflow: hidden; }}
.panel-title {{ position: absolute; top: 2px; left: 6px; font-size: 11px; color: #888; z-index: 10; }}
canvas {{ width: 100%; height: 100%; }}
.controls {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; flex-wrap: wrap; }}
.btn {{ background: #1a2a4a; color: #8ac; border: 1px solid #335; padding: 3px 8px; border-radius: 3px; cursor: pointer; font-size: 11px; }}
.btn:hover {{ background: #2a3a5a; }}
.btn.active {{ background: #3a2a5a; color: #f8f; border-color: #a6a; }}
.slider-box {{ display: inline-flex; align-items: center; gap: 4px; font-size: 11px; }}
.slider-box input[type=range] {{ width: 200px; }}
#logPanel {{ overflow-y: auto; padding: 4px 6px; font-size: 10px; line-height: 1.4; }}
.log-born {{ color: #8cf; }}
.log-hit {{ color: #4f4; font-weight: bold; }}
.log-invalidated {{ color: #f66; }}
.log-expired {{ color: #886; }}
#info {{ font-size: 11px; color: #888; height: 14px; margin: 2px 0; }}
</style></head><body>

<h2>动态对称预测 v4.0 | EURUSD H1 | {dt_start} ~ {dt_end}</h2>
<div class="info">
{n_bars} bars | 预测总数: {n_total} | 
<span style="color:#4f4">命中: {n_hit}</span> | 
<span style="color:#f66">失效: {n_inv}</span> | 
<span style="color:#886">过期: {n_exp}</span> | 
<span style="color:#8cf">活跃: {n_act}</span> | 
命中率: {hit_rate:.1%} (命中/已结束)
</div>

<div class="controls">
  <button class="btn" onclick="stepTo(0)">|&lt;</button>
  <button class="btn" onclick="stepBy(-10)">&lt;&lt;</button>
  <button class="btn" onclick="stepBy(-1)">&lt;</button>
  <button class="btn" id="playBtn" onclick="togglePlay()">▶ Play</button>
  <button class="btn" onclick="stepBy(1)">&gt;</button>
  <button class="btn" onclick="stepBy(10)">&gt;&gt;</button>
  <button class="btn" onclick="stepTo({n_bars-1})">&gt;|</button>
  <span class="slider-box">
    Bar: <input type="range" id="barSlider" min="0" max="{n_bars-1}" value="{n_bars-1}" 
         oninput="stepTo(parseInt(this.value))">
    <span id="barNum">{n_bars-1}</span>/{n_bars-1}
  </span>
  <span class="slider-box">
    Speed: <input type="range" id="speedSlider" min="1" max="20" value="5">
    <span id="speedVal">5</span>
  </span>
</div>

<div id="info"></div>

<div class="grid">
  <div class="panel">
    <div class="panel-title">大尺度 (A_time ≥ 30)</div>
    <canvas id="c0"></canvas>
  </div>
  <div class="panel">
    <div class="panel-title">中尺度 (10 ≤ A_time < 30)</div>
    <canvas id="c1"></canvas>
  </div>
  <div class="panel">
    <div class="panel-title">小尺度 (A_time < 10)</div>
    <canvas id="c2"></canvas>
  </div>
  <div class="panel" id="logPanel">
    <div class="panel-title">预测日志</div>
    <div id="logContent" style="margin-top:16px;"></div>
  </div>
</div>

<script>
const K = {json.dumps(kline_data)};
const PIVOTS = {json.dumps(frame_pivots)};
const PREDS = {json.dumps(pred_data)};
const LOG = {json.dumps(log_data)};
const N = K.length;

// 尺度分类
const PRED_LARGE = PREDS.filter(p => p.A_time >= 30);
const PRED_MED   = PREDS.filter(p => p.A_time >= 10 && p.A_time < 30);
const PRED_SMALL = PREDS.filter(p => p.A_time < 10);

let curBar = N - 1;
let playing = false;
let playTimer = null;

// 价格范围
let pMin = Infinity, pMax = -Infinity;
for(const k of K) {{ pMin = Math.min(pMin, k.l); pMax = Math.max(pMax, k.h); }}
const pRange = pMax - pMin;
pMin -= pRange * 0.03; pMax += pRange * 0.03;

function setupCanvas(id) {{
    const cv = document.getElementById(id);
    const rect = cv.parentElement.getBoundingClientRect();
    cv.width = rect.width;
    cv.height = rect.height;
    return cv;
}}

function xS(bar, W, mg) {{ return mg.l + (bar / N) * (W - mg.l - mg.r); }}
function yS(price, H, mg) {{ return mg.t + (H - mg.t - mg.b) - ((price - pMin) / (pMax - pMin)) * (H - mg.t - mg.b); }}

function drawPanel(canvasId, preds, label) {{
    const cv = document.getElementById(canvasId);
    const rect = cv.parentElement.getBoundingClientRect();
    cv.width = rect.width;
    cv.height = rect.height;
    const cx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    const mg = {{l: 55, r: 10, t: 18, b: 15}};
    
    // Background
    cx.fillStyle = '#0d0d20';
    cx.fillRect(0, 0, W, H);
    
    // Grid
    cx.strokeStyle = '#161630'; cx.lineWidth = 0.5;
    cx.fillStyle = '#444'; cx.font = '9px monospace';
    for(let i = 0; i <= 8; i++) {{
        const p = pMin + (pMax - pMin) * i / 8;
        const y = yS(p, H, mg);
        cx.beginPath(); cx.moveTo(mg.l, y); cx.lineTo(W - mg.r, y); cx.stroke();
        cx.fillText(p.toFixed(4), 2, y + 3);
    }}
    
    // K-lines up to curBar
    for(let i = 0; i <= curBar && i < N; i++) {{
        const k = K[i];
        const x = xS(i, W, mg);
        cx.strokeStyle = k.c >= k.o ? '#1a4a1a' : '#4a1a1a';
        cx.lineWidth = 0.6;
        cx.beginPath(); cx.moveTo(x, yS(k.l, H, mg)); cx.lineTo(x, yS(k.h, H, mg)); cx.stroke();
    }}
    
    // Current bar marker
    if(curBar < N) {{
        const x = xS(curBar, W, mg);
        cx.strokeStyle = '#ffff0033'; cx.lineWidth = 1;
        cx.beginPath(); cx.moveTo(x, mg.t); cx.lineTo(x, H - mg.b); cx.stroke();
    }}
    
    // Pivots at curBar
    const pvts = PIVOTS[Math.min(curBar, N-1)];
    if(pvts.length >= 2) {{
        cx.strokeStyle = '#FFD700'; cx.lineWidth = 0.8; cx.globalAlpha = 0.5;
        cx.beginPath();
        cx.moveTo(xS(pvts[0][0], W, mg), yS(pvts[0][1], H, mg));
        for(let j = 1; j < pvts.length; j++) {{
            cx.lineTo(xS(pvts[j][0], W, mg), yS(pvts[j][1], H, mg));
        }}
        cx.stroke();
        cx.globalAlpha = 1;
    }}
    
    // Predictions
    for(let i = 0; i < preds.length; i++) {{
        const p = preds[i];
        // 只画已出生且在当前视野中的
        if(p.birth > curBar) continue;
        
        const isActive = p.status === 'active' ? (p.death === -1 || p.death > curBar) : (p.death > curBar || p.death === -1);
        const isDead = p.death !== -1 && p.death <= curBar;
        
        // 确定是否在当前时刻active
        let liveNow = p.birth <= curBar && (p.death === -1 || p.death > curBar);
        
        let alpha = liveNow ? 0.8 : 0.15;
        let lw = liveNow ? 1.8 : 0.5;
        
        // 颜色: mirror=橙, center=青
        const col = p.type === 'mirror' 
            ? (liveNow ? `rgba(255,160,40,${{alpha}})` : `rgba(255,160,40,${{alpha}})`)
            : (liveNow ? `rgba(40,200,255,${{alpha}})` : `rgba(40,200,255,${{alpha}})`);
        const predCol = p.type === 'mirror'
            ? `rgba(255,120,0,${{alpha}})`
            : `rgba(0,180,255,${{alpha}})`;
        
        // A段 (实线)
        cx.strokeStyle = col; cx.lineWidth = lw; cx.setLineDash([]);
        cx.beginPath();
        cx.moveTo(xS(p.A_s, W, mg), yS(p.A_ps, H, mg));
        cx.lineTo(xS(p.A_e, W, mg), yS(p.A_pe, H, mg));
        cx.stroke();
        
        // B段 (center型)
        if(p.type === 'center' && p.B_s !== undefined) {{
            cx.strokeStyle = `rgba(150,150,150,${{alpha*0.5}})`; 
            cx.lineWidth = lw * 0.4; cx.setLineDash([3, 2]);
            cx.beginPath();
            cx.moveTo(xS(p.B_s, W, mg), yS(p.B_ps, H, mg));
            cx.lineTo(xS(p.B_e, W, mg), yS(p.B_pe, H, mg));
            cx.stroke();
        }}
        
        // C' 预测 (虚线+箭头)
        cx.strokeStyle = predCol; cx.lineWidth = lw * 1.2; cx.setLineDash([6, 3]);
        cx.beginPath();
        cx.moveTo(xS(p.ps_bar, W, mg), yS(p.ps_prc, H, mg));
        cx.lineTo(xS(p.pt_bar, W, mg), yS(p.pt_prc, H, mg));
        cx.stroke();
        
        // 目标区间 (半透明矩形)
        if(liveNow) {{
            cx.fillStyle = p.type === 'mirror' ? 'rgba(255,160,40,0.06)' : 'rgba(40,200,255,0.06)';
            const y1 = yS(p.tgt_hi, H, mg), y2 = yS(p.tgt_lo, H, mg);
            const x1 = xS(p.ps_bar, W, mg), x2 = xS(p.pt_bar, W, mg);
            cx.fillRect(x1, Math.min(y1,y2), x2-x1, Math.abs(y2-y1));
        }}
        
        // 失效价格线 (红色虚线)
        if(liveNow) {{
            cx.strokeStyle = 'rgba(255,80,80,0.25)'; cx.lineWidth = 0.5; cx.setLineDash([2, 3]);
            const invY = yS(p.inv_prc, H, mg);
            cx.beginPath(); cx.moveTo(xS(p.ps_bar, W, mg), invY); cx.lineTo(xS(p.pt_bar, W, mg), invY); cx.stroke();
        }}
        
        // 状态标记
        if(isDead && p.death <= curBar) {{
            const dx = xS(p.death, W, mg);
            if(p.status === 'hit') {{
                cx.fillStyle = 'rgba(0,255,0,0.8)';
                cx.font = 'bold 10px monospace';
                cx.fillText('✓', dx - 4, yS(p.pt_prc, H, mg) + 4);
            }} else if(p.status === 'invalidated') {{
                cx.fillStyle = 'rgba(255,60,60,0.8)';
                cx.font = 'bold 10px monospace';
                cx.fillText('✗', dx - 4, yS(p.inv_prc, H, mg) + 4);
            }}
        }}
        
        cx.setLineDash([]);
    }}
    cx.globalAlpha = 1;
}}

function drawLog() {{
    const el = document.getElementById('logContent');
    let html = '';
    for(let i = LOG.length - 1; i >= 0; i--) {{
        const [bar, evt, pid, detail] = LOG[i];
        if(bar > curBar) continue;
        const cls = 'log-' + evt;
        html += `<div class="${{cls}}">bar${{bar}} [${{evt}}] #${{pid}} ${{detail}}</div>`;
    }}
    el.innerHTML = html;
}}

function drawAll() {{
    drawPanel('c0', PRED_LARGE, 'Large');
    drawPanel('c1', PRED_MED, 'Medium');
    drawPanel('c2', PRED_SMALL, 'Small');
    drawLog();
    
    document.getElementById('barNum').textContent = curBar;
    document.getElementById('barSlider').value = curBar;
    
    // Info bar
    if(curBar < N) {{
        const k = K[curBar];
        const pvts = PIVOTS[curBar];
        const actives = PREDS.filter(p => p.birth <= curBar && (p.death === -1 || p.death > curBar));
        document.getElementById('info').textContent = 
            `Bar ${{curBar}} | O:${{k.o}} H:${{k.h}} L:${{k.l}} C:${{k.c}} | ` +
            `Pivots:${{pvts.length}} | Active preds: ${{actives.length}}`;
    }}
}}

function stepTo(bar) {{
    curBar = Math.max(0, Math.min(N-1, bar));
    drawAll();
}}

function stepBy(d) {{
    stepTo(curBar + d);
}}

function togglePlay() {{
    playing = !playing;
    document.getElementById('playBtn').textContent = playing ? '⏸ Pause' : '▶ Play';
    if(playing) {{
        playTimer = setInterval(() => {{
            const speed = parseInt(document.getElementById('speedSlider').value);
            document.getElementById('speedVal').textContent = speed;
            stepBy(speed);
            if(curBar >= N - 1) {{
                playing = false;
                clearInterval(playTimer);
                document.getElementById('playBtn').textContent = '▶ Play';
            }}
        }}, 100);
    }} else {{
        clearInterval(playTimer);
    }}
}}

// Handle resize
window.addEventListener('resize', drawAll);

// Initial draw
drawAll();
</script></body></html>"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Saved: {output_path} ({len(html)//1024}KB)")


def main():
    print("=" * 70)
    print("动态对称预测 v4.0")
    print("=" * 70)
    
    df = load_kline("/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv", limit=200)
    print(f"数据: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]} ({len(df)} bars)")
    
    print(f"\n运行动态引擎...")
    frames, tracker, engine = run_dynamic_predictions(
        df, 
        pred_interval=5,      # 每5根K线做一次预测
        min_pool_for_pred=10,  # 至少10个线段才开始预测
        top_pred_per_step=20,  # 每次最多注册20个新预测
    )
    
    print(f"\n生成HTML...")
    generate_dynamic_html(df, frames, tracker, 
                          "/home/ubuntu/stage2_abc/merge_v4.html")


if __name__ == '__main__':
    main()
