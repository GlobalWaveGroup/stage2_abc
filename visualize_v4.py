#!/usr/bin/env python3
"""
动态对称预测可视化 v4.2

核心:
1. 2000K线, 从bar50开始预测, 窗口150K+50K空白, 单K线滚动
2. 三窗口: 幅度归并ZG / 时间归并ZG / Fusion(amp+lat的再归并)
3. 同类预测替换, 固定数量预测线, 预测不超过50K
4. 明亮配色

架构:
- DynamicZG + DynamicMerger 逐K线推进 (0.4s for 2000K)
- 不做 pool_fusion (太慢), 直接用 base/amp/lat 线段
- 每有新拐点确认时做predict, 中间帧复用
- 三窗口对应: amp来源(base+amp) / lat来源 / amp+lat全部
"""

import json
import sys
import time
sys.path.insert(0, '/home/ubuntu/stage2_abc')
from merge_engine_v3 import load_kline, predict_symmetric_image
from dynamic_engine import DynamicZG, DynamicMerger


def build_pivot_info_simple(segments):
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


def encode_pred(p):
    """编码单个预测为紧凑dict"""
    entry = {
        't': p['type'][0],  # 'm' or 'c'
        'as': p['A_start'], 'ae': p['A_end'],
        'aps': round(p['A_price_start'], 5), 'ape': round(p['A_price_end'], 5),
        'ad': p['A_dir'],
        'psb': p['pred_start_bar'], 'psp': round(p['pred_start_price'], 5),
        'ptb': p['pred_target_bar'], 'ptp': round(p['pred_target_price'], 5),
        'sc': round(p['score'], 5),
    }
    if p['type'] == 'center':
        entry['bs'] = p['B_start']
        entry['be'] = p['B_end']
        entry['bps'] = round(p['B_price_start'], 5)
        entry['bpe'] = round(p['B_price_end'], 5)
    return entry


def run_engine(df, start_pred=50, max_preds=15, pred_horizon=50):
    """
    逐K线推进ZG+Merger, 在关键帧做预测。
    
    三组预测源:
    - amp: base + amp 线段 (幅度归并级别)
    - lat: lat 线段 (时间归并级别)
    - all: base + amp + lat 全部 (综合)
    """
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    
    zg = DynamicZG()
    merger = DynamicMerger()
    
    # 每帧数据 — 分离 confirmed/tentative 以压缩
    # confirmed_events: [(bar_idx, bar, price, dir)] — 只在新确认时追加
    # frame_tentative: [null | [bar, price, dir]] — 每帧的tentative点
    confirmed_events = []   # append-only list
    frame_tentative = []    # per-frame, small
    frame_preds = []        # per-frame predictions
    
    # 当前预测 (复用到下一个关键帧)
    cur_preds = {'amp': [], 'lat': [], 'all': []}
    
    t0 = time.time()
    
    for i in range(n):
        h_val, l_val = float(high[i]), float(low[i])
        
        # 1. ZG step
        events = zg.step(h_val, l_val)
        
        # 2. 处理确认拐点
        has_new = False
        for ev in events:
            if ev['type'] == 'confirmed':
                merger.add_pivot(ev['pivot'], i)
                has_new = True
                # 记录确认事件 (在哪个bar_idx确认的, 拐点信息)
                p = ev['pivot']
                confirmed_events.append((i, p['bar'], round(p['price'], 5), p['dir']))
        
        # 3. 当前tentative点
        if zg.tentative:
            frame_tentative.append((zg.tentative['bar'], round(zg.tentative['price'], 5), zg.tentative['dir']))
        else:
            frame_tentative.append(None)
        
        # 4. 在关键帧做预测 (有新拐点 且 i >= start_pred)
        if has_new and i >= start_pred and len(merger.segments) >= 5:
            all_segs = merger.segments
            
            # 截至当前bar出生的线段
            live_segs = [s for s in all_segs if s.get('birth_bar', 0) <= i]
            
            seg_amp = [s for s in live_segs if s['source'] in ('amp', 'base')]
            seg_lat = [s for s in live_segs if s['source'] == 'lat']
            
            for src_name, src_segs in [('amp', seg_amp), ('lat', seg_lat), ('all', live_segs)]:
                if len(src_segs) < 3:
                    cur_preds[src_name] = []
                    continue
                
                pivot_info = build_pivot_info_simple(src_segs)
                # 不限制pool size — 1000+个线段也只需0.01s
                raw = predict_symmetric_image(src_segs, pivot_info, current_bar=i,
                                              max_pool_size=99999)
                
                # 严格筛选: 起点已存在 + 目标在未来1~50K
                valid = []
                for p in raw:
                    if p['pred_start_bar'] > i:
                        continue
                    if p['pred_target_bar'] > i + pred_horizon:
                        continue
                    if p['pred_target_bar'] <= i:
                        continue
                    valid.append(p)
                
                # 按score取top, 但保证多样性
                valid.sort(key=lambda p: -p['score'])
                cur_preds[src_name] = valid[:max_preds]
        
        # 5. 存帧 (编码当前预测)
        frame_preds.append({
            'a': [encode_pred(p) for p in cur_preds['amp']],
            'l': [encode_pred(p) for p in cur_preds['lat']],
            'f': [encode_pred(p) for p in cur_preds['all']],
        })
        
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            na = len(cur_preds['amp'])
            nl = len(cur_preds['lat'])
            nf = len(cur_preds['all'])
            print(f"  bar {i+1}/{n} ({elapsed:.1f}s) pivots={len(zg.pivots)} "
                  f"segs={len(merger.segments)} preds: A={na} L={nl} All={nf}")
    
    elapsed = time.time() - t0
    print(f"完成: {n} bars, {elapsed:.1f}s, pivots={len(zg.pivots)}, segs={len(merger.segments)}")
    
    return confirmed_events, frame_tentative, frame_preds


def generate_html(df, confirmed_events, frame_tentative, frame_preds, output_path):
    """生成三窗口动态可视化HTML (增量编码, 目标<5MB)
    
    数据格式:
    - confirmed_events: [(frame_idx, bar, price, dir), ...] — append-only
    - frame_tentative: [null | [bar,price,dir], ...] — per frame
    - frame_preds: [{a:[...], l:[...], f:[...]}, ...] — per frame
    
    JS端重建逻辑:
    - confirmed pivots: 从 CONF 数组中取 frame_idx <= curBar 的所有拐点
    - tentative: TENT[curBar]
    - 合并为完整拐点序列
    """
    n = len(df)
    
    # K线: [o, h, l, c]
    klines = []
    for i in range(n):
        klines.append([
            round(df.iloc[i]['open'], 5),
            round(df.iloc[i]['high'], 5),
            round(df.iloc[i]['low'], 5),
            round(df.iloc[i]['close'], 5),
        ])
    
    # === 预测增量编码 ===
    pred_changes = {}
    prev_pred_key = None
    
    for i in range(n):
        preds = frame_preds[i]
        pred_key = json.dumps(preds, separators=(',', ':'))
        if pred_key != prev_pred_key:
            pred_changes[str(i)] = preds
            prev_pred_key = pred_key
    
    # === Tentative增量编码 ===
    tent_changes = {}
    prev_tent = None
    for i in range(n):
        t = frame_tentative[i]
        if t != prev_tent:
            tent_changes[str(i)] = t
            prev_tent = t
    
    # 统计
    conf_json = json.dumps(confirmed_events, separators=(',', ':'))
    tent_json = json.dumps(tent_changes, separators=(',', ':'))
    pred_json = json.dumps(pred_changes, separators=(',', ':'))
    klines_json = json.dumps(klines, separators=(',', ':'))
    
    print(f"  增量编码: {len(confirmed_events)} confirmed pivots, "
          f"{len(tent_changes)} tent变化帧, {len(pred_changes)} pred变化帧 / {n}总帧")
    print(f"  数据大小: K={len(klines_json)//1024}KB, CONF={len(conf_json)//1024}KB, "
          f"TENT={len(tent_json)//1024}KB, PRED={len(pred_json)//1024}KB")
    
    dt_start = str(df['datetime'].iloc[0])
    dt_end = str(df['datetime'].iloc[-1])
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>动态对称预测 v4.2 | EURUSD H1</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #111118; color: #eee; font-family: 'Consolas','Courier New',monospace; padding: 6px; }}
.header {{ background: #1a1a2e; padding: 5px 12px; border-radius: 4px; margin-bottom: 3px; font-size: 12px; border-left: 3px solid #58a6ff; }}
.controls {{ display: flex; align-items: center; gap: 5px; margin: 3px 0; flex-wrap: wrap; }}
.btn {{ background: #2a2a4a; color: #cdf; border: 1px solid #445; padding: 4px 10px; border-radius: 3px; cursor: pointer; font-size: 12px; font-family: inherit; user-select: none; }}
.btn:hover {{ background: #3a3a6a; }}
.slider-box {{ display: inline-flex; align-items: center; gap: 4px; font-size: 12px; color: #aaa; }}
.slider-box input[type=range] {{ width: 500px; }}
.grid {{ display: grid; grid-template-columns: 1fr; grid-template-rows: 1fr 1fr 1fr; gap: 3px; height: calc(100vh - 85px); }}
.panel {{ background: #14141f; border: 1px solid #2a2a3f; border-radius: 3px; position: relative; overflow: hidden; }}
.panel-title {{ position: absolute; top: 3px; left: 8px; font-size: 12px; z-index: 10; font-weight: bold; }}
canvas {{ width: 100%; height: 100%; display: block; }}
#infoBar {{ font-size: 11px; color: #999; height: 16px; margin: 1px 4px; }}
</style></head><body>

<div class="header">
EURUSD H1 | {dt_start} ~ {dt_end} | {n} bars | 窗口150K+50K空白 | 方向键←→ / 空格播放
</div>
<div class="controls">
  <button class="btn" onclick="stepTo(50)">|&lt;</button>
  <button class="btn" onclick="stepBy(-1)">&lt;</button>
  <button class="btn" id="playBtn" onclick="togglePlay()">&#9654; Play</button>
  <button class="btn" onclick="stepBy(1)">&gt;</button>
  <button class="btn" onclick="stepTo(N-1)">&gt;|</button>
  <span class="slider-box">
    <input type="range" id="barSlider" min="50" max="{n-1}" value="50"
         oninput="stepTo(parseInt(this.value))">
    <span id="barNum" style="min-width:40px">50</span>/<span>{n-1}</span>
  </span>
  <span class="slider-box" style="color:#666;">
    Speed: <input type="range" id="speedSlider" min="1" max="10" value="3" style="width:80px"
           oninput="document.getElementById('speedVal').textContent=this.value">
    <span id="speedVal">3</span>
  </span>
</div>
<div id="infoBar"></div>

<div class="grid">
  <div class="panel"><div class="panel-title" style="color:#ffcc44;">&#9650; 幅度归并 (Amp: base+amp segments)</div><canvas id="c0"></canvas></div>
  <div class="panel"><div class="panel-title" style="color:#55ddff;">&#9632; 时间归并 (Lat: lateral segments)</div><canvas id="c1"></canvas></div>
  <div class="panel"><div class="panel-title" style="color:#cc88ff;">&#9830; 综合 (All segments)</div><canvas id="c2"></canvas></div>
</div>

<script>
// === 数据 ===
const K = {klines_json};
const N = K.length;
const WIN = 150;
const FUTURE = 50;
const TOTAL_W = WIN + FUTURE;

// Confirmed pivots: [[frame_idx, bar, price, dir], ...]
// 按frame_idx排序 (append-only, 已排序)
const CONF = {conf_json};

// Tentative变化帧: {{ frame_idx_str: [bar,price,dir] | null }}
const TENT_CHG = {tent_json};

// 预测变化帧: {{ frame_idx_str: {{a:[...], l:[...], f:[...]}} }}
const PRED_CHG = {pred_json};

// 构建排序后的变化帧索引
const tentKeys = Object.keys(TENT_CHG).map(Number).sort((a,b)=>a-b);
const predKeys = Object.keys(PRED_CHG).map(Number).sort((a,b)=>a-b);

// 二分查找: 找 <= target 的最大 key
function findLE(keys, target) {{
    let lo = 0, hi = keys.length - 1, res = -1;
    while(lo <= hi) {{
        const mid = (lo + hi) >> 1;
        if(keys[mid] <= target) {{ res = mid; lo = mid + 1; }}
        else hi = mid - 1;
    }}
    return res >= 0 ? keys[res] : -1;
}}

// 获取bar对应的完整拐点序列 (confirmed + tentative)
function getPivots(bar) {{
    // confirmed pivots: 所有 frame_idx <= bar 的拐点
    // CONF已按frame_idx排序, 用二分找上界
    let hi = CONF.length;
    let lo = 0;
    while(lo < hi) {{
        const mid = (lo + hi) >> 1;
        if(CONF[mid][0] <= bar) lo = mid + 1;
        else hi = mid;
    }}
    // lo = 第一个 frame_idx > bar 的位置
    const pvts = [];
    for(let i = 0; i < lo; i++) {{
        pvts.push([CONF[i][1], CONF[i][2], CONF[i][3]]);
    }}
    // tentative
    const tk = findLE(tentKeys, bar);
    if(tk >= 0) {{
        const t = TENT_CHG[tk];
        if(t) pvts.push(t);
    }}
    return pvts;
}}

function getPreds(bar) {{
    const k = findLE(predKeys, bar);
    return k >= 0 ? PRED_CHG[k] : {{a:[], l:[], f:[]}};
}}

let curBar = 50;
let playing = false;
let playTimer = null;

function drawPanel(cid, preds, pvts, zigColor, predColorM, predColorC) {{
    const cv = document.getElementById(cid);
    const rect = cv.parentElement.getBoundingClientRect();
    cv.width = rect.width;
    cv.height = rect.height;
    const cx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    const mg = {{l: 56, r: 6, t: 20, b: 12}};
    const pw = W - mg.l - mg.r;
    const ph = H - mg.t - mg.b;
    
    const winStart = Math.max(0, curBar - WIN + 1);
    
    // 价格范围 (可见K线 + 预测)
    let pMin = Infinity, pMax = -Infinity;
    for(let b = winStart; b <= curBar && b < N; b++) {{
        pMin = Math.min(pMin, K[b][2]);
        pMax = Math.max(pMax, K[b][1]);
    }}
    for(const p of preds) {{
        pMin = Math.min(pMin, p.ptp, p.aps, p.ape, p.psp);
        pMax = Math.max(pMax, p.ptp, p.aps, p.ape, p.psp);
    }}
    if(pMin === Infinity) {{ pMin = 1.0; pMax = 1.1; }}
    const r = pMax - pMin;
    pMin -= r * 0.04; pMax += r * 0.04;
    
    function xS(bar) {{ return mg.l + ((bar - winStart) / TOTAL_W) * pw; }}
    function yS(price) {{ return mg.t + ph - ((price - pMin) / (pMax - pMin)) * ph; }}
    
    // Background
    cx.fillStyle = '#14141f';
    cx.fillRect(0, 0, W, H);
    
    // Future zone
    const fX = xS(curBar + 1);
    cx.fillStyle = '#1a1a28';
    cx.fillRect(fX, mg.t, W - mg.r - fX, ph);
    
    // Current bar line
    cx.strokeStyle = '#ffffff15';
    cx.lineWidth = 1;
    cx.beginPath(); cx.moveTo(xS(curBar), mg.t); cx.lineTo(xS(curBar), H - mg.b); cx.stroke();
    
    // Grid
    cx.fillStyle = '#555'; cx.font = '10px monospace';
    cx.strokeStyle = '#1c1c30'; cx.lineWidth = 0.5;
    for(let i = 0; i <= 6; i++) {{
        const p = pMin + (pMax - pMin) * i / 6;
        const y = yS(p);
        cx.beginPath(); cx.moveTo(mg.l, y); cx.lineTo(W - mg.r, y); cx.stroke();
        cx.fillText(p.toFixed(4), 2, y + 3);
    }}
    
    // K-lines (明亮)
    for(let b = winStart; b <= curBar && b < N; b++) {{
        const k = K[b];
        const x = xS(b);
        const up = k[3] >= k[0];
        // wick
        cx.strokeStyle = up ? '#33ee55' : '#ff4455';
        cx.lineWidth = 1;
        cx.beginPath(); cx.moveTo(x, yS(k[2])); cx.lineTo(x, yS(k[1])); cx.stroke();
        // body
        const bw = Math.max(1.5, pw / TOTAL_W * 0.7);
        const oY = yS(Math.max(k[0], k[3]));
        const cY = yS(Math.min(k[0], k[3]));
        cx.fillStyle = up ? '#22bb4488' : '#ee334488';
        cx.fillRect(x - bw/2, oY, bw, Math.max(1, cY - oY));
    }}
    
    // ZG拐点连线 (明亮粗线) — 三个panel共享同一ZG
    if(pvts && pvts.length >= 2) {{
        cx.strokeStyle = zigColor;
        cx.lineWidth = 2;
        cx.globalAlpha = 0.95;
        cx.setLineDash([]);
        cx.beginPath();
        let started = false;
        for(const pt of pvts) {{
            if(pt[0] > curBar) break;
            const x = xS(pt[0]), y = yS(pt[1]);
            if(x < mg.l - 20) continue;
            if(!started) {{ cx.moveTo(x, y); started = true; }}
            else cx.lineTo(x, y);
        }}
        cx.stroke();
        cx.globalAlpha = 1;
        
        // 拐点圆点
        for(const pt of pvts) {{
            if(pt[0] > curBar) break;
            const x = xS(pt[0]);
            if(x < mg.l - 5) continue;
            cx.fillStyle = pt[2] === 1 ? '#ff6666' : '#66ff66';
            cx.beginPath(); cx.arc(x, yS(pt[1]), 3, 0, Math.PI*2); cx.fill();
        }}
    }}
    
    // === 预测线 (在空白区) ===
    for(let i = 0; i < preds.length; i++) {{
        const p = preds[i];
        const isMirror = p.t === 'm';
        const alpha = Math.max(0.45, 0.95 - i * 0.035);
        const lw = Math.max(1.2, 2.8 - i * 0.1);
        
        const aCol = isMirror ? predColorM.replace('A', alpha) : predColorC.replace('A', alpha);
        const pCol = isMirror 
            ? predColorM.replace('A', Math.min(1, alpha + 0.1))
            : predColorC.replace('A', Math.min(1, alpha + 0.1));
        
        // A段 (实线, 已发生的部分)
        cx.strokeStyle = aCol; cx.lineWidth = lw; cx.setLineDash([]);
        cx.beginPath();
        cx.moveTo(xS(p.as), yS(p.aps));
        cx.lineTo(xS(p.ae), yS(p.ape));
        cx.stroke();
        
        // B段 (center预测有中间段)
        if(!isMirror && p.bs !== undefined) {{
            cx.strokeStyle = 'rgba(200,200,220,' + (alpha*0.35) + ')';
            cx.lineWidth = lw * 0.5; cx.setLineDash([3, 2]);
            cx.beginPath(); cx.moveTo(xS(p.bs), yS(p.bps)); cx.lineTo(xS(p.be), yS(p.bpe)); cx.stroke();
        }}
        
        // C' 预测线 (虚线+箭头, 亮色)
        cx.strokeStyle = pCol; cx.lineWidth = lw * 1.4; cx.setLineDash([7, 3]);
        cx.beginPath(); cx.moveTo(xS(p.psb), yS(p.psp)); cx.lineTo(xS(p.ptb), yS(p.ptp)); cx.stroke();
        
        // 箭头
        const ax1 = xS(p.psb), ay1 = yS(p.psp);
        const ax2 = xS(p.ptb), ay2 = yS(p.ptp);
        const angle = Math.atan2(ay2-ay1, ax2-ax1);
        cx.setLineDash([]); cx.lineWidth = lw;
        cx.beginPath();
        cx.moveTo(ax2, ay2);
        cx.lineTo(ax2 - 7*Math.cos(angle-0.35), ay2 - 7*Math.sin(angle-0.35));
        cx.moveTo(ax2, ay2);
        cx.lineTo(ax2 - 7*Math.cos(angle+0.35), ay2 - 7*Math.sin(angle+0.35));
        cx.stroke();
        
        // 目标价格标签
        if(i < 10) {{
            cx.fillStyle = pCol; cx.font = '10px monospace'; cx.textAlign = 'left';
            cx.fillText(p.ptp.toFixed(4), ax2 + 5, ay2 + 3);
        }}
    }}
    cx.setLineDash([]); cx.globalAlpha = 1; cx.textAlign = 'start';
}}

function drawAll() {{
    const pvts = getPivots(curBar);
    const fd = getPreds(curBar);
    drawPanel('c0', fd.a||[], pvts, '#ffd740', 'rgba(255,200,60,A)', 'rgba(120,230,255,A)');
    drawPanel('c1', fd.l||[], pvts, '#40e0ff', 'rgba(255,180,80,A)', 'rgba(80,210,255,A)');
    drawPanel('c2', fd.f||[], pvts, '#c080ff', 'rgba(255,160,100,A)', 'rgba(160,120,255,A)');
    
    document.getElementById('barNum').textContent = curBar;
    document.getElementById('barSlider').value = curBar;
    
    if(curBar < N) {{
        const k = K[curBar];
        const na = (fd.a||[]).length, nl = (fd.l||[]).length, nf = (fd.f||[]).length;
        document.getElementById('infoBar').textContent =
            `Bar ${{curBar}} | O:${{k[0]}} H:${{k[1]}} L:${{k[2]}} C:${{k[3]}} | ` +
            `Amp:${{na}} | Lat:${{nl}} | All:${{nf}} preds`;
    }}
}}

function stepTo(bar) {{ curBar = Math.max(50, Math.min(N-1, bar)); drawAll(); }}
function stepBy(d) {{ stepTo(curBar + d); }}

function togglePlay() {{
    playing = !playing;
    document.getElementById('playBtn').innerHTML = playing ? '&#9646;&#9646;' : '&#9654; Play';
    if(playing) {{
        playTimer = setInterval(() => {{
            const speed = parseInt(document.getElementById('speedSlider').value);
            stepBy(speed);
            if(curBar >= N - 1) {{
                playing = false; clearInterval(playTimer);
                document.getElementById('playBtn').innerHTML = '&#9654; Play';
            }}
        }}, 60);
    }} else {{ clearInterval(playTimer); }}
}}

document.addEventListener('keydown', e => {{
    if(e.key === 'ArrowRight') stepBy(1);
    else if(e.key === 'ArrowLeft') stepBy(-1);
    else if(e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
}});

window.addEventListener('resize', drawAll);
drawAll();
</script></body></html>"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    fsize = len(html)
    print(f"Saved: {output_path} ({fsize//1024}KB = {fsize/1024/1024:.1f}MB)")


def main():
    print("=" * 70)
    print("动态对称预测 v4.2")
    print("=" * 70)
    
    df = load_kline("/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv", limit=2000)
    print(f"数据: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]} ({len(df)} bars)")
    
    print(f"\n逐K线推进 (DynamicZG + DynamicMerger, 跳过pool_fusion)...")
    confirmed_events, frame_tentative, frame_preds = run_engine(
        df, start_pred=50, max_preds=15, pred_horizon=50,
    )
    
    print(f"\n生成HTML...")
    generate_html(df, confirmed_events, frame_tentative, frame_preds,
                  "/home/ubuntu/stage2_abc/merge_v4.html")


if __name__ == '__main__':
    main()
