#!/usr/bin/env python3
"""
归并引擎可视化 v3.1 — 池融合 + 冗余删除 + 动态更新

相比 V4 的改进:
  V4: 只用 DynamicMerger 的原始线段 (L0级别), 跳过 pool_fusion
  V3.1: 每帧完整流程:
    DynamicMerger → build_pool → remove_redundancy → limit_pool → pool_fusion → predict

新增:
  1. 冗余删除: 移除重叠/子集线段
  2. 池动态更新: 保持pool不超过 MAX_POOL_SIZE
  3. 使用fused pool做预测 (而不是raw segments)

输出: 2000K线 动态HTML, 单K线滚动, 单panel全景
"""

import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, '/home/ubuntu/stage2_abc')
from merge_engine_v3 import load_kline, predict_symmetric_image, pool_fusion
from dynamic_engine import DynamicZG, DynamicMerger


# ============================================================
# 1. 池构建 + 冗余删除 + 大小限制
# ============================================================

def build_pivot_info(segments, current_bar):
    """从线段列表构建pivot_info, 含importance"""
    pivot_info = {}
    for seg in segments:
        for bar, price, d in [(seg['bar_start'], seg['price_start'], seg['dir_start']),
                               (seg['bar_end'], seg['price_end'], seg['dir_end'])]:
            if bar not in pivot_info:
                pivot_info[bar] = {
                    'bar': bar, 'price': price, 'dir': d,
                    'importance': 0, 'max_amp': 0, 'max_span': 0,
                    'n_connections': 0,
                }
            info = pivot_info[bar]
            info['n_connections'] += 1
            info['max_amp'] = max(info['max_amp'], seg['amplitude'])
            info['max_span'] = max(info['max_span'], seg['span'])
            # 重要性 = amplitude × span × 连接数的对数
            amp_score = seg['amplitude'] * seg['span']
            info['importance'] = max(info['importance'], amp_score)

    # 归一化
    max_imp = max((p['importance'] for p in pivot_info.values()), default=1)
    if max_imp > 0:
        for p in pivot_info.values():
            # 加入时间衰减: 越近的拐点稍微加权
            recency = 1.0 + 0.1 * (p['bar'] / max(current_bar, 1))
            p['importance'] = (p['importance'] / max_imp) * recency

    return pivot_info


def build_pool_from_segments(segments, pivot_info):
    """从DynamicMerger线段直接构建去重池"""
    seg_dict = {}

    for seg in segments:
        key = (seg['bar_start'], seg['bar_end'])
        if key in seg_dict:
            continue

        info1 = pivot_info.get(seg['bar_start'], {})
        info2 = pivot_info.get(seg['bar_end'], {})
        imp1 = info1.get('importance', 0)
        imp2 = info2.get('importance', 0)

        seg_dict[key] = {
            'bar_start': seg['bar_start'],
            'price_start': seg['price_start'],
            'dir_start': seg['dir_start'],
            'bar_end': seg['bar_end'],
            'price_end': seg['price_end'],
            'dir_end': seg['dir_end'],
            'span': seg['span'],
            'amplitude': seg['amplitude'],
            'source': seg.get('source', 'unknown'),
            'source_label': seg.get('source', 'unknown'),
            'imp_start': imp1,
            'imp_end': imp2,
            'importance': imp1 * imp2,
            'birth_bar': seg.get('birth_bar', 0),
        }

    return list(seg_dict.values())


def remove_redundancies(pool, overlap_threshold=0.8, amp_diff_threshold=0.2):
    """
    冗余删除: 移除重叠的低重要性线段.

    两条线段"冗余"当:
    1. 子集关系: A完全包含B (A.start <= B.start, A.end >= B.end), 且 A.importance > B.importance
    2. 高度重叠: 时间重叠 > overlap_threshold, 且幅度差 < amp_diff_threshold * max(amp)
    """
    if not pool:
        return pool

    # 按importance降序排列
    pool_sorted = sorted(pool, key=lambda s: -s['importance'])
    keep = []
    removed = set()

    for i, seg_a in enumerate(pool_sorted):
        key_a = (seg_a['bar_start'], seg_a['bar_end'])
        if key_a in removed:
            continue

        # 检查是否被更重要的线段覆盖
        is_redundant = False
        for seg_b in keep:
            # 子集检查: seg_a 被 seg_b 包含
            if (seg_b['bar_start'] <= seg_a['bar_start'] and
                seg_b['bar_end'] >= seg_a['bar_end'] and
                seg_b['span'] > seg_a['span']):
                # 如果被包含且重要性明显更低
                if seg_a['importance'] < seg_b['importance'] * 0.3:
                    is_redundant = True
                    break

            # 高重叠检查
            overlap_start = max(seg_a['bar_start'], seg_b['bar_start'])
            overlap_end = min(seg_a['bar_end'], seg_b['bar_end'])
            if overlap_end > overlap_start:
                overlap = overlap_end - overlap_start
                ratio_a = overlap / max(seg_a['span'], 1)
                ratio_b = overlap / max(seg_b['span'], 1)

                if (ratio_a > overlap_threshold or ratio_b > overlap_threshold):
                    # 幅度也接近
                    max_amp = max(seg_a['amplitude'], seg_b['amplitude'])
                    amp_diff = abs(seg_a['amplitude'] - seg_b['amplitude'])
                    if max_amp > 0 and amp_diff / max_amp < amp_diff_threshold:
                        is_redundant = True
                        break

        if not is_redundant:
            keep.append(seg_a)
        else:
            removed.add(key_a)

    return keep


def limit_pool_size(pool, max_size=300, protect_recent_bars=100,
                     current_bar=0):
    """
    限制池大小: 保留最重要的max_size条线段.

    但始终保留最近 protect_recent_bars 内的线段 (不管importance).
    """
    if len(pool) <= max_size:
        return pool

    # 分成两组: 最近的 + 其余
    recent = []
    others = []
    for seg in pool:
        if seg['bar_end'] >= current_bar - protect_recent_bars:
            recent.append(seg)
        else:
            others.append(seg)

    # others按importance排序, 截断
    others.sort(key=lambda s: -s['importance'])
    remaining_slots = max(0, max_size - len(recent))
    others = others[:remaining_slots]

    result = recent + others
    return result


# ============================================================
# 2. 引擎: 逐K线推进 + 池融合 + 预测
# ============================================================

def run_engine(df, start_pred=50, max_preds=20, pred_horizon=50,
               max_pool_size=300, fusion_max_rounds=3):
    """
    逐K线推进, 在新拐点时重建池+融合+预测.

    关键改进 vs V4:
    - 使用 pool_fusion 产出更高级别线段
    - 冗余删除保持池整洁
    - 预测基于fused pool (不是raw segments)
    """
    n = len(df)
    high = df['high'].values
    low = df['low'].values

    zg = DynamicZG()
    merger = DynamicMerger()

    # 增量数据
    confirmed_events = []   # [(frame_idx, bar, price, dir)]
    frame_tentative = []    # [null | [bar,price,dir]]
    frame_preds = []        # per-frame predictions
    frame_pool_stats = []   # per-frame pool stats

    # 当前预测 (复用到下一个关键帧)
    cur_preds = []
    cur_pool_stat = {'raw': 0, 'after_dedup': 0, 'after_fusion': 0, 'n_fusion_new': 0}

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
                p = ev['pivot']
                confirmed_events.append((i, p['bar'], round(p['price'], 5), p['dir']))

        # 3. Tentative
        if zg.tentative:
            frame_tentative.append((zg.tentative['bar'],
                                     round(zg.tentative['price'], 5),
                                     zg.tentative['dir']))
        else:
            frame_tentative.append(None)

        # 4. 关键帧: 新拐点 + 足够线段
        if has_new and i >= start_pred and len(merger.segments) >= 5:
            all_segs = merger.segments
            live_segs = [s for s in all_segs if s.get('birth_bar', 0) <= i]

            if len(live_segs) >= 3:
                # a. 构建pivot_info
                pivot_info = build_pivot_info(live_segs, i)

                # b. 构建池
                pool = build_pool_from_segments(live_segs, pivot_info)
                raw_count = len(pool)

                # c. 冗余删除
                pool = remove_redundancies(pool)

                # d. 池大小限制
                pool = limit_pool_size(pool, max_size=max_pool_size,
                                        current_bar=i)
                after_dedup = len(pool)

                # e. 三波融合 (limited rounds for speed)
                try:
                    fused_pool, new_segs, log = pool_fusion(
                        pool, pivot_info, max_rounds=fusion_max_rounds)
                except Exception:
                    fused_pool = pool
                    new_segs = []

                n_fusion_new = len(new_segs)

                # f. 再次限制大小 (融合可能大幅增加)
                fused_pool = limit_pool_size(fused_pool, max_size=max_pool_size * 2,
                                              current_bar=i)

                # g. 预测 (使用fused pool!)
                raw_preds = predict_symmetric_image(
                    fused_pool, pivot_info, current_bar=i,
                    max_pool_size=99999)

                # h. 筛选
                valid = []
                for p in raw_preds:
                    if p['pred_start_bar'] > i:
                        continue
                    if p['pred_target_bar'] > i + pred_horizon:
                        continue
                    if p['pred_target_bar'] <= i:
                        continue
                    valid.append(p)

                valid.sort(key=lambda p: -p['score'])
                cur_preds = valid[:max_preds]

                cur_pool_stat = {
                    'raw': raw_count,
                    'after_dedup': after_dedup,
                    'after_fusion': len(fused_pool),
                    'n_fusion_new': n_fusion_new,
                }

        # 5. 存帧
        frame_preds.append([encode_pred(p) for p in cur_preds])
        frame_pool_stats.append(cur_pool_stat.copy())

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            np_ = len(cur_preds)
            ps = cur_pool_stat
            print(f"  bar {i+1}/{n} ({elapsed:.1f}s) pivots={len(zg.pivots)} "
                  f"segs={len(merger.segments)} pool={ps['raw']}/{ps['after_dedup']}/{ps['after_fusion']} "
                  f"preds={np_}")

    elapsed = time.time() - t0
    print(f"完成: {n} bars, {elapsed:.1f}s, pivots={len(zg.pivots)}, segs={len(merger.segments)}")

    return confirmed_events, frame_tentative, frame_preds, frame_pool_stats


def encode_pred(p):
    """编码单个预测"""
    entry = {
        't': p['type'][0],  # 'm' or 'c'
        'as': p['A_start'], 'ae': p['A_end'],
        'aps': round(p['A_price_start'], 5), 'ape': round(p['A_price_end'], 5),
        'ad': p['A_dir'],
        'psb': p['pred_start_bar'], 'psp': round(p['pred_start_price'], 5),
        'ptb': p['pred_target_bar'], 'ptp': round(p['pred_target_price'], 5),
        'sc': round(p['score'], 5),
        'pt': p['pred_time'],
        'pa': round(p['pred_amp'], 5),
        'pd': p['pred_dir'],
    }
    if p['type'] == 'center':
        entry['bs'] = p['B_start']
        entry['be'] = p['B_end']
        entry['bps'] = round(p['B_price_start'], 5)
        entry['bpe'] = round(p['B_price_end'], 5)
    return entry


# ============================================================
# 3. HTML 生成
# ============================================================

def generate_html(df, confirmed_events, frame_tentative, frame_preds,
                  frame_pool_stats, output_path):
    n = len(df)

    # K线
    klines = []
    for i in range(n):
        klines.append([
            round(df.iloc[i]['open'], 5),
            round(df.iloc[i]['high'], 5),
            round(df.iloc[i]['low'], 5),
            round(df.iloc[i]['close'], 5),
        ])

    # 预测增量编码
    pred_changes = {}
    prev_key = None
    for i in range(n):
        pkey = json.dumps(frame_preds[i], separators=(',', ':'))
        if pkey != prev_key:
            pred_changes[str(i)] = frame_preds[i]
            prev_key = pkey

    # Tentative增量编码
    tent_changes = {}
    prev_tent = None
    for i in range(n):
        t = frame_tentative[i]
        if t != prev_tent:
            tent_changes[str(i)] = t
            prev_tent = t

    # Pool stats增量
    pool_changes = {}
    prev_ps = None
    for i in range(n):
        ps = frame_pool_stats[i]
        ps_key = json.dumps(ps, separators=(',', ':'))
        if ps_key != prev_ps:
            pool_changes[str(i)] = ps
            prev_ps = ps_key

    conf_json = json.dumps(confirmed_events, separators=(',', ':'))
    tent_json = json.dumps(tent_changes, separators=(',', ':'))
    pred_json = json.dumps(pred_changes, separators=(',', ':'))
    klines_json = json.dumps(klines, separators=(',', ':'))
    pool_json = json.dumps(pool_changes, separators=(',', ':'))

    print(f"  增量编码: {len(confirmed_events)} pivots, "
          f"{len(pred_changes)} pred变化帧, {len(pool_changes)} pool变化帧 / {n}总帧")
    print(f"  大小: K={len(klines_json)//1024}KB, CONF={len(conf_json)//1024}KB, "
          f"PRED={len(pred_json)//1024}KB, POOL={len(pool_json)//1024}KB")

    dt_start = str(df['datetime'].iloc[0])
    dt_end = str(df['datetime'].iloc[-1])

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>v3.1 池融合+冗余删除 | EURUSD H1</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0d0d18; color: #eee; font-family: 'Consolas','Courier New',monospace; padding: 6px; }}
.hdr {{ background: #1a1a2e; padding: 5px 12px; border-radius: 4px; margin-bottom: 3px; font-size: 12px; border-left: 3px solid #ff8844; }}
.ctl {{ display: flex; align-items: center; gap: 5px; margin: 3px 0; flex-wrap: wrap; }}
.btn {{ background: #2a2a4a; color: #cdf; border: 1px solid #445; padding: 4px 10px; border-radius: 3px; cursor: pointer; font-size: 12px; font-family: inherit; }}
.btn:hover {{ background: #3a3a6a; }}
.sl {{ display: inline-flex; align-items: center; gap: 4px; font-size: 12px; color: #aaa; }}
.sl input[type=range] {{ width: 500px; }}
canvas {{ width: 100%; height: calc(100vh - 90px); display: block; }}
#info {{ font-size: 11px; color: #999; height: 16px; margin: 1px 4px; }}
</style></head><body>

<div class="hdr">
v3.1 池融合+冗余删除 | EURUSD H1 | {dt_start} ~ {dt_end} | {n} bars | ←→ / 空格播放
</div>
<div class="ctl">
  <button class="btn" onclick="stepTo(50)">|&lt;</button>
  <button class="btn" onclick="stepBy(-1)">&lt;</button>
  <button class="btn" id="playBtn" onclick="togglePlay()">&#9654; Play</button>
  <button class="btn" onclick="stepBy(1)">&gt;</button>
  <button class="btn" onclick="stepTo(N-1)">&gt;|</button>
  <span class="sl">
    <input type="range" id="barSlider" min="50" max="{n-1}" value="50"
         oninput="stepTo(parseInt(this.value))">
    <span id="barNum" style="min-width:40px">50</span>/<span>{n-1}</span>
  </span>
  <span class="sl" style="color:#666;">
    Speed: <input type="range" id="speedSlider" min="1" max="10" value="3" style="width:80px"
           oninput="document.getElementById('speedVal').textContent=this.value">
    <span id="speedVal">3</span>
  </span>
</div>
<div id="info"></div>
<canvas id="cv"></canvas>

<script>
const K = {klines_json};
const N = K.length;
const WIN = 150;
const FUTURE = 50;
const TW = WIN + FUTURE;

const CONF = {conf_json};
const TENT_CHG = {tent_json};
const PRED_CHG = {pred_json};
const POOL_CHG = {pool_json};

const tentKeys = Object.keys(TENT_CHG).map(Number).sort((a,b)=>a-b);
const predKeys = Object.keys(PRED_CHG).map(Number).sort((a,b)=>a-b);
const poolKeys = Object.keys(POOL_CHG).map(Number).sort((a,b)=>a-b);

function findLE(keys, target) {{
    let lo=0,hi=keys.length-1,res=-1;
    while(lo<=hi) {{ const m=(lo+hi)>>1; if(keys[m]<=target){{res=m;lo=m+1;}}else hi=m-1; }}
    return res>=0 ? keys[res] : -1;
}}

function getPivots(bar) {{
    let hi=CONF.length,lo=0;
    while(lo<hi) {{ const m=(lo+hi)>>1; if(CONF[m][0]<=bar)lo=m+1;else hi=m; }}
    const pvts=[];
    for(let i=0;i<lo;i++) pvts.push([CONF[i][1],CONF[i][2],CONF[i][3]]);
    const tk=findLE(tentKeys,bar);
    if(tk>=0) {{ const t=TENT_CHG[tk]; if(t) pvts.push(t); }}
    return pvts;
}}

function getPreds(bar) {{
    const k=findLE(predKeys,bar);
    return k>=0 ? PRED_CHG[k] : [];
}}

function getPool(bar) {{
    const k=findLE(poolKeys,bar);
    return k>=0 ? POOL_CHG[k] : {{raw:0,after_dedup:0,after_fusion:0,n_fusion_new:0}};
}}

let curBar=50, playing=false, playTimer=null;

function draw() {{
    const cv=document.getElementById('cv');
    const rect=cv.parentElement.getBoundingClientRect();
    cv.width=rect.width; cv.height=window.innerHeight-90;
    const cx=cv.getContext('2d');
    const W=cv.width, H=cv.height;
    const mg={{l:56,r:6,t:20,b:12}};
    const pw=W-mg.l-mg.r, ph=H-mg.t-mg.b;

    const winStart=Math.max(0,curBar-WIN+1);
    const preds=getPreds(curBar);

    let pMin=Infinity,pMax=-Infinity;
    for(let b=winStart;b<=curBar&&b<N;b++) {{ pMin=Math.min(pMin,K[b][2]); pMax=Math.max(pMax,K[b][1]); }}
    for(const p of preds) {{ pMin=Math.min(pMin,p.ptp,p.aps,p.ape,p.psp); pMax=Math.max(pMax,p.ptp,p.aps,p.ape,p.psp); }}
    if(pMin===Infinity) {{ pMin=1.0; pMax=1.1; }}
    const r=pMax-pMin; pMin-=r*0.04; pMax+=r*0.04;

    function xS(bar) {{ return mg.l+((bar-winStart)/TW)*pw; }}
    function yS(price) {{ return mg.t+ph-((price-pMin)/(pMax-pMin))*ph; }}

    // Background
    cx.fillStyle='#0d0d18'; cx.fillRect(0,0,W,H);
    const fX=xS(curBar+1);
    cx.fillStyle='#12121f'; cx.fillRect(fX,mg.t,W-mg.r-fX,ph);
    cx.strokeStyle='#ffffff15'; cx.lineWidth=1;
    cx.beginPath(); cx.moveTo(xS(curBar),mg.t); cx.lineTo(xS(curBar),H-mg.b); cx.stroke();

    // Grid
    cx.fillStyle='#555'; cx.font='10px monospace';
    cx.strokeStyle='#1c1c30'; cx.lineWidth=0.5;
    for(let i=0;i<=6;i++) {{
        const p=pMin+(pMax-pMin)*i/6, y=yS(p);
        cx.beginPath(); cx.moveTo(mg.l,y); cx.lineTo(W-mg.r,y); cx.stroke();
        cx.fillText(p.toFixed(4),2,y+3);
    }}

    // K-lines (明亮)
    for(let b=winStart;b<=curBar&&b<N;b++) {{
        const k=K[b], x=xS(b), up=k[3]>=k[0];
        cx.strokeStyle=up?'#33ee55':'#ff4455'; cx.lineWidth=1;
        cx.beginPath(); cx.moveTo(x,yS(k[2])); cx.lineTo(x,yS(k[1])); cx.stroke();
        const bw=Math.max(1.5,pw/TW*0.7);
        const oY=yS(Math.max(k[0],k[3])), cY=yS(Math.min(k[0],k[3]));
        cx.fillStyle=up?'#22bb4488':'#ee334488';
        cx.fillRect(x-bw/2,oY,bw,Math.max(1,cY-oY));
    }}

    // ZG (所有confirmed + tentative)
    const pvts=getPivots(curBar);
    if(pvts.length>=2) {{
        cx.strokeStyle='#ff884488'; cx.lineWidth=2; cx.setLineDash([]);
        cx.beginPath();
        let started=false;
        for(const pt of pvts) {{
            if(pt[0]>curBar) break;
            const x=xS(pt[0]),y=yS(pt[1]);
            if(x<mg.l-20) continue;
            if(!started) {{ cx.moveTo(x,y); started=true; }} else cx.lineTo(x,y);
        }}
        cx.stroke();
        for(const pt of pvts) {{
            if(pt[0]>curBar) break;
            const x=xS(pt[0]);
            if(x<mg.l-5) continue;
            cx.fillStyle=pt[2]===1?'#ff6666':'#66ff66';
            cx.beginPath(); cx.arc(x,yS(pt[1]),3,0,Math.PI*2); cx.fill();
        }}
    }}

    // === Predictions (from fused pool) ===
    for(let i=0;i<preds.length;i++) {{
        const p=preds[i];
        const isMirror=p.t==='m';
        const alpha=Math.max(0.4,0.95-i*0.035);
        const lw=Math.max(1.2,2.8-i*0.1);

        const mCol=`rgba(255,160,40,${{alpha}})`;
        const cCol=`rgba(60,220,255,${{alpha}})`;
        const aCol=isMirror?mCol:cCol;
        const pCol=isMirror?`rgba(255,120,0,${{alpha+0.05}})`:`rgba(30,200,255,${{alpha+0.05}})`;

        // A段
        cx.strokeStyle=aCol; cx.lineWidth=lw; cx.setLineDash([]);
        cx.beginPath(); cx.moveTo(xS(p.as),yS(p.aps)); cx.lineTo(xS(p.ae),yS(p.ape)); cx.stroke();

        // B段 (center)
        if(!isMirror && p.bs!==undefined) {{
            cx.strokeStyle=`rgba(180,180,200,${{alpha*0.35}})`;
            cx.lineWidth=lw*0.5; cx.setLineDash([3,2]);
            cx.beginPath(); cx.moveTo(xS(p.bs),yS(p.bps)); cx.lineTo(xS(p.be),yS(p.bpe)); cx.stroke();
        }}

        // 模长边界 (top 5 predictions)
        if(p.pa>0 && p.pt>0 && i<5) {{
            const nS=Math.min(p.pt,50);
            cx.strokeStyle=pCol.replace(/[\\d.]+\\)$/,(alpha*0.2)+')');
            cx.lineWidth=0.8; cx.setLineDash([3,4]);
            for(let side of [1,-1]) {{
                cx.beginPath();
                for(let k=0;k<=nS;k++) {{
                    const t=k/p.pt;
                    const bk=p.psb+k;
                    const eP=p.psp+t*(p.ptp-p.psp);
                    const R=p.pa*Math.sqrt(t)*0.5;
                    const bndP=eP+side*R;
                    if(k===0) cx.moveTo(xS(bk),yS(bndP)); else cx.lineTo(xS(bk),yS(bndP));
                }}
                cx.stroke();
            }}
        }}

        // 期望轨迹
        if(p.pa>0 && p.pt>0 && i<8) {{
            cx.strokeStyle=pCol.replace(/[\\d.]+\\)$/,(alpha*0.3)+')');
            cx.lineWidth=1; cx.setLineDash([2,3]);
            const nS2=Math.min(p.pt,50);
            cx.beginPath();
            for(let k=0;k<=nS2;k++) {{
                const t=k/p.pt; const bk=p.psb+k;
                const eP=p.psp+t*(p.ptp-p.psp);
                if(k===0) cx.moveTo(xS(bk),yS(eP)); else cx.lineTo(xS(bk),yS(eP));
            }}
            cx.stroke();
        }}

        // C' 预测线 (虚线+箭头)
        cx.strokeStyle=pCol; cx.lineWidth=lw*1.4; cx.setLineDash([7,3]);
        cx.beginPath(); cx.moveTo(xS(p.psb),yS(p.psp)); cx.lineTo(xS(p.ptb),yS(p.ptp)); cx.stroke();

        // 箭头
        const ax2=xS(p.ptb),ay2=yS(p.ptp);
        const angle=Math.atan2(ay2-yS(p.psp),ax2-xS(p.psb));
        cx.setLineDash([]); cx.lineWidth=lw;
        cx.beginPath();
        cx.moveTo(ax2,ay2);
        cx.lineTo(ax2-7*Math.cos(angle-0.35),ay2-7*Math.sin(angle-0.35));
        cx.moveTo(ax2,ay2);
        cx.lineTo(ax2-7*Math.cos(angle+0.35),ay2-7*Math.sin(angle+0.35));
        cx.stroke();

        // 离差标注
        if(p.pa>0 && p.pt>0 && i<6) {{
            const elapsed=curBar-p.psb;
            if(elapsed>0 && elapsed<=p.pt && curBar<N) {{
                const t=elapsed/p.pt;
                const eNow=p.psp+t*(p.ptp-p.psp);
                const aCl=K[curBar][3];
                const dev=(aCl-eNow)/p.pa;
                const dCol=Math.abs(dev)<0.3?'#44ff88':Math.abs(dev)<1.0?'#ffaa33':'#ff4466';
                cx.fillStyle=dCol; cx.font='9px monospace'; cx.textAlign='right';
                cx.strokeStyle=dCol; cx.lineWidth=0.8; cx.setLineDash([1,1]);
                cx.beginPath(); cx.moveTo(xS(curBar)+2,yS(eNow)); cx.lineTo(xS(curBar)+2,yS(aCl)); cx.stroke();
                cx.fillText((dev>=0?'+':'')+dev.toFixed(2),xS(curBar)-2,yS(eNow)+(dev>0?-4:12));
            }}
        }}

        // 目标价
        if(i<12) {{
            cx.fillStyle=pCol; cx.font='10px monospace'; cx.textAlign='left';
            cx.fillText(p.ptp.toFixed(4),ax2+5,ay2+3);
        }}
    }}
    cx.setLineDash([]); cx.globalAlpha=1; cx.textAlign='start';

    // Info bar
    const ps=getPool(curBar);
    const kc=curBar<N?K[curBar]:K[N-1];
    let devStr='';
    if(preds.length>0 && preds[0].pa>0 && preds[0].pt>0) {{
        const p0=preds[0], el=curBar-p0.psb;
        if(el>0 && el<=p0.pt) {{
            const t=el/p0.pt, eN=p0.psp+t*(p0.ptp-p0.psp), dev=(kc[3]-eN)/p0.pa;
            devStr=` | Top1 dev:${{dev>=0?'+':''}}${{dev.toFixed(3)}} (${{p0.t==='m'?'M':'C'}})`;
        }}
    }}
    document.getElementById('barNum').textContent=curBar;
    document.getElementById('barSlider').value=curBar;
    document.getElementById('info').textContent=
        `Bar ${{curBar}} | O:${{kc[0]}} H:${{kc[1]}} L:${{kc[2]}} C:${{kc[3]}} | `+
        `Pool: ${{ps.raw}}→${{ps.after_dedup}}→${{ps.after_fusion}} (+${{ps.n_fusion_new}} fused) | `+
        `Preds: ${{preds.length}}${{devStr}}`;
}}

function stepTo(bar) {{ curBar=Math.max(50,Math.min(N-1,bar)); draw(); }}
function stepBy(d) {{ stepTo(curBar+d); }}

function togglePlay() {{
    playing=!playing;
    document.getElementById('playBtn').innerHTML=playing?'&#9646;&#9646;':'&#9654; Play';
    if(playing) {{
        playTimer=setInterval(()=>{{
            const speed=parseInt(document.getElementById('speedSlider').value);
            stepBy(speed);
            if(curBar>=N-1) {{ playing=false; clearInterval(playTimer);
                document.getElementById('playBtn').innerHTML='&#9654; Play'; }}
        }},60);
    }} else clearInterval(playTimer);
}}

document.addEventListener('keydown',e=>{{
    if(e.key==='ArrowRight') stepBy(1);
    else if(e.key==='ArrowLeft') stepBy(-1);
    else if(e.key===' ') {{ e.preventDefault(); togglePlay(); }}
}});

window.addEventListener('resize',draw);
draw();
</script></body></html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    fsize = len(html)
    print(f"Saved: {output_path} ({fsize//1024}KB = {fsize/1024/1024:.1f}MB)")


# ============================================================
# 4. Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bars', type=int, default=2000)
    parser.add_argument('--pool-size', type=int, default=300,
                        help='Max pool size before fusion')
    parser.add_argument('--fusion-rounds', type=int, default=3,
                        help='Max fusion rounds per frame')
    parser.add_argument('--output', type=str, default='/home/ubuntu/stage2_abc/merge_v3_1.html')
    args = parser.parse_args()

    print("=" * 70)
    print(f"归并引擎可视化 v3.1 — 池融合+冗余删除")
    print(f"  bars={args.bars}, pool_size={args.pool_size}, fusion_rounds={args.fusion_rounds}")
    print("=" * 70)

    df = load_kline("/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv", limit=args.bars)
    print(f"数据: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]} ({len(df)} bars)")

    print(f"\n逐K线推进 (DynamicZG + DynamicMerger + pool_fusion)...")
    confirmed, tentative, preds, pool_stats = run_engine(
        df, start_pred=50, max_preds=20, pred_horizon=50,
        max_pool_size=args.pool_size, fusion_max_rounds=args.fusion_rounds,
    )

    print(f"\n生成HTML...")
    generate_html(df, confirmed, tentative, preds, pool_stats, args.output)


if __name__ == '__main__':
    main()
