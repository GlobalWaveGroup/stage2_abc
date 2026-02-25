#!/usr/bin/env python3
"""
归并引擎v3.0可视化 — 含波段池三波融合
- 快照(base/amp/lat): 完整zigzag连线
- extra_segments: 滑窗收集的被贪心跳过的线段（虚线）
- fusion线段: 波段池内三波归并产出的新线段（紫色系）
- Step+/- 可逐步回放归并过程
- Fusion Top N 滑块控制显示数量
- Top20 重要拐点标记
"""

import json
import sys
sys.path.insert(0, '/home/ubuntu/stage2_abc')
from merge_engine_v3 import *


def generate_html(df, results, output_path, pivot_info=None, fusion_segs=None, sym_structures=None):
    snapshots = results['all_snapshots']
    extra_segs = results.get('extra_segments', [])

    # 快照数据
    snap_data = []
    for snap_type, label, pvts in snapshots:
        points = [{'bar': int(p[0]), 'y': round(p[1], 5), 'dir': int(p[2])} for p in pvts]
        snap_data.append({'type': snap_type, 'label': label, 'n': len(pvts), 'pts': points})

    # extra线段按来源label分组
    extra_by_label = {}
    for p_start, p_end, label in extra_segs:
        if label not in extra_by_label:
            extra_by_label[label] = []
        extra_by_label[label].append({
            'b1': int(p_start[0]), 'p1': round(p_start[1], 5),
            'b2': int(p_end[0]),   'p2': round(p_end[1], 5),
        })
    extra_groups = []
    for label in sorted(extra_by_label.keys()):
        segs = extra_by_label[label]
        src = 'amp' if (label.startswith('A') and '_mid' not in label) else 'lat'
        extra_groups.append({'label': label, 'src': src, 'segs': segs})

    # Fusion 线段（按重要性排序，已经排好了）
    fusion_data = []
    if fusion_segs:
        # 按重要性排序
        sorted_fusion = sorted(fusion_segs, key=lambda s: -s['importance'])
        for s in sorted_fusion:
            via = s.get('fusion_via', (0, 0))
            fusion_data.append({
                'b1': s['bar_start'], 'p1': round(s['price_start'], 5),
                'd1': s['dir_start'],
                'b2': s['bar_end'],   'p2': round(s['price_end'], 5),
                'd2': s['dir_end'],
                'imp': round(s['importance'], 4),
                'span': s['span'],
                'amp': round(s['amplitude'], 5),
                'src': s['source_label'],
                'via': [via[0], via[1]] if via else [0, 0],
            })

    # Symmetry structures
    sym_data = []
    if sym_structures:
        for s in sym_structures:
            sym_data.append({
                'p1': s['p1'], 'p2': s['p2'], 'p3': s['p3'], 'p4': s['p4'],
                'pp1': round(s['price_p1'], 5), 'pp2': round(s['price_p2'], 5),
                'pp3': round(s['price_p3'], 5), 'pp4': round(s['price_p4'], 5),
                'score': s['score'], 'sym': s['sym_score'],
                'imp': s['endpoint_imp'],
                'va': s['vec']['amp'], 'vt': s['vec']['time'],
                'vm': s['vec']['mod'], 'vs': s['vec']['slope'],
                'vc': s['vec']['complexity'],
                'aA': s['amp_A'], 'aB': s['amp_B'], 'aC': s['amp_C'],
                'tA': s['time_A'], 'tB': s['time_B'], 'tC': s['time_C'],
                'dir': s['dir'], 'type': s['type'],
            })
    n_sym = len(sym_data)

    # K线
    kline_data = []
    for i in range(len(df)):
        kline_data.append({
            'o': round(df.iloc[i]['open'], 5),
            'h': round(df.iloc[i]['high'], 5),
            'l': round(df.iloc[i]['low'], 5),
            'c': round(df.iloc[i]['close'], 5),
        })

    # 所有拐点按重要性排序（峰和谷分别排序），传入JS端由滑块控制显示数量
    all_peaks = []
    all_valleys = []
    if pivot_info:
        peaks = sorted([v for v in pivot_info.values() if v['dir'] == 1], key=lambda x: -x['importance'])
        valleys = sorted([v for v in pivot_info.values() if v['dir'] == -1], key=lambda x: -x['importance'])
        for rank, p in enumerate(peaks):
            all_peaks.append({
                'bar': p['bar'], 'price': round(p['price'], 5), 'dir': 1,
                'rank': rank + 1, 'imp': round(p['importance'], 4),
                'label': f"H{rank+1}",
            })
        for rank, p in enumerate(valleys):
            all_valleys.append({
                'bar': p['bar'], 'price': round(p['price'], 5), 'dir': -1,
                'rank': rank + 1, 'imp': round(p['importance'], 4),
                'label': f"L{rank+1}",
            })
    top_marks = all_peaks + all_valleys
    n_peaks = len(all_peaks)
    n_valleys = len(all_valleys)

    n_snap = len(snap_data)
    n_amp = sum(1 for s in snap_data if s['type'] == 'amp')
    n_lat = sum(1 for s in snap_data if s['type'] == 'lat')
    n_extra = sum(len(g['segs']) for g in extra_groups)
    n_fusion = len(fusion_data)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>归并引擎 v3.1 + 波段池融合</title>
<style>
* {{ box-sizing: border-box; }}
body {{ background: #0a0a1a; color: #ddd; font-family: 'Consolas', monospace; margin: 0; padding: 10px; }}
h2 {{ color: #7eb8da; margin: 5px 0; font-size: 16px; }}
.info {{ background: #12122a; padding: 8px 12px; margin: 5px 0; border-radius: 4px; font-size: 13px; border-left: 3px solid #4488aa; }}
.section {{ display: flex; flex-wrap: wrap; gap: 2px 8px; margin: 4px 0; font-size: 12px; }}
.section label {{ cursor: pointer; padding: 2px 6px; border-radius: 3px; white-space: nowrap; }}
.section label:hover {{ background: #222244; }}
canvas {{ display: block; border: 1px solid #222; cursor: crosshair; }}
.btn {{ background: #1a2a4a; color: #8ac; border: 1px solid #335; padding: 3px 8px; border-radius: 3px; cursor: pointer; font-size: 11px; margin: 0 2px; }}
.btn:hover {{ background: #2a3a5a; }}
.btn.active {{ background: #3a2a5a; color: #f8f; border-color: #a6a; }}
.slider-box {{ display: inline-flex; align-items: center; gap: 4px; margin: 0 8px; font-size: 11px; }}
.slider-box input[type=range] {{ width: 120px; }}
</style></head><body>

<h2>归并引擎 v3.1 | EURUSD H1 | {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}</h2>
<div class="info">
{len(df)} bars | Base: {snap_data[0]['n']}pv |
Final: {len(results['final_pivots'])}pv |
Snap: {n_snap} (A:{n_amp} T:{n_lat}) |
Extra: {n_extra} segs |
Fusion: {n_fusion} new segs |
Symmetry: {n_sym} structures |
Peaks: {n_peaks} | Valleys: {n_valleys}
</div>

<div style="margin:4px 0;">
<button class="btn" onclick="showAll()">All</button>
<button class="btn" onclick="hideAll()">None</button>
<button class="btn" onclick="showType('amp')">Amp</button>
<button class="btn" onclick="showType('lat')">Lat</button>
<button class="btn" onclick="showBase()">Base</button>
<button class="btn" onclick="toggleExtra()">Extra</button>
<button class="btn" id="topBtn" onclick="toggleTop()" style="background:#2a1a4a;color:#f8f;">ImpPts</button>
<button class="btn" id="impValBtn" onclick="toggleImpVal()" style="background:#1a2a3a;color:#adf;">Values</button>
<button class="btn" id="fusionBtn" onclick="toggleFusion()" style="background:#3a1a3a;color:#c8f;">Fusion</button>
<button class="btn" id="symBtn" onclick="toggleSym()" style="background:#1a3a1a;color:#4f8;">Symmetry</button>
<button class="btn" onclick="showStep(1)">Step+</button>
<button class="btn" onclick="showStep(-1)">Step-</button>
</div>
<div style="margin:4px 0;">
<span class="slider-box" style="color:#f8f;">
  Peaks: <input type="range" id="peakSlider" min="0" max="{n_peaks}" value="{min(10, n_peaks)}" oninput="updatePeakN(this.value)">
  <span id="peakN">{min(10, n_peaks)}</span>/{n_peaks}
</span>
<span class="slider-box" style="color:#4f4;">
  Valleys: <input type="range" id="valleySlider" min="0" max="{n_valleys}" value="{min(10, n_valleys)}" oninput="updateValleyN(this.value)">
  <span id="valleyN">{min(10, n_valleys)}</span>/{n_valleys}
</span>
<span class="slider-box" style="color:#c8f;">
  Fusion Top: <input type="range" id="fusionSlider" min="0" max="{n_fusion}" value="50" oninput="updateFusionN(this.value)">
  <span id="fusionN">50</span>/{n_fusion}
</span>
<span class="slider-box" style="color:#4f8;">
  Sym Top: <input type="range" id="symSlider" min="0" max="{n_sym}" value="{min(20, n_sym)}" oninput="updateSymN(this.value)">
  <span id="symN">{min(20, n_sym)}</span>/{n_sym}
</span>
<span class="slider-box" style="color:#ffa;">
  Min imp: <input type="range" id="impSlider" min="0" max="100" value="0" oninput="updateMinImp(this.value)">
  <span id="impVal">0.00</span>
</span>
</div>

<div style="display:flex; gap:15px; flex-wrap:wrap; margin:6px 0;">
<div>
<div style="color:#fc8; font-size:11px; font-weight:bold;">Snapshots</div>
<div class="section" id="snap_ctl"></div>
</div>
<div>
<div style="color:#9e9; font-size:11px; font-weight:bold;">Extra (滑窗)</div>
<div class="section" id="extra_ctl"></div>
</div>
</div>

<div style="font-size:12px; color:#aaa; margin:2px 0;">
Step: <span id="cursor_info">all</span>
</div>

<canvas id="chart" width="1800" height="700"></canvas>
<div id="info" style="font-size:12px; color:#888; height:16px; margin:2px 0;"></div>

<script>
const K = {json.dumps(kline_data)};
const S = {json.dumps(snap_data)};
const EX = {json.dumps(extra_groups)};
const TOP = {json.dumps(top_marks)};
const FUS = {json.dumps(fusion_data)};
const SYM = {json.dumps(sym_data)};
// Separate peaks and valleys from TOP array
const PEAKS = TOP.filter(m => m.dir === 1);
const VALS = TOP.filter(m => m.dir === -1);
let showTop = true;
let showImpVal = true;
let peakTopN = Math.min(10, PEAKS.length);
let valleyTopN = Math.min(10, VALS.length);
let showFusion = true;
let fusionTopN = Math.min(50, FUS.length);
let showSym = true;
let symTopN = Math.min(20, SYM.length);
let minImp = 0;

function snapColor(i) {{
    const s = S[i];
    if(s.type === 'base') return '#FFD700';
    if(s.type === 'amp') {{
        let ai = 0; for(let j=0;j<i;j++) if(S[j].type==='amp') ai++;
        const t = ai/Math.max(S.filter(x=>x.type==='amp').length-1,1);
        return `rgb(${{Math.round(255-t*80)}},${{Math.round(100-t*60)}},30)`;
    }}
    if(s.type === 'lat') {{
        let li = 0; for(let j=0;j<i;j++) if(S[j].type==='lat') li++;
        const t = li/Math.max(S.filter(x=>x.type==='lat').length-1,1);
        return `rgb(${{Math.round(50+t*50)}},${{Math.round(220-t*80)}},${{Math.round(255-t*40)}})`;
    }}
    return '#888';
}}

function extraColor(gi) {{
    const g = EX[gi];
    if(g.label.includes('_mid')) {{
        const t = gi / Math.max(EX.length-1, 1);
        return `rgb(${{Math.round(80+t*40)}},${{Math.round(220-t*60)}},${{Math.round(120+t*40)}})`;
    }} else {{
        const t = gi / Math.max(EX.length-1, 1);
        return `rgb(${{Math.round(255-t*40)}},${{Math.round(160-t*60)}},${{Math.round(60)}})`;
    }}
}}

function fusionColor(rank, total) {{
    // 紫色系: 高重要性=亮紫, 低重要性=暗紫
    const t = rank / Math.max(total-1, 1);
    const r = Math.round(200 - t*100);
    const g = Math.round(80 - t*50);
    const b = Math.round(255 - t*80);
    return `rgb(${{r}},${{g}},${{b}})`;
}}

const vis = S.map(()=>true);
const exVis = EX.map(()=>true);
let stepCursor = S.length;

function showAll(){{ vis.fill(true); exVis.fill(true); showFusion=true; stepCursor=S.length; sync(); draw(); updateCursor(); }}
function hideAll(){{ vis.fill(false); exVis.fill(false); showFusion=false; stepCursor=0; sync(); draw(); updateCursor(); }}
function showType(t){{ vis.fill(false); exVis.fill(false); showFusion=false; S.forEach((s,i)=>vis[i]=s.type===t); sync(); draw(); }}
function showBase(){{ vis.fill(false); exVis.fill(false); showFusion=false; vis[0]=true; sync(); draw(); }}
function toggleExtra(){{ const on=exVis.some(v=>v); exVis.fill(!on); sync(); draw(); }}
function toggleTop(){{ showTop=!showTop; document.getElementById('topBtn').classList.toggle('active',showTop); draw(); }}
function toggleImpVal(){{ showImpVal=!showImpVal; document.getElementById('impValBtn').classList.toggle('active',showImpVal); draw(); }}
function toggleFusion(){{ showFusion=!showFusion; document.getElementById('fusionBtn').classList.toggle('active',showFusion); draw(); }}
function toggleSym(){{ showSym=!showSym; document.getElementById('symBtn').classList.toggle('active',showSym); draw(); }}

function updatePeakN(v) {{
    peakTopN = parseInt(v);
    document.getElementById('peakN').textContent = peakTopN;
    draw();
}}
function updateValleyN(v) {{
    valleyTopN = parseInt(v);
    document.getElementById('valleyN').textContent = valleyTopN;
    draw();
}}
function updateFusionN(v) {{
    fusionTopN = parseInt(v);
    document.getElementById('fusionN').textContent = fusionTopN;
    draw();
}}
function updateSymN(v) {{
    symTopN = parseInt(v);
    document.getElementById('symN').textContent = symTopN;
    draw();
}}
function updateMinImp(v) {{
    minImp = v / 100.0;
    document.getElementById('impVal').textContent = minImp.toFixed(2);
    draw();
}}

function showStep(d){{
    stepCursor=Math.max(0,Math.min(S.length,stepCursor+d));
    S.forEach((s,i)=>vis[i]=i<stepCursor);
    if(stepCursor>0){{
        const curLabel = S[stepCursor-1].label;
        EX.forEach((g,i)=>{{
            exVis[i] = false;
            for(let si=0; si<stepCursor; si++){{
                if(g.label===S[si].label || g.label===S[si].label+'_mid') exVis[i]=true;
            }}
        }});
    }} else {{ exVis.fill(false); }}
    sync(); draw(); updateCursor();
}}

function updateCursor(){{
    if(stepCursor>=S.length) document.getElementById('cursor_info').textContent='all';
    else if(stepCursor<=0) document.getElementById('cursor_info').textContent='none';
    else{{ const s=S[stepCursor-1]; document.getElementById('cursor_info').textContent=`${{stepCursor}}/${{S.length}} ${{s.label}}(${{s.n}}pv)`; }}
}}

function sync(){{
    vis.forEach((v,i)=>{{document.getElementById('s'+i).checked=v;}});
    exVis.forEach((v,i)=>{{document.getElementById('e'+i).checked=v;}});
}}

// Build snap controls
const ctl = document.getElementById('snap_ctl');
S.forEach((s,i)=>{{
    const lb=document.createElement('label');
    const cb=document.createElement('input');
    cb.type='checkbox'; cb.checked=true; cb.id='s'+i;
    cb.onchange=()=>{{vis[i]=cb.checked;draw();}};
    lb.appendChild(cb); lb.style.color=snapColor(i);
    lb.appendChild(document.createTextNode(` ${{s.label}}(${{s.n}})`));
    ctl.appendChild(lb);
}});

// Build extra controls
const ectl = document.getElementById('extra_ctl');
EX.forEach((g,i)=>{{
    const lb=document.createElement('label');
    const cb=document.createElement('input');
    cb.type='checkbox'; cb.checked=true; cb.id='e'+i;
    cb.onchange=()=>{{exVis[i]=cb.checked;draw();}};
    lb.appendChild(cb); lb.style.color=extraColor(i);
    lb.appendChild(document.createTextNode(` ${{g.label}}(${{g.segs.length}})`));
    ectl.appendChild(lb);
}});

const cv=document.getElementById('chart');
const cx=cv.getContext('2d');
const W=cv.width, H=cv.height;
const mg={{l:70,r:20,t:15,b:30}};
const pw=W-mg.l-mg.r, ph=H-mg.t-mg.b;

let mn=Infinity, mx=-Infinity;
for(const k of K){{ mn=Math.min(mn,k.l); mx=Math.max(mx,k.h); }}
const pr=mx-mn; mn-=pr*0.03; mx+=pr*0.03;

function xS(b){{ return mg.l+(b/K.length)*pw; }}
function yS(p){{ return mg.t+ph-((p-mn)/(mx-mn))*ph; }}

function draw(){{
    cx.fillStyle='#0a0a1a'; cx.fillRect(0,0,W,H);

    // Grid
    cx.strokeStyle='#181830'; cx.lineWidth=0.5;
    cx.fillStyle='#555'; cx.font='10px monospace';
    for(let i=0;i<=10;i++){{
        const p=mn+(mx-mn)*i/10, y=yS(p);
        cx.beginPath(); cx.moveTo(mg.l,y); cx.lineTo(W-mg.r,y); cx.stroke();
        cx.fillText(p.toFixed(4),5,y+3);
    }}

    // K-lines
    for(let i=0;i<K.length;i++){{
        const k=K[i],x=xS(i);
        cx.strokeStyle=k.c>=k.o?'#1a3a1a':'#3a1a1a'; cx.lineWidth=0.6;
        cx.beginPath(); cx.moveTo(x,yS(k.l)); cx.lineTo(x,yS(k.h)); cx.stroke();
    }}

    // === Fusion segments (draw first, behind everything) ===
    if(showFusion && FUS.length > 0) {{
        const n = Math.min(fusionTopN, FUS.length);
        for(let i=n-1; i>=0; i--) {{
            const f = FUS[i];
            if(f.imp < minImp) continue;
            const color = fusionColor(i, n);
            const t = i / Math.max(n-1, 1);
            cx.strokeStyle = color;
            cx.lineWidth = Math.max(0.3, 2.5 - t*2.0);
            cx.globalAlpha = Math.max(0.15, 0.7 - t*0.5);
            cx.setLineDash([10, 4]);
            cx.beginPath();
            cx.moveTo(xS(f.b1), yS(f.p1));
            cx.lineTo(xS(f.b2), yS(f.p2));
            cx.stroke();
        }}
        cx.setLineDash([]); cx.globalAlpha=1;
    }}

    // === Symmetry structures (three-wave A-B-C) ===
    if(showSym && SYM.length > 0) {{
        const n = Math.min(symTopN, SYM.length);
        for(let i=n-1; i>=0; i--) {{
            const s = SYM[i];
            const t = i / Math.max(n-1, 1);
            const alpha = Math.max(0.2, 0.85 - t*0.6);
            const lw = Math.max(1.0, 3.5 - t*2.5);
            
            // A段: p1→p2 (绿色系)
            const gA = s.dir === 1 ? `rgba(80,255,120,${{alpha}})` : `rgba(255,120,80,${{alpha}})`;
            cx.strokeStyle = gA;
            cx.lineWidth = lw;
            cx.setLineDash([]);
            cx.beginPath();
            cx.moveTo(xS(s.p1), yS(s.pp1));
            cx.lineTo(xS(s.p2), yS(s.pp2));
            cx.stroke();
            
            // B段: p2→p3 (灰色, 虚线)
            cx.strokeStyle = `rgba(180,180,180,${{alpha*0.6}})`;
            cx.lineWidth = lw * 0.6;
            cx.setLineDash([4, 3]);
            cx.beginPath();
            cx.moveTo(xS(s.p2), yS(s.pp2));
            cx.lineTo(xS(s.p3), yS(s.pp3));
            cx.stroke();
            
            // C段: p3→p4 (与A同色, 虚线表示"对称预期")
            cx.strokeStyle = gA;
            cx.lineWidth = lw;
            cx.setLineDash([6, 3]);
            cx.beginPath();
            cx.moveTo(xS(s.p3), yS(s.pp3));
            cx.lineTo(xS(s.p4), yS(s.pp4));
            cx.stroke();
            
            // 对称轴标记 (B段中点的竖线)
            if(i < 10) {{
                const midBar = (s.p2 + s.p3) / 2;
                const midPrice = (s.pp2 + s.pp3) / 2;
                cx.strokeStyle = `rgba(255,255,100,${{alpha*0.3}})`;
                cx.lineWidth = 0.5;
                cx.setLineDash([2, 2]);
                cx.beginPath();
                cx.moveTo(xS(midBar), yS(s.pp2));
                cx.lineTo(xS(midBar), yS(s.pp3));
                cx.stroke();
                
                // Score label
                cx.fillStyle = `rgba(200,255,100,${{alpha}})`;
                cx.font = '9px monospace';
                cx.textAlign = 'center';
                cx.fillText(`S${{i+1}} ${{s.sym.toFixed(2)}}`, xS(midBar), yS(midPrice) - 5);
                cx.textAlign = 'start';
            }}
        }}
        cx.setLineDash([]); cx.globalAlpha=1;
    }}

    // === Extra segments ===
    for(let gi=0; gi<EX.length; gi++){{
        if(!exVis[gi]) continue;
        const g=EX[gi];
        const color=extraColor(gi);
        const isLat=g.label.includes('_mid');

        cx.strokeStyle=color;
        cx.lineWidth=isLat?1.0:0.8;
        cx.globalAlpha=0.5;
        cx.setLineDash(isLat?[4,3]:[6,2]);

        for(const seg of g.segs){{
            cx.beginPath();
            cx.moveTo(xS(seg.b1),yS(seg.p1));
            cx.lineTo(xS(seg.b2),yS(seg.p2));
            cx.stroke();
        }}
        cx.setLineDash([]); cx.globalAlpha=1;
    }}

    // === Snapshots ===
    for(let si=0;si<S.length;si++){{
        if(!vis[si]) continue;
        const s=S[si], pts=s.pts;
        if(pts.length<2) continue;
        const color=snapColor(si);

        if(s.type==='base'){{
            cx.strokeStyle=color; cx.lineWidth=0.4; cx.globalAlpha=0.25; cx.setLineDash([]);
        }}else if(s.type==='amp'){{
            let ai=0; for(let j=0;j<si;j++) if(S[j].type==='amp') ai++;
            cx.strokeStyle=color;
            cx.lineWidth=Math.min(0.6+ai*0.3,3.5);
            cx.globalAlpha=Math.min(0.35+ai*0.08,0.9);
            cx.setLineDash([]);
        }}else{{
            let li=0; for(let j=0;j<si;j++) if(S[j].type==='lat') li++;
            cx.strokeStyle=color;
            cx.lineWidth=Math.min(1.5+li*0.3,3.5);
            cx.globalAlpha=0.85;
            cx.setLineDash([8+li*3,4+li*2]);
        }}

        cx.beginPath();
        cx.moveTo(xS(pts[0].bar),yS(pts[0].y));
        for(let j=1;j<pts.length;j++) cx.lineTo(xS(pts[j].bar),yS(pts[j].y));
        cx.stroke();

        // Markers
        if(pts.length<30){{
            const ms=Math.min(2+(30-pts.length)*0.1,5);
            cx.fillStyle=color;
            for(const p of pts){{
                cx.beginPath();
                if(s.type==='lat'){{
                    cx.moveTo(xS(p.bar),yS(p.y)-ms);
                    cx.lineTo(xS(p.bar)+ms,yS(p.y));
                    cx.lineTo(xS(p.bar),yS(p.y)+ms);
                    cx.lineTo(xS(p.bar)-ms,yS(p.y));
                    cx.closePath();
                }}else{{
                    cx.arc(xS(p.bar),yS(p.y),ms,0,Math.PI*2);
                }}
                cx.fill();
            }}
        }}
        cx.setLineDash([]); cx.globalAlpha=1;
    }}

    // === Important pivot markers (adjustable count) ===
    if(showTop) {{
        cx.globalAlpha = 1;
        cx.setLineDash([]);
        cx.textAlign = 'center';

        // Draw peaks (top peakTopN)
        for(let i=0; i<Math.min(peakTopN, PEAKS.length); i++) {{
            const m = PEAKS[i];
            const x = xS(m.bar);
            const y = yS(m.price);
            const color = '#FF4444';
            const yOff = -14;
            const sz = Math.max(3, 6 - i*0.3); // size decreases with rank

            // Marker circle (size by importance)
            cx.fillStyle = color;
            cx.globalAlpha = Math.max(0.4, 1.0 - i*0.03);
            cx.beginPath();
            cx.arc(x, y, sz, 0, Math.PI*2);
            cx.fill();
            cx.strokeStyle = '#fff';
            cx.lineWidth = i < 5 ? 1.2 : 0.6;
            cx.stroke();

            // Label: H1 + imp value
            cx.fillStyle = color;
            cx.font = i < 10 ? 'bold 11px monospace' : '10px monospace';
            cx.fillText(`${{m.label}}`, x, y + yOff);

            if(showImpVal) {{
                cx.font = '9px monospace';
                cx.fillStyle = '#faa';
                cx.fillText(`${{m.imp.toFixed(3)}}`, x, y + yOff - 11);
            }}

            // Price
            cx.font = '8px monospace';
            cx.fillStyle = '#888';
            cx.fillText(`${{m.price.toFixed(4)}}`, x, y + yOff - (showImpVal ? 21 : 11));

            // Vertical line
            cx.strokeStyle = color;
            cx.lineWidth = 0.5;
            cx.globalAlpha = 0.3;
            cx.beginPath();
            cx.moveTo(x, y);
            cx.lineTo(x, y + yOff * 0.6);
            cx.stroke();
            cx.globalAlpha = 1;
        }}

        // Draw valleys (top valleyTopN)
        for(let i=0; i<Math.min(valleyTopN, VALS.length); i++) {{
            const m = VALS[i];
            const x = xS(m.bar);
            const y = yS(m.price);
            const color = '#44FF44';
            const yOff = 16;
            const sz = Math.max(3, 6 - i*0.3);

            cx.fillStyle = color;
            cx.globalAlpha = Math.max(0.4, 1.0 - i*0.03);
            cx.beginPath();
            cx.arc(x, y, sz, 0, Math.PI*2);
            cx.fill();
            cx.strokeStyle = '#fff';
            cx.lineWidth = i < 5 ? 1.2 : 0.6;
            cx.stroke();

            cx.fillStyle = color;
            cx.font = i < 10 ? 'bold 11px monospace' : '10px monospace';
            cx.fillText(`${{m.label}}`, x, y + yOff);

            if(showImpVal) {{
                cx.font = '9px monospace';
                cx.fillStyle = '#afa';
                cx.fillText(`${{m.imp.toFixed(3)}}`, x, y + yOff + 11);
            }}

            cx.font = '8px monospace';
            cx.fillStyle = '#888';
            cx.fillText(`${{m.price.toFixed(4)}}`, x, y + yOff + (showImpVal ? 21 : 11));

            cx.strokeStyle = color;
            cx.lineWidth = 0.5;
            cx.globalAlpha = 0.3;
            cx.beginPath();
            cx.moveTo(x, y);
            cx.lineTo(x, y + yOff * 0.6);
            cx.stroke();
            cx.globalAlpha = 1;
        }}
        cx.textAlign = 'start';
    }}

    // === Hover info for fusion segments ===
    // (handled in mousemove)
}}

let hoveredFusion = null;

cv.addEventListener('mousemove',(e)=>{{
    const rect=cv.getBoundingClientRect();
    const mx_=e.clientX-rect.left;
    const my_=e.clientY-rect.top;
    const bar=Math.round((mx_-mg.l)/pw*K.length);

    let infoText = '';
    if(bar>=0 && bar<K.length){{
        const k=K[bar];
        infoText = `Bar ${{bar}} | O:${{k.o.toFixed(5)}} H:${{k.h.toFixed(5)}} L:${{k.l.toFixed(5)}} C:${{k.c.toFixed(5)}}`;
    }}

    // Check if hovering near an important point
    if(showTop) {{
        const allShown = PEAKS.slice(0, peakTopN).concat(VALS.slice(0, valleyTopN));
        let bestPt = null, bestPtDist = 12;
        for(const m of allShown) {{
            const px = xS(m.bar), py = yS(m.price);
            const d = Math.sqrt((mx_-px)*(mx_-px)+(my_-py)*(my_-py));
            if(d < bestPtDist) {{ bestPtDist=d; bestPt=m; }}
        }}
        if(bestPt) {{
            infoText += ` | ${{bestPt.label}} bar=${{bestPt.bar}} price=${{bestPt.price.toFixed(5)}} imp=${{bestPt.imp.toFixed(4)}}`;
        }}
    }}

    // Check if hovering near a symmetry structure
    if(showSym && SYM.length > 0) {{
        const n = Math.min(symTopN, SYM.length);
        let bestSym = null, bestSymDist = 20;
        for(let i=0; i<n; i++) {{
            const s = SYM[i];
            // Check each of the 3 segments
            const segs = [[s.p1,s.pp1,s.p2,s.pp2],[s.p2,s.pp2,s.p3,s.pp3],[s.p3,s.pp3,s.p4,s.pp4]];
            for(const seg of segs) {{
                const x1=xS(seg[0]),y1=yS(seg[1]),x2=xS(seg[2]),y2=yS(seg[3]);
                const dx=x2-x1,dy=y2-y1,len2=dx*dx+dy*dy;
                if(len2===0) continue;
                let t=((mx_-x1)*dx+(my_-y1)*dy)/len2;
                t=Math.max(0,Math.min(1,t));
                const px=x1+t*dx,py=y1+t*dy;
                const d=Math.sqrt((mx_-px)*(mx_-px)+(my_-py)*(my_-py));
                if(d<bestSymDist) {{ bestSymDist=d; bestSym=s; }}
            }}
        }}
        if(bestSym) {{
            const s = bestSym;
            const d = s.dir===1?'↑':'↓';
            infoText += ` | SYM[${{s.type}}${{d}}] ${{s.p1}}→${{s.p2}}→${{s.p3}}→${{s.p4}} score=${{s.score.toFixed(4)}} sym=${{s.sym.toFixed(3)}} | amp=${{s.va.toFixed(3)}} time=${{s.vt.toFixed(3)}} mod=${{s.vm.toFixed(3)}} slope=${{s.vs.toFixed(3)}} cplx=${{s.vc.toFixed(3)}}`;
        }}
    }}

    // Check if hovering near a fusion segment
    if(showFusion) {{
        const n = Math.min(fusionTopN, FUS.length);
        let best = null, bestDist = 15; // 15px threshold
        for(let i=0; i<n; i++) {{
            const f = FUS[i];
            if(f.imp < minImp) continue;
            const x1=xS(f.b1), y1=yS(f.p1), x2=xS(f.b2), y2=yS(f.p2);
            // Distance from point to line segment
            const dx=x2-x1, dy=y2-y1;
            const len2 = dx*dx+dy*dy;
            if(len2===0) continue;
            let t = ((mx_-x1)*dx+(my_-y1)*dy)/len2;
            t = Math.max(0, Math.min(1, t));
            const px=x1+t*dx, py=y1+t*dy;
            const d = Math.sqrt((mx_-px)*(mx_-px)+(my_-py)*(my_-py));
            if(d < bestDist) {{ bestDist=d; best=f; }}
        }}
        if(best) {{
            const d1 = best.d1===1?'H':'L';
            const d2 = best.d2===1?'H':'L';
            infoText += ` | FUSION[${{best.src}}] bar${{best.b1}}${{d1}}->bar${{best.b2}}${{d2}} span=${{best.span}} amp=${{best.amp}} imp=${{best.imp}} via(${{best.via[0]}},${{best.via[1]}})`;
        }}
    }}

    document.getElementById('info').textContent = infoText;
}});

draw();
document.getElementById('fusionBtn').classList.add('active');
document.getElementById('symBtn').classList.add('active');
document.getElementById('topBtn').classList.add('active');
document.getElementById('impValBtn').classList.add('active');
</script></body></html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Saved: {output_path} ({len(html)//1024}KB)")


def main():
    df = load_kline("/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv", limit=200)
    base = calculate_base_zg(df['high'].values, df['low'].values)
    results = full_merge_engine(base)
    pivot_info = compute_pivot_importance(results, total_bars=len(df))
    pool = build_segment_pool(results, pivot_info)
    full_pool, fusion_segs, fusion_log = pool_fusion(pool, pivot_info)

    print(f"Initial pool: {len(pool)}")
    print(f"After fusion: {len(full_pool)} (+{len(fusion_segs)} fusion)")
    for entry in fusion_log:
        print(f"  {entry}")

    # 对称结构识别
    sym_structures = find_symmetric_structures(full_pool, pivot_info, df=df, top_n=200)
    print(f"Symmetry structures: {len(sym_structures)}")

    generate_html(df, results, "/home/ubuntu/stage2_abc/merge_v3.html",
                  pivot_info=pivot_info, fusion_segs=fusion_segs, sym_structures=sym_structures)


if __name__ == '__main__':
    main()
