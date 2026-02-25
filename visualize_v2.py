#!/usr/bin/env python3
"""
归并引擎v2.1可视化
- 快照(base/amp/lat): 完整zigzag连线
- extra_segments: 滑窗收集的被贪心跳过的线段（虚线单独绘制）
- Step+/- 可逐步回放归并过程
- Extra开关单独控制
"""

import json
import sys
sys.path.insert(0, '/home/ubuntu/stage2_abc')
from merge_engine_v2 import *

def generate_html(df, results, output_path, pivot_info=None):
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
    # 转为有序列表
    extra_groups = []
    for label in sorted(extra_by_label.keys()):
        segs = extra_by_label[label]
        src = 'amp' if (label.startswith('A') and '_mid' not in label) else 'lat'
        extra_groups.append({'label': label, 'src': src, 'segs': segs})

    # K线
    kline_data = []
    for i in range(len(df)):
        kline_data.append({
            'o': round(df.iloc[i]['open'], 5),
            'h': round(df.iloc[i]['high'], 5),
            'l': round(df.iloc[i]['low'], 5),
            'c': round(df.iloc[i]['close'], 5),
        })

    # Top 20 重要拐点 (峰10 + 谷10)
    top_marks = []
    if pivot_info:
        peaks = sorted([v for v in pivot_info.values() if v['dir'] == 1], key=lambda x: -x['importance'])[:10]
        valleys = sorted([v for v in pivot_info.values() if v['dir'] == -1], key=lambda x: -x['importance'])[:10]
        for rank, p in enumerate(peaks):
            top_marks.append({
                'bar': p['bar'], 'price': round(p['price'], 5), 'dir': 1,
                'rank': rank + 1, 'imp': round(p['importance'], 3),
                'label': f"H{rank+1}",
            })
        for rank, p in enumerate(valleys):
            top_marks.append({
                'bar': p['bar'], 'price': round(p['price'], 5), 'dir': -1,
                'rank': rank + 1, 'imp': round(p['importance'], 3),
                'label': f"L{rank+1}",
            })

    n_snap = len(snap_data)
    n_amp = sum(1 for s in snap_data if s['type'] == 'amp')
    n_lat = sum(1 for s in snap_data if s['type'] == 'lat')
    n_extra = sum(len(g['segs']) for g in extra_groups)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>归并引擎 v2.1 - 完全归并</title>
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
</style></head><body>

<h2>归并引擎 v2.1 | EURUSD H1 | {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}</h2>
<div class="info">
{len(df)} bars | Base: {snap_data[0]['n']}pv |
Final: {len(results['final_pivots'])}pv |
Iter: {results['total_iterations']} |
Snap: {n_snap} (A:{n_amp} T:{n_lat}) |
Extra: {n_extra} segs ({len(extra_groups)} groups) |
Pool: 301 unique
</div>

<div style="margin:4px 0;">
<button class="btn" onclick="showAll()">All</button>
<button class="btn" onclick="hideAll()">None</button>
<button class="btn" onclick="showType('amp')">Amp</button>
<button class="btn" onclick="showType('lat')">Lat</button>
<button class="btn" onclick="showBase()">Base</button>
<button class="btn" onclick="toggleExtra()">Extra</button>
<button class="btn" onclick="toggleTop()" style="background:#2a1a4a;color:#f8f;">Top20</button>
<button class="btn" onclick="showStep(1)">Step+</button>
<button class="btn" onclick="showStep(-1)">Step-</button>
</div>

<div style="display:flex; gap:15px; flex-wrap:wrap; margin:6px 0;">
<div>
<div style="color:#fc8; font-size:11px; font-weight:bold;">Snapshots</div>
<div class="section" id="snap_ctl"></div>
</div>
<div>
<div style="color:#9e9; font-size:11px; font-weight:bold;">Extra (滑窗收集)</div>
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
let showTop = true;

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
        // lat-mid: green variants
        const t = gi / Math.max(EX.length-1, 1);
        return `rgb(${{Math.round(80+t*40)}},${{Math.round(220-t*60)}},${{Math.round(120+t*40)}})`;
    }} else {{
        // amp extra: orange variants
        const t = gi / Math.max(EX.length-1, 1);
        return `rgb(${{Math.round(255-t*40)}},${{Math.round(160-t*60)}},${{Math.round(60)}})`;
    }}
}}

const vis = S.map(()=>true);
const exVis = EX.map(()=>true);
let stepCursor = S.length;

function showAll(){{ vis.fill(true); exVis.fill(true); stepCursor=S.length; sync(); draw(); updateCursor(); }}
function hideAll(){{ vis.fill(false); exVis.fill(false); stepCursor=0; sync(); draw(); updateCursor(); }}
function showType(t){{ vis.fill(false); exVis.fill(false); S.forEach((s,i)=>vis[i]=s.type===t); sync(); draw(); }}
function showBase(){{ vis.fill(false); exVis.fill(false); vis[0]=true; sync(); draw(); }}
function toggleExtra(){{ const on=exVis.some(v=>v); exVis.fill(!on); sync(); draw(); }}
function toggleTop(){{ showTop=!showTop; draw(); }}

function showStep(d){{
    stepCursor=Math.max(0,Math.min(S.length,stepCursor+d));
    S.forEach((s,i)=>vis[i]=i<stepCursor);
    // show extra up to current step label
    if(stepCursor>0){{
        const curLabel = S[stepCursor-1].label;
        EX.forEach((g,i)=>{{
            // show extra if its source label <= current snapshot label
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

    // === Extra segments (draw first, behind snapshots) ===
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

    // === Top 20 pivot markers ===
    if(showTop && TOP.length > 0) {{
        cx.globalAlpha = 1;
        cx.setLineDash([]);
        cx.font = 'bold 11px monospace';
        cx.textAlign = 'center';

        for(const m of TOP) {{
            const x = xS(m.bar);
            const y = yS(m.price);
            const isPeak = m.dir === 1;
            const color = isPeak ? '#FF4444' : '#44FF44';
            const yOff = isPeak ? -14 : 16;

            // Marker circle
            cx.fillStyle = color;
            cx.beginPath();
            cx.arc(x, y, 5, 0, Math.PI*2);
            cx.fill();
            cx.strokeStyle = '#fff';
            cx.lineWidth = 1;
            cx.stroke();

            // Label: H1/L1 + price
            cx.fillStyle = color;
            cx.fillText(`${{m.label}}`, x, y + yOff);
            cx.font = '9px monospace';
            cx.fillStyle = '#aaa';
            cx.fillText(`${{m.price.toFixed(4)}}`, x, y + yOff + (isPeak ? -11 : 11));
            cx.font = 'bold 11px monospace';

            // Vertical line to price
            cx.strokeStyle = color;
            cx.lineWidth = 0.5;
            cx.globalAlpha = 0.4;
            cx.beginPath();
            cx.moveTo(x, y);
            cx.lineTo(x, y + yOff * 0.6);
            cx.stroke();
            cx.globalAlpha = 1;
        }}
        cx.textAlign = 'start';
    }}
}}

cv.addEventListener('mousemove',(e)=>{{
    const rect=cv.getBoundingClientRect();
    const mx_=e.clientX-rect.left;
    const bar=Math.round((mx_-mg.l)/pw*K.length);
    if(bar>=0&&bar<K.length){{
        const k=K[bar];
        document.getElementById('info').textContent=
            `Bar ${{bar}} | O:${{k.o.toFixed(5)}} H:${{k.h.toFixed(5)}} L:${{k.l.toFixed(5)}} C:${{k.c.toFixed(5)}}`;
    }}
}});

draw();
</script></body></html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Saved: {output_path}")

def main():
    df = load_kline("/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv", limit=200)
    base = calculate_base_zg(df['high'].values, df['low'].values)
    results = full_merge_engine(base)
    pivot_info = compute_pivot_importance(results)
    generate_html(df, results, "/home/ubuntu/stage2_abc/merge_v2.html", pivot_info=pivot_info)

if __name__ == '__main__':
    main()
