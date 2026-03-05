#!/usr/bin/env python3
"""
emerge v3.2 — 多品种四窗口归并可视化

功能:
  0. 使用database2的normalized数据 (close[0]=1.0, 无时间戳)
  1. 四窗口同时显示 M5, M15, M30, H1
  2. 每个窗口: 鼠标滚轮缩放, 点击左/右半区平移, 键盘+/-/←→
  3. 完整保留: 归并快照、重要性计算、fusion、重要拐点标注
  4. 品种选择下拉框 (自动扫描106个品种)

数据源: /home/ubuntu/database2/{TF}/{SYMBOL}_{TF}_norm.csv
格式: open,high,low,close,return (无时间戳, 纯bar index)

用法:
  python3 emerge_v32.py                     # 默认 EURUSD, 最近5000 bars
  python3 emerge_v32.py XAUUSD              # 指定品种
  python3 emerge_v32.py EURUSD 10000        # 指定品种和bar数(对H1)
"""

import json, sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ubuntu/stage2_abc')
from merge_engine_v3 import (
    calculate_base_zg, full_merge_engine,
    compute_pivot_importance, build_segment_pool, pool_fusion,
)

NORM_DIR = '/home/ubuntu/database2'
TFS = ['M5', 'M15', 'M30', 'H1']


def discover_symbols():
    """扫描database2/H1目录, 返回所有有归一化数据的品种"""
    h1_dir = os.path.join(NORM_DIR, 'H1')
    symbols = set()
    for f in os.listdir(h1_dir):
        if f.endswith('_H1_norm.csv'):
            symbols.add(f.replace('_H1_norm.csv', ''))
    return sorted(symbols)


def load_norm_csv(symbol, tf, limit=None):
    """加载normalized CSV → DataFrame[open,high,low,close]"""
    path = os.path.join(NORM_DIR, tf, f'{symbol}_{tf}_norm.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # 有些文件有time列, 有些没有 — 统一只保留OHLC
    cols_keep = []
    for c in ['open', 'high', 'low', 'close']:
        if c in df.columns:
            cols_keep.append(c)
    if len(cols_keep) < 4:
        return None
    df = df[cols_keep].copy()
    df = df.dropna()
    if limit and len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    return df


def process_tf(df, tf_name):
    """对单个TF运行完整引擎, 返回可视化数据"""
    t0 = time.time()
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)

    base = calculate_base_zg(high, low, rb=0.5)
    results = full_merge_engine(base)
    pivot_info = compute_pivot_importance(results, total_bars=len(df))
    pool = build_segment_pool(results, pivot_info)
    full_pool, fusion_segs, fusion_log = pool_fusion(pool, pivot_info)

    elapsed = time.time() - t0
    print(f"  [{tf_name}] {len(df)} bars, {len(base)} base pivots, "
          f"{len(pool)}→{len(full_pool)} pool ({elapsed:.1f}s)")

    # K线
    klines = []
    for i in range(len(df)):
        klines.append({
            'o': round(float(df.iloc[i]['open']), 6),
            'h': round(float(df.iloc[i]['high']), 6),
            'l': round(float(df.iloc[i]['low']), 6),
            'c': round(float(df.iloc[i]['close']), 6),
        })

    # 快照
    snaps = []
    for snap_type, label, pvts in results['all_snapshots']:
        pts = [{'bar': int(p[0]), 'y': round(p[1], 6), 'dir': int(p[2])} for p in pvts]
        snaps.append({'type': snap_type, 'label': label, 'n': len(pvts), 'pts': pts})

    # Extra segments
    extra_by_label = {}
    for p_start, p_end, label in results.get('extra_segments', []):
        if label not in extra_by_label:
            extra_by_label[label] = []
        extra_by_label[label].append({
            'b1': int(p_start[0]), 'p1': round(p_start[1], 6),
            'b2': int(p_end[0]),   'p2': round(p_end[1], 6),
        })
    extra_groups = []
    for label in sorted(extra_by_label.keys()):
        segs = extra_by_label[label]
        src = 'amp' if (label.startswith('A') and '_mid' not in label) else 'lat'
        extra_groups.append({'label': label, 'src': src, 'segs': segs})

    # Fusion segments (Top 200 by importance)
    sorted_fusion = sorted(fusion_segs, key=lambda s: -s['importance'])[:200]
    fus_data = []
    for s in sorted_fusion:
        fus_data.append({
            'b1': s['bar_start'], 'p1': round(s['price_start'], 6),
            'd1': s['dir_start'],
            'b2': s['bar_end'],   'p2': round(s['price_end'], 6),
            'd2': s['dir_end'],
            'imp': round(s['importance'], 4),
            'span': s['span'],
            'amp': round(s['amplitude'], 6),
            'src': s['source_label'],
        })

    # 重要拐点
    peaks = sorted([v for v in pivot_info.values() if v['dir'] == 1],
                   key=lambda x: -x['importance'])
    valleys = sorted([v for v in pivot_info.values() if v['dir'] == -1],
                     key=lambda x: -x['importance'])
    top_marks = []
    for rank, p in enumerate(peaks):
        top_marks.append({
            'bar': p['bar'], 'price': round(p['price'], 6), 'dir': 1,
            'rank': rank + 1, 'imp': round(p['importance'], 4),
            'label': f"H{rank+1}",
        })
    for rank, p in enumerate(valleys):
        top_marks.append({
            'bar': p['bar'], 'price': round(p['price'], 6), 'dir': -1,
            'rank': rank + 1, 'imp': round(p['importance'], 4),
            'label': f"L{rank+1}",
        })

    return {
        'tf': tf_name,
        'n_bars': len(df),
        'n_base': len(base),
        'n_pool': len(full_pool),
        'klines': klines,
        'snaps': snaps,
        'extras': extra_groups,
        'fusions': fus_data,
        'marks': top_marks,
    }


def generate_html(symbol, tf_data_list, all_symbols, output_path):
    """生成四窗口HTML"""

    tf_json = json.dumps(tf_data_list)
    symbols_json = json.dumps(all_symbols)

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>emerge v3.2 | __SYMBOL__</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0a0a1a; color: #ddd; font-family: 'Consolas', 'Courier New', monospace; overflow: hidden; }
.header { display: flex; align-items: center; gap: 12px; padding: 5px 12px; background: #0d0d20; border-bottom: 1px solid #222; height: 32px; }
.header h1 { font-size: 14px; color: #7eb8da; white-space: nowrap; }
.header select { background: #1a1a30; color: #8ac; border: 1px solid #335; padding: 2px 6px; font-family: inherit; font-size: 12px; border-radius: 3px; }
.header .info { font-size: 11px; color: #888; }
.controls { display: flex; flex-wrap: wrap; gap: 3px 8px; padding: 3px 12px; background: #0c0c1e; border-bottom: 1px solid #1a1a2a; align-items: center; height: 30px; }
.btn { background: #1a2a4a; color: #8ac; border: 1px solid #335; padding: 1px 6px; border-radius: 3px; cursor: pointer; font-size: 11px; font-family: inherit; }
.btn:hover { background: #2a3a5a; }
.btn.active { background: #3a2a5a; color: #f8f; border-color: #a6a; }
.slider-box { display: inline-flex; align-items: center; gap: 3px; font-size: 10px; color: #999; }
.slider-box input[type=range] { width: 70px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; width: 100vw; height: calc(100vh - 62px); }
.pane { position: relative; border: 1px solid #1a1a30; overflow: hidden; }
.pane canvas { width: 100%; height: 100%; display: block; cursor: crosshair; }
.pane-label { position: absolute; top: 3px; left: 8px; font-size: 13px; font-weight: bold; pointer-events: none; z-index: 2; }
.pane-info { position: absolute; bottom: 1px; left: 8px; font-size: 9px; color: #555; pointer-events: none; z-index: 2; }
.hover-info { position: absolute; top: 3px; right: 8px; font-size: 10px; color: #aaa; pointer-events: none; z-index: 2; text-align: right; max-width: 70%; }
</style></head><body>

<div class="header">
  <h1>emerge v3.2</h1>
  <select id="symSel" onchange="location.href='emerge_v32_'+this.value+'.html'">
  </select>
  <span class="info" id="gInfo"></span>
  <span style="flex:1"></span>
  <span class="info" style="color:#555;">Wheel: zoom | Click L/R: pan | +- Keys: zoom | Arrows: scroll</span>
</div>

<div class="controls">
  <button class="btn active" id="bSnap" onclick="tgl('snaps')">Snaps</button>
  <button class="btn" id="bExtra" onclick="tgl('extras')">Extra</button>
  <button class="btn active" id="bFus" onclick="tgl('fusion')">Fusion</button>
  <button class="btn active" id="bMark" onclick="tgl('marks')">ImpPts</button>
  <button class="btn active" id="bVal" onclick="tgl('values')">Values</button>
  <span class="slider-box" style="color:#f8f">Pk:<input type="range" id="sP" min="0" max="50" value="10" oninput="pkN=+this.value;document.getElementById('nP').textContent=pkN;DA()"><span id="nP">10</span></span>
  <span class="slider-box" style="color:#4f4">Vl:<input type="range" id="sV" min="0" max="50" value="10" oninput="vlN=+this.value;document.getElementById('nV').textContent=vlN;DA()"><span id="nV">10</span></span>
  <span class="slider-box" style="color:#c8f">Fus:<input type="range" id="sF" min="0" max="200" value="30" oninput="fuN=+this.value;document.getElementById('nF').textContent=fuN;DA()"><span id="nF">30</span></span>
  <span class="slider-box" style="color:#ffa">MinImp:<input type="range" id="sI" min="0" max="100" value="0" oninput="mImp=this.value/100;document.getElementById('nI').textContent=mImp.toFixed(2);DA()"><span id="nI">0.00</span></span>
</div>

<div class="grid" id="grid">
  <div class="pane" id="P0"><canvas id="C0"></canvas><div class="pane-label" id="L0" style="color:#5a8ab0"></div><div class="pane-info" id="I0"></div><div class="hover-info" id="H0"></div></div>
  <div class="pane" id="P1"><canvas id="C1"></canvas><div class="pane-label" id="L1" style="color:#8ab05a"></div><div class="pane-info" id="I1"></div><div class="hover-info" id="H1"></div></div>
  <div class="pane" id="P2"><canvas id="C2"></canvas><div class="pane-label" id="L2" style="color:#b08a5a"></div><div class="pane-info" id="I2"></div><div class="hover-info" id="H2"></div></div>
  <div class="pane" id="P3"><canvas id="C3"></canvas><div class="pane-label" id="L3" style="color:#b05a8a"></div><div class="pane-info" id="I3"></div><div class="hover-info" id="H3"></div></div>
</div>

<script>
const TF=__TF_JSON__;
const SYM=__SYMBOLS_JSON__;
const CUR='__SYMBOL__';

// State
const ly={snaps:true,extras:false,fusion:true,marks:true,values:true};
let pkN=10,vlN=10,fuN=30,mImp=0,aP=-1;

function tgl(k){ly[k]=!ly[k];const m={snaps:'bSnap',extras:'bExtra',fusion:'bFus',marks:'bMark',values:'bVal'};
const b=document.getElementById(m[k]);if(b)b.classList.toggle('active',ly[k]);DA();}

// Canvas
const cvs=[0,1,2,3].map(i=>document.getElementById('C'+i));
const cxs=cvs.map(c=>c.getContext('2d'));
const pns=[0,1,2,3].map(i=>document.getElementById('P'+i));

const vw=TF.map((t,i)=>({i,vs:Math.max(0,t.n_bars-300),ve:t.n_bars,tf:t}));

function resize(){
  cvs.forEach((c,i)=>{const r=pns[i].getBoundingClientRect();
    c.width=r.width*devicePixelRatio;c.height=r.height*devicePixelRatio;
    c.style.width=r.width+'px';c.style.height=r.height+'px';
    cxs[i].setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);});DA();}
window.addEventListener('resize',resize);

const MG={l:55,r:8,t:20,b:14};
const tfColors=['#5a8ab0','#8ab05a','#b08a5a','#b05a8a'];

function sCol(si,snaps){const s=snaps[si];
  if(s.type==='base')return'#FFD700';
  if(s.type==='amp'){let a=0;for(let j=0;j<si;j++)if(snaps[j].type==='amp')a++;
    const n=snaps.filter(x=>x.type==='amp').length;const t=a/Math.max(n-1,1);
    return`rgb(${Math.round(255-t*80)},${Math.round(100-t*60)},30)`;}
  if(s.type==='lat'){let l=0;for(let j=0;j<si;j++)if(snaps[j].type==='lat')l++;
    const n=snaps.filter(x=>x.type==='lat').length;const t=l/Math.max(n-1,1);
    return`rgb(${Math.round(50+t*50)},${Math.round(220-t*80)},${Math.round(255-t*40)})`;}
  return'#888';}
function fCol(r,n){const t=r/Math.max(n-1,1);
  return`rgb(${Math.round(200-t*100)},${Math.round(80-t*50)},${Math.round(255-t*80)})`;}

function DP(pi){
  const cv=cvs[pi],cx=cxs[pi],W=cv.width/devicePixelRatio,H=cv.height/devicePixelRatio;
  const v=vw[pi],tf=v.tf;if(!tf)return;
  const vs=Math.max(0,Math.floor(v.vs)),ve=Math.min(tf.n_bars,Math.ceil(v.ve));
  const nV=ve-vs;if(nV<=0)return;
  const pw=W-MG.l-MG.r,ph=H-MG.t-MG.b;

  let pMn=Infinity,pMx=-Infinity;
  for(let i=vs;i<ve;i++){const k=tf.klines[i];if(k.l<pMn)pMn=k.l;if(k.h>pMx)pMx=k.h;}
  const pr=pMx-pMn;pMn-=pr*0.04;pMx+=pr*0.04;

  const xS=b=>MG.l+((b-vs)/nV)*pw;
  const yS=p=>MG.t+ph-((p-pMn)/(pMx-pMn))*ph;
  const bFx=x=>vs+(x-MG.l)/pw*nV;

  cx.fillStyle='#0a0a1a';cx.fillRect(0,0,W,H);
  if(pi===aP){cx.strokeStyle='#335588';cx.lineWidth=2;cx.strokeRect(1,1,W-2,H-2);}

  // Grid
  cx.strokeStyle='#181830';cx.lineWidth=0.5;cx.fillStyle='#444';cx.font='9px monospace';
  for(let i=0;i<=6;i++){const p=pMn+(pMx-pMn)*i/6,y=yS(p);
    cx.beginPath();cx.moveTo(MG.l,y);cx.lineTo(W-MG.r,y);cx.stroke();
    cx.fillText(pr<0.1?p.toFixed(6):pr<1?p.toFixed(5):pr<100?p.toFixed(3):p.toFixed(1),2,y+3);}

  // Bar index labels
  cx.fillStyle='#444';cx.font='9px monospace';cx.textAlign='center';
  const lStep=Math.max(1,Math.floor(nV/8));
  for(let i=vs;i<ve;i+=lStep){cx.fillText(i,xS(i),H-2);}
  cx.textAlign='start';

  // K-lines
  const bW=Math.max(0.5,pw/nV*0.6);
  for(let i=vs;i<ve;i++){const k=tf.klines[i],x=xS(i),up=k.c>=k.o;
    cx.strokeStyle=up?'#2a6a2a':'#6a2a2a';cx.lineWidth=0.6;
    cx.beginPath();cx.moveTo(x,yS(k.l));cx.lineTo(x,yS(k.h));cx.stroke();
    if(bW>1.5){cx.fillStyle=up?'#1a4a1a':'#4a1a1a';
      const yt=yS(Math.max(k.o,k.c)),yb=yS(Math.min(k.o,k.c));
      cx.fillRect(x-bW/2,yt,bW,Math.max(1,yb-yt));}}

  // Fusion
  if(ly.fusion&&tf.fusions.length){const n=Math.min(fuN,tf.fusions.length);
    for(let i=n-1;i>=0;i--){const f=tf.fusions[i];if(f.imp<mImp||f.b2<vs||f.b1>ve)continue;
      const t=i/Math.max(n-1,1);cx.strokeStyle=fCol(i,n);cx.lineWidth=Math.max(0.3,2-t*1.5);
      cx.globalAlpha=Math.max(0.15,0.6-t*0.4);cx.setLineDash([8,3]);
      cx.beginPath();cx.moveTo(xS(f.b1),yS(f.p1));cx.lineTo(xS(f.b2),yS(f.p2));cx.stroke();}
    cx.setLineDash([]);cx.globalAlpha=1;}

  // Extras
  if(ly.extras){for(const g of tf.extras){
    const isL=g.label.includes('_mid');cx.strokeStyle=isL?'#4a8a5a':'#aa8a40';
    cx.lineWidth=isL?0.8:0.6;cx.globalAlpha=0.4;cx.setLineDash(isL?[4,3]:[5,2]);
    for(const s of g.segs){if(s.b2<vs||s.b1>ve)continue;
      cx.beginPath();cx.moveTo(xS(s.b1),yS(s.p1));cx.lineTo(xS(s.b2),yS(s.p2));cx.stroke();}
  }cx.setLineDash([]);cx.globalAlpha=1;}

  // Snapshots
  if(ly.snaps){for(let si=0;si<tf.snaps.length;si++){const s=tf.snaps[si],pts=s.pts;
    if(pts.length<2)continue;const col=sCol(si,tf.snaps);
    if(s.type==='base'){cx.strokeStyle=col;cx.lineWidth=0.3;cx.globalAlpha=0.2;}
    else if(s.type==='amp'){let a=0;for(let j=0;j<si;j++)if(tf.snaps[j].type==='amp')a++;
      cx.strokeStyle=col;cx.lineWidth=Math.min(0.5+a*0.25,3);cx.globalAlpha=Math.min(0.3+a*0.07,0.85);}
    else{let l=0;for(let j=0;j<si;j++)if(tf.snaps[j].type==='lat')l++;
      cx.strokeStyle=col;cx.lineWidth=Math.min(1.2+l*0.25,3);cx.globalAlpha=0.8;cx.setLineDash([6+l*2,3+l]);}
    cx.beginPath();let st=false;
    for(const p of pts){if(p.bar<vs-nV*0.1||p.bar>ve+nV*0.1)continue;
      if(!st){cx.moveTo(xS(p.bar),yS(p.y));st=true;}else cx.lineTo(xS(p.bar),yS(p.y));}
    if(st)cx.stroke();
    if(pts.length<40&&s.type!=='base'){const ms=Math.min(2+(40-pts.length)*0.08,4);cx.fillStyle=col;
      for(const p of pts){if(p.bar<vs||p.bar>ve)continue;
        cx.beginPath();cx.arc(xS(p.bar),yS(p.y),ms,0,Math.PI*2);cx.fill();}}
    cx.setLineDash([]);cx.globalAlpha=1;}}

  // Important points
  if(ly.marks){const pks=tf.marks.filter(m=>m.dir===1),vls=tf.marks.filter(m=>m.dir===-1);
    cx.textAlign='center';
    for(let i=0;i<Math.min(pkN,pks.length);i++){const m=pks[i];if(m.bar<vs||m.bar>ve)continue;
      const x=xS(m.bar),y=yS(m.price),sz=Math.max(2.5,5-i*0.25),al=Math.max(0.4,1-i*0.04);
      cx.fillStyle='#FF4444';cx.globalAlpha=al;cx.beginPath();cx.arc(x,y,sz,0,Math.PI*2);cx.fill();
      cx.strokeStyle='#fff';cx.lineWidth=i<5?1:0.5;cx.stroke();
      cx.fillStyle='#FF4444';cx.font=i<10?'bold 10px monospace':'9px monospace';cx.fillText(m.label,x,y-10);
      if(ly.values){cx.font='8px monospace';cx.fillStyle='#faa';cx.fillText(m.imp.toFixed(3),x,y-20);}
      cx.globalAlpha=1;}
    for(let i=0;i<Math.min(vlN,vls.length);i++){const m=vls[i];if(m.bar<vs||m.bar>ve)continue;
      const x=xS(m.bar),y=yS(m.price),sz=Math.max(2.5,5-i*0.25),al=Math.max(0.4,1-i*0.04);
      cx.fillStyle='#44FF44';cx.globalAlpha=al;cx.beginPath();cx.arc(x,y,sz,0,Math.PI*2);cx.fill();
      cx.strokeStyle='#fff';cx.lineWidth=i<5?1:0.5;cx.stroke();
      cx.fillStyle='#44FF44';cx.font=i<10?'bold 10px monospace':'9px monospace';cx.fillText(m.label,x,y+14);
      if(ly.values){cx.font='8px monospace';cx.fillStyle='#afa';cx.fillText(m.imp.toFixed(3),x,y+24);}
      cx.globalAlpha=1;}
    cx.textAlign='start';}

  // Pane label
  document.getElementById('L'+pi).textContent=tf.tf+' | '+nV+' bars';
  document.getElementById('I'+pi).textContent='base:'+tf.n_base+' pool:'+tf.n_pool+' fus:'+tf.fusions.length;

  v._xS=xS;v._yS=yS;v._bFx=bFx;v._vs=vs;v._ve=ve;v._pMn=pMn;v._pMx=pMx;
}

function DA(){vw.forEach((_,i)=>DP(i));}

function clamp(v){const sp=v.ve-v.vs;
  if(v.vs<0){v.vs=0;v.ve=sp;}if(v.ve>v.tf.n_bars){v.ve=v.tf.n_bars;v.vs=v.tf.n_bars-sp;}
  if(v.vs<0)v.vs=0;}
function zoom(i,f){const v=vw[i],mid=(v.vs+v.ve)/2,h=(v.ve-v.vs)/2*f,nh=Math.max(10,h);
  v.vs=mid-nh;v.ve=mid+nh;clamp(v);DP(i);}
function pan(i,d){const v=vw[i];v.vs+=d;v.ve+=d;clamp(v);DP(i);}

document.addEventListener('keydown',e=>{if(aP<0)return;const v=vw[aP],sp=v.ve-v.vs,st=Math.max(1,Math.floor(sp*0.15));
  if(e.key==='+'||e.key==='='){zoom(aP,0.7);e.preventDefault();}
  else if(e.key==='-'||e.key==='_'){zoom(aP,1.4);e.preventDefault();}
  else if(e.key==='ArrowLeft'){pan(aP,-st);e.preventDefault();}
  else if(e.key==='ArrowRight'){pan(aP,st);e.preventDefault();}
  else if(e.key==='Home'){v.vs=0;v.ve=sp;clamp(v);DP(aP);e.preventDefault();}
  else if(e.key==='End'){v.ve=v.tf.n_bars;v.vs=v.tf.n_bars-sp;clamp(v);DP(aP);e.preventDefault();}
});

pns.forEach((pn,i)=>{const cv=cvs[i];
  cv.addEventListener('mousedown',e=>{aP=i;DA();const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,mid=r.width/2;
    const v=vw[i],sp=v.ve-v.vs,st=Math.max(1,Math.floor(sp*0.25));
    if(mx<mid)pan(i,-st);else pan(i,st);});
  cv.addEventListener('wheel',e=>{e.preventDefault();aP=i;zoom(i,e.deltaY>0?1.2:0.83);},{passive:false});
  cv.addEventListener('mousemove',e=>{const v=vw[i];if(!v._bFx)return;
    const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    const bar=Math.round(v._bFx(mx));let info='';
    if(bar>=0&&bar<v.tf.n_bars){const k=v.tf.klines[bar];
      info=`bar ${bar} | O:${k.o.toFixed(5)} H:${k.h.toFixed(5)} L:${k.l.toFixed(5)} C:${k.c.toFixed(5)}`;}
    if(ly.marks&&v._xS){const pks=v.tf.marks.filter(m=>m.dir===1).slice(0,pkN);
      const vls=v.tf.marks.filter(m=>m.dir===-1).slice(0,vlN);
      let bd=15,bm=null;for(const m of pks.concat(vls)){if(m.bar<v._vs||m.bar>v._ve)continue;
        const d=Math.sqrt((mx-v._xS(m.bar))**2+(my-v._yS(m.price))**2);
        if(d<bd){bd=d;bm=m;}}
      if(bm)info+=` | ${bm.label} imp=${bm.imp.toFixed(4)} price=${bm.price.toFixed(5)}`;}
    if(ly.fusion&&v._xS){const n=Math.min(fuN,v.tf.fusions.length);let bd=12,bf=null;
      for(let fi=0;fi<n;fi++){const f=v.tf.fusions[fi];if(f.imp<mImp||f.b2<v._vs||f.b1>v._ve)continue;
        const x1=v._xS(f.b1),y1=v._yS(f.p1),x2=v._xS(f.b2),y2=v._yS(f.p2);
        const dx=x2-x1,dy=y2-y1,l2=dx*dx+dy*dy;if(!l2)continue;
        let t=((mx-x1)*dx+(my-y1)*dy)/l2;t=Math.max(0,Math.min(1,t));
        const d=Math.sqrt((mx-x1-t*dx)**2+(my-y1-t*dy)**2);if(d<bd){bd=d;bf=f;}}
      if(bf)info+=` | FUS[${bf.src}] bar${bf.b1}-${bf.b2} span=${bf.span} amp=${bf.amp} imp=${bf.imp}`;}
    document.getElementById('H'+i).textContent=info;});
});

// Init
const sel=document.getElementById('symSel');
sel.innerHTML=SYM.map(s=>`<option value="${s}" ${s===CUR?'selected':''}>${s}</option>`).join('');
document.getElementById('gInfo').textContent=TF.map(t=>t.tf+':'+t.n_bars).join(' | ');
resize();
</script>
</body></html>"""

    html = html.replace('__TF_JSON__', tf_json)
    html = html.replace('__SYMBOLS_JSON__', symbols_json)
    html = html.replace('__SYMBOL__', symbol)

    with open(output_path, 'w') as f:
        f.write(html)
    size_kb = len(html) // 1024
    print(f"\nSaved: {output_path} ({size_kb}KB)")


def main():
    symbol = 'EURUSD'
    h1_limit = 5000  # 控制数据量: H1的bar数, 其他TF按比例

    if len(sys.argv) >= 2:
        symbol = sys.argv[1].upper()
    if len(sys.argv) >= 3:
        h1_limit = int(sys.argv[2])

    print(f"emerge v3.2 | {symbol} | normalized data")
    print(f"{'='*60}")

    all_symbols = discover_symbols()
    if symbol not in all_symbols:
        print(f"ERROR: {symbol} not found in database2. Available: {', '.join(all_symbols[:10])}...")
        sys.exit(1)

    # 各TF数据量: pool_fusion是O(n^3), 必须控制bar数
    # 经验: 500 bars → ~2s, 1000 bars → ~10s, 2000 bars → 超时
    # 每个TF独立取最近N根, 保证总计算<30s
    tf_limits = {
        'H1':  min(h1_limit, 1000),
        'M30': min(h1_limit * 2, 1000),
        'M15': min(h1_limit * 4, 1000),
        'M5':  min(h1_limit * 12, 1000),
    }

    print(f"\nLoading normalized data from database2...")
    tf_data_list = []
    for tf_name in TFS:
        limit = tf_limits.get(tf_name, h1_limit)
        df = load_norm_csv(symbol, tf_name, limit=limit)
        if df is None or len(df) < 10:
            print(f"  [{tf_name}] SKIPPED: no data or too few bars")
            continue
        print(f"  [{tf_name}] loaded {len(df)} bars")
        data = process_tf(df, tf_name)
        tf_data_list.append(data)

    if not tf_data_list:
        print("ERROR: No data available for any timeframe!")
        sys.exit(1)

    output_path = f'/home/ubuntu/stage2_abc/emerge_v32_{symbol}.html'
    generate_html(symbol, tf_data_list, all_symbols, output_path)
    print(f"\nDone! Open: {output_path}")


if __name__ == '__main__':
    main()
