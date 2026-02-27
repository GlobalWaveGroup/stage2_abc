#!/usr/bin/env python3
"""
visualize_v3.1 — v3完整pipeline + 冗余删除 + 滑动窗口 + 动态时间轴

核心改进 (vs v4/v5):
  v4/v5 只用 DynamicZG+DynamicMerger (161段), 丢失了v3的全部归并+融合体系
  v3.1 每个窗口做完整的 v3 pipeline:
    base_zg → full_merge_engine → importance → build_pool → pool_fusion → prune → predict

核心改进 (vs v3):
  v3 是静态200 bars, v3.1 是滑动窗口2000 bars + 动态播放

可视化保留v3的全部层级:
  - 逐级归并快照 (base/A1/A2.../T1/T2...) — 不同颜色和线型
  - Extra segments (贪心跳过的)
  - Fusion segments (跨级别三波连接)
  - 预测 (mirror + center symmetric image)
  - 拐点重要性标注

数据流: 滑动窗口 → 完整v3 pipeline → 增量编码 → HTML/JS
"""

import json
import sys
import time
import math
import numpy as np
sys.path.insert(0, '/home/ubuntu/stage2_abc')
from merge_engine_v3 import (load_kline, calculate_base_zg, full_merge_engine,
                              build_segment_pool, compute_pivot_importance,
                              pool_fusion, predict_symmetric_image)
from fsd_engine import prune_redundant


def run_windowed_pipeline(df, window=200, stride=10, max_pool=300, max_preds=50):
    """
    滑动窗口运行完整v3 pipeline, 收集每帧数据。
    
    每stride根K线做一次完整重算:
      base_zg → full_merge → importance → pool → fusion → prune → predict
    中间帧复用上一窗口的结果。
    """
    n = len(df)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    opens = df['open'].values.astype(float)
    closes = df['close'].values.astype(float)
    
    klines = []
    for i in range(n):
        klines.append([round(opens[i],5), round(highs[i],5), 
                        round(lows[i],5), round(closes[i],5)])
    
    # 每帧的数据
    frame_snapshots = []   # 当前窗口的归并快照列表
    frame_pool = []        # 当前窗口的pruned pool
    frame_preds = []       # 当前窗口的top predictions
    frame_pivots = []      # 当前窗口的重要拐点 (top by importance)
    
    # 当前缓存
    cur_snaps = []
    cur_pool = []
    cur_preds = []
    cur_pivots = []
    
    t0 = time.time()
    for i in range(n):
        should_compute = (i >= window and (i - window) % stride == 0)
        
        if should_compute:
            ws = i - window
            hw = highs[ws:i]
            lw = lows[ws:i]
            
            # 完整v3 pipeline
            base = calculate_base_zg(hw, lw, rb=0.5)
            results = full_merge_engine(base)
            pi = compute_pivot_importance(results, window)
            pool = build_segment_pool(results, pi)
            fused_all, _, _ = pool_fusion(pool, pi)
            pruned = prune_redundant(fused_all, pi, merge_dist=3, 
                                      max_per_start=5, max_total=max_pool)
            preds = predict_symmetric_image(pruned, pi, current_bar=window-1, 
                                             max_pool_size=99999)
            
            # 坐标变换: 窗口内 → 全局
            snapshots_global = []
            for snap_type, label, pvts in results['all_snapshots']:
                gpvts = [(p[0]+ws, round(p[1],5), p[2]) for p in pvts]
                snapshots_global.append({
                    't': snap_type[0],  # 'b','a','l'
                    'lb': label,
                    'pts': gpvts
                })
            
            pool_global = []
            for seg in pruned:
                pool_global.append({
                    'bs': seg['bar_start'] + ws,
                    'be': seg['bar_end'] + ws,
                    'ps': round(seg['price_start'], 5),
                    'pe': round(seg['price_end'], 5),
                    'src': seg['source'][0],  # b/a/l/f
                    'imp': round(seg.get('importance', 0), 5),
                    'amp': round(seg['amplitude'], 5),
                })
            
            preds_global = []
            for p in preds[:max_preds]:
                pg = {
                    'tp': p['type'][0],  # m/c
                    'pd': p['pred_dir'],
                    'as': p['A_start'] + ws,
                    'ae': p['A_end'] + ws,
                    'aps': round(p['A_price_start'], 5),
                    'ape': round(p['A_price_end'], 5),
                    'psb': p['pred_start_bar'] + ws,
                    'psp': round(p['pred_start_price'], 5),
                    'ptb': p['pred_target_bar'] + ws,
                    'ptp': round(p['pred_target_price'], 5),
                    'pa': round(p['pred_amp'], 5),
                    'pt': p['pred_time'],
                    'sc': round(p['score'], 5),
                }
                if p['type'] == 'center':
                    pg['bs'] = p['B_start'] + ws
                    pg['be'] = p['B_end'] + ws
                    pg['bps'] = round(p['B_price_start'], 5)
                    pg['bpe'] = round(p['B_price_end'], 5)
                preds_global.append(pg)
            
            # 重要拐点 (top 30)
            pivots_sorted = sorted(pi.values(), key=lambda x: -x['importance'])[:30]
            pivots_global = []
            for pv in pivots_sorted:
                pivots_global.append({
                    'b': pv['bar'] + ws,
                    'p': round(pv['price'], 5),
                    'd': pv['dir'],
                    'imp': round(pv['importance'], 4),
                })
            
            cur_snaps = snapshots_global
            cur_pool = pool_global
            cur_preds = preds_global
            cur_pivots = pivots_global
        
        frame_snapshots.append(cur_snaps)
        frame_pool.append(cur_pool)
        frame_preds.append(cur_preds)
        frame_pivots.append(cur_pivots)
        
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  bar {i+1}/{n} ({elapsed:.1f}s) pool={len(cur_pool)} preds={len(cur_preds)}")
    
    elapsed = time.time() - t0
    print(f"完成: {n} bars, {elapsed:.1f}s")
    
    return klines, frame_snapshots, frame_pool, frame_preds, frame_pivots


def generate_html(klines, frame_snapshots, frame_pool, frame_preds, frame_pivots,
                  df, output_path):
    """生成v3.1 HTML"""
    n = len(klines)
    WIN = 150
    FUT = 50
    
    # 增量编码: 只存变化帧
    def encode_changes(frames):
        changes = {}
        prev_key = None
        for i, f in enumerate(frames):
            key = json.dumps(f, separators=(',',':'))
            if key != prev_key:
                changes[str(i)] = f
                prev_key = key
        return changes
    
    snap_c = encode_changes(frame_snapshots)
    pool_c = encode_changes(frame_pool)
    pred_c = encode_changes(frame_preds)
    pvt_c = encode_changes(frame_pivots)
    
    kj = json.dumps(klines, separators=(',',':'))
    sj = json.dumps(snap_c, separators=(',',':'))
    pj = json.dumps(pool_c, separators=(',',':'))
    prj = json.dumps(pred_c, separators=(',',':'))
    pvj = json.dumps(pvt_c, separators=(',',':'))
    
    print(f"  增量: {len(snap_c)} snap帧, {len(pool_c)} pool帧, "
          f"{len(pred_c)} pred帧, {len(pvt_c)} pvt帧")
    print(f"  大小: K={len(kj)//1024}KB SNAP={len(sj)//1024}KB POOL={len(pj)//1024}KB "
          f"PRED={len(prj)//1024}KB PVT={len(pvj)//1024}KB")
    
    dt_start = str(df['datetime'].iloc[0])
    dt_end = str(df['datetime'].iloc[-1])
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>v3.1 | EURUSD H1 | Full Pipeline</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0a0a14; color: #eee; font-family: 'Consolas','Courier New',monospace; font-size: 11px; }}
.top {{ background: #12121f; padding: 4px 10px; border-bottom: 1px solid #252540; display: flex; justify-content: space-between; align-items: center; }}
.controls {{ display: flex; align-items: center; gap: 4px; padding: 3px 8px; background: #0e0e1a; flex-wrap: wrap; }}
.btn {{ background: #22224a; color: #aac; border: 1px solid #335; padding: 2px 6px; border-radius: 3px; cursor: pointer; font-size: 10px; font-family: inherit; }}
.btn:hover {{ background: #33336a; }}
.btn.on {{ background: #335588; color: #fff; }}
.slider-box {{ display: inline-flex; align-items: center; gap: 3px; color: #888; }}
.slider-box input[type=range] {{ width: 400px; }}
#mainCanvas {{ width: 100%; display: block; background: #0d0d18; }}
#infoBar {{ background: #12121f; padding: 3px 10px; border-top: 1px solid #252540; min-height: 20px; color: #888; }}
</style></head><body>

<div class="top">
  <span>v3.1 | EURUSD H1 | {dt_start} ~ {dt_end} | {n} bars | Full v3 Pipeline</span>
  <span style="color:#666">Window=200 Stride=10 Pool≤300</span>
</div>
<div class="controls">
  <button class="btn" onclick="stepTo(200)">|&lt;</button>
  <button class="btn" onclick="stepBy(-1)">&lt;</button>
  <button class="btn" id="playBtn" onclick="togglePlay()">&#9654;</button>
  <button class="btn" onclick="stepBy(1)">&gt;</button>
  <button class="btn" onclick="stepTo(N-1)">&gt;|</button>
  <span class="slider-box">
    <input type="range" id="barSlider" min="200" max="{n-1}" value="200"
         oninput="stepTo(parseInt(this.value))">
    <span id="barNum">200</span>/{n-1}
  </span>
  <span class="slider-box" style="color:#555">
    Spd:<input type="range" id="spdSlider" min="1" max="10" value="3" style="width:60px"
         oninput="this.nextElementSibling.textContent=this.value"><span>3</span>
  </span>
  |
  <button class="btn on" id="tbSnap" onclick="tog(this,'showSnap')">Merge</button>
  <button class="btn on" id="tbPool" onclick="tog(this,'showPool')">Pool</button>
  <button class="btn on" id="tbPred" onclick="tog(this,'showPred')">Pred</button>
  <button class="btn on" id="tbPvt" onclick="tog(this,'showPvt')">Pivots</button>
</div>

<canvas id="mainCanvas"></canvas>
<div id="infoBar"></div>

<script>
const K={kj};
const N=K.length;
const WIN={WIN}, FUT={FUT}, TOTALW=WIN+FUT;

const SNAP_C={sj};
const POOL_C={pj};
const PRED_C={prj};
const PVT_C={pvj};

const snapK=Object.keys(SNAP_C).map(Number).sort((a,b)=>a-b);
const poolK=Object.keys(POOL_C).map(Number).sort((a,b)=>a-b);
const predK=Object.keys(PRED_C).map(Number).sort((a,b)=>a-b);
const pvtK=Object.keys(PVT_C).map(Number).sort((a,b)=>a-b);

function findLE(keys,target){{
  let lo=0,hi=keys.length-1,r=-1;
  while(lo<=hi){{const m=(lo+hi)>>1;if(keys[m]<=target){{r=m;lo=m+1;}}else hi=m-1;}}
  return r>=0?keys[r]:-1;
}}
function getSnap(bar){{const k=findLE(snapK,bar);return k>=0?SNAP_C[k]:[];}}
function getPool(bar){{const k=findLE(poolK,bar);return k>=0?POOL_C[k]:[];}}
function getPred(bar){{const k=findLE(predK,bar);return k>=0?PRED_C[k]:[];}}
function getPvt(bar){{const k=findLE(pvtK,bar);return k>=0?PVT_C[k]:[];}}

let curBar=200, playing=false, playTimer=null;
let showSnap=true, showPool=true, showPred=true, showPvt=true;

function tog(btn,varName){{
  window[varName]=!window[varName];
  btn.classList.toggle('on',window[varName]);
  drawAll();
}}

// 归并层级颜色
const snapColors = {{
  'b': {{c:'rgba(255,215,0,A)', lw:0.5}},     // base: gold
  'a': [                                         // amp: red-orange gradient
    {{c:'rgba(255,100,30,A)', lw:1.0}},
    {{c:'rgba(255,120,40,A)', lw:1.3}},
    {{c:'rgba(255,140,50,A)', lw:1.6}},
    {{c:'rgba(255,160,60,A)', lw:2.0}},
    {{c:'rgba(240,170,70,A)', lw:2.3}},
    {{c:'rgba(230,180,80,A)', lw:2.6}},
    {{c:'rgba(220,190,90,A)', lw:3.0}},
  ],
  'l': [                                         // lat: cyan-blue gradient
    {{c:'rgba(50,220,255,A)', lw:1.5}},
    {{c:'rgba(70,200,245,A)', lw:2.0}},
    {{c:'rgba(90,180,235,A)', lw:2.5}},
    {{c:'rgba(110,160,225,A)', lw:3.0}},
  ],
}};

function draw(){{
  const cv=document.getElementById('mainCanvas');
  const cw=window.innerWidth;
  const ch=window.innerHeight-80;
  cv.width=cw; cv.height=Math.max(ch,400);
  const cx=cv.getContext('2d');
  const W=cv.width,H=cv.height;
  const mg={{l:56,r:8,t:14,b:14}};
  const pw=W-mg.l-mg.r, ph=H-mg.t-mg.b;
  
  const ws=Math.max(0,curBar-WIN+1);
  const snaps=getSnap(curBar);
  const pool=getPool(curBar);
  const preds=getPred(curBar);
  const pvts=getPvt(curBar);
  
  // 价格范围
  let pMn=Infinity,pMx=-Infinity;
  for(let b=ws;b<=curBar&&b<N;b++){{pMn=Math.min(pMn,K[b][2]);pMx=Math.max(pMx,K[b][1]);}}
  // 扩展: 预测目标
  for(const p of preds){{
    pMn=Math.min(pMn,p.ptp,p.psp);pMx=Math.max(pMx,p.ptp,p.psp);
  }}
  if(pMn===Infinity){{pMn=1.0;pMx=1.1;}}
  const rng=pMx-pMn; pMn-=rng*0.03; pMx+=rng*0.03;
  
  function xS(b){{return mg.l+((b-ws)/TOTALW)*pw;}}
  function yS(p){{return mg.t+ph-((p-pMn)/(pMx-pMn))*ph;}}
  
  // BG
  cx.fillStyle='#0d0d18'; cx.fillRect(0,0,W,H);
  cx.fillStyle='#10101f'; cx.fillRect(xS(curBar+1),mg.t,W-mg.r-xS(curBar+1),ph);
  
  // Grid
  cx.fillStyle='#444';cx.font='9px monospace';cx.strokeStyle='#161625';cx.lineWidth=0.5;
  for(let i=0;i<=5;i++){{const p=pMn+(pMx-pMn)*i/5;const y=yS(p);
    cx.beginPath();cx.moveTo(mg.l,y);cx.lineTo(W-mg.r,y);cx.stroke();
    cx.fillText(p.toFixed(4),2,y+3);}}
  
  // K-lines
  const bw=Math.max(1.2,pw/TOTALW*0.65);
  for(let b=ws;b<=curBar&&b<N;b++){{
    const k=K[b],x=xS(b),up=k[3]>=k[0];
    cx.strokeStyle=up?'#1a5a2a':'#5a1a2a';cx.lineWidth=0.6;
    cx.beginPath();cx.moveTo(x,yS(k[2]));cx.lineTo(x,yS(k[1]));cx.stroke();
    cx.fillStyle=up?'#1a3a1a55':'#3a1a1a55';
    const oY=yS(Math.max(k[0],k[3])),cY=yS(Math.min(k[0],k[3]));
    cx.fillRect(x-bw/2,oY,bw,Math.max(1,cY-oY));
  }}
  
  // === Pool segments ===
  if(showPool){{
    for(let i=0;i<pool.length;i++){{
      const s=pool[i];
      if(s.be<ws-10||s.bs>curBar+FUT)continue;
      const alpha=Math.max(0.1, 0.5*(1-i/pool.length));
      const lw=Math.max(0.3, 1.5*(1-i/pool.length));
      const isFusion=s.src==='f';
      cx.strokeStyle=isFusion?
        'rgba(180,120,255,'+alpha+')':
        'rgba(150,150,180,'+alpha*0.5+')';
      cx.lineWidth=lw;
      cx.setLineDash(isFusion?[8,4]:[3,2]);
      cx.beginPath();cx.moveTo(xS(s.bs),yS(s.ps));cx.lineTo(xS(s.be),yS(s.pe));cx.stroke();
    }}
  }}
  
  // === Merge snapshots ===
  if(showSnap){{
    let ampIdx=0, latIdx=0;
    for(const snap of snaps){{
      const pts=snap.pts;
      if(pts.length<2)continue;
      let col,lw,dash=[];
      if(snap.t==='b'){{
        col=snapColors.b.c.replace('A','0.25');lw=snapColors.b.lw;
      }}else if(snap.t==='a'){{
        const sc=snapColors.a[Math.min(ampIdx,snapColors.a.length-1)];
        col=sc.c.replace('A',String(0.35+ampIdx*0.08));lw=sc.lw;
        ampIdx++;
      }}else{{
        const sc=snapColors.l[Math.min(latIdx,snapColors.l.length-1)];
        col=sc.c.replace('A','0.85');lw=sc.lw;
        dash=[8+latIdx*2, 4+latIdx];
        latIdx++;
      }}
      cx.strokeStyle=col;cx.lineWidth=lw;cx.setLineDash(dash);
      cx.beginPath();
      let started=false;
      for(const pt of pts){{
        if(pt[0]<ws-20||pt[0]>curBar+FUT+20)continue;
        if(!started){{cx.moveTo(xS(pt[0]),yS(pt[1]));started=true;}}
        else cx.lineTo(xS(pt[0]),yS(pt[1]));
      }}
      cx.stroke();
      
      // 拐点标记 (高级快照)
      if(pts.length<=30&&snap.t!=='b'){{
        for(const pt of pts){{
          if(pt[0]<ws||pt[0]>curBar)continue;
          cx.fillStyle=col;cx.beginPath();
          if(snap.t==='a')cx.arc(xS(pt[0]),yS(pt[1]),2,0,Math.PI*2);
          else{{const x=xS(pt[0]),y=yS(pt[1]);cx.moveTo(x,y-3);cx.lineTo(x+3,y);cx.lineTo(x,y+3);cx.lineTo(x-3,y);}}
          cx.fill();
        }}
      }}
    }}
  }}
  cx.setLineDash([]);
  
  // === Important pivots ===
  if(showPvt){{
    for(let i=0;i<pvts.length;i++){{
      const pv=pvts[i];
      if(pv.b<ws||pv.b>curBar)continue;
      const x=xS(pv.b),y=yS(pv.p);
      const r=Math.max(2, 5-i*0.15);
      const alpha=Math.max(0.3, 1-i*0.025);
      cx.fillStyle=pv.d===1?'rgba(255,68,68,'+alpha+')':'rgba(68,255,68,'+alpha+')';
      cx.strokeStyle='rgba(255,255,255,'+alpha*0.5+')';cx.lineWidth=0.8;
      cx.beginPath();cx.arc(x,y,r,0,Math.PI*2);cx.fill();cx.stroke();
      // Label
      if(i<15){{
        const label=(pv.d===1?'H':'L')+(i+1);
        cx.fillStyle='rgba(255,255,255,'+alpha*0.7+')';cx.font='8px monospace';cx.textAlign='center';
        cx.fillText(label,x,pv.d===1?y-r-3:y+r+8);
        cx.fillText(pv.p.toFixed(4),x,pv.d===1?y-r-11:y+r+16);
      }}
    }}
    cx.textAlign='start';
  }}
  
  // === Predictions ===
  if(showPred){{
    for(let i=0;i<preds.length;i++){{
      const pr=preds[i];
      const isMirror=pr.tp==='m';
      const alpha=Math.max(0.2, 0.85-i*0.03);
      const lw=Math.max(0.8, 2.5-i*0.08);
      const aCol=isMirror?'rgba(255,180,60,'+alpha+')':'rgba(80,200,255,'+alpha+')';
      const pCol=pr.pd===1?'rgba(80,255,120,'+alpha+')':'rgba(255,80,100,'+alpha+')';
      
      // A段
      cx.strokeStyle=aCol;cx.lineWidth=lw;cx.setLineDash([]);
      cx.beginPath();cx.moveTo(xS(pr.as),yS(pr.aps));cx.lineTo(xS(pr.ae),yS(pr.ape));cx.stroke();
      
      // B段 (center only)
      if(!isMirror&&pr.bs>0){{
        cx.strokeStyle='rgba(180,180,200,'+alpha*0.3+')';cx.lineWidth=lw*0.5;cx.setLineDash([3,2]);
        cx.beginPath();cx.moveTo(xS(pr.bs),yS(pr.bps));cx.lineTo(xS(pr.be),yS(pr.bpe));cx.stroke();
      }}
      
      // C' 预测线
      cx.strokeStyle=pCol;cx.lineWidth=lw*1.3;cx.setLineDash([8,4]);
      cx.beginPath();cx.moveTo(xS(pr.psb),yS(pr.psp));cx.lineTo(xS(pr.ptb),yS(pr.ptp));cx.stroke();
      
      // 箭头
      const ax2=xS(pr.ptb),ay2=yS(pr.ptp);
      const ang=Math.atan2(ay2-yS(pr.psp),ax2-xS(pr.psb));
      cx.setLineDash([]);cx.lineWidth=lw;cx.strokeStyle=pCol;
      cx.beginPath();
      cx.moveTo(ax2,ay2);cx.lineTo(ax2-7*Math.cos(ang-0.35),ay2-7*Math.sin(ang-0.35));
      cx.moveTo(ax2,ay2);cx.lineTo(ax2-7*Math.cos(ang+0.35),ay2-7*Math.sin(ang+0.35));
      cx.stroke();
      
      // 目标价水平线
      if(i<10){{
        cx.strokeStyle=pCol.replace(/[\\d.]+\\)$/,'0.15)');
        cx.lineWidth=0.5;cx.setLineDash([3,6]);
        cx.beginPath();cx.moveTo(xS(pr.psb),yS(pr.ptp));cx.lineTo(xS(pr.ptb+10),yS(pr.ptp));cx.stroke();
      }}
      
      // 目标价标签
      if(i<15){{
        cx.fillStyle=pCol;cx.font='8px monospace';cx.textAlign='left';
        const label=(isMirror?'M':'C')+(i+1);
        cx.fillText(label+' '+pr.ptp.toFixed(4),ax2+4,ay2+3);
      }}
      
      // 离差标注 (当前K线 vs 预测期望)
      if(i<10&&pr.pt>0){{
        const elapsed=curBar-pr.psb;
        if(elapsed>0&&elapsed<=pr.pt&&curBar<N){{
          const t=elapsed/pr.pt;
          const eNow=pr.psp+t*(pr.ptp-pr.psp);
          const ac=K[curBar][3];
          const dev=(ac-eNow)/Math.max(pr.pa,0.0001);
          const dc=Math.abs(dev)<0.3?'#44ff88':Math.abs(dev)<1.0?'#ffaa33':'#ff4466';
          cx.strokeStyle=dc;cx.lineWidth=0.6;cx.setLineDash([1,1]);
          cx.beginPath();cx.moveTo(xS(curBar)+2+i*2,yS(eNow));cx.lineTo(xS(curBar)+2+i*2,yS(ac));cx.stroke();
        }}
      }}
    }}
  }}
  cx.setLineDash([]);cx.textAlign='start';
  
  // Current bar line
  cx.strokeStyle='#ffffff15';cx.lineWidth=1;
  cx.beginPath();cx.moveTo(xS(curBar),mg.t);cx.lineTo(xS(curBar),H-mg.b);cx.stroke();
  
  // Info bar
  const k=curBar<N?K[curBar]:[0,0,0,0];
  document.getElementById('infoBar').innerHTML=
    `Bar ${{curBar}} | O:${{k[0]}} H:${{k[1]}} L:${{k[2]}} C:${{k[3]}} | `+
    `Pool: ${{pool.length}} segs | Preds: ${{preds.length}} | `+
    `Snaps: ${{snaps.length}} levels (${{snaps.map(s=>s.lb).join(',')}})`;
  
  document.getElementById('barNum').textContent=curBar;
  document.getElementById('barSlider').value=curBar;
}}

function drawAll(){{draw();}}
function stepTo(b){{curBar=Math.max(200,Math.min(N-1,b));drawAll();}}
function stepBy(d){{stepTo(curBar+d);}}

function togglePlay(){{
  playing=!playing;
  document.getElementById('playBtn').innerHTML=playing?'&#9646;&#9646;':'&#9654;';
  if(playing){{
    playTimer=setInterval(()=>{{
      stepBy(parseInt(document.getElementById('spdSlider').value));
      if(curBar>=N-1){{playing=false;clearInterval(playTimer);
        document.getElementById('playBtn').innerHTML='&#9654;';}}
    }},60);
  }}else clearInterval(playTimer);
}}

document.addEventListener('keydown',e=>{{
  if(e.key==='ArrowRight')stepBy(1);
  else if(e.key==='ArrowLeft')stepBy(-1);
  else if(e.key===' '){{e.preventDefault();togglePlay();}}
}});

window.addEventListener('resize',drawAll);
drawAll();
</script></body></html>"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    fsize = len(html)
    print(f"Saved: {output_path} ({fsize//1024}KB = {fsize/1024/1024:.1f}MB)")


def main():
    print("=" * 60)
    print("v3.1 Visualizer — Full v3 Pipeline + Sliding Window")
    print("=" * 60)
    
    df = load_kline("/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv", limit=2000)
    print(f"数据: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]} ({len(df)} bars)")
    
    klines, frame_snapshots, frame_pool, frame_preds, frame_pivots = \
        run_windowed_pipeline(df, window=200, stride=10, max_pool=300, max_preds=50)
    
    print("\n生成HTML...")
    generate_html(klines, frame_snapshots, frame_pool, frame_preds, frame_pivots, df,
                  "/home/ubuntu/stage2_abc/merge_v31.html")


if __name__ == '__main__':
    main()
