#!/usr/bin/env python3
"""
FSD可视化 v5 — 基于FSD引擎的统一状态可视化

单窗口全景:
- K线 + ZG拐点
- 所有活跃轨迹 (A段 + 预测C' + 期望轨迹 + 模长边界)
- 实时离差标注 (每条轨迹)
- 底部状态栏: 共识方向, best deviation, 活跃轨迹数
- 上帝标注 (可开关): 未来最优方向箭头

数据流: FSD Engine → 逐帧snapshot → 增量编码 → HTML/JS
"""

import json
import sys
import time
import math
import numpy as np
sys.path.insert(0, '/home/ubuntu/stage2_abc')
from merge_engine_v3 import load_kline
from fsd_engine import FSDEngine, OracleLabeler


def run_engine(df, limit_bars=None):
    """运行FSD引擎, 收集每帧数据"""
    n = limit_bars or len(df)
    highs = df['high'].values.astype(float)[:n]
    lows = df['low'].values.astype(float)[:n]
    opens = df['open'].values.astype(float)[:n]
    closes = df['close'].values.astype(float)[:n]
    
    engine = FSDEngine(start_pred=200, max_trajs=30, pred_horizon=50,
                       fusion_window=200, fusion_stride=10)
    
    # 每帧数据
    klines = []        # [o,h,l,c]
    frame_pivots = []  # confirmed pivots at each frame
    frame_trajs = []   # active trajectories at each frame
    frame_meta = []    # meta info per frame
    
    t0 = time.time()
    for i in range(n):
        snap = engine.step(highs[i], lows[i], opens[i], closes[i])
        
        klines.append([
            round(opens[i], 5), round(highs[i], 5),
            round(lows[i], 5), round(closes[i], 5)
        ])
        
        # 拐点: confirmed + tentative
        pvts = [(p['bar'], round(p['price'], 5), p['dir']) for p in engine.zg.pivots]
        if engine.zg.tentative:
            t = engine.zg.tentative
            pvts.append((t['bar'], round(t['price'], 5), t['dir']))
        frame_pivots.append(pvts)
        
        # 轨迹
        trajs = []
        for tr in snap.trajs:
            trajs.append({
                'id': tr.traj_id,
                't': tr.pred_type[0],  # m/c
                'pd': tr.pred_dir,
                # A段
                'as': tr.a_start, 'ae': tr.a_end,
                'aps': round(tr.a_price_start, 5), 'ape': round(tr.a_price_end, 5),
                # 预测C'
                'psb': tr.pred_start_bar, 'psp': round(tr.pred_start_price, 5),
                'ptb': tr.pred_target_bar, 'ptp': round(tr.pred_target_price, 5),
                'pa': round(tr.pred_amp, 5), 'pt': tr.pred_time,
                # B段 (center)
                'bs': tr.b_start, 'be': tr.b_end,
                'bps': round(tr.b_price_start, 5), 'bpe': round(tr.b_price_end, 5),
                # 动态状态
                'prg': round(tr.progress, 4),
                'dev': round(tr.deviation, 4),
                'mfe': round(tr.max_favorable, 4),
                'mae': round(tr.max_adverse, 4),
                'sc': round(tr.score, 5),
            })
        frame_trajs.append(trajs)
        
        frame_meta.append({
            'nt': snap.n_active_trajs,
            'nb': snap.n_bullish,
            'ns': snap.n_bearish,
            'cd': snap.consensus_dir,
            'bd': round(snap.best_deviation, 4),
            'bp': round(snap.best_progress, 4),
        })
        
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  bar {i+1}/{n} ({elapsed:.1f}s) trajs={snap.n_active_trajs}")
    
    elapsed = time.time() - t0
    print(f"完成: {n} bars, {elapsed:.1f}s")
    
    # 上帝标注 (复用第一遍的snapshots)
    print("上帝标注...")
    # 需要收集snapshots for oracle
    snaps_for_oracle = []
    engine2 = FSDEngine(start_pred=200, max_trajs=30, pred_horizon=50,
                        fusion_window=200, fusion_stride=10)
    for i in range(n):
        snap = engine2.step(highs[i], lows[i], opens[i], closes[i])
        snaps_for_oracle.append(snap)
    oracle_labels = OracleLabeler.label_batch(snaps_for_oracle, highs, lows, closes)
    
    return klines, frame_pivots, frame_trajs, frame_meta, oracle_labels, df


def generate_html(klines, frame_pivots, frame_trajs, frame_meta, oracle_labels, df, output_path):
    """生成单窗口FSD可视化HTML"""
    n = len(klines)
    
    # === 增量编码 ===
    
    # 拐点: 拆分为 confirmed (append-only) + tentative (每帧变化)
    # confirmed pivots: 只记录新增事件 {bar_frame: [bar,price,dir]}
    # tentative: 记录变化帧 {bar_frame: [bar,price,dir] or null}
    pvt_confirmed_events = {}  # frame → [bar, price, dir] (new confirmed pivot)
    pvt_tentative_changes = {}  # frame → [bar, price, dir] or null
    prev_confirmed_len = 0
    prev_tent = None
    for i in range(n):
        fp = frame_pivots[i]
        # frame_pivots[i] includes confirmed + tentative(last element if exists)
        # We passed confirmed pivots from engine.zg.pivots, tentative from engine.zg.tentative
        # In run_engine, we appended tentative at the end. Need to tell them apart.
        # The confirmed pivots are engine.zg.pivots, tentative was appended at the end.
        # Since we don't have a flag, we use a heuristic: 
        # the data was built as: pvts = [(p['bar'],price,dir) for p in engine.zg.pivots] + maybe tentative
        # But we don't know which part is confirmed vs tentative in frame_pivots.
        # Simpler approach: just store the full list but MUCH more compressed.
        # Key insight: most of the confirmed pivots are OLD and far off-screen.
        # We only need pivots visible in the window [curBar-WIN, curBar+FUT].
        pass
    
    # Actually, let's just compress pivots by only storing visible window pivots
    # For each change-frame, store pivots that are within window range
    # This drastically reduces size since we only need ~150-200 bar range
    WIN = 150
    FUT = 50
    pvt_changes = {}
    prev_pvt_key = None
    for i in range(n):
        ws = max(0, i - WIN + 1)
        we = min(n - 1, i + FUT)
        # Filter to visible pivots only
        visible = [p for p in frame_pivots[i] if p[0] >= ws - 20 and p[0] <= we + 20]
        key = str(visible)
        if key != prev_pvt_key:
            pvt_changes[str(i)] = visible
            prev_pvt_key = key
    
    # 轨迹: 变化帧
    traj_changes = {}
    prev_traj = None
    for i in range(n):
        key = json.dumps(frame_trajs[i], separators=(',', ':'))
        if key != prev_traj:
            traj_changes[str(i)] = frame_trajs[i]
            prev_traj = key
    
    # Meta: 变化帧
    meta_changes = {}
    prev_meta = None
    for i in range(n):
        key = json.dumps(frame_meta[i], separators=(',', ':'))
        if key != prev_meta:
            meta_changes[str(i)] = frame_meta[i]
            prev_meta = key
    
    # Oracle: 变化帧 (相邻相同则跳过)
    oracle_data = {}
    prev_orc = None
    for i in range(n):
        lb = oracle_labels[i]
        orc = {
            'd10': lb.get('dir_10', 0),
            'd50': lb.get('dir_50', 0),
            'mu50': lb.get('mfe_up_50', 0),
            'md50': lb.get('mfe_dn_50', 0),
            'btd': lb.get('best_tp_dir_50', 0),
            'rr': lb.get('best_rr_50', 0),
        }
        key = json.dumps(orc, separators=(',', ':'))
        if key != prev_orc:
            oracle_data[str(i)] = orc
            prev_orc = key
    
    # 统计
    kj = json.dumps(klines, separators=(',', ':'))
    pj = json.dumps(pvt_changes, separators=(',', ':'))
    tj = json.dumps(traj_changes, separators=(',', ':'))
    mj = json.dumps(meta_changes, separators=(',', ':'))
    oj = json.dumps(oracle_data, separators=(',', ':'))
    
    print(f"  增量: {len(pvt_changes)} pvt帧, {len(traj_changes)} traj帧, "
          f"{len(meta_changes)} meta帧, {len(oracle_data)} oracle帧")
    print(f"  大小: K={len(kj)//1024}KB PVT={len(pj)//1024}KB TRAJ={len(tj)//1024}KB "
          f"META={len(mj)//1024}KB ORACLE={len(oj)//1024}KB")
    
    dt_start = str(df['datetime'].iloc[0])
    dt_end = str(df['datetime'].iloc[-1])
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>FSD v5 | EURUSD H1</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0a0a14; color: #eee; font-family: 'Consolas','Courier New',monospace; }}
.top {{ background: #12121f; padding: 4px 10px; font-size: 11px; border-bottom: 1px solid #252540; display: flex; justify-content: space-between; }}
.controls {{ display: flex; align-items: center; gap: 4px; padding: 3px 8px; background: #0e0e1a; }}
.btn {{ background: #22224a; color: #aac; border: 1px solid #335; padding: 3px 8px; border-radius: 3px; cursor: pointer; font-size: 11px; font-family: inherit; }}
.btn:hover {{ background: #33336a; }}
.slider-box {{ display: inline-flex; align-items: center; gap: 3px; font-size: 11px; color: #888; }}
.slider-box input[type=range] {{ width: 400px; }}
#mainCanvas {{ width: 100%; display: block; background: #0d0d18; }}
#stateBar {{ background: #12121f; padding: 4px 10px; font-size: 11px; border-top: 1px solid #252540; display: flex; gap: 15px; min-height: 22px; }}
.sb-item {{ display: inline-flex; gap: 4px; }}
.sb-val {{ font-weight: bold; }}
#trajPanel {{ background: #0e0e1a; padding: 4px 10px; font-size: 10px; border-top: 1px solid #1a1a30; max-height: 120px; overflow-y: auto; }}
.trow {{ display: flex; gap: 8px; padding: 1px 0; border-bottom: 1px solid #151528; }}
.trow:hover {{ background: #1a1a30; }}
label.chk {{ font-size: 11px; color: #888; cursor: pointer; }}
</style></head><body>

<div class="top">
  <span>FSD v5 | EURUSD H1 | {dt_start} ~ {dt_end} | {n} bars</span>
  <span>
    <label class="chk"><input type="checkbox" id="showOracle" onchange="drawAll()"> Oracle</label>
    <label class="chk"><input type="checkbox" id="showBoundary" checked onchange="drawAll()"> Boundary</label>
    <label class="chk"><input type="checkbox" id="showDeviation" checked onchange="drawAll()"> Deviation</label>
  </span>
</div>
<div class="controls">
  <button class="btn" onclick="stepTo(50)">|&lt;</button>
  <button class="btn" onclick="stepBy(-1)">&lt;</button>
  <button class="btn" id="playBtn" onclick="togglePlay()">&#9654;</button>
  <button class="btn" onclick="stepBy(1)">&gt;</button>
  <button class="btn" onclick="stepTo(N-1)">&gt;|</button>
  <span class="slider-box">
    <input type="range" id="barSlider" min="50" max="{n-1}" value="50"
         oninput="stepTo(parseInt(this.value))">
    <span id="barNum">50</span>/{n-1}
  </span>
  <span class="slider-box" style="color:#555;">
    Spd:<input type="range" id="spdSlider" min="1" max="10" value="3" style="width:60px"
         oninput="this.nextElementSibling.textContent=this.value"><span>3</span>
  </span>
</div>

<canvas id="mainCanvas"></canvas>
<div id="stateBar"></div>
<div id="trajPanel"></div>

<script>
const K={kj};
const N=K.length;
const WIN=150, FUT=50, TOTALW=WIN+FUT;

const PVT_C={pj};
const TRAJ_C={tj};
const META_C={mj};
const ORC={oj};

const pvtK=Object.keys(PVT_C).map(Number).sort((a,b)=>a-b);
const trajK=Object.keys(TRAJ_C).map(Number).sort((a,b)=>a-b);
const metaK=Object.keys(META_C).map(Number).sort((a,b)=>a-b);

function findLE(keys,target){{
  let lo=0,hi=keys.length-1,r=-1;
  while(lo<=hi){{const m=(lo+hi)>>1;if(keys[m]<=target){{r=m;lo=m+1;}}else hi=m-1;}}
  return r>=0?keys[r]:-1;
}}
function getP(bar){{const k=findLE(pvtK,bar);return k>=0?PVT_C[k]:[];}}
function getT(bar){{const k=findLE(trajK,bar);return k>=0?TRAJ_C[k]:[];}}
function getM(bar){{const k=findLE(metaK,bar);return k>=0?META_C[k]:{{nt:0,nb:0,ns:0,cd:0,bd:0,bp:0}};}}
function getO(bar){{return ORC[bar]||{{d10:0,d50:0,mu50:0,md50:0,btd:0,rr:0}};}}

let curBar=50,playing=false,playTimer=null;

// 轨迹颜色
function trajColor(tr,alpha){{
  if(tr.t==='m') return 'rgba(255,200,60,'+alpha+')';
  return 'rgba(100,220,255,'+alpha+')';
}}
function predColor(tr,alpha){{
  if(tr.pd===1) return 'rgba(80,255,120,'+alpha+')';
  return 'rgba(255,80,100,'+alpha+')';
}}

function draw(){{
  const cv=document.getElementById('mainCanvas');
  const cw=window.innerWidth;
  const ch=window.innerHeight-110;
  cv.width=cw; cv.height=Math.max(ch,300);
  const cx=cv.getContext('2d');
  const W=cv.width,H=cv.height;
  const mg={{l:56,r:8,t:14,b:14}};
  const pw=W-mg.l-mg.r, ph=H-mg.t-mg.b;
  
  const ws=Math.max(0,curBar-WIN+1);
  const trajs=getT(curBar);
  const pvts=getP(curBar);
  const meta=getM(curBar);
  const orc=getO(curBar);
  
  // 价格范围
  let pMn=Infinity,pMx=-Infinity;
  for(let b=ws;b<=curBar&&b<N;b++){{pMn=Math.min(pMn,K[b][2]);pMx=Math.max(pMx,K[b][1]);}}
  for(const tr of trajs){{
    pMn=Math.min(pMn,tr.ptp,tr.psp);pMx=Math.max(pMx,tr.ptp,tr.psp);
    // 边界扩展
    if(tr.pa>0){{pMn=Math.min(pMn,tr.psp-tr.pa*0.6);pMx=Math.max(pMx,tr.psp+tr.pa*0.6);}}
  }}
  if(pMn===Infinity){{pMn=1.0;pMx=1.1;}}
  const rng=pMx-pMn; pMn-=rng*0.03; pMx+=rng*0.03;
  
  function xS(b){{return mg.l+((b-ws)/TOTALW)*pw;}}
  function yS(p){{return mg.t+ph-((p-pMn)/(pMx-pMn))*ph;}}
  
  // BG
  cx.fillStyle='#0d0d18'; cx.fillRect(0,0,W,H);
  // Future zone
  cx.fillStyle='#10101f'; cx.fillRect(xS(curBar+1),mg.t,W-mg.r-xS(curBar+1),ph);
  // Current bar line
  cx.strokeStyle='#ffffff10'; cx.lineWidth=1;
  cx.beginPath();cx.moveTo(xS(curBar),mg.t);cx.lineTo(xS(curBar),H-mg.b);cx.stroke();
  
  // Grid
  cx.fillStyle='#444';cx.font='9px monospace';cx.strokeStyle='#161625';cx.lineWidth=0.5;
  for(let i=0;i<=5;i++){{const p=pMn+(pMx-pMn)*i/5;const y=yS(p);
    cx.beginPath();cx.moveTo(mg.l,y);cx.lineTo(W-mg.r,y);cx.stroke();
    cx.fillText(p.toFixed(4),2,y+3);}}
  
  // K-lines
  const bw=Math.max(1.2,pw/TOTALW*0.65);
  for(let b=ws;b<=curBar&&b<N;b++){{
    const k=K[b],x=xS(b),up=k[3]>=k[0];
    cx.strokeStyle=up?'#22cc44':'#ee3344';cx.lineWidth=1;
    cx.beginPath();cx.moveTo(x,yS(k[2]));cx.lineTo(x,yS(k[1]));cx.stroke();
    cx.fillStyle=up?'#22994466':'#cc224466';
    const oY=yS(Math.max(k[0],k[3])),cY=yS(Math.min(k[0],k[3]));
    cx.fillRect(x-bw/2,oY,bw,Math.max(1,cY-oY));
  }}
  
  // ZG
  if(pvts.length>=2){{
    cx.strokeStyle='#ffd740';cx.lineWidth=1.5;cx.globalAlpha=0.7;cx.setLineDash([]);
    cx.beginPath();let st=false;
    for(const pt of pvts){{if(pt[0]>curBar)break;const x=xS(pt[0]),y=yS(pt[1]);
      if(x<mg.l-20)continue;if(!st){{cx.moveTo(x,y);st=true;}}else cx.lineTo(x,y);}}
    cx.stroke();cx.globalAlpha=1;
    for(const pt of pvts){{if(pt[0]>curBar)break;const x=xS(pt[0]);if(x<mg.l-5)continue;
      cx.fillStyle=pt[2]===1?'#ff6666':'#66ff66';cx.beginPath();cx.arc(x,yS(pt[1]),2.5,0,Math.PI*2);cx.fill();}}
  }}
  
  // Oracle
  if(document.getElementById('showOracle').checked){{
    const dir50=orc.btd;
    if(dir50!==0){{
      const arrX=xS(curBar)+15, arrY=H/2;
      cx.fillStyle=dir50===1?'rgba(0,255,100,0.25)':'rgba(255,60,60,0.25)';
      cx.beginPath();
      if(dir50===1){{cx.moveTo(arrX-8,arrY+12);cx.lineTo(arrX+8,arrY+12);cx.lineTo(arrX,arrY-12);}}
      else{{cx.moveTo(arrX-8,arrY-12);cx.lineTo(arrX+8,arrY-12);cx.lineTo(arrX,arrY+12);}}
      cx.fill();
      cx.fillStyle='rgba(255,255,255,0.3)';cx.font='9px monospace';cx.textAlign='center';
      cx.fillText('RR:'+orc.rr.toFixed(1),arrX,dir50===1?arrY+22:arrY-16);
      cx.textAlign='start';
    }}
  }}
  
  const showBnd=document.getElementById('showBoundary').checked;
  const showDev=document.getElementById('showDeviation').checked;
  
  // === 轨迹 ===
  for(let i=0;i<trajs.length;i++){{
    const tr=trajs[i];
    const isMirror=tr.t==='m';
    const alpha=Math.max(0.35,0.9-i*0.04);
    const lw=Math.max(1,2.5-i*0.12);
    const tc=trajColor(tr,alpha);
    const pc=predColor(tr,alpha);
    
    // A段
    cx.strokeStyle=tc;cx.lineWidth=lw;cx.setLineDash([]);
    cx.beginPath();cx.moveTo(xS(tr.as),yS(tr.aps));cx.lineTo(xS(tr.ae),yS(tr.ape));cx.stroke();
    
    // B段 (center)
    if(!isMirror&&tr.bs>0){{
      cx.strokeStyle='rgba(180,180,200,'+(alpha*0.3)+')';cx.lineWidth=lw*0.5;cx.setLineDash([2,2]);
      cx.beginPath();cx.moveTo(xS(tr.bs),yS(tr.bps));cx.lineTo(xS(tr.be),yS(tr.bpe));cx.stroke();
    }}
    
    // 模长边界
    if(showBnd&&tr.pa>0&&tr.pt>0&&i<6){{
      const sa=tr.ptp-tr.psp;
      cx.strokeStyle=pc.replace(/[\\d.]+\\)$/,(alpha*0.18)+')');
      cx.lineWidth=0.7;cx.setLineDash([2,3]);
      const ns=Math.min(tr.pt,50);
      for(let side=-1;side<=1;side+=2){{
        cx.beginPath();
        for(let k=0;k<=ns;k++){{
          const t=k/tr.pt,bk=tr.psb+k;
          const ep=tr.psp+t*sa;
          const R=tr.pa*Math.sqrt(t)*0.5;
          const bp=ep+side*R;
          if(k===0)cx.moveTo(xS(bk),yS(bp));else cx.lineTo(xS(bk),yS(bp));
        }}
        cx.stroke();
      }}
    }}
    
    // 期望轨迹
    if(tr.pa>0&&tr.pt>0&&i<8){{
      cx.strokeStyle=pc.replace(/[\\d.]+\\)$/,(alpha*0.25)+')');
      cx.lineWidth=0.8;cx.setLineDash([1,2]);
      cx.beginPath();
      const ns=Math.min(tr.pt,50);
      for(let k=0;k<=ns;k++){{
        const t=k/tr.pt,bk=tr.psb+k,ep=tr.psp+t*(tr.ptp-tr.psp);
        if(k===0)cx.moveTo(xS(bk),yS(ep));else cx.lineTo(xS(bk),yS(ep));
      }}
      cx.stroke();
    }}
    
    // C' 预测线
    cx.strokeStyle=pc;cx.lineWidth=lw*1.3;cx.setLineDash([6,3]);
    cx.beginPath();cx.moveTo(xS(tr.psb),yS(tr.psp));cx.lineTo(xS(tr.ptb),yS(tr.ptp));cx.stroke();
    
    // 箭头
    const ax2=xS(tr.ptb),ay2=yS(tr.ptp);
    const ang=Math.atan2(ay2-yS(tr.psp),ax2-xS(tr.psb));
    cx.setLineDash([]);cx.lineWidth=lw;
    cx.beginPath();cx.moveTo(ax2,ay2);cx.lineTo(ax2-6*Math.cos(ang-0.35),ay2-6*Math.sin(ang-0.35));
    cx.moveTo(ax2,ay2);cx.lineTo(ax2-6*Math.cos(ang+0.35),ay2-6*Math.sin(ang+0.35));cx.stroke();
    
    // 离差标注
    if(showDev&&tr.pa>0&&tr.pt>0&&i<8){{
      const elapsed=curBar-tr.psb;
      if(elapsed>0&&elapsed<=tr.pt&&curBar<N){{
        const t=elapsed/tr.pt;
        const eNow=tr.psp+t*(tr.ptp-tr.psp);
        const ac=K[curBar][3];
        const dev=(ac-eNow)/tr.pa;
        const dc=Math.abs(dev)<0.3?'#44ff88':Math.abs(dev)<1.0?'#ffaa33':'#ff4466';
        // 小连线
        cx.strokeStyle=dc;cx.lineWidth=0.6;cx.setLineDash([1,1]);
        cx.beginPath();cx.moveTo(xS(curBar)+3,yS(eNow));cx.lineTo(xS(curBar)+3,yS(ac));cx.stroke();
        // 标签
        cx.fillStyle=dc;cx.font='8px monospace';cx.textAlign='left';
        cx.fillText((dev>=0?'+':'')+dev.toFixed(2),xS(curBar)+6,yS(eNow)+(dev>0?-3:10));
      }}
    }}
    
    // 目标价
    if(i<8){{
      cx.fillStyle=pc;cx.font='9px monospace';cx.textAlign='left';
      cx.fillText(tr.ptp.toFixed(4),ax2+4,ay2+3);
    }}
  }}
  cx.setLineDash([]);cx.globalAlpha=1;cx.textAlign='start';
  
  // === 状态栏 ===
  const k=curBar<N?K[curBar]:[0,0,0,0];
  const dirIcon=meta.cd===1?'\\u25B2':meta.cd===-1?'\\u25BC':'\\u25C6';
  const dirCol=meta.cd===1?'#44ff88':meta.cd===-1?'#ff4466':'#888';
  
  document.getElementById('stateBar').innerHTML=
    `<span class="sb-item">Bar <span class="sb-val">${{curBar}}</span></span>`+
    `<span class="sb-item">C:<span class="sb-val">${{k[3]}}</span></span>`+
    `<span class="sb-item" style="color:${{dirCol}}">${{dirIcon}} <span class="sb-val">consensus</span></span>`+
    `<span class="sb-item">Trajs:<span class="sb-val">${{meta.nt}}</span> (${{meta.nb}}\\u25B2 ${{meta.ns}}\\u25BC)</span>`+
    `<span class="sb-item">BestDev:<span class="sb-val" style="color:${{Math.abs(meta.bd)<0.3?'#44ff88':Math.abs(meta.bd)<1?'#ffaa33':'#ff4466'}}">${{meta.bd>=0?'+':''}}${{meta.bd.toFixed(3)}}</span></span>`+
    `<span class="sb-item" style="color:#555">Oracle50: ${{orc.btd===1?'\\u25B2':'\\u25BC'}} RR=${{orc.rr.toFixed(1)}} MFE\\u2191${{orc.mu50}}p \\u2193${{orc.md50}}p</span>`;
  
  document.getElementById('barNum').textContent=curBar;
  document.getElementById('barSlider').value=curBar;
  
  // === 轨迹详情面板 ===
  let tp='';
  for(const tr of trajs){{
    const tc2=tr.t==='m'?'#ffcc44':'#55ddff';
    const dc2=Math.abs(tr.dev)<0.3?'#44ff88':Math.abs(tr.dev)<1?'#ffaa33':'#ff4466';
    const prgBar=Math.min(Math.max(tr.prg,0),1.5);
    tp+=`<div class="trow">`+
      `<span style="color:${{tc2}};width:12px">${{tr.t==='m'?'M':'C'}}</span>`+
      `<span style="width:18px">${{tr.pd===1?'\\u25B2':'\\u25BC'}}</span>`+
      `<span style="width:60px">A[${{tr.as}}-${{tr.ae}}]</span>`+
      `<span style="width:55px;color:${{dc2}}">dev:${{tr.dev>=0?'+':''}}${{tr.dev.toFixed(2)}}</span>`+
      `<span style="width:50px">prg:${{tr.prg.toFixed(2)}}</span>`+
      `<span style="width:50px">mfe:${{tr.mfe.toFixed(2)}}</span>`+
      `<span style="width:50px">mae:${{tr.mae.toFixed(2)}}</span>`+
      `<span style="width:55px">sc:${{tr.sc.toFixed(3)}}</span>`+
      `<span style="width:70px">tp:${{tr.ptp.toFixed(4)}}</span>`+
      `<span style="flex:1;position:relative;height:8px;background:#1a1a30;border-radius:2px">`+
        `<span style="position:absolute;left:0;top:0;height:100%;width:${{Math.min(prgBar*100/1.5,100)}}%;background:${{tr.pd===1?'#2a8844':'#884422'}};border-radius:2px"></span>`+
      `</span>`+
    `</div>`;
  }}
  document.getElementById('trajPanel').innerHTML=tp||'<div style="color:#444;padding:4px">No active trajectories</div>';
}}

function drawAll(){{draw();}}
function stepTo(b){{curBar=Math.max(50,Math.min(N-1,b));drawAll();}}
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
    print("FSD Visualizer v5")
    print("=" * 60)
    
    df = load_kline("/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv", limit=2000)
    print(f"数据: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]} ({len(df)} bars)")
    
    klines, frame_pivots, frame_trajs, frame_meta, oracle_labels, df = run_engine(df)
    
    print("\n生成HTML...")
    generate_html(klines, frame_pivots, frame_trajs, frame_meta, oracle_labels, df,
                  "/home/ubuntu/stage2_abc/fsd_v5.html")


if __name__ == '__main__':
    main()
