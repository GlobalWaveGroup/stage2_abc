#!/usr/bin/env python3
"""
归并引擎 v3.2 — 四窗口多周期联动可视化

基于 v3 (merge_engine_v3.py):
  - 四窗口: M5, M15, M30, H1 (normalized数据, 无时间戳)
  - 每窗口200 bars, 支持缩放(+/-/滚轮)和平移(点击/方向键)
  - 完整保留: 归并快照, 重要性标注, fusion线段
  - 品种通过命令行指定

用法:
  python3 visualize_v3_2.py                  # 默认 EURUSD
  python3 visualize_v3_2.py XAUUSD           # 指定品种
  python3 visualize_v3_2.py EURUSD 300       # 指定品种和每窗口bar数
"""

import json, sys, os, time
import pandas as pd
sys.path.insert(0, '/home/ubuntu/stage2_abc')
from merge_engine_v3 import (
    calculate_base_zg, full_merge_engine,
    compute_pivot_importance, build_segment_pool, pool_fusion,
)

NORM_DIR = '/home/ubuntu/database2'
TFS = ['M5', 'M15', 'M30', 'H1']


def discover_symbols():
    h1_dir = os.path.join(NORM_DIR, 'H1')
    symbols = set()
    for f in os.listdir(h1_dir):
        if f.endswith('_H1_norm.csv'):
            symbols.add(f.replace('_H1_norm.csv', ''))
    return sorted(symbols)


def load_norm(symbol, tf, limit=200):
    path = os.path.join(NORM_DIR, tf, f'{symbol}_{tf}_norm.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    keep = [c for c in ['open','high','low','close'] if c in df.columns]
    if len(keep) < 4:
        return None
    df = df[keep].dropna()
    if limit and len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    return df


def process_one_tf(df, tf_name):
    """运行v3引擎, 打包为JS可用的数据"""
    t0 = time.time()
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)

    base = calculate_base_zg(high, low, rb=0.5)
    results = full_merge_engine(base)
    pivot_info = compute_pivot_importance(results, total_bars=len(df))
    pool = build_segment_pool(results, pivot_info)
    full_pool, fusion_segs, _ = pool_fusion(pool, pivot_info)

    elapsed = time.time() - t0
    print(f"  [{tf_name}] {len(df)} bars | {len(base)} base | "
          f"{len(pool)}→{len(full_pool)} pool | {elapsed:.1f}s")

    # K线
    klines = []
    for i in range(len(df)):
        row = df.iloc[i]
        klines.append([
            round(float(row['open']), 6),
            round(float(row['high']), 6),
            round(float(row['low']), 6),
            round(float(row['close']), 6),
        ])

    # 快照 (compact: 只存坐标)
    snaps = []
    for snap_type, label, pvts in results['all_snapshots']:
        pts = [[int(p[0]), round(p[1], 6)] for p in pvts]
        snaps.append([snap_type, label, pts])

    # Fusion top 100
    sorted_fus = sorted(fusion_segs, key=lambda s: -s['importance'])[:100]
    fus = []
    for s in sorted_fus:
        fus.append([
            s['bar_start'], round(s['price_start'], 6),
            s['bar_end'], round(s['price_end'], 6),
            round(s['importance'], 4), s['span'],
            round(s['amplitude'], 6), s['source_label'],
        ])

    # 重要拐点
    peaks = sorted([v for v in pivot_info.values() if v['dir'] == 1],
                   key=lambda x: -x['importance'])
    valleys = sorted([v for v in pivot_info.values() if v['dir'] == -1],
                     key=lambda x: -x['importance'])
    marks = []
    for rank, p in enumerate(peaks):
        marks.append([p['bar'], round(p['price'], 6), 1, rank+1,
                      round(p['importance'], 4), f"H{rank+1}"])
    for rank, p in enumerate(valleys):
        marks.append([p['bar'], round(p['price'], 6), -1, rank+1,
                      round(p['importance'], 4), f"L{rank+1}"])

    return {
        'tf': tf_name, 'n': len(df), 'nb': len(base), 'np': len(full_pool),
        'K': klines, 'S': snaps, 'F': fus, 'M': marks,
    }


def generate_html(symbol, tf_data, all_symbols, output_path):
    data_json = json.dumps(tf_data, separators=(',',':'))
    sym_json = json.dumps(all_symbols)

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>v3.2 | {symbol}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0a0a1a;color:#ddd;font-family:'Consolas',monospace;overflow:hidden}}
.hdr{{display:flex;align-items:center;gap:10px;padding:4px 10px;background:#0d0d20;border-bottom:1px solid #222;height:30px}}
.hdr h1{{font-size:13px;color:#7eb8da;white-space:nowrap}}
.hdr select{{background:#1a1a30;color:#8ac;border:1px solid #335;padding:1px 5px;font-family:inherit;font-size:11px;border-radius:3px}}
.ctl{{display:flex;gap:3px 8px;padding:2px 10px;background:#0c0c1e;border-bottom:1px solid #1a1a2a;align-items:center;height:26px;flex-wrap:wrap}}
.b{{background:#1a2a4a;color:#8ac;border:1px solid #335;padding:1px 5px;border-radius:3px;cursor:pointer;font-size:10px;font-family:inherit}}
.b:hover{{background:#2a3a5a}}.b.a{{background:#3a2a5a;color:#f8f;border-color:#a6a}}
.sb{{display:inline-flex;align-items:center;gap:2px;font-size:10px;color:#999}}
.sb input[type=range]{{width:60px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;width:100vw;height:calc(100vh - 82px)}}
.pn{{position:relative;border:1px solid #1a1a30;overflow:hidden}}
.pn canvas{{width:100%;height:100%;display:block;cursor:crosshair}}
.pl{{position:absolute;top:2px;left:6px;font-size:12px;font-weight:bold;pointer-events:none;z-index:2}}
.pi{{position:absolute;bottom:1px;left:6px;font-size:9px;color:#555;pointer-events:none;z-index:2}}
.hi{{position:absolute;top:2px;right:6px;font-size:10px;color:#aaa;pointer-events:none;z-index:2;text-align:right;max-width:70%}}
#abar{{display:flex;align-items:center;gap:8px;padding:3px 10px;background:#0e1225;border-bottom:1px solid #1a2a3a;height:26px;font-size:11px}}
#abar .ab-status{{color:#ff0;font-weight:bold}}
#abar .ab-info{{color:#888;font-size:10px}}
#abar .ab-hint{{color:#557;font-size:10px;margin-left:auto}}
#rpanel{{position:fixed;right:0;top:82px;width:340px;max-height:calc(100vh - 82px);background:#0d0d22;border-left:1px solid #2a2a4a;overflow-y:auto;z-index:10;padding:8px;display:none;font-size:10px}}
#rpanel h3{{color:#8af;font-size:12px;margin:6px 0 3px;border-bottom:1px solid #223}}
#rpanel .rr{{margin:2px 0;color:#bbb}}
#rpanel .rv{{color:#ff0;font-weight:bold}}
#rpanel .rg{{color:#4f4}}
#rpanel .rb{{color:#f44}}
#rpanel .rc{{color:#8cf}}
</style></head><body>
<div class="hdr">
<h1>v3.2 | {symbol}</h1>
<select id="ss" onchange="location.href='merge_v3_2_'+this.value+'.html'"></select>
<span style="font-size:10px;color:#666">{symbol} normalized | Wheel:zoom  +/-:zoom  Arrows:scroll</span>
</div>
<div class="ctl">
<button class="b a" id="bS" onclick="T('sn')">Snaps</button>
<button class="b a" id="bF" onclick="T('fu')">Fusion</button>
<button class="b a" id="bM" onclick="T('mk')">ImpPts</button>
<button class="b a" id="bV" onclick="T('vl')">Values</button>
<span class="sb" style="color:#f8f">Pk:<input type="range" id="rP" min="0" max="50" value="10" oninput="pN=+this.value;$('nP').textContent=pN;DA()"><span id="nP">10</span></span>
<span class="sb" style="color:#4f4">Vl:<input type="range" id="rV" min="0" max="50" value="10" oninput="vN=+this.value;$('nV').textContent=vN;DA()"><span id="nV">10</span></span>
<span class="sb" style="color:#c8f">Fu:<input type="range" id="rF" min="0" max="100" value="30" oninput="fN=+this.value;$('nF').textContent=fN;DA()"><span id="nF">30</span></span>
<span style="color:#333;margin-left:6px">|</span>
<button class="b" id="bABC" onclick="abcToggle()" style="color:#ff0;border-color:#660">ABC</button>
</div>
<div id="abar" style="display:none">
<span class="ab-status" id="abSt">--</span>
<span class="ab-info" id="abInf"></span>
<span class="ab-hint" id="abHint">Press [A] or click ABC to start. DblClick segment to select. Esc=reset</span>
</div>
<div class="grid">
<div class="pn" id="P0"><canvas id="C0"></canvas><div class="pl" id="L0" style="color:#5a8ab0"></div><div class="pi" id="I0"></div><div class="hi" id="H0"></div></div>
<div class="pn" id="P1"><canvas id="C1"></canvas><div class="pl" id="L1" style="color:#8ab05a"></div><div class="pi" id="I1"></div><div class="hi" id="H1r"></div></div>
<div class="pn" id="P2"><canvas id="C2"></canvas><div class="pl" id="L2" style="color:#b08a5a"></div><div class="pi" id="I2"></div><div class="hi" id="H2r"></div></div>
<div class="pn" id="P3"><canvas id="C3"></canvas><div class="pl" id="L3" style="color:#b05a8a"></div><div class="pi" id="I3"></div><div class="hi" id="H3r"></div></div>
</div>
<div id="rpanel"></div>
<script>
const D={data_json};
const SY={sym_json};
const CU='{symbol}';
const $=id=>document.getElementById(id);
const ly={{sn:1,fu:1,mk:1,vl:1}};
let pN=10,vN=10,fN=30,aP=0;
function T(k){{ly[k]=1-ly[k];const m={{sn:'bS',fu:'bF',mk:'bM',vl:'bV'}};
$(m[k]).classList.toggle('a',!!ly[k]);DA();}}
const cv=[0,1,2,3].map(i=>$('C'+i));
const cx=cv.map(c=>c.getContext('2d'));
const pn=[0,1,2,3].map(i=>$('P'+i));
const V=D.map((t,i)=>{{const nb=t.K.length;return{{i,s:0,e:nb,tf:t}};}});
function rsz(){{cv.forEach((c,i)=>{{const r=pn[i].getBoundingClientRect();
c.width=r.width*devicePixelRatio;c.height=r.height*devicePixelRatio;
c.style.width=r.width+'px';c.style.height=r.height+'px';
cx[i].setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);}});DA();}}
window.addEventListener('resize',rsz);
const MG={{l:52,r:6,t:18,b:12}};
const tfc=['#5a8ab0','#8ab05a','#b08a5a','#b05a8a'];

function scol(si,S){{const s=S[si];const tp=s[0];
if(tp==='base')return'#FFD700';
if(tp==='amp'){{let a=0;for(let j=0;j<si;j++)if(S[j][0]==='amp')a++;
const n=S.filter(x=>x[0]==='amp').length;const t=a/Math.max(n-1,1);
return`rgb(${{255-t*80|0}},${{100-t*60|0}},30)`;}}
if(tp==='lat'){{let l=0;for(let j=0;j<si;j++)if(S[j][0]==='lat')l++;
const n=S.filter(x=>x[0]==='lat').length;const t=l/Math.max(n-1,1);
return`rgb(${{50+t*50|0}},${{220-t*80|0}},${{255-t*40|0}})`;}}
return'#888';}}

function fcol(r,n){{const t=r/Math.max(n-1,1);
return`rgb(${{200-t*100|0}},${{80-t*50|0}},${{255-t*80|0}})`;}}

/* ==================== ABC ANNOTATION SYSTEM ==================== */
let abcMode=false;  // annotation mode active
// abcState: 'idle','selA','selB','selC','done'
let abcState='idle';
let abcPanel=-1;  // which panel is the annotation target
// Each seg: {{b0,p0,b1,p1}} in bar/price coords
let segA=null, segB=null, segC=null;
// C' predictions (computed after A+B selected)
let cPred=null;  // {{mirror:{{b0,p0,b1,p1}}, modArc:{{cx,cy,r}}, triConv:..., triDiv:...}}
// Saved annotations
const abcSaved=[];

function abcToggle(){{
  abcMode=!abcMode;
  $('bABC').classList.toggle('a',abcMode);
  $('abar').style.display=abcMode?'flex':'none';
  if(abcMode){{abcReset();}}
  else{{abcState='idle';segA=segB=segC=cPred=null;$('rpanel').style.display='none';}}
  // Adjust grid height
  document.querySelector('.grid').style.height=abcMode?'calc(100vh - 82px)':'calc(100vh - 56px)';
  rsz();
}}

function abcReset(){{
  abcState='selA';segA=segB=segC=cPred=null;abcPanel=-1;
  $('abSt').textContent='SELECT A';$('abSt').style.color='#f44';
  $('abInf').textContent='Click a panel, then double-click a segment for A';
  $('abHint').textContent='DblClick=select segment | Esc=reset | Right-click=cancel last';
  $('rpanel').style.display='none';$('rpanel').innerHTML='';
  DA();
}}

// Build all snappable segments for a panel (fusion + snapshot zigzag edges)
function getSnappableSegs(pi){{
  const tf=V[pi].tf, segs=[];
  // Fusion segments: F=[b0,p0,b1,p1,imp,span,amp,label]
  const n=Math.min(fN,tf.F.length);
  for(let i=0;i<n;i++){{
    const f=tf.F[i];
    segs.push({{b0:f[0],p0:f[1],b1:f[2],p1:f[3],imp:f[4],src:'fus#'+i+'['+f[7]+']',type:'fusion'}});
  }}
  // Snapshot zigzag edges (each consecutive pair of points = a segment)
  for(let si=0;si<tf.S.length;si++){{
    const s=tf.S[si],tp=s[0],pts=s[2];
    if(tp==='base')continue;  // skip base (too noisy)
    for(let j=0;j<pts.length-1;j++){{
      segs.push({{b0:pts[j][0],p0:pts[j][1],b1:pts[j+1][0],p1:pts[j+1][1],
                  imp:0,src:tp+'_'+s[1]+'#'+j,type:'snap'}});
    }}
  }}
  return segs;
}}

// Find nearest segment to mouse position (px coords)
function findNearestSeg(pi, mx, my){{
  const v=V[pi];
  if(!v._xS)return null;
  const segs=getSnappableSegs(pi);
  let best=null, bestD=25; // 25px threshold
  for(const s of segs){{
    if(s.b1<v._vs||s.b0>v._ve)continue;
    const x1=v._xS(s.b0),y1=v._yS(s.p0),x2=v._xS(s.b1),y2=v._yS(s.p1);
    const dx=x2-x1,dy=y2-y1,l2=dx*dx+dy*dy;
    if(!l2)continue;
    let t=((mx-x1)*dx+(my-y1)*dy)/l2;
    t=Math.max(0,Math.min(1,t));
    const d=Math.hypot(mx-x1-t*dx,my-y1-t*dy);
    if(d<bestD){{bestD=d;best=s;}}
  }}
  return best;
}}

// Compute C' predictions from A and B
function computeCpred(){{
  if(!segA||!segB)return null;
  const A={{b0:segA.b0,p0:segA.p0,b1:segA.b1,p1:segA.p1}};
  const B={{b0:segB.b0,p0:segB.p0,b1:segB.b1,p1:segB.p1}};

  // A vector
  const dA_b=A.b1-A.b0, dA_p=A.p1-A.p0;
  // B vector
  const dB_b=B.b1-B.b0, dB_p=B.p1-B.p0;
  // B endpoint = C start
  const cStart_b=B.b1, cStart_p=B.p1;

  // Price range for normalization (to make bar and price comparable in modulus)
  const v=V[abcPanel], K=v.tf.K;
  let pmin=1e9,pmax=-1e9;
  for(let i=0;i<K.length;i++){{if(K[i][2]<pmin)pmin=K[i][2];if(K[i][1]>pmax)pmax=K[i][1];}}
  const pRange=pmax-pmin||1;
  const nBars=K.length||200;
  // Normalize: bar_norm = bar/nBars, price_norm = price/pRange
  // Modulus in normalized space
  const modA=Math.sqrt((dA_b/nBars)**2+(dA_p/pRange)**2);
  const modB=Math.sqrt((dB_b/nBars)**2+(dB_p/pRange)**2);
  const ampA=Math.abs(dA_p);
  const ampB=Math.abs(dB_p);
  const spanA=Math.abs(dA_b);
  const spanB=Math.abs(dB_b);
  // A direction (sign of price change)
  const dirA=dA_p>0?1:-1;

  // --- C' Mirror (geometric center symmetry): C' = A mirrored through B endpoint ---
  // C' has same direction as A (impulse continues after correction)
  // C'_mirror: same dA vector starting from B.end
  const mirror={{
    b0:cStart_b, p0:cStart_p,
    b1:cStart_b+dA_b, p1:cStart_p+dA_p
  }};

  // --- C' Triangle convergent: amplitude shrinks, C' end toward B midline ---
  // Convergent: |dC_p| = |dA_p| * 0.618 (shrink), same direction, same time span
  const triConv={{
    b0:cStart_b, p0:cStart_p,
    b1:cStart_b+dA_b, p1:cStart_p+dA_p*0.618
  }};

  // --- C' Triangle divergent: amplitude expands ---
  const triDiv={{
    b0:cStart_b, p0:cStart_p,
    b1:cStart_b+dA_b, p1:cStart_p+dA_p*1.618
  }};

  // --- Modulus arc: circle centered at B.end, radius = modA (in normalized coords) ---
  // We store in bar/price coords but draw with normalization
  const modArc={{
    cb:cStart_b, cp:cStart_p,
    r_bars:modA*nBars,    // radius in bar units
    r_price:modA*pRange,  // radius in price units
    modA:modA, nBars:nBars, pRange:pRange
  }};

  return {{
    mirror, triConv, triDiv, modArc,
    metrics:{{modA,modB,ampA,ampB,spanA,spanB,dirA,
              modRatio:modB/modA,ampRatio:ampB/ampA,spanRatio:spanB/spanA}}
  }};
}}

// Evaluate C actual vs C' predictions
function evaluateC(){{
  if(!segA||!segB||!segC||!cPred)return null;
  const C={{b0:segC.b0,p0:segC.p0,b1:segC.b1,p1:segC.p1}};
  const dC_b=C.b1-C.b0, dC_p=C.p1-C.p0;
  const m=cPred.metrics;
  const nB=m.spanA>0?m.spanA:1;  // use A as reference scale
  const nP=m.ampA>0?m.ampA:0.001;

  const ampC=Math.abs(dC_p);
  const spanC=Math.abs(dC_b);
  const modC=Math.sqrt((dC_b/cPred.modArc.nBars)**2+(dC_p/cPred.modArc.pRange)**2);

  // Direction check: C should go same direction as A
  const dirC=dC_p>0?1:-1;
  const dirMatch=dirC===m.dirA;

  // Mirror deviation: how close is C endpoint to mirror prediction
  const mirDev_b=Math.abs((C.b1-cPred.mirror.b1))/Math.max(m.spanA,1);
  const mirDev_p=Math.abs((C.p1-cPred.mirror.p1))/Math.max(m.ampA,0.0001);
  const mirDev=Math.sqrt(mirDev_b**2+mirDev_p**2);

  // Modulus symmetry: mod_C / mod_A
  const modRatio_CA=modC/Math.max(m.modA,1e-9);

  // Amplitude symmetry: amp_C / amp_A
  const ampRatio_CA=ampC/Math.max(m.ampA,1e-9);

  // Span symmetry: span_C / span_A
  const spanRatio_CA=spanC/Math.max(m.spanA,1);

  // Convergence check
  const convDev_p=Math.abs((C.p1-cPred.triConv.p1))/Math.max(m.ampA,0.0001);

  // Scoring
  let score=0, notes=[];
  if(dirMatch){{score+=20;notes.push('Direction OK');}}
  else{{notes.push('Direction WRONG');}}
  if(modRatio_CA>=0.7&&modRatio_CA<=1.3){{score+=30;notes.push('ModSym GOOD ('+modRatio_CA.toFixed(3)+')');}}
  else if(modRatio_CA>=0.5&&modRatio_CA<=1.5){{score+=15;notes.push('ModSym FAIR ('+modRatio_CA.toFixed(3)+')');}}
  else{{notes.push('ModSym POOR ('+modRatio_CA.toFixed(3)+')');}}
  if(ampRatio_CA>=0.7&&ampRatio_CA<=1.3){{score+=20;notes.push('AmpSym GOOD');}}
  else if(ampRatio_CA>=0.5&&ampRatio_CA<=2.0){{score+=10;notes.push('AmpSym FAIR');}}
  if(mirDev<0.5){{score+=15;notes.push('MirrorClose');}}
  else if(mirDev<1.0){{score+=8;notes.push('MirrorMid');}}
  if(spanRatio_CA>=0.5&&spanRatio_CA<=2.0){{score+=15;notes.push('SpanSym OK');}}

  return {{
    dirMatch, modC, modRatio_CA, ampC, ampRatio_CA, spanC, spanRatio_CA,
    mirDev, convDev_p, score, notes,
    A_metrics:{{amp:m.ampA,span:m.spanA,mod:m.modA,dir:m.dirA}},
    B_metrics:{{amp:m.ampB,span:m.spanB,mod:m.modB,ratio_mod:m.modRatio,ratio_amp:m.ampRatio}},
    C_metrics:{{amp:ampC,span:spanC,mod:modC,dir:dirC}}
  }};
}}

function showResults(ev){{
  const rp=$('rpanel');
  rp.style.display='block';
  let h='<h3>ABC Symmetry Evaluation</h3>';
  h+='<div class="rr">Score: <span class="rv">'+ev.score+'/100</span> '
    +(ev.score>=60?'<span class="rg">SIGNIFICANT</span>':'<span class="rb">WEAK</span>')+'</div>';
  h+='<h3>Direction</h3>';
  h+='<div class="rr">'+(ev.dirMatch?'<span class="rg">A=C direction match</span>':'<span class="rb">A/C direction MISMATCH</span>')+'</div>';
  h+='<h3>Modulus (mod=sqrt(bar2+price2) norm)</h3>';
  h+='<div class="rr">mod_A='+ev.A_metrics.mod.toFixed(5)+' mod_C='+ev.modC.toFixed(5)+'</div>';
  h+='<div class="rr">mod_C/mod_A = <span class="rv">'+ev.modRatio_CA.toFixed(4)+'</span> '
    +(ev.modRatio_CA>=0.8&&ev.modRatio_CA<=1.2?'<span class="rg">[0.8-1.2]</span>':'<span class="rb">outside [0.8-1.2]</span>')+'</div>';
  h+='<h3>Amplitude</h3>';
  h+='<div class="rr">amp_A='+ev.A_metrics.amp.toFixed(6)+' amp_C='+ev.ampC.toFixed(6)+'</div>';
  h+='<div class="rr">amp_C/amp_A = <span class="rv">'+ev.ampRatio_CA.toFixed(4)+'</span></div>';
  h+='<h3>Span (time)</h3>';
  h+='<div class="rr">span_A='+ev.A_metrics.span+' span_C='+ev.spanC+'</div>';
  h+='<div class="rr">span_C/span_A = <span class="rv">'+ev.spanRatio_CA.toFixed(4)+'</span></div>';
  h+='<h3>Mirror Deviation</h3>';
  h+='<div class="rr">Normalized distance: <span class="rv">'+ev.mirDev.toFixed(4)+'</span> '
    +(ev.mirDev<0.5?'<span class="rg">CLOSE</span>':ev.mirDev<1.0?'<span class="rc">MID</span>':'<span class="rb">FAR</span>')+'</div>';
  h+='<h3>B Characteristics</h3>';
  h+='<div class="rr">mod_B/mod_A = '+ev.B_metrics.ratio_mod.toFixed(4)+'</div>';
  h+='<div class="rr">amp_B/amp_A = '+ev.B_metrics.ratio_amp.toFixed(4)+'</div>';
  h+='<h3>Notes</h3>';
  ev.notes.forEach(n=>{{h+='<div class="rr">'+n+'</div>';}});
  h+='<h3>Verdict</h3>';
  if(ev.score>=60){{
    h+='<div class="rg">This ABC shows meaningful symmetry. AB constraints worth analyzing.</div>';
    h+='<div class="rr rc">A: amp='+ev.A_metrics.amp.toFixed(6)+' span='+ev.A_metrics.span+' mod='+ev.A_metrics.mod.toFixed(5)+' dir='+(ev.A_metrics.dir>0?'UP':'DN')+'</div>';
    h+='<div class="rr rc">B: amp='+ev.B_metrics.amp.toFixed(6)+' span='+ev.B_metrics.span+' mod='+ev.B_metrics.mod.toFixed(5)+' B/A_mod='+ev.B_metrics.ratio_mod.toFixed(3)+'</div>';
  }}else{{
    h+='<div class="rb">Symmetry too weak for this ABC. Try another structure.</div>';
  }}
  h+='<br><button class="b" onclick="abcSave()" style="color:#ff0">Save This ABC</button>';
  h+='<button class="b" onclick="abcReset()" style="margin-left:5px">New ABC</button>';
  h+='<button class="b" onclick="abcExport()" style="margin-left:5px;color:#4f4">Export All</button>';
  rp.innerHTML=h;
}}

function abcSave(){{
  if(!segA||!segB||!segC||!cPred)return;
  const ev=evaluateC();
  abcSaved.push({{
    symbol:CU, tf:V[abcPanel].tf.tf, panel:abcPanel,
    A:{{b0:segA.b0,p0:segA.p0,b1:segA.b1,p1:segA.p1}},
    B:{{b0:segB.b0,p0:segB.p0,b1:segB.b1,p1:segB.p1}},
    C:{{b0:segC.b0,p0:segC.p0,b1:segC.b1,p1:segC.p1}},
    eval:ev
  }});
  $('abInf').textContent='Saved! ('+abcSaved.length+' total)';
}}

function abcExport(){{
  if(!abcSaved.length)return;
  const blob=new Blob([JSON.stringify(abcSaved,null,2)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='abc_annotations_'+CU+'.json';a.click();
}}

// Handle segment selection via double-click
function abcDblClick(pi, mx, my){{
  if(!abcMode)return;
  const seg=findNearestSeg(pi, mx, my);
  if(!seg){{$('abInf').textContent='No segment found near click. Try closer to a line.';return;}}

  if(abcState==='selA'){{
    abcPanel=pi;
    segA=seg;
    abcState='selB';
    $('abSt').textContent='A SET -> SELECT B';$('abSt').style.color='#4f4';
    $('abInf').textContent='A: bar'+seg.b0+'-'+seg.b1+' ['+seg.src+'] | Now double-click B segment';
    DA();
  }}
  else if(abcState==='selB'){{
    if(pi!==abcPanel){{$('abInf').textContent='B must be in same panel as A ('+V[abcPanel].tf.tf+')';return;}}
    segB=seg;
    // Compute C predictions
    cPred=computeCpred();
    abcState='selC';
    $('abSt').textContent='A+B SET -> SELECT C (actual)';$('abSt').style.color='#88f';
    $('abInf').textContent='B: bar'+seg.b0+'-'+seg.b1+' | C\\'predictions shown. Now double-click actual C segment.';
    DA();
  }}
  else if(abcState==='selC'){{
    if(pi!==abcPanel){{$('abInf').textContent='C must be in same panel ('+V[abcPanel].tf.tf+')';return;}}
    segC=seg;
    abcState='done';
    $('abSt').textContent='ABC COMPLETE';$('abSt').style.color='#ff0';
    $('abInf').textContent='C: bar'+seg.b0+'-'+seg.b1+' | Evaluating symmetry...';
    const ev=evaluateC();
    if(ev)showResults(ev);
    DA();
  }}
}}

// Draw ABC overlay on the active panel
function drawABC(pi,g,xS,yS,W,H){{
  if(!abcMode||pi!==abcPanel)return;
  const v=V[pi];

  // Draw A segment
  if(segA){{
    g.strokeStyle='#FF3333';g.lineWidth=3;g.globalAlpha=0.9;
    g.beginPath();g.moveTo(xS(segA.b0),yS(segA.p0));g.lineTo(xS(segA.b1),yS(segA.p1));g.stroke();
    // Label
    g.fillStyle='#FF3333';g.font='bold 13px monospace';g.textAlign='center';
    g.fillText('A',xS((segA.b0+segA.b1)/2),yS((segA.p0+segA.p1)/2)-10);
    // Endpoints
    g.beginPath();g.arc(xS(segA.b0),yS(segA.p0),4,0,Math.PI*2);g.fill();
    g.beginPath();g.arc(xS(segA.b1),yS(segA.p1),4,0,Math.PI*2);g.fill();
    g.globalAlpha=1;g.textAlign='start';
  }}

  // Draw B segment
  if(segB){{
    g.strokeStyle='#33FF33';g.lineWidth=3;g.globalAlpha=0.9;
    g.beginPath();g.moveTo(xS(segB.b0),yS(segB.p0));g.lineTo(xS(segB.b1),yS(segB.p1));g.stroke();
    g.fillStyle='#33FF33';g.font='bold 13px monospace';g.textAlign='center';
    g.fillText('B',xS((segB.b0+segB.b1)/2),yS((segB.p0+segB.p1)/2)-10);
    g.beginPath();g.arc(xS(segB.b0),yS(segB.p0),4,0,Math.PI*2);g.fill();
    g.beginPath();g.arc(xS(segB.b1),yS(segB.p1),4,0,Math.PI*2);g.fill();
    g.globalAlpha=1;g.textAlign='start';
  }}

  // Draw C' predictions
  if(cPred){{
    // Mirror (yellow dashed)
    g.strokeStyle='#FFD700';g.lineWidth=2;g.globalAlpha=0.7;g.setLineDash([6,4]);
    g.beginPath();g.moveTo(xS(cPred.mirror.b0),yS(cPred.mirror.p0));
    g.lineTo(xS(cPred.mirror.b1),yS(cPred.mirror.p1));g.stroke();
    g.fillStyle='#FFD700';g.font='10px monospace';g.textAlign='center';
    g.fillText("C'mir",xS(cPred.mirror.b1),yS(cPred.mirror.p1)-8);

    // Triangle convergent (cyan dashed)
    g.strokeStyle='#00CED1';g.lineWidth=1.5;
    g.beginPath();g.moveTo(xS(cPred.triConv.b0),yS(cPred.triConv.p0));
    g.lineTo(xS(cPred.triConv.b1),yS(cPred.triConv.p1));g.stroke();
    g.fillStyle='#00CED1';g.fillText("C'conv",xS(cPred.triConv.b1),yS(cPred.triConv.p1)-8);

    // Triangle divergent (orange dashed)
    g.strokeStyle='#FF8C00';g.lineWidth=1.5;
    g.beginPath();g.moveTo(xS(cPred.triDiv.b0),yS(cPred.triDiv.p0));
    g.lineTo(xS(cPred.triDiv.b1),yS(cPred.triDiv.p1));g.stroke();
    g.fillStyle='#FF8C00';g.fillText("C'div",xS(cPred.triDiv.b1),yS(cPred.triDiv.p1)-8);
    g.setLineDash([]);

    // Modulus arc (magenta ellipse)
    // Need to draw an ellipse: in screen coords, center at B.end
    // Radius in bar-direction = modA * nBars, in price-direction = modA * pRange
    // But we draw in screen pixels: convert
    const arc=cPred.modArc;
    const cx_scr=xS(arc.cb), cy_scr=yS(arc.cp);
    // Screen radius: how many px is r_bars and r_price?
    const pw=W-MG.l-MG.r, ph=H-MG.t-MG.b;
    const nv=v._ve-v._vs;
    const pmin2=v._yS?0:0; // we need price range from current view
    // px per bar = pw/nv, px per price = ph/(price range in view)
    const rx_scr=(arc.r_bars/nv)*pw;  // px radius in x
    // For y: we need the price range of current view
    // yS(p) = MG.t + ph - ((p-mn)/(mx-mn))*ph, so 1 unit price = ph/(mx-mn) px
    // But we don't have mn,mx here... let's estimate from arc.r_price
    // Actually let's compute: yS(cp) - yS(cp + r_price) = ph * r_price / (mx-mn)
    // We can get (mx-mn) from the view: yS is already captured
    const ry_scr=Math.abs(yS(arc.cp)-yS(arc.cp+arc.r_price));

    g.strokeStyle='#FF00FF';g.lineWidth=1.5;g.globalAlpha=0.5;
    g.setLineDash([4,4]);
    g.beginPath();
    g.ellipse(cx_scr, cy_scr, Math.max(rx_scr,2), Math.max(ry_scr,2), 0, 0, Math.PI*2);
    g.stroke();
    g.setLineDash([]);
    g.fillStyle='#FF00FF';g.font='9px monospace';
    g.fillText('mod='+arc.modA.toFixed(5), cx_scr+rx_scr+3, cy_scr);
    g.globalAlpha=1;g.textAlign='start';

    // Draw connecting line from A_end to B_start if they should be connected
    // (B typically starts where A ends, but may not in user selection)
    g.strokeStyle='#666';g.lineWidth=1;g.setLineDash([3,3]);g.globalAlpha=0.4;
    g.beginPath();g.moveTo(xS(segA.b1),yS(segA.p1));g.lineTo(xS(segB.b0),yS(segB.p0));g.stroke();
    g.setLineDash([]);g.globalAlpha=1;
  }}

  // Draw actual C segment
  if(segC){{
    g.strokeStyle='#4488FF';g.lineWidth=3;g.globalAlpha=0.9;
    g.beginPath();g.moveTo(xS(segC.b0),yS(segC.p0));g.lineTo(xS(segC.b1),yS(segC.p1));g.stroke();
    g.fillStyle='#4488FF';g.font='bold 13px monospace';g.textAlign='center';
    g.fillText('C',xS((segC.b0+segC.b1)/2),yS((segC.p0+segC.p1)/2)-10);
    g.beginPath();g.arc(xS(segC.b0),yS(segC.p0),4,0,Math.PI*2);g.fill();
    g.beginPath();g.arc(xS(segC.b1),yS(segC.p1),4,0,Math.PI*2);g.fill();
    g.globalAlpha=1;g.textAlign='start';
  }}
}}
/* ==================== END ABC SYSTEM ==================== */

function DP(pi){{
const c=cv[pi],g=cx[pi],W=c.width/devicePixelRatio,H=c.height/devicePixelRatio;
const v=V[pi],tf=v.tf,K=tf.K;
const vs=Math.max(0,v.s|0),ve=Math.min(K.length,Math.ceil(v.e)),nv=ve-vs;
if(nv<=0)return;
const pw=W-MG.l-MG.r,ph=H-MG.t-MG.b;
let mn=1e9,mx=-1e9;
for(let i=vs;i<ve;i++){{if(K[i][2]<mn)mn=K[i][2];if(K[i][1]>mx)mx=K[i][1];}}
const pr=mx-mn;mn-=pr*.04;mx+=pr*.04;
const xS=b=>MG.l+((b-vs)/nv)*pw;
const yS=p=>MG.t+ph-((p-mn)/(mx-mn))*ph;
const bF=x=>vs+(x-MG.l)/pw*nv;

g.fillStyle='#0a0a1a';g.fillRect(0,0,W,H);
// Active panel border (annotation mode: orange for target panel)
if(abcMode&&pi===abcPanel&&abcPanel>=0){{g.strokeStyle='#FF8800';g.lineWidth=3;g.strokeRect(1,1,W-2,H-2);}}
else if(pi===aP){{g.strokeStyle='#335588';g.lineWidth=2;g.strokeRect(1,1,W-2,H-2);}}

// Grid
g.strokeStyle='#181830';g.lineWidth=.5;g.fillStyle='#444';g.font='9px monospace';
for(let i=0;i<=6;i++){{const p=mn+(mx-mn)*i/6,y=yS(p);
g.beginPath();g.moveTo(MG.l,y);g.lineTo(W-MG.r,y);g.stroke();
g.fillText(pr<.1?p.toFixed(6):pr<1?p.toFixed(5):pr<100?p.toFixed(3):p.toFixed(1),2,y+3);}}
g.fillStyle='#444';g.textAlign='center';
const ls=Math.max(1,nv/8|0);
for(let i=vs;i<ve;i+=ls)g.fillText(i,xS(i),H-1);
g.textAlign='start';

// K-lines
const bw=Math.max(.5,pw/nv*.6);
for(let i=vs;i<ve;i++){{const k=K[i],x=xS(i),up=k[3]>=k[0];
g.strokeStyle=up?'#2a6a2a':'#6a2a2a';g.lineWidth=.6;
g.beginPath();g.moveTo(x,yS(k[2]));g.lineTo(x,yS(k[1]));g.stroke();
if(bw>1.5){{g.fillStyle=up?'#1a4a1a':'#4a1a1a';
const yt=yS(Math.max(k[0],k[3])),yb=yS(Math.min(k[0],k[3]));
g.fillRect(x-bw/2,yt,bw,Math.max(1,yb-yt));}}}}

// Fusion
if(ly.fu&&tf.F.length){{const n=Math.min(fN,tf.F.length);
for(let i=n-1;i>=0;i--){{const f=tf.F[i];if(f[4]<0||f[2]<vs||f[0]>ve)continue;
const t=i/Math.max(n-1,1);g.strokeStyle=fcol(i,n);g.lineWidth=Math.max(.3,2-t*1.5);
g.globalAlpha=Math.max(.15,.6-t*.4);g.setLineDash([8,3]);
g.beginPath();g.moveTo(xS(f[0]),yS(f[1]));g.lineTo(xS(f[2]),yS(f[3]));g.stroke();}}
g.setLineDash([]);g.globalAlpha=1;}}

// Snapshots
if(ly.sn){{for(let si=0;si<tf.S.length;si++){{const s=tf.S[si],tp=s[0],pts=s[2];
if(pts.length<2)continue;const col=scol(si,tf.S);
if(tp==='base'){{g.strokeStyle=col;g.lineWidth=.3;g.globalAlpha=.2;}}
else if(tp==='amp'){{let a=0;for(let j=0;j<si;j++)if(tf.S[j][0]==='amp')a++;
g.strokeStyle=col;g.lineWidth=Math.min(.5+a*.25,3);g.globalAlpha=Math.min(.3+a*.07,.85);}}
else{{let l=0;for(let j=0;j<si;j++)if(tf.S[j][0]==='lat')l++;
g.strokeStyle=col;g.lineWidth=Math.min(1.2+l*.25,3);g.globalAlpha=.8;g.setLineDash([6+l*2,3+l]);}}
g.beginPath();let st=0;
for(const p of pts){{if(p[0]<vs-nv*.1||p[0]>ve+nv*.1)continue;
if(!st){{g.moveTo(xS(p[0]),yS(p[1]));st=1;}}else g.lineTo(xS(p[0]),yS(p[1]));}}
if(st)g.stroke();
if(pts.length<40&&tp!=='base'){{const ms=Math.min(2+(40-pts.length)*.08,4);g.fillStyle=col;
for(const p of pts){{if(p[0]<vs||p[0]>ve)continue;
g.beginPath();g.arc(xS(p[0]),yS(p[1]),ms,0,Math.PI*2);g.fill();}}}}
g.setLineDash([]);g.globalAlpha=1;}}}}

// Important points: M=[bar,price,dir,rank,imp,label]
if(ly.mk){{const pks=tf.M.filter(m=>m[2]===1),vls=tf.M.filter(m=>m[2]===-1);
g.textAlign='center';
for(let i=0;i<Math.min(pN,pks.length);i++){{const m=pks[i];if(m[0]<vs||m[0]>ve)continue;
const x=xS(m[0]),y=yS(m[1]),sz=Math.max(2.5,5-i*.25),al=Math.max(.4,1-i*.04);
g.fillStyle='#FF4444';g.globalAlpha=al;g.beginPath();g.arc(x,y,sz,0,Math.PI*2);g.fill();
g.strokeStyle='#fff';g.lineWidth=i<5?1:.5;g.stroke();
g.fillStyle='#FF4444';g.font=i<10?'bold 10px monospace':'9px monospace';g.fillText(m[5],x,y-10);
if(ly.vl){{g.font='8px monospace';g.fillStyle='#faa';g.fillText(m[4].toFixed(3),x,y-20);}}
g.globalAlpha=1;}}
for(let i=0;i<Math.min(vN,vls.length);i++){{const m=vls[i];if(m[0]<vs||m[0]>ve)continue;
const x=xS(m[0]),y=yS(m[1]),sz=Math.max(2.5,5-i*.25),al=Math.max(.4,1-i*.04);
g.fillStyle='#44FF44';g.globalAlpha=al;g.beginPath();g.arc(x,y,sz,0,Math.PI*2);g.fill();
g.strokeStyle='#fff';g.lineWidth=i<5?1:.5;g.stroke();
g.fillStyle='#44FF44';g.font=i<10?'bold 10px monospace':'9px monospace';g.fillText(m[5],x,y+14);
if(ly.vl){{g.font='8px monospace';g.fillStyle='#afa';g.fillText(m[4].toFixed(3),x,y+24);}}
g.globalAlpha=1;}}
g.textAlign='start';}}

// ABC overlay (drawn last, on top)
drawABC(pi,g,xS,yS,W,H);

$('L'+pi).textContent=tf.tf+' | '+nv+' bars';
$('L'+pi).style.color=tfc[pi];
$('I'+pi).textContent='base:'+tf.nb+' pool:'+tf.np+' fus:'+tf.F.length;
v._xS=xS;v._yS=yS;v._bF=bF;v._vs=vs;v._ve=ve;
v._mn=mn;v._mx=mx;
}}

function DA(){{V.forEach((_,i)=>DP(i));}}
function cl(v){{const sp=v.e-v.s,nb=v.tf.K.length;
if(v.s<0){{v.s=0;v.e=sp;}}if(v.e>nb){{v.e=nb;v.s=nb-sp;}}if(v.s<0)v.s=0;}}
function zm(i,f){{const v=V[i],mid=(v.s+v.e)/2,h=Math.max(10,(v.e-v.s)/2*f);v.s=mid-h;v.e=mid+h;cl(v);DP(i);}}
function pn2(i,d){{const v=V[i];v.s+=d;v.e+=d;cl(v);DP(i);}}

document.addEventListener('keydown',e=>{{
  // ABC mode shortcuts
  if(e.key==='a'&&!abcMode){{abcToggle();e.preventDefault();return;}}
  if(e.key==='Escape'&&abcMode){{
    if(abcState==='done'||abcState==='selC')abcReset();
    else abcToggle();
    e.preventDefault();return;
  }}
  const v=V[aP],sp=v.e-v.s,st=Math.max(1,sp*.15|0);
  if(e.key==='+'||e.key==='='){{zm(aP,.7);e.preventDefault();}}
  else if(e.key==='-'||e.key==='_'){{zm(aP,1.4);e.preventDefault();}}
  else if(e.key==='ArrowLeft'){{pn2(aP,-st);e.preventDefault();}}
  else if(e.key==='ArrowRight'){{pn2(aP,st);e.preventDefault();}}
}});

// Right-click to undo last selection in ABC mode
document.addEventListener('contextmenu',e=>{{
  if(!abcMode)return;
  e.preventDefault();
  if(abcState==='done'){{segC=null;abcState='selC';$('abSt').textContent='A+B SET -> SELECT C';$('abSt').style.color='#88f';$('rpanel').style.display='none';DA();}}
  else if(abcState==='selC'){{segB=null;cPred=null;abcState='selB';$('abSt').textContent='A SET -> SELECT B';$('abSt').style.color='#4f4';DA();}}
  else if(abcState==='selB'){{segA=null;abcPanel=-1;abcState='selA';$('abSt').textContent='SELECT A';$('abSt').style.color='#f44';DA();}}
}});

[0,1,2,3].forEach(i=>{{
cv[i].addEventListener('mousedown',e=>{{
  aP=i;
  // In ABC mode, single click just sets active panel, no panning
  if(abcMode){{DA();return;}}
  DA();
  const r=cv[i].getBoundingClientRect(),mx=e.clientX-r.left;
  const v=V[i],sp=v.e-v.s,st=Math.max(1,sp*.25|0);
  if(mx<r.width/2)pn2(i,-st);else pn2(i,st);
}});
cv[i].addEventListener('dblclick',e=>{{
  e.preventDefault();
  if(!abcMode)return;
  const r=cv[i].getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  abcDblClick(i, mx, my);
}});
cv[i].addEventListener('wheel',e=>{{e.preventDefault();aP=i;zm(i,e.deltaY>0?1.2:.83);}},{{passive:false}});
cv[i].addEventListener('mousemove',e=>{{const v=V[i];if(!v._bF)return;
const r=cv[i].getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
const bar=Math.round(v._bF(mx));let info='';const K=v.tf.K;
if(bar>=0&&bar<K.length){{const k=K[bar];
info='bar '+bar+' O:'+k[0].toFixed(5)+' H:'+k[1].toFixed(5)+' L:'+k[2].toFixed(5)+' C:'+k[3].toFixed(5);}}
// In ABC mode, show nearest segment info
if(abcMode&&(abcState==='selA'||abcState==='selB'||abcState==='selC')){{
  const ns=findNearestSeg(i,mx,my);
  if(ns)info+=' | SNAP: bar'+ns.b0+'-'+ns.b1+' ['+ns.src+']';
}}
if(ly.mk&&v._xS){{const pks=v.tf.M.filter(m=>m[2]===1).slice(0,pN);
const vls=v.tf.M.filter(m=>m[2]===-1).slice(0,vN);
let bd=15,bm=null;for(const m of pks.concat(vls)){{if(m[0]<v._vs||m[0]>v._ve)continue;
const d=Math.hypot(mx-v._xS(m[0]),my-v._yS(m[1]));if(d<bd){{bd=d;bm=m;}}}}
if(bm)info+=' | '+bm[5]+' imp='+bm[4].toFixed(4)+' p='+bm[1].toFixed(5);}}
if(ly.fu&&v._xS){{const n=Math.min(fN,v.tf.F.length);let bd=12,bf=null;
for(let fi=0;fi<n;fi++){{const f=v.tf.F[fi];if(f[2]<v._vs||f[0]>v._ve)continue;
const x1=v._xS(f[0]),y1=v._yS(f[1]),x2=v._xS(f[2]),y2=v._yS(f[3]);
const dx=x2-x1,dy=y2-y1,l2=dx*dx+dy*dy;if(!l2)continue;
let t=((mx-x1)*dx+(my-y1)*dy)/l2;t=Math.max(0,Math.min(1,t));
const d=Math.hypot(mx-x1-t*dx,my-y1-t*dy);if(d<bd){{bd=d;bf=f;}}}}
if(bf)info+=' | FUS['+bf[7]+'] bar'+bf[0]+'-'+bf[2]+' span='+bf[5]+' amp='+bf[6]+' imp='+bf[4];}}
const hid=['H0','H1r','H2r','H3r'];
$(hid[i]).textContent=info;}});
}});

// Init
$('ss').innerHTML=SY.map(s=>'<option value="'+s+'"'+(s===CU?' selected':'')+'>'+s+'</option>').join('');
rsz();
</script></body></html>'''

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Saved: {output_path} ({len(html)//1024}KB)")


def main():
    symbol = 'EURUSD'
    bars_per_tf = 200

    if len(sys.argv) >= 2:
        symbol = sys.argv[1].upper()
    if len(sys.argv) >= 3:
        bars_per_tf = int(sys.argv[2])

    all_symbols = discover_symbols()
    if symbol not in all_symbols:
        print(f"ERROR: {symbol} not found. Available: {', '.join(all_symbols[:10])}...")
        sys.exit(1)

    print(f"v3.2 | {symbol} | {bars_per_tf} bars/tf")
    print('='*50)

    tf_data = []
    for tf in TFS:
        df = load_norm(symbol, tf, limit=bars_per_tf)
        if df is None or len(df) < 10:
            print(f"  [{tf}] SKIPPED")
            continue
        data = process_one_tf(df, tf)
        tf_data.append(data)

    if not tf_data:
        print("ERROR: No data!")
        sys.exit(1)

    output = f'/home/ubuntu/stage2_abc/merge_v3_2_{symbol}.html'
    generate_html(symbol, tf_data, all_symbols, output)
    print(f"Done! → {output}")


if __name__ == '__main__':
    main()
