#!/usr/bin/env python3
"""
v3.2 标注服务 — HTTP Server模式

启动:  python3 serve_v3_2.py [port]
访问:  http://localhost:8766/?symbol=EURUSD&bars=200

功能:
  - 四窗口 M5/M15/M30/H1 可视化 (normalized data)
  - 9个segment slot, 双击snap线段, M键手工输入(HxLy)
  - 连续性约束: 线段必须首尾相连
  - label分类: 用户提交时输入pattern名称
  - /annotate POST: 存储+特征提取+同label历史约束统计
  - 结果显示在页面底部log区域
"""

import json, sys, os, time, math
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import pandas as pd

sys.path.insert(0, '/home/ubuntu/stage2_abc')
from merge_engine_v3 import (
    calculate_base_zg, full_merge_engine,
    compute_pivot_importance, build_segment_pool, pool_fusion,
)

NORM_DIR = '/home/ubuntu/database2'
TFS = ['M5', 'M15', 'M30', 'H1']
ANNOT_FILE = '/home/ubuntu/stage2_abc/annotations.json'
PORT = 8766

# ====================== DATA ======================

def discover_symbols():
    h1_dir = os.path.join(NORM_DIR, 'H1')
    symbols = set()
    for f in os.listdir(h1_dir):
        if f.endswith('_H1_norm.csv'):
            symbols.add(f.replace('_H1_norm.csv', ''))
    return sorted(symbols)

ALL_SYMBOLS = discover_symbols()

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
          f"{len(pool)}->{len(full_pool)} pool | {elapsed:.1f}s")

    klines = []
    for i in range(len(df)):
        row = df.iloc[i]
        klines.append([
            round(float(row['open']), 6),
            round(float(row['high']), 6),
            round(float(row['low']), 6),
            round(float(row['close']), 6),
        ])

    snaps = []
    for snap_type, label, pvts in results['all_snapshots']:
        pts = [[int(p[0]), round(p[1], 6)] for p in pvts]
        snaps.append([snap_type, label, pts])

    sorted_fus = sorted(fusion_segs, key=lambda s: -s['importance'])[:100]
    fus = []
    for s in sorted_fus:
        fus.append([
            s['bar_start'], round(s['price_start'], 6),
            s['bar_end'], round(s['price_end'], 6),
            round(s['importance'], 4), s['span'],
            round(s['amplitude'], 6), s['source_label'],
        ])

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

# Cache computed data to avoid recomputing on every page load
_cache = {}

def get_tf_data(symbol, bars):
    key = f"{symbol}_{bars}"
    if key in _cache:
        return _cache[key]
    print(f"Computing {symbol} @ {bars} bars/tf...")
    tf_data = []
    for tf in TFS:
        df = load_norm(symbol, tf, limit=bars)
        if df is None or len(df) < 10:
            print(f"  [{tf}] SKIPPED")
            continue
        data = process_one_tf(df, tf)
        tf_data.append(data)
    _cache[key] = tf_data
    return tf_data


# ====================== ANNOTATIONS ======================

def load_annotations():
    if os.path.exists(ANNOT_FILE):
        with open(ANNOT_FILE) as f:
            return json.load(f)
    return []

def save_annotations(annots):
    with open(ANNOT_FILE, 'w') as f:
        json.dump(annots, f, indent=1, ensure_ascii=False)

def extract_features(segments):
    """Extract per-segment and inter-segment features."""
    features = []
    for i, seg in enumerate(segments):
        b1, p1, b2, p2 = seg['b1'], seg['p1'], seg['b2'], seg['p2']
        amp = abs(p2 - p1)
        span = abs(b2 - b1)
        direction = 1 if p2 > p1 else -1
        slope = (p2 - p1) / max(span, 1)
        features.append({
            'idx': i + 1,
            'b1': b1, 'p1': round(p1, 6), 'b2': b2, 'p2': round(p2, 6),
            'amp': round(amp, 6),
            'span': span,
            'dir': direction,
            'dir_label': 'UP' if direction > 0 else 'DN',
            'slope': round(slope, 8),
            'source': seg.get('source', ''),
        })
    return features

def compute_ratios(features):
    """Compute ratios between consecutive and key segment pairs."""
    ratios = []
    # Consecutive pairs
    for i in range(len(features) - 1):
        f1, f2 = features[i], features[i+1]
        amp_r = f2['amp'] / max(f1['amp'], 1e-9)
        time_r = f2['span'] / max(f1['span'], 1)
        # Retrace: how much does seg2 retrace seg1?
        retrace = f2['amp'] / max(f1['amp'], 1e-9) if f1['dir'] != f2['dir'] else None
        ratios.append({
            'pair': f'S{i+1}/S{i+2}',
            'amp_ratio': round(amp_r, 4),
            'time_ratio': round(time_r, 4),
            'retrace': round(retrace, 4) if retrace is not None else None,
        })
    # If >= 3 segments, also compute S1 vs S3 (A vs C style)
    if len(features) >= 3:
        f1, f3 = features[0], features[2]
        amp_r = f3['amp'] / max(f1['amp'], 1e-9)
        time_r = f3['span'] / max(f1['span'], 1)
        # Modulus comparison (normalized)
        total_bars = max(f['b2'] for f in features) - min(f['b1'] for f in features)
        total_amp = max(f['p1'] for f in features + [{'p1': f['p2']} for f in features]) - \
                    min(f['p1'] for f in features + [{'p1': f['p2']} for f in features])
        if total_bars > 0 and total_amp > 0:
            mod1 = math.sqrt((f1['span']/total_bars)**2 + (f1['amp']/total_amp)**2)
            mod3 = math.sqrt((f3['span']/total_bars)**2 + (f3['amp']/total_amp)**2)
            mod_r = mod3 / max(mod1, 1e-9)
        else:
            mod_r = None
        ratios.append({
            'pair': 'S1/S3 (A/C)',
            'amp_ratio': round(amp_r, 4),
            'time_ratio': round(time_r, 4),
            'retrace': None,
            'mod_ratio': round(mod_r, 4) if mod_r else None,
        })
    return ratios

def compute_constraint_stats(label, all_annots):
    """Compute statistics from all historical annotations with same label."""
    same = [a for a in all_annots if a['label'] == label]
    if len(same) < 2:
        return None

    # Collect feature vectors
    all_ratios = {}
    for a in same:
        for r in a.get('ratios', []):
            pair = r['pair']
            if pair not in all_ratios:
                all_ratios[pair] = {'amp_ratio': [], 'time_ratio': [], 'retrace': [], 'mod_ratio': []}
            all_ratios[pair]['amp_ratio'].append(r['amp_ratio'])
            all_ratios[pair]['time_ratio'].append(r['time_ratio'])
            if r.get('retrace') is not None:
                all_ratios[pair]['retrace'].append(r['retrace'])
            if r.get('mod_ratio') is not None:
                all_ratios[pair]['mod_ratio'].append(r['mod_ratio'])

    # Per-segment direction consistency
    dir_seqs = [a.get('dir_sequence', '') for a in same]

    stats = {}
    for pair, vals in all_ratios.items():
        s = {}
        for k, arr in vals.items():
            if len(arr) >= 2:
                import numpy as np
                arr = np.array(arr)
                s[k] = {
                    'mean': round(float(arr.mean()), 4),
                    'std': round(float(arr.std()), 4),
                    'min': round(float(arr.min()), 4),
                    'max': round(float(arr.max()), 4),
                    'n': len(arr),
                }
        if s:
            stats[pair] = s

    return {
        'n_historical': len(same),
        'dir_sequences': dir_seqs,
        'stats': stats,
    }


# ====================== HTML GENERATION ======================

def generate_page(symbol, bars, tf_data):
    data_json = json.dumps(tf_data, separators=(',', ':'))
    sym_json = json.dumps(ALL_SYMBOLS)

    # Using {{ and }} for literal braces in f-string, {x} for python vars
    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>v3.2 | {symbol}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0a0a1a;color:#ddd;font-family:'Consolas',monospace;overflow-x:hidden}}
.hdr{{display:flex;align-items:center;gap:10px;padding:4px 10px;background:#0d0d20;border-bottom:1px solid #222;height:30px}}
.hdr h1{{font-size:13px;color:#7eb8da;white-space:nowrap}}
.hdr select{{background:#1a1a30;color:#8ac;border:1px solid #335;padding:1px 5px;font-family:inherit;font-size:11px;border-radius:3px}}
.ctl{{display:flex;gap:3px 8px;padding:2px 10px;background:#0c0c1e;border-bottom:1px solid #1a1a2a;align-items:center;height:26px;flex-wrap:wrap}}
.b{{background:#1a2a4a;color:#8ac;border:1px solid #335;padding:1px 5px;border-radius:3px;cursor:pointer;font-size:10px;font-family:inherit}}
.b:hover{{background:#2a3a5a}}.b.a{{background:#3a2a5a;color:#f8f;border-color:#a6a}}
.sb{{display:inline-flex;align-items:center;gap:2px;font-size:10px;color:#999}}
.sb input[type=range]{{width:60px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;width:100vw;height:50vh}}
.pn{{position:relative;border:1px solid #1a1a30;overflow:hidden}}
.pn canvas{{width:100%;height:100%;display:block;cursor:crosshair}}
.pl{{position:absolute;top:2px;left:6px;font-size:12px;font-weight:bold;pointer-events:none;z-index:2}}
.pi{{position:absolute;bottom:1px;left:6px;font-size:9px;color:#555;pointer-events:none;z-index:2}}
.hi{{position:absolute;top:2px;right:6px;font-size:10px;color:#aaa;pointer-events:none;z-index:2;text-align:right;max-width:70%}}
#apanel{{background:#0d0d20;border:1px solid #335;border-radius:4px;padding:8px 12px;margin:6px 10px}}
#apanel .sb-row{{display:flex;gap:3px;flex-wrap:wrap;margin:4px 0}}
.sbox{{min-width:120px;height:48px;background:#111;border:2px solid #333;border-radius:3px;padding:3px 5px;cursor:pointer;font-size:10px;font-family:monospace;color:#777;display:flex;flex-direction:column;justify-content:center;text-align:center}}
.sbox.active{{border-color:#f0c040}}
.sbox.filled{{border-color:#4a4}}
.arow{{display:flex;align-items:center;gap:8px;margin-top:6px}}
#alog{{background:#080818;border:1px solid #222;border-radius:4px;padding:6px 10px;margin:4px 10px;max-height:400px;overflow-y:auto;font-size:11px;font-family:monospace;color:#999}}
</style></head><body>
<div class="hdr">
<h1>v3.2 | {symbol}</h1>
<select id="ss" onchange="location.href='/?symbol='+this.value+'&bars={bars}'"></select>
<span style="font-size:10px;color:#666">{symbol} norm | Wheel:zoom +/-:zoom Arrows:scroll | DblClick=select segment</span>
<span style="font-size:10px;color:#444;margin-left:auto" id="hInfo"></span>
</div>
<div class="ctl">
<button class="b a" id="bS" onclick="T('sn')">Snaps</button>
<button class="b a" id="bF" onclick="T('fu')">Fusion</button>
<button class="b a" id="bM" onclick="T('mk')">ImpPts</button>
<button class="b a" id="bV" onclick="T('vl')">Values</button>
<span class="sb" style="color:#f8f">Pk:<input type="range" id="rP" min="0" max="50" value="10" oninput="pN=+this.value;$('nP').textContent=pN;DA()"><span id="nP">10</span></span>
<span class="sb" style="color:#4f4">Vl:<input type="range" id="rV" min="0" max="50" value="10" oninput="vN=+this.value;$('nV').textContent=vN;DA()"><span id="nV">10</span></span>
<span class="sb" style="color:#c8f">Fu:<input type="range" id="rF" min="0" max="100" value="30" oninput="fN=+this.value;$('nF').textContent=fN;DA()"><span id="nF">30</span></span>
</div>
<div class="grid">
<div class="pn" id="P0"><canvas id="C0"></canvas><div class="pl" id="L0" style="color:#5a8ab0"></div><div class="pi" id="I0"></div><div class="hi" id="H0"></div></div>
<div class="pn" id="P1"><canvas id="C1"></canvas><div class="pl" id="L1" style="color:#8ab05a"></div><div class="pi" id="I1"></div><div class="hi" id="H1r"></div></div>
<div class="pn" id="P2"><canvas id="C2"></canvas><div class="pl" id="L2" style="color:#b08a5a"></div><div class="pi" id="I2"></div><div class="hi" id="H2r"></div></div>
<div class="pn" id="P3"><canvas id="C3"></canvas><div class="pl" id="L3" style="color:#b05a8a"></div><div class="pi" id="I3"></div><div class="hi" id="H3r"></div></div>
</div>

<!-- Annotation Panel -->
<div id="apanel">
<div style="display:flex;align-items:center;gap:10px;">
  <span style="color:#f0c040;font-weight:bold;font-size:13px;">Annotate</span>
  <span style="color:#888;font-size:11px;">Click panel to set TF | DblClick=snap segment | M=manual input(HxLy) | Esc=clear slot</span>
  <span style="color:#555;font-size:11px;" id="aMode">Mode: CLICK</span>
  <span id="aStatus" style="color:#888;font-size:11px;margin-left:auto;"></span>
</div>
<div class="sb-row" id="segBoxes"></div>
<div class="arow">
  <span style="color:#aaa;font-size:11px;">Label:</span>
  <input type="text" id="aLabel" placeholder="e.g. A(B)abcC, 12345, abcde, trend_impulse ..."
         style="width:320px;background:#111;color:#eee;border:1px solid #444;padding:3px 6px;font-size:12px;font-family:monospace;">
  <button onclick="submitAnnot()" style="background:#2a5a2a;color:#8f8;border:1px solid #4a4;padding:4px 14px;border-radius:3px;cursor:pointer;font-size:12px;font-weight:bold;">Submit</button>
  <button onclick="clearAnnot()" style="background:#3a1a1a;color:#f88;border:1px solid #644;padding:4px 10px;border-radius:3px;cursor:pointer;font-size:11px;">Clear</button>
  <button onclick="undoSeg()" style="background:#2a2a1a;color:#ff8;border:1px solid #554;padding:4px 10px;border-radius:3px;cursor:pointer;font-size:11px;">Undo</button>
</div>
</div>
<div id="alog"></div>

<script>
const D={data_json};
const SY={sym_json};
const CU='{symbol}';
const BARS={bars};
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

/* ==================== ANNOTATION SYSTEM ==================== */
const ANN_MAX=9;
let annSlots=[];  // [{{b1,p1,b2,p2,source}},...]
let annActive=0;
let annLocked=false;
let annPanel=0;  // which TF panel is the annotation target

// Build boxes
(function(){{
  const c=$('segBoxes');
  for(let i=0;i<ANN_MAX;i++){{
    const box=document.createElement('div');
    box.id='sbox'+i;
    box.className='sbox';
    box.innerHTML='<div style="color:#556;">S'+(i+1)+'</div>';
    box.onclick=function(){{if(!annLocked)setSlot(i);}};
    c.appendChild(box);
  }}
  setSlot(0);
}})();

function setSlot(idx){{
  annActive=idx;
  for(let i=0;i<ANN_MAX;i++){{
    const b=$('sbox'+i);
    b.className='sbox'+(i===idx?' active':'')+(annSlots[i]?' filled':'');
  }}
}}

function renderSlot(idx){{
  const box=$('sbox'+idx);
  const seg=annSlots[idx];
  if(!seg){{
    box.innerHTML='<div style="color:#556;">S'+(idx+1)+'</div>';
  }}else{{
    const d=seg.p2>seg.p1?'\\u2191':'\\u2193';
    const dc=seg.p2>seg.p1?'#4f4':'#f44';
    box.innerHTML=
      '<div style="color:#aaa;">S'+(idx+1)+' <span style="color:'+dc+';">'+d+'</span> <span style="color:#668;font-size:9px;">'+
      (seg.source||'')+'</span></div>'+
      '<div style="color:#8cf;font-size:9px;">b'+seg.b1+' '+seg.p1.toFixed(5)+'</div>'+
      '<div style="color:#fc8;font-size:9px;">b'+seg.b2+' '+seg.p2.toFixed(5)+'</div>';
  }}
  box.className='sbox'+(idx===annActive?' active':'')+(seg?' filled':'');
}}

function setStatus(msg,col){{$('aStatus').textContent=msg;$('aStatus').style.color=col||'#888';}}

// Collect all snappable segments for a panel
function getSegs(pi){{
  const tf=V[pi].tf, segs=[];
  // Fusion
  const n=Math.min(fN,tf.F.length);
  for(let i=0;i<n;i++){{
    const f=tf.F[i];
    segs.push({{b1:f[0],p1:f[1],b2:f[2],p2:f[3],source:'FUS:'+f[7]}});
  }}
  // Snapshot zigzag edges
  for(let si=0;si<tf.S.length;si++){{
    const s=tf.S[si],tp=s[0],pts=s[2];
    if(tp==='base')continue;
    for(let j=0;j<pts.length-1;j++){{
      segs.push({{b1:pts[j][0],p1:pts[j][1],b2:pts[j+1][0],p2:pts[j+1][1],source:tp+':'+s[1]}});
    }}
  }}
  return segs;
}}

// Distance from mouse to segment in pixel space
function segDist(pi,mx,my,seg){{
  const v=V[pi];if(!v._xS)return 999;
  const x1=v._xS(seg.b1),y1=v._yS(seg.p1),x2=v._xS(seg.b2),y2=v._yS(seg.p2);
  const dx=x2-x1,dy=y2-y1,l2=dx*dx+dy*dy;
  if(!l2)return Math.hypot(mx-x1,my-y1);
  let t=((mx-x1)*dx+(my-y1)*dy)/l2;t=Math.max(0,Math.min(1,t));
  return Math.hypot(mx-x1-t*dx,my-y1-t*dy);
}}

// Continuity check
function checkCont(idx,seg){{
  if(idx===0)return true;
  const prev=annSlots[idx-1];
  if(!prev)return true;
  return prev.b2===seg.b1;
}}

// Resolve HxLy label to bar,price from marks
function resolveLabel(label,pi){{
  const tf=V[pi].tf;
  const m=label.match(/^([HL])(\\d+)$/i);
  if(!m)return null;
  const dir=m[1].toUpperCase()==='H'?1:-1;
  const rank=parseInt(m[2]);
  const found=tf.M.find(mk=>mk[2]===dir&&mk[3]===rank);
  if(found)return {{bar:found[0],price:found[1]}};
  return null;
}}

// DblClick handler
function onDblClick(pi,mx,my){{
  if(annLocked)return;
  annPanel=pi;
  const allSegs=getSegs(pi);
  let best=null,bestD=25;
  for(const seg of allSegs){{
    const d=segDist(pi,mx,my,seg);
    if(d<bestD){{bestD=d;best=seg;}}
  }}
  if(!best){{setStatus('No segment near click','#f88');return;}}
  if(!checkCont(annActive,best)){{
    const prev=annSlots[annActive-1];
    setStatus('Not continuous! S'+annActive+' ends b'+prev.b2+' but clicked b'+best.b1,'#f88');
    return;
  }}
  annSlots[annActive]={{b1:best.b1,p1:best.p1,b2:best.b2,p2:best.p2,source:best.source}};
  renderSlot(annActive);
  setStatus('S'+(annActive+1)+': b'+best.b1+'->b'+best.b2+' ('+best.source+')','#8f8');
  DA(); // redraw with highlights
  if(annActive<ANN_MAX-1)setSlot(annActive+1);
}}

// Manual input (M key)
function startManual(){{
  const box=$('sbox'+annActive);
  if(box.querySelector('input'))return;
  const inp=document.createElement('input');
  inp.type='text';inp.placeholder='H1 L2 or b1 p1 b2 p2';
  inp.style.cssText='width:110px;background:#1a1a2a;color:#ff8;border:1px solid #554;padding:2px;font-size:10px;font-family:monospace;';
  inp.onkeydown=function(ev){{
    if(ev.key==='Enter'){{
      const v=inp.value.trim();
      // Try HxLy HxLy format first
      const labels=v.split(/\\s+/);
      if(labels.length===2){{
        const p1=resolveLabel(labels[0],annPanel);
        const p2=resolveLabel(labels[1],annPanel);
        if(p1&&p2){{
          const seg={{b1:p1.bar,p1:p1.price,b2:p2.bar,p2:p2.price,source:'MANUAL:'+v}};
          if(!checkCont(annActive,seg)){{setStatus('Not continuous!','#f88');return;}}
          annSlots[annActive]=seg;renderSlot(annActive);DA();
          setStatus('Manual S'+(annActive+1)+': '+v,'#8f8');
          if(annActive<ANN_MAX-1)setSlot(annActive+1);
          return;
        }}
      }}
      // Try 4 numbers: b1 p1 b2 p2
      const parts=v.split(/\\s+/);
      if(parts.length===4){{
        const b1=parseInt(parts[0]),p1=parseFloat(parts[1]),b2=parseInt(parts[2]),p2=parseFloat(parts[3]);
        if(!isNaN(b1)&&!isNaN(p1)&&!isNaN(b2)&&!isNaN(p2)){{
          const seg={{b1,p1,b2,p2,source:'MANUAL'}};
          if(!checkCont(annActive,seg)){{setStatus('Not continuous!','#f88');return;}}
          annSlots[annActive]=seg;renderSlot(annActive);DA();
          setStatus('Manual S'+(annActive+1)+': b'+b1+'->b'+b2,'#8f8');
          if(annActive<ANN_MAX-1)setSlot(annActive+1);
          return;
        }}
      }}
      setStatus('Parse error. Use: H1 L2 or b1 p1 b2 p2','#f88');
    }}
    if(ev.key==='Escape')renderSlot(annActive);
  }};
  box.innerHTML='';box.appendChild(inp);inp.focus();
  $('aMode').textContent='Mode: MANUAL';
}}

function clearAnnot(){{
  annSlots=[];annLocked=false;
  for(let i=0;i<ANN_MAX;i++)renderSlot(i);
  setSlot(0);setStatus('Cleared','#888');DA();
}}

function undoSeg(){{
  if(annLocked)return;
  let last=-1;
  for(let i=ANN_MAX-1;i>=0;i--)if(annSlots[i]){{last=i;break;}}
  if(last>=0){{delete annSlots[last];annSlots.length=last;renderSlot(last);setSlot(last);
  setStatus('Undone S'+(last+1),'#ff8');DA();}}
}}

async function submitAnnot(){{
  const filled=annSlots.filter(s=>s);
  if(filled.length<2){{setStatus('Need at least 2 segments','#f88');return;}}
  const label=$('aLabel').value.trim();
  if(!label){{setStatus('Enter a label first','#f88');$('aLabel').focus();return;}}
  // Validate continuity
  for(let i=1;i<filled.length;i++){{
    if(filled[i].b1!==filled[i-1].b2){{
      setStatus('Gap: S'+i+' end=b'+filled[i-1].b2+' != S'+(i+1)+' start=b'+filled[i].b1,'#f88');return;
    }}
  }}
  annLocked=true;setStatus('Submitting...','#ff8');
  try{{
    const resp=await fetch('/annotate',{{
      method:'POST',headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{
        segments:filled, label:label,
        symbol:CU, tf:V[annPanel].tf.tf, bars:BARS,
      }})
    }});
    const r=await resp.json();
    if(r.error){{setStatus('Error: '+r.error,'#f88');annLocked=false;return;}}
    showResult(r);
    setStatus('Saved! #'+r.record_id+' | total='+r.total_annotations+' | same_label='+r.same_label_count,'#8f8');
  }}catch(e){{setStatus('Network error: '+e.message,'#f88');annLocked=false;}}
}}

function showResult(r){{
  const log=$('alog');
  let h='<div style="border-bottom:1px solid #333;padding:6px 0;margin-bottom:4px;">';
  h+='<div><span style="color:#f0c040;font-weight:bold;">[#'+r.record_id+']</span> ';
  h+='<span style="color:#8cf;">'+r.n_segments+' segs</span> ';
  h+='<span style="color:#aaa;">label='+r.label+' dir='+r.dir_sequence+'</span> ';
  h+='<span style="color:#666;">'+r.symbol+' '+r.tf+'</span></div>';

  // Features table
  h+='<table style="font-size:10px;color:#aaa;border-collapse:collapse;margin:3px 0;">';
  h+='<tr style="color:#668;"><td style="padding:1px 6px;">Seg</td><td>Bar</td><td>Amp</td><td>Span</td><td>Dir</td><td>Slope</td><td>Src</td></tr>';
  for(const f of r.features){{
    const dc=f.dir===1?'#4f4':'#f44';
    h+='<tr><td style="padding:1px 6px;">S'+f.idx+'</td><td>b'+f.b1+'->b'+f.b2+'</td>';
    h+='<td>'+f.amp.toFixed(5)+'</td><td>'+f.time+'</td>';
    h+='<td style="color:'+dc+';">'+f.dir_label+'</td>';
    h+='<td>'+f.slope.toFixed(7)+'</td><td style="color:#556;">'+f.source+'</td></tr>';
  }}
  h+='</table>';

  // Ratios
  if(r.ratios&&r.ratios.length){{
    h+='<div style="color:#778;margin-top:3px;">Ratios:</div>';
    h+='<table style="font-size:10px;color:#aaa;border-collapse:collapse;margin:2px 0;">';
    h+='<tr style="color:#668;"><td style="padding:1px 6px;">Pair</td><td>AmpR</td><td>TimeR</td><td>Retrace</td><td>ModR</td></tr>';
    for(const rt of r.ratios){{
      h+='<tr><td style="padding:1px 6px;">'+rt.pair+'</td>';
      h+='<td>'+rt.amp_ratio.toFixed(4)+'</td>';
      h+='<td>'+rt.time_ratio.toFixed(4)+'</td>';
      h+='<td>'+(rt.retrace!==null&&rt.retrace!==undefined?rt.retrace.toFixed(4):'-')+'</td>';
      h+='<td>'+(rt.mod_ratio!==null&&rt.mod_ratio!==undefined?rt.mod_ratio.toFixed(4):'-')+'</td></tr>';
    }}
    h+='</table>';
  }}

  // Constraint stats
  if(r.constraint_stats){{
    const cs=r.constraint_stats;
    h+='<div style="color:#f80;margin-top:3px;">Historical constraints ('+cs.n_historical+' same-label cases):</div>';
    if(cs.stats){{
      h+='<table style="font-size:10px;color:#ca8;border-collapse:collapse;margin:2px 0;">';
      h+='<tr style="color:#886;"><td style="padding:1px 6px;">Pair</td><td>Metric</td><td>Mean</td><td>Std</td><td>Range</td><td>N</td></tr>';
      for(const[pair,metrics] of Object.entries(cs.stats)){{
        for(const[mk,mv] of Object.entries(metrics)){{
          h+='<tr><td style="padding:1px 6px;">'+pair+'</td><td>'+mk+'</td>';
          h+='<td style="color:#ff0;">'+mv.mean.toFixed(4)+'</td>';
          h+='<td>'+mv.std.toFixed(4)+'</td>';
          h+='<td>'+mv.min.toFixed(4)+'-'+mv.max.toFixed(4)+'</td>';
          h+='<td>'+mv.n+'</td></tr>';
        }}
      }}
      h+='</table>';
    }}
    h+='<div style="color:#668;font-size:9px;">dir patterns: '+JSON.stringify(cs.dir_sequences)+'</div>';
  }}

  h+='<div style="margin-top:3px;"><button class="b" onclick="clearAnnot()">New Annotation</button></div>';
  h+='</div>';
  log.innerHTML=h+log.innerHTML;
}}
/* ==================== END ANNOTATION ==================== */

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
if(pi===aP){{g.strokeStyle='#335588';g.lineWidth=2;g.strokeRect(1,1,W-2,H-2);}}

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

// Important points
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

// Annotation highlights
if(pi===annPanel){{
  g.save();
  for(let i=0;i<annSlots.length;i++){{
    const seg=annSlots[i];if(!seg)continue;
    const hue=(i*40)%360;
    g.strokeStyle='hsl('+hue+',100%,70%)';g.lineWidth=4;g.globalAlpha=0.85;g.setLineDash([]);
    g.beginPath();g.moveTo(xS(seg.b1),yS(seg.p1));g.lineTo(xS(seg.b2),yS(seg.p2));g.stroke();
    g.fillStyle='hsl('+hue+',100%,80%)';
    g.beginPath();g.arc(xS(seg.b1),yS(seg.p1),4,0,Math.PI*2);g.fill();
    g.beginPath();g.arc(xS(seg.b2),yS(seg.p2),4,0,Math.PI*2);g.fill();
    g.font='bold 11px monospace';g.textAlign='center';
    const mx2=(xS(seg.b1)+xS(seg.b2))/2,my2=(yS(seg.p1)+yS(seg.p2))/2;
    g.fillText('S'+(i+1),mx2,my2-8);
  }}
  g.restore();g.setLineDash([]);g.globalAlpha=1;g.textAlign='start';
}}

$('L'+pi).textContent=tf.tf+' | '+nv+' bars';
$('L'+pi).style.color=tfc[pi];
$('I'+pi).textContent='base:'+tf.nb+' pool:'+tf.np+' fus:'+tf.F.length;
v._xS=xS;v._yS=yS;v._bF=bF;v._vs=vs;v._ve=ve;
}}

function DA(){{V.forEach((_,i)=>DP(i));}}
function cl(v){{const sp=v.e-v.s,nb=v.tf.K.length;
if(v.s<0){{v.s=0;v.e=sp;}}if(v.e>nb){{v.e=nb;v.s=nb-sp;}}if(v.s<0)v.s=0;}}
function zm(i,f){{const v=V[i],mid=(v.s+v.e)/2,h=Math.max(10,(v.e-v.s)/2*f);v.s=mid-h;v.e=mid+h;cl(v);DP(i);}}
function pn2(i,d){{const v=V[i];v.s+=d;v.e+=d;cl(v);DP(i);}}

document.addEventListener('keydown',e=>{{
  if(e.target.tagName==='INPUT')return;
  if(e.key==='m'||e.key==='M'){{startManual();e.preventDefault();return;}}
  if(e.key==='Escape'){{
    if(annLocked){{clearAnnot();}}
    else{{
      // Clear current slot
      if(annSlots[annActive]){{delete annSlots[annActive];renderSlot(annActive);DA();}}
    }}
    e.preventDefault();return;
  }}
  const v=V[aP],sp=v.e-v.s,st=Math.max(1,sp*.15|0);
  if(e.key==='+'||e.key==='='){{zm(aP,.7);e.preventDefault();}}
  else if(e.key==='-'||e.key==='_'){{zm(aP,1.4);e.preventDefault();}}
  else if(e.key==='ArrowLeft'){{pn2(aP,-st);e.preventDefault();}}
  else if(e.key==='ArrowRight'){{pn2(aP,st);e.preventDefault();}}
}});

[0,1,2,3].forEach(i=>{{
  cv[i].addEventListener('mousedown',e=>{{aP=i;annPanel=i;DA();}});
  cv[i].addEventListener('dblclick',e=>{{
    e.preventDefault();
    const r=cv[i].getBoundingClientRect();
    onDblClick(i, e.clientX-r.left, e.clientY-r.top);
  }});
  cv[i].addEventListener('wheel',e=>{{e.preventDefault();aP=i;zm(i,e.deltaY>0?1.2:.83);}},{{passive:false}});
  cv[i].addEventListener('mousemove',e=>{{const v=V[i];if(!v._bF)return;
    const r=cv[i].getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    const bar=Math.round(v._bF(mx));let info='';const K=v.tf.K;
    if(bar>=0&&bar<K.length){{const k=K[bar];
    info='bar '+bar+' O:'+k[0].toFixed(5)+' H:'+k[1].toFixed(5)+' L:'+k[2].toFixed(5)+' C:'+k[3].toFixed(5);}}
    // Show nearest snap target
    const ns=getSegs(i);let bd=20,bf=null;
    for(const seg of ns){{const d=segDist(i,mx,my,seg);if(d<bd){{bd=d;bf=seg;}}}}
    if(bf)info+=' | SNAP: b'+bf.b1+'->b'+bf.b2+' ['+bf.source+']';
    if(ly.mk&&v._xS){{const pks=v.tf.M.filter(m=>m[2]===1).slice(0,pN);
    const vls=v.tf.M.filter(m=>m[2]===-1).slice(0,vN);
    let bm=null;bd=15;for(const m of pks.concat(vls)){{if(m[0]<v._vs||m[0]>v._ve)continue;
    const d=Math.hypot(mx-v._xS(m[0]),my-v._yS(m[1]));if(d<bd){{bd=d;bm=m;}}}}
    if(bm)info+=' | '+bm[5]+' imp='+bm[4].toFixed(4);}}
    const hid=['H0','H1r','H2r','H3r'];
    $(hid[i]).textContent=info;
  }});
}});

$('ss').innerHTML=SY.map(s=>'<option value="'+s+'"'+(s===CU?' selected':'')+'>'+s+'</option>').join('');
rsz();
</script></body></html>'''
    return html


# ====================== HTTP SERVER ======================

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == '/' or parsed.path == '':
            symbol = params.get('symbol', ['EURUSD'])[0].upper()
            bars = int(params.get('bars', ['200'])[0])
            if symbol not in ALL_SYMBOLS:
                self.send_error(404, f"Symbol {symbol} not found")
                return

            tf_data = get_tf_data(symbol, bars)
            if not tf_data:
                self.send_error(500, "No data")
                return

            html = generate_page(symbol, bars, tf_data)
            body = html.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            print(f"GET /?symbol={symbol}&bars={bars} -> {len(body)//1024}KB", flush=True)

        elif parsed.path == '/symbols':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(ALL_SYMBOLS).encode())

        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/annotate':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))

            segments = body.get('segments', [])
            label = body.get('label', '')
            symbol = body.get('symbol', 'UNKNOWN')
            tf = body.get('tf', 'UNKNOWN')
            bars = body.get('bars', 200)

            if not segments or not label:
                self.send_json({'error': 'segments and label required'})
                return

            # Validate continuity
            for i in range(1, len(segments)):
                if segments[i]['b1'] != segments[i-1]['b2']:
                    self.send_json({'error': f'Gap between S{i} and S{i+1}'})
                    return

            # Extract features
            features = extract_features(segments)
            ratios = compute_ratios(features)
            dir_seq = ''.join('U' if f['dir'] > 0 else 'D' for f in features)

            # Summary
            summary = {
                'net_dir': 'UP' if features[-1]['p2'] > features[0]['p1'] else 'DN',
                'total_bars': features[-1]['b2'] - features[0]['b1'],
                'total_amp': round(abs(features[-1]['p2'] - features[0]['p1']), 6),
                'n_segments': len(features),
            }

            # Load existing, add new record
            annots = load_annotations()
            record_id = len(annots) + 1
            record = {
                'id': record_id,
                'symbol': symbol,
                'tf': tf,
                'bars': bars,
                'label': label,
                'segments': segments,
                'features': features,
                'ratios': ratios,
                'dir_sequence': dir_seq,
                'summary': summary,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            annots.append(record)
            save_annotations(annots)

            # Compute constraint stats from all same-label records
            constraint_stats = compute_constraint_stats(label, annots)

            same_count = sum(1 for a in annots if a['label'] == label)

            result = {
                'record_id': record_id,
                'label': label,
                'symbol': symbol,
                'tf': tf,
                'n_segments': len(features),
                'dir_sequence': dir_seq,
                'features': features,
                'ratios': ratios,
                'summary': summary,
                'constraint_stats': constraint_stats,
                'same_label_count': same_count,
                'total_annotations': len(annots),
            }

            self.send_json(result)
            print(f"POST /annotate -> #{record_id} label={label} segs={len(segments)} same_label={same_count}")

        else:
            self.send_error(404)

    def send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True
    allow_reuse_port = True


def main():
    port = PORT
    if len(sys.argv) >= 2:
        port = int(sys.argv[1])

    print(f"v3.2 Annotation Server", flush=True)
    print(f"  Symbols: {len(ALL_SYMBOLS)}", flush=True)
    print(f"  Annotations file: {ANNOT_FILE}", flush=True)

    # Pre-warm cache for EURUSD
    print("  Pre-warming EURUSD 200...", flush=True)
    get_tf_data('EURUSD', 200)
    print("  Cache ready.", flush=True)

    print(f"  Starting on http://0.0.0.0:{port}", flush=True)
    print(f"  Open: http://localhost:{port}/?symbol=EURUSD&bars=200", flush=True)
    print(f"  Keys: DblClick=snap, M=manual, Esc=clear, +/-=zoom, Arrows=scroll", flush=True)

    server = ReusableHTTPServer(('0.0.0.0', port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
        server.server_close()


if __name__ == '__main__':
    main()
