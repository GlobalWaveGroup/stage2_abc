#!/usr/bin/env python3
"""
v33 Server — v3静态HTML + HTTP参数化交互 + 标注系统

每次请求 /?end=500&win=200 → 服务端跑v3完整pipeline → 返回完整静态HTML
标注: 双击选线段填入A-I框 → POST /annotate → 分析+存储

启动: python3 visualize_v33_server.py [--bars 5000] [--port 8765]
访问: http://10.10.0.4:8765/?end=500&win=200
"""

import sys
import time
import json
import os
import argparse
from datetime import datetime

sys.path.insert(0, '/home/ubuntu/stage2_abc')

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from merge_engine_v3 import (
    load_kline, calculate_base_zg, full_merge_engine,
    compute_pivot_importance, build_segment_pool,
    pool_fusion, predict_symmetric_image, find_symmetric_structures
)
from visualize_v3 import generate_html as v3_generate_html

ANNOTATIONS_DIR = '/home/ubuntu/stage2_abc/annotations'
ANNOTATIONS_FILE = os.path.join(ANNOTATIONS_DIR, 'all_annotations.json')

app = FastAPI()
G = {}


@app.on_event("startup")
def startup():
    bars = getattr(app, '_bars', 5000)
    print(f"Loading EURUSD H1 data (limit={bars})...", flush=True)
    df = load_kline("/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv", limit=bars)
    G['df'] = df
    G['n'] = len(df)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    print(f"Loaded: {G['n']} bars", flush=True)


@app.post("/annotate")
async def annotate(request: Request):
    """接收标注，分析几何特征，存储，返回分析结果"""
    data = await request.json()
    segments = data.get('segments', [])
    label = data.get('label', '').strip()
    window = data.get('window', {})

    if len(segments) < 2:
        return JSONResponse({'error': '至少需要2条线段'}, status_code=400)
    if not label:
        return JSONResponse({'error': '请输入分类描述'}, status_code=400)

    # 计算每段几何特征
    features = []
    for i, seg in enumerate(segments):
        b1, p1, b2, p2 = seg['b1'], seg['p1'], seg['b2'], seg['p2']
        amp = abs(p2 - p1)
        time_span = abs(b2 - b1)
        direction = 1 if p2 > p1 else -1
        slope = amp / max(time_span, 1)
        features.append({
            'idx': i + 1,
            'b1': b1, 'p1': round(p1, 5),
            'b2': b2, 'p2': round(p2, 5),
            'lbl1': seg.get('lbl1', f'b{b1}'),
            'lbl2': seg.get('lbl2', f'b{b2}'),
            'amp': round(amp, 5),
            'time': time_span,
            'dir': direction,
            'dir_label': 'UP' if direction == 1 else 'DN',
            'slope': round(slope, 7),
            'source': seg.get('source', '?'),
        })

    # 计算相邻段之间的比值
    ratios = []
    for i in range(1, len(features)):
        prev = features[i - 1]
        curr = features[i]
        amp_ratio = curr['amp'] / max(prev['amp'], 1e-8)
        time_ratio = curr['time'] / max(prev['time'], 1)
        # 回撤比: 当前段振幅 / 前段振幅 (方向相反时才有意义)
        retrace = None
        if prev['dir'] != curr['dir']:
            retrace = round(curr['amp'] / max(prev['amp'], 1e-8), 4)
        ratios.append({
            'pair': f'S{i}→S{i+1}',
            'amp_ratio': round(amp_ratio, 4),
            'time_ratio': round(time_ratio, 4),
            'dir_change': prev['dir'] != curr['dir'],
            'retrace': retrace,
        })

    # 总体特征
    total_bars = abs(segments[-1]['b2'] - segments[0]['b1'])
    total_amp = abs(segments[-1]['p2'] - segments[0]['p1'])
    n_segs = len(segments)

    # 方向序列
    dir_seq = ''.join('U' if f['dir'] == 1 else 'D' for f in features)

    # 构建记录
    record = {
        'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'timestamp': datetime.now().isoformat(),
        'label': label,
        'n_segments': n_segs,
        'dir_sequence': dir_seq,
        'window': window,
        'segments': features,
        'ratios': ratios,
        'summary': {
            'total_bars': total_bars,
            'total_amp': round(total_amp, 5),
            'net_dir': 'UP' if segments[-1]['p2'] > segments[0]['p1'] else 'DN',
        },
    }

    # 加载已有标注，追加
    all_annots = []
    if os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE, 'r') as f:
            try:
                all_annots = json.load(f)
            except json.JSONDecodeError:
                all_annots = []
    all_annots.append(record)
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(all_annots, f, indent=2, ensure_ascii=False)

    # 查找同label的历史案例
    same_label = [a for a in all_annots if a['label'] == label and a['id'] != record['id']]

    # 构建分析反馈
    analysis = {
        'status': 'ok',
        'record_id': record['id'],
        'n_segments': n_segs,
        'dir_sequence': dir_seq,
        'features': features,
        'ratios': ratios,
        'summary': record['summary'],
        'same_label_count': len(same_label),
        'total_annotations': len(all_annots),
    }

    # 如果同label有历史案例，计算约束统计
    if same_label:
        # 收集同段数的案例
        same_n = [a for a in same_label if a['n_segments'] == n_segs]
        if same_n:
            analysis['constraint_stats'] = compute_constraint_stats(same_n, record)

    print(f"  ANNOTATE: {label} | {n_segs} segs | {dir_seq} | total={len(all_annots)}", flush=True)
    return JSONResponse(analysis)


def compute_constraint_stats(historical, current):
    """对同类型同段数的历史案例，统计约束范围"""
    stats = {}
    n = len(historical)

    # 收集每段的amp/time/dir
    for seg_idx in range(current['n_segments']):
        key = f'S{seg_idx + 1}'
        amps = []
        times = []
        dirs = []
        for h in historical:
            if seg_idx < len(h['segments']):
                s = h['segments'][seg_idx]
                amps.append(s['amp'])
                times.append(s['time'])
                dirs.append(s['dir'])
        if amps:
            stats[key] = {
                'amp_range': [round(min(amps), 5), round(max(amps), 5)],
                'amp_mean': round(sum(amps) / len(amps), 5),
                'time_range': [min(times), max(times)],
                'time_mean': round(sum(times) / len(times), 1),
                'dir_consensus': sum(dirs) / len(dirs),  # 1.0=全UP, -1.0=全DN
                'n_samples': len(amps),
            }

    # 收集比值
    for ratio_idx in range(current['n_segments'] - 1):
        key = f'R{ratio_idx + 1}_{ratio_idx + 2}'
        amp_ratios = []
        time_ratios = []
        retraces = []
        for h in historical:
            if ratio_idx < len(h['ratios']):
                r = h['ratios'][ratio_idx]
                amp_ratios.append(r['amp_ratio'])
                time_ratios.append(r['time_ratio'])
                if r['retrace'] is not None:
                    retraces.append(r['retrace'])
        if amp_ratios:
            stats[key] = {
                'amp_ratio_range': [round(min(amp_ratios), 4), round(max(amp_ratios), 4)],
                'amp_ratio_mean': round(sum(amp_ratios) / len(amp_ratios), 4),
                'time_ratio_range': [round(min(time_ratios), 4), round(max(time_ratios), 4)],
                'time_ratio_mean': round(sum(time_ratios) / len(time_ratios), 4),
            }
            if retraces:
                stats[key]['retrace_range'] = [round(min(retraces), 4), round(max(retraces), 4)]
                stats[key]['retrace_mean'] = round(sum(retraces) / len(retraces), 4)

    return {'n_historical': n, 'stats': stats}


@app.get("/annotations")
def get_annotations():
    """返回所有标注记录"""
    if os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE, 'r') as f:
            return JSONResponse(json.load(f))
    return JSONResponse([])


@app.delete("/annotate/{record_id}")
def delete_annotation(record_id: str):
    """删除指定标注"""
    if not os.path.exists(ANNOTATIONS_FILE):
        return JSONResponse({'error': 'No annotations'}, status_code=404)
    with open(ANNOTATIONS_FILE, 'r') as f:
        all_annots = json.load(f)
    before = len(all_annots)
    all_annots = [a for a in all_annots if a['id'] != record_id]
    if len(all_annots) == before:
        return JSONResponse({'error': f'ID {record_id} not found'}, status_code=404)
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(all_annots, f, indent=2, ensure_ascii=False)
    print(f"  DELETE: {record_id} | remaining={len(all_annots)}", flush=True)
    return JSONResponse({'status': 'deleted', 'id': record_id, 'remaining': len(all_annots)})


@app.put("/annotate/{record_id}")
async def update_annotation(record_id: str, request: Request):
    """更新标注的label"""
    data = await request.json()
    new_label = data.get('label', '').strip()
    if not new_label:
        return JSONResponse({'error': '请输入新的label'}, status_code=400)
    if not os.path.exists(ANNOTATIONS_FILE):
        return JSONResponse({'error': 'No annotations'}, status_code=404)
    with open(ANNOTATIONS_FILE, 'r') as f:
        all_annots = json.load(f)
    found = False
    old_label = ''
    for a in all_annots:
        if a['id'] == record_id:
            old_label = a['label']
            a['label'] = new_label
            found = True
            break
    if not found:
        return JSONResponse({'error': f'ID {record_id} not found'}, status_code=404)
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(all_annots, f, indent=2, ensure_ascii=False)
    print(f"  UPDATE: {record_id} | '{old_label}' → '{new_label}'", flush=True)
    return JSONResponse({'status': 'updated', 'id': record_id, 'old_label': old_label, 'new_label': new_label})


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    df_full = G['df']
    n = G['n']

    # 从URL参数获取窗口
    end_bar = int(request.query_params.get('end', min(200, n)))
    win = int(request.query_params.get('win', min(200, end_bar)))
    win = max(10, min(win, 1000, n))
    end_bar = max(win, min(end_bar, n))
    start = end_bar - win

    t0 = time.time()

    # 切片dataframe
    df = df_full.iloc[start:end_bar].reset_index(drop=True)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)

    # v3 完整 pipeline
    base = calculate_base_zg(highs, lows, rb=0.5)
    results = full_merge_engine(base)
    pi = compute_pivot_importance(results, total_bars=len(df))
    pool = build_segment_pool(results, pi)
    full_pool, fusion_segs, fusion_log = pool_fusion(pool, pi)

    # Symmetry + Predictions
    try:
        sym_structures = find_symmetric_structures(full_pool, pi, df=df, top_n=200, max_pool_size=500)
    except Exception:
        sym_structures = []

    predictions = predict_symmetric_image(full_pool, pi, current_bar=len(df) - 1, max_pool_size=500)
    short_preds = sorted([p for p in predictions if p['pred_time'] <= 50], key=lambda p: -p['score'])

    elapsed = time.time() - t0

    # generate_html → 临时文件 → 读回
    import tempfile
    tmp = tempfile.mktemp(suffix='.html')
    v3_generate_html(df, results, tmp,
                     pivot_info=pi, fusion_segs=fusion_segs,
                     sym_structures=sym_structures,
                     predictions=short_preds[:200])
    with open(tmp, 'r') as f:
        html = f.read()
    os.unlink(tmp)

    # === 注入CSS ===
    extra_css = '<style>.lhItem{padding:3px 8px;cursor:pointer;color:#ccc;font-size:11px;font-family:monospace;border-bottom:1px solid #222;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}.lhItem:hover{background:#223;}</style>'
    html = html.replace('</head>', extra_css + '</head>', 1)

    # === 注入导航栏 ===
    nav_html = f'''
<div style="background:#1a1a2e;padding:6px 12px;margin-bottom:4px;border:1px solid #333;border-radius:4px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
  <span style="color:#7eb8da;font-weight:bold;font-size:13px;">v33</span>
  <span style="color:#888;font-size:12px;">|</span>
  <span style="color:#aaa;font-size:12px;">Bars: {start}~{end_bar} (win={win}) | Total: {n} | {elapsed*1000:.0f}ms</span>
  <span style="color:#888;font-size:12px;">|</span>
  <a href="/?end={max(win, end_bar-50)}&win={win}" style="color:#4af;font-size:12px;text-decoration:none;">&laquo;-50</a>
  <a href="/?end={max(win, end_bar-10)}&win={win}" style="color:#4af;font-size:12px;text-decoration:none;">&lsaquo;-10</a>
  <a href="/?end={max(win, end_bar-1)}&win={win}" style="color:#4af;font-size:12px;text-decoration:none;">-1</a>
  <a href="/?end={min(n, end_bar+1)}&win={win}" style="color:#4af;font-size:12px;text-decoration:none;">+1</a>
  <a href="/?end={min(n, end_bar+10)}&win={win}" style="color:#4af;font-size:12px;text-decoration:none;">+10&rsaquo;</a>
  <a href="/?end={min(n, end_bar+50)}&win={win}" style="color:#4af;font-size:12px;text-decoration:none;">+50&raquo;</a>
  <span style="color:#888;font-size:12px;">|</span>
  <span style="color:#aaa;font-size:11px;">Win:</span>
  <a href="/?end={end_bar}&win=25" style="color:{('#ff0' if win==25 else '#4af')};font-size:12px;text-decoration:none;">25</a>
  <a href="/?end={end_bar}&win=50" style="color:{('#ff0' if win==50 else '#4af')};font-size:12px;text-decoration:none;">50</a>
  <a href="/?end={end_bar}&win=100" style="color:{('#ff0' if win==100 else '#4af')};font-size:12px;text-decoration:none;">100</a>
  <a href="/?end={end_bar}&win=200" style="color:{('#ff0' if win==200 else '#4af')};font-size:12px;text-decoration:none;">200</a>
  <a href="/?end={end_bar}&win=500" style="color:{('#ff0' if win==500 else '#4af')};font-size:12px;text-decoration:none;">500</a>
  <span style="color:#888;font-size:12px;">|</span>
  <span style="color:#aaa;font-size:11px;">Goto:</span>
  <input type="number" id="gotoEnd" value="{end_bar}" min="{win}" max="{n}"
         style="width:60px;background:#111;color:#ccc;border:1px solid #444;padding:2px 4px;font-size:11px;"
         onchange="location.href='/?end='+this.value+'&win={win}'">
  <input type="number" id="gotoWin" value="{win}" min="10" max="1000"
         style="width:50px;background:#111;color:#ccc;border:1px solid #444;padding:2px 4px;font-size:11px;"
         onchange="location.href='/?end={end_bar}&win='+this.value">
  <span style="color:#888;font-size:12px;">|</span>
  <span style="color:#666;font-size:11px;">
    Sym:{len(sym_structures)} Pred:{len(short_preds)}
  </span>
</div>
'''
    html = html.replace('<body>', '<body>' + nav_html, 1)

    # === 注入标注面板 (在</script>之前) ===
    # 标注面板HTML放在canvas和info之后
    annotation_panel_html = '''
<!-- Annotation Panel -->
<div id="annotPanel" style="background:#0d0d20;border:1px solid #335;border-radius:4px;padding:8px 12px;margin-top:6px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
    <span style="color:#f0c040;font-weight:bold;font-size:13px;">Annotate</span>
    <span style="color:#888;font-size:11px;">双击图上线段选择 | 点击框激活 | 手工输入: bar1 price1 bar2 price2</span>
    <span style="color:#555;font-size:11px;">双击选线 | 框内输入: b1 p1 b2 p2 或 H3 L5</span>
  </div>
  <div style="display:flex;gap:4px;flex-wrap:wrap;align-items:flex-start;">
    <div id="segBoxes" style="display:flex;gap:3px;flex-wrap:wrap;"></div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-top:6px;">
    <span style="color:#aaa;font-size:11px;">Label:</span>
    <div style="position:relative;display:inline-block;">
      <input type="text" id="annotLabel" placeholder="e.g. A(B)abcC, 12345, abcde ..."
             style="width:280px;background:#111;color:#eee;border:1px solid #444;padding:3px 6px;font-size:12px;font-family:monospace;"
             onfocus="showLabelHist()" oninput="filterLabelHist()">
      <div id="labelHist" style="display:none;position:absolute;top:100%;left:0;width:320px;max-height:200px;overflow-y:auto;background:#111;border:1px solid #555;border-radius:3px;z-index:999;"></div>
    </div>
    <button onclick="submitAnnotation()" style="background:#2a5a2a;color:#8f8;border:1px solid #4a4;padding:4px 14px;border-radius:3px;cursor:pointer;font-size:12px;font-weight:bold;">Submit</button>
    <button onclick="clearAnnotation()" style="background:#3a1a1a;color:#f88;border:1px solid #644;padding:4px 10px;border-radius:3px;cursor:pointer;font-size:11px;">Clear</button>
    <button onclick="undoLastSeg()" style="background:#2a2a1a;color:#ff8;border:1px solid #554;padding:4px 10px;border-radius:3px;cursor:pointer;font-size:11px;">Undo</button>
    <span id="annotStatus" style="color:#888;font-size:11px;margin-left:8px;"></span>
  </div>
</div>
<div id="annotLog" style="background:#080818;border:1px solid #222;border-radius:4px;padding:6px 10px;margin-top:4px;max-height:300px;overflow-y:auto;font-size:11px;font-family:monospace;color:#999;display:none;"></div>
'''

    # 在 </script></body> 之前注入面板HTML（放到info div之后）
    html = html.replace(
        '<div id="info"',
        annotation_panel_html + '\n<div id="info"'
    )

    # === 注入标注JS (在最后的 draw(); 之后，</script>之前) ===
    annotation_js = '''

// ========== ANNOTATION SYSTEM ==========
const ANN_MAX = 9;
let annSlots = [];  // [{b1,p1,b2,p2,source},...] selected segments
let annActive = 0;  // which slot is active (0-indexed)
let annLocked = false;

// Build segment boxes UI
(function buildAnnotUI() {
  const container = document.getElementById('segBoxes');
  for(let i = 0; i < ANN_MAX; i++) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;flex-direction:column;gap:1px;';

    const box = document.createElement('div');
    box.id = 'segBox' + i;
    box.style.cssText = 'min-width:130px;height:40px;background:#111;border:2px solid #333;border-top-left-radius:3px;border-top-right-radius:3px;padding:3px 5px;cursor:pointer;font-size:10px;font-family:monospace;color:#777;display:flex;flex-direction:column;justify-content:center;';
    box.innerHTML = '<div style="color:#556;text-align:center;">S' + (i+1) + '</div>';
    box.onclick = function() { setActiveSlot(i); };

    const inp = document.createElement('input');
    inp.type = 'text';
    inp.id = 'segInp' + i;
    inp.placeholder = 'b1 p1 b2 p2';
    inp.style.cssText = 'width:126px;background:#0a0a18;color:#ff8;border:1px solid #333;border-top:none;border-bottom-left-radius:3px;border-bottom-right-radius:3px;padding:2px 4px;font-size:9px;font-family:monospace;';
    inp.onfocus = function() { setActiveSlot(i); };
    (function(idx) {
      inp.onkeydown = function(ev) {
        if(ev.key === 'Enter') {
          parseManualInput(idx, inp.value.trim());
          ev.preventDefault();
        }
      };
    })(i);

    wrapper.appendChild(box);
    wrapper.appendChild(inp);
    container.appendChild(wrapper);
  }
  setActiveSlot(0);
})();

function parseManualInput(idx, val) {
  if(annLocked) return;
  if(!val) return;
  const parts = val.split(/\\s+/);
  if(parts.length === 4) {
    const b1 = parseInt(parts[0]), p1 = parseFloat(parts[1]);
    const b2 = parseInt(parts[2]), p2 = parseFloat(parts[3]);
    if(!isNaN(b1) && !isNaN(p1) && !isNaN(b2) && !isNaN(p2)) {
      const seg = { b1, p1, b2, p2, source: 'MANUAL' };
      if(!checkContinuity(idx, seg)) {
        const prev = annSlots[idx - 1];
        setAnnotStatus('Not continuous! S' + idx + ' end=b' + prev.b2 + ' but input b' + b1, '#f88');
        return;
      }
      annSlots[idx] = seg;
      renderSlot(idx);
      drawAnnotHighlights();
      setAnnotStatus('Manual S' + (idx+1) + ': b' + b1 + '→b' + b2, '#8f8');
      document.getElementById('segInp' + idx).value = '';
      if(idx < ANN_MAX - 1) setActiveSlot(idx + 1);
      return;
    }
  }
  // Try H/L notation: e.g. "H3 L5" → lookup from TOP array
  if(parts.length === 2) {
    const p1Info = resolvePointLabel(parts[0]);
    const p2Info = resolvePointLabel(parts[1]);
    if(p1Info && p2Info) {
      const seg = { b1: p1Info.bar, p1: p1Info.price, b2: p2Info.bar, p2: p2Info.price, source: 'MANUAL:' + parts[0] + '-' + parts[1] };
      if(!checkContinuity(idx, seg)) {
        const prev = annSlots[idx - 1];
        setAnnotStatus('Not continuous! S' + idx + ' end=b' + prev.b2 + ' but ' + parts[0] + '=b' + p1Info.bar, '#f88');
        return;
      }
      annSlots[idx] = seg;
      renderSlot(idx);
      drawAnnotHighlights();
      setAnnotStatus('Manual S' + (idx+1) + ': ' + parts[0] + '→' + parts[1], '#8f8');
      document.getElementById('segInp' + idx).value = '';
      if(idx < ANN_MAX - 1) setActiveSlot(idx + 1);
      return;
    }
  }
  setAnnotStatus('Format: "b1 p1 b2 p2" or "H3 L5" (point labels)', '#f88');
}

// Resolve H3, L5 etc to {bar, price} from TOP array
function resolvePointLabel(label) {
  const m = label.match(/^([HL])(\\d+)$/i);
  if(!m) return null;
  const dir = m[1].toUpperCase();
  const rank = parseInt(m[2]);
  const arr = dir === 'H' ? PEAKS : VALS;
  if(rank < 1 || rank > arr.length) return null;
  const pt = arr[rank - 1];
  return { bar: pt.bar, price: pt.price };
}

function setActiveSlot(idx) {
  if(annLocked) return;
  annActive = idx;
  for(let i = 0; i < ANN_MAX; i++) {
    const box = document.getElementById('segBox' + i);
    box.style.borderColor = (i === idx) ? '#f0c040' : (annSlots[i] ? '#4a4' : '#333');
  }
}

function renderSlot(idx) {
  const box = document.getElementById('segBox' + idx);
  const seg = annSlots[idx];
  if(!seg) {
    box.innerHTML = '<div style="color:#556;text-align:center;">S' + (idx+1) + '</div>';
    box.style.borderColor = (idx === annActive) ? '#f0c040' : '#333';
    return;
  }
  const d = seg.p2 > seg.p1 ? '↑' : '↓';
  const dc = seg.p2 > seg.p1 ? '#4f4' : '#f44';
  const lbl1 = seg.lbl1 || barToLabel(seg.b1);
  const lbl2 = seg.lbl2 || barToLabel(seg.b2);
  box.innerHTML =
    '<div style="color:#aaa;">S' + (idx+1) + ' <span style="color:#ff0;font-weight:bold;">' + lbl1 + '→' + lbl2 + '</span> <span style="color:' + dc + ';">' + d + '</span></div>' +
    '<div style="color:#8cf;font-size:9px;">b' + seg.b1 + ' ' + seg.p1.toFixed(5) + ' <span style="color:#556;">' + (seg.source||'') + '</span></div>' +
    '<div style="color:#fc8;font-size:9px;">b' + seg.b2 + ' ' + seg.p2.toFixed(5) + '</div>';
  box.style.borderColor = (idx === annActive) ? '#f0c040' : '#4a4';
}

// Collect all visible segments for hit-test
function collectVisibleSegs() {
  const segs = [];

  // Snapshots: each snapshot is a polyline, extract consecutive segments
  for(let si = 0; si < S.length; si++) {
    if(!vis[si]) continue;
    const pts = S[si].pts;
    for(let j = 0; j < pts.length - 1; j++) {
      segs.push({
        b1: pts[j].bar, p1: pts[j].y,
        b2: pts[j+1].bar, p2: pts[j+1].y,
        source: S[si].label
      });
    }
  }

  // Extra segments
  for(let gi = 0; gi < EX.length; gi++) {
    if(!exVis[gi]) continue;
    for(const seg of EX[gi].segs) {
      segs.push({
        b1: seg.b1, p1: seg.p1,
        b2: seg.b2, p2: seg.p2,
        source: 'EX:' + EX[gi].label
      });
    }
  }

  // Fusion segments
  if(showFusion) {
    const fn = Math.min(fusionTopN, FUS.length);
    for(let i = 0; i < fn; i++) {
      const f = FUS[i];
      if(f.imp < minImp) continue;
      segs.push({
        b1: f.b1, p1: f.p1,
        b2: f.b2, p2: f.p2,
        source: 'FUS:' + f.src
      });
    }
  }

  // Symmetry structure segments (A, B, C)
  if(showSym) {
    const sn = Math.min(symTopN, SYM.length);
    for(let i = 0; i < sn; i++) {
      const s = SYM[i];
      segs.push({ b1: s.p1, p1: s.pp1, b2: s.p2, p2: s.pp2, source: 'SYM:A' + (i+1) });
      segs.push({ b1: s.p2, p1: s.pp2, b2: s.p3, p2: s.pp3, source: 'SYM:B' + (i+1) });
      segs.push({ b1: s.p3, p1: s.pp3, b2: s.p4, p2: s.pp4, source: 'SYM:C' + (i+1) });
    }
  }

  // Prediction segments (A and predicted C)
  if(showPred) {
    const pn = Math.min(predTopN, PRED.length);
    for(let i = 0; i < pn; i++) {
      const p = PRED[i];
      segs.push({ b1: p.A_s, p1: p.A_ps, b2: p.A_e, p2: p.A_pe, source: 'PRED:A' + (i+1) });
      segs.push({ b1: p.ps_bar, p1: p.ps_prc, b2: p.pt_bar, p2: p.pt_prc, source: 'PRED:C' + (i+1) });
      if(p.B_s !== undefined) {
        segs.push({ b1: p.B_s, p1: p.B_ps, b2: p.B_e, p2: p.B_pe, source: 'PRED:B' + (i+1) });
      }
    }
  }

  return segs;
}

// Distance from point to line segment (in pixel space)
function distToSeg(mx, my, seg) {
  const x1 = xS(seg.b1), y1 = yS(seg.p1);
  const x2 = xS(seg.b2), y2 = yS(seg.p2);
  const dx = x2 - x1, dy = y2 - y1;
  const len2 = dx*dx + dy*dy;
  if(len2 === 0) return Math.sqrt((mx-x1)*(mx-x1) + (my-y1)*(my-y1));
  let t = ((mx-x1)*dx + (my-y1)*dy) / len2;
  t = Math.max(0, Math.min(1, t));
  const px = x1 + t*dx, py = y1 + t*dy;
  return Math.sqrt((mx-px)*(mx-px) + (my-py)*(my-py));
}

// Check continuity: new segment's start must match previous segment's end
function checkContinuity(slotIdx, seg) {
  if(slotIdx === 0) return true;  // first segment, no constraint
  const prev = annSlots[slotIdx - 1];
  if(!prev) return true;  // previous slot empty, allow (will validate on submit)
  return prev.b2 === seg.b1;
}

// Find pivot label (H3/L5) for a given bar
function barToLabel(bar) {
  for(const p of PEAKS) { if(p.bar === bar) return p.label; }
  for(const v of VALS) { if(v.bar === bar) return v.label; }
  return 'b' + bar;
}

// Try to find a bridge segment connecting prevEnd to segStart
function findBridgeSeg(prevEndBar, segStartBar, allSegs) {
  let best = null, bestImp = -1;
  for(const s of allSegs) {
    if(s.b1 === prevEndBar && s.b2 === segStartBar) {
      // Prefer higher importance (fusion > lat > amp > base)
      const imp = s.source.startsWith('FUS:') ? 3 :
                  s.source.startsWith('L') ? 2 :
                  s.source.startsWith('A') ? 1 : 0;
      if(imp > bestImp) { bestImp = imp; best = s; }
    }
  }
  return best;
}

// Double-click handler for segment selection
cv.addEventListener('dblclick', function(e) {
  if(annLocked) return;
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  const allSegs = collectVisibleSegs();
  let best = null, bestDist = 20;

  for(const seg of allSegs) {
    const d = distToSeg(mx, my, seg);
    if(d < bestDist) {
      bestDist = d;
      best = seg;
    }
  }

  if(!best) {
    setAnnotStatus('No segment found near click', '#f88');
    return;
  }

  // Check continuity — if not continuous, try to auto-fill bridge (B segment)
  if(!checkContinuity(annActive, best)) {
    const prev = annSlots[annActive - 1];
    if(prev) {
      // Try to find a bridge segment: prev.b2 → best.b1
      const bridge = findBridgeSeg(prev.b2, best.b1, allSegs);
      if(bridge) {
        // Auto-fill bridge as current slot, then put best in next slot
        annSlots[annActive] = { b1: bridge.b1, p1: bridge.p1, b2: bridge.b2, p2: bridge.p2, source: bridge.source, lbl1: barToLabel(bridge.b1), lbl2: barToLabel(bridge.b2) };
        renderSlot(annActive);
        const bridgeSlot = annActive;
        annActive++;
        if(annActive < ANN_MAX) {
          annSlots[annActive] = { b1: best.b1, p1: best.p1, b2: best.b2, p2: best.p2, source: best.source, lbl1: barToLabel(best.b1), lbl2: barToLabel(best.b2) };
          renderSlot(annActive);
          setAnnotStatus('Auto-filled S' + (bridgeSlot+1) + ' as bridge (' + bridge.source + '), S' + (annActive+1) + ': ' + barToLabel(best.b1) + '→' + barToLabel(best.b2), '#8f8');
          drawAnnotHighlights();
          if(annActive < ANN_MAX - 1) setActiveSlot(annActive + 1);
          return;
        }
      }
      // No bridge found
      setAnnotStatus('Not continuous! ' + barToLabel(prev.b2) + '(b' + prev.b2 + ') → ' + barToLabel(best.b1) + '(b' + best.b1 + ') no bridge found', '#f88');
      return;
    }
  }

  // Fill slot
  annSlots[annActive] = { b1: best.b1, p1: best.p1, b2: best.b2, p2: best.p2, source: best.source, lbl1: barToLabel(best.b1), lbl2: barToLabel(best.b2) };
  renderSlot(annActive);
  setAnnotStatus('S' + (annActive+1) + ': ' + barToLabel(best.b1) + '→' + barToLabel(best.b2) + ' (' + best.source + ')', '#8f8');

  drawAnnotHighlights();

  if(annActive < ANN_MAX - 1) {
    setActiveSlot(annActive + 1);
  }
});

// Draw highlights for selected segments + K-line highlighting
function drawAnnotHighlights() {
  // Redraw base first
  draw();

  // Collect bar range covered by all selected segments
  let barMin = Infinity, barMax = -Infinity;
  for(let i = 0; i < annSlots.length; i++) {
    const seg = annSlots[i];
    if(!seg) continue;
    barMin = Math.min(barMin, seg.b1, seg.b2);
    barMax = Math.max(barMax, seg.b1, seg.b2);
  }

  cx.save();

  // Highlight K-lines in the covered range
  if(barMin <= barMax) {
    for(let b = barMin; b <= barMax && b < K.length; b++) {
      if(b < 0) continue;
      const k = K[b];
      const x = xS(b);
      // Bright candlestick body
      const isUp = k.c >= k.o;
      cx.strokeStyle = isUp ? '#00cc44' : '#cc3333';
      cx.lineWidth = 2;
      cx.globalAlpha = 0.7;
      // High-low wick
      cx.beginPath();
      cx.moveTo(x, yS(k.l));
      cx.lineTo(x, yS(k.h));
      cx.stroke();
      // Body
      const bodyTop = yS(Math.max(k.o, k.c));
      const bodyBot = yS(Math.min(k.o, k.c));
      const bodyH = Math.max(bodyBot - bodyTop, 1);
      const bw = Math.max(pw / K.length * 0.6, 2);
      cx.fillStyle = isUp ? '#00cc44' : '#cc3333';
      cx.globalAlpha = 0.5;
      cx.fillRect(x - bw/2, bodyTop, bw, bodyH);
    }
  }

  // Overlay selected segments
  for(let i = 0; i < annSlots.length; i++) {
    const seg = annSlots[i];
    if(!seg) continue;
    const hue = (i * 40) % 360;
    cx.strokeStyle = 'hsl(' + hue + ', 100%, 70%)';
    cx.lineWidth = 4;
    cx.globalAlpha = 0.85;
    cx.setLineDash([]);
    cx.beginPath();
    cx.moveTo(xS(seg.b1), yS(seg.p1));
    cx.lineTo(xS(seg.b2), yS(seg.p2));
    cx.stroke();

    // Endpoint dots
    cx.fillStyle = 'hsl(' + hue + ', 100%, 80%)';
    cx.beginPath();
    cx.arc(xS(seg.b1), yS(seg.p1), 4, 0, Math.PI*2);
    cx.fill();
    cx.beginPath();
    cx.arc(xS(seg.b2), yS(seg.p2), 4, 0, Math.PI*2);
    cx.fill();

    // Label with pivot names
    cx.fillStyle = 'hsl(' + hue + ', 100%, 85%)';
    cx.font = 'bold 11px monospace';
    cx.textAlign = 'center';
    const smx = (xS(seg.b1) + xS(seg.b2)) / 2;
    const smy = (yS(seg.p1) + yS(seg.p2)) / 2;
    const sl1 = seg.lbl1 || barToLabel(seg.b1);
    const sl2 = seg.lbl2 || barToLabel(seg.b2);
    cx.fillText('S' + (i+1) + ' ' + sl1 + '→' + sl2, smx, smy - 8);
  }
  cx.restore();
  cx.setLineDash([]);
  cx.globalAlpha = 1;
  cx.textAlign = 'start';
}

function setAnnotStatus(msg, color) {
  const el = document.getElementById('annotStatus');
  el.textContent = msg;
  el.style.color = color || '#888';
}

function clearAnnotation() {
  annSlots = [];
  annLocked = false;
  for(let i = 0; i < ANN_MAX; i++) renderSlot(i);
  setActiveSlot(0);
  setAnnotStatus('Cleared', '#888');
  draw();
}

function undoLastSeg() {
  if(annLocked) return;
  // Find last filled slot
  let last = -1;
  for(let i = ANN_MAX - 1; i >= 0; i--) {
    if(annSlots[i]) { last = i; break; }
  }
  if(last >= 0) {
    delete annSlots[last];
    annSlots.length = last;
    renderSlot(last);
    setActiveSlot(last);
    setAnnotStatus('Undone S' + (last+1), '#ff8');
    drawAnnotHighlights();
  }
}

async function submitAnnotation() {
  // Validate
  const filled = annSlots.filter(s => s);
  if(filled.length < 2) {
    setAnnotStatus('Need at least 2 segments', '#f88');
    return;
  }

  const label = document.getElementById('annotLabel').value.trim();
  if(!label) {
    setAnnotStatus('Please enter a label/description', '#f88');
    document.getElementById('annotLabel').focus();
    return;
  }

  // Validate continuity chain
  for(let i = 1; i < filled.length; i++) {
    if(filled[i].b1 !== filled[i-1].b2) {
      setAnnotStatus('Gap between S' + i + ' and S' + (i+1) + ': b' + filled[i-1].b2 + ' != b' + filled[i].b1, '#f88');
      return;
    }
  }

  annLocked = true;
  setAnnotStatus('Submitting...', '#ff8');

  try {
    const resp = await fetch('/annotate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        segments: filled,
        label: label,
        window: { start: WINDOW_START, end: WINDOW_END, win: WINDOW_WIN },
      })
    });
    const result = await resp.json();

    if(result.error) {
      setAnnotStatus('Error: ' + result.error, '#f88');
      annLocked = false;
      return;
    }

    // Show result in log
    showAnnotResult(result);
    setAnnotStatus('Saved! ID=' + result.record_id + ' | Total=' + result.total_annotations + ' | Same label=' + result.same_label_count, '#8f8');

    // Keep locked until user clicks Clear
  } catch(e) {
    setAnnotStatus('Network error: ' + e.message, '#f88');
    annLocked = false;
  }
}

function showAnnotResult(r) {
  const log = document.getElementById('annotLog');
  log.style.display = 'block';

  let html = '<div style="border-bottom:1px solid #333;padding:4px 0;margin-bottom:4px;">';
  html += '<span style="color:#f0c040;font-weight:bold;">[' + r.record_id + ']</span> ';
  html += '<span style="color:#8cf;">' + r.n_segments + ' segs</span> ';
  html += '<span style="color:#aaa;">dir=' + r.dir_sequence + '</span> ';
  html += '<span style="color:#888;">net=' + r.summary.net_dir + ' total_bars=' + r.summary.total_bars + ' total_amp=' + r.summary.total_amp + '</span>';
  html += '</div>';

  // Features table
  html += '<div style="color:#778;">Segments:</div>';
  html += '<table style="font-size:10px;color:#aaa;border-collapse:collapse;margin:2px 0;">';
  html += '<tr style="color:#668;"><td>Seg</td><td>Bar</td><td>Amp</td><td>Time</td><td>Dir</td><td>Slope</td><td>Src</td></tr>';
  for(const f of r.features) {
    const dc = f.dir === 1 ? '#4f4' : '#f44';
    html += '<tr><td>S' + f.idx + '</td><td>b' + f.b1 + '→b' + f.b2 + '</td>';
    html += '<td>' + f.amp.toFixed(5) + '</td><td>' + f.time + '</td>';
    html += '<td style="color:' + dc + ';">' + f.dir_label + '</td>';
    html += '<td>' + f.slope.toFixed(7) + '</td><td style="color:#556;">' + f.source + '</td></tr>';
  }
  html += '</table>';

  // Ratios
  if(r.ratios && r.ratios.length > 0) {
    html += '<div style="color:#778;margin-top:3px;">Ratios:</div>';
    html += '<table style="font-size:10px;color:#aaa;border-collapse:collapse;margin:2px 0;">';
    html += '<tr style="color:#668;"><td>Pair</td><td>AmpR</td><td>TimeR</td><td>Retrace</td></tr>';
    for(const r2 of r.ratios) {
      html += '<tr><td>' + r2.pair + '</td>';
      html += '<td>' + r2.amp_ratio.toFixed(4) + '</td>';
      html += '<td>' + r2.time_ratio.toFixed(4) + '</td>';
      html += '<td>' + (r2.retrace !== null ? r2.retrace.toFixed(4) : '-') + '</td></tr>';
    }
    html += '</table>';
  }

  // Historical constraints
  if(r.constraint_stats) {
    const cs = r.constraint_stats;
    html += '<div style="color:#f80;margin-top:3px;">Historical constraints (' + cs.n_historical + ' same-label cases):</div>';
    html += '<pre style="color:#ca8;font-size:10px;margin:2px 0;">' + JSON.stringify(cs.stats, null, 1) + '</pre>';
  }

  html += '<div style="color:#555;font-size:9px;">same_label=' + r.same_label_count + ' total=' + r.total_annotations + '</div>';

  // Action buttons
  html += '<div style="margin-top:3px;">';
  html += '<button onclick="deleteAnnot(\'' + r.record_id + '\', this)" style="background:#3a1a1a;color:#f88;border:1px solid #644;padding:2px 8px;border-radius:2px;cursor:pointer;font-size:10px;margin-right:4px;">Delete</button>';
  html += '<button onclick="editAnnotLabel(\'' + r.record_id + '\', this)" style="background:#1a2a3a;color:#8cf;border:1px solid #446;padding:2px 8px;border-radius:2px;cursor:pointer;font-size:10px;">Edit Label</button>';
  html += '</div>';

  // Prepend (newest first)
  log.innerHTML = html + log.innerHTML;
}

async function deleteAnnot(id, btn) {
  if(!confirm('Delete annotation ' + id + '?')) return;
  try {
    const resp = await fetch('/annotate/' + id, { method: 'DELETE' });
    const r = await resp.json();
    if(r.status === 'deleted') {
      btn.closest('div').parentElement.style.opacity = '0.3';
      btn.closest('div').parentElement.style.textDecoration = 'line-through';
      btn.disabled = true;
      setAnnotStatus('Deleted ' + id + ' | remaining=' + r.remaining, '#f88');
    } else {
      setAnnotStatus('Delete failed: ' + (r.error || 'unknown'), '#f88');
    }
  } catch(e) { setAnnotStatus('Delete error: ' + e.message, '#f88'); }
}

async function editAnnotLabel(id, btn) {
  const newLabel = prompt('New label for ' + id + ':');
  if(!newLabel || !newLabel.trim()) return;
  try {
    const resp = await fetch('/annotate/' + id, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: newLabel.trim() })
    });
    const r = await resp.json();
    if(r.status === 'updated') {
      setAnnotStatus('Updated ' + id + ': "' + r.old_label + '" -> "' + r.new_label + '"', '#8cf');
    } else {
      setAnnotStatus('Update failed: ' + (r.error || 'unknown'), '#f88');
    }
  } catch(e) { setAnnotStatus('Update error: ' + e.message, '#f88'); }
}

// Override the original draw to also show annotation highlights
const _origDraw = draw;
// We can't easily override since draw is declared with function keyword.
// Instead, we patch: after any draw() call from v3 controls, redraw highlights.
// Use MutationObserver or interval is ugly. Better: wrap toggle functions.
const _patchedFns = ['showAll','hideAll','showType','showBase','toggleExtra','toggleTop',
  'toggleImpVal','toggleFusion','toggleSym','togglePred','showStep'];
for(const fn of _patchedFns) {
  const orig = window[fn];
  if(orig) {
    window[fn] = function() {
      orig.apply(this, arguments);
      if(annSlots.some(s => s)) drawAnnotHighlights();
    };
  }
}
// Also patch slider updates
const _patchedSliders = ['updatePeakN','updateValleyN','updateFusionN','updateSymN','updatePredN','updateMinImp'];
for(const fn of _patchedSliders) {
  const orig = window[fn];
  if(orig) {
    window[fn] = function(v) {
      orig.call(this, v);
      if(annSlots.some(s => s)) drawAnnotHighlights();
    };
  }
}
// Label history dropdown
function showLabelHist() {
  const hist = document.getElementById('labelHist');
  if(!HIST_LABELS || HIST_LABELS.length === 0) { hist.style.display='none'; return; }
  filterLabelHist();
  hist.style.display = 'block';
}

function filterLabelHist() {
  const hist = document.getElementById('labelHist');
  const val = document.getElementById('annotLabel').value.toLowerCase();
  const filtered = val ? HIST_LABELS.filter(l => l.toLowerCase().includes(val)) : HIST_LABELS;
  if(filtered.length === 0) { hist.style.display='none'; return; }
  let h = '';
  for(const lb of filtered) {
    h += '<div class="lhItem" onclick="pickLabel(this)">' + lb + '</div>';
  }
  hist.innerHTML = h;
  hist.style.display = 'block';
}

function pickLabel(el) {
  document.getElementById('annotLabel').value = el.textContent;
  document.getElementById('labelHist').style.display = 'none';
}

// Hide dropdown when clicking elsewhere
document.addEventListener('click', function(e) {
  if(!e.target.closest || (!e.target.closest('#annotLabel') && !e.target.closest('#labelHist'))) {
    document.getElementById('labelHist').style.display = 'none';
  }
});

// ========== END ANNOTATION SYSTEM ==========
'''

    # 读取历史标注labels（去重，最近的在前，最多20条）
    hist_labels = []
    if os.path.exists(ANNOTATIONS_FILE):
        try:
            with open(ANNOTATIONS_FILE, 'r') as f:
                all_a = json.load(f)
            seen = set()
            for a in reversed(all_a):
                lb = a.get('label', '')
                if lb and lb not in seen:
                    seen.add(lb)
                    hist_labels.append(lb)
                if len(hist_labels) >= 20:
                    break
        except Exception:
            pass

    # 注入窗口信息常量（供JS使用）
    window_constants = f'''
const WINDOW_START = {start};
const WINDOW_END = {end_bar};
const WINDOW_WIN = {win};
const HIST_LABELS = {json.dumps(hist_labels, ensure_ascii=False)};
'''

    # 在 draw(); 之后注入
    html = html.replace(
        "document.getElementById('predBtn').classList.add('active');",
        "document.getElementById('predBtn').classList.add('active');\n"
        + window_constants
        + annotation_js
    )

    # 修改标题
    html = html.replace('归并引擎 v3.1', f'v33 | bars {start}~{end_bar} win={win}')

    # 键盘导航（不和标注冲突）
    kb_js = f'''
<script>
document.addEventListener("keydown", function(e) {{
  if(e.target.tagName === "INPUT") return;
  if(e.key === "ArrowRight" && !e.shiftKey) location.href = "/?end={min(n, end_bar+1)}&win={win}";
  if(e.key === "ArrowRight" && e.shiftKey) location.href = "/?end={min(n, end_bar+10)}&win={win}";
  if(e.key === "ArrowLeft" && !e.shiftKey) location.href = "/?end={max(win, end_bar-1)}&win={win}";
  if(e.key === "ArrowLeft" && e.shiftKey) location.href = "/?end={max(win, end_bar-10)}&win={win}";
}});
</script>
'''
    html = html.replace('</body>', kb_js + '</body>')

    print(f"  /?end={end_bar}&win={win} → {elapsed*1000:.0f}ms, {len(html)//1024}KB", flush=True)
    return HTMLResponse(content=html)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='v33 Static Server')
    parser.add_argument('--bars', type=int, default=5000)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()

    app._bars = args.bars
    print(f"Starting v33 server on port {args.port}...", flush=True)
    uvicorn.run(app, host='0.0.0.0', port=args.port)
