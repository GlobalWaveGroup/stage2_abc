#!/usr/bin/env python3
"""
v33 Server — v3静态HTML + HTTP参数化交互
每次请求 /?end=500&win=200 → 服务端跑v3完整pipeline → 返回完整静态HTML

这就是v3原版，只是参数从URL来而不是写死在main()里。
v3的全部功能原封不动保留。

启动: python3 visualize_v33_server.py [--bars 5000] [--port 8765]
访问: http://10.10.0.4:8765/?end=500&win=200
"""

import sys
import time
import argparse

sys.path.insert(0, '/home/ubuntu/stage2_abc')

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn

from merge_engine_v3 import (
    load_kline, calculate_base_zg, full_merge_engine,
    compute_pivot_importance, build_segment_pool,
    pool_fusion, predict_symmetric_image, find_symmetric_structures
)
from visualize_v3 import generate_html as v3_generate_html

import io

app = FastAPI()
G = {}


@app.on_event("startup")
def startup():
    bars = getattr(app, '_bars', 5000)
    print(f"Loading EURUSD H1 data (limit={bars})...", flush=True)
    df = load_kline("/home/ubuntu/DataBase/base_kline/EURUSD_H1.csv", limit=bars)
    G['df'] = df
    G['n'] = len(df)
    print(f"Loaded: {G['n']} bars", flush=True)


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

    # Symmetry + Predictions (max_pool_size限速，避免卡)
    try:
        sym_structures = find_symmetric_structures(full_pool, pi, df=df, top_n=200, max_pool_size=500)
    except Exception:
        sym_structures = []

    predictions = predict_symmetric_image(full_pool, pi, current_bar=len(df) - 1, max_pool_size=500)
    short_preds = sorted([p for p in predictions if p['pred_time'] <= 50], key=lambda p: -p['score'])

    elapsed = time.time() - t0

    # generate_html → 临时文件 → 读回
    import tempfile, os
    tmp = tempfile.mktemp(suffix='.html')
    v3_generate_html(df, results, tmp,
                     pivot_info=pi, fusion_segs=fusion_segs,
                     sym_structures=sym_structures,
                     predictions=short_preds[:200])
    with open(tmp, 'r') as f:
        html = f.read()
    os.unlink(tmp)

    # 注入导航控件到HTML (在<body>标签后插入)
    nav_html = f'''
<div style="background:#1a1a2e;padding:6px 12px;margin-bottom:4px;border:1px solid #333;border-radius:4px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
  <span style="color:#7eb8da;font-weight:bold;font-size:13px;">v33 Nav</span>
  <span style="color:#888;font-size:12px;">|</span>
  <span style="color:#aaa;font-size:12px;">Bars: {start}~{end_bar} (win={win}) | Total: {n} | Compute: {elapsed*1000:.0f}ms</span>
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
         onchange="location.href=&apos;/?end=&apos;+this.value+&apos;&win={win}&apos;">
  <input type="number" id="gotoWin" value="{win}" min="10" max="1000"
         style="width:50px;background:#111;color:#ccc;border:1px solid #444;padding:2px 4px;font-size:11px;"
         onchange="location.href=&apos;/?end={end_bar}&win=&apos;+this.value">
  <span style="color:#888;font-size:12px;">|</span>
  <span style="color:#666;font-size:11px;">
    Sym:{len(sym_structures)} Pred:{len(short_preds)}
    (M:{sum(1 for p in short_preds[:200] if p["type"]=="mirror")}
     C:{sum(1 for p in short_preds[:200] if p["type"]=="center")}
     T:{sum(1 for p in short_preds[:200] if p["type"]=="triangle")}
     O:{sum(1 for p in short_preds[:200] if p["type"]=="modonly")})
  </span>
</div>
<script>
document.addEventListener("keydown", function(e) {{
  if(e.target.tagName === "INPUT") return;
  if(e.key === "ArrowRight") location.href = "/?end={min(n, end_bar + (10 if 'shiftKey' else 1))}&win={win}";
  if(e.key === "ArrowLeft") location.href = "/?end={max(win, end_bar - (10 if 'shiftKey' else 1))}&win={win}";
}});
</script>
'''

    # 插入到<body>后面
    html = html.replace('<body>', '<body>' + nav_html, 1)

    # 修改标题
    html = html.replace('归并引擎 v3.1', f'v33 | bars {start}~{end_bar} win={win}')

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
