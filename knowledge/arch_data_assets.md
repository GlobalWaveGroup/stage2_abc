# 数据资产清单

## 1. 原始数据: /home/ubuntu/DataBase/base_kline/

- 48品种 × 5TF (M1/M5/M15/M30/H1)
- 格式: TSV, 列 = date time open high low close tickvol vol spread
- 数据范围: 2000年~2025年 (~25年)
- 例: EURUSD_M1.csv = 554MB, H1 = 9.7MB
- load函数: `merge_engine_v3.load_kline(path, limit=None)` → DataFrame[datetime,open,high,low,close]

## 2. 归一化数据: /home/ubuntu/database2/

### 2.1 归一化OHLC
- 路径: `database2/{TF}/{SYMBOL}_{TF}_norm.csv` (也在 `database2/normalization/{TF}/`)
- 106品种 × 4TF (M5/M15/M30/H1). **M1为空**.
- 格式: CSV, 列 = `open,high,low,close,return`
- 归一化方法: 全序列除以第一根K线的close价 → close[0]=1.000000
- `return`列: zigzag衍生的方向性标签, 连续值, 非简单涨跌分类
- 无时间戳列

### 2.2 数据量参考 (EURUSD)
| TF  | 行数      | 约年数 |
|-----|----------|--------|
| M5  | 1,856,492| ~14年  |
| M15 | 619,981  | ~14年  |
| M30 | 310,196  | ~14年  |
| H1  | 155,228  | ~14年  |

### 2.3 品种列表 (106个)
主要: EURUSD GBPUSD USDJPY USDCHF AUDUSD NZDUSD USDCAD
交叉: EURJPY GBPJPY EURGBP AUDNZD CHFJPY GBPCHF 等
异国: USDTRY USDZAR USDMXN USDNOK USDSEK USDPLN USDHUF EURTRY GBPTRY 等
贵金属: XAUUSD XAGUSD XAUAUD XAUCHF XAUGBP XAUJPY XAUCAD XAUCNH XAGAUD XAGCAD XAGCHF XAGEUR XAGGBP XAGJPY
亚洲: USDCNH USDRMB USDHKD USDSGD CNHJPY SGDJPY HKDJPY

## 3. 预建Zigzag图: /home/ubuntu/database2/build_zig_all/cl128/

- 窗口128 bars, 步长28 bars 的滑动窗口特征提取
- 每个.npz含: features(N,10), edges_bytes, n_records, n_bars, window_size, step_size
- 完成状态: H1✓ M15✓ M30✓ | M5部分(20/106) | M1空

## 4. Zigzag图 v2.1: /home/ubuntu/database2/zig_all_v21/cl128/

- **仅M15完成** (106品种全部)
- 更丰富的图结构: pivot_indices, point_features(N,6), adjacency(N,2), edges(E,10), birth_indices
- 使用ZZP2算法 (deviation=1%, max_merge_levels=8)
- 构建脚本: `zig_all_v21/build_v21_parallel.py`
- 核心依赖: `/home/ubuntu/tianshu_v10/kongfang_platform/features/zzp2.py`

## 5. ML特征: /home/ubuntu/database2/tokenizer/v10.1_multistep/

- 48品种 × 5TF = 240个.npz
- 每个含: features(N,8), return_2, return_5, return_10, return_32
- 总样本: 22,183,885
- 元数据: metadata.json

## 6. 关键设计原则

- **时间戳无意义**: K线的价值在于相对位置和几何关系
- **归一化的意义**: 不同品种的价格尺度不同, 归一化使zigzag结构跨品种可比
- **归并引擎不依赖时间戳**: calculate_base_zg 只需要 high/low 数组
