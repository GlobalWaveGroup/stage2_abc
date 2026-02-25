"""
TianShu Strategy Engine V3
============================
Production-oriented refactor of integrated_strategy_v2.
Verified: reproduces V2 results exactly (A_base through E_final).

Architecture:
  1. RAGProvider (abstract) — pluggable backend (KDTree/FAISS/external API)
  2. SignalGenerator — ABC extraction + scoring + quality metrics
  3. TradeExecutor — progressive exit with structural SL
  4. PortfolioManager — signal dedup, conflict resolution (future use)
  5. RiskManager — per-pair cap, total DD limit (future use)

This file can run standalone as a backtest, or be imported as modules
for live trading integration.

Key design: RAG is a black-box that takes a query vector and returns
neighbor statistics. Strategy doesn't care if it's KDTree, FAISS, or API.

Cost model is NOT part of strategy — each instrument learns its own
minimum profitable cycle as an operational parameter.
"""
import sys, os, csv, numpy as np, pickle, json, time
from collections import defaultdict
from abc import ABC, abstractmethod
from multiprocessing import Pool

DATA_DIR = "/home/ubuntu/DataBase/base_kline"
ZIG_DIR = "/home/ubuntu/database2/build_zig_all/cl128"
FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618, 2.000, 2.618]


def fib_distance(ratio):
    return min(abs(ratio - f) for f in FIB_LEVELS)


# =============================================================================
# 1. RAG PROVIDER (Abstract Interface)
# =============================================================================

class RAGProvider(ABC):
    """Abstract interface for RAG nearest-neighbor retrieval."""

    @abstractmethod
    def build(self, vectors, metadata):
        """Build index from training vectors + metadata."""
        pass

    @abstractmethod
    def query(self, vectors, k=50):
        """
        Query k nearest neighbors for each vector.
        Returns: (distances, indices) arrays.
        """
        pass

    @abstractmethod
    def get_metadata(self, indices):
        """Retrieve metadata for given indices."""
        pass

    def query_with_stats(self, vectors, k=50):
        """
        Query and return aggregated statistics per query.
        Returns list of dicts: {dir_prob, ca_median, ca_p25, ca_p75, dist_nearest}
        Vectorized for performance on large query sets.
        """
        distances, indices = self.query(vectors, k)
        n_queries = len(vectors)

        # Pre-extract metadata arrays for vectorized lookup
        if hasattr(self, '_c_continues_arr') and self._c_continues_arr is not None:
            c_cont = self._c_continues_arr
            ca_amp = self._ca_amp_arr
        else:
            # Build lookup arrays on first call
            n_meta = len(self.metadata_list) if hasattr(self, 'metadata_list') and self.metadata_list else 0
            if n_meta > 0:
                c_cont = np.array([m["c_continues"] for m in self.metadata_list], dtype=np.float64)
                ca_amp = np.array([m["ca_amp_r"] for m in self.metadata_list], dtype=np.float64)
                self._c_continues_arr = c_cont
                self._ca_amp_arr = ca_amp
            else:
                # Fallback to slow path
                return self._query_with_stats_slow(vectors, k)

        # Vectorized: lookup all neighbors at once
        neighbor_c_cont = c_cont[indices]  # (n_queries, k)
        dir_probs = neighbor_c_cont.mean(axis=1)  # (n_queries,)

        neighbor_ca = ca_amp[indices]  # (n_queries, k)
        ca_medians = np.median(neighbor_ca, axis=1)
        ca_p25 = np.percentile(neighbor_ca, 25, axis=1)
        ca_p75 = np.percentile(neighbor_ca, 75, axis=1)

        dist_nearest = distances[:, 0] if distances.ndim > 1 else distances

        results = []
        for qi in range(n_queries):
            results.append({
                "dir_prob": float(dir_probs[qi]),
                "ca_median": float(ca_medians[qi]),
                "ca_p25": float(ca_p25[qi]),
                "ca_p75": float(ca_p75[qi]),
                "dist_nearest": float(dist_nearest[qi]),
            })
        return results

    def _query_with_stats_slow(self, vectors, k=50):
        """Fallback slow path for query_with_stats."""
        distances, indices = self.query(vectors, k)
        results = []
        for qi in range(len(vectors)):
            ni = indices[qi]
            meta = [self.get_metadata(int(idx)) for idx in ni]
            dir_prob = np.mean([m["c_continues"] for m in meta])
            ca_vals = [m["ca_amp_r"] for m in meta]
            results.append({
                "dir_prob": dir_prob,
                "ca_median": np.median(ca_vals),
                "ca_p25": np.percentile(ca_vals, 25),
                "ca_p75": np.percentile(ca_vals, 75),
                "dist_nearest": distances[qi, 0] if distances.ndim > 1 else distances[qi],
            })
        return results


class KDTreeRAG(RAGProvider):
    """scipy KDTree backend — good for <50M vectors, exact search."""

    def __init__(self):
        self.tree = None
        self.metadata_list = None

    def build(self, vectors, metadata):
        from scipy.spatial import KDTree
        self.tree = KDTree(vectors)
        self.metadata_list = metadata
        print(f"  KDTreeRAG built: {len(vectors):,} vectors, {vectors.shape[1]}D")

    def query(self, vectors, k=50):
        k = min(k, len(self.metadata_list) - 1)
        return self.tree.query(vectors, k=k)

    def get_metadata(self, idx):
        return self.metadata_list[idx]


class FAISSPlaceholderRAG(RAGProvider):
    """
    Placeholder for your FAISS backend.
    Replace build/query with your FAISS index when ready.
    Expected metadata per entry: {c_continues: int, ca_amp_r: float}
    """

    def __init__(self, index_path=None):
        self.index_path = index_path
        self.index = None
        self.metadata_list = None

    def build(self, vectors, metadata):
        # TODO: Replace with FAISS IVF+PQ build
        # import faiss
        # quantizer = faiss.IndexFlatL2(vectors.shape[1])
        # self.index = faiss.IndexIVFPQ(quantizer, vectors.shape[1], 1024, 8, 8)
        # self.index.train(vectors)
        # self.index.add(vectors)
        raise NotImplementedError("Connect your FAISS index here")

    def query(self, vectors, k=50):
        raise NotImplementedError("Connect your FAISS query here")

    def get_metadata(self, idx):
        return self.metadata_list[idx]


# =============================================================================
# 2. COST MODEL
# =============================================================================

class CostModel:
    """
    Per-pair trading cost model.
    Spread in pips, commission per lot, estimated slippage.
    Converts to percentage of price for comparison with signal amplitude.
    """

    # Typical spreads in pips for major/minor/exotic (H1 execution)
    SPREAD_TABLE = {
        # Majors: 1-2 pips
        "EURUSD": 1.2, "GBPUSD": 1.5, "USDJPY": 1.3, "USDCHF": 1.5,
        "AUDUSD": 1.5, "NZDUSD": 1.8, "USDCAD": 1.8,
        # Crosses: 2-4 pips
        "EURGBP": 1.8, "EURJPY": 2.0, "GBPJPY": 2.5, "EURAUD": 2.5,
        "GBPAUD": 3.0, "EURNZD": 3.0, "GBPNZD": 3.5, "EURCAD": 2.5,
        "GBPCAD": 3.0, "AUDNZD": 2.5, "AUDCAD": 2.5, "AUDJPY": 2.0,
        "NZDJPY": 2.5, "CADJPY": 2.5, "CHFJPY": 2.5, "CADCHF": 2.5,
        "AUDCHF": 2.5, "NZDCHF": 3.0, "NZDCAD": 3.0, "GBPCHF": 3.0,
        "EURCHF": 2.0,
        # Exotics: 5-20 pips
        "USDMXN": 50, "USDZAR": 80, "USDTRY": 80, "EURTRY": 80,
        "USDNOK": 30, "USDSEK": 30, "USDSGD": 3.0, "USDHKD": 5.0,
        "USDCNH": 20, "USDRMB": 20, "EURHKD": 15, "EURNOK": 30,
        "EURSEK": 30, "EURPLN": 20, "USDPLN": 20, "GBPNOK": 35,
        "USDHUF": 20, "EURCNH": 25,
        # Metals
        "XAUUSD": 30, "XAGUSD": 2.0,  # in cents for XAG, pips for XAU
    }

    # Pip value as fraction of price (approximate)
    PIP_VALUE = {
        # JPY pairs: 1 pip = 0.01
        "USDJPY": 0.01, "EURJPY": 0.01, "GBPJPY": 0.01, "AUDJPY": 0.01,
        "NZDJPY": 0.01, "CADJPY": 0.01, "CHFJPY": 0.01,
        # Most others: 1 pip = 0.0001
        "_default": 0.0001,
        # Special
        "XAUUSD": 0.01,  # $0.01 per pip on gold
        "XAGUSD": 0.001,
        "USDMXN": 0.0001, "USDZAR": 0.0001, "USDTRY": 0.0001,
        "USDNOK": 0.0001, "USDSEK": 0.0001,
    }

    def __init__(self, commission_pct=0.001):
        """commission_pct: round-trip commission as % of notional (e.g., 0.001 = 0.1bp)"""
        self.commission_pct = commission_pct

    def get_cost_pct(self, pair, price=1.0):
        """
        Total round-trip cost as percentage of price.
        = spread_pct + slippage_pct + commission_pct
        """
        spread_pips = self.SPREAD_TABLE.get(pair, 5.0)
        pip_val = self.PIP_VALUE.get(pair, self.PIP_VALUE["_default"])

        # Convert spread from pips to price fraction
        spread_price = spread_pips * pip_val
        spread_pct = (spread_price / price) * 100.0 if price > 0 else 0.01

        # Slippage: assume 50% of spread
        slippage_pct = spread_pct * 0.5

        # Total round-trip cost
        total = spread_pct + slippage_pct + self.commission_pct
        return total

    def is_tradeable(self, pair, c_amp_pct, price=1.0, min_ratio=3.0):
        """
        Is the signal large enough relative to costs?
        c_amp_pct: expected C amplitude as % of price.
        min_ratio: minimum C_amplitude / cost ratio.
        """
        cost = self.get_cost_pct(pair, price)
        return c_amp_pct >= cost * min_ratio


# =============================================================================
# 3. QUALITY METRICS
# =============================================================================

def compute_quality(a_amp, b_amp, c_amp, a_dur, b_dur, c_dur,
                    a_slope, b_slope, lv, bar_idx, closes, n_bars):
    """Compute turning point quality metrics. Returns dict."""
    ca_ratio = c_amp / a_amp if a_amp > 1e-8 else 1.0
    fib_dist = fib_distance(ca_ratio)
    fib_score = max(0, 1.0 - fib_dist / 0.15)
    time_sym = min(a_dur, c_dur) / max(a_dur, c_dur) if max(a_dur, c_dur) > 0 else 0
    ba_ratio = b_amp / a_amp if a_amp > 1e-8 else 1.0
    if ba_ratio < 0.15: b_quality = ba_ratio / 0.15
    elif ba_ratio <= 1.0: b_quality = 1.0
    elif ba_ratio <= 2.0: b_quality = max(0, 1.0 - (ba_ratio - 1.0))
    else: b_quality = 0.0
    completion = fib_score * 0.4 + time_sym * 0.3 + b_quality * 0.3

    b_complexity = min(lv / 7.0, 1.0)
    dur_ratio_raw = b_dur / max(a_dur, 1)
    if dur_ratio_raw > 3.0: b_dur_score = 1.0
    elif dur_ratio_raw > 1.5: b_dur_score = 0.7
    elif dur_ratio_raw > 0.5: b_dur_score = 0.3
    else: b_dur_score = 0.0

    exhaustion = 0.5
    b_start_bar = bar_idx - b_dur
    b_mid_bar = bar_idx - b_dur // 2
    if b_start_bar >= 0 and b_mid_bar >= 0 and bar_idx < n_bars and b_dur >= 4:
        p_start = closes[max(0, b_start_bar)]
        p_mid = closes[max(0, min(b_mid_bar, n_bars-1))]
        p_end = closes[min(bar_idx, n_bars-1)]
        half1 = abs(p_mid - p_start)
        half2 = abs(p_end - p_mid)
        total = half1 + half2
        if total > 1e-8:
            exhaustion = half1 / total

    arrival = b_complexity * 0.3 + b_dur_score * 0.4 + (1.0 - exhaustion) * 0.3

    return {
        "completion": completion, "arrival": arrival, "exhaustion": exhaustion,
        "time_sym": time_sym, "fib_score": fib_score, "b_quality": b_quality,
    }


def compute_quality_multiplier(completion, time_sym, arrival, exhaustion):
    """
    Convert quality metrics to sizing multiplier.
    Based on empirical findings (F13a-e).
    """
    # Low completion = better (Q1 PF=2.43 vs Q5 PF=1.61)
    compl_mult = 1.0 + max(0, 0.6 - completion) * 1.5

    # High time symmetry = better (Q4 PF=2.25 vs Q1 PF=1.28)
    tsym_mult = 0.7 + time_sym * 0.6

    # Trend arrival = better
    arr_mult = 0.8 + arrival * 0.5

    # Exhaustion U-shape: extremes better
    exh_deviation = abs(exhaustion - 0.5)
    exh_mult = 0.9 + exh_deviation * 0.4

    return compl_mult * tsym_mult * arr_mult * exh_mult


# =============================================================================
# 4. TRADE EXECUTOR
# =============================================================================

def run_progressive_trade(eb, direction, tp_pct, sl_pct, ep, highs, lows, closes):
    """
    Progressive partial exit with structural SL.
    Returns weighted R-multiple, or None.
    """
    n_bars = len(closes)
    if tp_pct < 0.005 or sl_pct < 0.005: return None
    tp_d = ep * tp_pct / 100.0
    sl_d = ep * sl_pct / 100.0
    if tp_d <= 0 or sl_d <= 0: return None

    if direction == 1:
        sl_price = ep - sl_d
    else:
        sl_price = ep + sl_d

    targets = [(0.382, 0.25), (0.618, 0.25), (1.000, 0.25)]
    remaining_weight = 0.25

    max_hold = 200
    mf = 0.0
    end_bar = min(eb + max_hold, n_bars - 1)
    total_pnl = 0.0
    total_weight = 0.0
    targets_hit = 0

    for bar in range(eb + 1, end_bar + 1):
        h = highs[bar]; l = lows[bar]
        fav = (h - ep) if direction == 1 else (ep - l)
        if fav > mf: mf = fav

        if direction == 1:
            if l <= sl_price:
                rem_w = sum(t[1] for t in targets[targets_hit:]) + remaining_weight
                total_pnl += ((sl_price - ep) / sl_d) * rem_w
                total_weight += rem_w
                break
        else:
            if h >= sl_price:
                rem_w = sum(t[1] for t in targets[targets_hit:]) + remaining_weight
                total_pnl += ((ep - sl_price) / sl_d) * rem_w
                total_weight += rem_w
                break

        while targets_hit < len(targets):
            t_level, t_weight = targets[targets_hit]
            t_offset = tp_d * t_level
            hit = (h >= ep + t_offset) if direction == 1 else (l <= ep - t_offset)
            if hit:
                total_pnl += (t_offset / sl_d) * t_weight
                total_weight += t_weight
                targets_hit += 1
            else:
                break

        if targets_hit >= 1:
            if direction == 1: sl_price = max(sl_price, ep)
            else: sl_price = min(sl_price, ep)
        if targets_hit >= 2 and mf > 0:
            trail = mf * 0.618
            if direction == 1: sl_price = max(sl_price, ep + trail)
            else: sl_price = min(sl_price, ep - trail)
        if targets_hit >= 3 and mf > 0:
            trail = mf * 0.764
            if direction == 1: sl_price = max(sl_price, ep + trail)
            else: sl_price = min(sl_price, ep - trail)
    else:
        rem_w = sum(t[1] for t in targets[targets_hit:]) + remaining_weight
        pnl = (closes[end_bar] - ep) if direction == 1 else (ep - closes[end_bar])
        total_pnl += (pnl / sl_d) * rem_w
        total_weight += rem_w

    if total_weight <= 0: return None
    return total_pnl / total_weight


# =============================================================================
# 5. PORTFOLIO MANAGER — Signal Dedup & Conflict Resolution
# =============================================================================

class PortfolioManager:
    """
    Handles signal conflicts when multiple signals fire simultaneously.
    
    Rules:
    1. No overlapping trades on same pair (wait for current to close)
    2. Max N concurrent positions across portfolio
    3. Correlated pairs share allocation bucket (e.g., EURUSD and EURGBP)
    4. Priority: higher quality_mult × score → gets capital first
    """

    # Correlation groups — pairs that tend to move together
    CORR_GROUPS = {
        "EUR_USD": ["EURUSD", "EURGBP", "EURJPY", "EURAUD", "EURNZD", "EURCAD", "EURCHF"],
        "GBP": ["GBPUSD", "GBPJPY", "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF"],
        "JPY": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"],
        "AUD_NZD": ["AUDUSD", "NZDUSD", "AUDNZD", "AUDCAD", "AUDCHF", "NZDCAD", "NZDCHF"],
        "CHF": ["USDCHF", "EURCHF", "GBPCHF", "AUDCHF", "NZDCHF", "CADCHF", "CHFJPY"],
        "METALS": ["XAUUSD", "XAGUSD"],
        "EXOTIC": ["USDMXN", "USDZAR", "USDTRY", "EURTRY", "USDNOK", "USDSEK",
                    "USDSGD", "USDHKD", "USDCNH", "USDRMB", "EURHKD", "EURNOK",
                    "EURSEK", "EURPLN", "USDPLN", "GBPNOK", "USDHUF", "EURCNH"],
    }

    def __init__(self, max_concurrent=20, max_per_group=5, max_per_pair=1):
        self.max_concurrent = max_concurrent
        self.max_per_group = max_per_group
        self.max_per_pair = max_per_pair
        self._pair_to_group = {}
        for group, pairs in self.CORR_GROUPS.items():
            for pair in pairs:
                self._pair_to_group[pair] = group

    def resolve_conflicts(self, signals):
        """
        Given a list of signals at the same bar, select which to trade.
        Signals sorted by priority (score × quality_mult, descending).
        Returns filtered list respecting all limits.
        """
        signals = sorted(signals, key=lambda s: s.get("priority", 0), reverse=True)

        selected = []
        pair_count = defaultdict(int)
        group_count = defaultdict(int)
        total = 0

        for sig in signals:
            pair = sig["pair"]
            group = self._pair_to_group.get(pair, pair)

            if total >= self.max_concurrent:
                break
            if pair_count[pair] >= self.max_per_pair:
                continue
            if group_count[group] >= self.max_per_group:
                continue

            selected.append(sig)
            pair_count[pair] += 1
            group_count[group] += 1
            total += 1

        return selected


# =============================================================================
# 6. RISK MANAGER
# =============================================================================

class RiskManager:
    """
    Portfolio-level risk controls.
    """

    def __init__(self, max_dd_pct=20.0, risk_per_r=0.15, dd_scale_factor=0.5):
        """
        max_dd_pct: halt trading if portfolio DD exceeds this %
        risk_per_r: % of capital risked per 1R (e.g., 0.15 = 0.15% per R)
        dd_scale_factor: reduce size by this factor when in drawdown > 50% of max_dd
        """
        self.max_dd_pct = max_dd_pct
        self.risk_per_r = risk_per_r
        self.dd_scale_factor = dd_scale_factor
        self.peak_equity = 100.0
        self.equity = 100.0
        self.halted = False

    def update_equity(self, pnl_r, size):
        """Update equity after a trade. pnl_r is R-multiple, size is position multiplier."""
        pnl_pct = pnl_r * size * self.risk_per_r
        self.equity += pnl_pct
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = (self.peak_equity - self.equity) / self.peak_equity * 100
        if dd >= self.max_dd_pct:
            self.halted = True

    def get_size_adjustment(self):
        """Reduce position size during drawdown."""
        if self.halted:
            return 0.0
        dd = (self.peak_equity - self.equity) / self.peak_equity * 100
        if dd >= self.max_dd_pct * 0.5:
            return self.dd_scale_factor
        return 1.0


# =============================================================================
# 7. DATA LOADING (same as V2)
# =============================================================================

def load_ohlcv(pair, tf):
    fpath = os.path.join(DATA_DIR, f"{pair}_{tf}.csv")
    if not os.path.exists(fpath):
        return None, None
    dates, highs, lows, closes = [], [], [], []
    with open(fpath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 6: continue
            dates.append(row[0])
            highs.append(float(row[3]))
            lows.append(float(row[4]))
            closes.append(float(row[5]))
    return dates, (np.array(highs), np.array(lows), np.array(closes))


def load_zigzag_topo(pair, tf):
    fpath = os.path.join(ZIG_DIR, tf, f"{pair}_{tf}.npz")
    if not os.path.exists(fpath):
        return None, None
    d = np.load(fpath, allow_pickle=True)
    feats = d["features"]
    edges = pickle.loads(d["edges_bytes"].tobytes())
    return feats, edges


# =============================================================================
# 8. SIGNAL EXTRACTION (per-pair worker)
# =============================================================================

DELAY = 5


def extract_pair_data(args):
    """Worker: extract RAG chains + trade signals for one pair."""
    pair, tf = args
    dates, price_data = load_ohlcv(pair, tf)
    if dates is None: return pair, [], []
    highs, lows, closes = price_data
    n_bars = len(closes)
    feats, edges = load_zigzag_topo(pair, tf)
    if feats is None: return pair, [], []

    decay = np.exp(-0.08 * DELAY)

    win_bar_to_node = defaultdict(dict)
    for i in range(len(feats)):
        win = int(feats[i, 1])
        rel_idx = int(feats[i, 2])
        ml = int(feats[i, 5])
        if rel_idx not in win_bar_to_node[win] or ml > feats[win_bar_to_node[win][rel_idx], 5]:
            win_bar_to_node[win][rel_idx] = i

    chains = []
    node_levels = defaultdict(list)

    for i in range(len(feats)):
        bar_idx = int(feats[i, 0])
        win_id = int(feats[i, 1])
        rel_idx = int(feats[i, 2])
        max_level = int(feats[i, 5])
        if bar_idx < 50 or bar_idx >= n_bars - 300: continue

        in_edges_i, out_edges_i = edges[i]
        if not out_edges_i: continue

        in_by_lv = defaultdict(list)
        out_by_lv = defaultdict(list)
        for e in in_edges_i:
            if e["duration"] > 0: in_by_lv[e["level"]].append(e)
        for e in out_edges_i:
            if e["duration"] > 0: out_by_lv[e["level"]].append(e)

        for lv in range(max_level + 1):
            b_list = [e for e in in_by_lv.get(lv, []) if e["end_idx"] == rel_idx]
            c_list = [e for e in out_by_lv.get(lv, []) if e["start_idx"] == rel_idx]
            if not c_list or not b_list: continue
            c_edge = c_list[0]; b_edge = b_list[0]

            b_src_rel = b_edge["start_idx"]
            src_idx = win_bar_to_node.get(win_id, {}).get(b_src_rel, None)
            if src_idx is None: continue
            src_in, _ = edges[src_idx]
            a_cands = [e for e in src_in if e["level"] == lv and e["end_idx"] == b_src_rel and e["duration"] > 0]
            if not a_cands: continue
            a_edge = a_cands[0]

            a_amp = abs(a_edge["amplitude_pct"])
            b_amp = abs(b_edge["amplitude_pct"])
            c_amp = abs(c_edge["amplitude_pct"])
            a_dur = max(a_edge["duration"], 1)
            b_dur = max(b_edge["duration"], 1)
            c_dur = max(c_edge["duration"], 1)
            a_mod = a_edge["modulus"]
            b_mod = b_edge["modulus"]
            if a_amp < 1e-6 or a_mod < 1e-6: continue

            a_slope = a_amp / a_dur
            b_slope = b_amp / b_dur
            amp_r = b_amp / a_amp
            mod_r = b_mod / a_mod
            c_continues = 1 if (a_edge["direction"] == c_edge["direction"]) else 0
            ca_amp_r = c_amp / a_amp

            qual = compute_quality(a_amp, b_amp, c_amp, a_dur, b_dur, c_dur,
                                   a_slope, b_slope, lv, bar_idx, closes, n_bars)

            ab_vec = np.array([
                np.log1p(amp_r), np.log1p(b_dur / a_dur),
                np.log1p(b_slope / a_slope) if a_slope > 1e-10 else 0,
                np.log1p(mod_r) if a_mod > 1e-6 else 0,
                qual["completion"], qual["arrival"], qual["exhaustion"],
            ])

            year = int(dates[min(bar_idx, len(dates)-1)][:4])

            chains.append({
                "ab_vec": ab_vec,
                "c_continues": c_continues,
                "ca_amp_r": ca_amp_r,
                "year": year,
            })

            score = 0.5 + lv * 0.5
            score += max(0, 2.0 * (1.0 - fib_distance(amp_r) / 0.10))
            score += max(0, 1.0 * (1.0 - fib_distance(mod_r) / 0.10))
            if amp_r < 0.382: score += 1.5
            elif amp_r < 0.618: score += 0.8
            elif amp_r >= 1.0: score -= 0.5
            score += 1.0 if c_continues else -0.5

            node_levels[bar_idx].append({
                "lv": lv, "score": score,
                "c_amp": c_edge["amplitude_pct"], "c_dir": c_edge["direction"],
                "a_amp": a_amp, "b_amp": b_amp, "c_amp_abs": c_amp,
                "a_dur": a_dur, "b_dur": b_dur,
                "ab_vec": ab_vec,
                "completion": qual["completion"], "arrival": qual["arrival"],
                "exhaustion": qual["exhaustion"], "time_sym": qual["time_sym"],
                "c_continues": c_continues,
            })

    # Build signals
    signals = []
    for bar_idx, levels in node_levels.items():
        if not levels: continue
        n_lv = len(levels)
        total_score = sum(d["score"] for d in levels)
        dirs = [d["c_dir"] for d in levels]
        consensus = abs(sum(dirs)) / len(dirs)
        if consensus >= 0.9: total_score += n_lv * 0.5
        elif consensus < 0.5: total_score -= n_lv * 0.3
        if total_score < 10 or total_score >= 30: continue

        direction_sum = sum(d["c_dir"] * d["score"] for d in levels)
        direction = 1 if direction_sum > 0 else -1

        ws = [d["score"] for d in levels if d["score"] > 0]
        w_total = sum(ws)
        if w_total <= 0: continue

        def wmean(key):
            return sum(d["score"] * d[key] for d in levels if d["score"] > 0) / w_total

        v_amp = abs(wmean("c_amp"))
        v_b_amp = wmean("b_amp")
        c_amp_abs = wmean("c_amp_abs")

        if c_amp_abs < 0.20: continue

        ab_vec_agg = np.zeros(7)
        for d in levels:
            if d["score"] > 0:
                ab_vec_agg += d["score"] * d["ab_vec"]
        ab_vec_agg /= w_total

        completion = wmean("completion")
        arrival = wmean("arrival")
        exhaustion = wmean("exhaustion")
        time_sym = wmean("time_sym")
        quality_mult = compute_quality_multiplier(completion, time_sym, arrival, exhaustion)

        eb = bar_idx + DELAY
        if eb >= n_bars - 200: continue
        ep = closes[eb]
        year = int(dates[min(bar_idx, len(dates)-1)][:4])

        tp_pct = v_amp * 1.0 * decay
        sl_pct = v_b_amp * 0.8 * decay

        pnl_r = run_progressive_trade(eb, direction, tp_pct, sl_pct, ep, highs, lows, closes)
        if pnl_r is None: continue

        score_rounded = round(total_score, 2)
        if score_rounded < 15: base_size = 1.0
        elif score_rounded < 20: base_size = 1.5
        else: base_size = 2.0

        priority = total_score * quality_mult

        signals.append({
            "bar_idx": bar_idx, "year": year, "pair": pair,
            "direction": direction, "score": round(total_score, 2),
            "n_lv": n_lv, "pnl_r": round(pnl_r, 4),
            "ab_vec": ab_vec_agg,
            "quality_mult": round(quality_mult, 4),
            "base_size": base_size,
            "priority": round(priority, 2),
            "c_amp": round(c_amp_abs, 4),
        })

    print(f"  {pair}_{tf}: {len(chains):,} chains, {len(signals):,} signals")
    return pair, chains, signals


# =============================================================================
# 9. MAIN BACKTEST
# =============================================================================

def pf_wr(pnl_list):
    if not pnl_list: return 0, 0, 0, 0, 0
    w = [x for x in pnl_list if x > 0]
    lo = [x for x in pnl_list if x <= 0]
    pf = abs(sum(w) / sum(lo)) if lo and sum(lo) != 0 else 999
    return len(pnl_list), len(w)/len(pnl_list)*100, np.mean(pnl_list), pf, sum(pnl_list)


def max_drawdown_r(pnl_list):
    """Max drawdown in R-multiple terms (cumulative)."""
    if not pnl_list: return 0
    cum = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return abs(dd.min())


def main():
    print("=" * 80)
    print("  TIANSHU STRATEGY ENGINE V3 — COMPREHENSIVE BACKTEST")
    print("  48 pairs × H1 | IS: <=2018 | OOS: >2018")
    print("=" * 80)

    tf = "H1"
    tf_path = os.path.join(ZIG_DIR, tf)
    tasks = []
    for fname in sorted(os.listdir(tf_path)):
        if not fname.endswith(".npz"): continue
        pair = fname.replace(f"_{tf}.npz", "")
        if os.path.exists(os.path.join(DATA_DIR, f"{pair}_{tf}.csv")):
            tasks.append((pair, tf))
    print(f"  Pairs: {len(tasks)}")

    all_chains = []
    all_signals = []

    with Pool(min(48, len(tasks))) as pool:
        for pair, chains, signals in pool.imap_unordered(extract_pair_data, tasks):
            all_chains.extend(chains)
            all_signals.extend(signals)

    print(f"\n  Total: {len(all_chains):,} RAG chains, {len(all_signals):,} signals")

    is_chains = [c for c in all_chains if 0 < c["year"] <= 2018]
    is_signals = [s for s in all_signals if 0 < s["year"] <= 2018]
    oos_signals = [s for s in all_signals if s["year"] > 2018]
    print(f"  RAG IS: {len(is_chains):,}")
    print(f"  Signals: IS={len(is_signals):,} OOS={len(oos_signals):,}")

    # Build RAG
    rag = KDTreeRAG()
    is_vecs = np.array([c["ab_vec"] for c in is_chains])
    rag.build(is_vecs, is_chains)

    K = 50

    # Strategy variants — exactly reproducing V2's A→E layering
    # A = Baseline: flat size, no RAG
    # B = +Quality sizing
    # C = +RAG filter (>0.55)
    # D = +RAG confidence sizing
    # E = Final: RAG>0.60 + score sizing + quality + RAG bonus
    results = {s: {"IS": [], "OOS": []} for s in [
        "A_base", "B_quality", "C_rag_filt", "D_rag_conf", "E_final",
    ]}

    for split_name, sigs in [("IS", is_signals), ("OOS", oos_signals)]:
        if not sigs: continue
        sig_vecs = np.array([s["ab_vec"] for s in sigs])
        print(f"\n  RAG querying {split_name}: {len(sigs):,} signals...")
        rag_stats = rag.query_with_stats(sig_vecs, k=K)

        for qi, sig in enumerate(sigs):
            pnl_r = sig["pnl_r"]
            score = sig["score"]
            quality_mult = sig["quality_mult"]
            base_size = sig["base_size"]
            rs = rag_stats[qi]
            rag_dir_prob = rs["dir_prob"]

            # A: Baseline (flat size=1.0, no quality, no RAG)
            results["A_base"][split_name].append({
                "pnl_r": pnl_r, "size": 1.0, "pnl_sized": pnl_r,
                "year": sig["year"], "pair": sig["pair"], "score": score,
            })

            # B: Quality sizing only
            size_b = quality_mult
            results["B_quality"][split_name].append({
                "pnl_r": pnl_r, "size": size_b, "pnl_sized": pnl_r * size_b,
                "year": sig["year"], "pair": sig["pair"], "score": score,
            })

            # C: RAG filter (>0.55) + flat size
            if rag_dir_prob > 0.55:
                results["C_rag_filt"][split_name].append({
                    "pnl_r": pnl_r, "size": 1.0, "pnl_sized": pnl_r,
                    "year": sig["year"], "pair": sig["pair"], "score": score,
                })

            # D: RAG confidence + quality sizing
            if rag_dir_prob > 0.55:
                rag_bonus = 1.0 + max(0, rag_dir_prob - 0.55) * 2.0
                size_d = quality_mult * rag_bonus
                results["D_rag_conf"][split_name].append({
                    "pnl_r": pnl_r, "size": size_d, "pnl_sized": pnl_r * size_d,
                    "year": sig["year"], "pair": sig["pair"], "score": score,
                })

            # E: Final (tighter RAG>0.60 + score-based + quality + RAG confidence)
            if rag_dir_prob > 0.60:
                rag_bonus = 1.0 + max(0, rag_dir_prob - 0.60) * 2.5
                size_e = base_size * quality_mult * rag_bonus
                results["E_final"][split_name].append({
                    "pnl_r": pnl_r, "size": size_e, "pnl_sized": pnl_r * size_e,
                    "year": sig["year"], "pair": sig["pair"], "score": score,
                })

    # =============================================
    # REPORTS (matching V2 format)
    # =============================================
    print(f"\n{'='*80}")
    print("  REPORT 1: STRATEGY COMPARISON")
    print(f"{'='*80}")

    for strat in results:
        print(f"\n  === {strat} ===")
        for split in ["IS", "OOS"]:
            data = results[strat][split]
            if not data: continue
            pnl = [d["pnl_sized"] for d in data]
            pnl_r = [d["pnl_r"] for d in data]
            n, wr, ar, pf, sr = pf_wr(pnl)
            _, wr_r, _, pf_r, _ = pf_wr(pnl_r)
            avg_size = np.mean([d["size"] for d in data])
            md = max_drawdown_r(pnl)
            print(f"    {split:>3}: n={n:>7,}  WR_r={wr_r:.1f}%  PF_r={pf_r:.2f}  "
                  f"PF_sized={pf:.2f}  avgR={ar:.4f}  sumR={sr:>10,.0f}  "
                  f"avg_sz={avg_size:.2f}  maxDD={md:,.0f}")

    # Walk-forward by year (E_final)
    print(f"\n{'='*80}")
    print("  REPORT 2: WALK-FORWARD BY YEAR (E_final)")
    print(f"{'='*80}")
    all_e = results["E_final"]["IS"] + results["E_final"]["OOS"]
    if all_e:
        years = sorted(set(d["year"] for d in all_e))
        for year in years:
            yr_data = [d for d in all_e if d["year"] == year]
            if not yr_data: continue
            pnl = [d["pnl_sized"] for d in yr_data]
            pnl_r = [d["pnl_r"] for d in yr_data]
            n, wr, ar, pf, sr = pf_wr(pnl)
            _, wr_r, _, pf_r, _ = pf_wr(pnl_r)
            md = max_drawdown_r(pnl)
            oos_tag = " [OOS]" if year > 2018 else ""
            print(f"    {year}{oos_tag:>6}: n={n:>6,}  WR={wr_r:.1f}%  PF_r={pf_r:.2f}  "
                  f"PF_sz={pf:.2f}  sumR={sr:>8,.0f}  maxDD={md:>6,.0f}")

    # Per-pair summary (E_final, OOS)
    print(f"\n{'='*80}")
    print("  REPORT 3: PER-PAIR SUMMARY (E_final, OOS)")
    print(f"{'='*80}")
    oos_e = results["E_final"]["OOS"]
    if oos_e:
        pairs_oos = sorted(set(d["pair"] for d in oos_e))
        pair_stats = []
        for pair in pairs_oos:
            pdata = [d for d in oos_e if d["pair"] == pair]
            pnl = [d["pnl_sized"] for d in pdata]
            pnl_r = [d["pnl_r"] for d in pdata]
            n, wr, ar, pf, sr = pf_wr(pnl)
            _, wr_r, _, pf_r, _ = pf_wr(pnl_r)
            pair_stats.append((pair, n, wr_r, pf_r, pf, sr))
        pair_stats.sort(key=lambda x: -x[5])
        profitable = sum(1 for p in pair_stats if p[5] > 0)
        print(f"  Profitable pairs: {profitable}/{len(pair_stats)}")
        for pair, n, wr, pfr, pfs, sr in pair_stats:
            print(f"    {pair:>8}: n={n:>5,}  WR={wr:.1f}%  PF_r={pfr:.2f}  PF_sz={pfs:.2f}  sumR={sr:>7,.0f}")

    # Score tier analysis (E_final)
    print(f"\n{'='*80}")
    print("  REPORT 4: SCORE TIER (E_final)")
    print(f"{'='*80}")
    if oos_e:
        for slo, shi in [(10, 13), (13, 15), (15, 18), (18, 20), (20, 25), (25, 30)]:
            chunk = [d for d in oos_e if slo <= d["score"] < shi]
            if not chunk: continue
            pnl = [d["pnl_sized"] for d in chunk]
            pnl_r = [d["pnl_r"] for d in chunk]
            n, wr, ar, pf, sr = pf_wr(pnl)
            _, wr_r, _, pf_r, _ = pf_wr(pnl_r)
            print(f"    score [{slo:>2},{shi:>2}): n={n:>6,}  WR_r={wr_r:.1f}%  "
                  f"PF_r={pf_r:.2f}  PF_sz={pf:.2f}  sumR={sr:>8,.0f}")

    # Quality multiplier tiers (E_final, OOS)
    print(f"\n{'='*80}")
    print("  REPORT 5: QUALITY MULTIPLIER TIERS (E_final, OOS)")
    print(f"{'='*80}")
    if oos_e:
        sz_sorted = sorted(oos_e, key=lambda x: x["size"])
        qs = max(1, len(sz_sorted) // 5)
        for qi in range(5):
            chunk = sz_sorted[qi*qs:(qi+1)*qs] if qi < 4 else sz_sorted[qi*qs:]
            pnl = [d["pnl_sized"] for d in chunk]
            pnl_r = [d["pnl_r"] for d in chunk]
            n, wr, ar, pf, sr = pf_wr(pnl)
            _, wr_r, _, pf_r, _ = pf_wr(pnl_r)
            sz_lo = chunk[0]["size"]; sz_hi = chunk[-1]["size"]
            print(f"    Q{qi+1} [{sz_lo:.2f},{sz_hi:.2f}]: n={n:>6,}  "
                  f"WR_r={wr_r:.1f}%  PF_r={pf_r:.2f}  PF_sz={pf:.2f}  sumR={sr:>8,.0f}")

    # Final summary (E_final)
    print(f"\n{'='*80}")
    print("  REPORT 6: FINAL SUMMARY")
    print(f"{'='*80}")
    for split in ["IS", "OOS"]:
        data = results["E_final"][split]
        if not data: continue
        pnl = [d["pnl_sized"] for d in data]
        pnl_r = [d["pnl_r"] for d in data]
        n, wr, ar, pf, sr = pf_wr(pnl)
        _, wr_r, _, pf_r, _ = pf_wr(pnl_r)
        avg_size = np.mean([d["size"] for d in data])
        n_years = len(set(d["year"] for d in data))
        md = max_drawdown_r(pnl)

        print(f"\n  {split}:")
        print(f"    Trades: {n:,} ({n/max(n_years,1):.0f}/yr)")
        print(f"    WR (per R): {wr_r:.1f}%")
        print(f"    PF (per R): {pf_r:.2f}")
        print(f"    PF (sized): {pf:.2f}")
        print(f"    Sum R (sized): {sr:,.0f}")
        print(f"    Avg size: {avg_size:.2f}")
        print(f"    Max DD (R): {md:,.0f}")

    print(f"\n{'='*80}")
    print("  COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
