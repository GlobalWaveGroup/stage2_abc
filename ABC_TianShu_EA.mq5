//+------------------------------------------------------------------+
//|                                              ABC_TianShu_EA.mq5  |
//|                         TianShu ABC Wave Trading System           |
//|                                                                    |
//|  Strategy: Identify A-B-C waves using online zigzag, enter at     |
//|  B-wave completion (trend continuation), with dynamic TP/SL.      |
//|                                                                    |
//|  HOW TO USE:                                                       |
//|  1. Copy this file to: MT5/MQL5/Experts/                          |
//|  2. Compile in MetaEditor (F7)                                     |
//|  3. Open Strategy Tester (Ctrl+R)                                 |
//|  4. Select symbol, period (H1 recommended), date range            |
//|  5. Set "Modeling" to "Every tick" for best accuracy               |
//|  6. Run backtest                                                   |
//+------------------------------------------------------------------+
#property copyright "TianShu"
#property link      ""
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input double   ZZ_Deviation   = 1.0;     // ZigZag deviation (%)
input int      ZZ_ConfirmBars = 5;       // ZigZag confirmation bars
input double   RiskPercent    = 1.0;     // Risk per trade (% of balance)
input int      MagicNumber    = 20260214; // EA magic number
input bool     AllowLong      = true;    // Allow long trades
input bool     AllowShort     = true;    // Allow short trades

//--- Scoring weights (calibrated from backtests)
input double   W_Slope  = 0.40;  // Weight: slope_ratio score
input double   W_Time   = 0.35;  // Weight: time_ratio score
input double   W_Amp    = 0.25;  // Weight: amp_ratio score
input double   MinScore = 0.30;  // Minimum entry score

//--- Internal state
CTrade trade;

// ZigZag state
int    zz_trend;        // +1 tracking highs, -1 tracking lows, 0 init
double zz_ext_price;    // current extreme price
int    zz_ext_bar;      // bar of current extreme
double zz_tent_price;   // tentative pivot price
int    zz_tent_bar;     // tentative pivot bar
int    zz_tent_type;    // 1=High, -1=Low, 0=none
double zz_init_hi, zz_init_lo;
int    zz_init_hi_bar, zz_init_lo_bar;

// Pivot storage (last 4 pivots: indices 0,1,2,3 = oldest to newest)
struct PivotPoint {
    int    bar_idx;
    double price;
    int    direction;  // +1 = high pivot, -1 = low pivot
};
PivotPoint pivots[];
int pivot_count;

// Trade management
bool   in_trade;
int    trade_dir;       // +1=long, -1=short
double trade_entry;
double trade_a_amp;
int    trade_a_bars;
double trade_tp_dist;
double trade_sl_dist;
double trade_initial_sl_dist;
double trade_max_favorable;
bool   trade_hit_be;
double trade_score;
double trade_expected_bars;
int    trade_entry_bar;

//+------------------------------------------------------------------+
//| Expert initialization                                              |
//+------------------------------------------------------------------+
int OnInit()
{
    trade.SetExpertMagicNumber(MagicNumber);
    
    // Initialize zigzag
    zz_trend = 0;
    zz_tent_type = 0;
    zz_init_hi = 0; zz_init_lo = 0;
    zz_init_hi_bar = 0; zz_init_lo_bar = 0;
    
    pivot_count = 0;
    ArrayResize(pivots, 0);
    
    in_trade = false;
    trade_max_favorable = 0;
    trade_hit_be = false;
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Add a confirmed pivot                                              |
//+------------------------------------------------------------------+
void AddPivot(int bar_idx, double price, int direction)
{
    pivot_count++;
    ArrayResize(pivots, pivot_count);
    pivots[pivot_count - 1].bar_idx = bar_idx;
    pivots[pivot_count - 1].price = price;
    pivots[pivot_count - 1].direction = direction;
}

//+------------------------------------------------------------------+
//| Online ZigZag - process one bar                                    |
//| Returns true if a new pivot is confirmed                           |
//+------------------------------------------------------------------+
bool ZZ_ProcessBar(int idx, double high, double low)
{
    double dev = ZZ_Deviation / 100.0;
    
    // Initialization phase
    if(zz_trend == 0)
    {
        if(zz_init_hi == 0 && zz_init_lo == 0)
        {
            zz_init_hi = high; zz_init_hi_bar = idx;
            zz_init_lo = low;  zz_init_lo_bar = idx;
            return false;
        }
        if(high > zz_init_hi) { zz_init_hi = high; zz_init_hi_bar = idx; }
        if(low < zz_init_lo)  { zz_init_lo = low;  zz_init_lo_bar = idx; }
        
        if(zz_init_lo <= 0) return false;
        double rng = (zz_init_hi - zz_init_lo) / zz_init_lo;
        if(rng < dev) return false;
        
        if(zz_init_hi_bar > zz_init_lo_bar)
        {
            AddPivot(zz_init_lo_bar, zz_init_lo, -1);
            zz_trend = 1;
            zz_ext_price = high; zz_ext_bar = idx;
        }
        else
        {
            AddPivot(zz_init_hi_bar, zz_init_hi, 1);
            zz_trend = -1;
            zz_ext_price = low; zz_ext_bar = idx;
        }
        zz_tent_type = 0;
        return false;
    }
    
    // Uptrend: tracking high
    if(zz_trend == 1)
    {
        if(high > zz_ext_price)
        {
            zz_ext_price = high; zz_ext_bar = idx;
            zz_tent_type = 0;
        }
        double drop = (zz_ext_price > 0) ? (zz_ext_price - low) / zz_ext_price : 0;
        if(drop >= dev && zz_tent_type == 0)
        {
            zz_tent_bar = zz_ext_bar;
            zz_tent_price = zz_ext_price;
            zz_tent_type = 1; // High pivot tentative
        }
        if(zz_tent_type == 1)
        {
            if(high > zz_tent_price)
            {
                zz_ext_price = high; zz_ext_bar = idx;
                zz_tent_type = 0;
            }
            else if(idx - zz_tent_bar >= ZZ_ConfirmBars)
            {
                // Confirmed high pivot
                AddPivot(zz_tent_bar, zz_tent_price, +1);
                zz_trend = -1;
                zz_ext_price = low; zz_ext_bar = idx;
                zz_tent_type = 0;
                return true;
            }
        }
        return false;
    }
    
    // Downtrend: tracking low
    if(zz_trend == -1)
    {
        if(low < zz_ext_price)
        {
            zz_ext_price = low; zz_ext_bar = idx;
            zz_tent_type = 0;
        }
        double rise = (zz_ext_price > 0) ? (high - zz_ext_price) / zz_ext_price : 0;
        if(rise >= dev && zz_tent_type == 0)
        {
            zz_tent_bar = zz_ext_bar;
            zz_tent_price = zz_ext_price;
            zz_tent_type = -1; // Low pivot tentative
        }
        if(zz_tent_type == -1)
        {
            if(low < zz_tent_price)
            {
                zz_ext_price = low; zz_ext_bar = idx;
                zz_tent_type = 0;
            }
            else if(idx - zz_tent_bar >= ZZ_ConfirmBars)
            {
                // Confirmed low pivot
                AddPivot(zz_tent_bar, zz_tent_price, -1);
                zz_trend = 1;
                zz_ext_price = high; zz_ext_bar = idx;
                zz_tent_type = 0;
                return true;
            }
        }
        return false;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Compute entry score                                                |
//+------------------------------------------------------------------+
double ComputeScore(double slope_ratio, double time_ratio, double amp_ratio)
{
    // Slope score: lower = better
    double slope_score = MathMin(1.0, MathMax(0.0, 1.0 - slope_ratio * 0.8));
    
    // Time score: higher = better
    double time_score = MathMin(1.0, MathMax(0.0, (MathLog(1.0 + time_ratio) - 0.2) / 2.2));
    
    // Amp score: peak at 0.7-1.0
    double amp_score;
    if(amp_ratio < 0.3)
        amp_score = amp_ratio / 0.3 * 0.4;
    else if(amp_ratio < 0.7)
        amp_score = 0.4 + (amp_ratio - 0.3) / 0.4 * 0.6;
    else if(amp_ratio <= 1.0)
        amp_score = 1.0;
    else if(amp_ratio <= 1.5)
        amp_score = 1.0 - (amp_ratio - 1.0) / 0.5 * 0.3;
    else
        amp_score = MathMax(0.3, 0.7 - (amp_ratio - 1.5) * 0.2);
    
    double score = W_Slope * slope_score + W_Time * time_score + W_Amp * amp_score;
    return MathMin(1.0, MathMax(0.0, score));
}

//+------------------------------------------------------------------+
//| Score to trade parameters                                          |
//+------------------------------------------------------------------+
bool ScoreToParams(double score, double &pos_mult, double &tp_mult, double &sl_ratio)
{
    if(score < MinScore) return false;
    
    double t;
    if(score < 0.50) {
        t = (score - 0.30) / 0.20;
        pos_mult = 0.5 + t * 0.2;
        tp_mult  = 0.60 + t * 0.05;
        sl_ratio = 0.45 - t * 0.03;
    } else if(score < 0.70) {
        t = (score - 0.50) / 0.20;
        pos_mult = 0.7 + t * 0.3;
        tp_mult  = 0.65 + t * 0.10;
        sl_ratio = 0.42 - t * 0.04;
    } else if(score < 0.85) {
        t = (score - 0.70) / 0.15;
        pos_mult = 1.0 + t * 0.5;
        tp_mult  = 0.75 + t * 0.15;
        sl_ratio = 0.38 - t * 0.03;
    } else {
        t = MathMin((score - 0.85) / 0.15, 1.0);
        pos_mult = 1.5 + t * 0.5;
        tp_mult  = 0.90 + t * 0.20;
        sl_ratio = 0.35 - t * 0.03;
    }
    return true;
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk                                   |
//+------------------------------------------------------------------+
double CalcLots(double sl_distance, double pos_mult)
{
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double risk_amount = balance * RiskPercent / 100.0 * pos_mult;
    
    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    
    if(tick_value <= 0 || tick_size <= 0 || sl_distance <= 0) return 0;
    
    double lots = risk_amount / (sl_distance / tick_size * tick_value);
    
    double min_lot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_lot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    lots = MathMax(min_lot, MathMin(max_lot, lots));
    lots = MathFloor(lots / lot_step) * lot_step;
    
    return lots;
}

//+------------------------------------------------------------------+
//| Manage open trade - dynamic SL/TP                                  |
//+------------------------------------------------------------------+
void ManageTrade()
{
    if(!in_trade) return;
    
    // Get current position
    if(!PositionSelectByMagic(_Symbol, MagicNumber)) 
    {
        in_trade = false;
        return;
    }
    
    double current_price = (trade_dir == 1) ? 
        SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
        SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    
    int bars_elapsed = Bars(_Symbol, PERIOD_CURRENT) - trade_entry_bar;
    
    // Favorable excursion
    double favorable;
    if(trade_dir == 1)
        favorable = SymbolInfoDouble(_Symbol, SYMBOL_BID) - trade_entry;
    else
        favorable = trade_entry - SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    
    if(favorable > trade_max_favorable)
        trade_max_favorable = favorable;
    
    double progress = (trade_tp_dist > 0) ? trade_max_favorable / trade_tp_dist : 0;
    double time_frac = (trade_expected_bars > 0) ? (double)bars_elapsed / trade_expected_bars : 1;
    double speed = (time_frac > 0.05) ? progress / time_frac : progress * 20;
    
    double new_sl = PositionGetDouble(POSITION_SL);
    double new_tp = PositionGetDouble(POSITION_TP);
    bool modified = false;
    
    // Phase 2: Breakeven at 30% progress
    if(progress >= 0.30 && !trade_hit_be)
    {
        trade_hit_be = true;
        if(trade_dir == 1)
            new_sl = MathMax(new_sl, trade_entry + _Point);
        else
            new_sl = MathMin(new_sl, trade_entry - _Point);
        modified = true;
        
        if(speed > 1.2)
        {
            double expand = MathMin(0.10, (speed - 1.2) * 0.05) * trade_a_amp;
            if(trade_dir == 1)
                new_tp = MathMax(new_tp, new_tp + expand);
            else
                new_tp = MathMin(new_tp, new_tp - expand);
            modified = true;
        }
    }
    
    // Phase 3: Lock 50% at 60% progress
    if(progress >= 0.60)
    {
        double lock;
        if(trade_dir == 1)
        {
            lock = trade_entry + trade_max_favorable * 0.50;
            if(lock > new_sl) { new_sl = lock; modified = true; }
        }
        else
        {
            lock = trade_entry - trade_max_favorable * 0.50;
            if(lock < new_sl) { new_sl = lock; modified = true; }
        }
        
        if(speed > 1.5)
        {
            double expand = MathMin(0.25, (speed - 1.5) * 0.10) * trade_a_amp;
            if(trade_dir == 1)
                new_tp = MathMax(new_tp, new_tp + expand);
            else
                new_tp = MathMin(new_tp, new_tp - expand);
            modified = true;
        }
    }
    
    // Phase 4: Beyond TP at 100%
    if(progress >= 1.0)
    {
        double expand = 0.30 * trade_a_amp;
        if(trade_dir == 1)
        {
            new_tp = MathMax(new_tp, new_tp + expand);
            double lock65 = trade_entry + trade_max_favorable * 0.65;
            if(lock65 > new_sl) new_sl = lock65;
        }
        else
        {
            new_tp = MathMin(new_tp, new_tp - expand);
            double lock65 = trade_entry - trade_max_favorable * 0.65;
            if(lock65 < new_sl) new_sl = lock65;
        }
        modified = true;
        
        if(trade_score > 0.7)
        {
            double extra = (trade_score - 0.7) * 0.5 * trade_a_amp;
            if(trade_dir == 1)
                new_tp = MathMax(new_tp, new_tp + extra);
            else
                new_tp = MathMin(new_tp, new_tp - extra);
        }
    }
    
    // Phase 5: Strong beyond at 150%
    if(progress >= 1.5)
    {
        double expand = 0.50 * trade_a_amp;
        if(trade_dir == 1)
        {
            new_tp = MathMax(new_tp, new_tp + expand);
            double lock75 = trade_entry + trade_max_favorable * 0.75;
            if(lock75 > new_sl) new_sl = lock75;
        }
        else
        {
            new_tp = MathMin(new_tp, new_tp - expand);
            double lock75 = trade_entry - trade_max_favorable * 0.75;
            if(lock75 < new_sl) new_sl = lock75;
        }
        modified = true;
    }
    
    // Phase 6: Extreme at 200%
    if(progress >= 2.0)
    {
        if(trade_dir == 1)
        {
            double lock85 = trade_entry + trade_max_favorable * 0.85;
            if(lock85 > new_sl) { new_sl = lock85; modified = true; }
        }
        else
        {
            double lock85 = trade_entry - trade_max_favorable * 0.85;
            if(lock85 < new_sl) { new_sl = lock85; modified = true; }
        }
    }
    
    // Stagnation: shrink TP if too slow
    if(bars_elapsed > trade_expected_bars * 2.5 && progress < 0.40 && trade_max_favorable > 0)
    {
        double shrink;
        if(trade_dir == 1)
        {
            shrink = trade_entry + trade_max_favorable * 1.05;
            if(shrink < new_tp) { new_tp = shrink; modified = true; }
        }
        else
        {
            shrink = trade_entry - trade_max_favorable * 1.05;
            if(shrink > new_tp) { new_tp = shrink; modified = true; }
        }
    }
    
    // Normalize prices
    double digits_mult = MathPow(10, _Digits);
    new_sl = NormalizeDouble(new_sl, _Digits);
    new_tp = NormalizeDouble(new_tp, _Digits);
    
    if(modified)
    {
        double cur_sl = PositionGetDouble(POSITION_SL);
        double cur_tp = PositionGetDouble(POSITION_TP);
        
        // Only modify if significantly different
        if(MathAbs(new_sl - cur_sl) > _Point || MathAbs(new_tp - cur_tp) > _Point)
        {
            ulong ticket = PositionGetInteger(POSITION_TICKET);
            trade.PositionModify(ticket, new_sl, new_tp);
        }
    }
}

//+------------------------------------------------------------------+
//| Select position by magic number                                    |
//+------------------------------------------------------------------+
bool PositionSelectByMagic(string symbol, int magic)
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionGetSymbol(i) == symbol)
        {
            if(PositionGetInteger(POSITION_MAGIC) == magic)
                return true;
        }
    }
    return false;
}

//+------------------------------------------------------------------+
//| Check if we have an open position                                  |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
    return PositionSelectByMagic(_Symbol, MagicNumber);
}

//+------------------------------------------------------------------+
//| Expert tick function                                                |
//+------------------------------------------------------------------+
void OnTick()
{
    // Manage existing trade
    if(in_trade)
    {
        if(!HasOpenPosition())
        {
            in_trade = false;  // Position was closed (SL/TP hit)
        }
        else
        {
            ManageTrade();
            return;  // Don't open new trades while in one
        }
    }
    
    // Only process on new bar
    static datetime last_bar_time = 0;
    datetime current_bar_time = iTime(_Symbol, PERIOD_CURRENT, 0);
    if(current_bar_time == last_bar_time) return;
    last_bar_time = current_bar_time;
    
    // Get bar index (count from start)
    int bar_idx = Bars(_Symbol, PERIOD_CURRENT) - 1;
    
    // Get previous bar's high/low (completed bar)
    double high = iHigh(_Symbol, PERIOD_CURRENT, 1);
    double low  = iLow(_Symbol, PERIOD_CURRENT, 1);
    
    // Process through zigzag
    bool new_pivot = ZZ_ProcessBar(bar_idx - 1, high, low);
    
    // Check for entry signal
    if(new_pivot && pivot_count >= 3 && !in_trade)
    {
        // Get last 3 pivots: p0 (A start), p1 (A end/B start), p2 (B end = entry)
        PivotPoint p0 = pivots[pivot_count - 3];
        PivotPoint p1 = pivots[pivot_count - 2];
        PivotPoint p2 = pivots[pivot_count - 1];
        
        // A wave
        int a_bars = p1.bar_idx - p0.bar_idx;
        double a_amp = MathAbs(p1.price - p0.price);
        int a_dir = (p1.price > p0.price) ? 1 : -1;
        
        // B wave
        int b_bars = p2.bar_idx - p1.bar_idx;
        double b_amp = MathAbs(p2.price - p1.price);
        
        if(a_bars <= 0 || b_bars <= 0 || a_amp <= 0) return;
        
        double amp_ratio = b_amp / a_amp;
        double time_ratio = (double)b_bars / (double)a_bars;
        double a_slope = a_amp / a_bars;
        double b_slope = b_amp / b_bars;
        double slope_ratio = (a_slope > 0) ? b_slope / a_slope : 999;
        
        // Direction check
        if(a_dir == 1 && !AllowLong) return;
        if(a_dir == -1 && !AllowShort) return;
        
        // Compute score
        double score = ComputeScore(slope_ratio, time_ratio, amp_ratio);
        
        double pos_mult, tp_mult, sl_ratio;
        if(!ScoreToParams(score, pos_mult, tp_mult, sl_ratio)) return;
        
        // Calculate TP/SL
        double tp_distance = a_amp * tp_mult;
        double sl_distance = tp_distance * sl_ratio;
        
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        
        double entry_price, sl_price, tp_price;
        
        if(a_dir == 1) // Long
        {
            entry_price = ask;
            sl_price = NormalizeDouble(entry_price - sl_distance, _Digits);
            tp_price = NormalizeDouble(entry_price + tp_distance, _Digits);
        }
        else // Short
        {
            entry_price = bid;
            sl_price = NormalizeDouble(entry_price + sl_distance, _Digits);
            tp_price = NormalizeDouble(entry_price - tp_distance, _Digits);
        }
        
        // Calculate lot size
        double lots = CalcLots(sl_distance, pos_mult);
        if(lots <= 0) return;
        
        // Execute trade
        bool result;
        if(a_dir == 1)
            result = trade.Buy(lots, _Symbol, ask, sl_price, tp_price, 
                              StringFormat("ABC S=%.2f", score));
        else
            result = trade.Sell(lots, _Symbol, bid, sl_price, tp_price,
                               StringFormat("ABC S=%.2f", score));
        
        if(result)
        {
            in_trade = true;
            trade_dir = a_dir;
            trade_entry = entry_price;
            trade_a_amp = a_amp;
            trade_a_bars = a_bars;
            trade_tp_dist = tp_distance;
            trade_sl_dist = sl_distance;
            trade_initial_sl_dist = sl_distance;
            trade_max_favorable = 0;
            trade_hit_be = false;
            trade_score = score;
            trade_expected_bars = a_bars * 1.2;
            trade_entry_bar = bar_idx;
            
            PrintFormat("ABC Entry: dir=%d score=%.2f amp_r=%.2f time_r=%.2f slope_r=%.2f lots=%.2f",
                       a_dir, score, amp_ratio, time_ratio, slope_ratio, lots);
        }
    }
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                            |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    ArrayFree(pivots);
}
//+------------------------------------------------------------------+
