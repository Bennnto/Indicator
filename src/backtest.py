import ccxt
import pandas as pd
from data_engine import fetch_paginated_ohlcv
from indicators import (
    compute_ema, compute_rsi, compute_atr, compute_adx,
    compute_volume_ratio, compute_macd, update_trailing_stop, kelly_position_size,
    ATR_SL_MULTIPLIER, ATR_ENTRY_DISCOUNT, ATR_TRAIL_MULT,
    TP1_RR, TP2_RR, PARTIAL_EXIT_PCT
)

# ======================================================================
# THRESHOLDS
# ======================================================================
ADX_MIN       = 15
RSI_LONG_MAX  = 45
RSI_SHORT_MIN = 55
VOL_SPIKE_MAX = 1.8
CONSEC_LOSS_MAX = 2
CIRCUIT_PAUSE   = 8   # 15m bars = 2 hours


def run_backtest(symbol: str = 'BTC/USDT', days: int = 90,
                 capital: float = 10000.0) -> dict:
    """
    Full multi-timeframe backtest using paginated 90-day 15m data.
    Uses 1h candles for macro trend, 15m for entries.
    All V2 + V3 upgrades active.
    """
    exchange = ccxt.binance({'enableRateLimit': True})

    print(f"\n{'='*60}")
    print(f"⏳  Loading {days}-day history for {symbol}...")
    print(f"{'='*60}")

    # Paginated 15m data (90 days ≈ 8640 bars)
    df15 = fetch_paginated_ohlcv(exchange, symbol, timeframe='15m', days=days)
    if df15.empty or len(df15) < 300:
        print(f"  ❌ Not enough 15m data for {symbol}")
        return {}

    # 1h data for macro trend (fetch extra for EMA 200 warmup)
    df1h = fetch_paginated_ohlcv(exchange, symbol, timeframe='1h', days=days + 10)
    if df1h.empty:
        print(f"  ❌ Not enough 1h data for {symbol}")
        return {}

    # 1h EMAs
    df1h = df1h.rename(columns={'timestamp': 'time'})
    df1h['ema_50_1h']  = compute_ema(df1h['close'], 50)
    df1h['ema_200_1h'] = compute_ema(df1h['close'], 200)

    # Merge 1h trend into 15m (no lookahead bias via backward merge)
    df15 = df15.rename(columns={'timestamp': 'time'})
    df = pd.merge_asof(df15, df1h[['time', 'ema_50_1h', 'ema_200_1h']],
                       on='time', direction='backward')

    # 15m indicators
    df['rsi']     = compute_rsi(df['close'], 14)
    df['atr']     = compute_atr(df, 14)
    df['adx']     = compute_adx(df, 14)
    df['vol']     = compute_volume_ratio(df['volume'], 20)
    _, _, mh      = compute_macd(df['close'])
    df['macd_h']  = mh

    # Simulation loop
    trades         = []
    active         = None
    cons_loss      = 0
    pause_idx      = 0
    running_capital = capital

    for i in range(200, len(df) - 1):
        row = df.iloc[i]
        if pd.isna(row.get('ema_50_1h')) or pd.isna(row.get('ema_200_1h')):
            continue

        # ------------------------------------------------------------
        # Manage open trade
        # ------------------------------------------------------------
        if active is not None:
            side = active['side']

            # Update trailing stop
            active['trail_sl'] = update_trailing_stop(
                row['close'], active['trail_sl'], row['atr'], side, ATR_TRAIL_MULT
            )

            if side == 'LONG':
                hit_tp1 = row['high'] >= active['tp1'] and not active['tp1_hit']
                hit_tp2 = row['high'] >= active['tp2']
                hit_sl  = row['low']  <= active['trail_sl']
            else:
                hit_tp1 = row['low']  <= active['tp1'] and not active['tp1_hit']
                hit_tp2 = row['low']  <= active['tp2']
                hit_sl  = row['high'] >= active['trail_sl']

            # Partial exit at TP1
            if hit_tp1:
                active['tp1_hit']   = True
                active['trail_sl']  = active['entry']   # Move stop to breakeven
                active['partial_r'] = TP1_RR * PARTIAL_EXIT_PCT

            # Full exit at TP2
            if hit_tp2:
                active['pnl_r'] = round(
                    active.get('partial_r', 0) + TP2_RR * (1 - PARTIAL_EXIT_PCT), 2)
                active['exit']  = 'TP2'
                trades.append(active)
                cons_loss        = 0
                running_capital += active['risk_usdt'] * active['pnl_r']
                active = None

            elif hit_sl:
                partial = active.get('partial_r', 0)
                if active.get('tp1_hit'):
                    active['pnl_r'] = round(partial, 2)
                    active['exit']  = 'BE_TP1'
                    cons_loss       = 0
                else:
                    active['pnl_r'] = -1.0
                    active['exit']  = 'SL'
                    cons_loss += 1
                    if cons_loss >= CONSEC_LOSS_MAX:
                        pause_idx = i + CIRCUIT_PAUSE
                        cons_loss = 0
                trades.append(active)
                running_capital += active['risk_usdt'] * active['pnl_r']
                active = None

        # ------------------------------------------------------------
        # Look for new setups
        # ------------------------------------------------------------
        elif i >= pause_idx:
            c    = row['close']
            e50  = row['ema_50_1h']
            e200 = row['ema_200_1h']
            rsi  = row['rsi']
            atr  = row['atr']
            adx  = row['adx']
            vol  = row['vol']
            mh_  = row['macd_h']

            if any(pd.isna(v) for v in [rsi, atr, adx, vol, mh_]):
                continue
            if adx < ADX_MIN or vol > VOL_SPIKE_MAX:
                continue

            prev_mh = df['macd_h'].iloc[i - 1]
            macd_up   = mh_ > prev_mh
            macd_down = mh_ < prev_mh

            # Kelly sizing
            wr = 0.50
            if len(trades) >= 10:
                wr = sum(1 for t in trades if t.get('pnl_r', 0) > 0) / len(trades)
            kelly     = kelly_position_size(wr, TP1_RR, running_capital, c)
            risk_usdt = kelly['risk_usdt']

            # LONG setup
            if c > e50 > e200 and rsi < RSI_LONG_MAX and macd_up:
                e  = c - (ATR_ENTRY_DISCOUNT * atr)
                sl = e - (ATR_SL_MULTIPLIER * atr)
                r  = e - sl
                active = {
                    'side': 'LONG', 'entry': round(e, 4),
                    'trail_sl': round(sl, 4),
                    'tp1': round(e + TP1_RR * r, 4),
                    'tp2': round(e + TP2_RR * r, 4),
                    'tp1_hit': False, 'time': row['time'],
                    'risk_usdt': risk_usdt
                }

            # SHORT setup
            elif c < e50 < e200 and rsi > RSI_SHORT_MIN and macd_down:
                e  = c + (ATR_ENTRY_DISCOUNT * atr)
                sl = e + (ATR_SL_MULTIPLIER * atr)
                r  = sl - e
                active = {
                    'side': 'SHORT', 'entry': round(e, 4),
                    'trail_sl': round(sl, 4),
                    'tp1': round(e - TP1_RR * r, 4),
                    'tp2': round(e - TP2_RR * r, 4),
                    'tp1_hit': False, 'time': row['time'],
                    'risk_usdt': risk_usdt
                }

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------
    if not trades:
        print(f"  ⚪ No trades triggered for {symbol} in {days} days.")
        return {}

    dt       = pd.DataFrame(trades)
    total    = len(dt)
    wins     = len(dt[dt['pnl_r'] > 0])
    losses   = len(dt[dt['pnl_r'] < 0])
    be_count = len(dt[dt['exit'] == 'BE_TP1']) if 'exit' in dt.columns else 0
    net_r    = round(dt['pnl_r'].sum(), 2)
    avg_r    = round(dt['pnl_r'].mean(), 3)
    win_rate = round(wins / total * 100, 1)
    longs    = dt[dt['side'] == 'LONG']
    shorts   = dt[dt['side'] == 'SHORT']
    pos_r    = dt[dt['pnl_r'] > 0]['pnl_r'].sum()
    neg_r    = abs(dt[dt['pnl_r'] < 0]['pnl_r'].sum())
    pf       = round(pos_r / max(neg_r, 0.01), 2)

    est_pnl  = round(running_capital - capital, 2)
    pnl_pct  = round((est_pnl / capital) * 100, 2)

    date_from = dt['time'].iloc[0].strftime('%Y-%m-%d')
    date_to   = dt['time'].iloc[-1].strftime('%Y-%m-%d')

    print(f"\n{'='*60}")
    print(f"📊  BACKTEST: {symbol}  |  {days} days  ({date_from} → {date_to})")
    print(f"    Thresholds: ADX≥{ADX_MIN} | RSI<{RSI_LONG_MAX}L >{RSI_SHORT_MIN}S | Vol≤{VOL_SPIKE_MAX}x")
    print(f"{'='*60}")
    print(f"  Capital       : ${capital:,.0f}  →  ${running_capital:,.2f}  ({pnl_pct:+.1f}%)")
    print(f"  Total Trades  : {total}  ({len(longs)}L / {len(shorts)}S)")
    print(f"  Win / Loss    : {wins}W / {losses}L  (+{be_count} breakeven after TP1)")
    print(f"  Win Rate      : {win_rate}%")
    print(f"  Avg Trade     : {avg_r:+.3f}R")
    print(f"  Net Return    : {net_r:+.2f}R")
    print(f"  Profit Factor : {pf}")
    print(f"  Long  W/L     : {len(longs[longs['pnl_r']>0])}W / {len(longs[longs['pnl_r']<0])}L")
    print(f"  Short W/L     : {len(shorts[shorts['pnl_r']>0])}W / {len(shorts[shorts['pnl_r']<0])}L")
    verdict = "✅ PROFITABLE" if net_r > 0 and pf >= 1.2 else "⚠️ NEEDS TUNING"
    print(f"  Verdict       : {verdict}")

    return {
        'symbol': symbol, 'trades': total, 'win_rate': win_rate,
        'net_r': net_r, 'pf': pf, 'est_pnl': est_pnl, 'pnl_pct': pnl_pct
    }


if __name__ == "__main__":
    summary = []
    for coin in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']:
        result = run_backtest(symbol=coin, days=90, capital=10000.0)
        if result:
            summary.append(result)

    if summary:
        print(f"\n{'='*60}")
        print("📋  PORTFOLIO SUMMARY  (90-day | $10,000 start)")
        print(f"{'='*60}")
        for r in summary:
            icon = "✅" if r['net_r'] > 0 else "❌"
            print(f"  {icon} {r['symbol']:<12}  WR:{r['win_rate']}%  "
                  f"Net:{r['net_r']:+.1f}R  PF:{r['pf']}  "
                  f"P&L: ${r['est_pnl']:+,.2f} ({r['pnl_pct']:+.1f}%)")
