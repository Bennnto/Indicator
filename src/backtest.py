import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ccxt
import argparse
import pandas as pd
import numpy as np
from data_engine import fetch_paginated_ohlcv, get_binance_exchange, coins
from indicators import (
    compute_ema, compute_rsi, compute_atr, compute_adx,
    compute_volume_ratio, compute_macd, update_trailing_stop,
    check_candlestick_reversal, kelly_position_size,
    ATR_SL_MULTIPLIER, ATR_ENTRY_DISCOUNT, ATR_TRAIL_MULT,
    TP1_RR, TP2_RR, PARTIAL_EXIT_PCT
)
from ml_engine import predict_probability

# ======================================================================
# STRATEGY THRESHOLDS (Institutional Confluence Suite)
# ======================================================================
ADX_MIN         = 18
RSI_LONG_MAX    = 45
RSI_SHORT_MIN   = 55
VOL_SPIKE_MAX   = 1.8
MIN_LGBM_PROB   = 0.55  # Minimum statistical win probability required from LightGBM


def run_backtest(symbol: str = 'BTC/USDT', days: int = 45,
                 capital: float = 10000.0,
                 use_4h_filter: bool = True,
                 use_candlestick_filter: bool = True,
                 use_lgbm_filter: bool = True) -> dict:
    """
    Comprehensive historical backtest on Binance data testing:
    1. 4h Macro Trend Confluence (Close > 4h EMA50)
    2. 1h Intermediate Trend (EMA50 > EMA200)
    3. 15m RSI Pullback (< 45) + Volume (< 1.8x) + ADX (> 18) + MACD
    4. 15m Candlestick Reversal (Falling knife protection)
    5. LightGBM Quantitative Win Predictor
    6. Trade Management: Limit Entry, 2.0 ATR Stop, Breakeven at +1.0R, TP1 at 1.5R, TP2 at 2.5R
    """
    exchange = get_binance_exchange()

    print(f"\n{'='*65}")
    print(f"⏳ Loading {days}-Day Multi-Timeframe Binance History for {symbol}...")
    print(f"{'='*65}")

    # 1. Fetch 15m candles
    df15 = fetch_paginated_ohlcv(exchange, symbol, timeframe='15m', days=days)
    if df15.empty or len(df15) < 300:
        print(f"  ❌ Insufficient 15m data for {symbol}")
        return {}

    # 2. Fetch 1h candles
    df1h = fetch_paginated_ohlcv(exchange, symbol, timeframe='1h', days=days + 15)
    if df1h.empty or len(df1h) < 200:
        print(f"  ❌ Insufficient 1h data for {symbol}")
        return {}

    # 3. Fetch 4h candles
    df4h = fetch_paginated_ohlcv(exchange, symbol, timeframe='4h', days=days + 25)

    # Compute 1h Indicators
    df1h = df1h.rename(columns={'timestamp': 'time'})
    df1h['ema_50_1h']  = compute_ema(df1h['close'], 50)
    df1h['ema_200_1h'] = compute_ema(df1h['close'], 200)

    # Compute 4h Indicators
    if not df4h.empty and len(df4h) >= 50:
        df4h = df4h.rename(columns={'timestamp': 'time'})
        df4h['ema_50_4h'] = compute_ema(df4h['close'], 50)
        df15_merged = pd.merge_asof(
            df15.rename(columns={'timestamp': 'time'}).sort_values('time'),
            df4h[['time', 'ema_50_4h']].sort_values('time'),
            on='time', direction='backward'
        )
    else:
        df15_merged = df15.rename(columns={'timestamp': 'time'})
        df15_merged['ema_50_4h'] = np.nan

    # Merge 1h trend into 15m without lookahead bias
    df = pd.merge_asof(
        df15_merged.sort_values('time'),
        df1h[['time', 'ema_50_1h', 'ema_200_1h']].sort_values('time'),
        on='time', direction='backward'
    )

    # 15m Technicals
    df['rsi']    = compute_rsi(df['close'], 14)
    df['atr']    = compute_atr(df, 14)
    df['adx']    = compute_adx(df, 14)
    df['vol']    = compute_volume_ratio(df['volume'], 20)
    _, _, mh     = compute_macd(df['close'])
    df['macd_h'] = mh

    # Simulation State
    trades = []
    active = None
    running_capital = capital
    setups_encountered = 0
    setups_filtered_by_4h = 0
    setups_filtered_by_candle = 0
    setups_filtered_by_lgbm = 0

    for i in range(200, len(df) - 1):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        if pd.isna(row.get('ema_50_1h')) or pd.isna(row.get('ema_200_1h')):
            continue

        # ------------------------------------------------------------
        # TRADE MANAGEMENT: Active position simulation
        # ------------------------------------------------------------
        if active is not None:
            side = active['side']
            entry_p = active['entry']
            r_dist = active['r_dist']

            # Move stop to breakeven once price reaches +1.0R
            if side == 'LONG':
                if row['high'] >= (entry_p + (1.0 * r_dist)) and not active['be_moved']:
                    active['trail_sl'] = entry_p
                    active['be_moved'] = True

                hit_tp1 = row['high'] >= active['tp1'] and not active['tp1_hit']
                hit_tp2 = row['high'] >= active['tp2']
                hit_sl  = row['low']  <= active['trail_sl']
            else:
                if row['low'] <= (entry_p - (1.0 * r_dist)) and not active['be_moved']:
                    active['trail_sl'] = entry_p
                    active['be_moved'] = True

                hit_tp1 = row['low']  <= active['tp1'] and not active['tp1_hit']
                hit_tp2 = row['low']  <= active['tp2']
                hit_sl  = row['high'] >= active['trail_sl']

            # Partial exit at TP1 (50%)
            if hit_tp1:
                active['tp1_hit'] = True
                active['trail_sl'] = entry_p  # Ensure stop is at breakeven
                active['partial_r'] = TP1_RR * PARTIAL_EXIT_PCT

            # Full exit at TP2
            if hit_tp2:
                active['pnl_r'] = round(active.get('partial_r', 0) + (TP2_RR * (1 - PARTIAL_EXIT_PCT)), 2)
                active['exit']  = 'TP2'
                trades.append(active)
                running_capital += active['risk_usdt'] * active['pnl_r']
                active = None

            elif hit_sl:
                if active.get('tp1_hit'):
                    active['pnl_r'] = round(active.get('partial_r', 0), 2)
                    active['exit']  = 'BE_AFTER_TP1'
                elif active.get('be_moved'):
                    active['pnl_r'] = 0.0
                    active['exit']  = 'BREAKEVEN'
                else:
                    active['pnl_r'] = -1.0
                    active['exit']  = 'SL'

                trades.append(active)
                running_capital += active['risk_usdt'] * active['pnl_r']
                active = None

        # ------------------------------------------------------------
        # SIGNAL SCANNING: Look for new high-probability entries
        # ------------------------------------------------------------
        else:
            c    = row['close']
            e50  = row['ema_50_1h']
            e200 = row['ema_200_1h']
            rsi  = row['rsi']
            atr  = row['atr']
            adx  = row['adx']
            vol  = row['vol']
            mh_  = row['macd_h']
            e50_4h = row.get('ema_50_4h')

            if any(pd.isna(v) for v in [rsi, atr, adx, vol, mh_]):
                continue

            # Check for candidate pullback in 1h uptrend
            is_1h_bull = (c > e50) and (e50 > e200)
            is_rsi_dip = rsi < RSI_LONG_MAX
            macd_turning_up = mh_ > df['macd_h'].iloc[i - 1]

            if is_1h_bull and is_rsi_dip and macd_turning_up and adx >= ADX_MIN and vol <= VOL_SPIKE_MAX:
                setups_encountered += 1

                # 1. 4H MACRO CONFLUENCE FILTER
                if use_4h_filter and not pd.isna(e50_4h) and c < e50_4h:
                    setups_filtered_by_4h += 1
                    continue

                # 2. 15M CANDLESTICK CONFIRMATION FILTER
                recent_df = df.iloc[max(0, i - 5):i + 1]
                candle_ok, _ = check_candlestick_reversal(recent_df, side="LONG")
                if use_candlestick_filter and not candle_ok:
                    setups_filtered_by_candle += 1
                    continue

                # 3. LIGHTGBM STATISTICAL WIN PREDICTOR FILTER
                o, h, l = row['open'], row['high'], row['low']
                c_range = max(h - l, 1e-8)
                lower_wick = max(0.0, min(o, c) - l) / c_range
                is_green = 1.0 if c >= o else 0.0

                lgbm_features = {
                    'rsi_15m': rsi,
                    'macd_hist_15m': mh_,
                    'adx_15m': adx,
                    'volume_ratio': vol,
                    'extension_from_ema50': ((c - e50) / e50) * 100.0,
                    'ema50_over_ema200': 1.0,
                    'lower_wick_ratio': lower_wick,
                    'candle_is_green': is_green,
                    'ob_ratio': 1.0,
                    'btc_trend_bullish': 1.0
                }
                ml_pred = predict_probability(lgbm_features)

                if use_lgbm_filter and ml_pred['prob'] < MIN_LGBM_PROB:
                    setups_filtered_by_lgbm += 1
                    continue

                # Entry sizing
                e  = c - (ATR_ENTRY_DISCOUNT * atr)
                sl = e - (ATR_SL_MULTIPLIER * atr)
                r  = e - sl

                kelly = kelly_position_size(win_rate=ml_pred['prob'], rr_ratio=TP1_RR, capital=running_capital, price=c)
                risk_usdt = kelly['risk_usdt']

                active = {
                    'side': 'LONG',
                    'entry': round(e, 4),
                    'trail_sl': round(sl, 4),
                    'r_dist': round(r, 4),
                    'tp1': round(e + (TP1_RR * r), 4),
                    'tp2': round(e + (TP2_RR * r), 4),
                    'tp1_hit': False,
                    'be_moved': False,
                    'time': row['time'],
                    'risk_usdt': risk_usdt,
                    'ml_prob': ml_pred['prob_pct']
                }

    # ------------------------------------------------------------
    # Results Compilation
    # ------------------------------------------------------------
    if not trades:
        print(f"  ⚪ No approved trades executed for {symbol} ({setups_encountered} raw setups filtered by strict safety rules).")
        return {}

    dt       = pd.DataFrame(trades)
    total    = len(dt)
    wins     = len(dt[dt['pnl_r'] > 0])
    losses   = len(dt[dt['pnl_r'] < 0])
    be_count = len(dt[dt['pnl_r'] == 0.0])
    net_r    = round(dt['pnl_r'].sum(), 2)
    win_rate = round((wins / total) * 100, 1)

    pos_r = dt[dt['pnl_r'] > 0]['pnl_r'].sum()
    neg_r = abs(dt[dt['pnl_r'] < 0]['pnl_r'].sum())
    pf    = round(pos_r / max(neg_r, 0.01), 2)

    est_pnl = round(running_capital - capital, 2)
    pnl_pct = round((est_pnl / capital) * 100, 2)

    date_from = dt['time'].iloc[0].strftime('%Y-%m-%d')
    date_to   = dt['time'].iloc[-1].strftime('%Y-%m-%d')

    print(f"\n📊 RESULTS FOR {symbol} ({days} Days: {date_from} → {date_to}):")
    print(f"   Raw Pullback Setups Found   : {setups_encountered}")
    print(f"   Filtered by 4h Downtrend    : -{setups_filtered_by_4h}")
    print(f"   Filtered by Candlestick     : -{setups_filtered_by_candle}")
    print(f"   Filtered by LightGBM (<55%) : -{setups_filtered_by_lgbm}")
    print(f"   Approved Trades Executed    : {total}")
    print(f"   -------------------------------------------")
    print(f"   🏆 Realized Win Rate        : {win_rate}% ({wins}W / {losses}L / {be_count} Breakeven)")
    print(f"   📈 Profit Factor            : {pf}")
    print(f"   💰 Cumulative Return        : {pnl_pct:+.2f}% (${running_capital:,.2f})")
    print(f"   🛡️ Net Realized R           : {net_r:+.2f}R")

    return {
        'symbol': symbol,
        'setups': setups_encountered,
        'trades': total,
        'win_rate': win_rate,
        'wins': wins,
        'losses': losses,
        'breakevens': be_count,
        'net_r': net_r,
        'pf': pf,
        'pnl_pct': pnl_pct
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Timeframe Quantitative Backtester")
    parser.add_argument("--days", type=int, default=60, help="Number of historical days to backtest (default: 60)")
    parser.add_argument("--symbol", type=str, default="ALL", help="Coin symbol or ALL")
    args = parser.parse_args()

    test_coins = coins if args.symbol == "ALL" else [args.symbol]
    portfolio_results = []

    print("=" * 65)
    print("🚀 RUNNING HISTORICAL BACKTEST WITH LIGHTGBM + 4H + CANDLESTICKS")
    print(f"   Pairs: {', '.join(test_coins)} | Lookback: {args.days} Days")
    print("=" * 65)

    for coin in test_coins:
        res = run_backtest(symbol=coin, days=args.days, capital=10000.0)
        if res:
            portfolio_results.append(res)

    if portfolio_results:
        print("\n" + "=" * 65)
        print("🏆 PORTFOLIO BACKTEST SUMMARY")
        print("=" * 65)
        total_trades = sum(r['trades'] for r in portfolio_results)
        total_wins = sum(r['wins'] for r in portfolio_results)
        total_losses = sum(r['losses'] for r in portfolio_results)
        total_be = sum(r['breakevens'] for r in portfolio_results)
        avg_wr = round((total_wins / total_trades) * 100, 1) if total_trades > 0 else 0
        total_pnl = round(sum(r['pnl_pct'] for r in portfolio_results) / len(portfolio_results), 2)
        total_r = round(sum(r['net_r'] for r in portfolio_results), 2)

        print(f"Total Approved Trades: {total_trades}")
        print(f"Overall Win Rate     : {avg_wr}% ({total_wins} Wins / {total_losses} Losses / {total_be} Breakeven)")
        print(f"Total Net R          : {total_r:+.2f}R")
        print(f"Avg Portfolio Return : {total_pnl:+.2f}%")
        print("=" * 65)
